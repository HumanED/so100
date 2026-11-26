"""Minimal PPO (PyTorch) for batched GPU envs: Gaussian policy, GAE, timeout bootstrapping, adaptive learning rate."""
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    steps_per_env: int = 24            # rollout length per env per update
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    desired_kl: float = 0.01           # adaptive learning rate target (None = fixed lr)
    entropy_coef: float = 0.005
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    init_std: float = 1.0
    actor_hidden: Sequence[int] = (256, 256, 128)
    critic_hidden: Sequence[int] = (256, 256, 256)


def mlp(inp, hidden, out):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, n, eps=1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps), -10, 10)


class ActorCritic(nn.Module):
    def __init__(self, obs_size, act_size, cfg: PPOConfig):
        super().__init__()
        self.norm = RunningNorm(obs_size)
        self.actor = mlp(obs_size, cfg.actor_hidden, act_size)
        self.critic = mlp(obs_size, cfg.critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_size,), float(torch.log(torch.tensor(cfg.init_std)))))

    def dist(self, obs):
        mean = self.actor(self.norm(obs))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(self.norm(obs)).squeeze(-1)

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.actor(self.norm(obs))


class PPO:
    def __init__(self, obs_size, act_size, num_envs, device, cfg: PPOConfig = PPOConfig()):
        self.cfg, self.device, self.n = cfg, device, num_envs
        self.ac = ActorCritic(obs_size, act_size, cfg).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        T = cfg.steps_per_env
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs, self.actions, self.logp = z(T, num_envs, obs_size), z(T, num_envs, act_size), z(T, num_envs)
        self.rew, self.done, self.val = z(T, num_envs), z(T, num_envs), z(T, num_envs)
        self.mu, self.sigma = z(T, num_envs, act_size), z(T, num_envs, act_size)
        self.step = 0

    @torch.no_grad()
    def act(self, obs):
        self.ac.norm.update(obs)
        d = self.ac.dist(obs)
        a = d.sample()
        t = self.step
        self.obs[t], self.actions[t], self.logp[t] = obs, a, d.log_prob(a).sum(-1)
        self.val[t], self.mu[t], self.sigma[t] = self.ac.value(obs), d.mean, d.stddev
        return a

    @torch.no_grad()
    def record(self, reward, done, timeout):
        t = self.step
        # time-limit endings aren't failures: bootstrap with the value estimate (rsl_rl-style approximation)
        self.rew[t] = reward + self.cfg.gamma * self.val[t] * timeout.float()
        self.done[t] = done.float()
        self.step += 1

    def update(self, last_obs):
        cfg = self.cfg
        with torch.no_grad():
            adv = torch.zeros_like(self.rew)
            last_val = self.ac.value(last_obs)
            gae = torch.zeros(self.n, device=self.device)
            for t in reversed(range(cfg.steps_per_env)):
                next_val = last_val if t == cfg.steps_per_env - 1 else self.val[t + 1]
                nonterm = 1.0 - self.done[t]
                delta = self.rew[t] + cfg.gamma * next_val * nonterm - self.val[t]
                gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                adv[t] = gae
            ret = adv + self.val
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        O, A, LP, ADV, RET, V, MU, SG = map(flat, (self.obs, self.actions, self.logp, adv, ret, self.val, self.mu, self.sigma))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        N = O.shape[0]
        mb = N // cfg.minibatches
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_upd = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=self.device)
            for i in range(cfg.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                d = self.ac.dist(O[j])
                logp = d.log_prob(A[j]).sum(-1)
                with torch.no_grad():  # KL(old || new) for diagonal Gaussians -> adaptive lr
                    s_new = d.stddev
                    kl = (torch.log(s_new / SG[j]) + (SG[j] ** 2 + (MU[j] - d.mean) ** 2) / (2 * s_new ** 2) - 0.5).sum(-1).mean()
                    if cfg.desired_kl:
                        if kl > 2 * cfg.desired_kl:
                            self.lr = max(1e-5, self.lr / 1.5)
                        elif 0 < kl < cfg.desired_kl / 2:
                            self.lr = min(1e-2, self.lr * 1.5)
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr
                ratio = torch.exp(logp - LP[j])
                pl = -torch.min(ratio * ADV[j], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ADV[j]).mean()
                v = self.ac.value(O[j])
                v_clip = V[j] + (v - V[j]).clamp(-cfg.clip, cfg.clip)
                vl = torch.max((v - RET[j]) ** 2, (v_clip - RET[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + cfg.value_coef * vl - cfg.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()
                for k, x in (("policy_loss", pl), ("value_loss", vl), ("entropy", ent), ("kl", kl)):
                    stats[k] += float(x)
                n_upd += 1
        self.step = 0
        return {k: v / n_upd for k, v in stats.items()} | {"lr": self.lr, "action_std": float(self.ac.log_std.exp().mean())}
