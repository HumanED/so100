"""
Brax PPO training for SoFetchMjxEnv on GPU.

    python -m so_arm_mjx.train --run-name PPO-carl-mjx-1 --num-timesteps 100_000_000

Outputs, under <out_dir>:
    models/<run>/<step>.pkl     policy params (+ obs normaliser) every eval, loadable by visualise_mjx.py
    models/<run>/final.pkl
    models/<run>/config.json    env + network config needed to rebuild the policy
    checkpoints/<run>/...       Brax/orbax checkpoints (pass to --restore to resume training)
    logs/<run>.csv              metrics per eval
"""
import argparse
import csv
import functools
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import jax
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from .fetch_env import FetchConfig, SoFetchMjxEnv


@dataclass
class TrainConfig:
    run_name: str = "PPO-carl-mjx-1"
    out_dir: str = "mjx_out"
    num_timesteps: int = 100_000_000
    time_limit_hours: Optional[float] = None   # stop (after saving) at the first eval past this
    num_envs: int = 2048                       # parallel envs on the GPU
    num_evals: int = 25                        # evals == save points
    num_eval_envs: int = 128
    seed: int = 1
    # PPO
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.99
    unroll_length: int = 10
    batch_size: int = 256
    num_minibatches: int = 32                  # batch_size * num_minibatches must be divisible by num_envs
    num_updates_per_batch: int = 4
    clipping_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    reward_scaling: float = 1.0
    policy_hidden_layer_sizes: Sequence[int] = (256, 256, 128)
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256)
    restore_checkpoint_path: Optional[str] = None   # a checkpoints/<run>/<step> dir to resume from
    env: FetchConfig = field(default_factory=FetchConfig)


class _TimeLimit(Exception):
    pass


def train(cfg: TrainConfig, verbose: bool = True):
    print("JAX devices:", jax.devices())
    models_dir = os.path.join(cfg.out_dir, "models", cfg.run_name)
    ckpt_dir = os.path.abspath(os.path.join(cfg.out_dir, "checkpoints", cfg.run_name))
    logs_dir = os.path.join(cfg.out_dir, "logs")
    if cfg.restore_checkpoint_path is None and os.path.exists(models_dir) and os.listdir(models_dir):
        raise RuntimeError(f"{models_dir} already exists; change run_name or set restore_checkpoint_path")
    for d in (models_dir, ckpt_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    env = SoFetchMjxEnv(cfg.env)
    eval_env = SoFetchMjxEnv(cfg.env)

    config_out = asdict(cfg)
    config_out["env"] = cfg.env.to_dict()
    with open(os.path.join(models_dir, "config.json"), "w") as f:
        json.dump(config_out, f, indent=2, default=list)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(cfg.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.value_hidden_layer_sizes),
    )

    t0 = time.time()
    log_path = os.path.join(logs_dir, f"{cfg.run_name}.csv")
    log_rows = []
    last = {"step": 0, "params": None}

    def progress_fn(step, metrics):
        row = {"step": int(step), "time_s": round(time.time() - t0, 1)}
        row.update({k: float(v) for k, v in metrics.items() if k.startswith("eval/") and "std" not in k})
        log_rows.append(row)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in log_rows for k in r}, key=lambda k: (k != "step", k)))
            w.writeheader(); w.writerows(log_rows)
        if verbose:
            sps = step / max(time.time() - t0, 1e-6)
            print(f"[{row['time_s'] / 60:6.1f} min] step {step:>12,} | "
                  f"eval reward {row.get('eval/episode_reward', float('nan')):8.2f} | "
                  f"grasp {row.get('eval/episode_grasp', float('nan')):.2f} | "
                  f"success {row.get('eval/episode_success', float('nan')):.2f} | {sps:,.0f} steps/s overall")

    def policy_params_fn(step, make_policy, params):
        last["step"], last["params"] = step, params
        if step > 0:
            brax_model.save_params(os.path.join(models_dir, f"{step}.pkl"), params)
        if cfg.time_limit_hours is not None and time.time() - t0 > cfg.time_limit_hours * 3600:
            raise _TimeLimit()

    try:
        make_inference_fn, params, metrics = ppo.train(
            environment=env,
            eval_env=eval_env,
            num_timesteps=cfg.num_timesteps,
            episode_length=cfg.env.episode_length,
            num_envs=cfg.num_envs,
            num_eval_envs=cfg.num_eval_envs,
            num_evals=cfg.num_evals,
            learning_rate=cfg.learning_rate,
            entropy_cost=cfg.entropy_cost,
            discounting=cfg.discounting,
            unroll_length=cfg.unroll_length,
            batch_size=cfg.batch_size,
            num_minibatches=cfg.num_minibatches,
            num_updates_per_batch=cfg.num_updates_per_batch,
            clipping_epsilon=cfg.clipping_epsilon,
            max_grad_norm=cfg.max_grad_norm,
            reward_scaling=cfg.reward_scaling,
            normalize_observations=True,
            network_factory=network_factory,
            seed=cfg.seed,
            progress_fn=progress_fn,
            policy_params_fn=policy_params_fn,
            save_checkpoint_path=ckpt_dir,
            restore_checkpoint_path=cfg.restore_checkpoint_path,
        )
    except _TimeLimit:
        print(f"Time limit reached at step {last['step']:,}; stopping")
        params = last["params"]
    except KeyboardInterrupt:
        print("Interrupted; saving latest params")
        params = last["params"]

    if params is not None:
        final = os.path.join(models_dir, "final.pkl")
        brax_model.save_params(final, params)
        print(f"Saved {final} (step {last['step']:,}, {(time.time() - t0) / 60:.1f} min)")
    return models_dir, log_path


def main():
    p = argparse.ArgumentParser()
    d = TrainConfig()
    p.add_argument("--run-name", default=d.run_name)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--num-timesteps", type=int, default=d.num_timesteps)
    p.add_argument("--num-envs", type=int, default=d.num_envs)
    p.add_argument("--num-evals", type=int, default=d.num_evals)
    p.add_argument("--time-limit-hours", type=float, default=None)
    p.add_argument("--restore", default=None, help="checkpoints/<run>/<step> directory to resume from")
    a = p.parse_args()
    train(TrainConfig(run_name=a.run_name, out_dir=a.out_dir, num_timesteps=a.num_timesteps,
                      num_envs=a.num_envs, num_evals=a.num_evals, time_limit_hours=a.time_limit_hours,
                      restore_checkpoint_path=a.restore))


if __name__ == "__main__":
    main()
