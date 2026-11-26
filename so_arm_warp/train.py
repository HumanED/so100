"""
Train PPO on the Warp fetch env (NVIDIA GPU):

    python -m so_arm_warp.train --run-name carl-padmid-1 --reward pad_midpoint --case full --num-envs 8192 --hours 11

Writes to <out-dir>/<run-name>/:
    config.json         everything needed to rebuild the env and policy
    model_<step>.pt     checkpoints (policy + observation normaliser + optimiser), plus model_final.pt
    log.csv             per-update metrics: steps/s, episode return, grasp/success rates, losses
Resume with --resume path/to/model_X.pt.
"""
import argparse
import csv
import json
import os
import time
from dataclasses import asdict

import torch

from .env import FetchWarpEnv
from .physics import CASES, PhysicsConfig
from .ppo import PPO, PPOConfig
from .rewards import REWARDS


def train(run_name, out_dir="warp_out", case="full", reward="default", num_envs=8192, total_steps=2_000_000_000,
          hours=None, save_every_steps=50_000_000, resume=None, device=None, seed=0, ls_iterations=50,
          fixed_goal=(-0.3, -0.2, 0.025), ppo_cfg: PPOConfig = PPOConfig(), log_every=1):
    torch.manual_seed(seed)
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    phys = PhysicsConfig(case=case, ls_iterations=ls_iterations)
    env = FetchWarpEnv(num_envs, physics=phys, reward=reward, fixed_goal=fixed_goal, device=device, seed=seed)
    algo = PPO(env.obs_size, env.action_size, num_envs, env.device, ppo_cfg)
    start_step = 0
    if resume:
        ck = torch.load(resume, map_location=env.device)
        algo.ac.load_state_dict(ck["model"]); algo.opt.load_state_dict(ck["optimizer"])
        start_step = ck.get("step", 0)
        print(f"Resumed from {resume} at step {start_step:,}")
    config = dict(run_name=run_name, case=case, reward=reward, num_envs=num_envs, seed=seed, fixed_goal=fixed_goal,
                  physics=asdict(phys), ppo=asdict(ppo_cfg), obs_size=env.obs_size, action_size=env.action_size)
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2, default=list)

    def save(step, name=None):
        path = os.path.join(run_dir, name or f"model_{step}.pt")
        torch.save({"model": algo.ac.state_dict(), "optimizer": algo.opt.state_dict(), "step": step, "config": config}, path)
        return path

    print(f"Training '{run_name}': case={case} reward={reward} envs={num_envs} on {env.device}")
    log_path = os.path.join(run_dir, "log.csv")
    rows = []
    if resume and os.path.exists(log_path):
        rows = list(csv.DictReader(open(log_path)))
    obs = env.reset()
    step, t0, next_save = start_step, time.time(), (start_step // save_every_steps + 1) * save_every_steps
    ep_stats, it = {}, 0
    try:
        while step < total_steps:
            t_it = time.time()
            for _ in range(ppo_cfg.steps_per_env):
                action = algo.act(obs)
                obs, reward, done, info = env.step(action)
                algo.record(reward, done, info["timeout"])
                if "episode" in info:
                    for k, v in info["episode"].items():
                        ep_stats.setdefault(k, []).append(v)
            collect_s = time.time() - t_it
            stats = algo.update(obs)
            step += num_envs * ppo_cfg.steps_per_env
            it += 1
            if it % log_every == 0:
                row = {"step": step, "time_min": round((time.time() - t0) / 60, 2),
                       "steps_per_s": round(num_envs * ppo_cfg.steps_per_env / (time.time() - t_it)),
                       "collect_frac": round(collect_s / (time.time() - t_it), 2)}
                if ep_stats:
                    cat = {k: torch.cat(v) for k, v in ep_stats.items()}
                    row["episodes"] = int(cat["reward"].numel())
                    row["ep_return"] = round(float(cat["reward"].mean()), 3)
                    for k in ("grasp", "success"):
                        if k in cat:
                            row[f"{k}_rate"] = round(float((cat[k] > 0).float().mean()), 4)
                    for k, v in cat.items():
                        if k.startswith("rew_"):
                            row[f"ep_{k}"] = round(float(v.mean()), 3)
                    ep_stats = {}
                row.update({k: round(v, 5) for k, v in stats.items()})
                rows.append(row)
                keys = list(dict.fromkeys(k for r in rows for k in r))  # columns appear once episodes finish
                with open(log_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
                print(f"[{row['time_min']:7.1f} min] step {step:>13,} | {row['steps_per_s']:>8,} steps/s | "
                      f"return {row.get('ep_return', float('nan')):8.2f} | grasp {row.get('grasp_rate', 0):.3f} | "
                      f"success {row.get('success_rate', 0):.3f} | std {stats['action_std']:.2f} | lr {stats['lr']:.1e}", flush=True)
            if step >= next_save:
                save(step); next_save += save_every_steps
            if hours is not None and time.time() - t0 > hours * 3600:
                print("Time limit reached"); break
    except KeyboardInterrupt:
        print("Interrupted")
    final = save(step, "model_final.pt")
    print(f"Saved {final} ({step:,} steps, {(time.time() - t0) / 60:.1f} min)")
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-dir", default="warp_out")
    p.add_argument("--case", default="full", choices=CASES)
    p.add_argument("--reward", default="default", choices=sorted(REWARDS))
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--total-steps", type=int, default=2_000_000_000)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--save-every-steps", type=int, default=50_000_000)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-goal", action="store_true")
    a = p.parse_args()
    train(a.run_name, a.out_dir, a.case, a.reward, a.num_envs, a.total_steps, a.hours, a.save_every_steps,
          a.resume, a.device, a.seed, fixed_goal=None if a.random_goal else (-0.3, -0.2, 0.025))


if __name__ == "__main__":
    main()
