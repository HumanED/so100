"""
Speed (and a learning check) of Stable-Baselines3 PPO on MuJoCo Warp through SoFetchWarpVecEnv.

    python -m so_arm_warp.sb3_bench --mode speed --json sb3_speed.json        # configs below, ~2 min each
    python -m so_arm_warp.sb3_bench --mode env                                 # adapter only, random actions
    python -m so_arm_warp.sb3_bench --mode learn --minutes 40 --csv learn.csv  # one longer run, grasp / success rate
    python -m so_arm_warp.sb3_bench --mode eval --model learn_model.zip --physics original full   # play back a model

Every run uses train_parallel.py's setup: VecMonitor + VecNormalize(norm_obs, norm_reward, clip_obs=10), MlpPolicy,
SB3's default PPO hyperparameters except n_steps / batch_size / n_epochs, and train_parallel's TensorboardCallback.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from .sb3_vec_env import SoFetchWarpVecEnv

# (label, num_envs, n_steps, batch_size, n_epochs, physics)
SPEED_CONFIGS = [
    ("main's settings (16k rollout, batch 64)", 1024, 16, 64, 10, "full"),
    ("4096 envs, batch 4096", 4096, 16, 4096, 10, "full"),
    ("8192 envs, batch 8192", 8192, 16, 8192, 10, "full"),
    ("8192 envs, batch 16384, 5 epochs", 8192, 16, 16384, 5, "full"),
    ("8192 envs, batch 8192, original physics", 8192, 16, 8192, 10, "original"),
]


class Timing(BaseCallback):
    """Splits wall time into rollout collection (env + policy forward) and the PPO update. Counts grasps / successes."""

    def __init__(self, csv=None):
        super().__init__()
        self.rollout_s, self.t_start, self.csv, self.rows = 0.0, None, csv, []
        self.eps, self.grasps, self.succ = 0, 0, 0

    def _on_training_start(self):
        if not hasattr(self, "t0"):   # learn() is called repeatedly; time from the first call
            self.t0 = time.time()

    def _on_rollout_start(self):
        self.t_start = time.time()

    def _on_rollout_end(self):
        self.rollout_s += time.time() - self.t_start
        if self.csv and self.eps:
            row = dict(minutes=round((time.time() - self.t0) / 60, 2), steps=self.num_timesteps,
                       ep_rew_mean=float(np.mean([e["r"] for e in self.model.ep_info_buffer])),
                       grasp_rate=self.grasps / self.eps, success_rate=self.succ / self.eps,
                       steps_per_s=round(self.num_timesteps / (time.time() - self.t0)))
            self.rows.append(row)
            print(row, flush=True)
            import pandas as pd
            pd.DataFrame(self.rows).to_csv(self.csv, index=False)
            self.eps = self.grasps = self.succ = 0

    def _on_step(self):
        for i in np.flatnonzero(self.locals["dones"]):
            info = self.locals["infos"][i]
            self.eps += 1
            self.grasps += info.get("grasped", 0)
            self.succ += info.get("is_success", 0)
        return True


def make(num_envs, physics, device):
    return VecNormalize(VecMonitor(SoFetchWarpVecEnv(num_envs, physics=physics, device=device)),
                        norm_obs=True, norm_reward=True, clip_obs=10.0)


def tb_callback():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_parallel import TensorboardCallback
    return TensorboardCallback()


def run_speed(label, num_envs, n_steps, batch, epochs, physics, device, updates, max_s):
    venv = make(num_envs, physics, device)
    model = PPO("MlpPolicy", venv, n_steps=n_steps, batch_size=batch, n_epochs=epochs, device=device, verbose=0)
    per_update = num_envs * n_steps
    model.learn(per_update, callback=tb_callback())   # warm-up: Warp compile + CUDA graph capture
    tm = Timing()
    t = time.time()
    done = 0
    while done < updates and time.time() - t < max_s:
        model.learn(per_update, callback=CallbackList([tb_callback(), tm]), reset_num_timesteps=False)
        done += 1
    wall = time.time() - t
    steps = done * per_update
    r = dict(config=label, num_envs=num_envs, n_steps=n_steps, batch_size=batch, n_epochs=epochs, physics=physics,
             updates=done, steps_per_s=round(steps / wall), rollout_steps_per_s=round(steps / tm.rollout_s),
             rollout_frac=round(tm.rollout_s / wall, 2), s_per_update=round(wall / done, 1))
    print(f"{label:45s} {r['steps_per_s']:>8,} steps/s  (collect {r['rollout_steps_per_s']:>8,}/s, "
          f"{100 * r['rollout_frac']:.0f}% of time collecting, {r['s_per_update']} s per update)", flush=True)
    venv.close()
    del model, venv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r


def run_env(num_envs, physics, device, steps=200):
    venv = SoFetchWarpVecEnv(num_envs, physics=physics, device=device)
    venv.reset()
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 64, (8, num_envs, venv.action_space.shape[0]))
    for k in range(5):
        venv.step(acts[k % 8])
    t = time.time()
    for k in range(steps):
        venv.step(acts[k % 8])
    sps = num_envs * steps / (time.time() - t)
    print(f"adapter only, {physics:8s} {num_envs:6d} envs: {sps:>10,.0f} env steps/s", flush=True)
    return dict(config=f"adapter only ({physics})", num_envs=num_envs, steps_per_s=round(sps))


def run_eval(model_path, vecnorm_path, physics, device, num_envs=1024):
    """Plays a saved SB3 model for one episode in each of num_envs worlds; deterministic and sampled actions."""
    model = PPO.load(model_path, device=device)
    out = []
    for det in (True, False):
        venv = VecNormalize.load(vecnorm_path, SoFetchWarpVecEnv(num_envs, physics=physics, device=device))
        venv.training, venv.norm_reward = False, False
        obs = venv.reset()
        grasp = np.zeros(num_envs, bool); succ = np.zeros(num_envs, bool); ret = np.zeros(num_envs)
        for _ in range(MAX_EP):
            a, _ = model.predict(obs, deterministic=det)
            obs, _, dones, infos = venv.step(a)
            ret += venv.get_original_reward()
            for i in np.flatnonzero(dones):
                grasp[i] |= bool(infos[i].get("grasped")); succ[i] |= bool(infos[i].get("is_success"))
        r = dict(model=os.path.basename(model_path), physics=physics, deterministic=det, episodes=num_envs,
                 grasp_rate=float(grasp.mean()), success_rate=float(succ.mean()), mean_return=float(ret.mean()))
        print(f"eval {r['model']} on {physics:14s} {'deterministic' if det else 'sampled      '}: grasp {100 * r['grasp_rate']:5.1f}% "
              f"success {100 * r['success_rate']:5.1f}% return {r['mean_return']:+.2f}", flush=True)
        out.append(r)
    return out


MAX_EP = 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["speed", "env", "learn", "eval"], default="speed")
    p.add_argument("--physics", nargs="+", default=["original_20it"], help="learn: first entry; env / eval: each one")
    p.add_argument("--model", default=None, help="eval: SB3 .zip (VecNormalize stats: <model minus _model>_vecnorm.pkl)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--max-s", type=float, default=150)
    p.add_argument("--minutes", type=float, default=40)
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--json", default=None)
    p.add_argument("--csv", default="sb3_learn.csv")
    a = p.parse_args()
    if a.device.startswith("cuda"):
        print("GPU:", torch.cuda.get_device_name(torch.device(a.device)), flush=True)

    results = []
    if a.mode == "env":
        for physics in a.physics:
            for n in (1024, 8192):
                results.append(run_env(n, physics, a.device))
    elif a.mode == "speed":
        for cfg in SPEED_CONFIGS:
            try:
                results.append(run_speed(*cfg, device=a.device, updates=a.updates, max_s=a.max_s))
            except Exception as e:   # keep going: one failing config shouldn't lose the others
                print(f"{cfg[0]}: FAILED {type(e).__name__}: {e}", flush=True)
                results.append(dict(config=cfg[0], failed=f"{type(e).__name__}: {e}"))
            if a.json:
                json.dump(results, open(a.json, "w"), indent=1)
    elif a.mode == "eval":
        vn = a.model.replace("_model.zip", "_vecnorm.pkl")
        for physics in a.physics:
            results += run_eval(a.model, vn, physics, a.device, num_envs=min(a.num_envs, 1024))
    else:
        venv = make(a.num_envs, a.physics[0], a.device)
        model = PPO("MlpPolicy", venv, n_steps=16, batch_size=a.batch_size, n_epochs=10, device=a.device, verbose=0)
        tm = Timing(csv=a.csv)
        t = time.time()
        per_update = a.num_envs * 16
        while time.time() - t < a.minutes * 60:
            model.learn(per_update * 5, callback=CallbackList([tb_callback(), tm]), reset_num_timesteps=False)
        model.save(os.path.splitext(a.csv)[0] + "_model")
        venv.save(os.path.splitext(a.csv)[0] + "_vecnorm.pkl")
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
