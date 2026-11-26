"""
Recommended training: Stable-Baselines3 PPO on MuJoCo Warp (SoFetchWarpVecEnv), on one CUDA GPU.

Same setup as train_parallel.py (VecMonitor + VecNormalize, MlpPolicy, TensorboardCallback, SB3 default PPO
hyperparameters) except n_steps=16 / batch_size=8192 for 8192 envs. Default physics "original_20it": SoFetchEnv's
model and solver with 20 iterations instead of 100, ~2.3x SB3 + SoFetchVecEnv; its policies work in SoFetchEnv
(docs/TRAINING_OPTIONS.md section D).

    python -m so_arm_warp.train_sb3 --run-name carl-default-1 --hours 2
    python -m so_arm_warp.train_sb3 --run-name carl-x-1 --reward my_reward --physics full_elliptic --hours 2
    python -m so_arm_warp.train_sb3 --run-name carl-default-2 --resume warp_sb3_out/models/carl-default-1/final.zip

Output (same layout as train_parallel.py, so evaluate_model.py / visualise_model.py can read it):
    <out-dir>/models/<run>/<steps>.zip, final.zip          SB3 checkpoints
    <out-dir>/vec_norm_stats/<run>/<steps>.pkl, final.pkl  VecNormalize statistics (needed to run a model)
    <out-dir>/logs/<run>/log.csv                           steps, episode return, grasp / success rate, steps/s
    <out-dir>/logs/<run>_N/                                tensorboard (same tags as train_parallel.py)
    <out-dir>/logs/<run>/config.json
"""
import argparse
import json
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from .physics import CASES
from .rewards import REWARDS
from .sb3_bench import Timing, tb_callback
from .sb3_vec_env import SoFetchWarpVecEnv


def train(run_name, out_dir="warp_sb3_out", physics="original_20it", reward="default", num_envs=8192, n_steps=16,
          batch_size=8192, n_epochs=10, hours=None, total_steps=10**12, save_every_steps=25_000_000, resume=None,
          device=None, seed=1):
    models, stats, logs = (os.path.join(out_dir, d) for d in ("models", "vec_norm_stats", "logs"))
    for d in (models, stats, logs):
        os.makedirs(os.path.join(d, run_name), exist_ok=True)
    cfg = dict(run_name=run_name, physics=physics, reward=reward, num_envs=num_envs, n_steps=n_steps,
               batch_size=batch_size, n_epochs=n_epochs, resume=resume, seed=seed)
    json.dump(cfg, open(os.path.join(logs, run_name, "config.json"), "w"), indent=1)

    venv = VecMonitor(SoFetchWarpVecEnv(num_envs, physics=physics, reward=reward, device=device, seed=seed))
    dev = str(venv.venv.device)
    if resume:
        venv = VecNormalize.load(resume.replace("/models/", "/vec_norm_stats/").replace(".zip", ".pkl"), venv)
        venv.training, venv.norm_reward = True, True
        model = PPO.load(resume, venv, device=dev, seed=seed, tensorboard_log=logs,
                         custom_objects={"n_steps": n_steps, "batch_size": batch_size, "n_epochs": n_epochs})
        print(f"Resumed from {resume} at step {model.num_timesteps:,}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, device=dev,
                    seed=seed, tensorboard_log=logs, verbose=0)
    print(f"Training '{run_name}': physics={physics} reward={reward} envs={num_envs} on {dev}", flush=True)

    timing = Timing(csv=os.path.join(logs, run_name, "log.csv"))
    t0, next_save = time.time(), (model.num_timesteps // save_every_steps + 1) * save_every_steps
    chunk = num_envs * n_steps * 10   # ~1.3M steps between time-limit checks
    try:
        while model.num_timesteps < total_steps and (hours is None or time.time() - t0 < hours * 3600):
            model.learn(chunk, callback=CallbackList([tb_callback(), timing]), tb_log_name=run_name,
                        reset_num_timesteps=False)
            if model.num_timesteps >= next_save:
                model.save(os.path.join(models, run_name, str(model.num_timesteps)))
                venv.save(os.path.join(stats, run_name, f"{model.num_timesteps}.pkl"))
                next_save += save_every_steps
    except KeyboardInterrupt:
        print("Interrupted")
    model.save(os.path.join(models, run_name, "final"))
    venv.save(os.path.join(stats, run_name, "final.pkl"))
    print(f"Saved {models}/{run_name}/final.zip ({model.num_timesteps:,} steps, {(time.time() - t0) / 60:.1f} min)")
    return os.path.join(models, run_name, "final.zip")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-dir", default="warp_sb3_out")
    p.add_argument("--physics", default="original_20it", choices=CASES)
    p.add_argument("--reward", default="default", choices=sorted(REWARDS))
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--total-steps", type=int, default=10**12)
    p.add_argument("--save-every-steps", type=int, default=25_000_000)
    p.add_argument("--resume", default=None, help="a .zip under .../models/<run>/; its .pkl is found in vec_norm_stats")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    train(a.run_name, a.out_dir, a.physics, a.reward, a.num_envs, a.n_steps, a.batch_size, a.n_epochs, a.hours,
          a.total_steps, a.save_every_steps, a.resume, a.device, a.seed)


if __name__ == "__main__":
    main()
