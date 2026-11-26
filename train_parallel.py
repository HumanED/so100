import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecMonitor

from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv
from so_arm_rl.envs.fetch.so_arm_fetch_vec_env import SoFetchVecEnv

"""
Created by Ethan Cheam
"""

# SETTINGS
vectorized_env = True  # Set to True to use multiple environments
# "single_process": SoFetchVecEnv, all envs in one process stepped by MuJoCo's multithreaded rollout (~1.7x faster
#                   training on an M3; same observations/rewards/infos as SubprocVecEnv, see tests/test_vec_env.py)
# "subproc":        one process per env (SubprocVecEnv), the previous setup
# "warp":           SoFetchWarpVecEnv, thousands of envs on one CUDA GPU with MuJoCo Warp (needs `pip install
#                   mujoco==4.0.1 mujoco-warp==4.0.1` and an NVIDIA GPU, e.g. Kaggle T4). Same observations, actions
#                   and reward as SoFetchEnv; see so_arm_warp/sb3_vec_env.py and docs/TRAINING_OPTIONS.md
vec_env_backend = "single_process"
single_process_num_envs = 32
rollout_size = 16384  # env steps per PPO update: PPO's default 2048 per env x 8 envs, as "subproc" on an 8-core machine
# "warp" backend only: SB3's default batch_size of 64 would mean thousands of tiny gradient steps per update
warp_num_envs = 8192
warp_n_steps = 16           # rollout = warp_num_envs x warp_n_steps = 131,072 env steps per PPO update
warp_batch_size = 8192
warp_physics = "full"       # "full": all collision shapes, cheaper solver; "original": SoFetchEnv's exact solver (slower)
start_from_existing = False
old_model_file = "PPO-4-fetch-ethan/4750000"
# When you want to train PPO-20-shadowgym-ethan more and create PPO-21-shadowgym-ethan
# Set old_model_file="PPO-21-shadowgym-ethan" and this_run_name="PPO-20-shadowgym-ethan"

# Run name should have model, unique number, and your name
this_run_name = "PPO-8-fetch-ethan"
saving_timesteps_interval = 500_000
start_saving = 1_000_000
# Seed sets random number generators in model and environment
seed = 1


def make_env(rank, seed):
    """Creates gymnasium environment with necessary wrappers"""
    def inner():
        env = SoFetchEnv()
        # Ensure each environment has a different seed
        env.reset(seed=rank + seed)
        return env
    return inner

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        # Cumulative rewards of sub-reward component
        self.sub_rews_cumul = {}
        # Cumulative rewards of sub-rewards for ONE episode
        self.sub_rews_buffer = {}
        self.buffer_idx = 0
        self.ignore_reset_flag = True

    def _on_training_start(self) -> None:
        for k in self.training_env.get_attr("info")[0].keys():
            if k.startswith("rew_") or k.startswith("debug_"):
                self.sub_rews_cumul[k] = 0
                self.sub_rews_buffer[k] = np.zeros(self.training_env.get_attr("MAX_TIMESTEPS")[0])

    def _on_step(self) -> bool:
        # Warning!: In vectorized environments, on last step(), the reset() is called before _on_step
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
        # This code records each type of reward generated at each step.
        info = self.training_env.get_attr("info")[0]

        if info["reset_flag"] and not self.ignore_reset_flag:
            # Episode is terminated. Use buffer to only add records from completed episodes to cumulative sum (like ep_rew_mean)
            # The last timestep was not recorded so duplicate n-1th entry to form nth entry
            self.episode_count += 1
            for k, v in info.items():
                if k.startswith("rew_"):
                    self.sub_rews_buffer[k][self.buffer_idx] = self.sub_rews_buffer[k][self.buffer_idx - 1] # Duplicate last entry
                    self.sub_rews_cumul[k] += np.sum(self.sub_rews_buffer[k])
                    self.sub_rews_buffer[k] = np.zeros(self.training_env.get_attr("MAX_TIMESTEPS")[0])
            self.buffer_idx = 0
        else:
            self.ignore_reset_flag = False
            # Record sub rewards for this timestep
            for k, v in info.items():
                if k.startswith("rew_"):
                    self.sub_rews_buffer[k][self.buffer_idx] = v
            self.buffer_idx += 1

        return True

    def _on_rollout_end(self) -> None:
        # Tensorboard cannot print numpy floats.
        for k, v in self.sub_rews_cumul.items():
            if self.episode_count > 0:
                self.logger.record(f"rollout/{k}_mean", float(self.sub_rews_cumul[k]) / self.episode_count)
            else:
                self.logger.record(f"rollout/{k}_mean", 0)
        self.episode_count = 0
        for k in self.sub_rews_cumul.keys():
            self.sub_rews_cumul[k] = 0
def main(models_dir, vec_stats_dir, logs_dir):
    ppo_kwargs = {}
    if vectorized_env and vec_env_backend == "warp":
        from so_arm_warp.sb3_vec_env import SoFetchWarpVecEnv
        vec_env = SoFetchWarpVecEnv(warp_num_envs, physics=warp_physics)
        ppo_kwargs = {"batch_size": warp_batch_size, "device": str(vec_env.device)}
        print(f"Running {warp_num_envs} envs on {vec_env.device} with MuJoCo Warp ({warp_physics} physics)")
    elif vectorized_env and vec_env_backend == "single_process":
        vec_env = SoFetchVecEnv(single_process_num_envs)
        print(f"Running {single_process_num_envs} envs in one process on {vec_env.n_threads} threads")
    elif vectorized_env:
        num_envs = os.cpu_count() # Number of parallel environments. Equal to number of CPU cores
        print(f"Running on {num_envs} cores")
        vec_env = SubprocVecEnv([make_env(i, seed) for i in range(num_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0, seed)])
    # Keep env steps per PPO update the same however many envs run in parallel
    if vectorized_env and vec_env_backend == "warp":
        n_steps = warp_n_steps
    elif vectorized_env and vec_env_backend == "single_process":
        n_steps = max(1, rollout_size // vec_env.num_envs)
    else:
        n_steps = 2048

    vec_env = VecMonitor(vec_env, filename=None)


    # Load existing model or create a new model
    if start_from_existing:
        vec_env = VecNormalize.load(os.path.join(vec_stats_dir, old_model_file + ".pkl"), vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        custom_objects = {"lr_schedule": 3e-4, "clip_range": 0.2, "n_steps": n_steps, **{k: v for k, v in ppo_kwargs.items() if k != "device"}}
        model = PPO.load(os.path.join(models_dir, old_model_file), vec_env, seed=seed, tensorboard_log=os.path.normpath(logs_dir), custom_objects=custom_objects, device=ppo_kwargs.get("device", "auto"))
    else:
        # Normalize observation and rewards.
        # VecNormalize computes a RunningMeanStd (mean, std number) for observations and a RunningMeanStd for rewards
        # Uses the mean and std for z-normalisation. norm_obs_i = (raw_obs_i - mean_obs_i) / std_obs_i
        # i means the i-th dimension of the observation. Also means i-th value of the obs ndarray.
        vec_env = VecNormalize(vec_env,
                               norm_obs=True,
                               norm_reward=True,
                               clip_obs=10.0)
        model = PPO(policy="MlpPolicy", env=vec_env, n_steps=n_steps, tensorboard_log=os.path.normpath(logs_dir), verbose=2, **ppo_kwargs)

    # Training loop
    timesteps = 0
    while True:
        model.learn(saving_timesteps_interval, tb_log_name=this_run_name, reset_num_timesteps=False, callback=TensorboardCallback())
        timesteps += saving_timesteps_interval
        if timesteps >= start_saving:
            model.save(os.path.join(models_dir, this_run_name, str(timesteps)))
            vec_env.save(os.path.join(vec_stats_dir, this_run_name, str(timesteps) + ".pkl"))


if __name__ == "__main__":
    # Set up folders to store models and logs
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not start_from_existing and os.path.exists(f"{models_dir}/{this_run_name}"):
        raise Exception(
            "Error: model folder already exists. Change run_name to prevent overriding existing model folder")
    if not start_from_existing and os.path.exists(f"{logs_dir}/{this_run_name}"):
        raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")

    vec_stats_dir = os.path.join(os.path.dirname(__file__), 'vec_norm_stats')
    if (not start_from_existing) and os.path.exists(f"{vec_stats_dir}/{this_run_name}"):
        raise Exception(
            "Error: vec_stats folder already exists. Change run_name to prevent overriding existing vec_stats folder")
    os.mkdir(os.path.join(models_dir, this_run_name))
    os.mkdir(os.path.join(vec_stats_dir, this_run_name))
    main(models_dir, vec_stats_dir, logs_dir)
