import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecMonitor

from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv

"""
Created by Ethan Cheam
"""

# SETTINGS
vectorized_env = True  # Set to True to use multiple environments
start_from_existing = False
old_model_file = "PPO-4-fetch-ethan/4750000"
# When you want to train PPO-20-shadowgym-ethan more and create PPO-21-shadowgym-ethan
# Set old_model_file="PPO-21-shadowgym-ethan" and this_run_name="PPO-20-shadowgym-ethan"

# Run name should have model, unique number, and your name
this_run_name = "PPO-12-fetch-ethan"
saving_timesteps_interval = 500_000
start_saving = 1_000_000
# Seed sets random number generators in model and environment
seed = 1
# If running on DICE machine overnight, set to 10_000_000. Run forever is None
DICE_MAX_LIMIT = 10_000_000


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
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = None
        try:
            self.tb_formatter = next(
                formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
        except Exception as e:
            self.logger.warn("Unable to create tensorboard output format: {}".format(e))
        for k in self.training_env.get_attr("info")[0].keys():
            if k.startswith("rew_") or k.startswith('debug_'):
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
                if k.startswith("rew_") or k.startswith('debug_'):
                    self.sub_rews_buffer[k][self.buffer_idx] = self.sub_rews_buffer[k][self.buffer_idx - 1] # Duplicate last entry
                    self.sub_rews_cumul[k] += np.sum(self.sub_rews_buffer[k])
                    self.sub_rews_buffer[k] = np.zeros(self.training_env.get_attr("MAX_TIMESTEPS")[0])
            self.buffer_idx = 0
        else:
            self.ignore_reset_flag = False
            # Record sub rewards for this timestep

            for k, v in info.items():
                if k.startswith("rew_") or k.startswith('debug_'):
                    self.sub_rews_buffer[k][self.buffer_idx] = v
            self.buffer_idx += 1

        return True

    def _on_rollout_end(self) -> None:
        # Tensorboard cannot print numpy floats.

        debug_payload = dict()
        for k, v in self.sub_rews_cumul.items():
            if self.episode_count > 0:
                self.logger.record(f"rollout/{k}_mean", float(self.sub_rews_cumul[k]) / self.episode_count)
                if k.startswith("debug_"):
                    debug_payload[k[6:]] = float(self.sub_rews_cumul[k]) / self.episode_count
            else:
                self.logger.record(f"rollout/{k}_mean", 0)
                if k.startswith("debug_"):
                    debug_payload[k[6:]] = 0
        self.tb_formatter.writer.add_scalars('debug', debug_payload, self.num_timesteps)


        self.episode_count = 0
        for k in self.sub_rews_cumul.keys():
            self.sub_rews_cumul[k] = 0

def main(models_dir, vec_stats_dir, logs_dir):
    if vectorized_env:
        num_envs = os.cpu_count() # Number of parallel environments. Equal to number of CPU cores
        print(f"Running on {num_envs} cores")
        vec_env = SubprocVecEnv([make_env(i, seed) for i in range(num_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0, seed)])

    vec_env = VecMonitor(vec_env, filename=None)


    # Load existing model or create a new model
    if start_from_existing:
        vec_env = VecNormalize.load(os.path.join(vec_stats_dir, old_model_file + ".pkl"), vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        custom_objects = {"lr_schedule": 3e-4, "clip_range": 0.2}
        model = PPO.load(os.path.join(models_dir, old_model_file), vec_env, seed=seed, tensorboard_log=os.path.normpath(logs_dir), custom_objects=custom_objects)
    else:
        # Normalize observation and rewards.
        # VecNormalize computes a RunningMeanStd (mean, std number) for observations and a RunningMeanStd for rewards
        # Uses the mean and std for z-normalisation. norm_obs_i = (raw_obs_i - mean_obs_i) / std_obs_i
        # i means the i-th dimension of the observation. Also means i-th value of the obs ndarray.
        vec_env = VecNormalize(vec_env,
                               norm_obs=True,
                               norm_reward=True,
                               clip_obs=10.0)
        model = PPO(policy="MlpPolicy", env=vec_env, tensorboard_log=os.path.normpath(logs_dir), verbose=2)

    # Training loop
    timesteps = 0
    while DICE_MAX_LIMIT is None or timesteps < DICE_MAX_LIMIT:
        model.learn(saving_timesteps_interval, tb_log_name=this_run_name, reset_num_timesteps=False, callback=TensorboardCallback())
        timesteps += saving_timesteps_interval
        if timesteps >= start_saving:
            model.save(os.path.join(models_dir, this_run_name, str(timesteps)))
            vec_env.save(os.path.join(vec_stats_dir, this_run_name, str(timesteps) + ".pkl"))
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
