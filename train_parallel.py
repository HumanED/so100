import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers.transform_observation import TransformObservation
import numpy as np
import gymnasium

from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv
from stable_baselines3.common.callbacks import BaseCallback
"""
Created by Ethan Cheam
"""

# SETTINGS
vectorized_env = True  # Set to True to use multiple environments
start_from_existing = False
old_model_file = "PPO-33b-shadowgym-ethan/48500000"
# When you want to train PPO-20-shadowgym-ethan more and create PPO-21-shadowgym-ethan
# Set old_model_file="PPO-21-shadowgym-ethan" and this_run_name="PPO-20-shadowgym-ethan"

# Run name should have model, unique number, and your name
# PPO 34 is with ema
this_run_name = "PPO-0d-fetch-ethan"
saving_timesteps_interval = 250_000
start_saving = 500_000
# Seed sets random number generators in model and environment
seed = 1

# Set up folders to store models and logs
models_dir = os.path.join(os.path.dirname(__file__), 'models')
logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
if not start_from_existing and os.path.exists(f"{models_dir}/{this_run_name}"):
    raise Exception("Error: model folder already exists. Change run_name to prevent overriding existing model folder")
if not start_from_existing and os.path.exists(f"{logs_dir}/{this_run_name}"):
    raise Exception("Error: log folder already exists. Change run_name to prevent overriding existing log folder")

def make_env():
    """Creates gymnasium environment with necessary wrappers"""
    def clip_observation(obs):
        """
        clips observation to within 5 standard deviations of the mean
        Refer to section D.1 of Open AI paper
        """
        return np.clip(obs, a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))
    # env = gymnasium.make("ShadowEnv-v1")
    env = SoFetchEnv()
    env = NormalizeObservation(env)
    env = TransformObservation(env, clip_observation, env.observation_space)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

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
            if k.startswith("rew_"):
                self.sub_rews_cumul[k] = 0
                self.sub_rews_buffer[k] = np.zeros(self.training_env.get_attr("MAX_TIMESTEPS")[0])

    def _on_step(self) -> bool:
        # Warning!: In vectorized environments, on last step(), the reset() is called before _on_step
        # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
        # This code records rewards for the
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

def main():
    print(logs_dir)
    if vectorized_env:
        num_envs = os.cpu_count() # Number of parallel environments. Equal to number of CPU cores
        print(f"Running on {num_envs} cores")
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
    else:
        env = make_env()



    # Load existing model or create a new model
    if start_from_existing:

        model = PPO.load(os.path.join(models_dir, old_model_file), env, seed=seed, tensorboard_log=os.path.normpath(logs_dir))
    else:
        model = PPO(policy="MlpPolicy", env=env, tensorboard_log=os.path.normpath(logs_dir), verbose=1)

    # Training loop
    timesteps = 0
    while True:
        model.learn(saving_timesteps_interval, tb_log_name=this_run_name, reset_num_timesteps=False, callback=TensorboardCallback())
        timesteps += saving_timesteps_interval
        if timesteps >= start_saving:
            model.save(os.path.join(models_dir, this_run_name, str(timesteps)))


if __name__ == "__main__":
    main()
