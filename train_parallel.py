import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers.transform_observation import TransformObservation
import numpy as np
import gymnasium

from Shadow_Gym2.shadow_gym.envs.shadow_env_mujoco import ShadowEnvMujoco

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
this_run_name = "PPO-34-shadowgym-ethan"
saving_timesteps_interval = 250_000
start_saving = 1_000_000
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
    env = ShadowEnvMujoco()
    env = NormalizeObservation(env)
    env = TransformObservation(env, clip_observation, env.observation_space)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def main():
    print(logs_dir)
    if vectorized_env:
        num_envs = os.cpu_count()  # Number of parallel environments. Equal to number of CPU cores
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
        model.learn(saving_timesteps_interval, tb_log_name=this_run_name, reset_num_timesteps=False)
        timesteps += saving_timesteps_interval
        if timesteps >= start_saving:
            model.save(os.path.join(models_dir, this_run_name, str(timesteps)))


if __name__ == "__main__":
    main()
