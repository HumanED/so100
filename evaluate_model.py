import os
from statistics import mean, stdev

import numpy as np
from gymnasium.wrappers import NormalizeObservation, TransformObservation
from stable_baselines3 import PPO
from tqdm.auto import tqdm

from so_arm_rl.envs.fetch import SoFetchEnv

"""
Code to compute statistics on model performance.

"""
# SETTINGS
num_ep_evaluate = 100
model_folder_zip = "PPO-2b-fetch-ethan/10250000.zip"
FLAGS_TO_IGNORE = ("reset_flag")

def make_env():
    """Creates gymnasium environment for evaluation with necessary wrappers"""

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
    return env


def main():
    # Get model
    model_path = os.path.join(os.path.dirname(__file__), "models", model_folder_zip)
    if not os.path.exists(model_path):
        raise Exception("Error: model not found")

    env = make_env()
    model = PPO.load(model_path, env=env)

    # episode_info stores the key-value pair of the last info returned from each episode e.g. total total_timesteps key
    episode_info = {}
    episode_rewards = []
    obs, info = env.reset()
    for key in info.keys():
        if (key not in FLAGS_TO_IGNORE):
            episode_info[key] = []

    for _ in tqdm(range(num_ep_evaluate)):
        terminated = False
        truncated = False
        obs, info = env.reset()
        episode_reward = 0
        while not terminated and not truncated:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        for key in info.keys():
            episode_info[key].append(info[key])
        episode_rewards.append(episode_reward)

    print(f"episode_rewards", episode_rewards)
    for key, value in episode_info.items():
        print(key, value)
    print("-------------------------------------")
    print(f"episode_rewards                     mean: {mean(episode_rewards):.3f} std: {stdev(episode_rewards):.3f} ")
    for key in episode_info.keys():
        print(f"{key:35} mean: {mean(episode_info[key]):.3f} std: {stdev(episode_info[key]):.2f}")
    env.close()


if __name__ == "__main__":
    main()
