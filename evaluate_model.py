import os
from statistics import mean, stdev

import numpy as np
from gymnasium.wrappers import NormalizeObservation, TransformObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm.auto import tqdm

from so_arm_rl.envs.fetch import SoFetchEnv

"""
Code to compute statistics on model performance.

"""
# SETTINGS
num_ep_evaluate = 100
model_folder = "PPO-4b-fetch-ethan/9750000"
FLAGS_TO_IGNORE = ("reset_flag")

def make_env():
    def inner():
        env = SoFetchEnv()
        return env
    return inner


def main():
    # Get model
    model_path = os.path.join(os.path.dirname(__file__), "models", model_folder +".zip")
    if not os.path.exists(model_path):
        raise Exception("Error: model not found")

    vec_stats_path = os.path.join(os.path.dirname(__file__), "vec_norm_stats", model_folder + ".pkl")
    if not os.path.exists(vec_stats_path):
        raise Exception("Error: VecNormalize mean and std stats file not found")

    # Load environment
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize.load(vec_stats_path, vec_env)
    # now fix the mismatch
    vec_env.num_envs = 1
    vec_env.ret_rms.count = 1 # if you normalize rewards
    vec_env.obs_rms.count = 1
    vec_env.training = False
    vec_env.norm_reward = False  # Ensures fair comparison between different runs
    model = PPO.load(model_path, env=vec_env)

    # episode_info stores the key-value pair of the last info returned from each episode e.g. total total_timesteps key
    episode_info = {}
    episode_rewards = []
    obs_vector = vec_env.reset()
    info = vec_env.get_attr("info")[0]
    for key in info.keys():
        if (key not in FLAGS_TO_IGNORE):
            episode_info[key] = []

    for _ in tqdm(range(num_ep_evaluate)):
        terminated = False
        truncated = False
        obs_vector = vec_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action_vector, _ = model.predict(obs_vector)
            temp_info = vec_env.get_attr("info")[0]
            obs_vector, reward_vector, done_vector, info_vector = vec_env.step(action_vector)
            done = done_vector[0]
            reward = reward_vector[0]
            info = info_vector[0]
            # if not done:
            #     info = temp_info
            episode_reward += reward
        for key in episode_info.keys():
            episode_info[key].append(info[key])
        episode_rewards.append(episode_reward)

    print(f"episode_rewards", episode_rewards)
    for key, value in episode_info.items():
        print(key, value)
    print("-------------------------------------")
    print(f"episode_rewards                     mean: {mean(episode_rewards):.3f} std: {stdev(episode_rewards):.3f} ")
    for key in episode_info.keys():
        print(f"{key:35} mean: {mean(episode_info[key]):.3f} std: {stdev(episode_info[key]):.2f}")
    vec_env.close()


if __name__ == "__main__":
    main()
