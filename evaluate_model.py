import os
from statistics import mean, stdev

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm.auto import tqdm

from so_arm_rl.envs.fetch import SoFetchEnv

"""
Code to compute statistics on model performance.

"""
# SETTINGS
num_ep_evaluate = 100
model_folder = "PPO-9-fetch-varun/1000000"


def make_env():
    def inner():
        env = SoFetchEnv()
        return env

    return inner


def main():
    # Get model
    model_path = os.path.join(os.path.dirname(__file__), "models", model_folder + ".zip")
    if not os.path.exists(model_path):
        raise Exception("Error: model not found")

    # Get environment obs mean and std used to Normalize the environment
    vec_stats_path = os.path.join(os.path.dirname(__file__), "vec_norm_stats", model_folder + ".pkl")
    if not os.path.exists(vec_stats_path):
        raise Exception("Error: VecNormalize mean and std stats file not found")

    # Load environment
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize.load(vec_stats_path, vec_env)
    vec_env.num_envs = 1
    vec_env.ret_rms.count = 1
    vec_env.obs_rms.count = 1
    vec_env.training = False
    vec_env.norm_reward = False  # Ensures fair comparison between different runs
    custom_objects = {"lr_schedule": 3e-4, "clip_range": 0.2}
    model = PPO.load(model_path, env=vec_env, custom_objects=custom_objects)

    # episode_info stores per-episode totals for each type of reward and length of each episode.
    episode_info = {}
    # cur_episode_info is the accumulator for one episode
    cur_episode_info = {}

    vec_env.reset()
    info = vec_env.get_attr("info")[0]
    for key in info.keys():
        if key.startswith("rew") or key == "is_success":
            episode_info[key] = []
            cur_episode_info[key] = 0
    episode_info["ep_rew"] = []
    episode_info["total_timesteps"] = []

    for _ in tqdm(range(num_ep_evaluate)):
        # Run one episode.
        done = False
        obs_vector = vec_env.reset()
        episode_reward = 0
        total_timesteps = 0
        while not done:
            action_vector, _ = model.predict(obs_vector)
            obs_vector, reward_vector, done_vector, info_vector = vec_env.step(action_vector)
            done = done_vector[0]
            total_timesteps += 1
            for k, v in info_vector[0].items():
                if k == "is_success" and v == 1:
                    cur_episode_info[k] = 1
                elif k.startswith("rew"):
                    cur_episode_info[k] += v
            episode_reward += reward_vector[0]

        # Write accumulated values for this episode and reset cur_episode_info to episode_info
        for k, v in cur_episode_info.items():
            episode_info[k].append(v)
            cur_episode_info[k] = 0
        episode_info["ep_rew"].append(episode_reward)
        episode_info["total_timesteps"].append(total_timesteps)

    for key, value in episode_info.items():
        print(key, value)
    print("-------------------------------------")
    for k, v in episode_info.items():
        print(f"{k :35} mean: {mean(v):.3f} std: {stdev(v):.2f}")
    vec_env.close()


if __name__ == "__main__":
    main()
