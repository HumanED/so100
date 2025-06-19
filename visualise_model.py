import os
import time
from statistics import mean, stdev
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv

# SETTINGS
model_folder = "PPO-7-fetch-ethan/5500000"  # no .zip
extra_delay = 0  # seconds


def make_env():
    def inner():
        env = SoFetchEnv(render_mode="human")
        return env

    return inner


def main():
    model_path = os.path.join(os.path.dirname(__file__), "models", model_folder + ".zip")
    if not os.path.exists(model_path):
        raise Exception("Error: model not found")
    vec_stats_path = os.path.join(os.path.dirname(__file__), "vec_norm_stats", model_folder + ".pkl")
    if not os.path.exists(vec_stats_path):
        raise Exception("Error: VecNormalize mean and std stats file not found")

    # Load environment
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize.load(vec_stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False  # Ensures fair comparison between different runs

    # Load model
    model = PPO.load(model_path, env=vec_env)

    # Compute dt value
    temp_env = SoFetchEnv()
    dt = temp_env.N_SUBSTEPS * temp_env.model.opt.timestep

    # cur_episode_info is the accumulator for one episode
    cur_episode_info = {}
    vec_env.reset()
    info = vec_env.get_attr("info")[0]
    for key in info.keys():
        if key.startswith("rew") or key == "is_success":
            cur_episode_info[key] = 0

    while True:
        episode_reward = 0
        obs_vector = vec_env.reset()
        time_between_frames = dt
        print("DEBUG INFO")
        done = False
        # Each frame should have a gap of 80ms for the visualisation video to match real time. Each frame represents simulation moving by 80ms
        # The time.sleep delay ensures the simulation moves at same speed as if it were a real robot. info["dt"] should be 0.08

        while not done:
            start_time = time.time()
            action_vector, _ = model.predict(obs_vector)
            obs_vector, reward_vector, dones_vector, info_vector = vec_env.step(action_vector)
            done = dones_vector[0]

            for k, v in info_vector[0].items():
                if k == "is_success" and v == 1:
                    cur_episode_info[k] = 1
                elif k.startswith("rew"):
                    cur_episode_info[k] += v
            episode_reward += reward_vector[0]

            time_to_process = time.time() - start_time
            delay_time = time_between_frames - time_to_process + extra_delay
            if (delay_time > 0):
                time.sleep(delay_time)  # proper time

        print(f"Episode complete. episode_reward:   {episode_reward:.3f} ")
        for k, v in cur_episode_info.items():
            print(f"{k :35} {v:.3f}")
            cur_episode_info[k] = 0
        print("")
        time.sleep(1.5)  # Pause a bit before resetting environment


if __name__ == "__main__":
    main()
