import gymnasium
import numpy as np
from humaned_robotics.envs.fetch.so_arm_fetch_env import SoFetchEnv

env = SoFetchEnv(render_mode="human")
obs, info = env.reset()

while True:
    try:
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        # print(obs, rew, terminated, truncated, info)
    except KeyboardInterrupt:
        break
env.close()
print("DONE")