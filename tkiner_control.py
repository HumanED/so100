from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv
import numpy as np
import gymnasium
from gymnasium.wrappers import TransformObservation, NormalizeObservation
import os
import time
import tkinter as tk
import sys

def make_env():
    """Creates gymnasium environment for visualisation"""

    # env = gymnasium.make("ShadowEnv-v1")
    env = SoFetchEnv(render_mode="human")
    return env


class Statistic:
    def __init__(self):
        self.grasped_reward_issued = None
        self.target_reward_issued = None
        self.total_episode_reward = None
        self.reset()

    def reset(self):
        self.total_episode_reward = 0
        self.grasped_reward_issued = False
        self.target_reward_issued = False


def main():
    MAX_ACTION = 63 # Set to N_DISCRETE - 1
    # These are not exact text in .xml files
    SLIDER_LABELS = ["Base Rotation", "Pitch", "Elbow Pitch","Wrist Pitch", "Wrist Rotation","Jaw"]
    root = tk.Tk()
    root.geometry("600x800")
    root.title("MuJoco Simulation Control")
    env = make_env()
    env.reset()
    stat = Statistic()
    sliders = [tk.Scale(
        root,
        from_= 0,
        to=MAX_ACTION,
        orient='horizontal',
        length=300,
        label=label,
        font=('Consolas', 14)

    ) for label in SLIDER_LABELS]

    for slider in sliders:
        slider.set(int(MAX_ACTION / 2))
        slider.pack()
    obs_label = tk.Label(root, text=f"Default Text", font=('Consolas', 14))
    obs_label.pack(padx=10)


    def on_reset():
        env.reset()
        for slider in sliders:
            slider.set(int(MAX_ACTION / 2))
        obs_label.config(text="Environment reset")
        stat.reset()

    reset_button = tk.Button(root, text="Reset", command=on_reset, font=('Consolas',16,'normal') )
    reset_button.pack()

    def step_env():
        user_input = [slider.get() for slider in sliders]
        action = np.array(user_input)
        # print(action)
        info: dict = dict()
        obs, rew, terminated, truncated, info = env.step(action)
        stat.total_episode_reward += rew
        label_text = ""
        for k,v in info.items():
            label_text += f"{k}:{v:.3f}\n"
        label_text += f"total_episode_rew {stat.total_episode_reward:.3f}\n"
        label_text += f"grasp_reward_issued {stat.grasped_reward_issued}\n"
        label_text += f"target_reward_issued {stat.target_reward_issued}\n"
        if  info.get("rew_grasp", -1) == env.FIXED_GRASP_REWARD:
            print("Successful grasp")
            stat.grasped_reward_issued = True
        if info.get("rew_success", -1) == env.FIXED_TARGET_REACHED_REWARD:
            print("Goal achieved")
            stat.target_reward_issued = True
        obs_label.config(text=f"Info: {label_text}")
        # Calls the step_env every 40 milliseconds.
        root.after(40, step_env)

    step_env()

    root.mainloop()




if __name__ == "__main__":
    main()