from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv
import numpy as np
import gymnasium
from gymnasium.wrappers import TransformObservation, NormalizeObservation
import os
import time
import tkinter as tk

def make_env():
    """Creates gymnasium environment for visualisation with necessary wrappers"""

    def clip_observation(obs):
        """
        clips observation to within 5 standard deviations of the mean
        Refer to section D.1 of Open AI paper
        """
        return np.clip(obs, a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))

    # env = gymnasium.make("ShadowEnv-v1")
    env = SoFetchEnv(render_mode="human")
    env = NormalizeObservation(env)
    env = TransformObservation(env, clip_observation, env.observation_space)
    return env



def main():
    MAX_ACTION = 10
    # These are not exact text in .xml files
    SLIDER_LABELS = ["Base Rotation", "Pitch", "Elbow Pitch","Wrist Pitch", "Wrist Rotation","Jaw"]
    root = tk.Tk()
    root.geometry("600x800")
    root.title("MuJoco Simulation Control")
    env = make_env()
    env.reset()
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
        slider.set(5)
        slider.pack()
    obs_label = tk.Label(root, text=f"Default Text", font=('Consolas', 14))
    obs_label.pack(side=tk.LEFT, padx=10)

    def step_env():
        user_input = [slider.get() for slider in sliders]
        action = np.array(user_input)
        print(action)
        obs, rew, terminated, truncated, info = env.step(action)
        label_text = ""
        for k,v in info.items():
            label_text = label_text + f"{k}:{v:.3f}\n"
        obs_label.config(text=f"Info: {label_text}")
        # Calls the step_env every 40 milliseconds.
        root.after(40, step_env)

    step_env()
    root.mainloop()




if __name__ == "__main__":
    main()