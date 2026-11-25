"""
Play back a policy trained with so_arm_mjx (GPU/Brax) in ordinary CPU MuJoCo.

    python visualise_mjx.py models/PPO-carl-mjx-1/final.pkl                 # interactive viewer
    python visualise_mjx.py models/PPO-carl-mjx-1/final.pkl --episodes 10 --headless   # just print returns
    python visualise_mjx.py models/PPO-carl-mjx-1/final.pkl --video eval.mp4 --episodes 2

config.json must sit next to the .pkl (train.py writes it). Uses the same modified model
(collision filtering, cone, solver settings) the policy trained on.
On macOS the interactive viewer needs `mjpython visualise_mjx.py ...`.
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from so_arm_mjx.fetch_env import (FetchConfig, ObsIndices, actions_to_ctrl, build_mj_model,
                                  compute_obs, settled_initial_state)


def load_policy(pkl_path):
    with open(os.path.join(os.path.dirname(pkl_path), "config.json")) as f:
        cfg = json.load(f)
    env_cfg = FetchConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["env"].items()})
    nets = ppo_networks.make_ppo_networks(
        31, 6,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(cfg["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(cfg["value_hidden_layer_sizes"]),
    )
    params = brax_model.load_params(pkl_path)
    policy = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
    return env_cfg, policy


class Episode:
    """CPU re-implementation of SoFetchMjxEnv's step/reward, for playback."""
    def __init__(self, cfg: FetchConfig):
        self.cfg = cfg
        self.m = build_mj_model(cfg)
        self.d = mujoco.MjData(self.m)
        self.idx = ObsIndices.from_model(self.m)
        self.q0, self.v0, self.c0 = settled_initial_state(self.m, cfg)
        self.goal = np.asarray(cfg.fixed_goal if cfg.fixed_goal is not None
                               else np.random.uniform(cfg.goal_min, cfg.goal_max), dtype=np.float64)

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = self.q0, self.v0, self.c0
        # park the translucent target marker at the goal so it shows in the viewer
        tgt = self.m.joint("target:joint").qposadr[0]
        self.d.qpos[tgt:tgt + 7] = np.concatenate([self.goal, [1, 0, 0, 0]])
        mujoco.mj_forward(self.m, self.d)
        self.grasp_avail, self.target_avail = 1.0, 1.0
        return self.obs()

    def obs(self):
        return compute_obs(self.idx, self.d.qpos, self.d.qvel, self.d.site_xpos, self.goal, xp=np)

    def step(self, action):
        self.d.ctrl[:] = actions_to_ctrl(self.m.actuator_ctrlrange, np.asarray(action), xp=np)
        tgt = self.m.joint("target:joint")
        for _ in range(self.cfg.n_substeps):
            mujoco.mj_step(self.m, self.d)
            self.d.qvel[tgt.dofadr[0]:tgt.dofadr[0] + 6] = 0  # keep the marker from falling
            self.d.qpos[tgt.qposadr[0]:tgt.qposadr[0] + 7] = np.concatenate([self.goal, [1, 0, 0, 0]])
        o = self.obs()
        rj, rt = -np.linalg.norm(o[22:25]), -np.linalg.norm(o[25:28])
        grasp = self.grasp_avail * (-rj < self.cfg.grasp_threshold)
        r = (0.75 * rj + 0.25 * rt) if self.grasp_avail else (0.25 * rj + 0.75 * rt)
        r += self.cfg.grasp_reward * grasp
        success = self.target_avail * (-rt < self.cfg.target_threshold)
        r += self.cfg.target_reached_reward * success
        self.grasp_avail *= 1 - grasp
        self.target_avail *= 1 - success
        return o, r, {"grasp": grasp, "success": success}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("params", help="path to a .pkl written by so_arm_mjx.train")
    p.add_argument("--episodes", type=int, default=0, help="0 = loop forever (viewer)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", default=None, help="write an mp4 instead of opening a viewer")
    a = p.parse_args()

    cfg, policy = load_policy(a.params)
    ep = Episode(cfg)
    dt = cfg.timestep * cfg.n_substeps
    key = jax.random.PRNGKey(0)
    viewer = renderer = None
    frames = []
    if a.video:
        renderer = mujoco.Renderer(ep.m, 480, 640)
        cam = mujoco.MjvCamera()  # frame the arm base, cube and goal
        cam.lookat[:] = [0.0, -0.2, 0.05]
        cam.distance, cam.azimuth, cam.elevation = 0.95, 120.0, -35.0
    elif not a.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(ep.m, ep.d)

    n = 0
    while a.episodes == 0 or n < a.episodes:
        obs, ret, grasped, succeeded = ep.reset(), 0.0, 0.0, 0.0
        for _ in range(cfg.episode_length):
            t = time.time()
            action, _ = policy(jp.asarray(obs, dtype=jp.float32), key)
            obs, r, info = ep.step(np.asarray(action))
            ret += r; grasped += info["grasp"]; succeeded += info["success"]
            if renderer:
                renderer.update_scene(ep.d, camera=cam); frames.append(renderer.render())
            if viewer:
                if not viewer.is_running():
                    return
                viewer.sync()
                time.sleep(max(0.0, dt - (time.time() - t)))
        n += 1
        print(f"episode {n}: return {ret:8.2f} | grasped {bool(grasped)} | reached target {bool(succeeded)}")
        if viewer:
            time.sleep(1.5)

    if a.video and frames:
        import imageio
        imageio.mimsave(a.video, frames, fps=int(round(1 / dt)))
        print("wrote", a.video)


if __name__ == "__main__":
    main()
