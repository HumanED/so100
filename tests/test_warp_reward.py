"""so_arm_warp's "default" reward must equal SoFetchEnv._compute_reward, step by step, across episode resets.

Steps the real CPU SoFetchEnv with random actions and feeds each state into the torch reward.
    python tests/test_warp_reward.py        (or: pytest tests/)
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv  # noqa: E402
from so_arm_warp.rewards import REWARDS  # noqa: E402


def test_default_reward_matches_cpu_env(episodes=6, seed=0):
    env = SoFetchEnv()
    fake = SimpleNamespace(num_envs=1, device=torch.device("cpu"))
    reward_fn = REWARDS["default"](fake)
    m, d = env.model, env.data
    site = lambda name: torch.tensor(d.site_xpos[m.site(name).id][None], dtype=torch.float64)
    rng = np.random.default_rng(seed)
    worst, n_grasp, n_success = 0.0, 0, 0
    for _ in range(episodes):
        env.reset()
        reward_fn.reset(torch.ones(1, dtype=torch.bool))
        obj = m.joint("object:joint").qposadr[0]
        for t in range(env.MAX_TIMESTEPS):
            a = rng.integers(0, env.N_DISCRETE, 6)
            if 25 <= t <= 35:
                a[5] = env.N_DISCRETE - 1     # jaws wide open...
            if t in (30, 31):                 # ...and the cube teleported next to jaw_site -> grasp bonus (once)
                d.qpos[obj:obj + 3] = d.site_xpos[m.site("jaw_site").id] + np.array([0, 0, 0.005])
                d.qvel[m.joint("object:joint").dofadr[0]:][:6] = 0
            if t in (60, 61):                 # cube teleported onto the goal -> success bonus (once)
                d.qpos[obj:obj + 3] = env.goal
                d.qvel[m.joint("object:joint").dofadr[0]:][:6] = 0
            obs, r_cpu, _, _, info = env.step(a)
            fake.cube_pos = torch.tensor(obs[None, 12:15]); fake.goal = torch.tensor(env.goal[None])
            fake.jaw_pos, fake.jaw_top, fake.jaw_bottom = site("jaw_site"), site("jaw_top_site"), site("jaw_bottom_site")
            r_warp, metrics = reward_fn()
            worst = max(worst, abs(float(r_warp[0]) - r_cpu))
            assert bool(metrics["grasp"][0]) == (info["rew_grasp"] > 0)
            assert bool(metrics["success"][0]) == (info["rew_success"] > 0)
            n_grasp += info["rew_grasp"] > 0; n_success += info["rew_success"] > 0
    print(f"max |reward difference| {worst:.2e} over {episodes} episodes; grasp bonuses {n_grasp}, success bonuses {n_success}")
    assert worst < 1e-9
    assert n_grasp > 0 and n_success > 0, 'bonus paths not exercised'


if __name__ == "__main__":
    test_default_reward_matches_cpu_env()
    print("OK")
