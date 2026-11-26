"""The Warp scene (so_arm_warp/scene.py) must match the CPU env's task: layout and settled start pose.

    python tests/test_warp_scene.py        (or: pytest tests/)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv  # noqa: E402
from so_arm_warp.scene import FIXED_GOAL, GOAL_MAX, GOAL_MIN, INITIAL_CUBE_POSITION, FetchConfig, build_mj_model, settled_initial_state  # noqa: E402


def test_layout_matches_cpu_env():
    env = SoFetchEnv()
    assert np.allclose(INITIAL_CUBE_POSITION, env.initial_cube_position)
    assert np.allclose(FIXED_GOAL, env.FIXED_GOAL)
    assert np.allclose(GOAL_MIN, env.GOAL_MIN) and np.allclose(GOAL_MAX, env.GOAL_MAX)


def test_settled_start_matches_cpu_env():
    env = SoFetchEnv()
    env.reset()
    cfg = FetchConfig(collision_model="full", cone="elliptic", solver_iterations=100, ls_iterations=50)
    q0, v0, _ = settled_initial_state(env.model, cfg)
    assert np.array_equal(q0, env.data.qpos) and np.array_equal(v0, env.data.qvel)


if __name__ == "__main__":
    test_layout_matches_cpu_env()
    test_settled_start_matches_cpu_env()
    print("OK")
