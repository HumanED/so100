"""
Reward functions for the Warp training env. Pick one with `--reward NAME` (or `reward=` in the Kaggle notebook).

To try your own: subclass Reward (or DefaultReward), decorate it with @register("your_name"), and fill in
__call__. Everything is a torch tensor with one entry per env, on the GPU. Useful inputs (all (N, 3) unless noted):

    env.cube_pos      cube centre             env.jaw_pos       jaw_site (same point SoFetchEnv uses)
    env.jaw_top       jaw_top_site (inner edge of the moving jaw)
    env.jaw_bottom    jaw_bottom_site (inner edge of the fixed jaw)
    env.pad_mid       midpoint between the two jaws' finger pads
    env.goal          target position
    env.obs           the 31-value observation (N, 31), same layout as SoFetchEnv._get_obs
    env.t             step within the episode (N,)
    env.qpos/qvel     raw MuJoCo state views (N, nq) / (N, nv)

Per-episode state (e.g. "bonus already paid") goes in `state_init`; it's reset automatically when an episode ends.
`metrics` returned by __call__ are summed per episode and logged; "grasp" and "success" become grasp/success rates.
"""
from typing import Dict, Tuple

import torch

REWARDS: Dict[str, type] = {}


def register(name: str):
    def deco(cls):
        REWARDS[name] = cls
        return cls
    return deco


class Reward:
    state_init: Dict[str, float] = {}

    def __init__(self, env):
        self.env = env
        self.state = {k: torch.full((env.num_envs,), float(v), device=env.device) for k, v in self.state_init.items()}

    def reset(self, mask: torch.Tensor) -> None:
        for k, v in self.state_init.items():
            self.state[k][mask] = v

    def __call__(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


@register("default")
class DefaultReward(Reward):
    """Same as SoFetchEnv._compute_reward (so_arm_rl): 0.5 x jaw-to-cube distance + 0.5 x cube-to-goal distance
    (both negative), plus how far the jaws are opened beyond the cube's width, plus a once-per-episode +20 when the
    jaw centre is within 3 cm of the cube with the jaws at least 4 cm open, and a once-per-episode +30 when the cube
    is within 2 cm of the goal. tests/test_warp_reward.py checks it matches SoFetchEnv step by step."""
    state_init = {"grasp_avail": 1.0, "target_avail": 1.0}
    grasp_bonus, target_bonus = 20.0, 30.0
    object_width = 0.02
    grasp_distance, grasp_min_open, target_distance = 0.03, 0.04, 0.02

    def __call__(self):
        e, st = self.env, self.state
        rew_jaw = -torch.linalg.norm(e.cube_pos - e.jaw_pos, dim=-1)
        rew_tgt = -torch.linalg.norm(e.cube_pos - e.goal, dim=-1)
        width = torch.linalg.norm(e.jaw_top - e.jaw_bottom, dim=-1)
        rew_open = torch.clamp(width - self.object_width, min=0)
        reward = 0.5 * rew_jaw + 0.5 * rew_tgt + rew_open
        grasped = (st["grasp_avail"] > 0) & (-rew_jaw < self.grasp_distance) & (width >= self.grasp_min_open)
        success = (st["target_avail"] > 0) & (-rew_tgt < self.target_distance)
        reward = reward + self.grasp_bonus * grasped + self.target_bonus * success
        st["grasp_avail"] = st["grasp_avail"] * (~grasped)
        st["target_avail"] = st["target_avail"] * (~success)
        return reward, {"grasp": grasped.float(), "success": success.float(), "rew_jaw_center_to_object": rew_jaw,
                        "rew_object_to_target": rew_tgt, "rew_jaw_open": rew_open}


@register("legacy")
class LegacyReward(Reward):
    """The pre-PPO7 SoFetchEnv reward, for comparison: two-stage shaping (0.75/0.25 before the grasp bonus is paid,
    0.25/0.75 after), +20 when the cube centre is within 4 mm of jaw_site, +30 within 2 mm of the goal. Bonuses are
    re-armed every episode. The 4 mm grasp can't be met without the cube penetrating the fixed jaw."""
    state_init = {"grasp_avail": 1.0, "target_avail": 1.0}

    def __call__(self):
        e, st = self.env, self.state
        rew_jaw = -torch.linalg.norm(e.cube_pos - e.jaw_pos, dim=-1)
        rew_tgt = -torch.linalg.norm(e.cube_pos - e.goal, dim=-1)
        pre = st["grasp_avail"] > 0
        grasped = pre & (-rew_jaw < 0.004)
        success = (st["target_avail"] > 0) & (-rew_tgt < 0.002)
        reward = torch.where(pre, 0.75 * rew_jaw + 0.25 * rew_tgt, 0.25 * rew_jaw + 0.75 * rew_tgt)
        reward = reward + 20.0 * grasped + 30.0 * success
        st["grasp_avail"] = st["grasp_avail"] * (~grasped)
        st["target_avail"] = st["target_avail"] * (~success)
        return reward, {"grasp": grasped.float(), "success": success.float(),
                        "rew_jaw_center_to_object": rew_jaw, "rew_object_to_target": rew_tgt}
