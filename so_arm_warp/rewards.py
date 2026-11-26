"""
Reward functions for the Warp training env. Pick one with `--reward NAME` (or REWARD in the Kaggle notebook).

To try your own: subclass Reward (or DefaultReward), decorate it with @register("your_name"), and fill in
__call__. Everything is a torch tensor with one entry per env, on the GPU. Useful inputs (all (N, 3) unless noted):

    env.cube_pos      cube centre             env.jaw_pos       jaw_site (same point SoFetchEnv uses)
    env.goal          target position         env.pad_mid       midpoint between the two jaws' finger pads
    env.obs           the 31-value observation (N, 31), same layout as SoFetchEnv._get_obs
    env.t             step within the episode (N,)
    env.qpos/qvel     raw MuJoCo state views (N, nq) / (N, nv)

Per-episode state (e.g. "bonus already paid") goes in `state_init`; it's reset automatically when an episode ends.
`metrics` returned by __call__ are summed per episode and logged (e.g. grasp / success rates).
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
    """SoFetchEnv._compute_reward: two-stage distance shaping plus one-time grasp (+20) and target (+30) bonuses.
    Bonuses are re-armed every episode (SoFetchEnv only pays them once per worker lifetime).
    Note: the grasp condition (cube centre within 4 mm of jaw_site) can't be met without the cube penetrating the
    fixed jaw (jaw_site is 1.4 cm from the fixed pads, the cube is 4 cm wide); see "pad_midpoint"."""
    state_init = {"grasp_avail": 1.0, "target_avail": 1.0}
    grasp_bonus, target_bonus = 20.0, 30.0
    grasp_threshold, target_threshold = 0.004, 0.002

    def grasp_distance(self):
        return torch.linalg.norm(self.env.cube_pos - self.env.jaw_pos, dim=-1)

    def __call__(self):
        e, st = self.env, self.state
        rew_jaw = -torch.linalg.norm(e.cube_pos - e.jaw_pos, dim=-1)
        rew_tgt = -torch.linalg.norm(e.cube_pos - e.goal, dim=-1)
        pre = st["grasp_avail"] > 0
        grasped = pre & (self.grasp_distance() < self.grasp_threshold)
        reward = torch.where(pre, 0.75 * rew_jaw + 0.25 * rew_tgt, 0.25 * rew_jaw + 0.75 * rew_tgt)
        reward = reward + self.grasp_bonus * grasped
        success = (st["target_avail"] > 0) & (-rew_tgt < self.target_threshold)
        reward = reward + self.target_bonus * success
        st["grasp_avail"] = st["grasp_avail"] * (~grasped)
        st["target_avail"] = st["target_avail"] * (~success)
        return reward, {"grasp": grasped.float(), "success": success.float(),
                        "rew_jaw_center_to_object": rew_jaw, "rew_object_to_target": rew_tgt}


@register("pad_midpoint")
class PadMidpointReward(DefaultReward):
    """Same shaping, but 'grasped' means the cube centre is within 1.5 cm of the midpoint between the finger pads,
    which a real grip can reach (the default's 4 mm-from-jaw_site cannot)."""
    grasp_threshold = 0.015

    def grasp_distance(self):
        return torch.linalg.norm(self.env.cube_pos - self.env.pad_mid, dim=-1)
