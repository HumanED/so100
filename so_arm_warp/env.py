"""
Batched fetch env on MuJoCo Warp: N worlds on one GPU, state exposed as zero-copy torch tensors.

Same task as SoFetchEnv: 31-value observation (same layout), 100-step episodes, fixed goal by default.
Actions are continuous in [-1, 1] per motor, mapped to the actuator control range exactly as SoFetchEnv maps its
11 discrete bins (absolute position control). Rewards come from so_arm_warp.rewards.
"""
from typing import Dict, Optional, Sequence

import mujoco
import numpy as np
import torch

from .scene import FIXED_GOAL, GOAL_MAX, GOAL_MIN, ObsIndices

from .physics import PhysicsConfig, WarpBatch
from .rewards import REWARDS


class FetchWarpEnv:
    def __init__(self, num_envs: int, physics: PhysicsConfig = PhysicsConfig(), reward: str = "default",
                 episode_length: int = 100, fixed_goal: Optional[Sequence[float]] = FIXED_GOAL,
                 goal_low=GOAL_MIN, goal_high=GOAL_MAX, device: Optional[str] = None, seed: int = 0):
        import warp as wp
        self.wp = wp
        self.num_envs, self.episode_length = num_envs, episode_length
        self.batch = WarpBatch(physics, num_envs, device=device)
        m = self.m = self.batch.m
        self.device = torch.device("cuda", self.batch.device.ordinal) if self.batch.device.is_cuda else torch.device("cpu")
        dw = self.batch.dw
        # zero-copy views of the Warp state
        self.qpos, self.qvel, self.ctrl = wp.to_torch(dw.qpos), wp.to_torch(dw.qvel), wp.to_torch(dw.ctrl)
        self._site_xpos, self._geom_xpos = wp.to_torch(dw.site_xpos), wp.to_torch(dw.geom_xpos)
        self._time, self._warm = wp.to_torch(dw.time), wp.to_torch(dw.qacc_warmstart)
        self._act = wp.to_torch(dw.act) if m.na else None

        idx = ObsIndices.from_model(m)
        t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        self._i_rq, self._i_rv, self._i_obj, self._jaw = t(idx.robot_qpos), t(idx.robot_qvel), t(idx.object_qpos), idx.jaw_site
        self._jaw_top, self._jaw_bottom = m.site("jaw_top_site").id, m.site("jaw_bottom_site").id
        self._pads = t([m.geom(f"{s}_jaw_pad_{i}").id for s in ("fixed", "moving") for i in (3, 4)])
        lo, hi = t(m.actuator_ctrlrange[:, 0], torch.float32), t(m.actuator_ctrlrange[:, 1], torch.float32)
        self._lo, self._hi, self._center, self._half = lo, hi, (lo + hi) / 2, (hi - lo) / 2
        self._q0, self._v0, self._c0 = (t(x, torch.float32) for x in (self.batch.q0, self.batch.v0, self.batch.c0))
        # jaw / pad positions at the reset state (Warp's derived quantities aren't recomputed until the next step)
        d = mujoco.MjData(m); d.qpos[:] = self.batch.q0; mujoco.mj_kinematics(m, d)
        self._jaw0 = t(d.site_xpos[idx.jaw_site], torch.float32)
        self._jaw_top0 = t(d.site_xpos[self._jaw_top], torch.float32)
        self._jaw_bottom0 = t(d.site_xpos[self._jaw_bottom], torch.float32)
        self._pads0 = t(d.geom_xpos[[m.geom(f"{s}_jaw_pad_{i}").id for s in ("fixed", "moving") for i in (3, 4)]], torch.float32)

        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self._fixed_goal = None if fixed_goal is None else t(fixed_goal, torch.float32)
        self._goal_lo, self._goal_hi = t(goal_low, torch.float32), t(goal_high, torch.float32)
        self.goal = torch.zeros(num_envs, 3, device=self.device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reward_fn = REWARDS[reward](self)
        self._ep_sums: Dict[str, torch.Tensor] = {}
        self._fresh = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.obs_size, self.action_size = 31, m.nu

    # ------------------------------------------------------------------ derived quantities
    @property
    def cube_pos(self):
        return self.qpos[:, self._i_obj[:3]]

    @property
    def jaw_pos(self):
        return torch.where(self._fresh[:, None], self._jaw0, self._site_xpos[:, self._jaw])

    @property
    def jaw_top(self):
        return torch.where(self._fresh[:, None], self._jaw_top0, self._site_xpos[:, self._jaw_top])

    @property
    def jaw_bottom(self):
        return torch.where(self._fresh[:, None], self._jaw_bottom0, self._site_xpos[:, self._jaw_bottom])

    @property
    def pad_mid(self):
        pads = torch.where(self._fresh[:, None, None], self._pads0, self._geom_xpos[:, self._pads])
        return 0.5 * (pads[:, :2].mean(1) + pads[:, 2:].mean(1))

    @property
    def obs(self):
        obj = self.qpos[:, self._i_obj]
        jaw = self.jaw_pos
        return torch.cat([self.qpos[:, self._i_rq], self.qvel[:, self._i_rv], obj, jaw,
                          obj[:, :3] - jaw, obj[:, :3] - self.goal, self.goal], dim=1)

    # ------------------------------------------------------------------ API
    def _reset_worlds(self, mask: torch.Tensor):
        self.qpos[mask] = self._q0
        self.qvel[mask] = self._v0
        self.ctrl[mask] = self._c0
        self._warm[mask] = 0
        self._time[mask] = 0
        if self._act is not None:
            self._act[mask] = 0
        n = int(mask.sum())
        if self._fixed_goal is not None:
            self.goal[mask] = self._fixed_goal
        else:
            u = torch.rand(n, 3, generator=self.gen, device=self.device)
            self.goal[mask] = self._goal_lo + u * (self._goal_hi - self._goal_lo)
        self.t[mask] = 0
        self._fresh[mask] = True
        self.reward_fn.reset(mask)
        for v in self._ep_sums.values():
            v[mask] = 0

    def reset(self) -> torch.Tensor:
        self._reset_worlds(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.obs

    def step(self, action: torch.Tensor):
        self.ctrl.copy_(torch.clamp(self._center + action.clamp(-1, 1) * self._half, self._lo, self._hi))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)        # torch writes done before Warp reads
        self.batch.step_env()
        self.wp.synchronize_device(self.batch.device)  # Warp step done before torch reads
        self._fresh[:] = False
        self.t += 1

        reward, metrics = self.reward_fn()
        bad = ~torch.isfinite(self.qpos).all(1) | ~torch.isfinite(reward)
        reward = torch.where(bad, torch.zeros_like(reward), reward)
        obs = torch.nan_to_num(self.obs)
        timeout = self.t >= self.episode_length
        done = timeout | bad

        metrics = dict(metrics, reward=reward)
        self.last_metrics = metrics                    # per-step values, read by the SB3 adapter
        for k, v in metrics.items():
            self._ep_sums.setdefault(k, torch.zeros(self.num_envs, device=self.device))
            self._ep_sums[k] += v
        info = {"timeout": timeout & ~bad, "unstable": bad}
        if done.any():
            info["episode"] = {k: v[done].clone() for k, v in self._ep_sums.items()}
            info["terminal_obs"] = obs[done].clone()   # last obs of each finished episode, before auto-reset
            self._reset_worlds(done)
            obs = torch.where(done[:, None], self.obs, obs)
        return obs, reward, done, info
