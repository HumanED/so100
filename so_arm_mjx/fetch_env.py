"""
GPU (MuJoCo MJX + JAX) version of so_arm_rl's SoFetchEnv.

Thousands of copies of the env are stepped in parallel on one GPU and trained with Brax PPO.
It mirrors the CPU env as closely as is practical. Differences:

- Continuous actions in [-1, 1] (Brax PPO's tanh-Gaussian policy) instead of MultiDiscrete([11]*6).
  The CPU env maps its 11 bins onto the same [-1, 1] -> ctrlrange mapping, so this is the continuous
  version of the same control scheme.
- Cheaper collisions for speed (`FetchConfig.collision_model`, see `build_mj_model`). Collision detection was
  ~95% of a GPU physics step with the full meshes. Dynamics (masses, inertias, joints, actuators) are unchanged.
  - "filtered": original meshes, but no arm-vs-arm checks; big links only vs floor.
  - "simplified": the task contacts are unchanged (finger pads vs cube, cube vs floor). Safety contacts: the
    original collision meshes (hulls capped at `max_hull_verts`) vs the floor only, and capsules inside the
    wrist/jaw meshes vs the cube, so nothing passes through the floor or the gripper. Expensive mesh-vs-cube
    and mesh-vs-mesh pairs are gone. (Bounding boxes for the floor were tried and rejected: they stick out
    well beyond the tapered jaws.)
  - "full": the original collision setup.
  fidelity_check.py compares these against the original model.
- Pyramidal friction cone and 20 solver iterations by default (cheaper on GPU). 10 iterations drops grasps on the
  stiff finger-pad contacts (see fidelity_check.py). Configurable.
- The one-time grasp / target bonuses are re-armed every episode. In the CPU env they are only
  earned once per worker lifetime (they're zeroed and never reset in reset()).
- Observation normalisation is done by Brax PPO (`normalize_observations=True`), and the running
  statistics are saved with the policy, unlike the CPU env's NormalizeObservation wrapper.

The observation layout (31 values) and reward shaping are otherwise identical to SoFetchEnv.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from flax import struct
from mujoco import mjx

# Model building lives in model.py (no JAX); re-exported here so existing imports keep working
from .model import (INITIAL_ARM_POSITION, SCENE_XML, FetchConfig, ObsIndices, build_mj_model,  # noqa: F401
                    settled_initial_state)


def compute_obs(idx: ObsIndices, qpos, qvel, site_xpos, goal, xp=jp):
    """Same 31-value layout as SoFetchEnv._get_obs:
    [robot_qpos(6), robot_qvel(6), object_qpos(7), jaw_pos(3), object-jaw(3), object-goal(3), goal(3)]
    Works on jax (xp=jp) or numpy (xp=np) arrays."""
    object_qpos = qpos[idx.object_qpos]
    jaw_pos = site_xpos[idx.jaw_site]
    return xp.concatenate([
        qpos[idx.robot_qpos], qvel[idx.robot_qvel], object_qpos, jaw_pos,
        object_qpos[:3] - jaw_pos, object_qpos[:3] - goal, goal,
    ])


def actions_to_ctrl(m_ctrlrange, action, xp=jp):
    """Same mapping as SoFetchEnv._apply_action (absolute control): [-1, 1] -> ctrlrange"""
    center = (m_ctrlrange[:, 1] + m_ctrlrange[:, 0]) / 2.0
    half = (m_ctrlrange[:, 1] - m_ctrlrange[:, 0]) / 2.0
    return xp.clip(center + xp.clip(action, -1.0, 1.0) * half, m_ctrlrange[:, 0], m_ctrlrange[:, 1])


@struct.dataclass
class FetchPipelineState:
    """mjx.Data plus per-episode task state. Kept in pipeline_state so Brax's AutoResetWrapper
    restores it on reset (it only resets pipeline_state and obs)."""
    data: mjx.Data
    goal: jax.Array
    grasp_available: jax.Array     # 1.0 until the grasp bonus is paid this episode
    target_available: jax.Array    # 1.0 until the target bonus is paid this episode


class SoFetchMjxEnv(Env):
    def __init__(self, cfg: FetchConfig = FetchConfig()):
        self.cfg = cfg
        self.mj_model = build_mj_model(cfg)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.idx = ObsIndices.from_model(self.mj_model)
        q0, v0, c0 = settled_initial_state(self.mj_model, cfg)
        self._q0, self._v0, self._c0 = jp.asarray(q0), jp.asarray(v0), jp.asarray(c0)
        self._ctrlrange = jp.asarray(self.mj_model.actuator_ctrlrange)

    # --- brax Env interface
    @property
    def observation_size(self) -> int:
        return 31

    @property
    def action_size(self) -> int:
        return self.mj_model.nu

    @property
    def backend(self) -> str:
        return "mjx"

    def _goal(self, rng):
        if self.cfg.fixed_goal is not None:
            return jp.asarray(self.cfg.fixed_goal, dtype=jp.float32)
        return jax.random.uniform(rng, (3,), minval=jp.asarray(self.cfg.goal_min),
                                  maxval=jp.asarray(self.cfg.goal_max))

    def reset(self, rng: jax.Array) -> State:
        data = mjx.make_data(self.mjx_model).replace(qpos=self._q0, qvel=self._v0, ctrl=self._c0)
        data = mjx.forward(self.mjx_model, data)
        ps = FetchPipelineState(data=data, goal=self._goal(rng),
                                grasp_available=jp.float32(1.0), target_available=jp.float32(1.0))
        obs = compute_obs(self.idx, data.qpos, data.qvel, data.site_xpos, ps.goal)
        zero = jp.float32(0.0)
        metrics = {"rew_jaw_center_to_object": zero, "rew_object_to_target": zero,
                   "grasp": zero, "success": zero}
        return State(pipeline_state=ps, obs=obs, reward=zero, done=zero, metrics=metrics, info={})

    def step(self, state: State, action: jax.Array) -> State:
        ps: FetchPipelineState = state.pipeline_state
        data = ps.data.replace(ctrl=actions_to_ctrl(self._ctrlrange, action))
        data = jax.lax.fori_loop(0, self.cfg.n_substeps, lambda _, d: mjx.step(self.mjx_model, d), data)

        obs = compute_obs(self.idx, data.qpos, data.qvel, data.site_xpos, ps.goal)
        rew_jaw = -jp.linalg.norm(obs[22:25])
        rew_tgt = -jp.linalg.norm(obs[25:28])

        # Same two-stage shaping as SoFetchEnv._compute_reward
        grasped_now = ps.grasp_available * (-rew_jaw < self.cfg.grasp_threshold)
        reward = jp.where(ps.grasp_available > 0,
                          0.75 * rew_jaw + 0.25 * rew_tgt,
                          0.25 * rew_jaw + 0.75 * rew_tgt)
        reward += self.cfg.grasp_reward * grasped_now
        success_now = ps.target_available * (-rew_tgt < self.cfg.target_threshold)
        reward += self.cfg.target_reached_reward * success_now

        # Unstable physics -> end the episode instead of poisoning the batch with NaNs
        bad = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | jp.isnan(reward)
        obs = jp.where(bad, jp.zeros_like(obs), obs)
        reward = jp.where(bad, 0.0, reward)

        ps = ps.replace(data=data,
                        grasp_available=ps.grasp_available * (1 - grasped_now),
                        target_available=ps.target_available * (1 - success_now))
        metrics = dict(state.metrics)  # wrappers may add their own keys; keep the pytree structure
        metrics.update({"rew_jaw_center_to_object": rew_jaw, "rew_object_to_target": rew_tgt,
                        "grasp": grasped_now, "success": success_now})
        return state.replace(pipeline_state=ps, obs=obs, reward=reward,
                             done=bad.astype(jp.float32), metrics=metrics)
