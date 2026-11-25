"""
GPU (MuJoCo MJX + JAX) version of so_arm_rl's SoFetchEnv.

Thousands of copies of the env are stepped in parallel on one GPU and trained with Brax PPO.
It mirrors the CPU env as closely as is practical. Differences:

- Continuous actions in [-1, 1] (Brax PPO's tanh-Gaussian policy) instead of MultiDiscrete([11]*6).
  The CPU env maps its 11 bins onto the same [-1, 1] -> ctrlrange mapping, so this is the continuous
  version of the same control scheme.
- Collision filtering for speed (see `build_mj_model`): arm links no longer collide with each other,
  and the big upper links only collide with the floor. The wrist, jaw meshes and finger pads still
  collide with the cube and the floor. Every mesh-vs-cube pair is expensive on GPU, which is why.
- Pyramidal friction cone and fewer solver iterations by default (cheaper on GPU). Configurable.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.environ.get(
    "SO_ARM_SCENE_XML",
    os.path.join(_HERE, "..", "so_arm_rl", "envs", "resources", "fetch", "scene.xml"),
)

# Same as so_arm_fetch_env.INITIAL_ARM_POSITION
INITIAL_ARM_POSITION = {"robot_Rotation": 0, "robot_Pitch": -3.10, "robot_Elbow": 3.10,
                        "robot_Wrist_Pitch": 0.0, "robot_Wrist_Roll": 0.0, "robot_Jaw": 0.0}

# Collision bitmasks: two geoms collide if (contype1 & conaffinity2) or (contype2 & conaffinity1)
_CUBE, _FLOOR, _GRIPPER, _LINK = 1, 2, 4, 8
_COLLISION_MASKS = {
    "cube": (_CUBE, _FLOOR | _GRIPPER),       # cube <-> floor, gripper
    "floor": (_FLOOR, _GRIPPER | _LINK),      # floor <-> cube (via cube's mask), gripper, links
    "gripper": (_GRIPPER, 0),                 # gripper <-> cube, floor; never with other arm parts
    "link": (_LINK, 0),                       # big links <-> floor only
}
_GRIPPER_BODIES = ("Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw")


@dataclass(frozen=True)
class FetchConfig:
    n_substeps: int = 20               # physics steps per env step (CPU env: 20)
    timestep: float = 0.002            # physics timestep (CPU env: 0.002)
    cone: str = "pyramidal"            # "pyramidal" (faster) or "elliptic" (CPU env)
    solver_iterations: int = 10        # CPU env: 100 (it converges in far fewer)
    ls_iterations: int = 10
    episode_length: int = 100          # CPU env MAX_TIMESTEPS
    fixed_goal: Optional[Sequence[float]] = (-0.3, -0.2, 0.025)  # None -> uniform in [goal_min, goal_max]
    goal_min: Sequence[float] = (0.1, 0.1, 0.1)
    goal_max: Sequence[float] = (0.5, 0.5, 0.5)
    initial_cube_position: Sequence[float] = (0.3, -0.3, 0.025)
    grasp_reward: float = 20.0
    target_reached_reward: float = 30.0
    grasp_threshold: float = 0.004     # jaw_site <-> cube centre distance counted as a grasp
    target_threshold: float = 0.002    # cube <-> goal distance counted as success
    filter_collisions: bool = True     # see module docstring

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def build_mj_model(cfg: FetchConfig = FetchConfig()) -> mujoco.MjModel:
    """Loads the same scene.xml as the CPU env and applies the GPU-friendly settings.
    Also used by visualise_mjx.py, so CPU playback has exactly the physics the policy trained on."""
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    m.opt.timestep = cfg.timestep
    m.opt.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
                  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[cfg.cone]
    m.opt.iterations = cfg.solver_iterations
    m.opt.ls_iterations = cfg.ls_iterations

    if cfg.filter_collisions:
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue  # visual-only geom, or the target marker
            body = m.body(m.geom_bodyid[g]).name
            name = m.geom(g).name
            if name == "object":
                kind = "cube"
            elif name == "floor":
                kind = "floor"
            elif body in _GRIPPER_BODIES:
                kind = "gripper"
            else:
                kind = "link"
            m.geom_contype[g], m.geom_conaffinity[g] = _COLLISION_MASKS[kind]
    return m


def settled_initial_state(m: mujoco.MjModel, cfg: FetchConfig):
    """Replicates SoFetchEnv._reset_sim on CPU: place cube + arm, then settle 10 env steps with zero action.
    The result is deterministic, so it's computed once and reused for every reset."""
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    obj = m.joint("object:joint")
    d.qpos[obj.qposadr[0]:obj.qposadr[0] + 7] = np.concatenate([cfg.initial_cube_position, [1, 0, 0, 0]])
    for name, value in INITIAL_ARM_POSITION.items():
        d.qpos[m.joint(name).qposadr[0]] = value
    ctrl_center = m.actuator_ctrlrange.mean(axis=1)
    for _ in range(10):
        d.ctrl[:] = ctrl_center  # action 0 -> centre of ctrlrange
        # The CPU env steps twice per settle iteration (_apply_action steps, then _reset_sim steps again)
        mujoco.mj_step(m, d, nstep=2 * cfg.n_substeps)
    return d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()


@dataclass(frozen=True)
class ObsIndices:
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    object_qpos: np.ndarray
    jaw_site: int

    @staticmethod
    def from_model(m: mujoco.MjModel) -> "ObsIndices":
        robot = [j for j in range(m.njnt) if m.joint(j).name.startswith("robot")]
        obj = m.joint("object:joint").qposadr[0]
        return ObsIndices(
            robot_qpos=np.array([m.jnt_qposadr[j] for j in robot]),
            robot_qvel=np.array([m.jnt_dofadr[j] for j in robot]),
            object_qpos=np.arange(obj, obj + 7),
            jaw_site=m.site("jaw_site").id,
        )


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
