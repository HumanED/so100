"""
SoFetchVecEnv: N copies of SoFetchEnv in ONE process, as a single Stable-Baselines3 VecEnv.

Drop-in replacement for `SubprocVecEnv([make_env(i, seed) for i in range(N)])` in train_parallel.py. Wrap it in
VecMonitor / VecNormalize exactly as before. Where the time goes and why it's faster:

- Physics for all envs is stepped by MuJoCo's C++ `rollout` on a persistent thread pool: no worker processes,
  no pipes, no per-env Python during physics. The pool load-balances between performance and efficiency cores.
- Observations and rewards are numpy operations over all envs at once.
- The reset pose is deterministic, so it's computed once and copied. SoFetchEnv re-runs 400 settle substeps
  on every reset.

Behaviour matches raw SoFetchEnv inside SubprocVecEnv: same 31-value observation (float64), reward (incl. the
jaw-open term and the once-per-episode grasp / success bonuses), info dict keys and values (the rew_* keys
TensorboardCallback reads through get_attr("info")), truncation after MAX_TIMESTEPS, auto-reset with
"terminal_observation" and "TimeLimit.truncated".

physics="rollout" (default) is not bit-identical: `rollout` doesn't carry the solver's warm start between calls,
so results differ at solver-tolerance level, visibly only after contacts, and reset each episode.
physics="sequential" steps each env's own MjData exactly like SoFetchEnv; it exists to test that everything
else here is exact (tests/test_vec_env.py).
"""
import os
from typing import Any, List, Optional, Sequence

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from .so_arm_fetch_env import SoFetchEnv

_FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS
_SITES = ("jaw_site", "jaw_top_site", "jaw_bottom_site")


class SoFetchVecEnv(VecEnv):
    def __init__(self, num_envs: int, n_threads: Optional[int] = None, physics: str = "rollout",
                 chunk_size: Optional[int] = None):
        if physics not in ("rollout", "sequential"):
            raise ValueError("physics must be 'rollout' or 'sequential'")
        template = SoFetchEnv()
        template.reset()
        self.MAX_TIMESTEPS = template.MAX_TIMESTEPS
        self.N_SUBSTEPS = template.N_SUBSTEPS
        self.N_DISCRETE = template.N_DISCRETE
        self._goal = np.array(template.goal, dtype=np.float64)
        self._grasp_bonus = float(template.FIXED_GRASP_REWARD)
        self._target_bonus = float(template.FIXED_TARGET_REACHED_REWARD)
        self._physics = physics

        # Same scene plus framepos sensors on the three jaw sites: rollout returns sensordata, and a framepos sensor
        # is computed at the same point in mj_step as data.site_xpos, so it reads exactly what SoFetchEnv reads.
        spec = mujoco.MjSpec.from_file(template.FULLPATH)
        for site in _SITES:
            spec.add_sensor(name=f"{site}_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site)
        self.model = spec.compile()
        m = self.model
        assert (m.nq, m.nv, m.nu) == (template.model.nq, template.model.nv, template.model.nu)

        # Settled reset state, computed once by the real env
        self._nstate = mujoco.mj_stateSize(m, _FULLPHYSICS)
        self._init_state = np.empty(self._nstate)
        mujoco.mj_getState(template.model, template.data, self._init_state, _FULLPHYSICS)
        self._init_ctrl = template.data.ctrl.copy()
        self._init_warmstart = template.data.qacc_warmstart.copy()
        self._obs0 = template._get_obs()[0].astype(np.float64)

        # Observation indices into the FULLPHYSICS state vector: [time, qpos(nq), qvel(nv), ...]
        qpos0, qvel0 = 1, 1 + m.nq
        self._i_robot_qpos = qpos0 + template._robot_qpos_idx
        self._i_robot_qvel = qvel0 + template._robot_qvel_idx
        self._i_object = qpos0 + np.arange(template._object_qpos_slice.start, template._object_qpos_slice.stop)
        lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self._ctrl_lo, self._ctrl_hi = lo, hi
        self._ctrl_center, self._ctrl_half = (hi + lo) / 2.0, (hi - lo) / 2.0
        template.close()

        super().__init__(num_envs, template.observation_space, template.action_space)
        n = num_envs
        self._state = np.tile(self._init_state, (n, 1))
        self._t = np.zeros(n, dtype=np.int64)
        self._grasp = np.full(n, self._grasp_bonus)
        self._target = np.full(n, self._target_bonus)
        self._infos: List[dict] = [self._reset_info() for _ in range(n)]
        self._actions = np.zeros((n, template.action_space.shape[0]), dtype=np.int64)

        if physics == "rollout":
            self.n_threads = n_threads or os.cpu_count() or 1
            self._pool = mj_rollout.Rollout(nthread=self.n_threads)
            self._thread_data = [mujoco.MjData(m) for _ in range(self.n_threads)]
            self._models = [m] * n  # skip_checks=True needs one model entry per env (shared, read-only)
            self._out_state = np.empty((n, self.N_SUBSTEPS, self._nstate))
            self._out_sensor = np.empty((n, self.N_SUBSTEPS, m.nsensordata))
            self._chunk_size = chunk_size
        else:
            self.n_threads = 1
            self._datas = [mujoco.MjData(m) for _ in range(n)]
            for i in range(n):
                self._load_reset_state(i)
            self._site_ids = [m.site(s).id for s in _SITES]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _reset_info() -> dict:
        # Same dict SoFetchEnv.reset() creates
        return {"is_success": 0, "total_timesteps": 0, "rew_jaw_center_to_object": 0, "rew_object_to_target": 0,
                "rew_jaw_center_to_object_prop": 0, "rew_object_to_target_prop": 0, "reset_flag": True,
                "rew_grasp": 0, "rew_success": 0, "rew_jaw_open_prop": 0}

    def _load_reset_state(self, i: int) -> None:
        d = self._datas[i]
        mujoco.mj_setState(self.model, d, self._init_state, _FULLPHYSICS)
        d.ctrl[:] = self._init_ctrl
        d.qacc_warmstart[:] = self._init_warmstart

    def _raw_obs(self, state, jaw_pos) -> np.ndarray:
        obj = state[:, self._i_object]
        goal = np.broadcast_to(self._goal, (len(state), 3))
        return np.concatenate([state[:, self._i_robot_qpos], state[:, self._i_robot_qvel], obj, jaw_pos,
                               obj[:, :3] - jaw_pos, obj[:, :3] - goal, goal], axis=1)

    # ---------------------------------------------------------------- VecEnv API
    def reset(self) -> VecEnvObs:
        n = self.num_envs
        self._state[:] = self._init_state
        self._t[:] = 0
        self._grasp[:], self._target[:] = self._grasp_bonus, self._target_bonus
        self._infos = [self._reset_info() for _ in range(n)]
        if self._physics == "sequential":
            for i in range(n):
                self._load_reset_state(i)
        self._reset_seeds()
        self._reset_options()
        return np.tile(self._obs0, (n, 1))

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).reshape(self.num_envs, -1)

    def step_wait(self) -> VecEnvStepReturn:
        n = self.num_envs
        # Same action -> ctrl mapping as SoFetchEnv.step / _apply_action (absolute control)
        action = -1 + (self._actions * 2) / (self.N_DISCRETE - 1)
        ctrl = np.clip(self._ctrl_center + action * self._ctrl_half, self._ctrl_lo, self._ctrl_hi)

        if self._physics == "rollout":
            control = np.repeat(ctrl[:, None, :], self.N_SUBSTEPS, axis=1)
            self._pool.rollout(self._models, self._thread_data, self._state, control, nstep=self.N_SUBSTEPS,
                               state=self._out_state, sensordata=self._out_sensor, chunk_size=self._chunk_size,
                               skip_checks=True)
            self._state[:] = self._out_state[:, -1]
            sites = self._out_sensor[:, -1, :9].reshape(n, 3, 3).copy()
        else:
            sites = np.empty((n, 3, 3))
            for i, d in enumerate(self._datas):
                d.ctrl[:] = ctrl[i]
                mujoco.mj_step(self.model, d, nstep=self.N_SUBSTEPS)
                mujoco.mj_getState(self.model, d, self._state[i], _FULLPHYSICS)
                sites[i] = d.site_xpos[self._site_ids]
        jaw, jaw_top, jaw_bottom = sites[:, 0], sites[:, 1], sites[:, 2]
        obs = self._raw_obs(self._state, jaw)

        # Reward: same as SoFetchEnv._compute_reward (per-env norms keep it bit-identical)
        rew_jaw = -np.array([np.linalg.norm(r) for r in obs[:, 22:25]])
        rew_tgt = -np.array([np.linalg.norm(r) for r in obs[:, 25:28]])
        width = np.array([np.linalg.norm(t - b) for t, b in zip(jaw_top, jaw_bottom)])
        rew_open = np.maximum(0, width - 0.02)
        reward = 0 + ((0.5 * rew_jaw) + (0.5 * rew_tgt) + rew_open)
        grasped = (self._grasp > 0) & (np.abs(rew_jaw) < 0.03) & (width >= 0.04)
        reward = reward + np.where(grasped, self._grasp, 0.0)
        rew_grasp = np.where(grasped, self._grasp, 0.0)
        self._grasp[grasped] = 0
        success = (np.abs(rew_tgt) < 0.02) & (self._target > 0)
        reward = reward + np.where(success, self._target, 0.0)
        rew_success = np.where(success, self._target, 0.0)
        self._target[success] = 0

        self._t += 1
        truncated = self._t >= self.MAX_TIMESTEPS
        for i in range(n):
            info = self._infos[i]
            info["reset_flag"] = False
            info["total_timesteps"] = int(self._t[i])
            info["rew_jaw_center_to_object"] = rew_jaw[i]
            info["rew_object_to_target"] = rew_tgt[i]
            info["rew_grasp"] = int(rew_grasp[i]) if rew_grasp[i] else 0
            info["rew_success"] = int(rew_success[i]) if rew_success[i] else 0
            info["rew_jaw_center_to_object_prop"] = 0.5 * rew_jaw[i]
            info["rew_object_to_target_prop"] = 0.5 * rew_tgt[i]
            info["rew_jaw_open_prop"] = rew_open[i]
            if success[i]:
                info["is_success"] = 1
            info["TimeLimit.truncated"] = bool(truncated[i])

        out_infos = [dict(i) for i in self._infos]  # values are scalars or fresh arrays
        done_idx = np.flatnonzero(truncated)
        if len(done_idx):
            for i in done_idx:
                out_infos[i]["terminal_observation"] = obs[i].copy()
                self._infos[i] = self._reset_info()
                if self._physics == "sequential":
                    self._load_reset_state(i)
            self._state[done_idx] = self._init_state
            self._t[done_idx] = 0
            self._grasp[done_idx], self._target[done_idx] = self._grasp_bonus, self._target_bonus
            obs[done_idx] = self._obs0
        return obs, reward, truncated.copy(), out_infos

    def close(self) -> None:
        if self._physics == "rollout":
            self._pool.close()

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        idx = self._get_indices(indices)
        if attr_name == "info":
            return [self._infos[i] for i in idx]
        return [getattr(self, attr_name) for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        raise NotImplementedError("SoFetchVecEnv has no per-env sub-environments")

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return [None] * self.num_envs
