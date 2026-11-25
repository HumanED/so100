"""
SoFetchVecEnv: N copies of SoFetchEnv in ONE process, as a single Stable-Baselines3 VecEnv.

It replaces `SubprocVecEnv([make_env] * N)` from train_parallel.py, where make_env wraps SoFetchEnv in
NormalizeObservation -> TransformObservation(clip) -> Monitor. Where the time goes and why:

- Physics for all envs is stepped by MuJoCo's C++ `rollout` on a persistent thread pool, so there are no
  worker processes, no pipes, and no per-env Python during physics. The pool load-balances between
  the M-series performance and efficiency cores.
- Observations, rewards and the observation normalisation are numpy operations over all envs at once.
- The reset pose is deterministic, so it's computed once and copied. SoFetchEnv re-runs 400 settle
  substeps on every reset (about 17% of all physics).

Behaviour matches the wrapped SoFetchEnv: same 31-value observation, reward shaping, info dict keys
(including the rew_* keys TensorboardCallback reads through get_attr("info")), per-env
NormalizeObservation statistics, clipping, Monitor "episode" info, truncation and auto-reset.
That includes the one-time grasp/target bonus behaviour (set fix_bonus_bug=True to re-arm them each episode).

physics="rollout" (default) is not bit-identical to SoFetchEnv: `rollout` doesn't carry the solver's warm
start between calls, so results differ at solver-tolerance level and drift over an episode.
physics="sequential" steps each env's own MjData exactly like SoFetchEnv. It exists to test that
everything else here is exact.
"""
import os
import time
from typing import Any, List, Optional, Sequence

import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import rollout as mj_rollout
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from .so_arm_fetch_env import SoFetchEnv

_FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


def _clip_observation(obs):
    """Same as train_parallel.make_env's clip: within 5 std of the observation's own mean."""
    return np.clip(obs, a_min=obs.mean() - (5 * obs.std()), a_max=obs.mean() + (5 * obs.std()))


class SoFetchVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        n_threads: Optional[int] = None,
        physics: str = "rollout",
        normalize_obs: bool = True,
        clip_obs: bool = True,
        fix_bonus_bug: bool = False,
        initial_resets: int = 1,
        chunk_size: Optional[int] = None,
    ):
        """
        Args:
            num_envs: number of parallel envs.
            n_threads: physics threads (default: all cores).
            physics: "rollout" (fast) or "sequential" (exact match to SoFetchEnv, for testing).
            normalize_obs / clip_obs: apply train_parallel.py's NormalizeObservation / clip wrappers.
            fix_bonus_bug: re-arm the grasp/target bonuses every episode (SoFetchEnv only pays them once ever).
            initial_resets: resets each env saw before training. train_parallel's make_env calls
                env.reset(seed) once, then the VecEnv resets again, so the normaliser starts having seen 2.
                Set to 1 to match that (the VecEnv's own reset adds the second).
            chunk_size: envs per rollout work item (default: MuJoCo's choice).
        """
        if physics not in ("rollout", "sequential"):
            raise ValueError("physics must be 'rollout' or 'sequential'")
        template = SoFetchEnv()
        template.reset()
        self.MAX_TIMESTEPS = template.MAX_TIMESTEPS
        self.N_SUBSTEPS = template.N_SUBSTEPS
        self._dt = template.info["dt"]
        self._goal = np.array(template.goal, dtype=np.float64)
        self._grasp_bonus, self._target_bonus = float(template.grasp_reward), float(template.target_reached_reward)
        self._physics, self._fix_bonus = physics, fix_bonus_bug
        self._normalize, self._clip = normalize_obs, clip_obs

        # Same scene plus a framepos sensor on jaw_site: rollout returns sensordata, and the sensor is computed
        # at the same point in mj_step as data.site_xpos, so it reads exactly what SoFetchEnv reads.
        spec = mujoco.MjSpec.from_file(template.FULLPATH)
        spec.add_sensor(name="jaw_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="jaw_site")
        self.model = spec.compile()
        m = self.model
        assert (m.nq, m.nv, m.nu) == (template.model.nq, template.model.nv, template.model.nu)

        # Settled reset state, computed once by the real env
        self._nstate = mujoco.mj_stateSize(m, _FULLPHYSICS)
        self._init_state = np.empty(self._nstate)
        mujoco.mj_getState(template.model, template.data, self._init_state, _FULLPHYSICS)
        self._init_ctrl = template.data.ctrl.copy()
        self._init_warmstart = template.data.qacc_warmstart.copy()
        self._obs0 = template._get_obs().astype(np.float64)
        template.close()

        # Observation indices into the FULLPHYSICS state vector: [time, qpos(nq), qvel(nv), ...]
        qpos0, qvel0 = 1, 1 + m.nq
        self._i_robot_qpos = qpos0 + template._robot_qpos_idx
        self._i_robot_qvel = qvel0 + template._robot_qvel_idx
        self._i_object = qpos0 + np.arange(template._object_qpos_slice.start, template._object_qpos_slice.stop)
        lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self._ctrl_lo, self._ctrl_hi = lo, hi
        self._ctrl_center, self._ctrl_half = (hi + lo) / 2.0, (hi - lo) / 2.0

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,),
                                       dtype=np.float32 if normalize_obs else np.float64)
        super().__init__(num_envs, observation_space, template.action_space)
        n = num_envs

        # Per-env state
        self._state = np.tile(self._init_state, (n, 1))
        self._t = np.zeros(n, dtype=np.int64)
        self._grasp = np.full(n, self._grasp_bonus)
        self._target = np.full(n, self._target_bonus)
        self._infos: List[dict] = [self._reset_info() for _ in range(n)]
        self._ep_ret = np.zeros(n)
        self._ep_len = np.zeros(n, dtype=np.int64)
        self._t_start = np.full(n, time.time())
        # NormalizeObservation's RunningMeanStd, one per env (float64 after the first update, as in gymnasium)
        self._rms_mean = np.zeros((n, 31))
        self._rms_var = np.ones((n, 31))
        self._rms_count = np.full((n, 1), 1e-4)
        for _ in range(initial_resets):
            self._update_rms(np.arange(n), np.tile(self._obs0, (n, 1)))
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
            self._jaw_site = m.site("jaw_site").id

    # ---------------------------------------------------------------- helpers
    def _reset_info(self) -> dict:
        # Same dict SoFetchEnv.reset() creates
        return {"success": 0, "dt": self._dt, "total_timesteps": 0, "grasp": 0,
                "rew_jaw_center_to_object": 0, "rew_object_to_target": 0,
                "rew_jaw_center_to_object_prop": 0, "rew_object_to_target_prop": 0,
                "reset_flag": True, "rew_other": 0}

    def _load_reset_state(self, i: int) -> None:
        d = self._datas[i]
        mujoco.mj_setState(self.model, d, self._init_state, _FULLPHYSICS)
        d.ctrl[:] = self._init_ctrl
        d.qacc_warmstart[:] = self._init_warmstart

    def _update_rms(self, idx, obs) -> None:
        # gymnasium RunningMeanStd.update with a batch of one sample (batch_var = 0, batch_count = 1)
        mean, var, count = self._rms_mean[idx], self._rms_var[idx], self._rms_count[idx]
        delta = obs - mean
        tot_count = count + 1
        new_mean = mean + delta * 1 / tot_count
        m_a = var * count
        # gymnasium starts var as float32, so its first update computes var * count in float32
        first = (count == 1e-4)[:, 0]
        if first.any():
            m_a[first] = (var[first].astype(np.float32) * np.float32(1e-4)).astype(np.float64)
        M2 = m_a + 0.0 + np.square(delta) * count * 1 / tot_count
        self._rms_mean[idx], self._rms_var[idx], self._rms_count[idx] = new_mean, M2 / tot_count, tot_count

    def _process_obs(self, idx, raw) -> np.ndarray:
        """NormalizeObservation (updates stats) then the clip TransformObservation, for envs `idx`."""
        if not self._normalize:
            obs = raw
        else:
            self._update_rms(idx, raw)
            obs = np.float32((raw - self._rms_mean[idx]) / np.sqrt(self._rms_var[idx] + 1e-8))
        if self._clip:
            # Same as _clip_observation per row, but each row's mean/std computed once, for all rows together
            mean = obs.mean(axis=1, keepdims=True)
            std = obs.std(axis=1, keepdims=True)
            obs = np.clip(obs, a_min=mean - (5 * std), a_max=mean + (5 * std))
        return obs

    def _raw_obs(self, state, jaw_pos) -> np.ndarray:
        obj = state[:, self._i_object]
        goal = np.broadcast_to(self._goal, (len(state), 3))
        return np.concatenate([state[:, self._i_robot_qpos], state[:, self._i_robot_qvel], obj, jaw_pos,
                               obj[:, :3] - jaw_pos, obj[:, :3] - goal, goal], axis=1)

    # ---------------------------------------------------------------- VecEnv API
    def reset(self) -> VecEnvObs:
        n = self.num_envs
        all_idx = np.arange(n)
        self._state[:] = self._init_state
        self._t[:] = 0
        self._ep_ret[:] = 0
        self._ep_len[:] = 0
        self._t_start[:] = time.time()
        self._infos = [self._reset_info() for _ in range(n)]
        if self._fix_bonus:
            self._grasp[:], self._target[:] = self._grasp_bonus, self._target_bonus
        if self._physics == "sequential":
            for i in range(n):
                self._load_reset_state(i)
        self._reset_seeds()
        self._reset_options()
        return self._process_obs(all_idx, np.tile(self._obs0, (n, 1)))

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).reshape(self.num_envs, -1)

    def step_wait(self) -> VecEnvStepReturn:
        n = self.num_envs
        # Same action -> ctrl mapping as SoFetchEnv.step / _apply_action (absolute control)
        action = -1 + (self._actions * 2) / 10
        ctrl = np.clip(self._ctrl_center + action * self._ctrl_half, self._ctrl_lo, self._ctrl_hi)

        if self._physics == "rollout":
            control = np.repeat(ctrl[:, None, :], self.N_SUBSTEPS, axis=1)
            self._pool.rollout(self._models, self._thread_data, self._state, control, nstep=self.N_SUBSTEPS,
                               state=self._out_state, sensordata=self._out_sensor, chunk_size=self._chunk_size,
                               skip_checks=True)
            self._state[:] = self._out_state[:, -1]
            jaw = self._out_sensor[:, -1, :3].copy()
        else:
            jaw = np.empty((n, 3))
            for i, d in enumerate(self._datas):
                d.ctrl[:] = ctrl[i]
                mujoco.mj_step(self.model, d, nstep=self.N_SUBSTEPS)
                mujoco.mj_getState(self.model, d, self._state[i], _FULLPHYSICS)
                jaw[i] = d.site_xpos[self._jaw_site]

        raw = self._raw_obs(self._state, jaw)

        # Reward: same two-stage shaping and one-time bonuses as SoFetchEnv._compute_reward
        rew_jaw = -np.array([np.linalg.norm(r) for r in raw[:, 22:25]])
        rew_tgt = -np.array([np.linalg.norm(r) for r in raw[:, 25:28]])
        pre_grasp = self._grasp > 0
        grasped = pre_grasp & (np.abs(rew_jaw) < 0.004)
        jaw_w = np.where(pre_grasp, 0.75, 0.25)
        tgt_w = np.where(pre_grasp, 0.25, 0.75)
        reward = np.where(grasped, self._grasp, 0.0) + ((jaw_w * rew_jaw) + (tgt_w * rew_tgt))
        self._grasp[grasped] = 0
        success = (np.abs(rew_tgt) < 0.002) & (self._target > 0)
        reward = reward + np.where(success, self._target, 0.0)
        self._target[success] = 0

        self._t += 1
        self._ep_ret += reward
        self._ep_len += 1
        truncated = self._t >= self.MAX_TIMESTEPS

        for i in range(n):
            info = self._infos[i]
            info["reset_flag"] = False
            info["total_timesteps"] = int(self._t[i])
            info["rew_jaw_center_to_object"] = rew_jaw[i]
            info["rew_object_to_target"] = rew_tgt[i]
            info["rew_other"] = 0
            info["rew_jaw_center_to_object_prop"] = jaw_w[i] * rew_jaw[i]
            info["rew_object_to_target_prop"] = tgt_w[i] * rew_tgt[i]
            if success[i]:
                info["is_success"] = 1
            info["TimeLimit.truncated"] = bool(truncated[i])

        obs = self._process_obs(np.arange(n), raw)
        infos = self._infos
        done_idx = np.flatnonzero(truncated)
        if len(done_idx):
            now = time.time()
            for i in done_idx:
                infos[i]["episode"] = {"r": round(float(self._ep_ret[i]), 6), "l": int(self._ep_len[i]),
                                       "t": round(now - self._t_start[i], 6)}
                infos[i]["terminal_observation"] = obs[i].copy()
            out_infos = [dict(i) for i in infos]  # values are scalars or fresh arrays, so a shallow copy is enough
            # auto-reset finished envs
            self._state[done_idx] = self._init_state
            self._t[done_idx] = 0
            self._ep_ret[done_idx] = 0
            self._ep_len[done_idx] = 0
            self._t_start[done_idx] = now
            if self._fix_bonus:
                self._grasp[done_idx], self._target[done_idx] = self._grasp_bonus, self._target_bonus
            for i in done_idx:
                self._infos[i] = self._reset_info()
                if self._physics == "sequential":
                    self._load_reset_state(i)
            obs[done_idx] = self._process_obs(done_idx, np.tile(self._obs0, (len(done_idx), 1)))
        else:
            out_infos = [dict(i) for i in infos]

        return obs, reward.astype(np.float32), truncated.copy(), out_infos

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
        # Emulates Monitor (adds info["episode"]); nothing else
        return [wrapper_class is Monitor for _ in self._get_indices(indices)]

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return [None] * self.num_envs
