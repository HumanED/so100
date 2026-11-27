"""
SoFetchWarpVecEnv: FetchWarpEnv (N worlds on one GPU, MuJoCo Warp) as a Stable-Baselines3 VecEnv.

Drop-in for SoFetchVecEnv / SubprocVecEnv in train_parallel.py (vec_env_backend = "warp"): same observation space
(31 values), same action space (MultiDiscrete, 64 bins per motor, mapped with SoFetchEnv's -1 + 2a/63 formula),
same reward ("default" in so_arm_warp/rewards.py, which tests/test_warp_reward.py checks against SoFetchEnv),
100-step episodes, auto-reset with "terminal_observation" and "TimeLimit.truncated". Wrap it in VecMonitor and
VecNormalize exactly as before.

Differences from SoFetchVecEnv, all for speed:
- Physics is MuJoCo Warp (float32 on the GPU). physics="original_20it" (default) is SoFetchEnv's model and solver
  with 20 iterations instead of 100: policies trained on it work in SoFetchEnv. "full_elliptic" is ~1.4x faster and
  also transfers. "full" is fastest but its pyramidal friction cone gives policies that fail in SoFetchEnv.
- Per-env info dicts are empty except for envs whose episode just ended. The full SoFetchEnv info dict (the rew_*
  keys TensorboardCallback reads) is kept for env 0 only, which is all TensorboardCallback looks at.
- Needs a CUDA GPU (Kaggle T4 is fine).

One experiment on several GPUs: devices=["cuda:0", "cuda:1"] splits num_envs evenly across them. Each step launches
the physics on every GPU before waiting on any, so they run at the same time; the results are joined into one batch
in the same order (envs 0..n/2-1 on cuda:0, the rest on cuda:1). SB3's PPO update still runs on one GPU (the first).
"""
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from .env import FetchWarpEnv
from .physics import PhysicsConfig

N_DISCRETE = 64        # SoFetchEnv.N_DISCRETE
MAX_TIMESTEPS = 100    # SoFetchEnv.MAX_TIMESTEPS


def _reset_info() -> dict:
    # Same dict SoFetchEnv.reset() creates
    return {"is_success": 0, "total_timesteps": 0, "rew_jaw_center_to_object": 0, "rew_object_to_target": 0,
            "rew_jaw_center_to_object_prop": 0, "rew_object_to_target_prop": 0, "reset_flag": True,
            "rew_grasp": 0, "rew_success": 0, "rew_jaw_open_prop": 0}


class SoFetchWarpVecEnv(VecEnv):
    def __init__(self, num_envs: int = 4096, physics: str = "original_20it", reward: str = "default",
                 device: Optional[str] = None, seed: int = 0, devices: Optional[Sequence[str]] = None):
        self.MAX_TIMESTEPS, self.N_DISCRETE, self.render_mode = MAX_TIMESTEPS, N_DISCRETE, None
        devices = list(devices) if devices else [device]
        if num_envs % len(devices):
            raise ValueError(f"num_envs={num_envs} must split evenly across {len(devices)} devices")
        per = num_envs // len(devices)
        self.envs = [FetchWarpEnv(per, PhysicsConfig(case=physics), reward=reward, episode_length=MAX_TIMESTEPS,
                                  device=d, seed=seed + k) for k, d in enumerate(devices)]
        self.env = self.envs[0]                     # holds env 0 (TensorboardCallback's info dict)
        self._slices = [slice(k * per, (k + 1) * per) for k in range(len(devices))]
        nu = self.env.action_size
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.env.obs_size,), dtype=np.float32)
        super().__init__(num_envs, obs_space, spaces.MultiDiscrete([N_DISCRETE] * nu))
        self.device = self.env.device               # where the PPO policy goes
        self.devices = [e.device for e in self.envs]
        self._actions: Optional[np.ndarray] = None
        self._info0 = _reset_info()

    # ---------------------------------------------------------------- VecEnv API
    def reset(self) -> VecEnvObs:
        self._info0 = _reset_info()
        self._reset_seeds()
        self._reset_options()
        return np.concatenate([e.reset().cpu().numpy() for e in self.envs])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions).reshape(self.num_envs, -1)

    def step_wait(self) -> VecEnvStepReturn:
        a = -1 + torch.as_tensor(self._actions, dtype=torch.float32) * 2 / (N_DISCRETE - 1)
        for e, sl in zip(self.envs, self._slices):          # start every GPU's physics before waiting on any
            e.step_launch(a[sl].to(e.device, non_blocking=True))
        infos: List[dict] = [{} for _ in range(self.num_envs)]
        obs_l, rew_l, done_l = [], [], []
        for e, sl in zip(self.envs, self._slices):
            obs, reward, done, info = e.step_finish()
            done_np = done.cpu().numpy()
            obs_l.append(obs.cpu().numpy()); rew_l.append(reward.cpu().numpy()); done_l.append(done_np)
            if done_np.any():
                idx = np.flatnonzero(done_np) + sl.start
                term = info["terminal_obs"].cpu().numpy()
                trunc = info["timeout"][done].cpu().numpy()
                zero = torch.zeros(len(idx), device=e.device)
                succ = (info["episode"].get("success", zero) > 0).cpu().numpy()
                grasp = (info["episode"].get("grasp", zero) > 0).cpu().numpy()
                for j, i in enumerate(idx):
                    infos[i] = {"terminal_observation": term[j], "TimeLimit.truncated": bool(trunc[j]),
                                "is_success": int(succ[j]), "grasped": int(grasp[j])}
        done_np = np.concatenate(done_l)
        self._update_info0(done_np[0])
        return np.concatenate(obs_l), np.concatenate(rew_l), done_np, infos

    def _update_info0(self, done0: bool) -> None:
        """SoFetchEnv's info dict for env 0 (what TensorboardCallback reads via get_attr("info")[0])."""
        if done0:
            self._info0 = _reset_info()
            return
        m = {k: float(v[0]) for k, v in self.env.last_metrics.items()}   # custom rewards may omit some keys
        jaw, tgt = m.get("rew_jaw_center_to_object", 0.0), m.get("rew_object_to_target", 0.0)
        i = self._info0
        i["reset_flag"] = False
        i["total_timesteps"] = int(self.env.t[0])
        i["rew_jaw_center_to_object"] = jaw
        i["rew_object_to_target"] = tgt
        i["rew_grasp"] = 20 if m.get("grasp") else 0
        i["rew_success"] = 30 if m.get("success") else 0
        i["rew_jaw_center_to_object_prop"] = 0.5 * jaw
        i["rew_object_to_target_prop"] = 0.5 * tgt
        i["rew_jaw_open_prop"] = m.get("rew_jaw_open", 0.0)
        if m.get("success"):
            i["is_success"] = 1
        i["TimeLimit.truncated"] = False

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        idx = self._get_indices(indices)
        if attr_name == "info":
            return [self._info0 for _ in idx]   # only env 0 is tracked
        return [getattr(self, attr_name) for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        raise NotImplementedError("SoFetchWarpVecEnv has no per-env sub-environments")

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return [None] * self.num_envs
