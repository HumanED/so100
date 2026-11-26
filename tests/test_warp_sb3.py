"""SoFetchWarpVecEnv (MuJoCo Warp behind the SB3 VecEnv API) must look like SoFetchVecEnv to SB3.

Checks spaces, reset observation, the action mapping (first steps track the CPU env closely on the same actions),
auto-reset with terminal_observation / TimeLimit.truncated, the env-0 info dict TensorboardCallback reads, and that
SB3 PPO trains on it inside VecMonitor + VecNormalize. Runs on Warp's CPU device if there's no GPU (slow but fine).
    python tests/test_warp_sb3.py        (or: pytest tests/)
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so_arm_rl.envs.fetch.so_arm_fetch_vec_env import SoFetchVecEnv  # noqa: E402
from so_arm_warp.sb3_vec_env import SoFetchWarpVecEnv  # noqa: E402

N = 4


def _device():
    import warp as wp
    return "cuda:0" if wp.get_cuda_device_count() else "cpu"


def test_matches_cpu_vec_env(steps=120, seed=0):
    cpu = SoFetchVecEnv(N, physics="sequential")
    gpu = SoFetchWarpVecEnv(N, physics="original", device=_device())
    assert gpu.observation_space.shape == cpu.observation_space.shape
    assert gpu.action_space == cpu.action_space
    o_cpu, o_gpu = cpu.reset(), gpu.reset()
    assert np.abs(o_cpu - o_gpu).max() < 1e-5, "reset observation differs"
    assert set(gpu.get_attr("info")[0]) == set(cpu.get_attr("info")[0])

    rng = np.random.default_rng(seed)
    early_err, n_done = 0.0, 0
    for t in range(steps):
        a = rng.integers(0, 64, (N, 6))
        cpu.step_async(a); gpu.step_async(a)
        o_cpu, r_cpu, d_cpu, i_cpu = cpu.step_wait()
        o_gpu, r_gpu, d_gpu, i_gpu = gpu.step_wait()
        assert o_gpu.shape == o_cpu.shape and r_gpu.shape == r_cpu.shape
        assert (d_gpu == d_cpu).all(), f"episode boundaries differ at step {t}"
        if t < 3:   # same actions, same physics settings: arm joints agree until float32 / contact chaos takes over
            early_err = max(early_err, np.abs(o_cpu[:, :6] - o_gpu[:, :6]).max())
        if d_gpu.any():
            n_done += int(d_gpu.sum())
            for i in np.flatnonzero(d_gpu):
                assert i_gpu[i]["TimeLimit.truncated"] and "terminal_observation" in i_gpu[i]
                assert i_gpu[i]["terminal_observation"].shape == o_gpu[i].shape
            assert np.abs(o_gpu[d_gpu] - o_cpu[d_cpu]).max() < 1e-5, "auto-reset observation differs"
            assert gpu.get_attr("info")[0]["reset_flag"]
        else:
            info0 = gpu.get_attr("info")[0]
            assert not info0["reset_flag"] and info0["total_timesteps"] == i_cpu[0]["total_timesteps"]
    print(f"max |arm qpos difference| over the first 3 steps: {early_err:.2e}; {n_done} episodes ended")
    assert early_err < 1e-3
    assert n_done == N


def test_sb3_ppo_trains():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
    venv = VecNormalize(VecMonitor(SoFetchWarpVecEnv(N, device=_device())), norm_obs=True, norm_reward=True,
                        clip_obs=10.0)
    model = PPO("MlpPolicy", venv, n_steps=50, batch_size=100, n_epochs=1, verbose=0, device="cpu")
    model.learn(400)
    assert model.num_timesteps >= 400 and len(model.ep_info_buffer) == N


def test_split_across_devices_matches_halves(steps=110):
    """devices=[a, b] (one experiment's envs split over two GPUs) is exactly two half-size envs joined in order:
    obs, rewards, dones, per-env infos (env indices offset for the second half) and env 0's info dict.
    (Not bit-identical to one N-world batch: Warp's float32 results depend slightly on the batch size.)
    Splits over cuda:0 / cuda:1 when there are two GPUs, else two copies on one device."""
    import warp as wp
    d = _device()
    pair = ["cuda:0", "cuda:1"] if wp.get_cuda_device_count() >= 2 else [d, d]
    halves = [SoFetchWarpVecEnv(N // 2, device=pair[k], seed=k) for k in range(2)]
    two = SoFetchWarpVecEnv(N, devices=pair)
    np.testing.assert_array_equal(np.concatenate([h.reset() for h in halves]), two.reset())
    rng = np.random.default_rng(1)
    ended = 0
    for t in range(steps):   # crosses the 100-step episode boundary (auto-reset, terminal_observation)
        a = rng.integers(0, 64, (N, two.action_space.shape[0]))
        ref = [h.step(a[k * N // 2:(k + 1) * N // 2]) for k, h in enumerate(halves)]
        o, r, dn, info = two.step(a)
        np.testing.assert_allclose(o, np.concatenate([x[0] for x in ref]), atol=1e-6, err_msg=f"step {t}")
        np.testing.assert_allclose(r, np.concatenate([x[1] for x in ref]), atol=1e-6, err_msg=f"step {t}")
        assert (dn == np.concatenate([x[2] for x in ref])).all()
        for x, y in zip(info, ref[0][3] + ref[1][3]):
            assert x.keys() == y.keys()
            if "terminal_observation" in x:
                ended += 1
                np.testing.assert_allclose(x["terminal_observation"], y["terminal_observation"], atol=1e-6)
        assert two.get_attr("info")[0] == halves[0].get_attr("info")[0]
    assert ended == N    # every env finished one episode, in both halves


if __name__ == "__main__":
    test_matches_cpu_vec_env()
    test_sb3_ppo_trains()
    test_split_across_devices_matches_halves()
    print("OK")
