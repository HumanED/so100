"""
SoFetchVecEnv must behave like SoFetchEnv inside SubprocVecEnv (train_parallel.py's setup).

    python tests/test_vec_env.py        (or: pytest tests/)
"""
import os
import sys

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv  # noqa: E402
from so_arm_rl.envs.fetch.so_arm_fetch_vec_env import SoFetchVecEnv  # noqa: E402

N, STEPS = 8, 350


def make_env(rank, seed=1):  # same as train_parallel.make_env
    def inner():
        env = SoFetchEnv()
        env.reset(seed=rank + seed)
        return env
    return inner


def compare(physics):
    ref = VecMonitor(SubprocVecEnv([make_env(i) for i in range(N)]))
    new = VecMonitor(SoFetchVecEnv(N, physics=physics))
    o1, o2 = ref.reset(), new.reset()
    worst = dict(obs=np.abs(o1 - o2).max(), reward=0.0, info=0.0, episode_r=0.0)
    rng = np.random.default_rng(0)
    dones_equal, keys_equal, episodes = True, True, 0
    for _ in range(STEPS):
        a = rng.integers(0, 64, (N, 6))
        o1, r1, d1, i1 = ref.step(a)
        o2, r2, d2, i2 = new.step(a)
        assert o1.dtype == o2.dtype and r1.dtype == r2.dtype, (o1.dtype, o2.dtype, r1.dtype, r2.dtype)
        worst["obs"] = max(worst["obs"], np.abs(o1 - o2).max())
        worst["reward"] = max(worst["reward"], np.abs(r1 - r2).max())
        dones_equal &= bool((d1 == d2).all())
        for x, y in zip(i1, i2):
            keys_equal &= set(x) == set(y)
            for k in x:
                if k.startswith("rew_") or k == "is_success":
                    worst["info"] = max(worst["info"], abs(float(x[k]) - float(y[k])))
            if "episode" in x:
                episodes += 1
                worst["episode_r"] = max(worst["episode_r"], abs(x["episode"]["r"] - y["episode"]["r"]))
                worst["obs"] = max(worst["obs"], np.abs(x["terminal_observation"] - y["terminal_observation"]).max())
    current_info_keys = set(ref.get_attr("info")[0]) == set(new.get_attr("info")[0])
    ref.close(); new.close()
    return worst, dones_equal, keys_equal and current_info_keys, episodes


def test_sequential_is_exact():
    worst, dones, keys, eps = compare("sequential")
    print(f"sequential: {worst} dones_equal={dones} info_keys_equal={keys} episodes={eps}")
    assert all(v == 0 for v in worst.values()) and dones and keys and eps > 0


def test_rollout_is_close():
    worst, dones, keys, eps = compare("rollout")
    print(f"rollout:    {worst} dones_equal={dones} info_keys_equal={keys} episodes={eps}")
    assert dones and keys and eps > 0
    assert worst["episode_r"] < 1.0   # solver warm-start differences only show after contacts


if __name__ == "__main__":
    test_sequential_is_exact()
    test_rollout_is_close()
    print("OK")
