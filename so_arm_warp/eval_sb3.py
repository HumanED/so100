"""
Play an SB3 model (from train_sb3.py, train_parallel.py or Kaggle) in the real CPU SoFetchEnv: grasp / success
rate and return over N episodes, optionally an mp4. No GPU or Warp needed.

    python -m so_arm_warp.eval_sb3 warp_sb3_out/models/carl-default-1/final.zip --episodes 20
    python -m so_arm_warp.eval_sb3 <model.zip> --video out.mp4 --episodes 1

The VecNormalize statistics are found by swapping models/ -> vec_norm_stats/ and .zip -> .pkl (or pass --stats).
"""
import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv  # noqa: E402


def evaluate(model_path, stats=None, episodes=20, deterministic=True, video=None):
    stats = stats or model_path.replace("/models/", "/vec_norm_stats/").replace(".zip", ".pkl")
    if video:   # offscreen rendering needs EGL/OSMesa on Linux (MUJOCO_GL=egl on Kaggle); skip the video if it fails
        try:
            probe = SoFetchEnv(render_mode="rgb_array")
            probe.reset()
            probe.render()
            probe.close()
        except Exception as e:
            print("Offscreen rendering unavailable, evaluating without video:", type(e).__name__, e)
            video = None
    render = "rgb_array" if video else None
    venv = VecNormalize.load(stats, DummyVecEnv([lambda: SoFetchEnv(render_mode=render)]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(model_path, device="cpu")
    grasps = succ = 0
    rets, frames = [], []
    for _ in range(episodes):
        obs, ret, g, s = venv.reset(), 0.0, False, False
        for _ in range(venv.get_attr("MAX_TIMESTEPS")[0]):
            a, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, info = venv.step(a)
            ret += float(venv.get_original_reward()[0])
            g |= info[0].get("rew_grasp", 0) > 0
            s |= info[0].get("rew_success", 0) > 0
            if video:
                frames.append(venv.venv.envs[0].render())
            if done[0]:
                break
        grasps += g; succ += s; rets.append(ret)
    print(f"{os.path.basename(model_path)} on CPU SoFetchEnv ({'deterministic' if deterministic else 'sampled'}): "
          f"grasp {grasps}/{episodes}, success {succ}/{episodes}, mean return {np.mean(rets):+.2f}")
    if video and frames:
        import imageio
        imageio.mimsave(video, frames, fps=25)   # 20 substeps x 2 ms = 40 ms per step
        print("Wrote", video)
    return dict(grasp_rate=grasps / episodes, success_rate=succ / episodes, mean_return=float(np.mean(rets)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--stats", default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--sampled", action="store_true", help="sample actions instead of the most likely one")
    p.add_argument("--video", default=None)
    a = p.parse_args()
    evaluate(a.model, a.stats, a.episodes, not a.sampled, a.video)


if __name__ == "__main__":
    main()
