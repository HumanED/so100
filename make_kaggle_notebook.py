"""
Builds kaggle_train.ipynb: a self-contained Kaggle notebook that trains PPO on SoFetchEnv.

The so_arm_rl package (python + MuJoCo XML + STL meshes) is embedded in the notebook as a
base64 tarball, so nothing needs uploading to Kaggle besides the notebook itself.
Re-run this script whenever you change the environment code:

    python make_kaggle_notebook.py
"""
import base64
import io
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(HERE, "so_arm_rl")
OUT = os.path.join(HERE, "kaggle_train.ipynb")


def pack_package() -> str:
    def keep(ti: tarfile.TarInfo):
        name = ti.name
        if "__pycache__" in name or ".egg-info" in name or name.endswith((".pyc", ".DS_Store")):
            return None
        return ti

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(PKG_DIR, arcname="so_arm_rl", filter=keep)
    return base64.b64encode(buf.getvalue()).decode()


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


cells = [
    md("""
# SO-100 Fetch: PPO training on Kaggle

Trains PPO (stable-baselines3) on `SoFetchEnv` with the same setup as `train_parallel.py` (`VecMonitor` +
`VecNormalize`). Checkpoints are saved as `models/<run>/<steps>.zip` with `vec_norm_stats/<run>/<steps>.pkl`,
the layout `visualise_model.py` expects.

By default all envs run in one process with `SoFetchVecEnv` (MuJoCo's multithreaded rollout, no worker processes).
It's about 1.7x faster than one process per env, with the same observations, rewards and infos as `SoFetchEnv`.
Set `VEC_ENV = "subproc"` for the original `SubprocVecEnv`.

**Kaggle settings (right sidebar)**
- **Internet: On** (needed for `pip install`)
- **Accelerator: None (CPU)**, recommended. SB3 runs the policy once per env step, and with this small network
  each GPU round trip costs more than it saves (on Apple Metal it was ~10x slower). CPU sessions also don't use GPU quota.
  To try a GPU anyway (e.g. with a much bigger `NET_ARCH`), turn one on and set `USE_GPU = True`.
- Use **Save Version → Save & Run All (Commit)** so training keeps running in the background with the browser closed (12 h limit).
  The models, TensorBoard logs and a `results.zip` then appear under the version's **Output** tab.

**Resuming:** add a previous run's output (or any uploaded model `.zip`) as an input dataset, then set `RESUME_FROM`.
"""),
    code("""
%pip install -q "stable-baselines3==2.5.0" "gymnasium==1.0.0" "mujoco==3.3.5" tensorboard
"""),
    md("## Config"),
    code("""
import os

RUN_NAME = "PPO-carl-35-sofetch-kaggle"   # names the model + log folders
TOTAL_TIMESTEPS = 20_000_000              # upper bound; time limit below usually stops it first
TIME_LIMIT_HOURS = 11.25                   # Kaggle kills sessions at 12 h, so leave time to save + zip
SAVE_INTERVAL = 250_000                   # checkpoint every N timesteps...
START_SAVING = 500_000                    # ...once past this many
VEC_ENV = "single_process"                # "single_process" (SoFetchVecEnv, faster) or "subproc" (original)
N_ENVS = 32 if VEC_ENV == "single_process" else os.cpu_count()  # Kaggle CPU sessions have 4 cores
ROLLOUT_SIZE = 16384                      # env steps per PPO update (same as train_parallel.py); n_steps = ROLLOUT_SIZE // N_ENVS
SEED = 1

# Resume: path to a model .zip, e.g. "/kaggle/input/my-prev-run/models/PPO-carl-35-sofetch-kaggle/5000000.zip"
# Or a directory, in which case the checkpoint with the highest timestep number is used.
RESUME_FROM = None

import torch
USE_GPU = False                           # see the note at the top; CPU is faster for the default network
DEVICE = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
print("PPO device:", DEVICE, f"({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")

# Policy/value network sizes. None = SB3 default (64, 64). A bigger net, e.g. dict(pi=[256, 256], vf=[256, 256]),
# is where the GPU starts to pay off, but models will then differ in architecture from earlier runs.
NET_ARCH = None

# PPO hyperparameters (SB3 defaults except batch_size; tweak freely)
PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=max(1, ROLLOUT_SIZE // N_ENVS),  # per env
    batch_size=512 if DEVICE == "cuda" else 64,  # tiny minibatches leave a GPU idle; 512 is still 16 minibatches/epoch at 4 envs
    n_epochs=10,
    gamma=0.99,
)
if NET_ARCH is not None:
    PPO_KWARGS["policy_kwargs"] = dict(net_arch=NET_ARCH)

RECORD_VIDEO = True                       # record an evaluation video at the end (may fail without EGL)

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("kaggle_out")
MODELS_DIR = os.path.join(WORK, "models")
STATS_DIR = os.path.join(WORK, "vec_norm_stats")   # VecNormalize mean/std, saved next to each checkpoint
LOGS_DIR = os.path.join(WORK, "logs")
for _d in (MODELS_DIR, STATS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)
print("Output dir:", WORK, "| envs:", N_ENVS, "| device:", DEVICE)
"""),
    md("## Install the `so_arm_rl` environment package (embedded)"),
    code(f"""
import base64, io, sys, tarfile

PKG_B64 = "{pack_package()}"

SRC_DIR = os.path.join(WORK, "src") if WORK.startswith("/kaggle") else os.path.abspath("kaggle_src")
os.makedirs(SRC_DIR, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PKG_B64)), mode="r:gz") as tar:
    tar.extractall(SRC_DIR, **({{"filter": "data"}} if hasattr(tarfile, "data_filter") else {{}}))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# Subprocess workers also need to import it
os.environ["PYTHONPATH"] = SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")
print("Extracted to", SRC_DIR)
"""),
    md("## Env factory + callbacks"),
    code("""
import glob, re, time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from so_arm_rl.envs.fetch.so_arm_fetch_vec_env import SoFetchVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from so_arm_rl.envs.fetch.so_arm_fetch_env import SoFetchEnv


def make_env(rank=0, render_mode=None):
    \"\"\"Same as train_parallel.make_env: the raw env (normalisation and monitoring happen on the VecEnv)\"\"\"
    def _init():
        env = SoFetchEnv(render_mode=render_mode)
        env.reset(seed=SEED + rank)
        return env
    return _init


class TensorboardCallback(BaseCallback):
    \"\"\"Logs mean per-episode sub-rewards (rew_* and debug_*) of env 0, same as train_parallel.py\"\"\"
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.sub_rews_cumul, self.sub_rews_buffer = {}, {}
        self.buffer_idx = 0
        self.ignore_reset_flag = True

    def _on_training_start(self):
        self.max_t = self.training_env.get_attr("MAX_TIMESTEPS")[0]
        for k in self.training_env.get_attr("info")[0]:
            if k.startswith("rew_") or k.startswith("debug_"):
                self.sub_rews_cumul[k] = 0
                self.sub_rews_buffer[k] = np.zeros(self.max_t)

    def _on_step(self):
        info = self.training_env.get_attr("info")[0]
        if info["reset_flag"] and not self.ignore_reset_flag:
            self.episode_count += 1
            for k in self.sub_rews_cumul:
                self.sub_rews_buffer[k][self.buffer_idx] = self.sub_rews_buffer[k][self.buffer_idx - 1]
                self.sub_rews_cumul[k] += np.sum(self.sub_rews_buffer[k])
                self.sub_rews_buffer[k] = np.zeros(self.max_t)
            self.buffer_idx = 0
        else:
            self.ignore_reset_flag = False
            for k in self.sub_rews_cumul:
                if k in info and self.buffer_idx < self.max_t:
                    self.sub_rews_buffer[k][self.buffer_idx] = info[k]
            self.buffer_idx += 1
        return True

    def _on_rollout_end(self):
        for k, v in self.sub_rews_cumul.items():
            self.logger.record(f"rollout/{k}_mean", float(v) / self.episode_count if self.episode_count else 0.0)
            self.sub_rews_cumul[k] = 0
        self.episode_count = 0


class CheckpointAndTimeLimit(BaseCallback):
    \"\"\"Saves every SAVE_INTERVAL timesteps and stops cleanly before the Kaggle time limit.\"\"\"
    def __init__(self, save_dir, stats_dir, interval, start_saving, time_limit_s):
        super().__init__()
        self.save_dir, self.stats_dir, self.interval, self.start_saving = save_dir, stats_dir, interval, start_saving
        self.time_limit_s = time_limit_s
        self.t0 = time.time()
        self.next_save = None

    def _on_training_start(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        # first multiple of interval past current timestep count (works when resuming)
        self.next_save = (self.num_timesteps // self.interval + 1) * self.interval

    def _on_step(self):
        if self.num_timesteps >= self.next_save:
            if self.num_timesteps >= self.start_saving:
                path = os.path.join(self.save_dir, str(self.next_save))
                self.model.save(path)
                self.model.get_vec_normalize_env().save(os.path.join(self.stats_dir, f"{self.next_save}.pkl"))
                elapsed = (time.time() - self.t0) / 3600
                print(f"[ckpt] {self.next_save:,} steps -> {path}.zip  ({elapsed:.2f} h elapsed)")
            self.next_save += self.interval
        if time.time() - self.t0 > self.time_limit_s:
            print(f"[time limit] stopping at {self.num_timesteps:,} timesteps")
            return False
        return True
"""),
    md("## Smoke test the environment"),
    code("""
_env = make_env()()
obs, info = _env.reset()
for _ in range(5):
    obs, r, term, trunc, info = _env.step(_env.action_space.sample())
print("obs shape", obs.shape, "| reward", round(float(r), 4), "| action space", _env.action_space)
_env.close()

# quick throughput check of the vectorised env used for training (simulation only)
if VEC_ENV == "single_process":
    _venv = SoFetchVecEnv(N_ENVS)
    _venv.reset()
    _a = np.stack([_venv.action_space.sample() for _ in range(N_ENVS)])
    for _ in range(20):
        _venv.step(_a)
    t = time.time()
    for _ in range(200):
        _venv.step(_a)
    sps = 200 * N_ENVS / (time.time() - t)
    _venv.close()
    print(f"SoFetchVecEnv: {sps:,.0f} env steps/s with {N_ENVS} envs on {os.cpu_count()} cores "
          f"(~{sps * 3600 / 1e6:.0f}M/hour before PPO overhead; expect roughly a third of that end to end)")
"""),
    md("## Train"),
    code("""
def resolve_resume(path):
    if path is None:
        return None
    if os.path.isdir(path):
        zips = glob.glob(os.path.join(path, "**", "*.zip"), recursive=True)
        nums = [(int(m.group(1)), z) for z in zips if (m := re.match(r"(\\d+)", os.path.basename(z)))]
        if not nums:
            raise FileNotFoundError(f"No numbered checkpoints in {path}")
        return max(nums)[1]
    return path

run_models_dir = os.path.join(MODELS_DIR, RUN_NAME)
run_stats_dir = os.path.join(STATS_DIR, RUN_NAME)
resume_path = resolve_resume(RESUME_FROM)
if resume_path is None and glob.glob(os.path.join(run_models_dir, "*.zip")):
    raise RuntimeError(f"{run_models_dir} already has checkpoints; change RUN_NAME or set RESUME_FROM")

if VEC_ENV == "single_process":
    env = SoFetchVecEnv(N_ENVS)
else:
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)]) if N_ENVS > 1 else DummyVecEnv([make_env(0)])
env = VecMonitor(env)

if resume_path:
    # VecNormalize stats live in vec_norm_stats/<run>/<steps>.pkl next to models/<run>/<steps>.zip
    stats = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(resume_path))), "vec_norm_stats",
                         os.path.basename(os.path.dirname(resume_path)), os.path.basename(resume_path)[:-4] + ".pkl")
    print("Resuming from", resume_path, "with", stats)
    env = VecNormalize.load(stats, env)
    env.training, env.norm_reward = True, True
    model = PPO.load(resume_path, env=env, device=DEVICE, seed=SEED, tensorboard_log=LOGS_DIR,
                     custom_objects={"n_steps": PPO_KWARGS["n_steps"], "lr_schedule": 3e-4, "clip_range": 0.2})
else:
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)   # same as train_parallel.py
    model = PPO("MlpPolicy", env, device=DEVICE, seed=SEED, tensorboard_log=LOGS_DIR, verbose=1, **PPO_KWARGS)

callbacks = CallbackList([
    TensorboardCallback(),
    CheckpointAndTimeLimit(run_models_dir, run_stats_dir, SAVE_INTERVAL, START_SAVING, TIME_LIMIT_HOURS * 3600),
])

t0 = time.time()
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name=RUN_NAME,
        reset_num_timesteps=resume_path is None,
        progress_bar=False,
    )
except KeyboardInterrupt:
    print("Interrupted, saving what we have")
finally:
    final_path = os.path.join(run_models_dir, f"{model.num_timesteps}_final")
    model.save(final_path)
    os.makedirs(run_stats_dir, exist_ok=True)
    final_stats = os.path.join(run_stats_dir, f"{model.num_timesteps}_final.pkl")
    env.save(final_stats)
    env.close()
    print(f"Saved final model -> {final_path}.zip | {model.num_timesteps:,} timesteps in {(time.time()-t0)/3600:.2f} h")
"""),
    md("## Training curves"),
    code("""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ev_files = sorted(glob.glob(os.path.join(LOGS_DIR, f"{RUN_NAME}_*", "events.*")))
if not ev_files:
    print("No TensorBoard logs found")
else:
    ea = EventAccumulator(os.path.dirname(ev_files[-1])); ea.Reload()
    tags = [t for t in ["rollout/ep_rew_mean", "rollout/rew_jaw_center_to_object_mean",
                        "rollout/rew_object_to_target_mean", "train/value_loss"] if t in ea.Tags()["scalars"]]
    fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 3.5), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        s = ea.Scalars(tag)
        ax.plot([e.step for e in s], [e.value for e in s])
        ax.set_title(tag); ax.set_xlabel("timesteps"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(WORK, f"{RUN_NAME}_curves.png"), dpi=120); plt.show()
"""),
    md("## Evaluate (+ optional video)"),
    code("""
# Same as visualise_model.py: frozen VecNormalize stats, unnormalised rewards
if RECORD_VIDEO:
    os.environ.setdefault("MUJOCO_GL", "egl")
def make_eval(render):
    venv = VecNormalize.load(final_stats, DummyVecEnv([make_env(rank=100, render_mode="rgb_array" if render else None)]))
    venv.training, venv.norm_reward = False, False
    return venv
try:
    eval_env = make_eval(RECORD_VIDEO)
except Exception as e:
    print("Render env failed, evaluating without video:", e)
    RECORD_VIDEO = False
    eval_env = make_eval(False)
eval_model = PPO.load(final_path, env=eval_env, device=DEVICE)
returns, successes, frames = [], [], []
for ep in range(5):
    obs, ep_ret, done = eval_env.reset(), 0.0, False
    while not done:
        action, _ = eval_model.predict(obs, deterministic=True)
        obs, r, d, infos = eval_env.step(action)
        ep_ret += float(r[0]); done = bool(d[0])
        if RECORD_VIDEO and ep == 0:
            try:
                frames.append(eval_env.venv.envs[0].render())
            except Exception as e:
                print("Rendering failed, skipping video:", e); RECORD_VIDEO = False
    returns.append(ep_ret); successes.append(infos[0].get("is_success", 0))
eval_env.close()
print(f"Mean return {np.mean(returns):.2f} | success rate {np.mean(successes):.0%}")

if frames:
    import imageio
    vid = os.path.join(WORK, f"{RUN_NAME}_eval.mp4")
    imageio.mimsave(vid, frames, fps=25)   # 20 substeps x 2 ms = 40 ms per step
    print("Video saved ->", vid)
"""),
    md("## Package outputs\n\n`results.zip` has `models/`, `vec_norm_stats/` and `logs/`. Download it from the Output tab and unzip it into the repo root, so `visualise_model.py` can find the models."),
    code("""
# zip only models + logs (not the extracted source)
import zipfile
zip_path = os.path.join(WORK, "results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ("models", "vec_norm_stats", "logs"):
        for root, _, files in os.walk(os.path.join(WORK, sub)):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, WORK))
print("Wrote", zip_path, f"({os.path.getsize(zip_path)/1e6:.1f} MB)")
for p in sorted(glob.glob(os.path.join(run_models_dir, "*.zip"))):
    print("  ", os.path.relpath(p, WORK))
"""),
]

for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")
