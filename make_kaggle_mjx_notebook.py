"""
Builds kaggle_train_mjx.ipynb: a self-contained Kaggle notebook that trains the GPU (MJX + Brax PPO)
version of the fetch task. so_arm_mjx, visualise_mjx.py and the scene XML/STLs are embedded as a
base64 tarball. Re-run after changing any of them:

    python make_kaggle_mjx_notebook.py

Kaggle's API (kaggle kernels push) rejects notebooks over 1 MB, and the embedded one is ~1.2 MB. For API
pushes, build a slim notebook that git-clones the code from GitHub instead (the branch must be pushed):

    python make_kaggle_mjx_notebook.py --github carlgpu      # -> kaggle_train_mjx_github.ipynb
"""
import argparse
import base64
import io
import json
import os
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/HumanED/so100.git"
_args = argparse.ArgumentParser()
_args.add_argument("--github", metavar="BRANCH", default=None,
                   help="clone code from this GitHub branch instead of embedding it")
ARGS = _args.parse_args()
OUT = os.path.join(HERE, "kaggle_train_mjx_github.ipynb" if ARGS.github else "kaggle_train_mjx.ipynb")
EMBED = ["so_arm_mjx", "visualise_mjx.py", os.path.join("so_arm_rl", "envs", "resources")]


def pack() -> str:
    def keep(ti):
        if "__pycache__" in ti.name or ti.name.endswith((".pyc", ".DS_Store")):
            return None
        return ti

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in EMBED:
            tar.add(os.path.join(HERE, p), arcname=p, filter=keep)
    return base64.b64encode(buf.getvalue()).decode()


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


cells = [
    md("""
# SO-100 Fetch: GPU training (MuJoCo MJX + Brax PPO)

The whole simulation runs on the GPU: thousands of copies of the arm are stepped in parallel, and
PPO runs on the same device. See `so_arm_mjx/fetch_env.py` for how this differs from the CPU `SoFetchEnv`
(continuous actions, collision filtering, bonuses re-armed per episode).

**Kaggle settings (right sidebar)**
- **Accelerator: GPU T4 x1** (or P100). Only one GPU is used.
- **Internet: On** (for `pip install`).
- Run with **Save Version → Save & Run All (Commit)** so it trains in the background.
  Download `results.zip` from the version's **Output** tab.

**Outputs**
- `models/<run>/*.pkl`: policy + observation normaliser at each eval, plus `final.pkl`.
  Play one locally with `python visualise_mjx.py models/<run>/final.pkl` (needs `pip install brax mujoco-mjx`).
- `checkpoints/<run>/<step>/`: Brax checkpoints, for resuming via `RESTORE_CHECKPOINT`.
- `logs/<run>.csv` and `<run>_curves.png`: eval metrics.
"""),
    code("""
# Versions tested together (brax 0.14.2 breaks with jax >= 0.10). jax 0.9.2 needs Python >= 3.11.
%pip install -q "jax[cuda12]==0.9.2" "brax==0.14.2" "flax==0.12.6" "mujoco==3.3.5" "mujoco-mjx==3.3.5" "imageio[ffmpeg]"
"""),
    code("""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")   # recommended for MJX on GPU
os.environ.setdefault("MUJOCO_GL", "egl")                               # headless rendering for the video
import jax
print("JAX", jax.__version__, "devices:", jax.devices())
if jax.default_backend() != "gpu":
    print("WARNING: no GPU visible to JAX. Turn on a GPU accelerator. On CPU this will be extremely slow.")
"""),
    md(f"## Get the code (clone `{ARGS.github}` from GitHub)") if ARGS.github else md("## Unpack the embedded code"),
    code(f"""
import shutil, subprocess, sys
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("kaggle_mjx_out")
SRC = os.path.join(WORK, "src")
shutil.rmtree(SRC, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", "-b", "{ARGS.github}", "{REPO_URL}", SRC], check=True)
commit = subprocess.run(["git", "-C", SRC, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
sys.path.insert(0, SRC)
print("Code in", SRC, "at commit", commit)
""") if ARGS.github else code(f"""
import base64, io, sys, tarfile
PKG_B64 = "{pack()}"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("kaggle_mjx_out")
SRC = os.path.join(WORK, "src")
os.makedirs(SRC, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PKG_B64)), mode="r:gz") as tar:
    tar.extractall(SRC, **({{"filter": "data"}} if hasattr(tarfile, "data_filter") else {{}}))
sys.path.insert(0, SRC)
print("Code in", SRC)
"""),
    md("## Config"),
    code("""
from so_arm_mjx.fetch_env import FetchConfig, SoFetchMjxEnv
from so_arm_mjx.train import TrainConfig, train

RESTORE_CHECKPOINT = None   # e.g. "/kaggle/input/<prev-run>/checkpoints/PPO-carl-mjx-1/000050003968"

cfg = TrainConfig(
    run_name="PPO-carl-mjx-1",
    out_dir=WORK,
    num_timesteps=200_000_000,   # upper bound; the time limit below stops it at an eval/save point
    time_limit_hours=11.0,       # Kaggle sessions end at 12 h
    num_envs=2048,               # try 4096 if the benchmark below shows the GPU isn't saturated
    num_evals=40,                # = number of save points
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.99,
    policy_hidden_layer_sizes=(256, 256, 128),
    value_hidden_layer_sizes=(256, 256, 256),
    restore_checkpoint_path=RESTORE_CHECKPOINT,
    env=FetchConfig(
        n_substeps=20, timestep=0.002,   # same as the CPU env; (10, 0.004) is ~2x faster but coarser physics
        cone="pyramidal", solver_iterations=10, ls_iterations=10,
        fixed_goal=(-0.3, -0.2, 0.025),  # None = random goal per env
    ),
)
assert (cfg.batch_size * cfg.num_minibatches) % cfg.num_envs == 0, "batch_size * num_minibatches must be divisible by num_envs"
"""),
    md("## Quick throughput benchmark (simulation only)"),
    code("""
import time, jax.numpy as jp
_env = SoFetchMjxEnv(cfg.env)
_reset, _step = jax.jit(jax.vmap(_env.reset)), jax.jit(jax.vmap(_env.step))
_s = _reset(jax.random.split(jax.random.PRNGKey(0), cfg.num_envs))
_a = jp.zeros((cfg.num_envs, _env.action_size))
t = time.time(); _s = _step(_s, _a); jax.block_until_ready(_s.obs); print(f"compile: {time.time()-t:.0f} s")
t = time.time()
for _ in range(10):
    _s = _step(_s, _a)
jax.block_until_ready(_s.obs)
sps = 10 * cfg.num_envs / (time.time() - t)
print(f"{sps:,.0f} env steps/s with {cfg.num_envs} envs -> ~{sps*3600/1e6:,.0f}M steps/hour before PPO overhead")
del _env, _reset, _step, _s
"""),
    md("## Train"),
    code("""
models_dir, log_path = train(cfg)
"""),
    md("## Training curves"),
    code("""
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(log_path)
cols = [c for c in ["eval/episode_reward", "eval/episode_rew_jaw_center_to_object",
                    "eval/episode_rew_object_to_target", "eval/episode_grasp", "eval/episode_success"] if c in df]
fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 3.4), squeeze=False)
for ax, c in zip(axes[0], cols):
    ax.plot(df["step"], df[c]); ax.set_title(c.replace("eval/episode_", "")); ax.set_xlabel("timesteps"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(WORK, f"{cfg.run_name}_curves.png"), dpi=120); plt.show()
df.tail()
"""),
    md("## Evaluate in CPU MuJoCo + video"),
    code("""
import subprocess
final = os.path.join(models_dir, "final.pkl")
video = os.path.join(WORK, f"{cfg.run_name}_eval.mp4")
r = subprocess.run([sys.executable, os.path.join(SRC, "visualise_mjx.py"), final, "--episodes", "3", "--video", video],
                   cwd=SRC, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC, "JAX_PLATFORMS": "cpu"})
print(r.stdout[-2000:])
if r.returncode != 0:
    print("Video failed (no EGL?). Falling back to a headless eval:\\n", r.stderr[-1500:])
    r = subprocess.run([sys.executable, os.path.join(SRC, "visualise_mjx.py"), final, "--episodes", "5", "--headless"],
                       cwd=SRC, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC, "JAX_PLATFORMS": "cpu"})
    print(r.stdout[-2000:], r.stderr[-1000:] if r.returncode else "")
"""),
    md("## Package outputs"),
    code("""
import zipfile, glob
zip_path = os.path.join(WORK, "results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ("models", "checkpoints", "logs"):
        for root, _, files in os.walk(os.path.join(WORK, sub)):
            for f in files:
                full = os.path.join(root, f); zf.write(full, os.path.relpath(full, WORK))
    for f in glob.glob(os.path.join(WORK, f"{cfg.run_name}_*")):
        zf.write(f, os.path.basename(f))
print("Wrote", zip_path, f"({os.path.getsize(zip_path)/1e6:.1f} MB)")
"""),
]

for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")
