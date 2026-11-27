"""
Builds kaggle_train_warp.ipynb: Kaggle GPU notebook that trains SB3 PPO on MuJoCo Warp (so_arm_warp.train_sb3).
It git-clones the code (so it stays under Kaggle's 1 MB API limit and always runs the pushed branch):

    python make_kaggle_warp_notebook.py [--branch main]
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--branch", default="main")
a = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "kaggle_train_warp.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


cells = [
    md(f"""
# SO-100 Fetch: GPU training, Stable-Baselines3 PPO on MuJoCo Warp

Physics runs in MuJoCo Warp on the GPU (8,192 envs at once) behind a normal SB3 VecEnv (`so_arm_warp/sb3_vec_env.py`),
with the same PPO setup as `train_parallel.py` (VecMonitor + VecNormalize, MlpPolicy). A Kaggle GPU session has
**two T4s**, used one of two ways (`MODE` below):
- **`"variety"`** (default): **two experiments side by side**, one per T4 (e.g. two reward functions), ~16.6k steps/s each.
- **`"focus"`**: **one experiment on both T4s** (the first in `EXPERIMENTS`), ~24k steps/s, ~1.45× faster. Same PPO
  settings, so results are comparable. Use it once something looks promising and you want that one run sooner.

Code is cloned from the `{a.branch}` branch.

**Kaggle settings:** Accelerator **GPU T4 x2**, Internet **On**, then **Save Version → Save & Run All** (12 h limit).
Download `results.zip` from the version's Output tab.

**Your own reward:** add a class to `so_arm_warp/rewards.py` on your branch (see the docstring there), push it, set
`BRANCH` below and use its name in `EXPERIMENTS`.

**Physics (`case`):** `original_20it` (default) = the CPU env's model and solver with 20 iterations instead of 100,
~16.6k steps/s per T4, and policies work in the CPU env. `full_elliptic` = ~23k steps/s, also transfers (check it on
the CPU env). Don't use `full`: it's faster but its policies fail in the CPU env. See docs/TRAINING_OPTIONS.md.

**Send jobs from the command line instead:** `python tools/kaggle_job.py submit ...` (see docs/KAGGLE.md).

**Check a model locally (CPU, no GPU needed):** `python -m so_arm_warp.eval_sb3 results/models/<run>/final.zip`.
"""),
    code(f"""
BRANCH = "{a.branch}"
MODE = "variety"                  # "variety": one experiment per T4 | "focus": the first experiment, split over both T4s
HOURS = 11.0                      # stop each run after this long (Kaggle ends sessions at 12 h)
NUM_ENVS = 8192                   # parallel envs per experiment (in "focus" they're split across the T4s)
EXPERIMENTS = [                   # "variety": one per GPU; "focus": only the first one is trained
    dict(run_name="carl-default-1", reward="default", case="original_20it"),
    dict(run_name="carl-fast-1", reward="default", case="full_elliptic"),
]
RESUME = {{}}                      # e.g. {{"carl-default-1": "/kaggle/input/<prev-job>/results/models/carl-default-1/final.zip"}}
"""),
    code("""
%pip install -q "mujoco==4.0.1" "mujoco-warp==4.0.1" "stable-baselines3==2.7.0" "gymnasium==1.0.0" imageio
"""),
    code("""
import os, subprocess, sys, shutil, time, glob
SRC, OUT = "/kaggle/working/src", "/kaggle/working/results"
shutil.rmtree(SRC, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, "https://github.com/HumanED/so100.git", SRC], check=True)
print(subprocess.run(["git", "-C", SRC, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
import torch; n_gpu = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(n_gpu)])
assert n_gpu >= 1, "No GPU: set the accelerator to GPU T4 x2"
os.makedirs(OUT, exist_ok=True)
"""),
    md("## Train (variety: one process per GPU; focus: one process on all GPUs)"),
    code("""
assert MODE in ("variety", "focus"), MODE
focus = MODE == "focus" and n_gpu > 1
runs = EXPERIMENTS[:1] if MODE == "focus" else EXPERIMENTS[:n_gpu]
procs = []
for i, ex in enumerate(runs):
    cmd = [sys.executable, "-m", "so_arm_warp.train_sb3", "--run-name", ex["run_name"], "--reward", ex["reward"],
           "--physics", ex.get("case", "original_20it"), "--num-envs", str(NUM_ENVS), "--hours", str(HOURS),
           "--out-dir", OUT] + (["--gpus", str(n_gpu)] if focus else [])
    ck = RESUME.get(ex["run_name"])
    if ck and os.path.exists(ck):
        cmd += ["--resume", ck]
    elif ck:
        print("no checkpoint at", ck, "- starting", ex["run_name"], "from scratch")
    log = open(os.path.join(OUT, f"{ex['run_name']}.stdout"), "w")
    env = dict(os.environ) if focus else {**os.environ, "CUDA_VISIBLE_DEVICES": str(i)}
    procs.append((ex["run_name"], subprocess.Popen(cmd, cwd=SRC, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    print("started", ex["run_name"], f"on all {n_gpu} GPUs (focus)" if focus else f"on GPU {i}")
if len(EXPERIMENTS) > len(runs):
    print(f"{MODE}: not training", [e["run_name"] for e in EXPERIMENTS[len(runs):]])

def last_line(name):
    try:
        lines = [l for l in open(os.path.join(OUT, f"{name}.stdout")).read().splitlines() if l.startswith("{")]
        return lines[-1] if lines else "(starting)"
    except FileNotFoundError:
        return "(no output yet)"

while any(p.poll() is None for _, p, _ in procs):
    time.sleep(300)
    for name, p, _ in procs:
        print(f"{name}: {last_line(name)}", flush=True)
for name, p, log in procs:
    log.close()
    print(name, "exit code", p.returncode)
    if p.returncode:
        print(open(os.path.join(OUT, f"{name}.stdout")).read()[-3000:])
"""),
    md("## Training curves"),
    code("""
import pandas as pd, matplotlib.pyplot as plt
cols = ["ep_rew_mean", "grasp_rate", "success_rate", "steps_per_s"]
fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 3.4))
for ex in EXPERIMENTS:
    path = os.path.join(OUT, "logs", ex["run_name"], "log.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    for ax, c in zip(axes, cols):
        if c in df:
            ax.plot(df["steps"], df[c], label=ex["run_name"]); ax.set_title(c); ax.set_xlabel("env steps"); ax.grid(alpha=0.3)
axes[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "curves.png"), dpi=120); plt.show()
"""),
    md("## Evaluate in the real CPU env (SoFetchEnv)\n\nVideos: run `python -m so_arm_warp.eval_sb3 <model>.zip --video out.mp4` on your laptop. MuJoCo's offscreen renderer segfaults on Kaggle (EGL and OSMesa both tried)."),
    code("""
for ex in EXPERIMENTS:
    ck = os.path.join(OUT, "models", ex["run_name"], "final.zip")
    if not os.path.exists(ck):
        continue
    print(ex["run_name"])
    for extra in (["--episodes", "20"], ["--episodes", "20", "--sampled"]):
        r = subprocess.run([sys.executable, "-m", "so_arm_warp.eval_sb3", ck] + extra, cwd=SRC, capture_output=True, text=True)
        out = [l for l in r.stdout.splitlines() if "SoFetchEnv" in l or "Wrote" in l or "rendering" in l]
        for line in out:
            print(line)
        if r.returncode:
            print(f"(exit code {r.returncode})", r.stderr[-1500:])
"""),
    md("## Package outputs"),
    code("""
shutil.make_archive("/kaggle/working/results", "zip", OUT)
print("Wrote /kaggle/working/results.zip", f"({os.path.getsize('/kaggle/working/results.zip') / 1e6:.1f} MB)")
"""),
]
for i, c in enumerate(cells):
    c["id"] = f"cell-{i}"
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w"), indent=1)
print(f"Wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} KB)")
