"""
Builds kaggle_train_warp.ipynb: Kaggle GPU notebook that trains the MuJoCo Warp version (so_arm_warp).
It git-clones the code (so it stays under Kaggle's 1 MB API limit and always runs the pushed branch):

    python make_kaggle_warp_notebook.py [--branch warp-training]
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--branch", default="warp-training")
a = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "kaggle_train_warp.ipynb")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


cells = [
    md(f"""
# SO-100 Fetch: GPU training with MuJoCo Warp + PyTorch PPO

Physics runs in MuJoCo Warp and PPO in PyTorch, all on the GPU (no per-step CPU round trips).
A Kaggle GPU session has **two T4s** and each training run uses one, so this notebook trains up to **two experiments
side by side** (e.g. two reward functions). Code is cloned from the `{a.branch}` branch.

**Kaggle settings:** Accelerator **GPU T4 x2**, Internet **On**, then **Save Version → Save & Run All** (12 h limit).
Download `results.zip` from the version's Output tab.

**Your own reward:** add a class to `so_arm_warp/rewards.py` on your branch (see the docstring there), push it, set
`BRANCH` below and use its name in `EXPERIMENTS`.

**Physics (`case`):** `full` = all original collision shapes with a cheaper solver (default), `original` = exactly the
CPU env's physics (slower), `filtered` / `simplified` = cheaper collisions (fastest). See `so_arm_warp/physics.py`.

**Send jobs from the command line instead:** `python tools/kaggle_job.py submit ...` (see docs/KAGGLE.md).

**Watch a trained policy locally:** `python -m so_arm_warp.play results/<run>/model_final.pt` (CPU, no GPU needed).
"""),
    code(f"""
BRANCH = "{a.branch}"
HOURS = 11.0                      # stop each run after this long (Kaggle ends sessions at 12 h)
NUM_ENVS = 8192                   # parallel envs per GPU
EXPERIMENTS = [                   # one per GPU; remove one to train a single run
    dict(run_name="carl-default-1", reward="default", case="full"),
    dict(run_name="carl-padmid-1", reward="pad_midpoint", case="full"),
]
RESUME = {{}}                      # e.g. {{"carl-default-1": "/kaggle/input/<prev-job>/results/carl-default-1/model_final.pt"}}
"""),
    code("""
%pip install -q "mujoco==4.0.1" "mujoco-warp==4.0.1"
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
    md("## Train (one process per GPU)"),
    code("""
procs = []
for i, ex in enumerate(EXPERIMENTS[:n_gpu]):
    cmd = [sys.executable, "-m", "so_arm_warp.train", "--run-name", ex["run_name"], "--reward", ex["reward"],
           "--case", ex.get("case", "full"), "--num-envs", str(NUM_ENVS), "--hours", str(HOURS),
           "--out-dir", OUT, "--device", f"cuda:{i}"]
    ck = RESUME.get(ex["run_name"])
    if ck and os.path.exists(ck):
        cmd += ["--resume", ck]
    elif ck:
        print("no checkpoint at", ck, "- starting", ex["run_name"], "from scratch")
    log = open(os.path.join(OUT, f"{ex['run_name']}.stdout"), "w")
    procs.append((ex["run_name"], subprocess.Popen(cmd, cwd=SRC, stdout=log, stderr=subprocess.STDOUT), log))
    print("started", ex["run_name"], "on", f"cuda:{i}")
if len(EXPERIMENTS) > n_gpu:
    print("Only", n_gpu, "GPU(s): skipped", [e["run_name"] for e in EXPERIMENTS[n_gpu:]])

def last_line(name):
    try:
        lines = [l for l in open(os.path.join(OUT, f"{name}.stdout")).read().splitlines() if l.startswith("[")]
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
cols = ["ep_return", "grasp_rate", "success_rate", "steps_per_s"]
fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 3.4))
for ex in EXPERIMENTS:
    path = os.path.join(OUT, ex["run_name"], "log.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    for ax, c in zip(axes, cols):
        if c in df:
            ax.plot(df["step"], df[c], label=ex["run_name"]); ax.set_title(c); ax.set_xlabel("env steps"); ax.grid(alpha=0.3)
axes[0].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "curves.png"), dpi=120); plt.show()
"""),
    md("## Evaluate on the exact original physics (CPU MuJoCo) + video"),
    code("""
os.environ.setdefault("MUJOCO_GL", "egl")
for ex in EXPERIMENTS:
    ck = os.path.join(OUT, ex["run_name"], "model_final.pt")
    if not os.path.exists(ck):
        continue
    for extra in (["--physics", "original", "--episodes", "3", "--headless"], ["--episodes", "1", "--video", os.path.join(OUT, f"{ex['run_name']}.mp4")]):
        r = subprocess.run([sys.executable, "-m", "so_arm_warp.play", ck] + extra, cwd=SRC, capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        print(r.stdout[-1500:] or r.stderr[-800:])
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
