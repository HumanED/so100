"""
Send training jobs to Kaggle from the command line, check on them and download the results.
Wraps the official Kaggle CLI (`pip install kaggle`). See docs/KAGGLE.md for setup.

    python tools/kaggle_job.py submit --name so100-carl-rewards --exp carl-default:default --exp carl-fast:default:full_elliptic
    python tools/kaggle_job.py status so100-carl-rewards
    python tools/kaggle_job.py logs so100-carl-rewards            # streams the live log (Ctrl-C to stop)
    python tools/kaggle_job.py download so100-carl-rewards        # -> kaggle_results/so100-carl-rewards/
    python tools/kaggle_job.py submit --name so100-carl-rewards --resume-from so100-carl-rewards ...   # continue training

Each job is a private Kaggle notebook running kaggle_train_warp.ipynb (SB3 PPO on MuJoCo Warp) on GPU T4 x2: up to
two experiments, one per GPU. The code is cloned from --branch on GitHub, so push your branch (e.g. with a new reward) first.

Credentials: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_API_TOKEN (or KAGGLE_KEY) in the environment,
or --env-file path/to/.env containing them. Keys are never printed.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO, "kaggle_train_warp.ipynb")


def load_env(path):
    """Read KEY=VALUE lines into os.environ without printing them. Maps a few common token variable names."""
    if path:
        for line in open(path):
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
            if m:
                os.environ.setdefault(m.group(1), m.group(2).strip().strip("'\""))
    if "KAGGLE_API_TOKEN" not in os.environ and "KAGGLE_KEY" not in os.environ:
        for k in list(os.environ):
            if k.lower() in ("kaggle_api", "kaggle_token") and os.environ[k]:
                os.environ["KAGGLE_API_TOKEN"] = os.environ[k]


def kaggle(*args, check=True, capture=False):
    exe = shutil.which("kaggle")
    if not exe:
        sys.exit("Kaggle CLI not found: pip install kaggle")
    return subprocess.run([exe, *args], check=check, text=True, capture_output=capture)


def username(arg):
    user = arg or os.environ.get("KAGGLE_USERNAME")
    if not user:
        cfg = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(cfg):
            user = json.load(open(cfg)).get("username")
    if not user:
        sys.exit("Kaggle username unknown: pass --user or set KAGGLE_USERNAME")
    return user


def current_branch():
    try:
        return subprocess.run(["git", "-C", REPO, "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def config_cell(branch, hours, num_envs, experiments, resume):
    exps = ",\n".join(f'    dict(run_name="{r}", reward="{w}", case="{c}")' for r, w, c in experiments)
    res = ", ".join(f'"{k}": "{v}"' for k, v in resume.items())
    return (f'BRANCH = "{branch}"\nHOURS = {hours}                      # stop each run after this long (Kaggle ends sessions at 12 h)\n'
            f'NUM_ENVS = {num_envs}                   # parallel envs per GPU\n'
            f'EXPERIMENTS = [                   # one per GPU\n{exps},\n]\n'
            f'RESUME = {{{res}}}                      # run_name -> checkpoint (used if the file exists)')


def submit(a):
    user = username(a.user)
    slug = a.name.lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,49}", slug):
        sys.exit("--name: 5-50 characters, lowercase letters, digits and dashes")
    experiments = []
    for e in a.exp:
        parts = e.split(":")
        if len(parts) not in (2, 3):
            sys.exit(f"--exp {e!r}: use RUN_NAME:REWARD or RUN_NAME:REWARD:CASE")
        experiments.append((parts[0], parts[1], parts[2] if len(parts) == 3 else a.case))
    if not 1 <= len(experiments) <= 2:
        sys.exit("give one or two --exp (one per T4)")
    resume, sources = {}, []
    if a.resume_from:
        src = a.resume_from if "/" in a.resume_from else f"{user}/{a.resume_from}"
        sources.append(src)
        mount = src.split("/")[1]
        resume = {r: f"/kaggle/input/{mount}/results/models/{r}/final.zip" for r, _, _ in experiments}

    nb = json.load(open(NOTEBOOK))
    idx = next(i for i, c in enumerate(nb["cells"]) if "EXPERIMENTS = [" in "".join(c["source"]))
    nb["cells"][idx]["source"] = config_cell(a.branch, a.hours, a.num_envs, experiments, resume)
    out = a.dir or tempfile.mkdtemp(prefix="kaggle_job_")
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, f"{slug}.ipynb"), "w"), indent=1)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": sources, "model_sources": []}
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    print(f"Job '{user}/{slug}': branch {a.branch}, {a.hours} h, {a.num_envs} envs/GPU")
    for r, w, c in experiments:
        print(f"  {r}: reward={w} case={c}" + (f" (resume from {resume[r]})" if r in resume else ""))
    if a.dry_run:
        print("Dry run: job folder written to", out)
        return
    kaggle("kernels", "push", "-p", out, "--accelerator", a.accelerator)
    print(f"Watch: https://www.kaggle.com/code/{user}/{slug}  |  python tools/kaggle_job.py status {slug}")


def full_id(name, user_arg):
    return name if "/" in name else f"{username(user_arg)}/{name}"


def status(a):
    kaggle("kernels", "status", full_id(a.name, a.user))


def logs(a):
    try:
        kaggle("kernels", "logs", "-f", full_id(a.name, a.user), check=False)
    except KeyboardInterrupt:
        pass


def download(a):
    kid = full_id(a.name, a.user)
    out = a.out or os.path.join(REPO, "kaggle_results", kid.split("/")[1])
    os.makedirs(out, exist_ok=True)
    kaggle("kernels", "output", kid, "-p", out)
    print("Downloaded to", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-file", default=None, help=".env with KAGGLE_USERNAME / KAGGLE_API_TOKEN")
    p.add_argument("--user", default=None, help="Kaggle username (default: KAGGLE_USERNAME or kaggle.json)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="send a training job")
    s.add_argument("--name", required=True, help="job (notebook) name; reusing a name adds a new version")
    s.add_argument("--exp", action="append", required=True, help="RUN_NAME:REWARD[:CASE], once or twice")
    s.add_argument("--case", default="original_20it", help="default physics case for --exp without one")
    s.add_argument("--branch", default=current_branch() or "main", help="GitHub branch to clone")
    s.add_argument("--hours", type=float, default=11.0)
    s.add_argument("--num-envs", type=int, default=8192)
    s.add_argument("--resume-from", default=None, help="earlier job name whose results/models/<run>/final.zip to continue")
    s.add_argument("--accelerator", default="NvidiaTeslaT4")
    s.add_argument("--dir", default=None, help="where to write the job folder (default: a temp dir)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=submit)
    for name, fn, hlp in (("status", status, "job status"), ("logs", logs, "stream the live log"),
                          ("download", download, "download outputs")):
        c = sub.add_parser(name, help=hlp)
        c.add_argument("name")
        if name == "download":
            c.add_argument("--out", default=None)
        c.set_defaults(fn=fn)
    a = p.parse_args()
    load_env(a.env_file)
    a.fn(a)


if __name__ == "__main__":
    main()
