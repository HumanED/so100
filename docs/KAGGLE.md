# Training on Kaggle

Kaggle gives every account free GPU time (two NVIDIA T4s per session, roughly 30 hours a week; check
*Settings → Quotas*). We use it to run our recommended setup, **Stable-Baselines3 PPO on MuJoCo Warp** with `original_20it` physics
(16,600 training steps/s per T4, and its policies work in the real CPU env; see [TRAINING_OPTIONS.md](TRAINING_OPTIONS.md)).
One notebook session trains **up to two experiments at once**, one per GPU, for up to 12 hours.

There are two ways to start a job:

- **[A. Click through the notebook](#a-click-through-the-notebook)** on kaggle.com. No setup on your machine.
- **[B. Send jobs from the command line](#b-send-jobs-from-the-command-line)** with `tools/kaggle_job.py`. Best for starting several runs, resuming, and downloading results.

Either way, the notebook clones the code from **GitHub**. Whatever you want to run (for example a new reward
function) has to be **pushed to a branch** first.

## One-time setup

1. **Create a Kaggle account and verify your phone number** (*Settings → Phone verification*). Without it,
   notebooks can't use GPUs or the internet, and the notebook needs both.
2. **For option B only:** create an API token (*Settings → API → Create New Token*), then
   `pip install kaggle`. Credentials are covered under option B.

## A. Click through the notebook

1. Get [`kaggle_train_warp.ipynb`](../kaggle_train_warp.ipynb) from the repo: download it, or copy its GitHub URL.
2. On kaggle.com, go to **Create → New Notebook**, then **File → Import Notebook**, and upload the file or paste the URL.
3. In the right-hand sidebar under **Session options**:
   - **Accelerator: GPU T4 x2**
   - **Internet: On**
4. Edit the **settings cell** (the second cell):
   ```python
   BRANCH = "main"                   # the GitHub branch to run (yours, if you changed the reward)
   HOURS = 11.0                      # each run stops after this long (sessions end at 12 h)
   NUM_ENVS = 8192                   # parallel envs per GPU
   EXPERIMENTS = [                   # one per GPU
       dict(run_name="carl-default-1", reward="default", case="original_20it"),
       dict(run_name="carl-fast-1", reward="default", case="full_elliptic"),
   ]
   RESUME = {}
   ```
   - **`reward`:** a name from `so_arm_warp/rewards.py`: `default` (exactly `SoFetchEnv`'s reward), `legacy` (the
     pre-PPO7 reward), or your own.
   - **`case`:** the physics option.
     - `original_20it` (default): the CPU env's physics with 20 solver iterations instead of 100.
     - `full_elliptic`: 1.4× faster, also transfers; good for quick reward sweeps.
     - `original`: exact, slower.
     - **Not `full`:** it's faster, but its policies fail in the CPU env.
5. Click **Save Version → Save & Run All (Commit) → Save**. The job then runs in the background, and you can close the
   browser.
6. **Follow progress** from the notebook's page: open the version, then **Logs**. Every 5 minutes the notebook prints
   each run's latest steps/s, episode return, and grasp and success rates.
7. **Get the results** from the version's **Output** tab. Download `results.zip`, which contains for each run:
   - `models/<run>/<steps>.zip` checkpoints (every 25M steps) and `models/<run>/final.zip`
   - `vec_norm_stats/<run>/*.pkl`, the VecNormalize statistics each model needs
   - `logs/<run>/log.csv` metrics and tensorboard logs
   - the CPU-env evaluation (20 episodes, deterministic and sampled actions) in the notebook's log. For a video, run
     `python -m so_arm_warp.eval_sb3 <model>.zip --video out.mp4` on your laptop: MuJoCo's offscreen renderer crashes on Kaggle
   - `curves.png` comparing the runs

   Copy `models/<run>/` and `vec_norm_stats/<run>/` into the repo's `models/` and `vec_norm_stats/` folders to use
   `visualise_model.py` and `evaluate_model.py` as before.

**To resume a run:** in a new notebook (or a new version), click **Add Input → Your Work** and pick the earlier
notebook. Then set
`RESUME = {"carl-default-1": "/kaggle/input/<earlier-notebook>/results/models/carl-default-1/final.zip"}`.
The run name must match.

## B. Send jobs from the command line

`tools/kaggle_job.py` does option A for you. It fills in the settings cell, creates a private GPU notebook,
starts it, and lets you check on it and download the results.

### Credentials

Use one of these. **Never commit credentials.**
- **The `kaggle.json` file:** move it to `~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`.
- **Environment variables:** set `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` (or `KAGGLE_KEY` for older tokens).
- **A `.env` file:** put the same variables in a `.env` file and pass `--env-file path/to/.env`. Keep it outside the repo,
  or make sure it's git-ignored. The tool also accepts a token stored as `Kaggle_API` and never prints it.

### Commands

```bash
# Start a job: two experiments, one per T4 (physics original_20it unless given)
python tools/kaggle_job.py submit --name so100-carl-rewards \
    --exp carl-default-1:default --exp carl-fast-1:default:full_elliptic

# Other options
#   --exp RUN:REWARD:CASE     choose physics per run, e.g. carl-fast-1:default:full_elliptic
#   --branch my-branch        run your own branch (default: the branch you have checked out)
#   --hours 4                 shorter runs; --num-envs 4096 if a run is short of GPU memory
#   --dry-run                 write the job folder without sending it

python tools/kaggle_job.py status so100-carl-rewards       # queued / running / complete / error
python tools/kaggle_job.py logs so100-carl-rewards         # stream the live log (Ctrl-C to stop)
python tools/kaggle_job.py download so100-carl-rewards     # -> kaggle_results/so100-carl-rewards/results/...

# Continue both runs from that job's final checkpoints, in a new job
python tools/kaggle_job.py submit --name so100-carl-rewards-2 --resume-from so100-carl-rewards \
    --exp carl-default-1:default --exp carl-fast-1:default:full_elliptic
```

Some rules to know:
- **Reusing a `--name` creates a new version of that notebook.** Use a new name for each job you want to keep separate.
- **Resuming matches by run name:** each `--exp` resumes from the earlier job's `results/models/<run_name>/final.zip` if it
  exists, and starts fresh otherwise.

## Trying your own reward function

1. Make a branch: `git switch -c <you>-reward-x main`.
2. Add a class to [`so_arm_warp/rewards.py`](../so_arm_warp/rewards.py) with `@register("your_name")`. The file's
   docstring lists everything a reward can use: cube, jaw, pad and goal positions, the observation, and the step count.
3. Optionally check it runs on your laptop (CPU, slow but enough to catch errors):
   `python -m so_arm_warp.train_sb3 --run-name test --reward your_name --num-envs 16 --batch-size 64 --total-steps 3000 --device cpu`.
4. **Push the branch.** Then run it with option A (set `BRANCH`) or option B (`--branch <you>-reward-x --exp me-x:your_name`).

## Looking at results

```bash
pip install stable-baselines3==2.7.0 gymnasium==1.0.0      # plus the repo's usual mujoco; no GPU or Warp needed
python -m so_arm_warp.eval_sb3 kaggle_results/<job>/results/models/<run>/final.zip              # grasp / success in SoFetchEnv
python -m so_arm_warp.eval_sb3 <model>.zip --sampled --episodes 50
python -m so_arm_warp.eval_sb3 <model>.zip --video out.mp4 --episodes 1                          # works on macOS; crashes on Kaggle
# or copy models/<run>/ and vec_norm_stats/<run>/ into the repo and use visualise_model.py (viewer)
```

## Limits and troubleshooting

| What you see | Why / what to do |
|---|---|
| Status stays **QUEUED** | Kaggle is busy. GPU jobs usually start within minutes, but queues can be long at peak times, and TPUs can wait hours. Queue time doesn't use quota. |
| `Maximum batch ... session count ... reached` | Kaggle limits how many background sessions an account runs at once (only one for TPUs). Wait for one to finish, or stop one on its notebook page. |
| `The kernel source must be less than 1 megabytes` | The API only accepts notebooks under 1 MB. `kaggle_train_warp.ipynb` is 7 KB because it clones code from GitHub. The CPU notebook (`kaggle_train.ipynb`, 1.3 MB) has to be uploaded through the website. |
| Clone fails / branch not found | Push the branch to GitHub first. The repo must be readable without logging in. |
| CUDA out of memory | Lower `NUM_ENVS` (e.g. 4096). |
| No GPU / pip install fails | Check **Accelerator: GPU T4 x2** and **Internet: On**, and that your phone number is verified. |
| A run stops before `HOURS` | Kaggle ends sessions at 12 hours, so keep `HOURS` at 11 or less to leave time to save. Checkpoints are written every 25M steps and at the end. |

**No GPU quota left?** The CPU notebook [`kaggle_train.ipynb`](../kaggle_train.ipynb) (SB3 + `SoFetchVecEnv`, exact
physics) runs with **Accelerator: None**, which uses no quota. It's slower, and has to be uploaded through the website.
