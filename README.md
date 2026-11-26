# Reinforcement learning on the SO-100 arm

The task (`SoFetchEnv`): the SO-100 arm reaches for a cube, grasps it and moves it to a target, simulated in
MuJoCo. See the HumanED notion page for installation instructions.

For imitation learning on the SO-101 with LeRobot, see [imitation_learning/](imitation_learning/README.md).

## How to train

| | Where | Speed | Docs |
|---|---|---|---|
| **Recommended** | **MuJoCo Warp + PyTorch PPO** on Kaggle's free T4 GPUs (`so_arm_warp/`) | ~27× the original setup per experiment, two experiments per notebook | [docs/KAGGLE.md](docs/KAGGLE.md) |
| Free fallback | Stable-Baselines3 on a laptop or a Kaggle CPU session (`train_parallel.py`, `SoFetchVecEnv`) | ~1.4–3× the original setup, exact physics, no quota | below |

The full comparison of everything we tested, including speeds, physics fidelity and what failed, is in
[docs/TRAINING_OPTIONS.md](docs/TRAINING_OPTIONS.md).

### GPU training with MuJoCo Warp (recommended)

- **On Kaggle:** either click through [`kaggle_train_warp.ipynb`](kaggle_train_warp.ipynb) or send jobs with
  `python tools/kaggle_job.py submit ...`. Both are step by step in [docs/KAGGLE.md](docs/KAGGLE.md).
- **On any NVIDIA machine:** `pip install mujoco==4.0.1 mujoco-warp==4.0.1 torch`, then
  `python -m so_arm_warp.train --run-name me-1 --reward default --case full --hours 2`.
- **Reward experiments:** add a class to [`so_arm_warp/rewards.py`](so_arm_warp/rewards.py) and pass `--reward <name>`.
  `default` is exactly `SoFetchEnv`'s current reward (checked by `tests/test_warp_reward.py`); `legacy` is the pre-PPO7 one.
- **Physics (`--case`):**
  - `full` (default): all original collision shapes, cheaper solver.
  - `original`: exactly the CPU env's physics, about 3× slower.
  - `filtered` / `simplified`: fewer collision checks, fastest.

  `python -m so_arm_warp.fidelity` compares them with the original physics, and `python -m so_arm_warp.bench` measures speed.
- **Watch a policy (CPU, no GPU needed):** `python -m so_arm_warp.play <run>/model_final.pt [--physics original] [--video out.mp4]`.

### CPU training with Stable-Baselines3 (fallback)

- **Laptop:** `python train_parallel.py` (set the run name at the top). It trains with `VecMonitor` + `VecNormalize`
  and saves `models/<run>/` and `vec_norm_stats/<run>/`. By default it runs 32 envs in one process with
  `SoFetchVecEnv`, which uses MuJoCo's multithreaded rollout and gives the same observations, rewards and infos as
  `SubprocVecEnv` (`tests/test_vec_env.py`). Set `vec_env_backend = "subproc"` for one process per env.
- **Kaggle CPU (no quota):** upload [`kaggle_train.ipynb`](kaggle_train.ipynb) through the website with
  **Accelerator: None**. Rebuild it after changing the env with `python make_kaggle_notebook.py`.
- **Watch a model:** `python visualise_model.py` (set the model at the top).
- **Tests:** `python tests/test_vec_env.py`, `python tests/test_warp_reward.py`, `python tests/test_warp_scene.py` (or `pytest tests/`).

## Repository layout

| Path | What it is |
|---|---|
| `so_arm_rl/` | The original CPU env (`SoFetchEnv`) and `SoFetchVecEnv`, which runs N copies in one process |
| `so_arm_warp/` | GPU training: `scene.py` (scene and collision options), `physics.py` (Warp batches), `env.py`, `rewards.py`, `ppo.py`, `train.py`, `play.py`, plus `bench.py` / `fidelity*.py` |
| `tools/kaggle_job.py` | Send, monitor and download Kaggle training jobs from the command line |
| `kaggle_train_warp.ipynb` / `make_kaggle_warp_notebook.py` | Kaggle GPU notebook (clones the code from GitHub) and its generator |
| `kaggle_train.ipynb` / `make_kaggle_notebook.py` | Kaggle CPU notebook (code embedded) and its generator |
| `train_parallel.py` / `visualise_model.py` | SB3 training and viewing on CPU |
| `tests/` | `SoFetchVecEnv` and the Warp scene/reward checked against `SoFetchEnv` |
| `docs/` | [Training options and measurements](docs/TRAINING_OPTIONS.md), [Kaggle guide](docs/KAGGLE.md) |
