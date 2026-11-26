# Reinforcement learning on the SO-100 arm

See the HumanED notion page for installation instructions

## GPU training (MuJoCo MJX + Brax)

`so_arm_mjx/` is a JAX/MJX port of `SoFetchEnv` that runs thousands of environments in parallel on a GPU
and trains with Brax PPO. See the docstring in `so_arm_mjx/fetch_env.py` for how it differs from the CPU env.

- **Kaggle:** upload `kaggle_train_mjx.ipynb` (self-contained), set Accelerator to GPU and Internet to On, then Save & Run All.
  After changing `so_arm_mjx/`, `visualise_mjx.py` or the scene XML, rebuild it with `python make_kaggle_mjx_notebook.py`.
- **Local / other GPU machine:**
  `pip install "jax[cuda12]==0.9.2" "brax==0.14.2" "flax==0.12.6" "mujoco==3.3.5" "mujoco-mjx==3.3.5"`
  then `python -m so_arm_mjx.train --run-name PPO-carl-mjx-1`.
  These versions are tested together. brax 0.14.2 doesn't work with jax >= 0.10.
- **Watch a trained policy:** `python visualise_mjx.py mjx_out/models/<run>/final.pkl`
  (on macOS use `mjpython`), or add `--video out.mp4` / `--headless --episodes 10`.


## GPU training with MuJoCo Warp (fastest option)

`so_arm_warp/` runs the physics in MuJoCo Warp and PPO in PyTorch, all on an NVIDIA GPU. On a Kaggle T4,
Warp's physics measured 36k env steps/s with the exact original model and ~96k with all original collision
shapes plus a cheaper solver, against ~20k raw on an M3 with `SoFetchVecEnv`.

- **Kaggle:** upload `kaggle_train_warp.ipynb`, set GPU T4 x2 and Internet On, then Save & Run All. It trains two
  experiments at once, one per T4. Rebuild it with `python make_kaggle_warp_notebook.py --branch <your-branch>`.
- **Any NVIDIA machine:** `pip install mujoco==4.0.1 mujoco-warp==4.0.1 torch`, then
  `python -m so_arm_warp.train --run-name me-1 --reward default --case full --hours 2`.
- **Reward experiments:** add a class to `so_arm_warp/rewards.py` (see its docstring) and pass `--reward <name>`.
  `pad_midpoint` fixes the default reward's grasp check, which the cube can't reach without penetrating the jaw.
- **Physics choice (`--case`):** `full` (default), `original` (exact CPU-env physics, slower), `filtered`,
  `simplified`. `python -m so_arm_warp.fidelity` compares them with the original, and `python -m so_arm_warp.bench`
  measures speed.
- **Watch a policy:** `python -m so_arm_warp.play warp_out/<run>/model_final.pt [--physics original]` (CPU only).
