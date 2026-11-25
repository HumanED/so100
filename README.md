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
