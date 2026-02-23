# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Reinforcement learning (PPO via Stable Baselines3) to train a [SO-100 robotic arm](https://github.com/huggingface/lerobot/blob/main/examples/10_use_so100.md) in MuJoCo simulation to pick up a red cube and move it to a green target position. This is a learning project — code is intentionally iterative and experimental.

## Commands

```bash
# Install the environment package (run from repo root)
pip install -e so_arm_rl/

# Train a model (edit settings at top of file first)
python train_parallel.py

# Evaluate a trained model (edit model_folder at top of file first)
python evaluate_model.py

# Visualise a trained model with rendered window
python visualise_model.py

# Manual interactive control with GUI sliders
python tkiner_control.py
```

There are no test commands or linting setup in this project.

## Architecture

### Training flow

`train_parallel.py` → creates N parallel `SoFetchEnv` instances (one per CPU core) wrapped in `SubprocVecEnv` → wrapped in `VecMonitor` and `VecNormalize` (z-normalises observations and rewards) → trains a PPO `MlpPolicy` → saves model checkpoints to `models/<run_name>/<timesteps>.zip` and normalisation stats to `vec_norm_stats/<run_name>/<timesteps>.pkl`.

**Important**: both the `.zip` model file and the `.pkl` normalisation stats file are required together to evaluate or visualise a trained model.

### Run naming convention

Each run gets a unique name like `PPO-9-fetch-ethan`. The number must be incremented and the folder must not already exist before training — the script raises an exception otherwise. Set `start_from_existing = True` and point `old_model_file` at the checkpoint to continue training from a saved model.

### Core environment: `SoFetchEnv`

Defined in `so_arm_rl/envs/fetch/so_arm_fetch_env.py`. Implements the Gymnasium API.

**Observation space** (28-dimensional, indices matter for reward function):
- `[0:6]` — `robot_qpos`: joint angles (rad) for 6 motors in order: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw
- `[6:12]` — `robot_qvel`: joint velocities (rad/s)
- `[12:19]` — `object_qpos`: cube position (x,y,z) + quaternion (w,x,y,z)
- `[19:22]` — `jaw_pos`: jaw center position (x,y,z) from `jaw_site`
- `[22:25]` — `object_jaw_diff`: cube − jaw (dx,dy,dz)
- `[25:28]` — `object_target_diff`: cube − target (dx,dy,dz)

**Action space**: `MultiDiscrete([64]*6)` — each of the 6 motors gets an integer 0–63, rescaled to [−1, 1] before being sent to MuJoCo.

**Reward function** (`_compute_reward`):
```
reward = 0.5 * (−‖object_jaw_diff‖) + 0.5 * (−‖object_target_diff‖)
       + grasp_bonus  (once per episode: +20 when ‖object_jaw_diff‖ < 0.03 AND jaw_pos_rad >= 0.4)
       + success_bonus (once per episode: +30 when ‖object_target_diff‖ < 0.02)
```

**Episode**: max 100 timesteps (~8 seconds real time). Each `step()` runs 20 MuJoCo substeps (N_SUBSTEPS=20, dt=0.004s each → 0.08s per step).

### MuJoCo simulation files

- `so_arm_rl/envs/resources/fetch/scene.xml` — full scene: robot, red cube (`object:joint`), green target body (`target`), floor
- `so_arm_rl/envs/resources/fetch/so_arm100.xml` — robot kinematics, meshes, motor definitions and control ranges

### Utilities

- `so_arm_rl/envs/utils/mujoco_utils.py` — helpers to query/set joint positions, site positions, body positions
- `so_arm_rl/envs/utils/rotations.py` — Euler ↔ quaternion ↔ rotation matrix conversions
- `so_arm_rl/envs/utils/ema_util.py` — exponential moving average (optional action smoothing, `self.EMA`)

### TensorboardCallback

Logs per-rollout mean of every `info` key prefixed with `rew_` or `debug_`. Launch TensorBoard with:
```bash
tensorboard --logdir logs/
```

### Key settings to change between runs (top of `train_parallel.py`)

| Variable | Purpose |
|---|---|
| `this_run_name` | Must be unique; controls where models/logs are saved |
| `start_from_existing` / `old_model_file` | Continue training from a checkpoint |
| `DICE_MAX_LIMIT` | Total timesteps (set to `None` to run forever) |
| `saving_timesteps_interval` | How often to checkpoint |
