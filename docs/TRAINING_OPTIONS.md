# Training speed: options we tested

How fast can we train the SO-100 fetch task, and where? Everything below was measured in November 2026,
on an Apple M3 laptop (8 cores) and on free Kaggle notebooks (2× NVIDIA T4 GPUs, or a 4-core CPU).

**Bottom line:** train with **MuJoCo Warp on Kaggle's free T4 GPUs** (`so_arm_warp/`). It runs about
**27× faster per experiment** than our original laptop setup, and each Kaggle notebook trains two experiments at
once, one per GPU. It can use all of the original collision shapes, and every physics option passed the grasp test.
For running it, see [KAGGLE.md](KAGGLE.md).

**Measured before the PPO7/PPO8 changes on `main`:** 11 action bins, the older reward, and the scene without the
two jaw sites. The physics (masses, joints, collision shapes, solver) is unchanged, so all speeds and fidelity results
still apply. Everything on `warp-training` now trains `main`'s current task. `SoFetchVecEnv` and the Warp
`default` reward are checked against `SoFetchEnv` by the tests in `tests/`.

**Baseline ("1.0×"):** the original setup, `train_parallel.py` with SB3 PPO, 8 `SubprocVecEnv` workers and
batch size 64 on the M3, at **2,400 training steps/s** (counting the PPO updates).

![Training speed per experiment, relative to the original setup](training_speed.png)

## A. Training throughput

This is the number that decides how long training takes.

| # | Approach | Hardware | Physics | Training steps/s | vs baseline | Status |
|---|---|---|---|---|---|---|
| 1 | **MuJoCo Warp + PyTorch PPO** (`so_arm_warp`) | Kaggle T4 ×2 (free, ~30 GPU-hours/week each) | All original collision shapes, cheaper solver (`--case full`) | **65,000 per T4 = 130,000 per notebook** (2 experiments) | **27× per experiment** | ✅ Recommended. Learned to reach the cube in 15 min |
| 2 | MuJoCo Warp + PyTorch PPO, exact physics | Kaggle T4 ×2 | Exactly the CPU env's physics (`--case original`) | ~25–35k per T4 (estimated from 43k physics-only) | ~10–15× | ✅ Physics verified; training speed not yet measured |
| 3 | SB3 + `SoFetchVecEnv`, batch 512 | Any laptop | Exact original | 7,100 | 3.0× | ✅ Free fallback, no quota |
| 4 | SB3 + `SoFetchVecEnv`, batch 64 | Any laptop | Exact original | 3,300 | 1.4× | ✅ |
| 5 | SB3 + `SubprocVecEnv`, batch 512 | Any laptop | Exact original | 4,100 | 1.7× | ✅ Batch-size change only |
| – | **Baseline:** SB3 + `SubprocVecEnv`, batch 64 | M3 laptop | Exact original | **2,400** | 1.0× | Original setup |
| 6 | Kaggle CPU notebook (`kaggle_train.ipynb`) | Kaggle CPU, 4 cores (free, no quota) | Exact original | Not measured; likely about half a laptop | ~1–1.5× (est.) | ⚠️ Tested locally, not yet on Kaggle |
| 7 | MuJoCo MJX + Brax, simplified collisions | Kaggle T4 ×2 | Simplified, 20 solver iterations | 3,050 | 1.3× | ⚠️ Works but slower than a laptop. Dropped (on branch `carlgpu`) |
| 8 | MuJoCo MJX + Brax, original collision layout | Kaggle T4 ×2 | 10 solver iterations | ~1,000 | 0.4× | ❌ Too slow, and 10 iterations dropped grasped cubes |
| 9 | SB3 with PyTorch on Apple Metal (MPS) | M3 GPU | Exact original | 217–448 | 0.1–0.2× | ❌ Each step's round trip to the GPU costs more than the small network saves |

## B. Physics/env speed only, including dead ends

| Approach | Env steps/s | Status |
|---|---|---|
| **Warp, one T4:** simplified / filtered / full / exact original | **170k / 156k / 127k / 43k** | ✅ |
| **Warp, both T4s:** full / exact original | **273k / 95k** | ✅ Scales about 2.15× |
| `SoFetchVecEnv` (all envs in one process, MuJoCo's threaded `rollout`), M3 | 20,500 | ✅ |
| `SubprocVecEnv`, 8 processes, M3 | 8,900–16,300 | Baseline |
| Thread-pool vec env (a Python thread per env) | 7,100–8,300 | ❌ No faster than processes (Python's lock, waiting on the slowest env) |
| MJX on one T4: original layout / simplified collisions | 553 / 2,100 | ❌ / ⚠️ Collision detection was ~95% of the cost |
| MJX on the M3 CPU (JAX) | 26–36 | ❌ 100× slower than plain MuJoCo |
| MJX on Apple Metal (`jax-metal`) | – | ❌ Missing operations, then a compiler crash; the plugin is unmaintained |
| Kaggle TPU v5e-8 | – | ⏳ Never got through the queue. Moot now: Warp is NVIDIA-only and already wins |

## C. Physics options for Warp (`--case`)

Each option is compared with the exact original physics on CPU MuJoCo (`python -m so_arm_warp.fidelity`).

| `--case` | Speed (1 T4, physics) | Jaw drift in random motion (median) | Grasp & hold | Gripper pushes cube (original: 4.1 cm) | Arm-into-floor / cube-into-gripper leaks (original: 1.3% / 0.3%) |
|---|---|---|---|---|---|
| `original` (exact) | 43k | **0.85 cm** | ✅ Held | 4.0 cm | 1.2% / 0.3% |
| `full` (default) | 127k | 4.2 cm | ✅ Held | 5.8 cm | 1.2% / 0.0% |
| `filtered` | 156k | 6.2 cm | ✅ Held | 5.8 cm | 1.2% / 0.0% |
| `simplified` | 170k | 6.5 cm | ✅ Held | 5.6 cm | 1.6% / 0.0% |

- **`full` keeps every original collision shape** and only uses a cheaper contact solver: a pyramidal friction cone and 20 iterations instead of 100. `filtered` and `simplified` also drop collision checks that don't matter for the task. Masses, joints and motors are identical in every case.
- **The drift column measures sensitivity, not error.** Contact-heavy random motion is chaotic, so tiny differences grow. The tests that represent the task (grasp, push, leaks) all pass.

## D. SB3 PPO on Warp (`vec_env_backend = "warp"`)

`SoFetchWarpVecEnv` (`so_arm_warp/sb3_vec_env.py`) puts the Warp env behind SB3's VecEnv API. It uses the same 64-step actions, observation and reward as `SoFetchEnv`, and works inside `VecMonitor` and `VecNormalize`. Measured on one Kaggle T4 (`python -m so_arm_warp.sb3_bench`):

| SB3 setup | Training steps/s | Notes |
|---|---|---|
| 1,024 envs, main's PPO settings (batch 64) | 733 | ❌ 97% of the time goes to ~2,500 tiny gradient steps per update |
| 4,096 envs, batch 4,096, 10 epochs | 26k | |
| **8,192 envs, batch 8,192, 10 epochs** | **39k** | `train_parallel.py` default for `"warp"` |
| 8,192 envs, batch 16,384, 5 epochs | 61k | Same speed as our PyTorch PPO |
| 8,192 envs, batch 8,192, `original` physics | 9.1k | Exact physics: ~1.3× SB3 + `SoFetchVecEnv` |
| Adapter only, random actions (`full` / `original`) | 96k / 10k | |

**40-minute learning run, same T4, `full` physics, `default` reward:**

| Trainer | Steps in 40 min | Grasp rate at the end | Success rate | Episode return |
|---|---|---|---|---|
| SB3 PPO on Warp (8,192 envs, batch 8,192) | 75M | **99.8%** (from ~17 min) | 0% | +1.8 |
| Our PyTorch PPO on Warp (8,192 envs) | 147M | 0.02% | 0% | −20.7 |

- **SB3 learned the grasp bonus and our PPO didn't,** despite our PPO taking twice as many steps. The likely reasons are SB3's discrete actions and reward normalisation. "Grasp" here means the PPO8 bonus: the jaw centre within 3 cm of the cube with the jaws at least 4 cm open. Neither run moved the cube to the goal.
- **⚠️ Policies trained on `full` physics don't transfer to exact physics.** The same SB3 model, played back for 20 episodes:

  | Played on | Grasp (deterministic) | Grasp (sampled) |
  |---|---|---|
  | Warp `full` | 20/20 | 20/20 |
  | Warp `original` | 0/20 | 12/20 |
  | CPU `SoFetchEnv` | 0/20 | 6/20 |

  The adapter is consistent: Warp `original` and the CPU env agree.

**Finding the cause.** The `full`-trained model still grasps 100% on `full_2ms` (`full` with 2 ms timesteps), but drops to 0% deterministic on `full_elliptic` (`full` with `original`'s elliptic friction cone) and on `original_20it`. So the gap comes from `full`'s **pyramidal friction cone**, not the timestep or the collision hulls.

**Hybrid physics, 50-minute SB3 run on each, one T4:**

| Training physics | SB3 steps/s | Adapter only (8,192 envs) | Grasping by | Warp `original` (1,024 ep., det. / sampled) | CPU `SoFetchEnv` (20 ep., det. / sampled) |
|---|---|---|---|---|---|
| `full` | 39k | 96k | ~17 min | 0% / ~60% | 0/20 / 6/20 ❌ |
| `full_elliptic` | 23k | 49k | ~19 min | 100% / 85% | 20/20 / 20/20 ✅ |
| **`original_20it`** | **16.6k** | 24k | ~25 min | **100% / 99.7%** | **20/20 / 20/20** ✅ |
| `original` | 9.1k | 10k | — | (exact) | — |

- **`original_20it` is the default** for `vec_env_backend = "warp"`. It's `SoFetchEnv`'s exact model and solver, apart from 20 iterations instead of 100. That makes it 1.8× faster than `original` and about 2.3× SB3 + `SoFetchVecEnv`, and its policies transfer almost perfectly.
- **`full_elliptic`** is 1.4× faster again and also transfers. Its sampled-action grasp rate on `original` is lower (85%), so check models on `original`.
- **Don't train on `full`.** Our own PyTorch PPO (`so_arm_warp.train`) still defaults to `--case full`, so pass `--case full_elliptic` or `original_20it`.
- In every run so far the cube never reached the goal (0% success). That's now a reward and task question, not a physics one.

## Caveats

- **Only speed is compared.** Warp uses continuous actions and our own PyTorch PPO, while SB3 uses 64 discrete action steps per motor. We haven't yet compared how many steps each needs to actually solve the task.
- **PPO settings are untuned,** and the longest Warp training run so far is 15 minutes.
- **Two reward problems from the old env are fixed on `main` (PPO7/PPO8).** The +20 grasp bonus used to be unreachable: it needed the cube centre within 4 mm of `jaw_site`, which sits 1.4 cm from the fixed jaw's pads. Bonuses were also paid only once per worker lifetime. The current reward uses a 3 cm, jaws-open grasp check and re-arms the bonuses every episode. Warp's `default` reward is that same reward, and `legacy` keeps the old one for comparison.

## Recommendation

1. **Main workflow: Warp on Kaggle T4 ×2 with `--case full`.** Each person runs two reward experiments per notebook, using their own free weekly GPU hours.
2. **Confirm the best policies on the exact physics:** train or evaluate with `--case original`, or play them back with `python -m so_arm_warp.play <checkpoint> --physics original`.
3. **Free fallback:** `SoFetchVecEnv` (`train_parallel.py`) on a laptop or Kaggle CPU. That's 3× the baseline, exact physics and no quota.
4. **Dropped:** MJX/Brax (kept on branch `carlgpu`), TPU, Apple Metal, and the thread-pool env.
