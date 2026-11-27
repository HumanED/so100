# Training speed: options we tested

How fast can we train the SO-100 fetch task, and where? Everything below was measured in November 2026,
on an Apple M3 laptop (8 cores) and on free Kaggle notebooks (2× NVIDIA T4 GPUs, or a 4-core CPU).

**Bottom line:** train with **Stable-Baselines3 PPO on MuJoCo Warp, `original_20it` physics, on Kaggle's free T4
GPUs** (`python -m so_arm_warp.train_sb3`, or the Kaggle notebook / job tool in [KAGGLE.md](KAGGLE.md)). It trains at
**16,600 steps/s per T4, about 7× our original laptop setup**, two experiments per notebook. It learned the grasp in
~25 minutes, and the model grasps in **20/20 episodes in the real CPU `SoFetchEnv`**. The faster Warp options
(`full` physics, our own PyTorch PPO) produced policies that don't work in the real env, so speed alone is not the
measure.

**Baseline ("1.0×"):** the original setup, `train_parallel.py` with SB3 PPO, 8 `SubprocVecEnv` workers and
batch size 64 on the M3, at **2,400 training steps/s** (counting the PPO updates).

![Training speed of every option, coloured by whether its policies work in the real env](training_speed.png)

## Everything ranked

Training steps/s include the PPO updates. "Real env" means playing the trained model back in the CPU `SoFetchEnv`,
deterministic / sampled actions, 20 episodes. "Grasp" is the PPO8 +20 bonus: the jaw centre within 3 cm of the cube
with the jaws at least 4 cm open.

| # | Setup | Hardware | Training steps/s | vs baseline | Hours per 100M steps | Physics | Grasp in the real env | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | SB3 + Warp `full`, batch 16k, 5 epochs | T4 | 60,500 | 25× | 0.46 | Pyramidal friction, 4 ms steps | ❌ (same physics as #3) | Fast but its policies don't transfer |
| 2 | Our PyTorch PPO + Warp `full` | T4 | ~60,000 | 25× | 0.46 | Pyramidal friction, 4 ms steps | ❌ never learned (0.02% on Warp) | Dropped |
| 3 | SB3 + Warp `full`, batch 8k | T4 | 39,000 | 16× | 0.71 | Pyramidal friction, 4 ms steps | ❌ 0/20 / 6/20 | Dropped |
| 3b | **SB3 + Warp `original_20it`, one experiment on both T4s ("focus")** | **2 × T4** | **24,000** | **10×** | **1.2** | Same as #5 | ✅ **20/20 / 20/20** | **When one idea deserves both GPUs** |
| 4 | SB3 + Warp `full_elliptic` | T4 | 23,000 | 9.6× | 1.2 | Real friction model, 4 ms steps | ✅ 20/20 / 20/20 | For quick reward sweeps |
| 5 | **SB3 + Warp `original_20it`** | **T4** | **16,600** | **6.9×** | **1.7** | **Exact model and solver, 20 iterations instead of 100** | ✅ **20/20 / 20/20** | **Recommended default** |
| 6 | SB3 + Warp `original` | T4 | 9,100 | 3.8× | 3.1 | Exact (float32 on GPU) | Not trained (should transfer) | Too slow for what it adds |
| 7 | SB3 + `SoFetchVecEnv`, batch 512 | M3 CPU | 7,100 | 3.0× | 3.9 | Exact (within ~1e-8) | ✅ (it is the real env) | Fallback when out of GPU quota |
| 8 | SB3 + `SubprocVecEnv`, batch 512 | M3 CPU | 4,100 | 1.7× | 6.8 | Exact | ✅ | Batch-size change only |
| 9 | SB3 + `SoFetchVecEnv`, batch 64 | M3 CPU | 3,300 | 1.4× | 8.4 | Exact | ✅ | |
| 10 | MJX / Brax, simplified collisions | T4 | 3,050 | 1.3× | 9.1 | Simplified, 20 iterations | — | ❌ Slower than a laptop (branch `carlgpu`) |
| – | **Baseline:** SB3 + `SubprocVecEnv`, batch 64 | M3 CPU | **2,400** | 1.0× | 11.6 | Exact | ✅ | The original setup |
| 11 | MJX / Brax, original collision layout | T4 | ~1,000 | 0.4× | 28 | 10 iterations | — | ❌ Too slow; 10 iterations dropped grasped cubes |
| 12 | SB3 + Warp with main's batch size of 64 | T4 | 733 | 0.3× | 38 | — | — | ❌ 97% of the time goes to tiny gradient steps |
| 13 | SB3 with PyTorch on Apple Metal (MPS) | M3 GPU | 217–448 | 0.1–0.2× | 62+ | Exact | — | ❌ GPU round trips cost more than they save |
| – | Kaggle CPU notebook (`kaggle_train.ipynb`) | Kaggle CPU, 4 cores | Not measured | ~1–1.5× (est.) | — | Exact | ✅ | ⚠️ Tested locally only |
| – | Kaggle TPU v5e-8 | Kaggle TPU | — | — | — | — | — | ❌ Never ran (one TPU session per account) |

## How quickly each one learns (`default` reward, one T4)

| Setup | First reliable grasp | Steps by then | Grasp rate at the end of the run | Success (cube at goal) | Return at the end |
|---|---|---|---|---|---|
| SB3 + Warp `full` | ~17 min | ~29M | 99.8% (40 min) | 0% | +1.8 |
| SB3 + Warp `full_elliptic` | ~19 min | ~27M | 100% (50 min) | 0% | +0.7 to +1.0 |
| **SB3 + Warp `original_20it`** | **~25 min** | **~25M** | **100% (50 min)** | 0% | +1.0 to +1.6 |
| **Same, one experiment on both T4s (focus)** | **~22% sooner** (17.1 vs 21.9 min to a 90% grasp rate) | ~24M | 100% (40 min) | 0% | +5.4 (40 min) |
| Our PyTorch PPO + Warp `full` | Never, in 40 min | 147M seen | 0.02% | 0% | −20.7 |
| SB3 + `SoFetchVecEnv` (M3) | ~1 h *(estimate)* | ~25M | Not run | — | — |
| SB3 + `SubprocVecEnv` (M3) | ~3 h *(estimate)* | ~25M | Not run | — | — |

The CPU rows assume the same ~25M steps to learn, at their own speed; they weren't measured.
**No run has carried the cube to the goal yet.** That's now a reward and task question, not a speed or physics one.

## Do policies transfer to the exact physics?

| Trained on | On its own physics (det. / sampled) | Warp `original`, 1,024 episodes | CPU `SoFetchEnv`, 20 episodes |
|---|---|---|---|
| `full` | 100% / 100% | 0% / ~60% | 0/20 / 6/20 ❌ |
| `full_elliptic` | 100% / 84% | 100% / 85% | 20/20 / 20/20 ✅ |
| **`original_20it`** | 100% / 99.4% | **100% / 99.7%** | **20/20 / 20/20** ✅ |

**Finding the cause.** The `full`-trained model still grasps 100% on `full_2ms` (`full` with 2 ms timesteps), but
drops to 0% deterministic on `full_elliptic` (`full` with `original`'s elliptic friction cone) and on `original_20it`.
So the gap comes from `full`'s **pyramidal friction cone**, not the timestep or the collision hulls. The policy
learned to rely on how the pyramidal cone grips the cube. The SB3 adapter itself is consistent: Warp `original` and
the CPU env agree.

## Variety or focus: using both T4s

A Kaggle session has two T4s. **Variety** (the default) trains a different experiment on each. **Focus** splits one
experiment's envs over both (`train_sb3 --gpus 2`, notebook `MODE = "focus"`, `kaggle_job.py submit --focus`): each
step runs the physics on both GPUs at once and joins the results into one batch, and SB3's PPO update runs on the first
GPU. Measured on Kaggle's 2 × T4, `original_20it` physics:

| Setup | Envs | Batch | Training steps/s | vs one T4 | Env only (no PPO) |
|---|---|---|---|---|---|
| One T4 (variety, per experiment) | 8,192 | 8,192 | 17,760 | 1.0× | 24,041 |
| **Both T4s, same settings (focus default)** | **8,192 (4,096 each)** | **8,192** | **25,582** | **1.44×** | **41,336** |
| Both T4s, twice the envs | 16,384 (8,192 each) | 16,384 | 31,648 | 1.78× | 47,555 |
| Both T4s, twice the envs, smaller batch | 16,384 | 8,192 | 27,388 | 1.54× | |
| Both T4s, twice the envs, 5 epochs | 16,384 | 16,384 | 36,382 | 2.05× | Not tested for learning |

The PPO update isn't split, so the gain is less than 2×. **Learning (40-minute runs, `default` reward):**

| Run | Training steps/s | Grasp ≥ 50% | Grasp ≥ 90% | Return at 40 min | CPU `SoFetchEnv` grasps (det. / sampled) |
|---|---|---|---|---|---|
| One T4 (earlier run) | 16,600 | 20.0 min (19.5M steps) | 21.9 min (21.2M) | +1.4 | 20/20 / 20/20 |
| **Both T4s, same settings** | **24,000** | **16.4 min (23.3M)** | **17.1 min (24.4M)** | **+5.4** | **20/20 / 20/20** |
| Both T4s, twice the envs | 28,200 | 21.7 min (37.0M) | 26.0 min (44.0M) | +4.1 | 20/20 / 20/20 |

- **Same settings is the focus default.** It's the same PPO run, just 1.45× faster, so it reached the grasp ~22% sooner.
- **Twice the envs is faster per step but slower to learn.** Bigger batches mean fewer policy updates per step
  collected, so it needed ~44M steps instead of ~24M and reached the grasp later.
- **Variety still gets more total training:** 2 × 16.6k = 33k steps/s across two experiments, against 24k for one.
  Use focus when one result matters more than comparing two.
- One run each, so expect some variation. The one-T4 run shared the machine with a second run on the other T4.

## Physics speed only (no learning)

| Physics | Through the SB3 adapter, 8,192 envs, 1 T4 | Raw physics, 1 T4 | Raw physics, 2 T4s | Jaw drift vs exact (median) | Grasp & hold |
|---|---|---|---|---|---|
| `simplified` | — | 170k | — | 6.5 cm | ✅ |
| `filtered` | — | 156k | — | 6.2 cm | ✅ |
| `full` | 96k | 127k | 273k | 4.2 cm | ✅ (but policies don't transfer) |
| `full_2ms` | 51k | — | — | — | — |
| `full_elliptic` | 49k | — | — | — | — |
| `original_20it` | 24k | — | — | — | — |
| `original` | 10k | 43k | 95k | 0.85 cm (float32 only) | ✅ |
| `SoFetchVecEnv` (one process, MuJoCo's threaded `rollout`), M3 | — | 20,500 | — | 0 | ✅ |
| `SubprocVecEnv`, 8 processes, M3 | — | 8,900–16,300 | — | 0 | ✅ |
| CPU `SoFetchEnv`, one env (with cached observation lookups) | — | 3,500 (was 2,100) | — | 0 | ✅ |
| Thread-pool vec env (a Python thread per env) | — | 7,100–8,300 | — | 0 | ❌ No faster than processes |
| MJX on one T4: original layout / simplified collisions | — | 553 / 2,100 | — | — | ❌ / ⚠️ |
| MJX on the M3 CPU (JAX) / on Apple Metal (`jax-metal`) | — | 26–36 / – | — | — | ❌ |

The adapter column is lower than raw physics because it includes resets, the reward and observation, the copy back
to the CPU for SB3, and realistic contact states. It's the one that matters for training. Drift measures
sensitivity, not error: contact-heavy random motion is chaotic, so tiny differences grow
(`python -m so_arm_warp.fidelity`).

### Physics options (`--physics` / `--case`)

| Option | What it is |
|---|---|
| `original` | The untouched `scene.xml` exactly as `SoFetchEnv` runs it: elliptic friction cone, 100 solver iterations, 20 × 2 ms per step |
| **`original_20it`** | `original` with 20 solver iterations. **Default** |
| `full_elliptic` | Original collision shapes (hulls capped at 16 vertices), elliptic cone, 20 iterations, 10 × 4 ms |
| `full_2ms` | `full` with 20 × 2 ms steps (used to find the cause above) |
| `full` | Pyramidal cone, 20 iterations, 10 × 4 ms. Fast, but don't train on it |
| `filtered` / `simplified` | `full` with fewer collision checks |

Masses, joints, motors and the reward are identical in every option.

## SB3-on-Warp settings

| SB3 setup (1 T4, `full` physics) | Training steps/s | Notes |
|---|---|---|
| 1,024 envs, main's PPO settings (batch 64) | 733 | ❌ ~2,500 tiny gradient steps per update |
| 4,096 envs, batch 4,096, 10 epochs | 26k | |
| **8,192 envs, batch 8,192, 10 epochs** | **39k** | The default (`train_sb3.py`, `train_parallel.py`'s `"warp"` backend) |
| 8,192 envs, batch 16,384, 5 epochs | 61k | Untested for learning |

## What a Kaggle session gets you (12 h, 2 × T4)

| Setup | Steps per GPU per session | Experiments per session |
|---|---|---|
| `original_20it` | ~720M | 2 |
| `full_elliptic` | ~1.0B | 2 |
| `original_20it`, focus (both T4s on one experiment) | ~1.0B for that one | 1 |
| CPU `SoFetchVecEnv` on a laptop, 12 h overnight | ~300M | 1 |

## Caveats

- **One run each,** seed 0 / 1. Expect some variation between runs.
- **PPO settings are SB3's defaults** apart from the batch size. Nothing is tuned.
- **Hardware differs:** the CPU rows were measured on an M3, the Warp rows on one Kaggle T4.
- **Two reward problems from the old env are fixed on `main` (PPO7/PPO8).** The +20 grasp bonus used to be
  unreachable (cube centre within 4 mm of `jaw_site`, 1.4 cm from the fixed jaw's pads), and bonuses were paid only
  once per worker lifetime. Warp's `default` reward is `main`'s current reward (`tests/test_warp_reward.py`), and
  `legacy` keeps the old one.

## Recommendation

1. **Main workflow: SB3 + Warp `original_20it` on Kaggle T4 ×2.** Each person runs two reward experiments per
   notebook on their own free weekly GPU hours ([KAGGLE.md](KAGGLE.md)). When one looks promising, switch to **focus**
   to give it both T4s (1.45× faster).
2. **Quick reward sweeps:** `full_elliptic` (1.4× faster). Check the winners with
   `python -m so_arm_warp.eval_sb3 <model>.zip`, or retrain them on `original_20it`.
3. **No GPU quota left:** `train_parallel.py` with `SoFetchVecEnv` on a laptop. That's 3× the baseline, exact physics.
4. **Dropped:** `full` physics for training, our own PyTorch PPO (`so_arm_warp.train`, kept for reference), MJX/Brax
   (branch `carlgpu`), TPU, Apple Metal, and the thread-pool env.
