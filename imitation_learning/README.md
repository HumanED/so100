# Imitation learning on the SO-101

Setup notes for collecting demonstrations with a leader/follower pair of SO-101 arms and
training a policy on them with [LeRobot](https://github.com/huggingface/lerobot).

The SO-101 is the updated SO-100: same 6 joints and STS3215 motors on the follower, and
LeRobot drives both with the same code. The parts differ slightly, so the SO-100 sim model
used by the RL work in this repo is not an exact match for these arms, and datasets from
SO-100 and SO-101 arms should not be mixed.

Guides this follows:
- [SO-101 setup](https://huggingface.co/docs/lerobot/en/so101)
- [Imitation learning on real robots](https://huggingface.co/docs/lerobot/en/il_robots)

## Install

Tested on an Apple Silicon Mac with LeRobot 0.6.2, Python 3.12 and PyTorch 2.11 (MPS).

```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y ffmpeg -c conda-forge
cd ~/lerobot && pip install -e ".[core_scripts,feetech]"
```

Run `conda activate lerobot` in every new terminal.

## Ports

Each controller board has its own serial port, which stays the same whichever USB socket it
is plugged into. Find a board's port by running `lerobot-find-port` and unplugging that board
when asked.

Our boards:

| Arm      | Port                           |
| -------- | ------------------------------ |
| Follower | `/dev/tty.usbmodem58FA0828801` |
| Leader   | `/dev/tty.usbmodem5A7C1224381` |

These are specific to our boards on macOS; on Linux they show up as `/dev/ttyACM*`.

## Motor IDs (once per arm)

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58FA0828801
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5A7C1224381
```

Connect one motor at a time when asked, gripper first. IDs end up as:

| ID | Joint         |
| -- | ------------- |
| 1  | shoulder_pan  |
| 2  | shoulder_lift |
| 3  | elbow_flex    |
| 4  | wrist_flex    |
| 5  | wrist_roll    |
| 6  | gripper       |

Then daisy-chain them: board → 1 → 2 → 3 → 4 → 5 → 6. If only some IDs answer afterwards,
the chain is broken just before the first missing one; check that cable is fully seated.

## Calibration

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58FA0828801 --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5A7C1224381 --teleop.id=my_leader
```

Calibration is saved under `~/.cache/huggingface/lerobot/calibration/` by `id`, so use the same
ids in every later command.

Our calibrations for these two arms are in [`calibration/`](calibration/). To use the arms from
another computer without recalibrating, copy them into LeRobot's calibration folder:

```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration
cp -R imitation_learning/calibration/. ~/.cache/huggingface/lerobot/calibration/
```

They only fit these two arms. Calibrating also writes each joint's zero point into the motors,
so if anyone recalibrates, commit the new files here too, or everyone else's copies stop
matching and LeRobot asks them to recalibrate.

Things that went wrong for us:
- Pressing Enter before moving every joint end to end. LeRobot stops with
  `Some motors have the same min and max values`, or saves ranges too narrow to use. Watch
  the MIN/MAX table and only press Enter once every row has a wide range (most joints span
  more than 1000 steps).
- Wrist flex recorded an almost full-circle range (about 0–4095) on both arms, which means its
  reading wrapped past 0. Put the wrist straight at the first "middle position" step. If the
  follower's wrist jumps during teleop, recalibrate that arm.

## Teleoperation

```bash
lerobot-teleoperate \
    --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58FA0828801 --robot.id=my_follower \
    --robot.max_relative_target=10 \
    --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5A7C1224381 --teleop.id=my_leader
```

`max_relative_target` caps each step at 10° so a bad calibration can't make the follower jump;
drop it once everything tracks properly. Ctrl-C stops teleop and the follower goes limp, so
hold it first.

## Cameras

We need at least two:
1. A fixed wide view of the workspace (an iPhone via Continuity Camera works). It must not
   move between recordings and rollouts.
2. A wrist camera on the **follower**, since the follower is what the policy sees and moves.

`lerobot-find-cameras opencv` lists the cameras and saves a frame from each, to find their
indices.

## Recording, training, rollout

Log in with a write token first: `hf auth login`.

```bash
lerobot-record \
    --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58FA0828801 --robot.id=my_follower \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5A7C1224381 --teleop.id=my_leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_pick_place \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the cube and place it in the bin" \
    --dataset.episode_time_s=30 --dataset.reset_time_s=10
```

While recording: `→`/`n` ends the episode early, `←`/`r` re-records it, `Esc`/`q` stops.
Aim for 50+ episodes to start (about 10 per object position), with the cameras fixed and a
consistent grasp.

```bash
lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_pick_place \
    --policy.type=act --policy.device=mps \
    --output_dir=outputs/train/act_so101 --job_name=act_so101 \
    --policy.repo_id=${HF_USER}/act_so101
```

Training locally on a laptop is slow; the guide's Colab notebook or Hugging Face Jobs
(`--job.target=a10g-small`) are faster.

```bash
lerobot-rollout --strategy.type=base \
    --policy.path=${HF_USER}/act_so101 \
    --robot.type=so101_follower --robot.port=/dev/tty.usbmodem58FA0828801 --robot.id=my_follower \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --task="Pick up the cube and place it in the bin" --duration=60
```

Use the same camera names, resolutions and ids for recording and rollout.
