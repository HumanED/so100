"""Show the sim arm's wrist camera view and drive its actuators with sliders.

Usage (from the repo root):
    python imitation_learning/wrist_camera_teleop.py

Each slider sets the target of one position actuator; the physics runs in real time, so the
arm moves towards the targets and can push the cube around.

Keyboard (hold a key to keep moving):
    q / w   Rotation         o / p   Wrist_Roll
    e / r   Pitch            [ / ]   Jaw
    t / y   Elbow
    u / i   Wrist_Pitch
    1-9, 0  Speed (1 = slowest, 0 = fastest)
    space   Start recording an episode / stop and save it
    backspace  Discard the episode being recorded

Each episode is saved to <out>/episode_NNN/ as wrist.mp4 and front.mp4 (the two camera views)
plus data.npz, sampled at --record-fps in simulation time:
    timestamp           (T,)    seconds since the episode started
    observation.state   (T, 6)  joint positions, in actuator order
    action              (T, 6)  actuator targets (what the keys/sliders set)
    object_pose         (T, 7)  cube position and quaternion
    joint_names         (6,)
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.widgets import Button, Slider

SCENE_PATH = Path(__file__).parent / "models" / "scene.xml"
EPISODES_DIR = Path(__file__).parent / "episodes"
CAMERA_NAME = "wrist_camera"
FPS = 30

# (decrease, increase) key for each actuator, in actuator order.
JOINT_KEYS = [("q", "w"), ("e", "r"), ("t", "y"), ("u", "i"), ("o", "p"), ("[", "]")]
SPEED_KEYS = "1234567890"
# Joint speed in rad/s at speed level 1; level n moves n times as fast (0 is level 10).
BASE_SPEED = 0.2
DEFAULT_SPEED_LEVEL = 5
RECORD_KEYS = (" ", "space")
DISCARD_KEY = "backspace"


class EpisodeRecorder:
    """Writes one episode: a video per camera, streamed to disk, plus the per-frame arrays."""

    def __init__(self, episode_dir, model, fps, render):
        self.episode_dir = episode_dir
        self.model = model
        self.render = render
        episode_dir.mkdir(parents=True)
        self.videos = {
            name: imageio.get_writer(episode_dir / f"{name}.mp4", fps=fps, macro_block_size=1)
            for name in ("wrist", "front")
        }
        self.joint_qpos = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
        self.object_qpos = model.jnt_qposadr[model.joint("object:joint").id]
        self.start_time = None
        self.frames = {"timestamp": [], "observation.state": [], "action": [], "object_pose": []}

    def capture(self, data):
        if self.start_time is None:
            self.start_time = data.time
        wrist, front = self.render()
        self.videos["wrist"].append_data(wrist)
        self.videos["front"].append_data(front)
        self.frames["timestamp"].append(data.time - self.start_time)
        self.frames["observation.state"].append(data.qpos[self.joint_qpos].copy())
        self.frames["action"].append(data.ctrl.copy())
        self.frames["object_pose"].append(data.qpos[self.object_qpos : self.object_qpos + 7].copy())

    @property
    def duration(self):
        timestamps = self.frames["timestamp"]
        return timestamps[-1] if timestamps else 0.0

    def save(self):
        for video in self.videos.values():
            video.close()
        arrays: dict[str, Any] = {key: np.array(values) for key, values in self.frames.items()}
        arrays["joint_names"] = np.array([self.model.actuator(i).name for i in range(self.model.nu)])
        np.savez(self.episode_dir / "data.npz", **arrays)
        return len(self.frames["timestamp"])

    def discard(self):
        for video in self.videos.values():
            video.close()
        shutil.rmtree(self.episode_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=SCENE_PATH, help="MJCF scene to load")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--out", type=Path, default=EPISODES_DIR, help="folder to save episodes in")
    parser.add_argument("--record-fps", type=int, default=30, help="frames per second of simulation time")
    parser.add_argument("--record-width", type=int, default=640, help="width of the recorded videos")
    parser.add_argument("--record-height", type=int, default=480, help="height of the recorded videos")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, args.height, args.width)
    # Hide the debug sites (jaw/object centres) so they don't clutter the camera image.
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = 0
    overview = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, overview)

    def render(camera):
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        return renderer.render()

    # Recorded frames get their own renderer so they can be smaller than the on-screen view.
    record_renderer = mujoco.Renderer(model, args.record_height, args.record_width)

    def render_for_recording():
        frames = []
        for camera in (CAMERA_NAME, overview):
            record_renderer.update_scene(data, camera=camera, scene_option=scene_option)
            frames.append(record_renderer.render())
        return frames

    # Free up keys that matplotlib binds by default (q quits, o zooms, p pans, r resets the view,
    # backspace goes back).
    used_keys = {key for pair in JOINT_KEYS for key in pair} | set(SPEED_KEYS) | set(RECORD_KEYS) | {DISCARD_KEY}
    for name, keys in mpl.rcParams.items():
        if name.startswith("keymap."):
            mpl.rcParams[name] = [key for key in keys if key not in used_keys]

    fig = plt.figure("SO-ARM100 wrist camera", figsize=(18, 10))
    image_bottom = 0.04 + 0.035 * model.nu
    wrist_ax = fig.add_axes((0.01, image_bottom, 0.485, 0.96 - image_bottom))
    overview_ax = fig.add_axes((0.505, image_bottom, 0.485, 0.96 - image_bottom))
    for ax, title in ((wrist_ax, "Wrist camera"), (overview_ax, "Overview (scene camera)")):
        ax.set_title(title)
        ax.axis("off")
    wrist_image = wrist_ax.imshow(render(CAMERA_NAME))
    overview_image = overview_ax.imshow(render(overview))

    # One slider per actuator, limited to the range of the joint it drives.
    sliders = []
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        low, high = model.jnt_range[joint_id]
        start = float(min(max(data.qpos[model.jnt_qposadr[joint_id]], low), high))
        slider_ax = fig.add_axes((0.15, 0.02 + 0.035 * (model.nu - 1 - i), 0.65, 0.025))
        sliders.append(Slider(slider_ax, model.actuator(i).name, low, high, valinit=start))

    def reset(_event):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for slider in sliders:
            slider.reset()

    reset_button = Button(fig.add_axes((0.85, 0.03, 0.1, 0.05)), "Reset")
    reset_button.on_clicked(reset)

    # Keys currently held down; the timer moves their joints each tick.
    held_keys = set()
    speed_level = DEFAULT_SPEED_LEVEL
    speed_text = fig.text(0.85, 0.1, "")

    def show_speed():
        speed_text.set_text(f"Speed {speed_level} ({BASE_SPEED * speed_level:.1f} rad/s)")

    recorder = None
    next_frame_time = 0.0
    record_text = fig.text(0.85, 0.14, "", fontweight="bold")

    def next_episode_dir():
        index = 0
        while (args.out / f"episode_{index:03d}").exists():
            index += 1
        return args.out / f"episode_{index:03d}"

    def show_recording(message=None):
        if recorder is not None:
            record_text.set_text(f"● REC {recorder.episode_dir.name}  {recorder.duration:.1f} s")
            record_text.set_color("red")
        else:
            record_text.set_text(message or "space: record")
            record_text.set_color("black")

    def toggle_recording():
        nonlocal recorder, next_frame_time
        if recorder is None:
            recorder = EpisodeRecorder(next_episode_dir(), model, args.record_fps, render_for_recording)
            next_frame_time = data.time
            show_recording()
        else:
            frame_count = recorder.save()
            print(f"Saved {recorder.episode_dir} ({frame_count} frames, {recorder.duration:.1f} s)")
            show_recording(f"Saved {recorder.episode_dir.name}")
            recorder = None

    def discard_recording():
        nonlocal recorder
        if recorder is not None:
            recorder.discard()
            print(f"Discarded {recorder.episode_dir}")
            show_recording(f"Discarded {recorder.episode_dir.name}")
            recorder = None

    def on_key_press(event):
        nonlocal speed_level
        if event.key in RECORD_KEYS:
            toggle_recording()
        elif event.key == DISCARD_KEY:
            discard_recording()
        elif event.key in SPEED_KEYS:
            speed_level = SPEED_KEYS.index(event.key) + 1
            show_speed()
        else:
            held_keys.add(event.key)

    def on_key_release(event):
        held_keys.discard(event.key)

    show_speed()
    show_recording()
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    last_time = time.perf_counter()

    def update():
        nonlocal last_time, next_frame_time
        now = time.perf_counter()
        # Catch the simulation up with wall-clock time, capped so a stall doesn't freeze the GUI.
        # A tick takes ~150 ms at 1280x960 (two renders plus the matplotlib redraw), so the cap
        # has to sit above that or the arm and the keys run slower than real time.
        sim_time = min(now - last_time, 0.25)
        last_time = now

        for slider, (decrease_key, increase_key) in zip(sliders, JOINT_KEYS):
            direction = (increase_key in held_keys) - (decrease_key in held_keys)
            if direction:
                step = direction * BASE_SPEED * speed_level * sim_time
                slider.set_val(min(max(slider.val + step, slider.valmin), slider.valmax))

        data.ctrl[:] = [slider.val for slider in sliders]
        nstep = max(1, round(sim_time / model.opt.timestep))
        if recorder is None:
            mujoco.mj_step(model, data, nstep=nstep)
        else:
            # Step one physics step at a time so frames land on a fixed grid of simulation time,
            # however long the GUI took to draw.
            for _ in range(nstep):
                if data.time >= next_frame_time:
                    recorder.capture(data)
                    next_frame_time += 1 / args.record_fps
                mujoco.mj_step(model, data)
            show_recording()

        wrist_image.set_data(render(CAMERA_NAME))
        overview_image.set_data(render(overview))
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=1000 // FPS)
    timer.add_callback(update)
    timer.start()
    plt.show()
    # Closing the window mid-episode keeps what was recorded rather than losing it.
    if recorder is not None:
        toggle_recording()
    renderer.close()
    record_renderer.close()


if __name__ == "__main__":
    main()
