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
"""

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mujoco
from matplotlib.widgets import Button, Slider

SCENE_PATH = Path(__file__).parent / "models" / "scene.xml"
CAMERA_NAME = "wrist_camera"
FPS = 30

# (decrease, increase) key for each actuator, in actuator order.
JOINT_KEYS = [("q", "w"), ("e", "r"), ("t", "y"), ("u", "i"), ("o", "p"), ("[", "]")]
SPEED_KEYS = "1234567890"
# Joint speed in rad/s at speed level 1; level n moves n times as fast (0 is level 10).
BASE_SPEED = 0.2
DEFAULT_SPEED_LEVEL = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=SCENE_PATH, help="MJCF scene to load")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
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

    # Free up keys that matplotlib binds by default (q quits, o zooms, p pans, r resets the view).
    used_keys = {key for pair in JOINT_KEYS for key in pair} | set(SPEED_KEYS)
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

    def on_key_press(event):
        nonlocal speed_level
        if event.key in SPEED_KEYS:
            speed_level = SPEED_KEYS.index(event.key) + 1
            show_speed()
        else:
            held_keys.add(event.key)

    def on_key_release(event):
        held_keys.discard(event.key)

    show_speed()
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    last_time = time.perf_counter()

    def update():
        nonlocal last_time
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
        mujoco.mj_step(model, data, nstep=max(1, round(sim_time / model.opt.timestep)))

        wrist_image.set_data(render(CAMERA_NAME))
        overview_image.set_data(render(overview))
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=1000 // FPS)
    timer.add_callback(update)
    timer.start()
    plt.show()
    renderer.close()


if __name__ == "__main__":
    main()
