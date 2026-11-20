"""Scripted IK expert: place the red cube at a random reachable pose, grasp it, lift it.

Sim only. Drives MjModel/MjData directly; the per-step (q_arm, jaw) targets in
`plan_pick` are the stream to record later for imitation learning.
"""
import argparse
import os
import time

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

SCENE_PATH = os.path.join(os.path.dirname(__file__), "imitation_learning", "models", "scene.xml")
# Start pose (pan is set per episode to face the cube): the wrist camera, aimed at the grasp point, sees the cube.
START_POSE = {"robot_Rotation": 0.0, "robot_Pitch": -2.7, "robot_Elbow": 2.3, "robot_Wrist_Pitch": 1.6,
              "robot_Wrist_Roll": 0.0, "robot_Jaw": 0.0}
ARM_JOINTS = ["robot_Rotation", "robot_Pitch", "robot_Elbow", "robot_Wrist_Pitch", "robot_Wrist_Roll"]
JAW_JOINT = "robot_Jaw"
N_SUBSTEPS = 20  # same as SoFetchEnv; control dt = 20 * 0.002 = 0.04 s
GRASP_OFFSET = np.array([-0.020, -0.085, 0.0])  # grasp point in Fixed_Jaw body frame
JAW_OPEN = 0.8  # rad, ~5.8 cm gap; cube is 4 cm
JAW_CLOSED = 0.0  # rad; jaw stalls on the cube at ~0.3, which gives the squeeze
CUBE_HALF = 0.02  # scene.xml box size 0.02 (half-extent)
CUBE_R_RANGE = (0.12, 0.24)  # m, radial distance from pan axis
CUBE_AZ_RANGE = (-np.pi / 3, np.pi / 3)  # rad, around the arm's forward (-y) direction
HOVER_HEIGHT = 0.08  # grasp point above cube centre for approach and lift
GRASP_HEIGHT = 0.006  # grasp point above cube centre at grasp
SUCCESS_LIFT = 0.05  # cube must end >= 5 cm above its start height
IK_ITERS = 300
IK_DAMPING = 0.01
IK_POS_TOL = 1e-3  # m
IK_ROT_TOL = 2e-2  # rad
SETTLE_STEPS, APPROACH_STEPS, DWELL_STEPS, DESCEND_STEPS, CLOSE_STEPS, LIFT_STEPS, HOLD_STEPS = 10, 40, 5, 30, 15, 30, 25
CAMERA_CONFIG = {"distance": 0.7, "azimuth": 140.0, "elevation": -30.0, "lookat": np.array([0.0, -0.2, 0.05])}
MAX_SAMPLE_TRIES = 100


def min_jerk(t):
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def top_down_rotation(psi):
    """Fingers point straight down, jaw-opening axis at heading psi."""
    x = np.array([np.cos(psi), np.sin(psi), 0.0])
    y = np.array([0.0, 0.0, 1.0])
    return np.column_stack([x, y, np.cross(x, y)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)  # kinematics only; IK never disturbs the simulation

    arm_ids = [model.joint(n).id for n in ARM_JOINTS]
    arm_qadr = np.array([model.joint(n).qposadr[0] for n in ARM_JOINTS])
    arm_dadr = np.array([model.joint(n).dofadr[0] for n in ARM_JOINTS])
    lo, hi = model.jnt_range[arm_ids].T
    fixed_jaw_id = model.body("Fixed_Jaw").id
    cube_qadr = model.joint("object:joint").qposadr[0]
    cube_dadr = model.joint("object:joint").dofadr[0]

    # Hide the unused debug "target" box.
    model.geom_rgba[model.geom("target").id, 3] = 0.0

    mujoco.mj_forward(model, data)
    pan_xy = data.body("Rotation_Pitch").xpos[:2].copy()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    quat = np.zeros(4)
    e_r = np.zeros(3)

    def solve_ik(q_seed, p_target, R_target):
        q = np.array(q_seed, dtype=float)
        for _ in range(IK_ITERS):
            ik_data.qpos[arm_qadr] = q
            mujoco.mj_kinematics(model, ik_data)
            mujoco.mj_comPos(model, ik_data)
            R = ik_data.xmat[fixed_jaw_id].reshape(3, 3)
            p = ik_data.xpos[fixed_jaw_id] + R @ GRASP_OFFSET
            e_p = p_target - p
            mujoco.mju_mat2Quat(quat, (R_target @ R.T).flatten())
            mujoco.mju_quat2Vel(e_r, quat, 1.0)
            if np.linalg.norm(e_p) < IK_POS_TOL and np.linalg.norm(e_r) < IK_ROT_TOL:
                return q, True
            mujoco.mj_jac(model, ik_data, jacp, jacr, p, fixed_jaw_id)
            J = np.vstack([jacp[:, arm_dadr], jacr[:, arm_dadr]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), e)
            q = np.clip(q + dq, lo, hi)
        return q, False

    def plan_pick(q_home, cube_pos, cube_yaw):
        cx, cy = cube_pos[:2]
        psi_nat = np.arctan2(pan_xy[1] - cy, pan_xy[0] - cx)
        psi = cube_yaw + (np.pi / 2) * round((psi_nat - cube_yaw) / (np.pi / 2))
        R_target = top_down_rotation(psi)
        plan = []

        # 1. Hover
        hover = cube_pos + np.array([0, 0, HOVER_HEIGHT])
        pan_guess = np.arctan2(cx - pan_xy[0], -(cy - pan_xy[1]))
        q_hover, ok = solve_ik(np.array([pan_guess, -1.5, 1.5, 1.5, 0.0]), hover, R_target)
        if not ok:
            return None
        for k in range(1, APPROACH_STEPS + 1):
            s = min_jerk(k / APPROACH_STEPS)
            plan.append((q_home + (q_hover - q_home) * s, JAW_OPEN * s))
        plan += [(q_hover, JAW_OPEN)] * DWELL_STEPS

        # 2. Descend
        grasp = cube_pos + np.array([0, 0, GRASP_HEIGHT])
        q = q_hover
        for k in range(1, DESCEND_STEPS + 1):
            s = min_jerk(k / DESCEND_STEPS)
            q, ok = solve_ik(q, hover + (grasp - hover) * s, R_target)
            if not ok:
                return None
            plan.append((q, JAW_OPEN))
        plan += [(q, JAW_OPEN)] * DWELL_STEPS

        # 3. Close
        for k in range(1, CLOSE_STEPS + 1):
            s = min_jerk(k / CLOSE_STEPS)
            plan.append((q, JAW_OPEN + (JAW_CLOSED - JAW_OPEN) * s))
        plan += [(q, JAW_CLOSED)] * DWELL_STEPS

        # 4. Lift
        lift = cube_pos + np.array([0, 0, HOVER_HEIGHT])
        for k in range(1, LIFT_STEPS + 1):
            s = min_jerk(k / LIFT_STEPS)
            q, ok = solve_ik(q, grasp + (lift - grasp) * s, R_target)
            if not ok:
                return None
            plan.append((q, JAW_CLOSED))
        plan += [(q, JAW_CLOSED)] * HOLD_STEPS
        return plan

    renderer = None if args.no_render else MujocoRenderer(
        model, data, default_cam_config=CAMERA_CONFIG, visual_options={mujoco.mjtVisFlag.mjVIS_CAMERA: True})

    def control_step(q_arm, jaw):
        t0 = time.time()
        data.ctrl[:5] = q_arm
        data.ctrl[5] = jaw
        mujoco.mj_step(model, data, nstep=N_SUBSTEPS)
        if renderer is not None:
            renderer.render("human")
            time.sleep(max(0.0, N_SUBSTEPS * model.opt.timestep - (time.time() - t0)))

    def run_episode(rng):
        for _ in range(MAX_SAMPLE_TRIES):
            mujoco.mj_resetData(model, data)
            for name, value in START_POSE.items():
                data.qpos[model.joint(name).qposadr[0]] = value
            r = rng.uniform(*CUBE_R_RANGE)
            az = rng.uniform(*CUBE_AZ_RANGE)
            yaw = rng.uniform(0, np.pi / 2)
            data.qpos[arm_qadr[0]] = az  # pan toward the cube: azimuth az is exactly the pan angle that faces it
            data.qpos[cube_qadr:cube_qadr + 3] = [pan_xy[0] + r * np.sin(az), pan_xy[1] - r * np.cos(az), CUBE_HALF]
            data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            data.qvel[cube_dadr:cube_dadr + 6] = 0
            data.ctrl[:] = data.qpos[:6]
            mujoco.mj_forward(model, data)
            q_home = data.qpos[arm_qadr].copy()
            for _ in range(SETTLE_STEPS):
                control_step(q_home, data.ctrl[5])
            cube_pos = data.qpos[cube_qadr:cube_qadr + 3].copy()
            plan = plan_pick(q_home, cube_pos, yaw)
            if plan is not None:
                break
        else:
            raise RuntimeError("no reachable cube placement found")

        for q_arm, jaw in plan:
            control_step(q_arm, jaw)
        lift = data.qpos[cube_qadr + 2] - cube_pos[2]
        return lift >= SUCCESS_LIFT, {"cube_xy": cube_pos[:2], "yaw": yaw, "lift": lift}

    rng = np.random.default_rng(args.seed)
    successes = 0
    try:
        for i in range(args.episodes):
            ok, info = run_episode(rng)
            successes += ok
            print(f"episode {i}: {'success' if ok else 'FAILED'} cube=({info['cube_xy'][0]:.3f}, "
                  f"{info['cube_xy'][1]:.3f}) yaw={np.degrees(info['yaw']):.0f}deg lift={info['lift']:.3f}m")
        print(f"success {successes}/{args.episodes}")
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
