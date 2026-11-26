"""
How close are the GPU collision models to the original SO-100 physics? Runs on CPU MuJoCo (no GPU needed):

    python -m so_arm_mjx.fidelity_check

Every model is compared with the ORIGINAL scene (the physics SoFetchEnv uses: full collision meshes, elliptic
cone, 100 solver iterations):

1. start pose      settled reset state vs the original
2. random motions  30 random open-loop command sequences; jaw/cube drift from the original, plus "leaks":
                   each trajectory is replayed through the ORIGINAL collision meshes to count states where an arm
                   part is >5 mm into the floor or the cube is >5 mm into the gripper (what the safety contacts prevent)
3. grasp & hold    close the jaws on the cube in mid-air, hold, then swing the base; does the grip hold?
4. push tests      sweep the gripper (then the lower arm) into a cube on the floor; how far is it pushed?
                   "simplified" keeps gripper-vs-cube contacts but drops link-vs-cube, so the link test shows that gap.
"""
import argparse

import mujoco
import numpy as np

from .fetch_env import SCENE_XML, FetchConfig, build_mj_model, settled_initial_state

ARM = ["robot_Rotation", "robot_Pitch", "robot_Elbow", "robot_Wrist_Pitch", "robot_Wrist_Roll", "robot_Jaw"]
_ORIG_CFG = FetchConfig(cone="elliptic", solver_iterations=100, ls_iterations=50, collision_model="full")

MODELS = {
    "original": _ORIG_CFG,
    "filtered (GPU solver)": FetchConfig(collision_model="filtered"),
    "simplified, original solver": FetchConfig(collision_model="simplified", cone="elliptic",
                                               solver_iterations=100, ls_iterations=50),
    "simplified hull32 (GPU)": FetchConfig(collision_model="simplified", max_hull_verts=32),
    "simplified hull16 (GPU)": FetchConfig(collision_model="simplified", max_hull_verts=16),
    "simplified hull32, 10 substeps": FetchConfig(collision_model="simplified", max_hull_verts=32,
                                                  n_substeps=10, timestep=0.004),
}


class Sim:
    def __init__(self, cfg: FetchConfig):
        self.cfg = cfg
        self.m = build_mj_model(cfg) if cfg.collision_model != "full" else self._original(cfg)
        self.d = mujoco.MjData(self.m)
        self.q0, self.v0, self.c0 = settled_initial_state(self.m, cfg)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in ARM])
        self.obj = self.m.joint("object:joint").qposadr[0]
        self.obj_dof = self.m.joint("object:joint").dofadr[0]
        self.jaw = self.m.site("jaw_site").id

    @staticmethod
    def _original(cfg):
        m = mujoco.MjModel.from_xml_path(SCENE_XML)  # untouched: elliptic, default iterations, full meshes
        m.opt.timestep = cfg.timestep
        return m

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = self.q0, self.v0, self.c0
        mujoco.mj_forward(self.m, self.d)

    def run(self, ctrl, seconds, hold_cube_at=None):
        """Apply ctrl for `seconds` (in 0.04 s env steps); returns qpos per env step."""
        out = []
        steps_per_env = int(round(0.04 / self.m.opt.timestep))
        for _ in range(int(round(seconds / 0.04))):
            self.d.ctrl[:] = ctrl
            for _ in range(steps_per_env):
                if hold_cube_at is not None:  # kinematically hold the cube (used while the jaws close)
                    self.d.qpos[self.obj:self.obj + 7] = np.concatenate([hold_cube_at, [1, 0, 0, 0]])
                    self.d.qvel[self.obj_dof:self.obj_dof + 6] = 0
                mujoco.mj_step(self.m, self.d)
            out.append(self.d.qpos.copy())
        return np.array(out)

    def jaw_pos(self, qpos):
        d = mujoco.MjData(self.m)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.jaw].copy()


class LeakChecker:
    """Measures penetration using the ORIGINAL model's collision meshes."""
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.d = mujoco.MjData(self.m)
        self.floor = self.m.geom("floor").id
        self.cube = self.m.geom("object").id
        gripper = {self.m.body(b).id for b in ("Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw")}
        self.arm_geoms = {g for g in range(self.m.ngeom)
                          if (self.m.geom_contype[g] or self.m.geom_conaffinity[g]) and g not in (self.floor, self.cube)
                          and self.m.geom_bodyid[g] != self.m.body("Base").id}
        self.gripper_geoms = {g for g in self.arm_geoms if self.m.geom_bodyid[g] in gripper}

    def leaks(self, qpos, depth=0.005):
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        floor_leak = cube_leak = False
        for c in self.d.contact[:self.d.ncon]:
            pair = {c.geom1, c.geom2}
            if c.dist < -depth and self.floor in pair and pair & self.arm_geoms:
                floor_leak = True
            if c.dist < -depth and self.cube in pair and pair & self.gripper_geoms:
                cube_leak = True
        return floor_leak, cube_leak


def random_ctrls(seed, m, n_segments=10):
    rng = np.random.default_rng(seed)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    # random targets over the joints' useful range (ctrlrange is +-pi for all joints)
    jlo = np.array([m.jnt_range[m.actuator_trnid[i, 0]][0] for i in range(m.nu)])
    jhi = np.array([m.jnt_range[m.actuator_trnid[i, 0]][1] for i in range(m.nu)])
    return [np.clip(rng.uniform(jlo, jhi), lo, hi) for _ in range(n_segments)]


def grasp_pose_ctrl():
    # arm raised with the gripper in mid-air (found by forward kinematics); jaw open
    return np.array([0.0, -1.2, 1.4, 0.6, 0.0, 1.2])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    a = p.parse_args()
    sims = {k: Sim(c) for k, c in MODELS.items()}
    ref = sims["original"]
    leak = LeakChecker()

    print("\n1) START POSE (max |difference| from original)")
    for k, s in sims.items():
        print(f"   {k:32s} arm joints {np.abs(s.q0[s.qadr] - ref.q0[ref.qadr]).max():.2e} rad | "
              f"cube {np.abs(s.q0[s.obj:s.obj + 3] - ref.q0[ref.obj:ref.obj + 3]).max():.2e} m")

    print(f"\n2) RANDOM MOTIONS ({a.seeds} sequences x 4 s): drift from original, and leaks through the ORIGINAL meshes")
    print(f"   {'model':32s} {'jaw drift median/max':>22s} {'cube moved (mean)':>18s} {'cube drift max':>15s} {'arm->floor leaks':>17s} {'cube->gripper leaks':>20s}")
    ref_traj = {}
    for k, s in sims.items():
        jaw_d, cube_d, cube_moved, fl, cl, n = [], [], [], 0, 0, 0
        for seed in range(a.seeds):
            s.reset()
            traj = np.concatenate([s.run(c, 0.4) for c in random_ctrls(seed, s.m)])
            if k == "original":
                ref_traj[seed] = traj
            r = ref_traj[seed]
            jaw_d.append(max(np.linalg.norm(s.jaw_pos(q) - ref.jaw_pos(rq)) for q, rq in zip(traj[::5], r[::5])))
            cube_d.append(np.abs(traj[:, s.obj:s.obj + 3] - r[:, ref.obj:ref.obj + 3]).max())
            cube_moved.append(np.linalg.norm(traj[-1, s.obj:s.obj + 3] - s.q0[s.obj:s.obj + 3]))
            for q in traj[::2]:
                f, c = leak.leaks(q)
                fl += f; cl += c; n += 1
        print(f"   {k:32s} {np.median(jaw_d) * 100:8.2f} / {np.max(jaw_d) * 100:5.2f} cm "
              f"{np.mean(cube_moved) * 100:15.2f} cm {np.max(cube_d) * 100:12.2f} cm "
              f"{100 * fl / n:15.1f} % {100 * cl / n:18.1f} %")

    print("\n3) GRASP & HOLD (cube placed against the fixed jaw's pads in mid-air, jaws close, hold 1.5 s, swing the base 1 rad)")
    for k, s in sims.items():
        s.reset()
        ctrl = grasp_pose_ctrl()
        s.run(ctrl, 1.5)                                   # move to pose, jaw open
        pad = lambda side: np.mean([s.d.geom_xpos[s.m.geom(f"{side}_jaw_pad_{i}").id] for i in (3, 4)], axis=0)
        fp, mp = pad("fixed"), pad("moving")
        place = fp + (mp - fp) / np.linalg.norm(mp - fp) * 0.0215   # cube face just touching the fixed pads
        closed = ctrl.copy(); closed[5] = -0.2
        s.run(closed, 0.5, hold_cube_at=place)             # close the moving jaw on the held cube
        if k == "original":
            grip_dist = np.linalg.norm(s.d.qpos[s.obj:s.obj + 3] - s.d.site_xpos[s.jaw])
        s.run(closed, 1.5)                                 # release the cube: does the grip hold it?
        held = np.linalg.norm(s.d.qpos[s.obj:s.obj + 3] - s.d.site_xpos[s.jaw])
        swing = closed.copy(); swing[0] = 1.0
        s.run(swing, 1.5)
        after = np.linalg.norm(s.d.qpos[s.obj:s.obj + 3] - s.d.site_xpos[s.jaw])
        z = s.d.qpos[s.obj + 2]
        verdict = "HELD" if after < 0.03 and z > 0.05 else "DROPPED"
        print(f"   {k:32s} cube-to-jaw_site after hold {held * 100:5.2f} cm, after swing {after * 100:5.2f} cm, cube height {z * 100:5.1f} cm -> {verdict}")
    print(f"   NOTE: in a real grip the cube centre sits {grip_dist * 100:.1f} cm from jaw_site, but the grasp reward needs < 0.4 cm"
          " (jaw_site is 1.4 cm from the fixed pads; the cube is 4 cm wide).")

    print("\n4) PUSH TESTS (cube on the floor in the sweep path; base swings 0.7 rad; cube displacement)")
    for k, s in sims.items():
        res = []
        for part in ("jaw_site", "Lower_Arm"):
            s.reset()
            pose = np.array([0.0, -0.8, 1.6, 1.0, 0.0, 0.0])    # gripper low over the floor
            s.run(pose, 1.5)
            base = s.d.xpos[s.m.body("Base").id]
            pt = s.d.site_xpos[s.jaw] if part == "jaw_site" else s.d.xpos[s.m.body("Lower_Arm").id]
            rel = pt[:2] - base[:2]
            c, sn = np.cos(0.3), np.sin(0.3)                    # 0.3 rad ahead of the part along the sweep
            target = base[:2] + np.array([c * rel[0] - sn * rel[1], sn * rel[0] + c * rel[1]])
            start_pos = np.array([target[0], target[1], 0.0201])
            s.d.qpos[s.obj:s.obj + 7] = np.concatenate([start_pos, [1, 0, 0, 0]])
            s.d.qvel[s.obj_dof:s.obj_dof + 6] = 0
            mujoco.mj_forward(s.m, s.d)
            sweep = pose.copy(); sweep[0] = 0.7
            s.run(sweep, 2.0)
            res.append(np.linalg.norm(s.d.qpos[s.obj:s.obj + 2] - start_pos[:2]))
        print(f"   {k:32s} gripper push: cube moved {res[0] * 100:6.2f} cm | lower-arm push: cube moved {res[1] * 100:6.2f} cm")


if __name__ == "__main__":
    main()
