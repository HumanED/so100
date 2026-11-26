"""
The fidelity tests from so_arm_mjx/fidelity_check.py, with physics stepped by MuJoCo Warp (float32, GPU),
compared against the exact original model on CPU MuJoCo:

    python -m so_arm_warp.fidelity --cases full original --seeds 20
"""
import argparse

import mujoco
import numpy as np

from so_arm_mjx.fidelity_check import _ORIG_CFG, Sim, run_all

from .physics import CASES, PhysicsConfig, build


class WarpSim(Sim):
    """Same interface as fidelity_check.Sim, but run() steps a single Warp world.
    State is copied in/out of Warp around each run() so the tests can read/teleport via self.d."""

    def __init__(self, pc: PhysicsConfig, device=None):
        import warp as wp
        import mujoco_warp as mjw
        self.wp, self.mjw, self.pc = wp, mjw, pc
        m, cfg = build(pc)
        self.cfg = cfg
        self.m = m
        self.d = mujoco.MjData(m)
        from so_arm_mjx.model import settled_initial_state
        self.q0, self.v0, self.c0 = settled_initial_state(m, cfg)
        self.qadr = np.array([m.joint(j).qposadr[0] for j in
                              ["robot_Rotation", "robot_Pitch", "robot_Elbow", "robot_Wrist_Pitch", "robot_Wrist_Roll", "robot_Jaw"]])
        self.obj = m.joint("object:joint").qposadr[0]
        self.obj_dof = m.joint("object:joint").dofadr[0]
        self.jaw = m.site("jaw_site").id
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            self.mw = mjw.put_model(m)
            self.mw.opt.warn_overflow = 0

    def run(self, ctrl, seconds, hold_cube_at=None):
        wp, mjw = self.wp, self.mjw
        steps_per_env = int(round(0.04 / self.m.opt.timestep))
        self.d.ctrl[:] = ctrl
        out = []
        with wp.ScopedDevice(self.device):
            dw = mjw.put_data(self.m, self.d, nworld=1, naconmax=self.pc.contacts(), njmax=self.pc.njmax)
            for _ in range(int(round(seconds / 0.04))):
                for _ in range(steps_per_env):
                    if hold_cube_at is not None:
                        q = dw.qpos.numpy(); q[0, self.obj:self.obj + 7] = np.concatenate([hold_cube_at, [1, 0, 0, 0]])
                        v = dw.qvel.numpy(); v[0, self.obj_dof:self.obj_dof + 6] = 0
                        dw.qpos.assign(q); dw.qvel.assign(v)
                    mjw.step(self.mw, dw)
                out.append(dw.qpos.numpy()[0].astype(np.float64))
            self.d.qpos[:] = dw.qpos.numpy()[0]
            self.d.qvel[:] = dw.qvel.numpy()[0]
            self.d.time = float(dw.time.numpy()[0])
        mujoco.mj_forward(self.m, self.d)
        return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["full", "original"], choices=CASES)
    p.add_argument("--ls", type=int, default=50)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    sims = {"original": Sim(_ORIG_CFG, exact=True)}           # reference: CPU MuJoCo, exact original model
    for case in a.cases:
        sims[f"warp {case}" + (f" ls{a.ls}" if case != "original" else "")] = WarpSim(PhysicsConfig(case=case, ls_iterations=a.ls), a.device)
    run_all(sims, a.seeds)


if __name__ == "__main__":
    main()
