"""
Physics setups ("cases") for MuJoCo Warp, and a batched stepper used by the benchmark and the training env.

Cases (all build from scene.py, i.e. the same scene.xml SoFetchEnv loads):
  simplified   simplified collision model (see scene.py), GPU solver, 10 x 4 ms substeps
  filtered     original collision meshes, no arm-vs-arm checks, GPU solver, 10 x 4 ms substeps
  full         ALL original collisions, GPU solver (pyramidal cone, `iterations`), 10 x 4 ms substeps
  original     the untouched scene.xml exactly as SoFetchEnv runs it (elliptic cone, 100 iterations), 20 x 2 ms
  full_2ms     `full` (collisions + GPU solver) but original's 20 x 2 ms substeps: isolates the timestep
  original_20it  `original` with 20 solver iterations instead of 100: isolates the iteration count
  full_elliptic  `full` with original's elliptic friction cone (impratio 10 from the XML): isolates the cone
"""
from dataclasses import dataclass, replace
from typing import Optional

import mujoco
import numpy as np

from .scene import SCENE_XML, FetchConfig, build_mj_model, settled_initial_state

CASES = ("simplified", "filtered", "full", "original", "full_2ms", "original_20it", "full_elliptic")

# Contacts actually seen are ~4.5 per world (Kaggle T4 benchmark); these leave a wide margin without
# reserving 20x too much memory (the cause of the out-of-memory failures at 32k worlds).
CONTACTS_PER_WORLD = {"simplified": 16, "filtered": 24, "full": 24, "original": 24, "full_2ms": 24, "original_20it": 24, "full_elliptic": 24}
NJMAX = 192  # constraint rows per world (contacts x up to 6 rows + joint limits + joint friction)


@dataclass(frozen=True)
class PhysicsConfig:
    case: str = "full"
    solver_iterations: int = 20
    ls_iterations: int = 50     # 20 hit the line-search cap constantly on the T4
    contacts_per_world: Optional[int] = None
    njmax: int = NJMAX

    def fetch_config(self) -> FetchConfig:
        if self.case in ("original", "original_20it"):
            return FetchConfig(collision_model="full", n_substeps=20, timestep=0.002, cone="elliptic",
                               solver_iterations=100 if self.case == "original" else 20, ls_iterations=50)
        if self.case not in CASES:
            raise ValueError(f"case must be one of {CASES}")
        if self.case == "full_2ms":
            return FetchConfig(collision_model="full", max_hull_verts=16, n_substeps=20, timestep=0.002,
                               cone="pyramidal", solver_iterations=self.solver_iterations,
                               ls_iterations=self.ls_iterations)
        if self.case == "full_elliptic":
            return FetchConfig(collision_model="full", max_hull_verts=16, n_substeps=10, timestep=0.004,
                               cone="elliptic", solver_iterations=self.solver_iterations,
                               ls_iterations=self.ls_iterations)
        return FetchConfig(collision_model=self.case, max_hull_verts=16, n_substeps=10, timestep=0.004,
                           cone="pyramidal", solver_iterations=self.solver_iterations,
                           ls_iterations=self.ls_iterations)

    def contacts(self) -> int:
        return self.contacts_per_world or CONTACTS_PER_WORLD[self.case]


def build(pc: PhysicsConfig):
    """Returns (MjModel, FetchConfig). 'original' is the untouched XML (its own solver settings)."""
    cfg = pc.fetch_config()
    if pc.case in ("original", "original_20it"):
        m = mujoco.MjModel.from_xml_path(SCENE_XML)
        m.opt.timestep = cfg.timestep
        if pc.case == "original_20it":
            m.opt.iterations = cfg.solver_iterations
    else:
        m = build_mj_model(cfg)
    return m, cfg


class WarpBatch:
    """nworld copies of the scene on one GPU. step_env() advances every world by one env step (n_substeps)."""

    def __init__(self, pc: PhysicsConfig, nworld: int, device: Optional[str] = None, quiet: bool = True):
        import warp as wp
        import mujoco_warp as mjw
        self.wp, self.mjw = wp, mjw
        self.pc, self.nworld = pc, nworld
        self.m, self.cfg = build(pc)
        self.device = wp.get_device(device)
        q0, v0, c0 = settled_initial_state(self.m, self.cfg)
        d = mujoco.MjData(self.m)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q0, v0, c0
        mujoco.mj_forward(self.m, d)
        self.q0, self.v0, self.c0 = q0, v0, c0
        with wp.ScopedDevice(self.device):
            self.mw = mjw.put_model(self.m)
            if quiet:
                self.mw.opt.warn_overflow = 0  # we check buffers/convergence ourselves (see stats())
            self.dw = mjw.put_data(self.m, d, nworld=nworld, naconmax=nworld * pc.contacts(), njmax=pc.njmax)
        self._graph = None

    def step_env(self):
        wp = self.wp
        with wp.ScopedDevice(self.device):
            if self.device.is_cuda:
                if self._graph is None:
                    for _ in range(self.cfg.n_substeps):   # compile kernels before capturing
                        self.mjw.step(self.mw, self.dw)
                    with wp.ScopedCapture() as cap:
                        for _ in range(self.cfg.n_substeps):
                            self.mjw.step(self.mw, self.dw)
                    self._graph = cap.graph
                wp.capture_launch(self._graph)
            else:
                for _ in range(self.cfg.n_substeps):
                    self.mjw.step(self.mw, self.dw)

    def stats(self) -> dict:
        """Buffer use and solver convergence for the current state."""
        nacon = int(self.dw.nacon.numpy()[0])
        nefc = self.dw.nefc.numpy()
        niter = self.dw.solver_niter.numpy()
        it_cap = self.m.opt.iterations
        return dict(contacts=nacon, contacts_cap=self.nworld * self.pc.contacts(),
                    max_rows=int(nefc.max()), rows_cap=self.pc.njmax,
                    mean_solver_iters=float(niter.mean()), frac_at_iter_cap=float((niter >= it_cap).mean()),
                    finite=bool(np.isfinite(self.dw.qpos.numpy()).all()))
