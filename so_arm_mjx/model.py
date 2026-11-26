"""
SO-100 fetch scene for the GPU versions: config, collision models, settled start state and observation indices.
Pure MuJoCo + numpy (no JAX), so it's shared by the MJX env (fetch_env.py), MuJoCo Warp, CPU playback
(visualise_mjx.py) and fidelity_check.py. See fetch_env.py's docstring for what each option changes.
"""
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_XML = os.environ.get(
    "SO_ARM_SCENE_XML",
    os.path.join(_HERE, "..", "so_arm_rl", "envs", "resources", "fetch", "scene.xml"),
)

# Same as so_arm_fetch_env.INITIAL_ARM_POSITION
INITIAL_ARM_POSITION = {"robot_Rotation": 0, "robot_Pitch": -3.10, "robot_Elbow": 3.10,
                        "robot_Wrist_Pitch": 0.0, "robot_Wrist_Roll": 0.0, "robot_Jaw": 0.0}

# Collision bitmasks: two geoms collide if (contype1 & conaffinity2) or (contype2 & conaffinity1)
_CUBE, _FLOOR, _GRIPPER, _LINK = 1, 2, 4, 8
_COLLISION_MASKS = {
    "cube": (_CUBE, _FLOOR | _GRIPPER),       # cube <-> floor, gripper
    "floor": (_FLOOR, _GRIPPER | _LINK),      # floor <-> cube (via cube's mask), gripper, links
    "gripper": (_GRIPPER, 0),                 # gripper <-> cube, floor; never with other arm parts
    "link": (_LINK, 0),                       # big links <-> floor only
}
_GRIPPER_BODIES = ("Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw")


@dataclass(frozen=True)
class FetchConfig:
    n_substeps: int = 20               # physics steps per env step (CPU env: 20)
    timestep: float = 0.002            # physics timestep (CPU env: 0.002)
    cone: str = "pyramidal"            # "pyramidal" (faster) or "elliptic" (CPU env)
    solver_iterations: int = 20        # CPU env: 100. 10 is too few: grasps slip (fidelity_check.py)
    ls_iterations: int = 20
    episode_length: int = 100          # CPU env MAX_TIMESTEPS
    fixed_goal: Optional[Sequence[float]] = (-0.3, -0.2, 0.025)  # None -> uniform in [goal_min, goal_max]
    goal_min: Sequence[float] = (0.1, 0.1, 0.1)
    goal_max: Sequence[float] = (0.5, 0.5, 0.5)
    initial_cube_position: Sequence[float] = (0.3, -0.3, 0.025)
    grasp_reward: float = 20.0
    target_reached_reward: float = 30.0
    grasp_threshold: float = 0.004     # jaw_site <-> cube centre distance counted as a grasp
    target_threshold: float = 0.002    # cube <-> goal distance counted as success
    filter_collisions: bool = True     # legacy flag (False == collision_model "full"); kept so old config.json files load
    collision_model: str = "filtered"  # "full", "filtered" or "simplified"; see module docstring
    max_hull_verts: Optional[int] = 32  # "simplified": cap on collision-mesh hull vertices (None = uncapped)

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


# Masks for collision_model="simplified"
_S_CUBE, _S_FLOOR, _S_PAD, _S_FLOOR_SAFETY, _S_CUBE_SAFETY = 1, 2, 4, 8, 16
_SIMPLE_MASKS = {
    "cube": (_S_CUBE, _S_FLOOR | _S_PAD | _S_CUBE_SAFETY),
    "floor": (_S_FLOOR, _S_FLOOR_SAFETY),
    "pad": (_S_PAD, 0),                   # finger pads: cube only (unchanged geoms)
    "floor_safety": (_S_FLOOR_SAFETY, 0),  # original link/wrist/jaw collision meshes: floor only
    "cube_safety": (_S_CUBE_SAFETY, 0),    # wrist/jaw capsules: cube only
}
_FLOOR_SAFETY_BODIES = ("Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw")
_CUBE_SAFETY_BODIES = ("Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw")
# rotations taking a capsule's local z axis onto the box's x / y / z axis
_AXIS_QUAT = {0: np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0]), 1: np.array([np.sqrt(0.5), -np.sqrt(0.5), 0, 0]),
              2: np.array([1.0, 0, 0, 0])}


def _simplified_model(max_hull_verts: Optional[int]) -> mujoco.MjModel:
    """See collision_model="simplified" in the module docstring. Added capsules have zero density, so
    masses/inertias are untouched."""
    base = mujoco.MjModel.from_xml_path(SCENE_XML)
    spec = mujoco.MjSpec.from_file(SCENE_XML)
    collision_meshes = set()
    for g in range(base.ngeom):
        if base.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or not (base.geom_contype[g] or base.geom_conaffinity[g]):
            continue
        body = base.body(base.geom_bodyid[g]).name
        if body not in _FLOOR_SAFETY_BODIES:
            continue  # Base is welded to the world
        collision_meshes.add(base.mesh(base.geom_dataid[g]).name)
        if body in _CUBE_SAFETY_BODIES:
            # capsule inside the mesh's bounding box: radius = smallest half-extent, along the longest axis
            center, half = base.geom_aabb[g, :3], base.geom_aabb[g, 3:]
            rot = np.zeros(3)
            mujoco.mju_rotVecQuat(rot, center, base.geom_quat[g])
            pos = base.geom_pos[g] + rot
            axis = int(np.argmax(half))
            r = float(np.min(half))
            half_len = max(float(half[axis]) - r, 1e-4)
            cq = np.zeros(4)
            mujoco.mju_mulQuat(cq, base.geom_quat[g], _AXIS_QUAT[axis])
            spec.body(body).add_geom(name=f"cube_safety_{g}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, half_len, 0],
                                     pos=pos, quat=cq, density=0, group=3, rgba=[1, 0.4, 0, 0.3])
    if max_hull_verts:
        for mesh in spec.meshes:
            if mesh.name in collision_meshes:
                mesh.maxhullvert = max_hull_verts
    m = spec.compile()
    floor_bodies = {m.body(b).id for b in _FLOOR_SAFETY_BODIES}
    for g in range(m.ngeom):
        name = m.geom(g).name
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) and not name.startswith("cube_safety_"):
            continue  # visual-only
        if name == "object":
            kind = "cube"
        elif name == "floor":
            kind = "floor"
        elif "jaw_pad" in name:
            kind = "pad"
        elif name.startswith("cube_safety_"):
            kind = "cube_safety"
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_bodyid[g] in floor_bodies:
            kind = "floor_safety"
        else:
            m.geom_contype[g] = m.geom_conaffinity[g] = 0  # e.g. Base mesh
            continue
        m.geom_contype[g], m.geom_conaffinity[g] = _SIMPLE_MASKS[kind]
    return m


def build_mj_model(cfg: FetchConfig = FetchConfig()) -> mujoco.MjModel:
    """Loads the same scene.xml as the CPU env and applies the GPU-friendly settings.
    Also used by visualise_mjx.py, so CPU playback has exactly the physics the policy trained on."""
    model_kind = cfg.collision_model if cfg.filter_collisions else "full"
    if model_kind not in ("full", "filtered", "simplified"):
        raise ValueError(f"unknown collision_model {model_kind!r}")
    m = _simplified_model(cfg.max_hull_verts) if model_kind == "simplified" else mujoco.MjModel.from_xml_path(SCENE_XML)
    m.opt.timestep = cfg.timestep
    m.opt.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
                  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[cfg.cone]
    m.opt.iterations = cfg.solver_iterations
    m.opt.ls_iterations = cfg.ls_iterations

    if model_kind == "filtered":
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue  # visual-only geom, or the target marker
            body = m.body(m.geom_bodyid[g]).name
            name = m.geom(g).name
            if name == "object":
                kind = "cube"
            elif name == "floor":
                kind = "floor"
            elif body in _GRIPPER_BODIES:
                kind = "gripper"
            else:
                kind = "link"
            m.geom_contype[g], m.geom_conaffinity[g] = _COLLISION_MASKS[kind]
    return m


def settled_initial_state(m: mujoco.MjModel, cfg: FetchConfig):
    """Replicates SoFetchEnv._reset_sim on CPU: place cube + arm, then settle 10 env steps with zero action.
    The result is deterministic, so it's computed once and reused for every reset."""
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    obj = m.joint("object:joint")
    d.qpos[obj.qposadr[0]:obj.qposadr[0] + 7] = np.concatenate([cfg.initial_cube_position, [1, 0, 0, 0]])
    for name, value in INITIAL_ARM_POSITION.items():
        d.qpos[m.joint(name).qposadr[0]] = value
    ctrl_center = m.actuator_ctrlrange.mean(axis=1)
    for _ in range(10):
        d.ctrl[:] = ctrl_center  # action 0 -> centre of ctrlrange
        # The CPU env steps twice per settle iteration (_apply_action steps, then _reset_sim steps again)
        mujoco.mj_step(m, d, nstep=2 * cfg.n_substeps)
    return d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()


@dataclass(frozen=True)
class ObsIndices:
    robot_qpos: np.ndarray
    robot_qvel: np.ndarray
    object_qpos: np.ndarray
    jaw_site: int

    @staticmethod
    def from_model(m: mujoco.MjModel) -> "ObsIndices":
        robot = [j for j in range(m.njnt) if m.joint(j).name.startswith("robot")]
        obj = m.joint("object:joint").qposadr[0]
        return ObsIndices(
            robot_qpos=np.array([m.jnt_qposadr[j] for j in robot]),
            robot_qvel=np.array([m.jnt_dofadr[j] for j in robot]),
            object_qpos=np.arange(obj, obj + 7),
            jaw_site=m.site("jaw_site").id,
        )
