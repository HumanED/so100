"""
Run a trained Warp policy in ordinary CPU MuJoCo (no GPU needed):

    python -m so_arm_warp.play warp_out/my-run/model_final.pt                       # viewer, training physics
    python -m so_arm_warp.play warp_out/my-run/model_final.pt --physics original --episodes 10 --headless
    python -m so_arm_warp.play warp_out/my-run/model_final.pt --video eval.mp4

--physics original runs the exact SoFetchEnv physics, a check that the policy works beyond its training physics.
On macOS the interactive viewer needs `mjpython -m so_arm_warp.play ...`.
"""
import argparse
import time

import mujoco
import numpy as np
import torch

from .scene import ObsIndices

from .physics import PhysicsConfig, build
from .ppo import ActorCritic, PPOConfig


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    ppo = PPOConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["ppo"].items()})
    ac = ActorCritic(cfg["obs_size"], cfg["action_size"], ppo)
    ac.load_state_dict(ck["model"]); ac.eval()
    return ac, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--physics", default=None, help="'original' or a case name; default: the case it trained on")
    p.add_argument("--episodes", type=int, default=0, help="0 = loop forever (viewer)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", default=None)
    a = p.parse_args()

    ac, cfg = load(a.checkpoint)
    case = a.physics or cfg["case"]
    m, fcfg = build(PhysicsConfig(**{**cfg["physics"], "case": case}))
    from .scene import settled_initial_state
    q0, v0, c0 = settled_initial_state(m, fcfg)
    d = mujoco.MjData(m)
    idx = ObsIndices.from_model(m)
    pads = [m.geom(f"{s}_jaw_pad_{i}").id for s in ("fixed", "moving") for i in (3, 4)]
    goal = np.asarray(cfg["fixed_goal"] if cfg["fixed_goal"] is not None else np.random.uniform(0.1, 0.5, 3))
    tgt = m.joint("target:joint")
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    dt = fcfg.timestep * fcfg.n_substeps

    viewer = renderer = None
    frames = []
    if a.video:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0.0, -0.2, 0.05]
        cam.distance, cam.azimuth, cam.elevation = 0.95, 120.0, -35.0
    elif not a.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)

    def obs():
        obj = d.qpos[idx.object_qpos]; jaw = d.site_xpos[idx.jaw_site]
        return np.concatenate([d.qpos[idx.robot_qpos], d.qvel[idx.robot_qvel], obj, jaw, obj[:3] - jaw, obj[:3] - goal, goal])

    print(f"policy trained on '{cfg['case']}' with reward '{cfg['reward']}'; playing on '{case}' physics")
    n = 0
    while a.episodes == 0 or n < a.episodes:
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q0, v0, c0
        mujoco.mj_forward(m, d)
        best_goal, gripped, lifted = 1e9, False, False
        for _ in range(100):
            t = time.time()
            act = ac.act_deterministic(torch.as_tensor(obs(), dtype=torch.float32)[None])[0].numpy()
            d.ctrl[:] = np.clip((lo + hi) / 2 + np.clip(act, -1, 1) * (hi - lo) / 2, lo, hi)
            for _ in range(fcfg.n_substeps):
                mujoco.mj_step(m, d)
                d.qpos[tgt.qposadr[0]:tgt.qposadr[0] + 7] = np.concatenate([goal, [1, 0, 0, 0]])  # show the goal marker
                d.qvel[tgt.dofadr[0]:tgt.dofadr[0] + 6] = 0
            cube = d.qpos[idx.object_qpos[:3]]
            pad_mid = 0.5 * (d.geom_xpos[pads[:2]].mean(0) + d.geom_xpos[pads[2:]].mean(0))
            gripped |= np.linalg.norm(cube - pad_mid) < 0.015
            lifted |= cube[2] > 0.05
            best_goal = min(best_goal, np.linalg.norm(cube - goal))
            if renderer:
                renderer.update_scene(d, camera=cam); frames.append(renderer.render())
            if viewer:
                if not viewer.is_running():
                    return
                viewer.sync(); time.sleep(max(0.0, dt - (time.time() - t)))
        n += 1
        print(f"episode {n}: cube in gripper {gripped} | lifted {lifted} | closest to goal {best_goal * 100:.1f} cm "
              f"| success {best_goal < 0.002}")
    if a.video and frames:
        import imageio
        imageio.mimsave(a.video, frames, fps=int(round(1 / dt)))
        print("wrote", a.video)


if __name__ == "__main__":
    main()
