"""
Physics-only throughput of the Warp cases on the current GPU(s).

    python -m so_arm_warp.bench --cases full original --nworld 8192 32768 --ls 20 50
    python -m so_arm_warp.bench --cases full --nworld 32768 --multi-gpu     # one process per GPU, summed
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from .physics import CASES, PhysicsConfig, WarpBatch


def bench_one(case, nworld, ls, device=None, reps=20):
    import warp as wp
    pc = PhysicsConfig(case=case, ls_iterations=ls)
    t = time.time()
    b = WarpBatch(pc, nworld, device=device)
    jr = np.array([b.m.jnt_range[b.m.actuator_trnid[i, 0]] for i in range(b.m.nu)])
    b.dw.ctrl.assign(np.random.default_rng(0).uniform(jr[:, 0], jr[:, 1], (nworld, b.m.nu)).astype(np.float32))
    b.step_env()                      # compile + capture
    for _ in range(5):                # move into contact-rich states
        b.step_env()
    wp.synchronize_device(b.device)
    setup = time.time() - t
    t = time.time()
    for _ in range(reps):
        b.step_env()
    wp.synchronize_device(b.device)
    dt = (time.time() - t) / reps
    st = b.stats()
    return dict(case=case, nworld=nworld, ls=ls if case != "original" else 50, device=str(b.device),
                env_sps=nworld / dt, ms_per_env_step=dt * 1000, setup_s=setup, **st)


def fmt(r):
    return (f"{r['case']:10s} ls {r['ls']:3d} nworld {r['nworld']:6d} [{r['device']}]: {r['env_sps']:10,.0f} env steps/s"
            f" | contacts {r['contacts']}/{r['contacts_cap']} | max rows/world {r['max_rows']}/{r['rows_cap']}"
            f" | solver iters {r['mean_solver_iters']:.1f} ({100 * r['frac_at_iter_cap']:.0f}% at cap) | finite {r['finite']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    p.add_argument("--nworld", nargs="+", type=int, default=[8192, 32768])
    p.add_argument("--ls", nargs="+", type=int, default=[50])
    p.add_argument("--device", default=None)
    p.add_argument("--multi-gpu", action="store_true", help="run each config on every GPU at once (one process each)")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    results = []
    if a.multi_gpu:
        import warp as wp
        n = wp.get_cuda_device_count()
        for case in a.cases:
            for nworld in a.nworld:
                for ls in a.ls:
                    procs = [subprocess.Popen([sys.executable, "-m", "so_arm_warp.bench", "--cases", case, "--nworld",
                                               str(nworld), "--ls", str(ls), "--device", f"cuda:{i}", "--json", f"/tmp/_b{i}.json"],
                                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for i in range(n)]
                    errs = [pr.communicate()[1] for pr in procs]
                    try:
                        parts = [json.load(open(f"/tmp/_b{i}.json"))[0] for i in range(n)]
                    except Exception:
                        print(f"{case} nworld {nworld}: multi-GPU run FAILED\n", errs[0][-800:], flush=True)
                        continue
                    r = dict(parts[0]); r["device"] = f"{n} GPUs"; r["env_sps"] = sum(x["env_sps"] for x in parts)
                    r["nworld"] = nworld * n
                    print(fmt(r), flush=True); results.append(r)
    else:
        for case in a.cases:
            for nworld in a.nworld:
                for ls in (a.ls if case != "original" else [50]):
                    try:
                        r = bench_one(case, nworld, ls, a.device)
                        print(fmt(r), flush=True); results.append(r)
                    except Exception as e:
                        print(f"{case:10s} ls {ls:3d} nworld {nworld:6d}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
