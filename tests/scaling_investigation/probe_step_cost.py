#!/usr/bin/env python3
"""
tests/scaling_investigation/probe_step_cost.py — isolate what a single simulation step costs, and
which knobs drive it, at large particle counts.

Why this exists
---------------
``benchmark_scaling.py`` measures whole operations (action, settle, reset).
Its *settle* numbers scale cleanly, but its *action* numbers are dominated by
two confounds: ``plate_velocity_translation`` derives its step count from the
largest sampled sweep distance across all envs (so cost depends on the random
action, and grows with n_envs for reasons unrelated to GPU scaling), and the
loop exits early only when *every* env reaches its goal.

This probe removes both confounds by timing:

  raw        ``scene.step()`` with nothing else — the physics cost floor
  sweep-loop one iteration of ``plate_velocity_translation``'s body, including
             the per-step ``set_dofs_position`` and the ``.item()`` sync
  delta      sweep-loop minus raw = per-step control + synchronisation tax

and sweeps ``max_collision_pairs`` / ``box_box_detection``, which set the
solver's preallocated constraint-array sizes. If step time tracks
``max_collision_pairs``, the solver is iterating allocated slots rather than
active contacts, and the option has to be tuned tightly rather than set
generously.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_step_cost
    python -m tests.scaling_investigation.probe_step_cost --particles 200 --mcp 200 800 2000 --envs 1
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# This script lives outside Genesis/, so paths to the simulator's configs are
# resolved explicitly rather than relative to this file.
GENESIS_DIR = Path(__file__).resolve().parents[2] / "Genesis"
REPO_ROOT = Path(__file__).resolve().parents[2]



def _config(n_particles, particle_size, mcp, box_box):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg["rigid_options"]["max_collision_pairs"] = mcp
    cfg["rigid_options"]["box_box_detection"] = box_box
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def run_cell(n_particles, n_envs, particle_size, mcp, box_box, n_steps):
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    def sync():
        torch.cuda.synchronize()

    def vram():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2 ** 30

    base = vram()
    sim = SandboxManipulation(config=_config(n_particles, particle_size, mcp, box_box),
                              n_envs=n_envs, debug=False)
    sim.build()
    out = dict(n_particles=n_particles, n_envs=n_envs, mcp=mcp,
               box_box=box_box, particle_size=particle_size)
    try:
        sim.shuffle_particles()
        sim.update_material_state()          # settle the pile + warm kernels
        out["vram_gib"] = vram() - base

        # ---- raw scene.step() ------------------------------------------
        for _ in range(5):
            sim._step_scene()
        sync(); t = time.perf_counter()
        for _ in range(n_steps):
            sim._step_scene()
        sync()
        out["ms_per_raw_step"] = (time.perf_counter() - t) / n_steps * 1000

        # ---- one sweep-loop iteration ----------------------------------
        # Same per-step work plate_velocity_translation does, at the same
        # plate pose, so only the control/sync overhead is added.
        p = sim.plate.get_pos()
        v = torch.zeros_like(p)
        sim.plate.control_dofs_position_velocity(p, v, dofs_idx_local=[0, 1, 2])
        p_end = p.clone()
        reached = torch.zeros(n_envs, dtype=torch.bool, device=p.device)
        best = torch.full((n_envs,), float("inf"), device=p.device)

        for _ in range(5):
            sim.plate.set_dofs_position(sim._horizontal_dof_fix,
                                        dofs_idx_local=sim._horizontal_dofs_local)
            sim._step_scene()
        sync(); t = time.perf_counter()
        for _ in range(n_steps):
            sim.plate.set_dofs_position(sim._horizontal_dof_fix,
                                        dofs_idx_local=sim._horizontal_dofs_local)
            sim._step_scene()
            cur = sim.plate.get_pos()
            d = torch.linalg.norm(cur[:, :2] - p_end[:, :2], axis=1)
            best = torch.where(d < best, d, best)
            reached |= (d < sim._goal_threshold)
            _ = int(reached.sum().item())          # the per-step sync
        sync()
        out["ms_per_sweep_step"] = (time.perf_counter() - t) / n_steps * 1000
        out["ms_control_overhead"] = out["ms_per_sweep_step"] - out["ms_per_raw_step"]
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", nargs="+", type=int, default=[50, 200])
    p.add_argument("--envs", nargs="+", type=int, default=[1, 4])
    p.add_argument("--mcp", nargs="+", type=int, default=[200, 800, 2000])
    p.add_argument("--box-box", nargs="+", type=int, default=[1, 0],
                   help="1 = box_box_detection on (16 contacts/pair), 0 = off (5)")
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cell", nargs=5, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()

    if args.cell is not None:
        n_p, n_e, mcp, bb, n_steps = (int(x) for x in args.cell)
        print("###JSON###" + json.dumps(
            run_cell(n_p, n_e, args.particle_size, mcp, bool(bb), n_steps)))
        return

    rows = []
    for n_p, n_e, mcp, bb in itertools.product(
            args.particles, args.envs, args.mcp, args.box_box):
        print(f"  n_p={n_p:>4} envs={n_e:>3} mcp={mcp:>5} box_box={bb} ...",
              end="", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "tests.scaling_investigation.probe_step_cost", "--cell",
             str(n_p), str(n_e), str(mcp), str(bb), str(args.n_steps),
             "--particle-size", str(args.particle_size)],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT))
        line = next((l for l in proc.stdout.splitlines()
                     if l.startswith("###JSON###")), None)
        if line is None:
            print(" CRASHED")
            rows.append(dict(n_particles=n_p, n_envs=n_e, mcp=mcp, box_box=bool(bb),
                             ok=False, error="crashed"))
            continue
        r = json.loads(line[len("###JSON###"):])
        rows.append(r)
        if r["ok"]:
            print(f" raw {r['ms_per_raw_step']:7.1f} ms  sweep "
                  f"{r['ms_per_sweep_step']:7.1f} ms  overhead "
                  f"{r['ms_control_overhead']:6.1f} ms  vram {r['vram_gib']:.2f} GiB")
        else:
            print(f" FAIL {r.get('error','')[:70]}")

    print("\n### ms per raw scene.step()")
    print(f"{'n_p':>5} {'envs':>5} {'box_box':>8}" +
          "".join(f"{'mcp=' + str(m):>12}" for m in args.mcp))
    for n_p in args.particles:
        for n_e in args.envs:
            for bb in args.box_box:
                line = f"{n_p:>5} {n_e:>5} {bool(bb)!s:>8}"
                for m in args.mcp:
                    r = next((x for x in rows
                              if x["n_particles"] == n_p and x["n_envs"] == n_e
                              and x["mcp"] == m and x["box_box"] == bool(bb)), None)
                    line += (f"{'-':>12}" if r is None
                             else f"{'FAIL':>12}" if not r["ok"]
                             else f"{r['ms_per_raw_step']:>12.1f}")
                print(line)

    print("\n### per-step control+sync overhead of the sweep loop (ms)")
    print(f"{'n_p':>5} {'envs':>5} {'box_box':>8}" +
          "".join(f"{'mcp=' + str(m):>12}" for m in args.mcp))
    for n_p in args.particles:
        for n_e in args.envs:
            for bb in args.box_box:
                line = f"{n_p:>5} {n_e:>5} {bool(bb)!s:>8}"
                for m in args.mcp:
                    r = next((x for x in rows
                              if x["n_particles"] == n_p and x["n_envs"] == n_e
                              and x["mcp"] == m and x["box_box"] == bool(bb)), None)
                    line += (f"{'-':>12}" if r is None
                             else f"{'FAIL':>12}" if not r["ok"]
                             else f"{r['ms_control_overhead']:>12.1f}")
                print(line)

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nraw -> {args.out}")


if __name__ == "__main__":
    main()
