#!/usr/bin/env python3
"""
tests/scaling_investigation/probe_settle_convergence.py — how many simulation steps a pile actually
needs to come to rest, as a function of particle count.

Why
---
``update_material_state`` used to run a fixed ``settle_steps`` (default 100) and
then read the state, with no check that anything had stopped moving. Measured at
200 cubes of 5 mm, the pile is still moving at ~60 mm/s linear and ~18 rad/s
angular when those 100 steps run out. That matters more than it looks: each
transition's ``s`` is the previous transition's ``s'``, so a state read
mid-motion contaminates the next sample too.

``settle_steps`` is now a *cap* and the settle exits early once the pile is at
rest. This probe measures where that early exit actually lands, so the cap can
be set from data instead of habit.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_settle_convergence
    python -m tests.scaling_investigation.probe_settle_convergence --particles 50 200 --cap 3000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

# This script lives outside Genesis/, so paths to the simulator's configs are
# resolved explicitly rather than relative to this file.
GENESIS_DIR = Path(__file__).resolve().parents[2] / "Genesis"
REPO_ROOT = Path(__file__).resolve().parents[2]



def _config(n_particles, particle_size, cap, check_every):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg["simulation"]["settle_steps"] = cap
    cfg["simulation"]["settle_check_every"] = check_every
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def run_cell(n_particles, particle_size, cap, check_every, n_envs, n_trials):
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    sim = SandboxManipulation(
        config=_config(n_particles, particle_size, cap, check_every),
        n_envs=n_envs, debug=False)
    out = {"n_particles": n_particles, "cap": cap, "n_envs": n_envs}
    try:
        sim.build()
        steps, times, lins, angs = [], [], [], []
        for _ in range(n_trials):
            sim.shuffle_particles()
            # step manually so the convergence point can be observed directly
            plate_frozen = sim.plate.get_dofs_position()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            reached = cap
            for step in range(cap):
                sim.plate.set_dofs_position(plate_frozen)
                sim._step_scene()
                if (step + 1) % check_every == 0 and sim._pile_is_at_rest():
                    reached = step + 1
                    break
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            steps.append(reached)
            lin, ang = sim._pile_motion()
            lins.append(lin); angs.append(ang)
        out.update(steps_to_rest=steps, seconds=times,
                   final_linear=lins, final_angular=angs,
                   converged=[s < cap for s in steps], ok=True)
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {str(e)[:120]}")
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", nargs="+", type=int, default=[50, 100, 200])
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--cap", type=int, default=3000,
                   help="hard step limit while measuring (well above the "
                        "production default of 100)")
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cell", nargs=1, type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cell is not None:
        print("###JSON###" + json.dumps(run_cell(
            args.cell[0], args.particle_size, args.cap, args.check_every,
            args.n_envs, args.n_trials)))
        return

    rows = []
    for n in args.particles:
        print(f"  n_particles={n:>4} ...", end="", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "tests.scaling_investigation.probe_settle_convergence", "--cell", str(n),
             "--particle-size", str(args.particle_size), "--cap", str(args.cap),
             "--check-every", str(args.check_every), "--n-envs", str(args.n_envs),
             "--n-trials", str(args.n_trials)],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT))
        line = next((l for l in proc.stdout.splitlines()
                     if l.startswith("###JSON###")), None)
        if line is None:
            print(" CRASHED")
            continue
        r = json.loads(line[len("###JSON###"):])
        rows.append(r)
        if r["ok"]:
            print(f" steps to rest {r['steps_to_rest']}  "
                  f"({'converged' if all(r['converged']) else 'HIT CAP'})")
        else:
            print(f" FAIL {r.get('error','')[:70]}")

    print("\n### steps needed to reach rest (production default is 100)")
    print(f"{'n_p':>5} {'median':>8} {'max':>6} {'seconds':>9} {'vs 100':>18}")
    for r in rows:
        if not r.get("ok"):
            continue
        steps = sorted(r["steps_to_rest"])
        med = steps[len(steps) // 2]
        verdict = "ok" if max(steps) <= 100 else f"NEEDS {max(steps)} ({max(steps)/100:.1f}x)"
        print(f"{r['n_particles']:>5} {med:>8} {max(steps):>6} "
              f"{sum(r['seconds'])/len(r['seconds']):>9.2f} {verdict:>18}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nraw -> {args.out}")


if __name__ == "__main__":
    main()
