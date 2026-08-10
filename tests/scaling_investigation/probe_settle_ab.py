#!/usr/bin/env python3
"""
tests/scaling_investigation/probe_settle_ab.py — isolate what causes the settling noise floor.

The finding this exists to explain (tests/scaling_investigation/settling_investigation.md §1.3): a
200-cube pile's motion drops 16x by 2 s of simulated time and then stops
decreasing, fluctuating around ~1.2 mm/s and ~0.3 rad/s out to 12 s. That is a
persistent excitation the solver re-injects, not slow physical settling.

Two suspects, both testable by changing one thing at a time:

  warm-start discard   Holding the plate with a per-step ``set_dofs_position``
                       calls ``collider.reset()`` and
                       ``constraint_solver.reset()`` on EVERY settle step,
                       throwing away the constraint solver's warm start. The
                       plate is out of contact during a settle and its PD holds
                       it to a 5.3 um sag, so the teleport buys nothing.

  solver budget        ``iterations: 10, tolerance: 1e-4`` against Genesis
                       defaults of 50 and 1e-6. A contact solve restarted from
                       scratch each step with a fifth of the default budget has
                       little chance of converging.

Every variant settles the SAME initial arrangement — the spawn is snapshotted
once and restored before each run — so differences are attributable to the
variant and not to a different random pile.

``iterations`` is a build-time RigidOptions field, so each variant runs in its
own subprocess, seeded identically.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_settle_ab
    python -m tests.scaling_investigation.probe_settle_ab --n-particles 200 --n-envs 8 --steps 3000
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


VARIANTS = {
    "baseline":      dict(hold_by_control=False, iterations=10),
    "pd_hold":       dict(hold_by_control=True,  iterations=10),
    "more_solver":   dict(hold_by_control=False, iterations=50),
    "both":          dict(hold_by_control=True,  iterations=50),
}


def _config(n_particles, particle_size, steps, hold_by_control, iterations):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg["simulation"]["settle_steps"] = steps
    cfg["rigid_options"]["iterations"] = int(iterations)
    cfg["rigid_options"]["ls_iterations"] = int(iterations)
    if iterations >= 50:
        cfg["rigid_options"]["tolerance"] = 1e-6
        cfg["rigid_options"]["ls_tolerance"] = 0.01
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def run_variant(name, n_particles, particle_size, n_envs, steps, seed, checkpoints):
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    v = VARIANTS[name]
    torch.manual_seed(seed)

    sim = SandboxManipulation(
        config=_config(n_particles, particle_size, steps, **v),
        n_envs=n_envs, debug=False)
    out = {"variant": name, **v, "n_particles": n_particles, "n_envs": n_envs}
    try:
        sim.build()
        torch.manual_seed(seed)          # re-seed so the spawn matches exactly
        sim.shuffle_particles()

        # Snapshot the spawn so every variant settles an identical arrangement.
        spawn_pos = sim._get_particle_positions()[0:1].clone()
        spawn_quat = sim._get_particle_quats()[0:1].clone()
        sim.set_particle_state(spawn_pos, spawn_quat)
        out["spawn_checksum"] = float(spawn_pos.sum())

        frozen = sim.plate.get_dofs_position()
        sim.plate.zero_all_dofs_velocity()
        sim.plate.control_dofs_position_velocity(
            frozen, torch.zeros_like(frozen), dofs_idx_local=[0, 1, 2, 3, 4, 5])

        curve, settled_at = [], None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for step in range(steps):
            if not v["hold_by_control"]:
                sim.plate.set_dofs_position(frozen)
            sim._step_scene()
            if (step + 1) in checkpoints:
                lin, ang = sim._pile_motion(quantile=0.995)
                lin_max, ang_max = sim._pile_motion()
                curve.append({"step": step + 1, "t": (step + 1) * sim._scene.dt,
                              "q995_lin_mms": lin * 1000, "q995_ang": ang,
                              "max_lin_mms": lin_max * 1000, "max_ang": ang_max})
            if settled_at is None and (step + 1) % 10 == 0 and sim._pile_is_at_rest():
                settled_at = step + 1
        torch.cuda.synchronize()
        out["seconds"] = time.perf_counter() - t0
        out["curve"] = curve
        out["settled_at"] = settled_at
        out["ms_per_step"] = out["seconds"] / steps * 1000
        out["ok"] = True
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {str(e)[:150]}")
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-particles", type=int, default=200)
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--checkpoints", nargs="+", type=int,
                   default=[125, 250, 500, 1000, 1500, 2000, 3000])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.variant is not None:
        print("###JSON###" + json.dumps(run_variant(
            args.variant, args.n_particles, args.particle_size, args.n_envs,
            args.steps, args.seed, set(args.checkpoints))))
        return

    rows = []
    for name in args.variants:
        print(f"  {name:>12} (hold_by_control={VARIANTS[name]['hold_by_control']}, "
              f"iterations={VARIANTS[name]['iterations']}) ...", end="", flush=True)
        cmd = [sys.executable, "-m", "tests.scaling_investigation.probe_settle_ab", "--variant", name,
               "--n-particles", str(args.n_particles),
               "--particle-size", str(args.particle_size),
               "--n-envs", str(args.n_envs), "--steps", str(args.steps),
               "--seed", str(args.seed), "--checkpoints",
               *[str(c) for c in args.checkpoints]]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(REPO_ROOT))
        line = next((l for l in proc.stdout.splitlines()
                     if l.startswith("###JSON###")), None)
        if line is None:
            print(" CRASHED")
            print("    " + (proc.stderr.strip().splitlines() or ["no output"])[-1][:160])
            continue
        r = json.loads(line[len("###JSON###"):])
        rows.append(r)
        if r["ok"]:
            print(f" {r['ms_per_step']:.1f} ms/step, settled_at="
                  f"{r['settled_at']}, {r['seconds']:.0f} s")
        else:
            print(f" FAIL {r.get('error','')[:80]}")

    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return 1

    print("\n### q99.5 particle speed vs simulated time (mm/s linear)")
    cps = [c["step"] for c in ok[0]["curve"]]
    print(f"{'variant':>12}" + "".join(f"{c*4e-3:>8.1f}s" for c in cps))
    for r in ok:
        print(f"{r['variant']:>12}" +
              "".join(f"{c['q995_lin_mms']:>9.2f}" for c in r["curve"]))

    print("\n### q99.5 angular (rad/s)")
    print(f"{'variant':>12}" + "".join(f"{c*4e-3:>8.1f}s" for c in cps))
    for r in ok:
        print(f"{r['variant']:>12}" +
              "".join(f"{c['q995_ang']:>9.3f}" for c in r["curve"]))

    print("\n### worst single particle at the end (the sustained-motion outliers)")
    print(f"{'variant':>12} {'max linear mm/s':>17} {'max angular rad/s':>19}")
    for r in ok:
        last = r["curve"][-1]
        print(f"{r['variant']:>12} {last['max_lin_mms']:>17.1f} {last['max_ang']:>19.2f}")

    print("\n### cost and convergence")
    print(f"{'variant':>12} {'ms/step':>9} {'settled at':>12} {'total s':>9}")
    base = next((r for r in ok if r["variant"] == "baseline"), None)
    for r in ok:
        rel = ""
        if base and base["ms_per_step"]:
            rel = f"  ({r['ms_per_step']/base['ms_per_step']:.2f}x baseline)"
        print(f"{r['variant']:>12} {r['ms_per_step']:>9.1f} "
              f"{str(r['settled_at']):>12} {r['seconds']:>9.0f}{rel}")

    print("\n### verdict")
    if base:
        b_end = base["curve"][-1]["q995_lin_mms"]
        for r in ok:
            if r["variant"] == "baseline":
                continue
            end = r["curve"][-1]["q995_lin_mms"]
            print(f"  {r['variant']:>12}: floor {end:.2f} vs {b_end:.2f} mm/s "
                  f"({end/max(b_end,1e-9):.2f}x) — "
                  f"{'LOWER, mechanism implicated' if end < 0.7*b_end else 'no clear improvement'}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nraw -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
