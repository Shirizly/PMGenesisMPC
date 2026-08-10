#!/usr/bin/env python3
"""
tests/scaling_investigation/probe_settle_truncation.py — can the settle be truncated early and the
remaining velocity simply zeroed?

The question
------------
Residual motion at 6 s of simulated time is clearly numerical, not granular —
a real pile is dead still in well under a second. That invites an obvious
shortcut: stop the settle early, force every velocity to zero, and record that.
The compute saving is large.

Whether it is *correct* hinges on a distinction that velocity thresholds hide:

  jitter          particles vibrating in place; positions already final.
                  Zeroing velocity is free — you are projecting onto the
                  equilibrium the solver is failing to sit still at.

  real relaxation particles still moving somewhere; positions not yet final.
                  Zeroing velocity freezes a NON-equilibrium configuration.
                  It will start moving again on the next step, so the recorded
                  s' is not a fixed point of the dynamics — and because each
                  transition's s is the previous transition's s', that
                  unfinished motion leaks into the next sample's input.

So the metric is *position* drift, not speed. This probe measures two things.

1. Convergence: how far is the configuration at time t from the fully-settled
   configuration? Reported as a distribution over particles, in mm and as a
   fraction of a particle size, because "max over 6400 particles" is an
   extreme-value statistic and says little about the pile.

2. The proposal itself, directly: restore the checkpoint at time t with
   velocities zeroed (which is exactly what set_particle_state does), then step
   freely and see how far it moves. A truncation point is safe iff the pile
   stays put — and the fully-settled state is measured the same way as a
   baseline, since even a converged pile drifts a little.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_settle_truncation
    python -m tests.scaling_investigation.probe_settle_truncation --n-particles 200 --n-envs 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from Genesis.sandbox_manipulation_clean import SandboxManipulation

# This script lives outside Genesis/, so paths to the simulator's configs are
# resolved explicitly rather than relative to this file.
GENESIS_DIR = Path(__file__).resolve().parents[2] / "Genesis"
REPO_ROOT = Path(__file__).resolve().parents[2]



def _config(n_particles, particle_size, cap):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg["simulation"]["settle_steps"] = cap
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def _stats(disp_m, particle_size):
    """Distribution of per-particle displacement, in mm and particle fractions."""
    d = disp_m.flatten()
    q = torch.tensor([0.5, 0.9, 0.99], device=d.device, dtype=d.dtype)
    med, p90, p99 = (float(x) for x in torch.quantile(d, q))
    mx = float(d.max())
    return {"median_mm": med * 1000, "p90_mm": p90 * 1000, "p99_mm": p99 * 1000,
            "max_mm": mx * 1000, "p99_frac_particle": p99 / particle_size,
            "max_frac_particle": mx / particle_size}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-particles", type=int, default=200)
    ap.add_argument("--particle-size", type=float, default=0.005)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--cap", type=int, default=3000, help="steps to full settle")
    ap.add_argument("--checkpoints", nargs="+", type=int,
                    default=[125, 250, 500, 1000, 1500],
                    help="settle step counts to test truncating at "
                         "(dt=4e-3, so 250 steps = 1 s of simulated time)")
    ap.add_argument("--hold-steps", type=int, default=250,
                    help="steps to run after zeroing velocity, to see whether "
                         "the truncated state actually stays put")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sim = SandboxManipulation(config=_config(args.n_particles, args.particle_size,
                                             args.cap),
                              n_envs=args.n_envs, debug=False)
    out = {"n_particles": args.n_particles, "n_envs": args.n_envs,
           "particle_size": args.particle_size, "checkpoints": {}}
    try:
        sim.build()
        sim.shuffle_particles()
        dt = sim._scene.dt

        # ---- phase 1: settle to the cap, snapshotting on the way -----------
        print(f"\n--- settling {args.n_particles} particles x {args.n_envs} envs "
              f"to {args.cap} steps ({args.cap*dt:.1f} s) ---", flush=True)
        frozen = sim.plate.get_dofs_position()
        snaps = {}
        for step in range(args.cap):
            sim.plate.set_dofs_position(frozen)
            sim._step_scene()
            if (step + 1) in args.checkpoints:
                snaps[step + 1] = (sim._get_particle_positions().clone(),
                                   sim._get_particle_quats().clone())
                lin, ang = sim._pile_motion(quantile=0.995)
                print(f"  t={(step+1)*dt:5.2f}s ({step+1:>4} steps): "
                      f"q99.5 speed {lin*1000:6.2f} mm/s, {ang:5.2f} rad/s",
                      flush=True)
        final_pos = sim._get_particle_positions().clone()
        final_quat = sim._get_particle_quats().clone()
        lin, ang = sim._pile_motion(quantile=0.995)
        print(f"  t={args.cap*dt:5.2f}s (converged ref): q99.5 speed "
              f"{lin*1000:6.2f} mm/s, {ang:5.2f} rad/s", flush=True)

        # ---- phase 2: how far from converged, and does it stay put? --------
        print(f"\n--- is the truncated state already the equilibrium? ---")
        print(f"{'truncate at':>14} {'dist from converged':>34} "
              f"{'moves after zeroing velocity':>34}")
        print(f"{'':>14} {'p50 / p99 / max (mm)':>34} "
              f"{'p50 / p99 / max (mm)':>34}")

        rows = []
        for cp in sorted(snaps):
            pos, quat = snaps[cp]
            conv = _stats((pos - final_pos).norm(dim=-1), args.particle_size)

            # Restore this checkpoint WITH velocities zeroed -- set_particle_state
            # does exactly that, so this is the proposed shortcut verbatim --
            # then let it run and see whether it holds.
            sim.set_particle_state(pos[0:1], quat[0:1])
            before = sim._get_particle_positions().clone()
            for _ in range(args.hold_steps):
                sim._step_scene()
            hold = _stats((sim._get_particle_positions() - before).norm(dim=-1),
                          args.particle_size)

            print(f"{cp:>6} ({cp*dt:4.1f}s) "
                  f"{conv['median_mm']:>9.3f} /{conv['p99_mm']:>8.3f} /"
                  f"{conv['max_mm']:>8.3f}     "
                  f"{hold['median_mm']:>9.3f} /{hold['p99_mm']:>8.3f} /"
                  f"{hold['max_mm']:>8.3f}")
            rows.append({"steps": cp, "seconds": cp * dt,
                         "dist_from_converged": conv, "hold_drift": hold})
            out["checkpoints"][cp] = rows[-1]

        # Baseline: even a fully converged pile moves a little when restarted.
        sim.set_particle_state(final_pos[0:1], final_quat[0:1])
        before = sim._get_particle_positions().clone()
        for _ in range(args.hold_steps):
            sim._step_scene()
        base = _stats((sim._get_particle_positions() - before).norm(dim=-1),
                      args.particle_size)
        print(f"{'converged':>14} {'(reference)':>34}     "
              f"{base['median_mm']:>9.3f} /{base['p99_mm']:>8.3f} /"
              f"{base['max_mm']:>8.3f}")
        out["converged_hold_drift"] = base

        print(f"\n--- verdict ---")
        print(f"  A truncation point is safe if its 'moves after zeroing' is "
              f"comparable to the converged\n  baseline "
              f"(p99 {base['p99_mm']:.3f} mm), and its distance from converged "
              f"is small next to a\n  {args.particle_size*1000:.0f} mm particle.")
        for r in rows:
            safe = (r["hold_drift"]["p99_mm"] <= max(2 * base["p99_mm"], 0.05)
                    and r["dist_from_converged"]["p99_frac_particle"] < 0.05)
            print(f"    t={r['seconds']:4.1f}s: "
                  f"{'SAFE to truncate' if safe else 'not yet settled'} "
                  f"(p99 hold {r['hold_drift']['p99_mm']:.3f} mm, "
                  f"p99 from converged {r['dist_from_converged']['p99_mm']:.3f} mm "
                  f"= {r['dist_from_converged']['p99_frac_particle']*100:.1f}% "
                  f"of a particle)")
    finally:
        try:
            sim.destroy()
        except Exception:
            pass

    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nraw -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
