#!/usr/bin/env python3
"""
tests/scaling_investigation/probe_persistent_movers.py — identify the individual particles that
never stop moving, and characterise what they are doing.

Why
---
Settling statistics say a 200-cube pile keeps moving for seconds, but a
percentile over 200 particles hides *who*. Measured separately
(tests/scaling_investigation/settling_investigation.md §1.6), the median particle is already at its
final position at 0.5 s while the 99th percentile is more than a diameter away —
so the residual is a handful of particles, not the pile. Friction should stop
everything within about a second, so a particle still relocating at 6 s is
either doing something physically interesting or, more likely, is a numerical
artifact. This tells them apart by naming the particles and watching them.

Per snapshot (default every 1 s of simulated time) it reports, for the fastest
particles:

  identity      particle index, and whether it was fast in the previous
                snapshot too — a stable set implicates a localized defect
                (a wedged cube, a bad contact pair); a rotating cast implicates
                a global property (solver noise).
  location      x, y, z; height above the tray floor; clearance to the nearest
                wall. A mover on top of the pile means the two-layer spawn is
                still collapsing; one jammed in a corner or against a wall
                means something else.
  motion type   net displacement vs path length over the window. Path much
                greater than net = vibrating in place (jitter). Path close to
                net = genuinely travelling (creep). This is the distinction
                velocity thresholds cannot make.
  overlap       distance to the nearest neighbour against the sum of their
                inscribed and circumscribed radii. A particle continuously
                overlapping another is being pushed apart by constraint forces
                every step, which is the classic signature of a solver that
                never resolves a contact.

Modes
-----
``--mode respawn``  settle after shuffle_particles() — a fresh two-layer spawn
                    that has to collapse. This is the worst case, and it is what
                    every settling measurement so far has used.
``--mode push``     settle fully first, then execute one plate sweep and watch
                    the settle that follows. This is the case that actually
                    occurs per recorded transition, and it has never been
                    measured. Note the median particle is expected to be at its
                    final position here simply because the sweep never touched
                    it, so read the p50 accordingly.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_persistent_movers --mode respawn
    python -m tests.scaling_investigation.probe_persistent_movers --mode push --n-particles 200
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


def _speeds(sim):
    """(linear m/s, angular rad/s) per particle, shape (n_envs, n_active)."""
    vel = sim._scene.rigid_solver.get_dofs_velocity(
        dofs_idx=sim._particle_dofs_idx).reshape(sim._n_envs, -1, 6)
    n_active = getattr(sim, "_n_active", vel.shape[1])
    vel = vel[:, :n_active]
    return vel[..., :3].norm(dim=-1), vel[..., 3:].norm(dim=-1)


def _neighbour_overlap(pos_env, idx, size):
    """Distance to nearest neighbour, and how it compares to contact radii."""
    d = (pos_env - pos_env[idx]).norm(dim=-1)
    d[idx] = float("inf")
    nearest = int(d.argmin())
    dist = float(d[nearest])
    inscribed = size            # sum of two half-sides
    circumscribed = size * np.sqrt(3)   # sum of two half body-diagonals
    if dist < inscribed:
        verdict = "OVERLAPPING"
    elif dist < circumscribed:
        verdict = "touching"
    else:
        verdict = "free"
    return nearest, dist, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["respawn", "push"], default="respawn")
    ap.add_argument("--n-particles", type=int, default=200)
    ap.add_argument("--particle-size", type=float, default=0.005)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--snapshot-seconds", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--env", type=int, default=0, help="env to report in detail")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cap = int(args.seconds / 4e-3) + 10
    sim = SandboxManipulation(config=_config(args.n_particles, args.particle_size, cap),
                              n_envs=args.n_envs, debug=False)
    out = {"mode": args.mode, "n_particles": args.n_particles,
           "n_envs": args.n_envs, "snapshots": []}
    try:
        sim.build()
        dt = sim._scene.dt
        snap_every = max(1, int(round(args.snapshot_seconds / dt)))
        n_steps = int(round(args.seconds / dt))
        size = args.particle_size
        floor_z = sim._wall_thickness / 2
        half_box = sim._box_params["vol"][0] / 2

        sim.shuffle_particles()
        if args.mode == "push":
            print("\n--- settling the spawn fully before the push ---", flush=True)
            sim.update_material_state()
            dev = sim._particle_state.device
            p_start = torch.tensor([[-0.030, 0.0, sim._operation_height]],
                                   device=dev).expand(args.n_envs, 3).contiguous()
            p_stop = torch.tensor([[0.030, 0.0, sim._operation_height]],
                                  device=dev).expand(args.n_envs, 3).contiguous()
            print("--- executing one sweep, then watching the settle ---", flush=True)
            sim.execute_action(p_start, p_stop, torch.zeros(args.n_envs, device=dev))
            frozen = sim.plate.get_dofs_position()
            sim.plate.zero_all_dofs_velocity()
            sim.plate.control_dofs_position_velocity(
                frozen, torch.zeros_like(frozen), dofs_idx_local=[0, 1, 2, 3, 4, 5])
        else:
            print("\n--- watching the settle after a fresh respawn ---", flush=True)
            frozen = sim.plate.get_dofs_position()
            sim.plate.zero_all_dofs_velocity()
            sim.plate.control_dofs_position_velocity(
                frozen, torch.zeros_like(frozen), dofs_idx_local=[0, 1, 2, 3, 4, 5])

        e = args.env
        prev_pos = sim._get_particle_positions().clone()
        window_start = prev_pos.clone()
        path = torch.zeros(prev_pos.shape[:2], device=prev_pos.device)
        prev_fast = set()

        for step in range(n_steps):
            sim._step_scene()
            pos = sim._get_particle_positions()
            path += (pos - prev_pos).norm(dim=-1)     # accumulated on GPU, no sync
            prev_pos = pos.clone()

            if (step + 1) % snap_every != 0:
                continue

            t = (step + 1) * dt
            lin, ang = _speeds(sim)
            net = (pos - window_start).norm(dim=-1)
            order = torch.argsort(lin[e], descending=True)[:args.top_k]

            print(f"\n=== t = {t:.1f} s "
                  f"(env {e}, {args.n_particles} particles) ===")
            print(f"  pile: q99.5 lin {float(torch.quantile(lin.flatten(), 0.995))*1000:6.2f} mm/s, "
                  f"median lin {float(lin.median())*1000:6.3f} mm/s, "
                  f"{int((lin > 1e-3).sum())}/{lin.numel()} particles above 1 mm/s")
            print(f"  {'idx':>5} {'lin mm/s':>9} {'ang rad/s':>10} "
                  f"{'z-floor mm':>11} {'wall gap mm':>12} "
                  f"{'net mm':>8} {'path mm':>9} {'motion':>7} "
                  f"{'nearest':>9} {'contact':>12} {'was fast':>9}")
            snap = {"t": t, "movers": []}
            fast_now = set()
            for j in order:
                j = int(j)
                if float(lin[e, j]) < 1e-4 and float(ang[e, j]) < 1e-2:
                    continue
                fast_now.add(j)
                p = pos[e, j]
                z_above = float(p[2]) - floor_z
                wall_gap = half_box - max(abs(float(p[0])), abs(float(p[1])))
                nj, ndist, verdict = _neighbour_overlap(pos[e], j, size)
                net_j, path_j = float(net[e, j]), float(path[e, j])
                motion = ("still" if path_j < 1e-4 else
                          "jitter" if net_j < 0.25 * path_j else "creep")
                print(f"  {j:>5} {float(lin[e,j])*1000:>9.2f} {float(ang[e,j]):>10.3f} "
                      f"{z_above*1000:>11.2f} {wall_gap*1000:>12.2f} "
                      f"{net_j*1000:>8.2f} {path_j*1000:>9.2f} {motion:>7} "
                      f"{nj:>9} {verdict + ' ' + format(ndist*1000, '.1f'):>12} "
                      f"{'yes' if j in prev_fast else '-':>9}")
                snap["movers"].append(
                    {"idx": j, "lin_mms": float(lin[e, j]) * 1000,
                     "ang": float(ang[e, j]), "z_above_floor_mm": z_above * 1000,
                     "wall_gap_mm": wall_gap * 1000, "net_mm": net_j * 1000,
                     "path_mm": path_j * 1000, "motion": motion,
                     "nearest_idx": nj, "nearest_mm": ndist * 1000,
                     "contact": verdict, "was_fast_before": j in prev_fast})
            if not snap["movers"]:
                print("   (nothing above the reporting floor — pile is at rest)")
            persist = (len(fast_now & prev_fast) / max(len(fast_now), 1)) if prev_fast else None
            if persist is not None:
                print(f"  identity persistence vs previous snapshot: "
                      f"{len(fast_now & prev_fast)}/{len(fast_now)} "
                      f"({persist*100:.0f}%) — "
                      f"{'same particles, localized defect' if persist > 0.5 else 'different particles each time'}")
                snap["identity_persistence"] = persist
            prev_fast = fast_now
            out["snapshots"].append(snap)

            window_start = pos.clone()
            path.zero_()
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
