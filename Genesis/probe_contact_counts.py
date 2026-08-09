#!/usr/bin/env python3
"""
Genesis/probe_contact_counts.py — measure how many collision pairs a settled
pile and an active push actually generate, so ``max_collision_pairs`` can be
set from data instead of guessed.

Why this matters
----------------
``RigidOptions.max_collision_pairs`` defaults to 150 in
``sandbox_manipulation_clean.py:208`` and no config overrides it. On overflow,
Genesis's broadphase sets an error bit and **stops adding pairs** — the
remaining contacts are silently dropped, and the simulation continues with
wrong contact physics.

Worse, that error bit is normally surfaced by a periodic ``check_errno()`` at
the start of ``Simulator.step``, but ``RigidSolver.set_dofs_position`` clears
``_errno`` as a side effect — and both ``update_material_state``'s settle loop
and ``plate_velocity_translation``'s sweep loop call it on *every* step. So in
this codebase the overflow is expected to be reported **never**. This probe
therefore reads the contact counters directly rather than relying on Genesis
to complain.

Reported per particle count:
    pairs_settled   contacts with the pile at rest
    pairs_pushing   peak contacts during an actual plate sweep
    n_possible      C(n_geoms, 2)-style upper bound Genesis caps against
    overflowed      whether the configured cap was hit

Usage
-----
From the REPO ROOT::

    python -m Genesis.probe_contact_counts
    python -m Genesis.probe_contact_counts --particles 50 100 200 --cap 6000
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _config(n_particles, particle_size, cap):
    with open(Path(__file__).parent / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg["rigid_options"]["max_collision_pairs"] = cap
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def _read_counts(sim):
    """Peak (broad_pairs, contact_points) across envs, or None."""
    try:
        u = sim.contact_budget_usage()
        return u["broad_pairs"], u["contact_points"]
    except Exception:
        return None


def run_cell(n_particles, particle_size, cap, n_envs):
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    sim = SandboxManipulation(config=_config(n_particles, particle_size, cap),
                              n_envs=n_envs, debug=False)
    sim.build()
    out = dict(n_particles=n_particles, cap=cap, n_envs=n_envs,
               particle_size=particle_size)
    try:
        solver = sim._scene.rigid_solver
        info = solver.collider._collider_info
        out["n_geoms"] = int(len(solver.geoms))
        for k in ("max_collision_pairs", "max_collision_pairs_broad"):
            try:
                out[k] = int(getattr(info, k).to_numpy())
            except Exception:
                pass

        sim.shuffle_particles()
        sim.update_material_state()
        u = sim.contact_budget_usage()
        out.update(n_contacts_per_pair=u["n_contacts_per_pair"],
                   broad_cap=u["broad_cap"], contact_cap=u["contact_cap"])
        c = _read_counts(sim)
        out["broad_settled"], out["points_settled"] = c if c else (None, None)

        # Peak contacts across a full plate sweep straight through the pile.
        dev = sim._particle_state.device
        p_start = torch.tensor([[-0.045, 0.0, sim._operation_height]],
                               device=dev).expand(n_envs, 3).contiguous()
        p_stop = torch.tensor([[0.045, 0.0, sim._operation_height]],
                              device=dev).expand(n_envs, 3).contiguous()
        angle = torch.zeros(n_envs, device=dev)

        peak_broad = peak_pts = 0
        sim._horizontal_dof_fix[:, -1] = 0.0
        delta = p_stop - p_start
        dist = torch.linalg.norm(delta, axis=1, keepdim=True)
        v = delta / (dist + 1e-8) * sim._plate_params["speed"]
        sim.plate.set_pos(p_start)
        sim.plate.control_dofs_position_velocity(p_stop, v, dofs_idx_local=[0, 1, 2])
        n_steps = int(float(dist.max()) / (sim._plate_params["speed"] * sim._scene.dt))
        for _ in range(n_steps):
            sim.plate.set_dofs_position(sim._horizontal_dof_fix,
                                        dofs_idx_local=sim._horizontal_dofs_local)
            sim._step_scene()
            c = _read_counts(sim)
            if c is not None:
                peak_broad = max(peak_broad, c[0])
                peak_pts = max(peak_pts, c[1])
        out["broad_pushing"], out["points_pushing"] = peak_broad, peak_pts
        out["sweep_steps"] = n_steps
        # What mcp would have been needed: mcp bounds broad pairs at mcp*8 and
        # contact points at mcp*n_contacts_per_pair, so the binding requirement
        # is the larger of the two implied minima.
        out["mcp_required"] = int(max(
            math.ceil(peak_broad / 8),
            math.ceil(peak_pts / max(out["n_contacts_per_pair"], 1)),
        ))
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", nargs="+", type=int, default=[50, 70, 100, 150, 200])
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--cap", type=int, default=1500,
                   help="max_collision_pairs to measure under — generous "
                        "enough that nothing overflows, but the cap sizes the "
                        "constraint Jacobian, so a very large value OOMs at "
                        "high n_particles rather than measuring anything")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cell", nargs=1, type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()

    if args.cell is not None:
        print("###JSON###" + json.dumps(
            run_cell(args.cell[0], args.particle_size, args.cap, args.n_envs)))
        return

    rows = []
    for n in args.particles:
        print(f"  n_particles={n:>4} ...", end="", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "Genesis.probe_contact_counts", "--cell", str(n),
             "--particle-size", str(args.particle_size), "--cap", str(args.cap),
             "--n-envs", str(args.n_envs)],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent))
        line = next((l for l in proc.stdout.splitlines()
                     if l.startswith("###JSON###")), None)
        if line is None:
            print(" CRASHED")
            continue
        r = json.loads(line[len("###JSON###"):])
        rows.append(r)
        if r["ok"]:
            print(f" broad {r['broad_settled']}/{r['broad_pushing']}  "
                  f"points {r['points_settled']}/{r['points_pushing']} "
                  f"(settled/pushing)  -> needs mcp >= {r['mcp_required']}")
        else:
            print(f" FAIL {r.get('error','')[:80]}")

    print("\n### collider occupancy (settled / peak during a push)")
    print(f"{'n_p':>5} {'geoms':>6} {'broad pairs':>13} {'contact pts':>13} "
          f"{'mcp needed':>11} {'default 150?':>13}")
    for r in rows:
        if not r.get("ok"):
            continue
        need = r["mcp_required"]
        verdict = "ok" if need <= 150 else f"OVERFLOWS ({need/150:.1f}x)"
        print(f"{r['n_particles']:>5} {r['n_geoms']:>6} "
              f"{str(r['broad_settled']) + '/' + str(r['broad_pushing']):>13} "
              f"{str(r['points_settled']) + '/' + str(r['points_pushing']):>13} "
              f"{need:>11} {verdict:>13}")
    print(f"\n(mcp bounds broad pairs at mcp*8 and contact points at "
          f"mcp*n_contacts_per_pair; n_contacts_per_pair="
          f"{rows[0].get('n_contacts_per_pair') if rows else '?'} here)")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nraw -> {args.out}")


if __name__ == "__main__":
    main()
