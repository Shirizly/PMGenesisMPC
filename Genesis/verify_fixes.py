#!/usr/bin/env python3
"""
Genesis/verify_fixes.py — end-to-end verification of the correctness fixes made
to ``sandbox_manipulation_clean.py`` while scaling data collection to 200
objects.

Each check asserts an observable property of a live Genesis scene, so it fails
loudly if a fix regresses. Run from the REPO ROOT::

    python -m Genesis.verify_fixes
    python -m Genesis.verify_fixes --n-particles 200 --particle-size 0.005

Checks
------
1.  ``dt``/``substeps`` fallbacks resolve to 4e-3 / 5, not 4e3 / 1
2.  ``max_collision_pairs`` default scales with the particle count
3.  the plate has an explicit friction (was pinned at Genesis' default 1.0,
    which masked every sampled particle friction at the tool interface)
4.  particle mass matches the *configured* density (was 0.8x it)
5.  200 particles can be placed at all (layered spawn)
6.  batched pose writes round-trip exactly
7.  the plate retains x/y velocity through a sweep (was zeroed every step)
8.  parked particles are spread out, not heaped on one point
9.  contact-pair usage is within the configured budget
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from Genesis.sandbox_manipulation_clean import SandboxManipulation

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"{PASS if ok else FAIL}  {name}" + (f"  --  {detail}" if detail else ""),
          flush=True)


def _config(n_particles, particle_size, drop_timing=False):
    with open(Path(__file__).parent / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    if drop_timing:
        # exercise the fallbacks rather than the configured values
        cfg["simulation"].pop("dt", None)
        cfg["simulation"].pop("substeps", None)
    return cfg


def check_timing_defaults(particle_size):
    """Check 1 — the dt fallback used to be 4e3 (4000 seconds)."""
    sim = SandboxManipulation(config=_config(8, particle_size, drop_timing=True),
                              n_envs=1, debug=False)
    try:
        sim.build()
        dt, sub = sim._scene.dt, sim._scene.sim.substeps
        check("dt fallback is 4e-3 (not 4e3)", abs(dt - 4e-3) < 1e-9, f"dt={dt}")
        check("substeps fallback is 5 (not 1)", sub == 5, f"substeps={sub}")
    finally:
        sim.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-particles", type=int, default=200)
    ap.add_argument("--particle-size", type=float, default=0.005)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--skip-timing-check", action="store_true",
                    help="skip check 1, which needs its own scene build")
    args = ap.parse_args()

    n, s = args.n_particles, args.particle_size

    if not args.skip_timing_check:
        print("\n--- timing fallbacks (small scene) ---")
        check_timing_defaults(s)

    print(f"\n--- main scene: n_particles={n}, size={s*1000:.1f} mm ---")
    sim = SandboxManipulation(config=_config(n, s), n_envs=2, debug=False)
    try:
        sim.build()

        # 2 -- contact budget default scales with the pile
        cap = sim._scene.rigid_solver.max_collision_pairs
        check("max_collision_pairs scales with n_particles",
              cap >= 4 * n, f"cap={cap} (Genesis default would be 150)")

        # 3 -- plate friction is explicit, not Genesis' default 1.0
        pf = float(sim.plate.geoms[0].friction)
        check("plate friction is set explicitly (not 1.0)",
              abs(pf - 0.3) < 1e-6, f"mu_plate={pf}")

        # 5 -- placement at this particle count works at all
        sim.shuffle_particles()
        sim.update_material_state()
        pos = sim._particle_state[:, :, 0:3]
        inside = ((pos[..., 0].abs() < 0.075) & (pos[..., 1].abs() < 0.075)).all()
        check(f"{n} particles placed inside the box", bool(inside),
              f"z range {float(pos[...,2].min()):.4f}..{float(pos[...,2].max()):.4f} m")

        # 4 -- mass reflects the configured density
        sim.set_material_properties(dict(particle_friction=0.3,
                                         sampled_particle_friction=None,
                                         particle_density=args.density,
                                         sampled_particle_density=None,
                                         box_friction=0.3))
        expected = args.density * s ** 3
        got = float(sim.material[0].get_mass())
        check("particle mass matches configured density",
              abs(got - expected) / expected < 0.02,
              f"expected {expected*1e6:.2f} mg, got {got*1e6:.2f} mg "
              f"(ratio {got/expected:.3f}; pre-fix was 0.800)")

        # 6 -- batched pose writes round-trip
        ref_pos = sim._particle_state[0:1, :, 0:3].clone()
        ref_quat = sim._particle_state[0:1, :, 3:7].clone()
        sim.set_particle_state(ref_pos, ref_quat)
        back_pos = sim._get_particle_positions()
        err = float((back_pos - ref_pos.expand_as(back_pos)).abs().max())
        check("batched pose write round-trips", err < 1e-5,
              f"max position error {err:.2e} m")

        # 8 -- parked particles are spread, not heaped
        keep = max(1, n // 2)
        sim.set_n_active(keep)
        sim.shuffle_particles()
        parked = sim._get_particle_positions()[0, keep:, :2]
        if parked.shape[0] > 1:
            spread = float(torch.cdist(parked, parked).max())
            check("parked particles are spread out", spread > 1e-3,
                  f"{parked.shape[0]} parked, max separation {spread*1000:.1f} mm "
                  f"(pre-fix: 0.0)")
        sim.set_n_active(n)
        sim.shuffle_particles()
        sim.update_material_state()

        # 7 -- the plate keeps momentum through a sweep
        dev = sim._particle_state.device
        p_start = torch.tensor([[-0.045, 0.0, sim._operation_height]],
                               device=dev).expand(2, 3).contiguous()
        p_stop = torch.tensor([[0.045, 0.0, sim._operation_height]],
                              device=dev).expand(2, 3).contiguous()
        angle = torch.zeros(2, device=dev)

        sim._horizontal_dof_fix[:, -1] = 0.0
        delta = p_stop - p_start
        dist = torch.linalg.norm(delta, axis=1, keepdim=True)
        v = delta / (dist + 1e-8) * sim._plate_params["speed"]
        sim.plate.set_pos(p_start)
        sim.plate.control_dofs_position_velocity(p_stop, v, dofs_idx_local=[0, 1, 2])
        speeds = []
        for _ in range(40):
            sim.plate.set_dofs_position(sim._horizontal_dof_fix,
                                        dofs_idx_local=sim._horizontal_dofs_local,
                                        zero_velocity=False)
            sim._step_scene()
            speeds.append(float(torch.linalg.norm(
                sim.plate.get_dofs_velocity()[0, :2])))
        mean_speed = float(np.mean(speeds))
        check("plate retains x/y velocity during sweep", mean_speed > 1e-3,
              f"mean |v_xy| = {mean_speed*1000:.2f} mm/s "
              f"(commanded {sim._plate_params['speed']*1000:.0f} mm/s; "
              f"pre-fix this was 0 at the start of every step)")

        # 9 -- contact usage within budget
        try:
            st = sim._scene.rigid_solver.collider._collider_state
            peak = int(torch.as_tensor(st.n_contacts.to_torch()).max())
            check("contact usage within configured budget", peak < cap,
                  f"peak {peak} pairs vs cap {cap} "
                  f"(Genesis default of 150 would {'OVERFLOW' if peak >= 150 else 'hold'})")
        except Exception as e:
            check("contact usage within configured budget", False, f"unreadable: {e}")

        # a real action still runs end to end
        sim.execute_action(p_start, p_stop, angle)
        sim.update_material_state()
        check("execute_action + settle completes", True)

    finally:
        sim.destroy()

    n_pass = sum(1 for _, ok in _results if ok)
    print(f"\n=== {n_pass}/{len(_results)} checks passed ===")
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
