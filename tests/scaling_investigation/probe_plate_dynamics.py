#!/usr/bin/env python3
"""
tests/scaling_investigation/probe_plate_dynamics.py — measure how faithfully the pusher plate
tracks its commanded trajectory, and how much the pile perturbs it.

Motivation
----------
The plate is a free-floating rigid box driven by a 6-DOF PD controller
(``build()`` sets kp=0.8, kv=1.0), with z/roll/pitch/yaw hard-set every step
and x/y left to the controller. Whether that is "stiff enough to stand in for
a robot arm holding a plate" is an empirical question, and it is the one this
probe answers, by comparing three quantities for the same commanded sweep:

  commanded   the straight line p_start -> p_stop at ``plate.speed``
  free        the plate's realized path with NO particles in the box
  loaded      the plate's realized path through an actual pile

  free vs commanded   -> controller tracking error (actuator model quality)
  loaded vs free      -> how much the granular reaction moves the tool
                         (the quantity a stiffly-held real plate would show
                         as ~zero)

It also reports the plate's actual mass and the peak contact-pair usage, so
``max_collision_pairs`` can be set from measurement rather than guesswork.

Usage
-----
From the REPO ROOT (see ``benchmark_n_envs.py`` for why)::

    python -m tests.scaling_investigation.probe_plate_dynamics
    python -m tests.scaling_investigation.probe_plate_dynamics --n-particles 200 --particle-size 0.005
    python -m tests.scaling_investigation.probe_plate_dynamics --density 5000 --armature 0.1
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from Genesis.sandbox_manipulation_clean import SandboxManipulation

# This script lives outside Genesis/, so paths to the simulator's configs are
# resolved explicitly rather than relative to this file.
GENESIS_DIR = Path(__file__).resolve().parents[2] / "Genesis"
REPO_ROOT = Path(__file__).resolve().parents[2]



def _config(n_particles: int, particle_size: float, density: float,
            friction: float, max_collision_pairs: int) -> dict:
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=density,
                           friction=friction)
    cfg["box"]["friction"] = friction
    if max_collision_pairs is not None:
        cfg["rigid_options"]["max_collision_pairs"] = max_collision_pairs
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def _trace_production(sim: SandboxManipulation, p_start, p_stop, angle):
    """Run the real ``plate_velocity_translation`` and record what it did.

    Uses the production method via its ``on_step`` hook rather than a local
    copy of the control law, so this probe cannot drift out of sync with the
    code it is supposed to be measuring.

    Returns (realized_xy, reference_xy, speeds) per step.
    """
    realized, reference, speeds = [], [], []

    def on_step(step, p_ref, v_ref):
        realized.append(sim.plate.get_pos()[0, :2].clone())
        reference.append(p_ref[0, :2].clone())
        speeds.append(torch.linalg.norm(sim.plate.get_dofs_velocity()[0, :2]).clone())

    sim.plate_velocity_translation(p_start, p_stop, angle, on_step=on_step)
    return (torch.stack(realized).cpu().numpy(),
            torch.stack(reference).cpu().numpy(),
            torch.stack(speeds).cpu().numpy())


def _trace_legacy(sim: SandboxManipulation, p_start, p_stop, angle,
                  zero_velocity: bool = True):
    """Reproduce the PREVIOUS control law for comparison.

    The old sweep commanded the *endpoint* as the PD position target and let
    the per-step ``set_dofs_position`` zero all six dofs' velocities, so the
    plate restarted from rest every step. Kept here (and only here) so the two
    actuator models can be compared within one build.
    """
    sim._horizontal_dof_fix[:, -1] = angle
    delta = p_stop - p_start
    dist = torch.linalg.norm(delta, axis=1, keepdim=True)
    v = delta / (dist + 1e-8) * sim._plate_params["speed"]

    sim.plate.set_pos(p_start)
    sim.plate.control_dofs_position_velocity(p_stop, v, dofs_idx_local=[0, 1, 2])

    import math
    n_steps = max(1, math.ceil(float(dist.max().item())
                               / (sim._plate_params["speed"] * sim._scene.dt) * 1.7))
    path, speeds = [], []
    for _ in range(n_steps):
        sim.plate.set_dofs_position(sim._horizontal_dof_fix,
                                    dofs_idx_local=sim._horizontal_dofs_local,
                                    zero_velocity=zero_velocity)
        sim._step_scene()
        path.append(sim.plate.get_pos()[0, :2].clone())
        speeds.append(torch.linalg.norm(
            sim.plate.get_dofs_velocity()[0, :2]).clone())
    return (torch.stack(path).cpu().numpy(),
            torch.stack(speeds).cpu().numpy(), n_steps)


def _commanded_path(p_start, p_stop, speed, dt, n_steps):
    """Where a perfectly-tracking plate would be at each step."""
    start = p_start[0, :2].cpu().numpy()
    stop = p_stop[0, :2].cpu().numpy()
    total = np.linalg.norm(stop - start)
    direction = (stop - start) / (total + 1e-12)
    travelled = np.minimum(speed * dt * np.arange(1, n_steps + 1), total)
    return start[None, :] + direction[None, :] * travelled[:, None]


def _peak_contacts(sim) -> int | None:
    """Best-effort read of how many contact pairs the collider actually used."""
    try:
        collider = sim._scene.rigid_solver.collider
        n = collider._collider_state.n_contacts.to_numpy()
        return int(np.max(n))
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--density", type=float, default=1000.0)
    p.add_argument("--friction", type=float, default=0.3)
    p.add_argument("--max-collision-pairs", type=int, default=None,
                   help="default: the wrapper's n_particles-scaled value. A "
                        "large fixed cap is not free — at n=200 a cap of 2000 "
                        "sizes the constraint Jacobian past this GPU's memory")
    p.add_argument("--armature", type=float, default=None,
                   help="if set, apply set_dofs_armature(<v>) to the plate — "
                        "the candidate fix for a tool that gets pushed around")
    p.add_argument("--kp", type=float, default=None, help="override plate kp")
    p.add_argument("--kv", type=float, default=None, help="override plate kv")
    p.add_argument("--sweep-settle-steps", type=int, default=None,
                   help="steps held at the goal after the reference ends; "
                        "raising it separates servo settling from a genuine "
                        "steady-state offset")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = _config(args.n_particles, args.particle_size, args.density,
                  args.friction, args.max_collision_pairs)

    sim = SandboxManipulation(config=cfg, n_envs=1, debug=False)
    sim.build()

    if args.armature is not None:
        sim.plate.set_dofs_armature((args.armature,) * 6, [0, 1, 2, 3, 4, 5])
    if args.kp is not None:
        sim.plate.set_dofs_kp((args.kp,) * 6, [0, 1, 2, 3, 4, 5])
    if args.kv is not None:
        sim.plate.set_dofs_kv((args.kv,) * 6, [0, 1, 2, 3, 4, 5])
    if args.sweep_settle_steps is not None:
        sim._sweep_settle_steps = args.sweep_settle_steps

    plate_mass = float(sim.plate.get_mass())
    p_size = args.particle_size
    particle_mass = args.density * p_size ** 3
    print("\n=== tool / particle mass ===")
    print(f"  plate mass                 : {plate_mass*1000:.3f} g")
    print(f"  one particle ({p_size*1000:.1f} mm cube) : {particle_mass*1000:.4f} g")
    print(f"  plate / particle mass ratio: {plate_mass/particle_mass:.1f}")
    print(f"  whole pile ({args.n_particles}) mass    : "
          f"{particle_mass*args.n_particles*1000:.2f} g")
    print(f"  plate / pile mass ratio    : "
          f"{plate_mass/(particle_mass*args.n_particles):.2f}")
    if args.armature is not None:
        print(f"  + armature                 : {args.armature} kg per DOF "
              f"(effective inertia {args.armature/plate_mass:.0f}x the plate mass)")

    # A single fixed, fully-crossing sweep so free and loaded are comparable.
    dev = sim._particle_state.device
    # see verify_fixes.py: at yaw=0 the blade reaches 0.02 m beyond the
    # commanded centre, and the wall is at 0.064 m, so 0.045 is unreachable
    p_start = torch.tensor([[-0.030, 0.0, sim._operation_height]], device=dev)
    p_stop = torch.tensor([[0.030, 0.0, sim._operation_height]], device=dev)
    angle = torch.zeros(1, device=dev)

    total = float(np.linalg.norm((p_stop - p_start)[0, :2].cpu().numpy()))
    commanded_speed = sim._plate_params["speed"]

    for label in ("CURRENT (trapezoid reference, gantry actuator)",
                  "LEGACY  (endpoint target, velocity zeroed every step)"):
        is_current = label.startswith("CURRENT")

        def run():
            if is_current:
                path, ref, spd = _trace_production(sim, p_start, p_stop, angle)
                return path, spd, ref
            # Restore the ORIGINAL actuator too, not just the original
            # reference: soft gains, no reflected inertia, no force limit.
            # Otherwise "legacy" would silently inherit the new gantry model
            # and understate how much the old tool was pushed around.
            dofs = [0, 1, 2]
            sim.plate.set_dofs_armature((0.0,) * 3, dofs)
            sim.plate.set_dofs_kp((0.8,) * 3, dofs)
            sim.plate.set_dofs_kv((1.0,) * 3, dofs)
            sim.plate.set_dofs_force_range((-1e9,) * 3, (1e9,) * 3, dofs)
            try:
                path, spd, _n = _trace_legacy(sim, p_start, p_stop, angle)
            finally:
                sim._configure_plate_actuator()
            # legacy commanded a constant speed from t=0
            return path, spd, _commanded_path(
                p_start, p_stop, commanded_speed, sim._scene.dt, len(path))

        # free run: park every particle outside the box
        sim.set_n_active(0)
        sim.shuffle_particles()
        sim.update_material_state()
        free_path, free_v, cmd = run()

        # loaded run: full pile
        sim.set_n_active(args.n_particles)
        sim.shuffle_particles()
        sim.update_material_state()
        loaded_path, loaded_v, _ = run()

        n = min(len(free_path), len(loaded_path), len(cmd))
        free_path, loaded_path, cmd = free_path[:n], loaded_path[:n], cmd[:n]

        # tracking error is measured against the reference the servo was
        # actually given, not against a constant-velocity ramp -- comparing a
        # trapezoid against a ramp would report the acceleration phase as error
        track_err = np.linalg.norm(free_path - cmd, axis=1)
        react_err = np.linalg.norm(loaded_path - free_path, axis=1)

        print(f"\n=== {label} ===")
        print(f"    ({total*1000:.0f} mm commanded at "
              f"{commanded_speed*1000:.0f} mm/s, {n} steps)")
        print(f"  realized speed         : free {free_v.mean()*1000:6.1f} mm/s   "
              f"loaded {loaded_v.mean()*1000:6.1f} mm/s   "
              f"(commanded {commanded_speed*1000:.0f})")
        print(f"  tracking error (free vs commanded)   : "
              f"mean {track_err.mean()*1000:7.2f} mm  max {track_err.max()*1000:7.2f} mm")
        print(f"  reaction displacement (loaded - free): "
              f"mean {react_err.mean()*1000:7.2f} mm  max {react_err.max()*1000:7.2f} mm")
        goal = p_stop[0, :2].cpu().numpy()
        fe_free = np.linalg.norm(free_path[-1] - goal)
        fe_load = np.linalg.norm(loaded_path[-1] - goal)
        thr = sim._goal_threshold
        print(f"  final error vs goal    : free {fe_free*1000:6.2f} mm   "
              f"loaded {fe_load*1000:6.2f} mm   "
              f"(goal_threshold {thr*1000:.1f} mm -> reached_goal="
              f"{bool(fe_load < thr)})")
        print(f"  distance travelled     : free "
              f"{np.linalg.norm(free_path[-1]-free_path[0])*1000:6.1f} mm   "
              f"loaded {np.linalg.norm(loaded_path[-1]-loaded_path[0])*1000:6.1f} mm   "
              f"commanded {total*1000:.1f} mm")

    peak = _peak_contacts(sim)
    print("\n=== collider usage ===")
    print(f"  max_collision_pairs configured: {args.max_collision_pairs}")
    print(f"  peak contacts observed        : "
          f"{peak if peak is not None else 'unavailable'}")

    sim.destroy()


if __name__ == "__main__":
    main()
