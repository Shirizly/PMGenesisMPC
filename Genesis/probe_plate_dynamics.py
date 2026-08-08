#!/usr/bin/env python3
"""
Genesis/probe_plate_dynamics.py — measure how faithfully the pusher plate
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

    python -m Genesis.probe_plate_dynamics
    python -m Genesis.probe_plate_dynamics --n-particles 200 --particle-size 0.005
    python -m Genesis.probe_plate_dynamics --density 5000 --armature 0.1
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from Genesis.sandbox_manipulation_clean import SandboxManipulation


def _config(n_particles: int, particle_size: float, density: float,
            friction: float, max_collision_pairs: int) -> dict:
    with open(Path(__file__).parent / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=density,
                           friction=friction)
    cfg["box"]["friction"] = friction
    cfg["rigid_options"]["max_collision_pairs"] = max_collision_pairs
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def _trace_sweep(sim: SandboxManipulation, p_start, p_stop, angle,
                 zero_velocity: bool = False):
    """Run one sweep, recording the plate's realized xy path each step.

    Reimplements ``plate_velocity_translation``'s loop rather than calling it,
    because the per-step plate pose is exactly what we need and the production
    method only returns the endpoint.

    ``zero_velocity=True`` reproduces the pre-fix behaviour, where the per-step
    ``set_dofs_position`` reset all six dofs' velocities and the plate
    restarted from rest every step. Kept so the two actuator models can be
    compared in one build rather than argued about.
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
    p.add_argument("--max-collision-pairs", type=int, default=2000)
    p.add_argument("--armature", type=float, default=None,
                   help="if set, apply set_dofs_armature(<v>) to the plate — "
                        "the candidate fix for a tool that gets pushed around")
    p.add_argument("--kp", type=float, default=None, help="override plate kp")
    p.add_argument("--kv", type=float, default=None, help="override plate kv")
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
    p_start = torch.tensor([[-0.045, 0.0, sim._operation_height]], device=dev)
    p_stop = torch.tensor([[0.045, 0.0, sim._operation_height]], device=dev)
    angle = torch.zeros(1, device=dev)

    total = float(np.linalg.norm((p_stop - p_start)[0, :2].cpu().numpy()))
    commanded_speed = sim._plate_params["speed"]

    for label, zero_v in (("CURRENT (velocity retained)", False),
                          ("LEGACY  (velocity zeroed every step)", True)):
        # free run: park every particle outside the box
        sim.set_n_active(0)
        sim.shuffle_particles()
        sim.update_material_state()
        free_path, free_v, n_steps = _trace_sweep(
            sim, p_start, p_stop, angle, zero_velocity=zero_v)

        # loaded run: full pile
        sim.set_n_active(args.n_particles)
        sim.shuffle_particles()
        sim.update_material_state()
        loaded_path, loaded_v, _ = _trace_sweep(
            sim, p_start, p_stop, angle, zero_velocity=zero_v)

        n = min(len(free_path), len(loaded_path))
        cmd = _commanded_path(p_start, p_stop, commanded_speed, sim._scene.dt, n)
        free_path, loaded_path = free_path[:n], loaded_path[:n]

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
