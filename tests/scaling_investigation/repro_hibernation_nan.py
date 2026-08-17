#!/usr/bin/env python3
"""
repro_hibernation_nan.py — standalone reproduction of a Genesis 1.3.3 failure:
with ``use_hibernation=True``, driving a body with ``set_dofs_position`` produces
``Invalid constraint forces causing 'nan'`` once it contacts a hibernated body.
Driving the same motion with ``set_pos`` instead is clean.

Depends on nothing in this repository — it is written to be pasted into an
upstream issue. Run it directly::

    python tests/scaling_investigation/repro_hibernation_nan.py

Why it matters here: every transition in this project's data collection lowers a
pusher plate onto a settled pile, and that is exactly the motion that fails, so
hibernation cannot be used at all. It is otherwise very attractive — with
hibernation the largest contact island during a push drops from 61 entities to 7
and the step cost falls 57x (see docs/genesis_world_migration.md).

The distinction is the WRITE API, not the motion
------------------------------------------------
Identical trajectory, identical scene, hibernation on:

    set_pos every step                      -> clean
    set_dofs_position every step            -> NaN
    set_pos + set_dofs_position (subset)    -> NaN
    set_pos + set_dofs_position (all 6)     -> NaN

and with ``use_hibernation=False`` every one of those is clean.

Ruled out along the way:

* Not ``enable_torsional_friction`` (fails with it on or off).
* Not the motion profile: waking every step via ``rigid_solver._wake_dofs``,
  dropping ``zero_velocity``, and quartering the step size all still fail.
* Not the actuator: fails with no armature, no PD gains, no force range.
* Not a budget: fails at ``max_collision_pairs`` 150, 1000 and 3000.
* Not scale: fails with 2 cubes as readily as with 50.
* A settle alone never fails, and a free body dropped onto a sleeping pile is
  fine — a body driven by ``set_dofs_position`` is required.

Note the failure is normally INVISIBLE. ``RigidSolver.set_dofs_position`` clears
``_errno``, so a loop that calls it every step wipes the error bit before
``check_errno`` can read it; the exception only surfaces on the first step that
does not call it. This script therefore reads ``_errno`` directly after each
step, which is how the failing step is identified.
"""

import numpy as np
import genesis as gs
from genesis.utils.array_class import ErrorCode
from genesis.utils.misc import qd_to_numpy

DT, SUBSTEPS = 4e-3, 5
CUBE = 0.005                 # 5 mm cubes
TRAY_W, WALL_T, TRAY_H = 0.128, 0.02, 0.04
N_CUBES = 2
SETTLE_STEPS = 200           # long enough for everything to hibernate
DESCENT_STEPS = 25


def errno_bits(scene):
    e = int(np.bitwise_or.reduce(qd_to_numpy(scene.rigid_solver._errno)))
    names = [n for n, c in (("INVALID_CONTACT_NAN", ErrorCode.INVALID_CONTACT_NAN),
                            ("INVALID_FORCE_NAN", ErrorCode.INVALID_FORCE_NAN),
                            ("INVALID_ACC_NAN", ErrorCode.INVALID_ACC_NAN),
                            ("OVERFLOW_HIBERNATION_ISLANDS",
                             ErrorCode.OVERFLOW_HIBERNATION_ISLANDS))
             if e & int(c)]
    return e, names


def build(use_hibernation, with_tray):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,            # required by use_hibernation
            use_hibernation=use_hibernation,
            box_box_detection=True,
            iterations=10, tolerance=1e-4,
            ls_iterations=10, ls_tolerance=0.05,
            max_collision_pairs=150,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    floor_z = WALL_T / 2
    if with_tray:
        # tray floor plus four walls, all fixed
        scene.add_entity(gs.morphs.Box(pos=(0, 0, 0), size=(TRAY_W, TRAY_W, WALL_T),
                                       fixed=True))
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            scene.add_entity(gs.morphs.Box(
                pos=(sx * (TRAY_W + WALL_T) / 2, sy * (TRAY_W + WALL_T) / 2,
                     TRAY_H / 2),
                size=(WALL_T if sx else TRAY_W, WALL_T if sy else TRAY_W, TRAY_H),
                fixed=True))

    cubes = [scene.add_entity(gs.morphs.Box(
        pos=(0.012 * i, 0.0, floor_z + CUBE / 2), size=(CUBE,) * 3))
        for i in range(N_CUBES)]

    # The driven body: a thin blade, moved by set_pos every step (kinematic),
    # which is how a gantry-mounted tool is commonly scripted.
    blade = scene.add_entity(gs.morphs.Box(
        pos=(0.0, 0.0, floor_z + 0.05), size=(0.04, 0.002, 0.01)))

    scene.build(n_envs=1)
    return scene, cubes, blade, floor_z


def run(use_hibernation=True, with_tray=True, pin_dofs=False, verbose=True):
    """pin_dofs mirrors how a gantry tool is usually held: as well as commanding
    the pose with set_pos, the loop re-asserts a SUBSET of the dofs every step
    (here z + orientation) with set_dofs_position, so contact cannot tilt or
    lift the tool. That is two writes to the same body in one step, one of them
    partial."""
    scene, cubes, blade, floor_z = build(use_hibernation, with_tray)

    for _ in range(SETTLE_STEPS):
        scene.step()

    hib = qd_to_numpy(scene.rigid_solver.dyn_state.links.is_hibernated)
    hib = hib.reshape(hib.shape[0], -1)[:, 0]
    if verbose:
        print(f"  after settle: {int(hib.sum())}/{hib.size} links hibernated, "
              f"errno={errno_bits(scene)}")

    # Lower the blade onto the cubes, one interpolated teleport per step.
    z_top = floor_z + 0.05
    z_bot = floor_z + CUBE / 2
    failed_at = None
    for i in range(DESCENT_STEPS):
        z = z_top + (z_bot - z_top) * (i + 1) / DESCENT_STEPS
        if pin_dofs != "dofs_only":
            blade.set_pos(np.array([[0.0, 0.0, z]], dtype=np.float32))
        if pin_dofs == "dofs_only":
            # ONE write: x/y/z and orientation together, no set_pos at all
            blade.set_dofs_position(
                position=np.array([[0.0, 0.0, z, 0.0, 0.0, 0.0]], dtype=np.float32),
                zero_velocity=False)
        if pin_dofs == "partial":
            # re-assert z + orientation only, leaving x/y to the solver
            blade.set_dofs_position(
                position=np.array([[z, 0.0, 0.0, 0.0]], dtype=np.float32),
                dofs_idx_local=[2, 3, 4, 5], zero_velocity=False)
        elif pin_dofs == "full":
            # same intent, but writing ALL six dofs in one call
            blade.set_dofs_position(
                position=np.array([[0.0, 0.0, z, 0.0, 0.0, 0.0]], dtype=np.float32),
                zero_velocity=False)
        scene.step()
        e, names = errno_bits(scene)
        if names and failed_at is None:
            failed_at = (i, DESCENT_STEPS, names)
            if verbose:
                print(f"  FAILED at descent step {i}/{DESCENT_STEPS}: {names}")
            break
    if failed_at is None and verbose:
        print(f"  descent completed clean ({DESCENT_STEPS} steps)")
    return failed_at


if __name__ == "__main__":
    gs.init(backend=gs.cuda, logging_level="error")
    for hib in (False, True):
        for pin in (False, "dofs_only"):   # set_pos only vs set_dofs_position only
            print(f"use_hibernation={hib}, pin_dofs={pin}")
            try:
                run(use_hibernation=hib, with_tray=True, pin_dofs=pin)
            except Exception as exc:                   # noqa: BLE001
                print(f"  raised: {type(exc).__name__}: {str(exc)[:70]}")
