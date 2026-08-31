#!/usr/bin/env python3
"""
record_simulation_video.py — render the simulation to video so a human can look
for anything unphysical that the numbers do not catch.

Every measurement in this directory checks a quantity someone thought to ask
about. This one exists for the failures nobody thought to ask about: particles
tunnelling through a wall and reappearing, the pile jittering when it should be
still, the blade dragging a cube it should have pushed past, a second layer
sinking into the first, the tool descending through a particle instead of onto
it.

Records three moments, because they fail differently:

  spawn + settle   the layered drop. Particles are written directly into two or
                   three stacked grids and released, which is the least
                   physically natural thing in the pipeline — a real pile is
                   never assembled that way. Worth watching for interpenetration
                   at t=0 and for the collapse looking like a collapse.
  descent          the tool is teleported to clearance height and then
                   interpolated down. If placement-aware sampling has failed and
                   the blade is coming down onto a cube, this is where it shows.
  push             the sweep itself, frame by frame. The blade should meet the
                   pile, not pass through it, and should not visibly deflect.

Three camera views, each answering a different question:

  bird       top-down. Does the pile stay inside the tray? Is the swath the
             blade cuts the width it should be?
  observer   three-quarter. General plausibility.
  leveled    side-on at pile height. The one that shows layer structure and
             whether the blade rides at the right height through a stack —
             ``operation_height`` puts the blade centre at half a particle, so
             a two-layer pile is engaged by the blade's upper edge, and only a
             side view makes that visible.

Runs at n_envs=1: the rasterizer draws every environment into the same image, so
a batched run renders as a pile of overlapping tray copies.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.record_simulation_video
    python -m tests.scaling_investigation.record_simulation_video \
        --n-particles 200 --n-pushes 3 --views leveled bird
    python -m tests.scaling_investigation.record_simulation_video \
        --n-particles 50 --source library --every 4

Output goes to ``tests/scaling_investigation/results/video/``.
"""

import argparse
import inspect
import math
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GENESIS_DIR = REPO_ROOT / "Genesis"
OUT_DIR = Path(__file__).resolve().parent / "results" / "video"


def _config(n_particles, particle_size):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def camera_poses(box_vol):
    """(pos, lookat, fov) per view, sized from the tray so they stay framed
    when the box dimensions change."""
    w, d, h = box_vol
    return {
        "bird": (( 0.0, -1e-3, 3.2 * w), (0.0, 0.0, 0.0), 30),
        "observer": (( 1.9 * w, -1.9 * w, 1.6 * w), (0.0, 0.0, h / 3), 32),
        # The lowest angle that still clears the near wall. A genuinely level
        # camera sees nothing: the tray walls are h (40 mm) and the pile is one
        # or two 5 mm layers, so the near rim occludes the entire interior. The
        # sight line from here passes over the outer wall face at ~4 mm of
        # clearance, which is as flat as this geometry allows while still
        # showing layer structure and where the blade rides in the stack.
        "leveled": (( 2.0 * w, 0.0, 3.2 * h), (0.0, 0.0, 0.25 * h), 30),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-particles", type=int, default=200)
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--n-pushes", type=int, default=3)
    p.add_argument("--views", nargs="+", default=["leveled", "bird", "observer"],
                   choices=["bird", "observer", "leveled"])
    p.add_argument("--source", choices=["spawn", "library"], default="spawn",
                   help="spawn drops a fresh layered pile and settles it "
                        "(the thing most worth watching); library restores a "
                        "recorded settled state and skips straight to pushes")
    p.add_argument("--library-root", default="data/dry_run")
    p.add_argument("--every", type=int, default=8,
                   help="render one frame per N simulation steps. 8 at dt=4 ms "
                        "gives ~31 fps of real-time playback")
    p.add_argument("--res", type=int, nargs=2, default=[960, 720])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default=None, help="suffix for the output filenames")
    args = p.parse_args()

    import numpy as np
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation
    from Genesis.state_library import StateLibrary, default_library_path

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"n{args.n_particles}_{args.source}"

    cfg = _config(args.n_particles, args.particle_size)
    sim = SandboxManipulation(config=cfg, n_envs=1, debug=False)

    # Cameras must exist before build(); adding one afterwards is not supported.
    poses = camera_poses(sim._box_params["vol"])
    cams = {}
    for view in args.views:
        pos, lookat, fov = poses[view]
        cams[view] = sim._scene.add_camera(res=tuple(args.res), pos=pos,
                                           lookat=lookat, fov=fov, GUI=False)
    sim.build()

    n_frames = 0

    def _render():
        nonlocal n_frames
        for cam in cams.values():
            cam.render()
        n_frames += 1

    def grab_settle(step=None):
        """Frame hook for update_material_state: on_step(step)."""
        if step is None or step % args.every == 0:
            _render()

    def grab_push(phase, step):
        """Frame hook for execute_action: on_step(phase, step).

        Every phase is sampled at the SAME cadence. An earlier version rendered
        the descent and lift at full rate, on the reasoning that a touchdown
        into a particle would show there -- but from Genesis 1.3 the recorder
        streams to disk and drops frames to honour the requested fps against
        simulation time, so those extra frames were silently discarded (206
        rendered, 79 written). Rendering uniformly means the frame rate and the
        requested fps agree and nothing is dropped. Use a smaller --every to see
        the descent in more detail.
        """
        if step % args.every == 0:
            _render()

    # Genesis moved the output arguments from stop_recording to start_recording
    # in 1.3.0. Detect rather than branch on a version string, so this runs on
    # both the pinned 0.4.5 and 1.3.x.
    fps = max(1, round(1.0 / (sim._scene.dt * args.every)))
    _rec_args_at_start = "save_to_filename" in inspect.signature(
        type(next(iter(cams.values()))).start_recording).parameters
    paths = {view: out_dir / f"{view}_{tag}.mp4" for view in cams}

    try:
        for view, cam in cams.items():
            if _rec_args_at_start:
                cam.start_recording(save_to_filename=str(paths[view]), fps=fps)
            else:
                cam.start_recording()

        t0 = time.perf_counter()
        if args.source == "spawn":
            print(f"spawning {args.n_particles} particles ...", flush=True)
            sim.shuffle_particles()
            grab_settle()                            # t=0, before anything falls
            print("settling (recording) ...", flush=True)
            sim.update_material_state(on_step=grab_settle)
        else:
            lib_path = default_library_path(GENESIS_DIR / args.library_root, "cube",
                                            args.n_particles, args.particle_size)
            print(f"restoring settled state from {lib_path}", flush=True)
            StateLibrary.load(lib_path).apply(sim, rng=np.random.default_rng(args.seed))
            sim.update_material_state(on_step=grab_settle)
        print(f"  settle done in {time.perf_counter() - t0:.1f} s, "
              f"{n_frames} frames", flush=True)

        # Real sampled actions, not hand-picked ones: the point is to see what
        # the collection actually does, including any awkward action it draws.
        starts, stops, angles = sim.generate_action_samples(
            args.n_pushes, placement_aware=True, shared_travel_distance=False)

        for i in range(args.n_pushes):
            p_start, p_stop = starts[:, i, :], stops[:, i, :]
            ang = angles[:, i]
            travel = float((p_stop[0, :2] - p_start[0, :2]).norm()) * 1000
            print(f"push {i+1}/{args.n_pushes}: "
                  f"({p_start[0,0]*1000:.0f}, {p_start[0,1]*1000:.0f}) -> "
                  f"({p_stop[0,0]*1000:.0f}, {p_stop[0,1]*1000:.0f}) mm, "
                  f"{travel:.0f} mm, yaw {math.degrees(float(ang[0])):.0f} deg",
                  flush=True)
            sim.execute_action(p_start, p_stop, ang, on_step=grab_push)
            # Settling after the push is part of the recorded transition — if
            # the pile is still creeping when s' is read, it is visible here.
            sim.update_material_state(on_step=grab_settle)

        for view, cam in cams.items():
            if _rec_args_at_start:
                cam.stop_recording()
            else:
                cam.stop_recording(save_to_filename=str(paths[view]), fps=fps)
            print(f"  -> {paths[view]}")
        print(f"\n{n_frames} frames per view at {fps} fps "
              f"(= real time; dt={sim._scene.dt*1000:.0f} ms x {args.every})")
    finally:
        try:
            sim.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
