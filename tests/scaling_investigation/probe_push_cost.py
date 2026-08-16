#!/usr/bin/env python3
"""
Genesis/probe_push_cost.py — why does a PUSH step cost far more than a SETTLE
step, and why does it scale so badly with n_envs?

The observation
---------------
At 200 particles and 32 envs, one sweep step was measured at ~120 s (py-spy
`step: 88` of `sweep_steps: 229` after ~3.5 h), against ~1.4 s for a settle step
at the same particle count and env count. Meanwhile settling batches almost
perfectly: a 100-step settle at n=200 took 40.05 s at 1 env, 44.36 s at 2 and
53.72 s at 4 — 4x the work for 1.34x the time. So the simulator *can* batch, and
an 85x gap between two kinds of step is structural, not saturation.

Leading hypothesis: **contact-island percolation.** With
``use_contact_island: True`` the constraint solver partitions constraints into
independent islands. While settling that is ~n_particles tiny islands (each cube
on the floor) — cheap and parallel. During a push the plate touches particles
that touch other particles; in a dense two-layer pile the contact graph
percolates and collapses into one island spanning the whole pile. If the
per-island solve is superlinear in island size, that is a large factor on its
own, and it multiplies with the solver iteration count.

What this measures
------------------
For each variant, inside a single build:

  settle ms/step   plate lifted clear, pile at rest — the cheap regime
  push ms/step     plate driven through the pile — the expensive regime
  ratio            push / settle. **This is the number of interest.** It
                   isolates the contact regime from everything that scales with
                   env count or particle count, because both are measured in the
                   same build. If the ratio grows with n_envs, islands are
                   implicated; if it is flat, the problem is ordinary saturation
                   and the answer is simply "use fewer envs".
  contacts         broad-phase pairs and contact points during the push, against
                   their caps — a push that overflows max_collision_pairs is
                   both wrong and potentially slow.

Knobs swept (all build-time, hence a subprocess per variant):
  n_envs, iterations, use_contact_island, box_box_detection

``box_box_detection`` is included because it sets contacts-per-pair to 16 rather
than 5, i.e. it multiplies the constraint row count by ~3.2 — a plausible second
contributor that costs nothing to test alongside.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_push_cost
    python -m tests.scaling_investigation.probe_push_cost --envs 1 4 16 --steps 5
    python -m tests.scaling_investigation.probe_push_cost --variants baseline islands_off
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

GENESIS_DIR = Path(__file__).resolve().parents[2] / "Genesis"
REPO_ROOT = Path(__file__).resolve().parents[2]

# name -> rigid_options overrides.
#
# RESULTS (200 cubes, broadside push, library-seeded) — recorded so these are
# not re-run blindly:
#   baseline (as shipped)        770 ms/step at 1 env, 1358 at 4
#   iters50   (50 solver iters)  1.6-1.8x SLOWER than 10 — reverted, see basic.yaml
#   islands_off                  844734 ms/step (~1100x slower). Never turn off.
#   boxbox_off                   CUDA illegal memory access — not a Genesis bug:
#                                it cuts n_contacts_per_pair 16->5, so the
#                                contact-point cap drops 2400->750 while the
#                                pile needs ~826. Needs max_collision_pairs>=166.
VARIANTS = {
    "baseline":      {},                                     # as shipped
    "iters50":       dict(iterations=50, ls_iterations=50,
                          tolerance=1e-6, ls_tolerance=0.01),
    "islands_off":   dict(use_contact_island=False),
    # only meaningful with max_collision_pairs raised to >=166; left here so the
    # crash is reproducible rather than mysterious
    "boxbox_off":    dict(box_box_detection=False),
    "boxbox_off_bigcap": dict(box_box_detection=False, max_collision_pairs=400),
}


def _config(n_particles, particle_size, overrides):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg["rigid_options"].update(overrides)
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def run_variant(name, n_particles, particle_size, n_envs, steps, seed,
                warmup=30, yaw=None, library_root=None):
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation
    from Genesis.state_library import StateLibrary, default_library_path

    torch.manual_seed(seed)
    sim = SandboxManipulation(config=_config(n_particles, particle_size,
                                             VARIANTS[name]),
                              n_envs=n_envs, debug=False)
    out = {"variant": name, "n_envs": n_envs, "n_particles": n_particles,
           **VARIANTS[name]}
    try:
        sim.build()
        torch.manual_seed(seed)

        # Seed the pile from a recorded settled-state library rather than
        # settling a fresh one. Settling a 200-particle two-layer respawn takes
        # ~1500 steps and dominates this probe's runtime, while contributing
        # nothing to what it measures -- every variant only needs to start from
        # the same at-rest pile. Falls back to shuffle+settle if no compatible
        # library exists.
        seeded = False
        lib_path = (default_library_path(library_root, "cube", n_particles,
                                         particle_size)
                    if library_root else None)
        if lib_path is not None and lib_path.exists():
            try:
                lib = StateLibrary.load(lib_path)
                if lib.n_particles == len(sim.material):
                    # One shared state across envs, matching collection.
                    import numpy as _np
                    lib.apply(sim, rng=_np.random.default_rng(seed))
                    sim._particle_state[:, :, 0:3] = sim._get_particle_positions()
                    sim._particle_state[:, :, 3:] = sim._get_particle_quats()
                    seeded = True
                    out["seeded_from"] = str(lib_path)
            except Exception as e:
                out["library_error"] = str(e)[:120]
        if not seeded:
            sim.shuffle_particles()
            sim.update_material_state()      # reach the at-rest regime
        out["seeded_from_library"] = seeded

        # ---- settle step: plate lifted clear, pile quiet -----------------
        frozen = sim.plate.get_dofs_position()
        sim.plate.zero_all_dofs_velocity()
        sim.plate.control_dofs_position_velocity(
            frozen, torch.zeros_like(frozen), dofs_idx_local=[0, 1, 2, 3, 4, 5])
        for _ in range(3):
            sim._step_scene()                # warm up
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(steps):
            sim._step_scene()
        torch.cuda.synchronize()
        out["settle_ms_per_step"] = (time.perf_counter() - t0) / steps * 1000
        u = sim.contact_budget_usage()
        out["settle_contacts"] = {"broad": u["broad_pairs"], "points": u["contact_points"]}

        # ---- push step: plate driven through the pile --------------------
        dev = sim._particle_state.device
        p_start = torch.tensor([[-0.030, 0.0, sim._operation_height]],
                               device=dev).expand(n_envs, 3).contiguous()
        p_stop = torch.tensor([[0.030, 0.0, sim._operation_height]],
                              device=dev).expand(n_envs, 3).contiguous()
        # BLADE ORIENTATION MATTERS ENORMOUSLY and is easy to get wrong.
        # The plate is 40 mm along its local x and 2 mm along local y, and the
        # yaw is independent of the push direction. With yaw=0 and motion along
        # +x the blade's LONG axis is parallel to travel, so it slices through
        # the pile edge-on with a 2 mm swath and barely loads at all — which is
        # not a push in any meaningful sense. yaw=pi/2 puts the blade broadside,
        # sweeping its full 40 mm width, which is the representative case.
        yaw_val = (math.pi / 2) if yaw is None else float(yaw)
        angle = torch.full((n_envs,), yaw_val, device=dev)
        out["yaw"] = yaw_val
        out["swath_mm"] = (abs(math.sin(yaw_val)) * 0.04
                           + abs(math.cos(yaw_val)) * 0.002) * 1000

        # lower the plate into place, then drive it far enough to be loaded
        sim._vertical_dof_fix[:, 0] = p_start[:, 0]
        sim._vertical_dof_fix[:, 1] = p_start[:, 1]
        sim._vertical_dof_fix[:, 4] = angle
        sim.plate.set_pos(p_start, zero_velocity=True)
        sim._horizontal_dof_fix[:, -1] = yaw_val

        delta = p_stop - p_start
        dist = torch.linalg.norm(delta, axis=1)
        direction = delta / (dist.unsqueeze(1) + 1e-8)
        prof = sim._trapezoid_profile(dist)
        dt = sim._scene.dt

        # advance far enough that the blade is genuinely inside the pile.
        # 30 steps of the trapezoid covers ~14 mm from a -30 mm start, so the
        # blade is well into a tray-spanning pile by then.
        WARMUP = warmup
        for k in range(WARMUP):
            s, v = sim._trapezoid_at(prof, (k + 1) * dt)
            sim.plate.control_dofs_position_velocity(
                p_start + direction * s.unsqueeze(1),
                direction * v.unsqueeze(1), dofs_idx_local=[0, 1, 2])
            sim.plate.set_dofs_position(sim._horizontal_dof_fix,
                                        dofs_idx_local=sim._horizontal_dofs_local,
                                        zero_velocity=False)
            sim._step_scene()

        torch.cuda.synchronize(); t0 = time.perf_counter()
        follow_err, realized = [], []
        for k in range(WARMUP, WARMUP + steps):
            s, v = sim._trapezoid_at(prof, (k + 1) * dt)
            p_ref = p_start + direction * s.unsqueeze(1)
            sim.plate.control_dofs_position_velocity(
                p_ref, direction * v.unsqueeze(1), dofs_idx_local=[0, 1, 2])
            sim.plate.set_dofs_position(sim._horizontal_dof_fix,
                                        dofs_idx_local=sim._horizontal_dofs_local,
                                        zero_velocity=False)
            sim._step_scene()
            # Following error: is the tool tracking, or parked against a heap
            # it cannot move? This is the real machine's stall condition.
            cur = sim.plate.get_pos()
            follow_err.append(float((cur[:, :2] - p_ref[:, :2]).norm(dim=1).max()))
            realized.append(float((cur[:, :2] - p_start[:, :2]).norm(dim=1).mean()))
        torch.cuda.synchronize()
        ref_travel = float(sim._trapezoid_at(prof, (WARMUP + steps) * dt)[0].max())
        out["follow_err_mm"] = [e * 1000 for e in follow_err]
        out["follow_err_max_mm"] = max(follow_err) * 1000
        out["realized_travel_mm"] = realized[-1] * 1000
        out["reference_travel_mm"] = ref_travel * 1000
        out["tracking_fraction"] = realized[-1] / max(ref_travel, 1e-9)
        out["stalled"] = out["tracking_fraction"] < 0.5
        out["push_ms_per_step"] = (time.perf_counter() - t0) / steps * 1000
        u = sim.contact_budget_usage()
        out["push_contacts"] = {"broad": u["broad_pairs"], "points": u["contact_points"],
                                "broad_cap": u["broad_cap"], "points_cap": u["contact_cap"]}
        out["ratio"] = out["push_ms_per_step"] / max(out["settle_ms_per_step"], 1e-9)
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
    p.add_argument("--envs", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--variants", nargs="+", default=["baseline"])
    p.add_argument("--steps", type=int, default=5,
                   help="timed steps per regime; keep small, a push step at "
                        "32 envs has been measured at ~120 s")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--library-root", default="Genesis/data/dry_run",
                   help="root of recorded settled-state libraries, laid out as "
                        "<root>/<shape>/n<N>/size<S>/settled_states.pt. Seeding "
                        "from these skips the ~1500-step respawn settle that "
                        "otherwise dominates this probe. Pass '' to disable.")
    p.add_argument("--yaw", type=float, default=None,
                   help="blade yaw in radians; default pi/2 = broadside (full "
                        "40 mm swath). yaw=0 with x-motion is edge-on and "
                        "barely loads the tool at all")
    p.add_argument("--warmup", type=int, default=30,
                   help="sweep steps to advance before timing. Deeper = the "
                        "plate is bulldozing a denser, more compressed heap, "
                        "which is a different contact regime entirely")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cell", nargs=2, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cell is not None:
        name, n_envs = args.cell[0], int(args.cell[1])
        print("###JSON###" + json.dumps(run_variant(
            name, args.n_particles, args.particle_size, n_envs, args.steps,
            args.seed, args.warmup, args.yaw,
            str(REPO_ROOT / args.library_root) if args.library_root else None)))
        return

    rows = []
    for name in args.variants:
        for n_envs in args.envs:
            print(f"  {name:>20} n_envs={n_envs:>3} ...", end="", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m", "tests.scaling_investigation.probe_push_cost",
                 "--cell", name, str(n_envs),
                 "--n-particles", str(args.n_particles),
                 "--particle-size", str(args.particle_size),
                 "--steps", str(args.steps), "--seed", str(args.seed),
                 "--warmup", str(args.warmup)]
                + ([] if args.yaw is None else ["--yaw", str(args.yaw)])
                + ["--library-root", args.library_root],
                capture_output=True, text=True, cwd=str(REPO_ROOT))
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("###JSON###")), None)
            if line is None:
                print(" CRASHED")
                print("    " + (proc.stderr.strip().splitlines() or ["?"])[-1][:150])
                continue
            r = json.loads(line[len("###JSON###"):])
            rows.append(r)
            if r["ok"]:
                print(f" settle {r['settle_ms_per_step']:8.1f} ms  "
                      f"push {r['push_ms_per_step']:9.1f} ms  "
                      f"ratio {r['ratio']:6.1f}x  "
                      f"track {r['tracking_fraction']*100:5.1f}%  "
                      f"follow_err {r['follow_err_max_mm']:6.2f} mm"
                      f"{'  STALLED' if r['stalled'] else ''}")
            else:
                print(f" FAIL {r.get('error','')[:70]}")

    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return 1

    print("\n### push/settle cost ratio — the structural signal")
    print(f"{'variant':>20}" + "".join(f"{e:>10}" for e in args.envs))
    for name in args.variants:
        line = f"{name:>20}"
        for e in args.envs:
            r = next((x for x in ok if x["variant"] == name and x["n_envs"] == e), None)
            line += f"{'-':>10}" if r is None else f"{r['ratio']:>9.1f}x"
        print(line)
    print("  A ratio that GROWS with n_envs implicates contact islands.")
    print("  A FLAT ratio means ordinary saturation -> just use fewer envs.")

    for label, key in (("settle ms/step", "settle_ms_per_step"),
                       ("push ms/step", "push_ms_per_step")):
        print(f"\n### {label}")
        print(f"{'variant':>20}" + "".join(f"{e:>10}" for e in args.envs))
        for name in args.variants:
            line = f"{name:>20}"
            for e in args.envs:
                r = next((x for x in ok if x["variant"] == name and x["n_envs"] == e), None)
                line += f"{'-':>10}" if r is None else f"{r[key]:>10.1f}"
            print(line)

    print("\n### per-env push cost (ms/step/env) — flat means perfect batching")
    print(f"{'variant':>20}" + "".join(f"{e:>10}" for e in args.envs))
    for name in args.variants:
        line = f"{name:>20}"
        for e in args.envs:
            r = next((x for x in ok if x["variant"] == name and x["n_envs"] == e), None)
            line += f"{'-':>10}" if r is None else f"{r['push_ms_per_step']/e:>10.1f}"
        print(line)

    print("\n### contact usage during the push (a cap hit is a separate bug)")
    print(f"{'variant':>20} {'n_envs':>7} {'broad':>14} {'points':>16}")
    for r in ok:
        c = r["push_contacts"]
        print(f"{r['variant']:>20} {r['n_envs']:>7} "
              f"{str(c['broad']) + '/' + str(c['broad_cap']):>14} "
              f"{str(c['points']) + '/' + str(c['points_cap']):>16}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nraw -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
