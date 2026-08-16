#!/usr/bin/env python3
"""
probe_contact_islands.py — why does a transition cost 85x more at 200 objects
than at 100?

The observation
---------------
Measured end to end (probe_collection_health.py), seconds per transition:

    100 objects, 64 envs     10.5
    150 objects,  2 envs    106
    200 objects,  1 env     896

That is 85x the cost for 2x the particles and only 2.4x the contact points, so
contact *count* does not explain it. Per simulation step it is worse: at 200
objects in ONE env a sweep step costs seconds, while 64 envs x 100 objects —
32x more bodies — costs less per step.

The answer (measured — see below for how)
-----------------------------------------
**Confirmed.** Push cost is a function of the largest contact island and of
nothing else: ``cost ~ island_size^2.64`` across every configuration tested.
The decisive comparison is two cells with the same island size but twice the
particles::

    n=100, standard tray, largest island 24  ->  437 ms/step
    n=200, 1.5x tray,     largest island 21  ->  483 ms/step

Twice the objects, same cost. Meanwhile the same 200 cubes in a 1.5x tray run
**17.5x faster** than in the standard one (483 vs 8440 ms/step) purely because
the pile stops percolating (largest island 57 -> 21). Settle cost, where islands
are singletons, stays linear in particle count throughout (33 -> 161 ms for
n=50 -> 200).

So the lever is **packing fraction, not particle count**. Neither solver escape
is available: CG is broken in Genesis 0.4.5 (its kernel references
``RigidSolver.func_solve_mass_batch``, which does not exist), and disabling
islands replaces the per-island blocks with one global ``n_dofs^2`` Hessian —
12x more dense entries than the islands sum to at n=200, consistent with the
~1100x slowdown ``probe_push_cost.py`` measured for that variant.

The hypothesis, as originally stated
------------------------------------
**Contact-island percolation against a dense Newton Hessian.**

Genesis defaults to the Newton constraint solver. With ``use_contact_island``
the solver partitions constraints into independent islands and, per island,
builds a DENSE Hessian over that island's dofs
(``constraint/solver_island.py::_func_nt_hessian_direct``): it zeroes,
accumulates and symmetrizes an ``island_dofs x island_dofs`` block, where
``island_dofs = 6 x entities in the island``. That is **quadratic in island
size**.

While settling, the contact graph is ~n tiny islands — each cube resting on the
floor, touching nobody — so islands are size 1-2 and the quadratic term is
nothing. During a push the blade touches particles that touch other particles;
in a dense pile the contact graph **percolates** and islands grow to span much
of the tray. The predicted scaling is therefore not in n_particles but in the
size of the largest island.

Two details the measurement corrected. Percolation is **partial, not total**:
the largest island reaches ~0.28 x n (56 of 200), never the whole pile, so the
growth is smooth rather than a cliff. And the exponent is **2.64, not 2** —
between the quadratic cost of zeroing and symmetrizing the block and the cubic
cost of factorizing it, which is what a mix of the two terms looks like.

What this measures
------------------
Directly, by reading the solver's own island structures
(``rigid_solver.constraint_solver.contact_island``) at rest and during a push:

  n_islands            how many islands the contact graph decomposes into
  max_island_entities  the largest island. THE number the hypothesis is about.
  sum sq               sum over islands of size^2 — what the quadratic term
                       actually pays, and the quantity that should track time
  ms/step              measured cost of the same steps

Confirmation looks like: at rest, many small islands; during a push at 200
objects, one island containing most of the pile, with ms/step tracking sum-sq
rather than particle count or contact count.

Refutation looks like: island sizes stay small at 200 objects (so the cost is
elsewhere), or ms/step fails to track sum-sq across particle counts.

The variants also test the two available escapes, since a confirmed diagnosis is
only useful with one:

  baseline        islands on, Newton      — current configuration
  islands_off     islands off, Newton     — one global Hessian instead of one
                  per island. Previously measured ~1100x slower at small n,
                  but that was with many small islands, where decomposition
                  wins; the interesting question is whether it still loses once
                  a single island contains everything anyway.
  cg              islands on, CG          — CG never forms a dense Hessian, so
                  if the quadratic term is the cost this should be flat in
                  island size
  cg_islands_off  islands off, CG

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_contact_islands
    python -m tests.scaling_investigation.probe_contact_islands \
        --n-particles 50 100 200 --variants baseline cg
    python -m tests.scaling_investigation.probe_contact_islands \
        --n-particles 200 --variants baseline islands_off cg cg_islands_off
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GENESIS_DIR = REPO_ROOT / "Genesis"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# name -> rigid_options overrides
VARIANTS = {
    "baseline":       {},
    "islands_off":    {"use_contact_island": False},
    "cg":             {"constraint_solver": "CG"},
    "cg_islands_off": {"constraint_solver": "CG", "use_contact_island": False},
    # Hibernation is the principled version of "split the island when it gets
    # too big": bodies below a velocity/acceleration threshold are frozen and
    # dropped from the active island instead of being carried in its dense
    # block. During a push most of the pile is genuinely at rest, so the active
    # island should shrink to the neighbourhood of the blade. Requires
    # use_contact_island (Genesis silently disables it otherwise).
    "hibernation":    {"use_hibernation": True},
    # Genesis >=1.2.x only: CG *with* islands. Blocked in 0.4.5 by two bugs in
    # solver_island.py, which 1.2.0 deleted outright. Upstream the per-island
    # linear solve looks Newton-only, so islands may partition without
    # accelerating CG -- worth measuring rather than assuming either way.
    "cg_islands_on":  {"constraint_solver": "CG"},
}


def _config(n_particles, particle_size, overrides, box_scale=1.0,
            fence_scale=1.0, pile_scale=1.0):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size * pile_scale,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    cfg.setdefault("rigid_options", {}).update(overrides)
    if box_scale != 1.0:
        # Widen the tray without changing the particles. If cost is driven by
        # island size and island size by percolation, then the SAME 200 cubes
        # spread over more floor should cost dramatically less — which would
        # make tray size, not particle count, the thing that decides whether a
        # dataset is affordable.
        for key in ("box", "material"):
            w, d, h = cfg[key]["vol"]
            cfg[key]["vol"] = [w * box_scale, d * box_scale, h]
    if fence_scale != 1.0:
        # Move the WALLS out without touching the pile. box_scale confounds two
        # things — it makes islands smaller AND it takes the walls away — so on
        # its own it cannot distinguish "cost follows the number of objects in
        # contact" from "cost follows confinement against a wall". This scales
        # only the fence; the pile is then restored from the library exactly as
        # recorded, so the contact graph is the dense one and the walls are far
        # from it.
        w, d, h = cfg["box"]["vol"]
        cfg["box"]["vol"] = [w * fence_scale, d * fence_scale, h]
    if pile_scale != 1.0:
        # The blade scales with the pile too, so engagement stays geometrically
        # similar — otherwise a fixed-width blade would sweep a relatively wider
        # swath through a shrunken pile and couple a different number of cubes,
        # which is the one thing this test must hold fixed.
        l, w_, t = cfg["plate"]["size"]
        cfg["plate"]["size"] = [l * pile_scale, w_ * pile_scale, t * pile_scale]
    return cfg


def island_stats(sim):
    """Island decomposition of the current contact graph, read from the solver.

    Two incompatible layouts, because Genesis replaced the implementation in
    1.2.0 (PR #2972 deleted ``ConstraintSolverIsland`` / ``ContactIsland`` and
    the whole of ``solver_island.py``):

    * <=1.1.x  ``constraint_solver.contact_island`` — membership per ENTITY,
      island sizes read from ``island_entity.n``
    * >=1.2.x  ``constraint_solver.constraint_state.island`` — membership per
      LINK, via ``links_island_idx``. Sizes have to be counted rather than
      looked up, which is if anything more direct here: every particle is a
      single-link entity, so a link count IS a particle count.

    Supporting both keeps the 0.4.5 measurements in results/ reproducible
    instead of stranding them. Returns None when islands are disabled — there
    is then no decomposition to read, which is itself the answer.
    """
    import numpy as np
    cs = sim._scene.rigid_solver.constraint_solver

    state = getattr(cs, "constraint_state", None)
    island = getattr(state, "island", None) if state is not None else None
    if island is not None:                                   # >= 1.2.x
        n_islands = int(np.asarray(island.n_islands.to_numpy()).reshape(-1)[0])
        if n_islands <= 0:
            return {"n_islands": 0, "max_entities": 0, "mean_entities": 0.0,
                    "sum_sq_entities": 0, "sum_cube_entities": 0}
        idx = np.asarray(island.links_island_idx.to_numpy())
        idx = idx.reshape(idx.shape[0], -1)[:, 0]            # env 0
        sizes = np.bincount(idx[idx >= 0], minlength=n_islands)[:n_islands]
        sizes = sizes[sizes > 0].astype("int64")
        if sizes.size == 0:
            return {"n_islands": 0, "max_entities": 0, "mean_entities": 0.0,
                    "sum_sq_entities": 0, "sum_cube_entities": 0}
        return {
            "n_islands": int(sizes.size),
            "max_entities": int(sizes.max()),
            "mean_entities": float(sizes.mean()),
            "sum_sq_entities": int((sizes ** 2).sum()),
            "sum_cube_entities": int((sizes.astype("float64") ** 3).sum()),
        }

    ci = getattr(cs, "contact_island", None)                 # <= 1.1.x
    if ci is None:
        return None
    n_islands = int(ci.n_islands.to_numpy()[0])
    if n_islands <= 0:
        return {"n_islands": 0, "max_entities": 0, "mean_entities": 0.0,
                "sum_sq_entities": 0}
    sizes = ci.island_entity.n.to_numpy()[:n_islands, 0].astype("int64")
    return {
        "n_islands": n_islands,
        "max_entities": int(sizes.max()),
        "mean_entities": float(sizes.mean()),
        # Two cost terms, because the solver pays both: building/zeroing the
        # dense island block is quadratic in its dofs, while factorizing it is
        # cubic. Which one dominates is exactly what the fit below decides.
        "sum_sq_entities": int((sizes ** 2).sum()),
        "sum_cube_entities": int((sizes.astype("float64") ** 3).sum()),
    }


def run_cell(n_particles, n_envs, variant, particle_size, library_root, seed,
             sample_every, warmup, box_scale=1.0, fence_scale=1.0,
             pile_scale=1.0):
    import numpy as np
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation
    from Genesis.state_library import StateLibrary, default_library_path

    torch.manual_seed(seed)
    np.random.seed(seed)
    out = {"n_particles": n_particles, "n_envs": n_envs, "variant": variant,
           "overrides": VARIANTS[variant], "box_scale": box_scale,
           "fence_scale": fence_scale, "pile_scale": pile_scale}

    sim = SandboxManipulation(
        config=_config(n_particles, particle_size, VARIANTS[variant], box_scale,
                       fence_scale, pile_scale),
        n_envs=n_envs, debug=False)
    sim.build()

    try:
        lib_path = default_library_path(GENESIS_DIR / library_root, "cube",
                                        n_particles, particle_size)
        # A library is recorded for one tray geometry; only box_scale changes
        # the volume the pile was packed into, so only it has to spawn afresh.
        # fence_scale and pile_scale both keep the recorded pile — that is the
        # point of them.
        if box_scale == 1.0 and lib_path.exists():
            lib = StateLibrary.load(lib_path)
            idx = lib.next_index(np.random.default_rng(seed))
            state = lib.states[idx].to(sim._particle_state.device)
            pos, quat = state[None, :, 0:3], state[None, :, 3:7]
            if pile_scale != 1.0:
                # Shrink the recorded pile and its particles by the same factor:
                # geometrically similar, so the contact graph is IDENTICAL, but
                # it now occupies a small patch in the middle of an unchanged
                # tray and touches no wall. Rotations are scale-invariant, so
                # only positions move.
                pos = pos * pile_scale
            sim.set_particle_state(pos, quat)
            out["seeded_from"] = "library"
        else:
            sim.shuffle_particles()
            out["seeded_from"] = "spawn"
        sim.update_material_state()
        eff_size = particle_size * pile_scale
        out["packing_fraction"] = (
            n_particles * eff_size ** 2
            / (sim._box_params["vol"][0] * sim._box_params["vol"][1]))
        # How much of the pile is actually in contact with a wall — the thing
        # fence_scale is supposed to drive to zero while leaving islands alone.
        p = sim._get_particle_positions()[0, :n_particles]
        half_w = sim._box_params["vol"][0] / 2
        near_wall = (p[:, :2].abs() > half_w - eff_size).any(dim=-1)
        out["particles_touching_wall"] = int(near_wall.sum())

        # ---- the cheap regime: pile at rest, plate lifted clear ------------
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(warmup):
            sim._step_scene()
        torch.cuda.synchronize()
        out["settle_ms_per_step"] = (time.perf_counter() - t0) / warmup * 1e3
        out["settle_islands"] = island_stats(sim)
        u = sim.contact_budget_usage()
        out["settle_contacts"] = {"broad": u["broad_pairs"],
                                  "points": u["contact_points"]}

        # ---- the expensive regime: blade driven broadside through the pile --
        # Fixed action, identical at every particle count and every variant, so
        # the comparison is not polluted by the 2-9x swing between blade
        # orientations or by a different travel distance.
        dev = sim._particle_state.device
        reach = 0.030 * pile_scale          # scales with the pile, see _config
        p_start = torch.tensor([[-reach, 0.0, sim._operation_height]],
                               device=dev).expand(n_envs, 3).contiguous()
        p_stop = torch.tensor([[reach, 0.0, sim._operation_height]],
                              device=dev).expand(n_envs, 3).contiguous()
        angle = torch.full((n_envs,), math.pi / 2, device=dev)

        samples = []
        state = {"t": None}

        def on_step(step, *_):
            if step % sample_every:
                return
            torch.cuda.synchronize()
            now = time.perf_counter()
            if state["t"] is not None:
                rec = {"step": step,
                       "ms_per_step": (now - state["t"]) / sample_every * 1e3}
                isl = island_stats(sim)
                if isl:
                    rec.update(isl)
                samples.append(rec)
            state["t"] = time.perf_counter()

        # descend first so the blade starts in contact, then sweep
        sim.plate_position_translation(
            p_start + sim._clearance_offset, p_start, sim._clearance_ctrl_steps)
        sim.plate_velocity_translation(p_start, p_stop, angle, on_step=on_step)

        out["push_samples"] = samples
        if samples:
            ms = [s["ms_per_step"] for s in samples]
            out["push_ms_per_step_mean"] = sum(ms) / len(ms)
            out["push_ms_per_step_max"] = max(ms)
            out["push_settle_ratio"] = (out["push_ms_per_step_mean"]
                                        / out["settle_ms_per_step"])
            if "max_entities" in samples[0]:
                out["push_max_island_entities"] = max(
                    s["max_entities"] for s in samples)
                out["push_sum_sq_peak"] = max(s["sum_sq_entities"] for s in samples)
                out["push_sum_cube_peak"] = max(
                    s.get("sum_cube_entities", 0) for s in samples)
                out["push_n_islands_min"] = min(s["n_islands"] for s in samples)
        u = sim.contact_budget_usage()
        out["push_contacts"] = {"broad": u["broad_pairs"],
                                "points": u["contact_points"]}
        out["ok"] = True
    except Exception as e:
        import traceback
        out.update(ok=False, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1200:])
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
    return out


def report(cells):
    print("\n" + "=" * 92)
    print("CONTACT ISLANDS vs PUSH COST")
    print("=" * 92)
    print(f"{'variant':<12}{'n':>5}{'box':>5}{'fence':>6}{'pile':>6}"
          f"{'wall':>6}{'settle ms':>11}{'push ms':>10}{'ratio':>7}"
          f"{'max isl':>9}{'points':>8}")
    for c in cells:
        if not c.get("ok"):
            print(f"{c['variant']:<15}{c['n_particles']:>5}   FAILED: "
                  f"{str(c.get('error'))[:60]}")
            continue
        print(f"{c['variant']:<12}{c['n_particles']:>5}"
              f"{c.get('box_scale', 1.0):>5.2f}{c.get('fence_scale', 1.0):>6.2f}"
              f"{c.get('pile_scale', 1.0):>6.2f}"
              f"{c.get('particles_touching_wall', '-'):>6}"
              f"{c['settle_ms_per_step']:>11.1f}"
              f"{c.get('push_ms_per_step_mean', float('nan')):>10.1f}"
              f"{c.get('push_settle_ratio', float('nan')):>7.1f}"
              f"{c.get('push_max_island_entities', '-'):>9}"
              f"{c['push_contacts']['points']:>8}")

    # The test of the hypothesis: does cost track the quadratic term?
    quad = [c for c in cells
            if c.get("ok") and c.get("push_sum_sq_peak")
            and c["variant"] == "baseline"]
    if len(quad) > 1:
        print("\nbaseline: cost against the two terms dense island linear algebra pays")
        print(f"  {'n':>5}{'max isl':>9}{'push ms':>10}{'ms/sum-sq':>12}"
              f"{'ms/sum-cube':>14}")
        for c in sorted(quad, key=lambda c: c["n_particles"]):
            cube = c.get("push_sum_cube_peak") or 0
            print(f"  {c['n_particles']:>5}"
                  f"{c.get('push_max_island_entities', 0):>9}"
                  f"{c['push_ms_per_step_mean']:>10.1f}"
                  f"{c['push_ms_per_step_mean']/c['push_sum_sq_peak']:>12.4f}"
                  + (f"{c['push_ms_per_step_mean']/cube:>14.5f}" if cube
                     else f"{'-':>14}"))
        print("  Whichever column is CONSTANT down the rows is the term that\n"
              "  explains the cost: sum-sq = building the dense block,\n"
              "  sum-cube = factorizing it.")
    print()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-particles", nargs="+", type=int,
                   default=[50, 100, 150, 200])
    p.add_argument("--n-envs", type=int, default=1,
                   help="1 by default: the cost cliff was measured at 1 env, "
                        "and a single env keeps island size the only thing "
                        "that varies")
    p.add_argument("--variants", nargs="+", default=["baseline"],
                   choices=sorted(VARIANTS))
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--library-root", default="data/dry_run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-every", type=int, default=5,
                   help="sample island stats and timing every N sweep steps")
    p.add_argument("--box-scale", nargs="+", type=float, default=[1.0],
                   help="widen the tray by this factor, keeping the particles "
                        "unchanged, to vary packing fraction independently of "
                        "particle count. Scaled trays spawn their own pile "
                        "(a state library is specific to one geometry).")
    p.add_argument("--fence-scale", nargs="+", type=float, default=[1.0],
                   help="move the WALLS out by this factor while restoring the "
                        "recorded pile unchanged. Separates 'many objects in "
                        "contact' from 'confined against a wall', which "
                        "--box-scale confounds.")
    p.add_argument("--pile-scale", nargs="+", type=float, default=[1.0],
                   help="shrink the recorded pile, its particles and the blade "
                        "by this factor inside an unchanged tray: a "
                        "geometrically similar pile, identical contact graph, "
                        "touching no wall.")
    p.add_argument("--warmup", type=int, default=20,
                   help="steps timed in the at-rest regime")
    p.add_argument("--out", default=None)
    p.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()

    if args.cell:                                   # child: one (n, variant)
        n_particles, variant, box_scale, fence_scale, pile_scale = \
            args.cell.split(":")
        res = run_cell(int(n_particles), args.n_envs, variant,
                       args.particle_size, args.library_root, args.seed,
                       args.sample_every, args.warmup, float(box_scale),
                       float(fence_scale), float(pile_scale))
        print("__RESULT__" + json.dumps(res))
        return

    out = Path(args.out) if args.out else RESULTS_DIR / "contact_islands.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in args.variants:
      for scale in args.box_scale:
       for fence in args.fence_scale:
        for pile in args.pile_scale:
         for n in args.n_particles:
            cell = f"{n}:{variant}:{scale}:{fence}:{pile}"
            print(f">>> {cell}", flush=True)
            # subprocess per cell: rigid_options are build-time, and a CUDA
            # fault in one variant must not take the sweep with it
            proc = subprocess.run(
                [sys.executable, "-m",
                 "tests.scaling_investigation.probe_contact_islands",
                 "--cell", cell, "--n-envs", str(args.n_envs),
                 "--particle-size", str(args.particle_size),
                 "--library-root", args.library_root, "--seed", str(args.seed),
                 "--sample-every", str(args.sample_every),
                 "--warmup", str(args.warmup)],
                cwd=REPO_ROOT, capture_output=True, text=True)
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("__RESULT__")), None)
            if line:
                results.append(json.loads(line[len("__RESULT__"):]))
            else:
                results.append({"n_particles": n, "variant": variant,
                                "box_scale": scale, "fence_scale": fence,
                                "pile_scale": pile, "ok": False,
                                "error": (proc.stderr or proc.stdout)[-400:]})
            out.write_text(json.dumps(results, indent=2))
            report(results[-1:])

    print("\n" + "#" * 92)
    report(results)
    print(f"raw results -> {out}")


if __name__ == "__main__":
    main()
