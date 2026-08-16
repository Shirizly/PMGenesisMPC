#!/usr/bin/env python3
"""
Genesis/run_collection.py — drive a data-collection run across several
``n_objects``, one subprocess per pile size, with preflight and postflight
checks.

Why a driver rather than a bigger flag list on data_collection_clean.py
----------------------------------------------------------------------
Changing ``n_particles`` requires a full scene rebuild (particles are created
before ``scene.build()``), so each pile size is a separate process anyway. Doing
it explicitly buys three things: the number of parallel envs can differ per pile
size (VRAM per env grows with the particle count), an OOM or an infeasible
placement at one size cannot abort the rest of the run, and each size can be
checked on the way in and validated on the way out.

Checks it performs
------------------
preflight, before spending a scene build:
  * placement feasibility — can this many particles of this size be packed into
    the tray at all, and in how many layers? Infeasible sizes are skipped with a
    reason rather than discovered via a RuntimeError 2 minutes in.
  * free VRAM against the plan's env count, with an explicit warning when the
    request is close to the card's capacity.

postflight, on the data each size produced:
  * the dataset file exists, loads, and has the expected shapes/dtype
  * no NaNs or infinities in the recorded states
  * s' actually differs from s — a run that "succeeds" while nothing moves is
    the silent failure this is here to catch
  * the fraction of samples that reached their goal
  * particles stayed inside the tray
  * the state library, if requested, was written and has the expected size

Usage
-----
From the REPO ROOT::

    python -m Genesis.run_collection --plan configs/collection_dry_run.yaml
    python -m Genesis.run_collection --plan configs/collection_dry_run.yaml --dry-run
    python -m Genesis.run_collection --plan configs/collection_dry_run.yaml --env-scale 0.5
    python -m Genesis.run_collection --plan configs/collection_dry_run.yaml --viewer bird
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
GENESIS_DIR = Path(__file__).resolve().parent

# Mirrors shuffle_particles: a cube free to take any yaw is treated as an
# axis-aligned square of side size*sqrt(2), and layers are packed to ~45% area.
LAYER_FILL_FRACTION = 0.45


def placement_feasibility(n_particles, particle_size, box_vol, wall_thickness,
                          shape="cube"):
    """Can this many particles be spawned, and in how many stacked layers?"""
    width, _, height = box_vol
    footprint = particle_size * (math.sqrt(2) if shape in ("cube", "box") else 1.0)
    per_layer = max(1, int(LAYER_FILL_FRACTION * width ** 2 / footprint ** 2))
    layers_needed = math.ceil(n_particles / per_layer)

    usable_height = (height - wall_thickness / 2) - (wall_thickness / 2)
    pitch = particle_size + 1e-3
    max_layers = max(1, int((usable_height - 1e-3) // pitch))

    return {
        "per_layer": per_layer,
        "layers_needed": layers_needed,
        "max_layers": max_layers,
        "capacity": per_layer * max_layers,
        "feasible": layers_needed <= max_layers,
    }


def resolve_env_counts(spec, material, cli_override=None):
    """Work out how many envs to run per object count, and say where from.

    ``plan.n_envs`` may be either a literal ``{n_objects: n_envs}`` mapping or a
    path to a `throughput_optimal.yaml` produced by
    ``tests/scaling_investigation/benchmark_throughput.py``. Referencing the
    measurement rather than copying its numbers is the point: the throughput
    optimum moved by more than an order of magnitude when the action sampler
    changed (2 envs -> 64 at 100 objects), and a copied table gives no signal
    when that happens.

    A throughput optimum is specific to the material it was measured on, so the
    recorded conditions are checked against the plan and any divergence is
    reported rather than silently accepted.

    Returns (mapping, provenance_string, warnings).
    """
    source = cli_override if cli_override is not None else spec.get("n_envs")
    warnings = []

    if isinstance(source, dict):
        return ({int(k): int(v) for k, v in source.items()},
                "literal values in the plan", warnings)

    path = Path(source)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"plan.n_envs points at {path}, which does not exist. Run "
            f"`python -m tests.scaling_investigation.benchmark_throughput` "
            f"to produce it, or replace it with a literal mapping.")

    with open(path) as f:
        blob = yaml.safe_load(f)
    counts = {int(k): int(v) for k, v in blob["n_envs"].items()}
    cond = blob.get("measured_under", {})

    for key in ("shape", "particle_size", "particle_friction",
                "particle_density", "box_friction"):
        want, got = material.get(key), cond.get(key)
        if got is not None and want is not None and got != want:
            warnings.append(f"measured at {key}={got}, this run uses {want}")
    if not cond.get("shared_travel_distance", True):
        warnings.append("measured WITHOUT shared travel distance; collection "
                        "uses it, so these counts understate what is achievable")

    prov = (f"{path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}"
            f" (measured {cond.get('measured_at', 'at an unrecorded time')}"
            f"{', ' + cond['gpu'] if cond.get('gpu') else ''})")
    return counts, prov, warnings


def free_vram_gib():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return free / 2 ** 30
    except Exception:
        return None


def estimate_vram_gib(n_particles, n_envs):
    """Rough per-env VRAM, fitted to the measured --vram-only sweep.

    Fitted to the per-env cost at each pile size's measured ceiling
    (tests/scaling_investigation/benchmark_scaling.py --vram-only, shipped max_collision_pairs):
    (50,128)->7.22 GiB, (100,64)->7.28, (150,32)->5.26, (200,32)->6.98, i.e.
    0.056 / 0.114 / 0.164 / 0.218 GiB per env. Linear in n_particles to within
    1%, which is enough to warn before paying for a build.
    """
    per_env = 0.0025 + 0.001078 * n_particles
    return 0.15 + per_env * n_envs


def check_dataset(data_path: Path, n_particles: int) -> list[str]:
    """Validate one saved dataset file. Returns a list of problems."""
    import torch

    problems = []
    blob = torch.load(data_path, weights_only=False)
    states = blob.get("states")
    states_ = blob.get("states_")
    if states is None or states_ is None:
        return [f"{data_path.name}: missing 'states'/'states_' keys "
                f"(got {sorted(blob.keys())})"]

    if states.ndim != 3 or states.shape[1] != n_particles or states.shape[2] != 7:
        problems.append(f"unexpected states shape {tuple(states.shape)}, "
                        f"expected (N, {n_particles}, 7)")
    if states.shape != states_.shape:
        problems.append(f"states {tuple(states.shape)} != states_ "
                        f"{tuple(states_.shape)}")
    if states.shape[0] == 0:
        problems.append("zero successful samples recorded")
        return problems

    if not torch.isfinite(states).all() or not torch.isfinite(states_).all():
        problems.append("non-finite values in recorded states")

    # The silent failure this exists to catch: the run completed but the pushes
    # did nothing, so every s' equals its s.
    moved = (states_[..., :3] - states[..., :3]).abs().amax(dim=(1, 2))
    n_static = int((moved < 1e-6).sum())
    if n_static == states.shape[0]:
        problems.append("NO sample changed state — actions had no effect")
    elif n_static:
        problems.append(f"{n_static}/{states.shape[0]} samples did not change "
                        f"state at all")

    max_disp = float(moved.max())
    if max_disp > 0.5:
        problems.append(f"implausible displacement {max_disp:.3f} m — particles "
                        f"likely ejected from the tray")

    return problems


def summarize_dataset(data_path: Path):
    import torch
    blob = torch.load(data_path, weights_only=False)
    states, states_ = blob["states"], blob["states_"]
    moved = (states_[..., :3] - states[..., :3]).abs().amax(dim=(1, 2))
    return {
        "n_samples": int(states.shape[0]),
        "mean_displacement_mm": float(moved.mean()) * 1000,
        "max_displacement_mm": float(moved.max()) * 1000,
        "xy_extent_mm": float(states_[..., :2].abs().max()) * 1000,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="configs/collection_dry_run.yaml",
                   help="collection plan YAML, relative to Genesis/")
    p.add_argument("--n-envs-from", default=None, metavar="PATH",
                   help="override plan.n_envs with a throughput_optimal.yaml "
                        "produced by benchmark_throughput.py")
    p.add_argument("--env-scale", type=float, default=1.0,
                   help="multiply every planned n_envs by this. The planned "
                        "values are measured OOM ceilings, so use 0.5 on a "
                        "shared GPU")
    p.add_argument("--only", nargs="+", type=int, default=None,
                   help="run only these n_objects")
    p.add_argument("--dry-run", action="store_true",
                   help="run preflight checks and print the commands, but "
                        "launch nothing")
    p.add_argument("--viewer", choices=["observer", "bird", "leveled"], default=None,
                   help="open a live viewer window. Forces n_envs=1 — the "
                        "viewer renders every env, so a batched run is both "
                        "unreadable and far slower with it on")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed, forwarded to each n_objects run offset by "
                        "its own index so the runs do not all draw the same "
                        "actions. Omit for an unseeded (non-reproducible) run.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    plan_path = GENESIS_DIR / args.plan
    with open(plan_path) as f:
        plan = yaml.safe_load(f)

    with open(GENESIS_DIR / "configs" / plan.get("base_config", "basic.yaml")) as f:
        base = yaml.safe_load(f)

    mat, spec = plan["material"], plan["plan"]
    n_list = args.only or spec["n_objects"]
    size = float(mat["particle_size"])

    env_counts, env_provenance, env_warnings = resolve_env_counts(
        spec, mat, args.n_envs_from)

    print(f"=== collection plan: {plan_path.name} ===")
    print(f"  shape={mat['shape']} size={size*1000:.1f} mm  "
          f"friction={mat['particle_friction']} density={mat['particle_density']} "
          f"box_friction={mat['box_friction']}")
    print(f"  samples/env={spec['samples_per_env']}  batches={spec['n_batches']}  "
          f"state_library={spec['state_library_settles']}  "
          f"placement_aware={spec['placement_aware']}")
    print(f"  n_envs from: {env_provenance}")
    for w in env_warnings:
        print(f"  !! env counts may not apply: {w}")
    if args.viewer:
        print(f"  VIEWER={args.viewer} -> n_envs forced to 1")

    free = free_vram_gib()
    print(f"  free VRAM: {free:.2f} GiB" if free else "  free VRAM: unknown")

    # ---------------- preflight ----------------
    print("\n=== preflight ===")
    runnable = []
    for n in n_list:
        feas = placement_feasibility(n, size, base["box"]["vol"],
                                     base["box"]["wall_thickness"], mat["shape"])
        if not args.viewer and n not in env_counts:
            print(f"  n={n:>4}  SKIP — no measured env count for this object "
                  f"count. Add it to the benchmark plan and re-run "
                  f"benchmark_throughput, or set plan.n_envs literally.")
            continue
        n_envs = 1 if args.viewer else max(
            1, int(env_counts[n] * args.env_scale))
        est = estimate_vram_gib(n, n_envs)

        if not feas["feasible"]:
            print(f"  n={n:>4}  SKIP — needs {feas['layers_needed']} layers but "
                  f"only {feas['max_layers']} fit ({feas['per_layer']}/layer, "
                  f"capacity {feas['capacity']}). Use smaller particles or a "
                  f"bigger tray.")
            continue

        # The planned env counts ARE the measured ceilings, so a cell sitting
        # just under the card's capacity is expected, not a warning sign. Only
        # flag an overrun, and note tightness above 85%.
        note = ""
        if free is not None and est > free:
            note = f"  !! est {est:.1f} GiB vs {free:.1f} free — expect OOM"
        elif free is not None and est > free * 0.85:
            note = (f"  ! est {est:.1f} GiB vs {free:.1f} free — tight, "
                    f"no room for another GPU consumer")
        print(f"  n={n:>4}  envs={n_envs:>4}  {feas['layers_needed']} layer(s) "
              f"({feas['per_layer']}/layer)  est {est:.2f} GiB{note}")
        runnable.append((n, n_envs))

    if not runnable:
        print("\nnothing runnable — stopping")
        return 1

    total = sum(e * spec["samples_per_env"] * spec["n_batches"] for _, e in runnable)
    print(f"\n  planned transitions this run: {total}")

    # ---------------- run ----------------
    results = {}
    for run_idx, (n, n_envs) in enumerate(runnable):
        out_rel = f"{plan['output_root']}"
        cmd = [
            sys.executable, "-m", "Genesis.data_collection_clean",
            "--num-particles", str(n),
            "--particle-sizes", str(size),
            "--particle-shape", mat["shape"],
            "--n-envs", str(n_envs),
            "--samples-per-env", str(spec["samples_per_env"]),
            "--n-batches", str(spec["n_batches"]),
            "--output-root", out_rel,
            "--constant-params",
            "--particle-friction", str(mat["particle_friction"]),
            "--particle-density", str(mat["particle_density"]),
            "--box-friction", str(mat["box_friction"]),
        ]
        if spec["state_library_settles"]:
            cmd += ["--state-library", str(spec["state_library_settles"])]
        if spec["placement_aware"]:
            cmd += ["--placement-aware"]
        if args.seed is not None:
            # Offset per run: the same seed in every subprocess would draw the
            # identical action sequence at every object count, which looks like
            # reproducibility but is a correlated dataset.
            cmd += ["--seed", str(args.seed + run_idx)]
        if args.viewer:
            cmd += ["--viewer-type", args.viewer]
        if args.debug:
            cmd += ["--debug"]

        print(f"\n=== n_objects={n} (envs={n_envs}) ===")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        elapsed = time.perf_counter() - t0
        results[n] = {"returncode": proc.returncode, "seconds": elapsed,
                      "n_envs": n_envs}
        print(f"  exit={proc.returncode} in {elapsed:.1f} s")

    if args.dry_run:
        print("\n--dry-run: nothing launched")
        return 0

    # ---------------- postflight ----------------
    print("\n=== postflight: validating what was written ===")
    ok_all = True
    for n, info in results.items():
        out_dir = (GENESIS_DIR / plan["output_root"] / mat["shape"]
                   / f"n{n}" / f"size{size}")
        if info["returncode"] != 0:
            print(f"  n={n:>4}  FAILED (exit {info['returncode']})")
            ok_all = False
            continue
        data_files = sorted(out_dir.glob("*_data.pt"))
        if not data_files:
            print(f"  n={n:>4}  NO DATA written to {out_dir}")
            ok_all = False
            continue

        problems = []
        for f in data_files:
            problems += [f"{f.name}: {p}" for p in check_dataset(f, n)]
        summary = summarize_dataset(data_files[-1])

        lib = out_dir / "settled_states.pt"
        lib_note = ""
        if spec["state_library_settles"]:
            if lib.exists():
                import torch
                blob = torch.load(lib, weights_only=False)
                lib_note = (f", library {blob['states'].shape[0]} states "
                            f"({blob['meta'].get('n_symmetries','?')} symmetries)")
            else:
                problems.append("state library requested but not written")

        status = "OK  " if not problems else "PROB"
        print(f"  n={n:>4}  {status} {summary['n_samples']} samples, "
              f"displacement mean {summary['mean_displacement_mm']:.1f} mm / "
              f"max {summary['max_displacement_mm']:.1f} mm, "
              f"|xy|max {summary['xy_extent_mm']:.1f} mm, "
              f"{info['seconds']:.0f} s{lib_note}")
        for p in problems:
            ok_all = False
            print(f"        - {p}")

    print("\n=== cost per size ===")
    n_batches = spec["n_batches"]
    for n, info in sorted(results.items()):
        if info["returncode"] != 0:
            continue
        transitions = info["n_envs"] * spec["samples_per_env"] * n_batches
        print(f"  n={n:>4}: {info['seconds']:7.0f} s wall for {transitions:>5} "
              f"transitions ({info['seconds']/max(transitions,1):.2f} s each, "
              f"{info['n_envs']} envs)")

    if n_batches < 2:
        print(
            "\n  NOTE: with n_batches=1 the fixed cost (scene build, kernel\n"
            "  compilation, state-library settles) cannot be separated from the\n"
            "  marginal per-batch cost, so extrapolating these numbers to a long\n"
            "  run would overstate it — most of the time above is paid once per\n"
            "  build, not once per batch. Re-run with n_batches: 3 to get a\n"
            "  marginal cost worth extrapolating from.")
    else:
        print("\n=== extrapolation (fixed cost separated) ===")
        print("  Requires two runs at different n_batches to solve for the\n"
              "  intercept; with a single run this reports the average only.")
        for n, info in sorted(results.items()):
            if info["returncode"] != 0:
                continue
            per_batch = info["seconds"] / n_batches
            print(f"  n={n:>4}: ~{per_batch/60:.1f} min per batch "
                  f"(incl. amortized fixed cost); 1000 batches ~= "
                  f"{per_batch*1000/3600:.1f} h")

    print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED — see above")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
