#!/usr/bin/env python3
"""
benchmark_throughput.py — find the n_envs that maximises transitions/second for
each object count, and record the full grid behind that choice.

Why this is not benchmark_scaling.py
------------------------------------
``benchmark_scaling.py --vram-only`` answers "how many envs FIT". That is an
upper bound, not an operating point, and the collection plan originally took its
env counts from it. The two are not the same number: settling batches
sublinearly while pushing is closer to linear, so throughput can peak well below
the memory ceiling. This measures the thing that matters — completed transitions
per wall-clock second — and reports the whole grid, not just the winner.

What one measurement is
-----------------------
A transition is ``execute_action`` (lower, sweep, lift) followed by
``update_material_state`` (settle to rest) — the same pair
``collect_data_samples`` performs per sample. Actions come from
``generate_action_samples``, i.e. the real distribution of yaw and travel
distance. That matters more than it sounds: push cost varies 2-9x with blade
orientation alone, so a fixed action would be unrepresentative.

Two things keep the grid affordable:

* **Library seeding.** Each cell restores a settled pile from a recorded
  state library instead of settling a fresh two-layer respawn (~1500 steps at
  200 particles). All envs share one state, matching what collection does, so
  the measurement reflects the configuration actually being timed.
* **Early termination.** Throughput rises while the GPU is underused and falls
  once saturated, so the ladder stops climbing once it has stopped improving.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.benchmark_throughput \\
        --plan configs/throughput_benchmark.yaml
    python -m tests.scaling_investigation.benchmark_throughput --dry-run
"""

import argparse
import json
from datetime import datetime
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

GENESIS_DIR = Path(__file__).resolve().parents[2] / "Genesis"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(plan, n_particles):
    with open(GENESIS_DIR / "configs" / plan.get("base_config", "basic.yaml")) as f:
        cfg = yaml.safe_load(f)
    m = plan["material"]
    cfg["material"].update(shape=m["shape"], particle_size=m["particle_size"],
                           n_particles=n_particles,
                           density=m["particle_density"],
                           friction=m["particle_friction"])
    cfg["box"]["friction"] = m["box_friction"]
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def estimate_vram_gib(n_particles, n_envs):
    """Fitted to the measured OOM ceilings (benchmark_scaling.py --vram-only)."""
    return 0.15 + (0.0025 + 0.001078 * n_particles) * n_envs


def free_vram_gib():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()[0] / 2 ** 30
    except Exception:
        return None


def run_cell(plan, n_particles, n_envs, n_transitions, seed):
    import numpy as np
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation
    from Genesis.state_library import StateLibrary, default_library_path

    torch.manual_seed(seed)
    m = plan["material"]
    sim = SandboxManipulation(config=_config(plan, n_particles),
                              n_envs=n_envs, debug=False)
    out = {"n_particles": n_particles, "n_envs": n_envs}
    try:
        t0 = time.perf_counter()
        sim.build()
        out["build_s"] = time.perf_counter() - t0

        root = plan.get("library_root")
        lib_path = (default_library_path(GENESIS_DIR / root, m["shape"],
                                         n_particles, m["particle_size"])
                    if root else None)
        seeded = False
        if lib_path is not None and lib_path.exists():
            lib = StateLibrary.load(lib_path)
            if lib.n_particles == len(sim.material):
                # One shared state across envs, matching collection: this must
                # measure what production does, and production broadcasts.
                lib.apply(sim, rng=np.random.default_rng(seed))
                sim._particle_state[:, :, 0:3] = sim._get_particle_positions()
                sim._particle_state[:, :, 3:] = sim._get_particle_quats()
                seeded = True
        if not seeded:
            sim.shuffle_particles()
            sim.update_material_state()
        out["seeded_from_library"] = seeded

        # shared_travel_distance mirrors collect_data_samples' default: envs step
        # in lockstep, so independent push lengths make every env run for the
        # longest one's duration. Measuring without it would be measuring a
        # configuration collection does not use.
        starts, stops, angles = sim.generate_action_samples(
            n_transitions + 1, shared_travel_distance=True)

        # warm up: one full transition, untimed (kernel specialization, caches)
        sim.execute_action(starts[:, 0, :], stops[:, 0, :], angles[:, 0])
        sim.update_material_state()

        per_transition = []
        for i in range(1, n_transitions + 1):
            torch.cuda.synchronize(); t = time.perf_counter()
            sim.execute_action(starts[:, i, :], stops[:, i, :], angles[:, i])
            sim.update_material_state()
            torch.cuda.synchronize()
            per_transition.append(time.perf_counter() - t)

        med = float(np.median(per_transition))
        out["seconds_per_batch"] = med
        out["seconds_per_transition"] = med / n_envs
        out["transitions_per_sec"] = n_envs / med
        out["all_batches_s"] = per_transition
        u = sim.contact_budget_usage()
        out["contacts"] = {"broad": u["broad_pairs"], "broad_cap": u["broad_cap"],
                           "points": u["contact_points"], "points_cap": u["contact_cap"]}
        out["ok"] = True
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {str(e)[:140]}")
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default="configs/throughput_benchmark.yaml")
    p.add_argument("--only", nargs="+", type=int, default=None,
                   help="restrict to these object counts")
    p.add_argument("--dry-run", action="store_true",
                   help="print the grid and VRAM estimates, run nothing")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--cell", nargs=2, type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    with open(GENESIS_DIR / args.plan) as f:
        plan = yaml.safe_load(f)
    spec = plan["plan"]

    if args.cell is not None:
        n_p, n_e = args.cell
        print("###JSON###" + json.dumps(run_cell(
            plan, n_p, n_e, spec["transitions_per_cell"], args.seed)))
        return

    counts = args.only or spec["n_objects"]
    free = free_vram_gib()
    print(f"=== throughput benchmark: {plan['plan']['n_objects']} objects x "
          f"{spec['n_envs']} envs ===")
    print(f"  {spec['transitions_per_cell']} transitions/cell, "
          f"stop after {spec['stop_after_declines']} non-improving steps, "
          f"free VRAM {free:.2f} GiB" if free else "  free VRAM unknown")

    rows = []
    for n_p in counts:
        print(f"\n--- n_objects={n_p} ---")
        best, declines = 0.0, 0
        for n_e in spec["n_envs"]:
            est = estimate_vram_gib(n_p, n_e)
            if free is not None and est > free * spec["vram_fraction_limit"]:
                print(f"  n_envs={n_e:>4}  SKIP (est {est:.1f} GiB > "
                      f"{spec['vram_fraction_limit']:.0%} of {free:.1f} free)")
                break
            if args.dry_run:
                print(f"  n_envs={n_e:>4}  est {est:.2f} GiB")
                continue

            print(f"  n_envs={n_e:>4} ...", end="", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m",
                 "tests.scaling_investigation.benchmark_throughput",
                 "--cell", str(n_p), str(n_e), "--plan", args.plan,
                 "--seed", str(args.seed)],
                capture_output=True, text=True, cwd=str(REPO_ROOT))
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("###JSON###")), None)
            if line is None:
                print(" CRASHED")
                rows.append({"n_particles": n_p, "n_envs": n_e, "ok": False,
                             "error": "crashed"})
                break
            r = json.loads(line[len("###JSON###"):])
            rows.append(r)
            if not r["ok"]:
                print(f" FAIL {r.get('error','')[:60]}")
                break
            tps = r["transitions_per_sec"]
            mark = ""
            if tps > best:
                best, declines, mark = tps, 0, "  <-- best so far"
            else:
                declines += 1
                mark = f"  (no improvement, {declines})"
            print(f" {r['seconds_per_batch']:7.1f} s/batch  "
                  f"{tps:7.3f} transitions/s{mark}")
            if spec["stop_after_declines"] and declines >= spec["stop_after_declines"]:
                print(f"  stopping: throughput has not improved in "
                      f"{declines} steps")
                break

    if args.dry_run:
        print("\n--dry-run: nothing launched")
        return 0

    ok = [r for r in rows if r.get("ok")]
    if not ok:
        print("\nno successful cells")
        return 1

    print("\n### transitions/second (full grid)")
    envs = sorted({r["n_envs"] for r in ok})
    print(f"{'n_obj':>6}" + "".join(f"{e:>10}" for e in envs))
    for n_p in counts:
        line = f"{n_p:>6}"
        for e in envs:
            r = next((x for x in ok if x["n_particles"] == n_p and x["n_envs"] == e), None)
            line += f"{'-':>10}" if r is None else f"{r['transitions_per_sec']:>10.3f}"
        print(line)

    print("\n### seconds per transition (lower is better)")
    print(f"{'n_obj':>6}" + "".join(f"{e:>10}" for e in envs))
    for n_p in counts:
        line = f"{n_p:>6}"
        for e in envs:
            r = next((x for x in ok if x["n_particles"] == n_p and x["n_envs"] == e), None)
            line += f"{'-':>10}" if r is None else f"{r['seconds_per_transition']:>10.2f}"
        print(line)

    print("\n### optimal n_envs per object count")
    print(f"{'n_obj':>6} {'best n_envs':>12} {'transitions/s':>14} "
          f"{'s/transition':>13} {'vs 1 env':>9}")
    optimal = {}
    for n_p in counts:
        cand = [r for r in ok if r["n_particles"] == n_p]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["transitions_per_sec"])
        one = next((r for r in cand if r["n_envs"] == 1), None)
        speedup = (best["transitions_per_sec"] / one["transitions_per_sec"]
                   if one else float("nan"))
        optimal[n_p] = best["n_envs"]
        print(f"{n_p:>6} {best['n_envs']:>12} "
              f"{best['transitions_per_sec']:>14.3f} "
              f"{best['seconds_per_transition']:>13.2f} {speedup:>8.1f}x")

    out_full = REPO_ROOT / plan["output"]["results"]
    out_opt = REPO_ROOT / plan["output"]["optimal"]
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_full.write_text(json.dumps(rows, indent=2))

    # Record WHAT these counts were measured under, not just the counts. A
    # throughput optimum is specific to the material and the hardware, so a
    # consumer (run_collection.py) has to be able to tell whether the numbers
    # still apply to the run it is about to launch.
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu = None
    best_rows = {}
    for n_p, n_e in optimal.items():
        r = next(x for x in ok if x["n_particles"] == n_p and x["n_envs"] == n_e)
        best_rows[int(n_p)] = {
            "n_envs": int(n_e),
            "transitions_per_sec": round(r["transitions_per_sec"], 4),
            "seconds_per_transition": round(r["seconds_per_transition"], 3),
        }
    out_opt.write_text(yaml.safe_dump({
        "n_envs": {int(k): int(v) for k, v in optimal.items()},
        "best": best_rows,
        "measured_under": {
            "shape": plan["material"]["shape"],
            "particle_size": plan["material"]["particle_size"],
            "particle_friction": plan["material"]["particle_friction"],
            "particle_density": plan["material"]["particle_density"],
            "box_friction": plan["material"]["box_friction"],
            "shared_travel_distance": True,
            "transitions_per_cell": spec["transitions_per_cell"],
            "gpu": gpu,
            "free_vram_gib": round(free, 2) if free else None,
            "measured_at": datetime.now().isoformat(timespec="seconds"),
            "source": "tests/scaling_investigation/benchmark_throughput.py",
        },
        "note": "Throughput-optimal env counts. Reference from a collection "
                "plan with `n_envs: <path to this file>` rather than copying "
                "the numbers, so they cannot silently go stale.",
    }, sort_keys=True))
    print(f"\nfull grid -> {out_full}\noptimal    -> {out_opt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
