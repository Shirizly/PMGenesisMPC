#!/usr/bin/env python3
"""
Genesis/benchmark_scaling.py — measure how data-collection cost scales with
particle count and parallel-environment count, to plan large-``n_particles``
dataset collection.

Where ``benchmark_n_envs.py`` answers "what ``n_envs`` should oracle MPC use?"
for a fixed small pile, this answers the dataset-collection question: for
``n_particles`` in {50, 70, 100, 150, 200}, how many envs fit in VRAM, what
does one transition cost, and how does the cost of *resetting* an environment
(``shuffle_particles`` — re-randomize + re-settle) compare against the cost of
just executing another action in the state we already have?

Each (n_particles, n_envs) cell is timed in its own subprocess so that an OOM
or a Genesis buffer-overflow exception at one cell doesn't abort the sweep;
the parent collects one JSON line per cell and prints the summary tables.

Measured per cell
-----------------
build       scene construction + kernel compilation + ``scene.build()``
reset       ``shuffle_particles()`` — RSA re-placement of every particle
resettle    ``update_material_state()`` — ``settle_steps`` of free simulation
action      ``execute_action()`` — lower / sweep / lift
transition  action + resettle (what one recorded sample actually costs)
restore     ``set_particle_state()`` — the cheap snapshot-restore reset path
            used by ``simple_mpc.genesis_oracle``, for comparison against
            the full ``shuffle_particles`` reset
vram        device memory attributable to this process, via ``mem_get_info``

Usage
-----
Run as a module from the REPO ROOT (see ``benchmark_n_envs.py`` for why)::

    python -m Genesis.benchmark_scaling
    python -m Genesis.benchmark_scaling --particles 50 100 200 --envs 1 2 4 8
    python -m Genesis.benchmark_scaling --out results.json

Internal single-cell mode (invoked by the driver, not by hand)::

    python -m Genesis.benchmark_scaling --cell N_PARTICLES N_ENVS
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PARTICLES = [50, 70, 100, 150, 200]
DEFAULT_ENVS = [1, 2, 4, 8, 16, 32]

# RSA saturation for aligned squares is ~0.56 area fraction; stay well below it
# so per-layer placement succeeds reliably instead of burning retries.
LAYER_FILL_FRACTION = 0.45


def monolayer_capacity(box_side: float, particle_size: float,
                       *, is_cube: bool = True) -> int:
    """How many particles of ``particle_size`` fit in one layer of the box.

    Mirrors ``shuffle_particles``' collision test: a cube free to take any yaw
    is treated as an axis-aligned square of side ``size * sqrt(2)``.
    """
    footprint = particle_size * (math.sqrt(2) if is_cube else 1.0)
    return max(1, int(LAYER_FILL_FRACTION * box_side ** 2 / footprint ** 2))


def _build_config(n_particles: int, particle_size: float,
                  settle_steps: int | None = None) -> dict:
    base_dir = Path(__file__).parent
    with open(base_dir / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"]["shape"] = "cube"
    cfg["material"]["particle_size"] = particle_size
    cfg["material"]["n_particles"] = n_particles
    cfg["material"]["density"] = 1000.0
    cfg["material"]["friction"] = 0.3
    cfg["box"]["friction"] = 0.3
    if settle_steps is not None:
        cfg["simulation"]["settle_steps"] = settle_steps
    # Recording is irrelevant to timing and would write junk into data/mpc_runs.
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


# --------------------------------------------------------------------------
# single-cell measurement (runs in its own subprocess)
# --------------------------------------------------------------------------

def run_cell(n_particles: int, n_envs: int, particle_size: float,
             settle_steps: int, n_trials: int, max_collision_pairs: int) -> dict:
    import torch
    import genesis as gs

    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    def vram_used_gib() -> float:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2 ** 30

    def sync():
        torch.cuda.synchronize()

    vram_baseline = vram_used_gib()

    cfg = _build_config(n_particles, particle_size, settle_steps)
    cfg["rigid_options"]["max_collision_pairs"] = max_collision_pairs

    out = {
        "n_particles": n_particles,
        "n_envs": n_envs,
        "particle_size": particle_size,
        "settle_steps": settle_steps,
        "max_collision_pairs": max_collision_pairs,
        "vram_baseline_gib": vram_baseline,
    }

    t0 = time.perf_counter()
    sim = SandboxManipulation(config=cfg, n_envs=n_envs, debug=False)
    sim.build()
    sync()
    out["t_build"] = time.perf_counter() - t0
    out["vram_after_build_gib"] = vram_used_gib() - vram_baseline

    try:
        # ---- reset (full RSA re-placement) --------------------------------
        ts = []
        for _ in range(n_trials):
            sync(); t = time.perf_counter()
            sim.shuffle_particles()
            sync(); ts.append(time.perf_counter() - t)
        out["t_reset"] = float(np.median(ts))
        out["t_reset_all"] = ts

        # ---- resettle (settle_steps of free sim, no action) ---------------
        sync(); t = time.perf_counter()
        sim.update_material_state()          # warmup: compiles settle kernels
        sync()
        out["t_settle_warmup"] = time.perf_counter() - t

        ts = []
        for _ in range(n_trials):
            sync(); t = time.perf_counter()
            sim.update_material_state()
            sync(); ts.append(time.perf_counter() - t)
        out["t_settle"] = float(np.median(ts))

        # ---- restore from snapshot (cheap reset path) ---------------------
        snap_pos = sim._particle_state[0:1, :, 0:3].clone()
        snap_quat = sim._particle_state[0:1, :, 3:7].clone()
        sim.set_particle_state(snap_pos, snap_quat)   # warmup
        sync()
        ts = []
        for _ in range(n_trials):
            sync(); t = time.perf_counter()
            sim.set_particle_state(snap_pos, snap_quat)
            sync(); ts.append(time.perf_counter() - t)
        out["t_restore"] = float(np.median(ts))

        # ---- action (lower / sweep / lift) --------------------------------
        starts, stops, angles = sim.generate_action_samples(n_trials + 1)
        sync(); t = time.perf_counter()
        sim.execute_action(starts[:, 0, :], stops[:, 0, :], angles[:, 0])
        sync()
        out["t_action_warmup"] = time.perf_counter() - t

        ts = []
        for i in range(1, n_trials + 1):
            sync(); t = time.perf_counter()
            sim.execute_action(starts[:, i, :], stops[:, i, :], angles[:, i])
            sync(); ts.append(time.perf_counter() - t)
        out["t_action"] = float(np.median(ts))

        out["t_transition"] = out["t_action"] + out["t_settle"]
        out["transitions_per_sec"] = n_envs / out["t_transition"]
        out["vram_peak_gib"] = vram_used_gib() - vram_baseline
        out["ok"] = True
    except Exception as e:                       # OOM, collider overflow, ...
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        out["vram_peak_gib"] = vram_used_gib() - vram_baseline
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
        gs = None  # noqa: F841
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", nargs="+", type=int, default=DEFAULT_PARTICLES)
    p.add_argument("--envs", nargs="+", type=int, default=DEFAULT_ENVS)
    p.add_argument("--particle-size", type=float, default=0.005,
                   help="cube side (m). 5 mm is the only size in the current "
                        "sweep at which 150-200 particles fit the 128 mm box "
                        "in <=2 layers")
    p.add_argument("--settle-steps", type=int, default=100,
                   help="settle steps per transition (dataset default is 100)")
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--max-collision-pairs", type=int, default=None,
                   help="default: auto (4 x n_particles, floor 150) — the "
                        "wrapper's 150 default overflows well before 200 "
                        "particles")
    p.add_argument("--out", type=Path, default=None, help="write raw JSON here")
    p.add_argument("--cell", nargs=2, type=int, default=None,
                   help=argparse.SUPPRESS)   # internal single-cell mode
    return p.parse_args()


def _fmt(v, spec=".3f", width=9, missing="-"):
    return f"{missing:>{width}}" if v is None else f"{v:>{width}{spec}}"


def print_tables(rows, particles, envs):
    by = {(r["n_particles"], r["n_envs"]): r for r in rows}

    def table(title, key, spec=".3f", note=""):
        print(f"\n### {title}{note}")
        print("{:>13}".format("n_p \\ n_envs") + "".join(f"{e:>9}" for e in envs))
        for n in particles:
            line = f"{n:>13}"
            for e in envs:
                r = by.get((n, e))
                if r is None:
                    line += f"{'-':>9}"
                elif not r.get("ok"):
                    line += f"{'FAIL':>9}"
                else:
                    line += _fmt(r.get(key), spec)
            print(line)

    table("scene build (s)", "t_build", note=" — paid once per rebuild")
    table("reset: shuffle_particles (s)", "t_reset")
    table("restore: set_particle_state (s)", "t_restore", ".4f")
    table("settle: update_material_state (s)", "t_settle")
    table("action: execute_action (s)", "t_action")
    table("one transition = action+settle (s)", "t_transition")
    table("throughput (transitions/s)", "transitions_per_sec", ".1f")
    table("VRAM attributable to process (GiB)", "vram_peak_gib", ".2f")

    print("\n### reset amortization — resets cost this many transitions")
    print("{:>13}".format("n_p \\ n_envs") + "".join(f"{e:>9}" for e in envs))
    for n in particles:
        line = f"{n:>13}"
        for e in envs:
            r = by.get((n, e))
            if r is None or not r.get("ok"):
                line += f"{'-' if r is None else 'FAIL':>9}"
            else:
                # a reset also needs a settle before the state is usable
                cost = (r["t_reset"] + r["t_settle"]) / r["t_transition"]
                line += f"{cost:>9.2f}"
        print(line)

    print("\n### failures")
    any_fail = False
    for r in rows:
        if not r.get("ok"):
            any_fail = True
            print(f"  n_p={r['n_particles']:>4} n_envs={r['n_envs']:>3}: "
                  f"{r.get('error', 'unknown')[:160]}")
    if not any_fail:
        print("  none")


def main():
    args = parse_args()

    if args.cell is not None:
        n_particles, n_envs = args.cell
        mcp = args.max_collision_pairs or max(150, 4 * n_particles)
        result = run_cell(n_particles, n_envs, args.particle_size,
                          args.settle_steps, args.n_trials, mcp)
        print("###JSON###" + json.dumps(result))
        return

    rows = []
    for n in args.particles:
        cap = monolayer_capacity(0.128, args.particle_size)
        layers = math.ceil(n / cap)
        print(f"\n=== n_particles={n} (size {args.particle_size*1000:.1f} mm, "
              f"~{cap}/layer, needs {layers} layer(s)) ===", flush=True)
        for e in args.envs:
            cmd = [sys.executable, "-m", "Genesis.benchmark_scaling",
                   "--cell", str(n), str(e),
                   "--particle-size", str(args.particle_size),
                   "--settle-steps", str(args.settle_steps),
                   "--n-trials", str(args.n_trials)]
            if args.max_collision_pairs is not None:
                cmd += ["--max-collision-pairs", str(args.max_collision_pairs)]
            print(f"  n_envs={e} ...", end="", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=str(Path(__file__).resolve().parent.parent))
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("###JSON###")), None)
            if line is None:
                rows.append({"n_particles": n, "n_envs": e, "ok": False,
                             "error": (proc.stderr.strip().splitlines() or
                                       ["no output"])[-1]})
                print(" CRASHED", flush=True)
                # A hard crash (OOM kill) at this n_envs means larger will too.
                break
            r = json.loads(line[len("###JSON###"):])
            rows.append(r)
            if r.get("ok"):
                print(f" {r['t_transition']:.3f}s/transition, "
                      f"{r['transitions_per_sec']:.1f}/s, "
                      f"{r['vram_peak_gib']:.2f} GiB", flush=True)
            else:
                print(f" FAIL ({r.get('error','')[:80]})", flush=True)
                break

    print_tables(rows, args.particles, args.envs)

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nraw results -> {args.out}")


if __name__ == "__main__":
    main()
