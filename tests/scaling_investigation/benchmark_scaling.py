#!/usr/bin/env python3
"""
tests/scaling_investigation/benchmark_scaling.py — measure how data-collection cost scales with
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

    python -m tests.scaling_investigation.benchmark_scaling
    python -m tests.scaling_investigation.benchmark_scaling --particles 50 100 200 --envs 1 2 4 8
    python -m tests.scaling_investigation.benchmark_scaling --out results.json

Internal single-cell mode (invoked by the driver, not by hand)::

    python -m tests.scaling_investigation.benchmark_scaling --cell N_PARTICLES N_ENVS
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

# This script lives outside Genesis/, so paths to the simulator's configs are
# resolved explicitly rather than relative to this file.
GENESIS_DIR = Path(__file__).resolve().parents[2] / "Genesis"
REPO_ROOT = Path(__file__).resolve().parents[2]


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
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
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
             settle_steps: int, n_trials: int, max_collision_pairs: int,
             vram_only: bool = False) -> dict:
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
    # None -> let SandboxManipulation pick its shipped default, so the sweep
    # measures the configuration that will actually be used to collect. An
    # oversized cap is not free: the constraint Jacobian is
    # O(mcp x contacts_per_pair x n_dofs x n_envs), so it directly reduces how
    # many envs fit, without making the physics any more correct.
    if max_collision_pairs is not None:
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
        if vram_only:
            sim.shuffle_particles()
            sim.update_material_state()
            out["vram_peak_gib"] = vram_used_gib() - vram_baseline
            out["max_collision_pairs_used"] = int(
                sim._scene.rigid_solver.max_collision_pairs)
            out["contact_usage"] = sim.contact_budget_usage()
            out["ok"] = True
            return out

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
        # A FIXED crossing sweep, identical in every env and every cell.
        # Randomly sampled actions are not comparable across cells: the sweep's
        # step count follows the travel distance, so a cell that happened to
        # draw long pushes looks slower for reasons that have nothing to do
        # with n_particles or n_envs. (Before this was fixed, two draws in one
        # cell differed by 5x.)
        dev = sim._particle_state.device
        half = 0.045
        p_start = torch.tensor([[-half, 0.0, sim._operation_height]],
                               device=dev).expand(n_envs, 3).contiguous()
        p_stop = torch.tensor([[half, 0.0, sim._operation_height]],
                              device=dev).expand(n_envs, 3).contiguous()
        angle = torch.zeros(n_envs, device=dev)
        out["sweep_distance_m"] = 2 * half

        sync(); t = time.perf_counter()
        sim.execute_action(p_start, p_stop, angle)
        sync()
        out["t_action_warmup"] = time.perf_counter() - t

        ts = []
        for _ in range(n_trials):
            sync(); t = time.perf_counter()
            sim.execute_action(p_start, p_stop, angle)
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
                   help="default: whatever SandboxManipulation ships, so the "
                        "sweep measures the real collection configuration")
    p.add_argument("--vram-only", action="store_true",
                   help="build + one settle, then report VRAM and stop. Finds "
                        "the env ceiling in ~40 s/cell instead of minutes, "
                        "which is the right trade when only the memory "
                        "boundary has changed (per-step cost is independent "
                        "of max_collision_pairs)")
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

    if any("contact_usage" in r for r in rows):
        print("\n### contact budget occupancy (peak used / cap)")
        print(f"{'n_p':>5} {'n_envs':>7} {'mcp':>6} {'broad pairs':>14} "
              f"{'contact points':>16} {'mcp needed':>11}")
        for r in rows:
            u = r.get("contact_usage")
            if not u:
                continue
            need = max(math.ceil(u["broad_pairs"] / 8),
                       math.ceil(u["contact_points"] / u["n_contacts_per_pair"]))
            print(f"{r['n_particles']:>5} {r['n_envs']:>7} "
                  f"{r.get('max_collision_pairs_used', '?'):>6} "
                  f"{str(u['broad_pairs']) + '/' + str(u['broad_cap']):>14} "
                  f"{str(u['contact_points']) + '/' + str(u['contact_cap']):>16} "
                  f"{need:>11}")

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
            # --vram-only cells are 'ok' but carry no timings, so read
            # defensively rather than indexing — otherwise the whole summary
            # (and the --out JSON written after it) is lost to a KeyError.
            if r is None or not r.get("ok") or "t_reset" not in r:
                line += f"{'-' if r is None or r.get('ok') else 'FAIL':>9}"
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
        mcp = args.max_collision_pairs
        result = run_cell(n_particles, n_envs, args.particle_size,
                          args.settle_steps, args.n_trials, mcp,
                          vram_only=args.vram_only)
        print("###JSON###" + json.dumps(result))
        return

    rows = []
    for n in args.particles:
        cap = monolayer_capacity(0.128, args.particle_size)
        layers = math.ceil(n / cap)
        print(f"\n=== n_particles={n} (size {args.particle_size*1000:.1f} mm, "
              f"~{cap}/layer, needs {layers} layer(s)) ===", flush=True)
        for e in args.envs:
            cmd = [sys.executable, "-m", "tests.scaling_investigation.benchmark_scaling",
                   "--cell", str(n), str(e),
                   "--particle-size", str(args.particle_size),
                   "--settle-steps", str(args.settle_steps),
                   "--n-trials", str(args.n_trials)]
            if args.max_collision_pairs is not None:
                cmd += ["--max-collision-pairs", str(args.max_collision_pairs)]
            if args.vram_only:
                cmd += ["--vram-only"]
            print(f"  n_envs={e} ...", end="", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=str(REPO_ROOT))
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
            if r.get("ok") and args.vram_only:
                u = r.get("contact_usage", {})
                print(f" {r['vram_peak_gib']:.2f} GiB  (mcp="
                      f"{r.get('max_collision_pairs_used')}, broad "
                      f"{u.get('broad_pairs')}/{u.get('broad_cap')}, points "
                      f"{u.get('contact_points')}/{u.get('contact_cap')})",
                      flush=True)
            elif r.get("ok"):
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
