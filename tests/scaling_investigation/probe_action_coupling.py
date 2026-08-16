#!/usr/bin/env python3
"""
probe_action_coupling.py — why does per-batch cost explode between 2 and 4 envs?

The throughput grid shows a cliff that gets worse with object count
(seconds per batch, `outputs/scaling_benchmark/throughput_full.json`):

    n_obj   1 env    2      4       8
       20     2.3    2.7    5.5     6.3
      100    11.0   13.8   92.1   113.6
      200    37.6   88.4  2706.6     --

That is not saturation. The hypothesis is **worst-case coupling**: every env in a
batch steps together, so the batch pays for the worst env in it, twice over —

  step count      ``sweep_steps`` is derived from the LARGEST travel distance
                  across envs, so every env runs for as long as the longest one.
  per-step cost   each step costs what the densest contact graph in the batch
                  costs. With randomly sampled yaw, more envs means a higher
                  chance of including one broadside push through a dense pile.

Those two are separable, and separating them says what to do about it:

  shared_action   one identical action in every env — the floor.
  matched_dist    per-env random yaw and direction, but all travel the SAME
                  distance. Isolates contact complexity from step count: if this
                  is as cheap as shared_action, only step count matters and the
                  fix is to sample matched travel distances. If it is expensive,
                  the coupling is in the contact graph and no sampling change
                  helps.
  per_env         fully random per env — what collection actually does.

All three use a shared initial state from the library, as collection does, so
the only variable is the action distribution.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_action_coupling
    python -m tests.scaling_investigation.probe_action_coupling --n-particles 100 --envs 1 2 4 8
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
MODES = ("shared_action", "matched_dist", "per_env")


def _config(n_particles, particle_size):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def run_cell(mode, n_particles, particle_size, n_envs, library_root, seed):
    import numpy as np
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation
    from Genesis.state_library import StateLibrary, default_library_path

    torch.manual_seed(seed)
    sim = SandboxManipulation(config=_config(n_particles, particle_size),
                              n_envs=n_envs, debug=False)
    out = {"mode": mode, "n_envs": n_envs, "n_particles": n_particles}
    try:
        sim.build()

        lib = StateLibrary.load(default_library_path(
            GENESIS_DIR / library_root, "cube", n_particles, particle_size))
        lib.apply(sim, rng=np.random.default_rng(seed))   # shared state, as collection does
        sim._particle_state[:, :, 0:3] = sim._get_particle_positions()
        sim._particle_state[:, :, 3:] = sim._get_particle_quats()

        torch.manual_seed(seed)
        starts, stops, angles = sim.generate_action_samples(1)
        p_start, p_stop, angle = starts[:, 0, :], stops[:, 0, :], angles[:, 0]

        if mode == "shared_action":
            p_start = p_start[0:1].expand(n_envs, 3).contiguous()
            p_stop = p_stop[0:1].expand(n_envs, 3).contiguous()
            angle = angle[0:1].expand(n_envs).contiguous()
        elif mode == "matched_dist":
            # keep each env's own yaw and direction, equalise travel distance by
            # shrinking every push to the shortest one (always in bounds)
            delta = p_stop - p_start
            dist = torch.linalg.norm(delta, dim=1, keepdim=True)
            target = dist.min()
            p_stop = p_start + delta / (dist + 1e-9) * target

        dist = torch.linalg.norm(p_stop - p_start, dim=1)
        out["travel_min_mm"] = float(dist.min()) * 1000
        out["travel_max_mm"] = float(dist.max()) * 1000
        out["travel_mean_mm"] = float(dist.mean()) * 1000

        # sweep_steps is derived from the longest travel in the batch — compute
        # it the same way plate_velocity_translation does, so cost can be split
        # into "how many steps" and "how expensive is a step".
        prof = sim._trapezoid_profile(dist)
        dt = sim._scene.dt
        sweep_steps = max(1, math.ceil(float(prof["duration"].max()) / dt)) \
            + sim._sweep_settle_steps
        out["sweep_steps"] = sweep_steps

        torch.cuda.synchronize(); t0 = time.perf_counter()
        sim.execute_action(p_start, p_stop, angle)
        torch.cuda.synchronize()
        out["sweep_s"] = time.perf_counter() - t0
        out["ms_per_step"] = out["sweep_s"] / sweep_steps * 1000
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
    p.add_argument("--n-particles", type=int, default=100)
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--envs", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--library-root", default="data/dry_run")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cell", nargs=2, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cell is not None:
        mode, n_envs = args.cell[0], int(args.cell[1])
        print("###JSON###" + json.dumps(run_cell(
            mode, args.n_particles, args.particle_size, n_envs,
            args.library_root, args.seed)))
        return

    rows = []
    for n_envs in args.envs:
        for mode in args.modes:
            print(f"  n_envs={n_envs:>3} {mode:>14} ...", end="", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m",
                 "tests.scaling_investigation.probe_action_coupling",
                 "--cell", mode, str(n_envs),
                 "--n-particles", str(args.n_particles),
                 "--particle-size", str(args.particle_size),
                 "--library-root", args.library_root, "--seed", str(args.seed)],
                capture_output=True, text=True, cwd=str(REPO_ROOT))
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("###JSON###")), None)
            if line is None:
                print(" CRASHED")
                print("    " + (proc.stderr.strip().splitlines() or ["?"])[-1][:140])
                continue
            r = json.loads(line[len("###JSON###"):])
            rows.append(r)
            if r["ok"]:
                print(f" {r['sweep_s']:8.2f}s  {r['sweep_steps']:>4} steps  "
                      f"{r['ms_per_step']:7.1f} ms/step  "
                      f"travel {r['travel_min_mm']:.0f}-{r['travel_max_mm']:.0f} mm")
            else:
                print(f" FAIL {r.get('error','')[:70]}")

    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return 1

    def table(title, key, spec):
        print(f"\n### {title}")
        print(f"{'n_envs':>7}" + "".join(f"{m:>16}" for m in args.modes))
        for n_envs in args.envs:
            line = f"{n_envs:>7}"
            for m in args.modes:
                r = next((x for x in ok if x["n_envs"] == n_envs and x["mode"] == m), None)
                line += f"{'-':>16}" if r is None else f"{r[key]:>16{spec}}"
            print(line)

    table("total sweep time (s)", "sweep_s", ".2f")
    table("sweep steps (follows the LONGEST travel in the batch)", "sweep_steps", "d")
    table("ms per step (follows the DENSEST contact graph in the batch)",
          "ms_per_step", ".1f")
    table("longest travel in batch (mm)", "travel_max_mm", ".0f")

    print("\n### which coupling dominates")
    for n_envs in args.envs:
        got = {m: next((x for x in ok if x["n_envs"] == n_envs and x["mode"] == m), None)
               for m in args.modes}
        if not all(got.get(m) for m in ("shared_action", "matched_dist", "per_env")):
            continue
        sh, md, pe = got["shared_action"], got["matched_dist"], got["per_env"]
        step_factor = pe["sweep_steps"] / sh["sweep_steps"]
        cost_factor = md["ms_per_step"] / sh["ms_per_step"]
        total = pe["sweep_s"] / sh["sweep_s"]
        print(f"  n_envs={n_envs:>3}: total {total:5.2f}x  = "
              f"step-count {step_factor:4.2f}x x per-step {cost_factor:4.2f}x"
              f"   -> {'STEP COUNT dominates (fix by matching travel distances)' if step_factor > cost_factor else 'CONTACT COMPLEXITY dominates (sampling cannot fix)'}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nraw -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
