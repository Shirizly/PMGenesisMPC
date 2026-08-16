#!/usr/bin/env python3
"""
probe_shared_vs_distinct.py — does giving every environment the SAME initial
state actually make a batch cheaper to simulate?

Collection broadcasts one settled state to all envs rather than giving each its
own, trading a little within-batch diversity for speed. That trade was justified
by a comparison between two runs that also differed in seeding method and warmup
depth, so the speed claim was never cleanly measured. This measures it: identical
scene, identical action, identical library — the only thing that varies is
whether the envs start from one shared state or one state each.

Times three phases separately, because they answer different questions:

  setup   scene construction + build (kernel compilation). Independent of
          seeding; reported to show it is not what differs.
  seed    loading the library and writing the poses.
  sweep   one execute_action (lower, sweep, lift) — the phase where identical
          contact graphs could plausibly help.

The action is FIXED (broadside, same for every env and every cell) so the
comparison is not polluted by the 2-9x cost swing between blade orientations,
nor by batch step count following the largest sampled travel distance.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_shared_vs_distinct
    python -m tests.scaling_investigation.probe_shared_vs_distinct --envs 4 128 --n-particles 50
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
    out = {"mode": mode, "n_envs": n_envs, "n_particles": n_particles}

    t0 = time.perf_counter()
    sim = SandboxManipulation(config=_config(n_particles, particle_size),
                              n_envs=n_envs, debug=False)
    sim.build()
    torch.cuda.synchronize()
    out["setup_s"] = time.perf_counter() - t0

    try:
        lib_path = default_library_path(GENESIS_DIR / library_root, "cube",
                                        n_particles, particle_size)
        lib = StateLibrary.load(lib_path)
        if lib.n_particles != len(sim.material):
            raise ValueError(f"library has {lib.n_particles} particles, "
                             f"scene has {len(sim.material)}")

        rng = np.random.default_rng(seed)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if mode == "shared":
            used = [lib.apply(sim, rng=rng)]
        else:
            used = lib.apply_per_env(sim, rng=rng)
        torch.cuda.synchronize()
        out["seed_s"] = time.perf_counter() - t0
        out["n_distinct_states"] = len(set(used))
        sim._particle_state[:, :, 0:3] = sim._get_particle_positions()
        sim._particle_state[:, :, 3:] = sim._get_particle_quats()

        # FIXED broadside action, identical in every env and every cell.
        dev = sim._particle_state.device
        p_start = torch.tensor([[-0.030, 0.0, sim._operation_height]],
                               device=dev).expand(n_envs, 3).contiguous()
        p_stop = torch.tensor([[0.030, 0.0, sim._operation_height]],
                              device=dev).expand(n_envs, 3).contiguous()
        angle = torch.full((n_envs,), math.pi / 2, device=dev)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        sim.execute_action(p_start, p_stop, angle)
        torch.cuda.synchronize()
        out["sweep_s"] = time.perf_counter() - t0

        u = sim.contact_budget_usage()
        out["contacts"] = {"broad": u["broad_pairs"], "points": u["contact_points"]}
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
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--envs", nargs="+", type=int, default=[4, 128])
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
        for mode in ("shared", "per_env"):
            print(f"  n_envs={n_envs:>4} {mode:>8} ...", end="", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m",
                 "tests.scaling_investigation.probe_shared_vs_distinct",
                 "--cell", mode, str(n_envs),
                 "--n-particles", str(args.n_particles),
                 "--particle-size", str(args.particle_size),
                 "--library-root", args.library_root, "--seed", str(args.seed)],
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
                print(f" setup {r['setup_s']:6.1f}s  seed {r['seed_s']*1000:7.1f}ms  "
                      f"sweep {r['sweep_s']:8.2f}s  "
                      f"({r['n_distinct_states']} distinct states)")
            else:
                print(f" FAIL {r.get('error','')[:70]}")

    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return 1

    print(f"\n### {args.n_particles} objects, fixed broadside action")
    print(f"{'n_envs':>7} {'phase':>8} {'shared':>12} {'per-env':>12} "
          f"{'per-env / shared':>18}")
    for n_envs in args.envs:
        sh = next((r for r in ok if r["n_envs"] == n_envs and r["mode"] == "shared"), None)
        pe = next((r for r in ok if r["n_envs"] == n_envs and r["mode"] == "per_env"), None)
        if not (sh and pe):
            continue
        for label, key, scale, unit in (("setup", "setup_s", 1.0, "s"),
                                        ("seed", "seed_s", 1000.0, "ms"),
                                        ("sweep", "sweep_s", 1.0, "s")):
            ratio = pe[key] / sh[key] if sh[key] else float("nan")
            print(f"{n_envs:>7} {label:>8} "
                  f"{sh[key]*scale:>10.2f}{unit:<2} {pe[key]*scale:>10.2f}{unit:<2} "
                  f"{ratio:>17.2f}x")

    print("\n### verdict")
    for n_envs in args.envs:
        sh = next((r for r in ok if r["n_envs"] == n_envs and r["mode"] == "shared"), None)
        pe = next((r for r in ok if r["n_envs"] == n_envs and r["mode"] == "per_env"), None)
        if not (sh and pe):
            continue
        gain = pe["sweep_s"] / sh["sweep_s"] if sh["sweep_s"] else float("nan")
        if gain > 1.15:
            v = f"shared is {gain:.2f}x faster — the trade pays"
        elif gain < 0.87:
            v = f"shared is {1/gain:.2f}x SLOWER — the trade does not pay"
        else:
            v = f"no meaningful difference ({gain:.2f}x) — diversity is ~free"
        print(f"  n_envs={n_envs:>4}: {v}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"\nraw -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
