#!/usr/bin/env python3
"""
Genesis/benchmark_n_envs.py — measure Genesis multi-env rollout throughput to
pick a default ``mpc.n_envs`` for ``simple_mpc.oracle_mpc``
(see docs/oracle_mpc_design.md's config reference for ``n_envs``).

For each ``n_envs`` in a doubling sweep (default 4, 8, 16, 32, 64), builds a
fresh ``SandboxManipulation(n_envs=n_envs)``, times several batches of "one
candidate push per env, in parallel, at rollout (reduced-settle) fidelity" —
the operation ``oracle_mpc`` performs once per optimizer iteration — and
reports wall-clock seconds/batch and candidates/sec.

Usage
-----
Run as a module from the REPO ROOT (not as a bare script, and not with cwd
inside Genesis/) — this is what lets ``sandbox_manipulation_clean.py``'s
``from .utilities.materials import *`` resolve, with no sys.path/PYTHONPATH
manipulation needed:

    python -m Genesis.benchmark_n_envs
    python -m Genesis.benchmark_n_envs --envs 4 8 16 32 --n-trials 5
    python -m Genesis.benchmark_n_envs --max-envs 64 --n-particles 20

(configs/basic.yaml is resolved relative to this file's own path, so it's
found regardless of the working directory you launch from.)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Genesis.sandbox_manipulation_clean import SandboxManipulation


def _build_config(n_particles: int) -> dict:
    base_dir = Path(__file__).parent
    with open(base_dir / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    # basic.yaml leaves these as None (normally filled in per-experiment by
    # data_collection_clean.py's sweep loop) — set representative defaults.
    cfg["material"]["shape"]         = "cube"
    cfg["material"]["particle_size"] = 0.008
    cfg["material"]["n_particles"]   = n_particles
    cfg["material"]["density"]       = 1000.0
    cfg["material"]["friction"]      = 0.3
    cfg["box"]["friction"]           = 0.3
    return cfg


def _time_one_rollout_batch(sim: SandboxManipulation, settle_steps: int) -> float:
    """Time one 'candidate push per env, in parallel' batch — mirrors the
    inner loop of GenesisOracleEnv.rollout_candidates for a single
    look-ahead step (execute_action + reduced-settle update_material_state)."""
    action_starts, action_stops, angles = sim.generate_action_samples(1)
    p_start = action_starts[:, 0, :]
    p_stop  = action_stops[:, 0, :]
    angle   = angles[:, 0]

    sim._settle_steps = settle_steps
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    sim.execute_action(p_start, p_stop, angle)
    sim.update_material_state()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time() - t0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--envs', nargs='+', type=int, default=None,
                    help='explicit n_envs values to test (default: 4 8 16 32 64)')
    p.add_argument('--max-envs', type=int, default=None,
                    help='skip values above this (e.g. to stop before OOM)')
    p.add_argument('--n-particles', type=int, default=20)
    p.add_argument('--settle-steps', type=int, default=50,
                    help='settle steps per batch (rollout-time value — '
                         'half of the dataset.yaml default of 100)')
    p.add_argument('--n-warmup', type=int, default=1,
                    help='untimed warmup batches (kernel compilation, caches)')
    p.add_argument('--n-trials', type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    envs_to_test = args.envs or [4, 8, 16, 32, 64]
    if args.max_envs is not None:
        envs_to_test = [n for n in envs_to_test if n <= args.max_envs]

    results = []
    for n_envs in envs_to_test:
        print(f"\n=== n_envs={n_envs} ===", flush=True)
        cfg = _build_config(args.n_particles)
        sim = SandboxManipulation(config=cfg, n_envs=n_envs, debug=False)
        try:
            sim.build()
            sim.shuffle_particles()
            sim.update_material_state()

            for _ in range(args.n_warmup):
                _time_one_rollout_batch(sim, args.settle_steps)

            times = [_time_one_rollout_batch(sim, args.settle_steps)
                     for _ in range(args.n_trials)]
            mean_t = float(np.mean(times))
            cand_per_sec = n_envs / mean_t
            print(f"  {mean_t:.3f}s/batch  ->  {cand_per_sec:.1f} candidates/sec  "
                  f"(min={min(times):.3f}s max={max(times):.3f}s over {args.n_trials} trials)")
            results.append((n_envs, mean_t, cand_per_sec))
        except Exception as e:
            print(f"  FAILED at n_envs={n_envs}: {e}")
            break
        finally:
            sim.destroy()

    print("\n=== summary ===")
    print(f"{'n_envs':>8}  {'s/batch':>10}  {'cand/sec':>10}")
    for n_envs, mean_t, cps in results:
        print(f"{n_envs:>8}  {mean_t:>10.3f}  {cps:>10.1f}")


if __name__ == '__main__':
    main()
