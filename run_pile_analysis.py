"""One-shot comparison: piled+contact-sampled data vs scattered+blind data.

Runs the three analyses that matter, on both datasets, and prints them side by
side:

  1. physical units  -- error as a share of the change that actually occurred
                        (interpret_foresight.py), because occupancy-per-pixel
                        differences are uninterpretable;
  2. variance decomposition -- how much of per-push displacement is predictable
                        at all, and how much of that is reachable LINEARLY. This
                        is the number that decides whether the paper's central
                        claim holds in a given regime;
  3. leave-one-run-out -- the operator against warped and raw persistence.

Usage:
    python run_pile_analysis.py --pile-glob '...' --pile-cfg '...'
"""
from __future__ import annotations
import argparse, subprocess, sys

def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + ("\n" + r.stderr if r.returncode else "")
    print(out, flush=True)
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--pile-cfg", default="configs/dataset/genesis_foresight_pile30.yaml")
ap.add_argument("--pile-glob",
                default="Genesis/data/foresight/pile30/cube/n30/size0.005/_*_data.pt")
ap.add_argument("--pile-n", type=int, default=30)
ap.add_argument("--skip-scattered", action="store_true")
ap.add_argument("--folds", type=int, default=6)
a = ap.parse_args()

print("=" * 78)
print("A. VARIANCE DECOMPOSITION -- is displacement predictable, and linearly?")
print("=" * 78)
run([sys.executable, "-u", "variance_decomposition.py", "--glob", a.pile_glob,
     "--label", "PILED 30 + contact-sampled pushes"])
if not a.skip_scattered:
    run([sys.executable, "-u", "variance_decomposition.py",
         "--glob", "Genesis/data/foresight/L040*/cube/n50/size0.005/_*_data.pt",
         "--label", "SCATTERED 50 + blind pushes (reference)"])

print("=" * 78)
print("B. PHYSICAL UNITS -- error as a share of the real change")
print("=" * 78)
run([sys.executable, "-u", "interpret_foresight.py", "--dataset", a.pile_cfg,
     "--n-particles", str(a.pile_n), "--folds", str(a.folds)])

print("=" * 78)
print("C. LEAVE-ONE-RUN-OUT -- operator vs warped and raw persistence")
print("=" * 78)
run([sys.executable, "-u", "loro_foresight.py", "--dataset", a.pile_cfg,
     "--res", "32", "--crop", "0.5", "--blur", "1.0", "--bins", "3",
     "--folds", str(a.folds), "--ridge", "1.0", "--iters", "2000"])
