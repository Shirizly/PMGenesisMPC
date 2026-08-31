"""Does the linear share of predictable dynamics rise with local packing?

The pile hypothesis says the paper's linear-model claim should hold better on a
dense pile than on sparse scattered objects. That can be tested WITHOUT new
data: in the scattered dataset some pushes happen to strike locally clustered
cubes and some strike isolated ones. If the linear share of predictable
displacement rises with local packing density, the hypothesis has support before
a single piled transition is collected.

Density is measured as the mean nearest-neighbour distance among the particles in
the blade's swath, in cube widths -- small = tightly packed, large = isolated.
"""
from __future__ import annotations
import argparse
import numpy as np, torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from variance_decomposition import DEFAULT_GLOB, band_frame, features, load, r2, MM, BAND_HALF, CUBE

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default=DEFAULT_GLOB)
ap.add_argument("--bins", type=int, default=3)
ap.add_argument("--by", default="packing", choices=["packing", "contact"],
                help="stratify by local packing (pile hypothesis) or by how much "
                     "material the blade meets (perturbation-size hypothesis: a "
                     "linear operator is a first-order approximation, so it "
                     "should suit SMALL perturbations best -- which would imply "
                     "shorter pushes rather than denser piles)")
a = ap.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
s0, s1, quat, ps, pe, run, nf = load(a.glob)
s0, s1, quat, ps, pe = (t.to(dev) for t in (s0, s1, quat, ps, pe))
Xo, Xp, Xq, inband = features(s0, quat, ps, pe)
runs = run.numpy()

# local packing: mean nearest-neighbour distance among swath particles
big = 1e3
dmat = (s0[:, :, None, :2] - s0[:, None, :, :2]).norm(dim=-1)
dmat = dmat + torch.eye(s0.shape[1], device=dev)[None] * big
nn = dmat.min(dim=-1).values
cnt = inband.float().sum(1).clamp_min(1)
packing = (torch.where(inband, nn, torch.zeros_like(nn)).sum(1) / cnt / CUBE).cpu().numpy()
if a.by == "contact":
    # exogenous perturbation size: particles the blade will sweep. Stratifying
    # by the OUTCOME (actual displacement) would be circular -- it restricts the
    # range of the very quantity being predicted.
    packing = inband.float().sum(1).cpu().numpy()

disp = (s1 - s0)[..., :2].norm(dim=-1)
y = ((disp * inband.float()).sum(1) / cnt * MM).cpu().numpy()

gkf = GroupKFold(n_splits=5)
def fit(X, yy, groups):
    rl, rg = [], []
    for tr, te in gkf.split(X, yy, groups=groups):
        sc = X[tr].std(0); sc[sc == 0] = 1; mu = X[tr].mean(0)
        A, B = (X[tr] - mu) / sc, (X[te] - mu) / sc
        rl.append(r2(yy[te], RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(A, yy[tr]).predict(B)))
        rg.append(r2(yy[te], HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, random_state=0).fit(A, yy[tr]).predict(B)))
    return float(np.mean(rl)), float(np.mean(rg))

qs = np.quantile(packing, np.linspace(0, 1, a.bins + 1))
_unit = "cube widths (1.0=touching)" if a.by == "packing" else "particles in swath"
print(f"n={len(y)}  stratifying by {a.by}: {_unit}")
print(f"  overall median {np.median(packing):.2f}")
print(f"\n{'packing stratum':>22s} {'n':>6s} {'NN dist':>9s} {'linear R2':>10s} "
      f"{'boosted R2':>11s} {'linear share':>13s}")
print("-" * 78)
for i in range(a.bins):
    m = (packing >= qs[i]) & (packing <= qs[i + 1] if i == a.bins - 1 else packing < qs[i + 1])
    if m.sum() < 200 or len(np.unique(runs[m])) < 5:
        print(f"  stratum {i}: too few samples/runs"); continue
    L, G = fit(Xo[m], y[m], runs[m])
    share = f"{100 * L / G:.0f}%" if G > 0.02 else "n/a"
    if a.by == "packing":
        lbl = 'densest' if i == 0 else ('sparsest' if i == a.bins - 1 else 'middle')
    else:
        lbl = 'smallest perturb' if i == 0 else ('largest perturb' if i == a.bins - 1 else 'middle')
    print(f"{lbl:>22s} {int(m.sum()):6d} {packing[m].mean():9.2f} {L:10.3f} "
          f"{G:11.3f} {share:>13s}")
if a.by == "packing":
    print("\nIf the linear share RISES toward the densest stratum, the pile "
          "hypothesis\nhas support: linearity suits packed material better than "
          "isolated objects.")
else:
    print("\nIf the linear share RISES toward the SMALLEST perturbation, "
          "linearity suits\nsmall pushes -- implying shorter pushes, not denser "
          "piles, is the lever.")
