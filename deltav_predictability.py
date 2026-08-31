"""How predictable is the CONTROL-relevant quantity, and how much of it linearly?

variance_decomposition.py asks this of per-push displacement. But a greedy
Lyapunov controller does not care about displacement; it cares about dV, the
change in the image-space cost. If dV is highly predictable AND largely
linearly predictable, a linear operator is a perfectly good controller even
though its predicted images are poor -- which would reconcile our per-pixel
results with the paper's control result.

V is computed on the PARTICLES rather than on an occupancy grid (their eq. 3
rather than eq. 4), so the number is free of grid resolution and of the warp:

    V(X) = mean_i  d(p_i, S_goal)

The feature sets are the same nested OCC / +PART / +yaw sets as
variance_decomposition.py, so the linear-vs-nonlinear split is directly
comparable between the two targets.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from variance_decomposition import DEFAULT_GLOB, features, load, r2

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default=DEFAULT_GLOB)
ap.add_argument("--label", default="")
ap.add_argument("--half-extent", type=float, default=0.064,
                help="tray half-width in metres (goal geometry is defined on it)")
a = ap.parse_args()
if a.label:
    print(f"### {a.label}")

dev = "cuda" if torch.cuda.is_available() else "cpu"
s0, s1, quat, ps, pe, run, nf = load(a.glob)
s0, s1, quat, ps, pe = (t.to(dev) for t in (s0, s1, quat, ps, pe))
Xo, Xp, Xq, inband = features(s0, quat, ps, pe)
runs = run.numpy()
E = a.half_extent


def dist_to_goal(xy, goal):
    """Distance from each particle to the target set, metres."""
    x, y = xy[..., 0], xy[..., 1]
    if goal == "center":          # centred square, half-width E/2
        h = E / 2
        return (x.abs() - h).clamp_min(0).square().add(
            (y.abs() - h).clamp_min(0).square()).sqrt()
    if goal == "corner":          # quadrant x<0, y<0
        return x.clamp_min(0).square().add(y.clamp_min(0).square()).sqrt()
    if goal == "stripe":          # band |y| < E/4
        return (y.abs() - E / 4).clamp_min(0)
    if goal == "point":           # collect to the origin (hardest)
        return xy.norm(dim=-1)
    raise ValueError(goal)


gkf = GroupKFold(n_splits=5)
sets = [("OCC (grid-visible)", Xo),
        ("OCC + PART (exact pos)", np.hstack([Xo, Xp])),
        ("OCC + PART + yaw", np.hstack([Xo, Xp, Xq]))]

print(f"n={s0.shape[0]} transitions, {nf} runs, tray half-width {E * 1000:.0f} mm")
for goal in ("center", "corner", "stripe", "point"):
    v0 = dist_to_goal(s0[..., :2], goal).mean(dim=1)
    v1 = dist_to_goal(s1[..., :2], goal).mean(dim=1)
    y = ((v1 - v0) * 1000).cpu().numpy()          # mm of mean distance change
    print(f"\n=== goal '{goal}' : dV in mm of mean particle-to-goal distance ===")
    print(f"    dV: mean {y.mean():+.3f}  sd {y.std():.3f}  "
          f"helpful (dV<0) {100 * (y < 0).mean():.0f}%")
    print(f"    {'feature set':26s} {'linear R2':>10s} {'boosted R2':>11s} {'linear share':>13s}")
    for sname, X in sets:
        rl, rg = [], []
        for tr, te in gkf.split(X, y, groups=runs):
            sc = X[tr].std(0); sc[sc == 0] = 1; mu = X[tr].mean(0)
            A, B = (X[tr] - mu) / sc, (X[te] - mu) / sc
            rl.append(r2(y[te], RidgeCV(alphas=np.logspace(-3, 4, 20))
                         .fit(A, y[tr]).predict(B)))
            rg.append(r2(y[te], HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.06, random_state=0)
                .fit(A, y[tr]).predict(B)))
        L, G = float(np.mean(rl)), float(np.mean(rg))
        share = f"{100 * L / G:.0f}%" if G > 0.02 else "n/a"
        print(f"    {sname:26s} {L:10.3f} {G:11.3f} {share:>13s}")
print("\n'linear share' = what fraction of the achievable prediction a LINEAR "
      "model captures.\nHigh share => the paper's linear-model claim is "
      "appropriate for this regime.")

# ---------------------------------------------------------------------------
# What does linearity cost in CONTROL terms, not R^2 terms?
# ---------------------------------------------------------------------------
# R^2 is not the currency a controller spends. Convert both models' held-out
# predictions into the quantity a greedy controller actually realises: pick the
# best of K candidates by predicted dV, and measure the fraction of an oracle's
# advantage over a random pick that you capture.
print("\n" + "=" * 74)
print("CONTROL COST OF LINEARITY: best-of-K action selection on held-out data")
print("=" * 74)


def slate_utility(pred, true, K, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(true), size=(4000, K))
    ct, cp = true[idx], pred[idx]
    chosen = np.take_along_axis(ct, cp.argmin(axis=1, keepdims=True), axis=1)[:, 0]
    oracle = ct.min(axis=1)
    rand = ct.mean(axis=1)
    return float(np.mean((rand - chosen) / np.maximum(rand - oracle, 1e-9)))


print(f"{'goal':10s} {'model':10s} {'R2':>7s} {'best-of-4':>10s} {'best-of-16':>11s} "
      f"{'dV realised (mm)':>18s}")
print("-" * 74)
for goal in ("center", "corner", "stripe", "point"):
    v0 = dist_to_goal(s0[..., :2], goal).mean(dim=1)
    v1 = dist_to_goal(s1[..., :2], goal).mean(dim=1)
    y = ((v1 - v0) * 1000).cpu().numpy()
    X = Xo
    oof = {"linear": np.zeros_like(y), "boosted": np.zeros_like(y)}
    for tr, te in gkf.split(X, y, groups=runs):
        sc = X[tr].std(0); sc[sc == 0] = 1; mu = X[tr].mean(0)
        A, B = (X[tr] - mu) / sc, (X[te] - mu) / sc
        oof["linear"][te] = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(A, y[tr]).predict(B)
        oof["boosted"][te] = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, random_state=0).fit(A, y[tr]).predict(B)
    for nm in ("linear", "boosted"):
        r = r2(y, oof[nm])
        u4, u16 = slate_utility(oof[nm], y, 4), slate_utility(oof[nm], y, 16)
        # dV actually realised by best-of-16 selection, in mm
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(y), size=(4000, 16))
        realised = float(np.mean(np.take_along_axis(
            y[idx], oof[nm][idx].argmin(axis=1, keepdims=True), axis=1)[:, 0]))
        print(f"{goal:10s} {nm:10s} {r:7.3f} {u4:10.3f} {u16:11.3f} {realised:+18.3f}")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(4000, 16))
    print(f"{'':10s} {'oracle':10s} {1.0:7.3f} {1.0:10.3f} {1.0:11.3f} "
          f"{float(np.mean(y[idx].min(axis=1))):+18.3f}")
    print(f"{'':10s} {'random':10s} {0.0:7.3f} {0.0:10.3f} {0.0:11.3f} "
          f"{float(np.mean(y[idx].mean(axis=1))):+18.3f}")
print("\nNegative dV = the pile got closer to the goal. A controller that picks "
      "the best of 16\ncandidates realises the dV in the last column; oracle "
      "and random bracket it.")
