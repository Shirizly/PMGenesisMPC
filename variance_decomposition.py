"""Is per-push displacement variability irreducible, or just invisible to the
occupancy grid?

The linear operator reduces one-step error by only 3-4% even though a 40 mm push
moves the cubes in its path ~19 mm. Displacement per push runs 0-49 mm (median
16, p95 42), so the operator captures a conditional mean that is weakly
informative about any individual push. This asks WHY:

  (a) genuinely stochastic given a top-down view -- then no model over occupancy
      can do better and the ceiling is real; or
  (b) predictable from information the particle state carries but the 64x64
      binary occupancy destroys (sub-pixel position, packing, cube yaw) -- then
      the representation is the limitation and is fixable.

Method: regress per-push outcomes on three nested feature sets --
  OCC   : derivable from the occupancy grid + action (what the model sees)
  PART  : exact particle positions (what occupancy quantises away)
  PART+ : plus cube orientation (absent from occupancy entirely)
-- with a linear model and a gradient-boosted one, plus a pure-noise control to
calibrate what R^2 looks like at zero signal. Grouped train/test split by run.
"""
from __future__ import annotations
import glob, math
import numpy as np
import torch

CUBE = 0.005
BAND_HALF = 0.021          # plate half-length + margin, metres
MM = 1000.0


def load():
    S, E, P, Q, A, R = [], [], [], [], [], []
    files = sorted(glob.glob('Genesis/data/foresight/L040*/cube/n50/size0.005/_*_data.pt'))
    for i, f in enumerate(files):
        d = torch.load(f, weights_only=False)
        S.append(d['states'][..., :3]); E.append(d['states_'][..., :3])
        Q.append(d['states'][..., 3:7])
        P.append(d['p_starts']); A.append(d['p_stops'])
        R.append(torch.full((d['states'].shape[0],), i))
    return (torch.cat(S), torch.cat(E), torch.cat(Q), torch.cat(P),
            torch.cat(A), torch.cat(R), len(files))


def band_frame(s0, ps, pe):
    """Per-particle coordinates in the push frame: along-axis and lateral."""
    d = pe[:, :2] - ps[:, :2]
    L = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    u = d / L
    rel = s0[..., :2] - ps[:, None, :2]
    along = (rel * u[:, None, :]).sum(-1)
    lat = rel[..., 0] * (-u[:, None, 1]) + rel[..., 1] * u[:, None, 0]
    return along, lat, L.squeeze(-1), u


def features(s0, quat, ps, pe):
    """Return (occ_features, part_features, partplus_features, names)."""
    along, lat, L, u = band_frame(s0, ps, pe)
    inband = (along >= -0.005) & (along <= L[:, None] + 0.005) & (lat.abs() <= BAND_HALF)
    n = s0.shape[0]
    f_occ, f_part, f_pp, no, npr, npp = [], [], [], [], [], []

    def add(lst, names, v, nm):
        lst.append(v.reshape(n, -1)); names.append(nm)

    # ---- OCC: things a 2 mm binary grid preserves ----
    # mass profile along the push axis, in 2 mm bins (grid resolution)
    edges = torch.arange(-0.01, 0.061, 0.004, device=s0.device)
    prof = torch.stack([((along >= edges[i]) & (along < edges[i + 1]) &
                         (lat.abs() <= BAND_HALF)).float().sum(1)
                        for i in range(len(edges) - 1)], dim=-1)
    add(f_occ, no, prof, 'along-profile(18)')
    # lateral profile
    lprof = torch.stack([((lat >= -BAND_HALF + j * 0.014) & (lat < -BAND_HALF + (j + 1) * 0.014)
                          & (along >= 0) & (along <= L[:, None])).float().sum(1)
                         for j in range(3)], dim=-1)
    add(f_occ, no, lprof, 'lateral-profile(3)')
    add(f_occ, no, inband.float().sum(1), 'n-in-band')
    add(f_occ, no, torch.ones(n, device=s0.device) * s0.shape[1], 'total-pile')
    # geometry of the action relative to the tray (walls) -- fully in the grid
    add(f_occ, no, (0.064 - pe[:, :2].abs()).min(dim=-1).values, 'end-dist-to-wall')
    add(f_occ, no, (0.064 - ps[:, :2].abs()).min(dim=-1).values, 'start-dist-to-wall')
    add(f_occ, no, torch.atan2(u[:, 1], u[:, 0]), 'push-heading')
    # mass just beyond the blade's stopping point (blocking material)
    add(f_occ, no, ((along > L[:, None]) & (along < L[:, None] + 0.02)
                    & (lat.abs() <= BAND_HALF)).float().sum(1), 'mass-ahead')

    # ---- PART: exact positions, i.e. what quantisation destroys ----
    big = 1e3
    a_in = torch.where(inband, along, torch.full_like(along, big))
    l_in = torch.where(inband, lat.abs(), torch.full_like(lat, big))
    cnt = inband.float().sum(1).clamp_min(1)
    add(f_part, npr, (a_in.clamp(max=1.0).sum(1) / cnt), 'mean-along')
    add(f_part, npr, a_in.min(dim=1).values.clamp(max=1.0), 'min-along')
    add(f_part, npr, (l_in.clamp(max=1.0).sum(1) / cnt), 'mean-|lat|')
    # sub-pixel: fractional part of position within a 2 mm cell
    frac = ((s0[..., :2] / 0.002) % 1.0)
    add(f_part, npr, torch.where(inband[..., None], frac, torch.zeros_like(frac)).sum(1) / cnt[:, None],
        'subpixel-frac(2)')
    # packing: nearest-neighbour distance among band particles
    dmat = (s0[:, :, None, :2] - s0[:, None, :, :2]).norm(dim=-1)
    dmat = dmat + torch.eye(s0.shape[1], device=s0.device)[None] * big
    nn = dmat.min(dim=-1).values
    add(f_part, npr, torch.where(inband, nn, torch.zeros_like(nn)).sum(1) / cnt, 'mean-nn-dist')
    add(f_part, npr, torch.where(inband, nn, torch.full_like(nn, big)).min(dim=1).values.clamp(max=1.0),
        'min-nn-dist')
    # how many neighbours within 1.2 cube widths = contact chains
    add(f_part, npr, ((dmat < 1.2 * CUBE).float() * inband[:, :, None].float()).sum((1, 2)),
        'n-contacts')

    # ---- PART+: orientation, absent from occupancy entirely ----
    yaw = 2.0 * torch.atan2(quat[..., 3], quat[..., 0])
    add(f_pp, npp, (torch.cos(4 * yaw) * inband.float()).sum(1) / cnt, 'yaw-cos4')
    add(f_pp, npp, (torch.sin(4 * yaw) * inband.float()).sum(1) / cnt, 'yaw-sin4')
    rel_yaw = yaw - torch.atan2(u[:, 1], u[:, 0])[:, None]
    add(f_pp, npp, (torch.cos(4 * rel_yaw) * inband.float()).sum(1) / cnt, 'relyaw-cos4')

    cat = lambda L: torch.cat(L, dim=-1).cpu().numpy()
    return cat(f_occ), cat(f_part), cat(f_pp), inband


def r2(y, yh):
    return 1.0 - float(((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    s0, s1, quat, ps, pe, run, nfiles = load()
    s0, s1, quat, ps, pe = (t.to(dev) for t in (s0, s1, quat, ps, pe))
    print(f"n={len(ps)} transitions from {nfiles} runs, {s0.shape[1]} particles")

    Xo, Xp, Xq, inband = features(s0, quat, ps, pe)
    # targets
    disp = (s1 - s0)[..., :2].norm(dim=-1)
    cnt = inband.float().sum(1).clamp_min(1)
    y_mean = ((disp * inband.float()).sum(1) / cnt * MM).cpu().numpy()   # mm
    # forward component of the band's displacement
    d = pe[:, :2] - ps[:, :2]
    u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    fwd = ((s1 - s0)[..., :2] * u[:, None, :]).sum(-1)
    y_fwd = ((fwd * inband.float()).sum(1) / cnt * MM).cpu().numpy()
    y_max = ((disp * inband.float()).max(dim=1).values * MM).cpu().numpy()
    runs = run.numpy()

    rng = np.random.default_rng(0)
    Xnoise = rng.normal(size=(len(ps), Xo.shape[1]))

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    sets = [("noise control", Xnoise),
            ("OCC (grid-visible)", Xo),
            ("OCC + PART (exact pos)", np.hstack([Xo, Xp])),
            ("OCC + PART + yaw", np.hstack([Xo, Xp, Xq]))]
    targets = [("mean displacement in band (mm)", y_mean),
               ("mean FORWARD displacement (mm)", y_fwd),
               ("max displacement in band (mm)", y_max)]

    gkf = GroupKFold(n_splits=min(5, nfiles))
    for tname, y in targets:
        print(f"\n=== target: {tname} ===")
        print(f"    spread: mean {y.mean():.2f}  sd {y.std():.2f}  "
              f"p5 {np.percentile(y,5):.2f}  p95 {np.percentile(y,95):.2f}")
        print(f"    {'feature set':26s} {'linear R2':>10s} {'boosted R2':>11s}")
        for sname, X in sets:
            rl, rg = [], []
            for tr, te in gkf.split(X, y, groups=runs):
                sc = X[tr].std(0); sc[sc == 0] = 1
                mu = X[tr].mean(0)
                Xtr, Xte = (X[tr] - mu) / sc, (X[te] - mu) / sc
                rl.append(r2(y[te], RidgeCV(alphas=np.logspace(-3, 4, 20))
                             .fit(Xtr, y[tr]).predict(Xte)))
                rg.append(r2(y[te], HistGradientBoostingRegressor(
                    max_iter=250, learning_rate=0.06, random_state=0)
                    .fit(Xtr, y[tr]).predict(Xte)))
            print(f"    {sname:26s} {np.mean(rl):10.3f} {np.mean(rg):11.3f}")


if __name__ == '__main__':
    main()
