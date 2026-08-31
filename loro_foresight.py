"""Leave-one-run-out CV for the switched (per-contact-bin) foresight operator.

Why LORO and not a single holdout: with 8 data files a single holdout is one
file (320 samples) and the split-to-split std of rms is ~0.004, which swamps
the ~0.001-0.003 effects being compared. A single-split "win" here is noise --
measured the hard way (a gating variant won on seed 0 and lost 3 of 6 seeds).
Every model is scored on every fold and compared PAIRED, fold by fold.

The primary baseline is `identity` -- persistence pushed through the SAME warp
(and blur) as the operator -- not raw persistence. The operator and identity
then pay identical resampling costs, so the comparison isolates what the
operator learns. Raw persistence is reported alongside as the task-level floor.

Usage:
    PYTHONPATH=. python loro_foresight.py [--crop 0.5] [--res 32] [--blur 1.0]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, contact_score, fit_operator,
    fit_operator_nonneg, predict_heuristic, predict_world, swept_region_mask,
)

BIN_NAMES = {2: ["low", "high"], 3: ["barely", "mildly", "significantly"]}


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma * sigma))
    g = g / g.sum()
    x = F.conv2d(x.unsqueeze(1), g.view(1, 1, 1, -1), padding=(0, k // 2))
    return F.conv2d(x, g.view(1, 1, -1, 1), padding=(k // 2, 0)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="configs/dataset/genesis_foresight_L040.yaml")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--crop", type=float, default=0.5)
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--bins", type=int, default=3)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="shrinkage strength; the target is A=I, not A=0")
    args = ap.parse_args()

    d = load_transition_arrays(args.dataset, "train")
    H, W = d.occ_t.shape[-2:]
    d.occ_t, d.occ_t1 = gaussian_blur(d.occ_t, args.blur), gaussian_blur(d.occ_t1, args.blur)
    s, e = actions_to_pixels(d.actions, d.workspace_min, d.workspace_max, (H, W))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    o0, o1, s, e = d.occ_t.to(dev), d.occ_t1.to(dev), s.to(dev), e.to(dev)
    eid = d.episode_ids.to(dev)
    runs = eid.unique()
    plate = 0.04 / 0.128 * W
    R, CR, B = args.res, args.crop, args.bins
    D = R * R
    names = BIN_NAMES.get(B, [f"bin{i}" for i in range(B)])

    print(f"dataset {args.dataset}")
    print(f"  n={len(o0)} transitions, {len(runs)} runs, grid {H}x{W}")
    print(f"  operator {D}x{D}, canonical window {CR * W:.0f}px -> {R}x{R}"
          f"{' (1:1 pixel scale)' if abs(CR * W - R) < 1 else ''}, blur sigma={args.blur}")
    per_bin_M = len(o0) * (len(runs) - 1) / len(runs) / B
    print(f"  least squares per bin: M~{per_bin_M:.0f} vs D={D}  ->  M/D={per_bin_M / D:.2f}"
          f"  {'OK' if per_bin_M / D >= 2 else '** UNDERDETERMINED, needs more data **'}")

    def rms_per_sample(p, truth, reg):
        n = p.shape[0]
        dd = ((p - truth) * reg).reshape(n, -1)
        return (dd.pow(2).sum(1) / reg.reshape(n, -1).sum(1).clamp_min(1)).sqrt()

    fold_rms: dict[str, list] = {}
    bin_rms: dict[tuple, list] = {}

    for r in runs:
        mte = eid == r
        mtr = ~mte
        ote, o1te, ste, ete = o0[mte], o1[mte], s[mte], e[mte]
        reg = swept_region_mask(ste, ete, (H, W),
                                half_width_px=0.5 * plate + 2.0, pad_px=0.5 * plate)

        Y0 = canonicalise(o0[mtr], s[mtr], e[mtr], R, CR).reshape(int(mtr.sum()), -1).T
        Y1 = canonicalise(o1[mtr], s[mtr], e[mtr], R, CR).reshape(int(mtr.sum()), -1).T
        A = fit_operator_nonneg(Y0, Y1, max_iter=args.iters)
        Ai = fit_operator_nonneg(Y0, Y1, max_iter=args.iters, ridge=args.ridge) \
            if args.ridge > 0 else None

        eye = torch.eye(D, device=dev)
        preds = {
            "persistence (raw)": ote,
            "identity (warp only)": predict_world(eye, ote, ste, ete, R, (H, W), CR),
            "linear-nonneg": predict_world(A, ote, ste, ete, R, (H, W), CR),
            "ridge->0 (1e-2)": predict_world(
                fit_operator(Y0, Y1, 1e-2, toward_identity=False),
                ote, ste, ete, R, (H, W), CR),
            "ridge->I (1e-2)": predict_world(
                fit_operator(Y0, Y1, 1e-2, toward_identity=True),
                ote, ste, ete, R, (H, W), CR),
            "heur-cumulative": predict_heuristic(
                "cumulative", ote.cpu(), ste.cpu(), ete.cpu()).to(dev),
        }

        # switched: one operator per contact bin, edges from TRAIN only
        ctr = contact_score(o0[mtr], s[mtr], e[mtr], (H, W), plate)
        cte = contact_score(ote, ste, ete, (H, W), plate)
        edges = torch.quantile(ctr, torch.linspace(0, 1, B + 1, device=dev)[1:-1])
        btr, bte = torch.bucketize(ctr, edges), torch.bucketize(cte, edges)
        sw = ote.clone()
        for b in range(B):
            a, t = btr == b, bte == b
            if int(a.sum()) < 20 or int(t.sum()) == 0:
                continue
            Yb0 = canonicalise(o0[mtr][a], s[mtr][a], e[mtr][a], R, CR
                               ).reshape(int(a.sum()), -1).T
            Yb1 = canonicalise(o1[mtr][a], s[mtr][a], e[mtr][a], R, CR
                               ).reshape(int(a.sum()), -1).T
            Ab = fit_operator_nonneg(Yb0, Yb1, max_iter=args.iters)
            sw[t] = predict_world(Ab, ote[t], ste[t], ete[t], R, (H, W), CR)
        preds["switched-nonneg"] = sw
        if Ai is not None:
            preds[f"nonneg,shrink->I {args.ridge:g}"] = predict_world(
                Ai, ote, ste, ete, R, (H, W), CR)
            swi = ote.clone()
            for b in range(B):
                a, t = btr == b, bte == b
                if int(a.sum()) < 20 or int(t.sum()) == 0:
                    continue
                Yb0 = canonicalise(o0[mtr][a], s[mtr][a], e[mtr][a], R, CR
                                   ).reshape(int(a.sum()), -1).T
                Yb1 = canonicalise(o1[mtr][a], s[mtr][a], e[mtr][a], R, CR
                                   ).reshape(int(a.sum()), -1).T
                Ab = fit_operator_nonneg(Yb0, Yb1, max_iter=args.iters,
                                         ridge=args.ridge)
                swi[t] = predict_world(Ab, ote[t], ste[t], ete[t], R, (H, W), CR)
            preds[f"switched,shrink->I {args.ridge:g}"] = swi

        for name, p in preds.items():
            v = rms_per_sample(p, o1te, reg)
            fold_rms.setdefault(name, []).append(v.mean())
            for b in range(B):
                m = bte == b
                if int(m.sum()):
                    bin_rms.setdefault((name, b), []).append(v[m].mean())

    ref = torch.stack(fold_rms["identity (warp only)"])
    raw = torch.stack(fold_rms["persistence (raw)"])
    print(f"\n=== leave-one-run-out, {len(runs)} folds, swept region ===")
    print(f"{'model':22s} {'rms mean':>10s} {'sd':>8s} {'vs identity':>12s} "
          f"{'folds won':>10s} {'vs raw persist':>15s}")
    print("-" * 82)
    for name in fold_rms:
        t = torch.stack(fold_rms[name])
        di, dr = ref - t, raw - t
        print(f"{name:22s} {float(t.mean()):10.5f} {float(t.std()):8.5f} "
              f"{float(di.mean()):+12.5f} {int((di > 0).sum()):>5d}/{len(runs):<4d} "
              f"{float(dr.mean()):+15.5f}")

    print(f"\n=== per bin: rms, and gap to identity-in-that-bin ===")
    print(f"{'model':22s} " + "".join(f"{n[:13]:>18s}" for n in names))
    print("-" * (22 + 18 * B))
    for name in fold_rms:
        row = f"{name:22s} "
        for b in range(B):
            t = torch.stack(bin_rms[(name, b)])
            g = torch.stack(bin_rms[("identity (warp only)", b)]) - t
            row += f"{float(t.mean()):9.5f}{float(g.mean()):+9.5f}"
        print(row)
    print("\nSecond number in each cell is the gap to identity in that bin "
          "(+ = better than warped persistence).")


if __name__ == "__main__":
    main()
