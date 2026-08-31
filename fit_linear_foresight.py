"""
Fit and falsify the switched-linear visual-foresight operator at PIXEL level.

Suh & Tedrake 2020 (arXiv:2002.09093) §3.1: predict occupancy-image dynamics
with an action-dependent linear map, applied in the canonical push frame so the
action collapses to push length alone:

    I_{k+1} ~= blend( T^-1( A_l . T(I_k) ), I_k, M )

This script fits the single operator `A_l` for a dataset collected at ONE push
length with perpendicular pushes (see
Genesis/configs/collection_foresight_single_operator.yaml) and answers the
stage-2 gate question of docs/linear_visual_foresight_baseline.md §10:

    does the pixel operator beat persistence AND the existing geometric
    push heuristics on held-out one-step error?

Everything is reported in the WORLD frame after the round trip, because that is
what the controller actually sees — a canonical-frame-only number would flatter
the operator by hiding warp loss.

Usage
-----
    python fit_linear_foresight.py configs/dataset/genesis_foresight_L040.yaml \
        --res 32 --constraint ridge
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from dmdc_baseline import load_transition_arrays, split_by_episode
from transforms.functional import (
    blend_push_prediction, from_push_frame, push_frame_validity_mask,
    to_push_frame,
)


# =====================================================================
# 1. World action -> pixel coordinates
# =====================================================================
#
# Dataset grid convention, MEASURED rather than assumed: dim0 (rows) tracks
# world_x and dim1 (cols) tracks world_y. Note this is the transpose of what
# INTERFACES.md 4.1 states for "dataset/grid convention" -- verified by
# brute-forcing all 24 (point, flip, flip, swap) hypotheses against the
# dataset's own rasterised plate channel: this one lands at 3.4 px median, the
# best alternative at 9.5 px. `verify_pixel_mapping` re-checks it on every run
# so a future convention change fails loudly instead of silently fitting a
# transposed operator.


def actions_to_pixels(actions: torch.Tensor, ws_min, ws_max, grid_res):
    """[sx,sy,ex,ey] metres -> (start_px, end_px) in (col, row) pixels."""
    H, W = int(grid_res[0]), int(grid_res[1])
    x_min, y_min = float(ws_min[0]), float(ws_min[1])
    x_max, y_max = float(ws_max[0]), float(ws_max[1])

    def to_px(vx, vy):
        row = (vx - x_min) / (x_max - x_min) * H - 0.5   # dim0 <- world_x
        col = (vy - y_min) / (y_max - y_min) * W - 0.5   # dim1 <- world_y
        return torch.stack([col, row], dim=-1)           # warp wants (col, row)

    return (to_px(actions[:, 0], actions[:, 1]),
            to_px(actions[:, 2], actions[:, 3]))


def verify_pixel_mapping(data, start_px, end_px, grid_res, n_check=64):
    """Cross-check the world->pixel mapping against the dataset's own
    rasterised plate channel, rather than trusting the convention.

    Returns the median distance in pixels between the push midpoint we compute
    and the centroid of the rasterised action channel. A few pixels is expected
    (the channel draws the plate, not the push segment); tens of pixels means a
    flipped or transposed axis.
    """
    import yaml as _yaml

    from registry.dataset_registry import build_dataset

    cfg = _yaml.safe_load(open(data.cfg_path).read())
    raw = build_dataset(cfg, data.split).raw_dataset
    H, W = int(grid_res[0]), int(grid_res[1])

    d_start, d_mid = [], []
    n = min(n_check, len(raw))
    for i in range(n):
        (inp, _), _ = raw[i]
        plate = inp[1]
        m = plate.sum()
        if float(m) < 1e-6:
            continue
        rows = torch.arange(H, dtype=torch.float32)
        cols = torch.arange(W, dtype=torch.float32)
        cr = float((plate.sum(dim=1) * rows).sum() / m)
        cc = float((plate.sum(dim=0) * cols).sum() / m)
        mid = 0.5 * (start_px[i] + end_px[i])
        d_mid.append(math.hypot(float(mid[0]) - cc, float(mid[1]) - cr))
        d_start.append(math.hypot(float(start_px[i][0]) - cc,
                                  float(start_px[i][1]) - cr))
    if not d_mid:
        return float('nan'), float('nan')
    d_mid.sort(); d_start.sort()
    return d_mid[len(d_mid) // 2], d_start[len(d_start) // 2]


# =====================================================================
# 2. Canonicalise, fit, predict
# =====================================================================


def canonicalise(occ, start_px, end_px, res, batch=512):
    """Warp to the push frame and resample to `res` x `res`."""
    outs = []
    for i in range(0, occ.shape[0], batch):
        sl = slice(i, i + batch)
        outs.append(to_push_frame(occ[sl], start_px[sl], end_px[sl], (res, res)))
    return torch.cat(outs, dim=0)


def fit_operator(Y0, Y1, ridge: float = 0.0):
    """Matrix least squares  A = argmin ||Y1 - A Y0||_F,  closed form.

    Y0, Y1 : (D, M) — columns are vectorised canonical-frame images.

    Their eq. (8) with an optional ridge term. The ridge is not decoration: the
    row decomposition (their eq. 10) gives each of the D output pixels its own
    problem with D unknowns, so at D = 1024 the paper's own 800 training pairs
    left every row underdetermined. Regularisation here is structural.
    """
    D = Y0.shape[0]
    G = Y0 @ Y0.T
    if ridge > 0:
        G = G + ridge * torch.eye(D, dtype=G.dtype, device=G.device)
    return torch.linalg.solve(G.T, (Y1 @ Y0.T).T).T


def fit_operator_nonneg(Y0, Y1, max_iter: int = 300, lr: float = 1.0):
    """Their eq. (9) non-negativity constraint, A >= 0.

    Projected gradient rather than D independent QPs: the row decomposition
    makes the exact QP tractable in principle, but D=1024 scipy.optimize.nnls
    calls of 1024 variables is minutes-to-hours, while projected gradient on
    the whole matrix is seconds on GPU and reaches the same feasible set. The
    objective is convex with a Lipschitz gradient, so this converges to the
    constrained optimum.
    """
    G = Y0 @ Y0.T
    C = Y1 @ Y0.T
    step = lr / float(torch.linalg.eigvalsh(G)[-1])
    A = torch.zeros_like(C)
    for _ in range(max_iter):
        A = (A - step * (A @ G - C)).clamp_min_(0.0)
    return A


def predict_world(A, occ, start_px, end_px, res, grid_res, batch=256):
    """Full pipeline: warp -> apply A -> unwarp -> blend with the original."""
    H, W = grid_res
    outs = []
    for i in range(0, occ.shape[0], batch):
        sl = slice(i, i + batch)
        o, s, e = occ[sl], start_px[sl], end_px[sl]
        canon = to_push_frame(o, s, e, (res, res))
        pred_c = (A @ canon.reshape(canon.shape[0], -1).T).T.reshape(-1, res, res)
        back = from_push_frame(pred_c, s, e, (H, W))
        mask = push_frame_validity_mask(s, e, (H, W), (res, res))
        outs.append(blend_push_prediction(back, o, mask).clamp_(0.0, 1.0))
    return torch.cat(outs, dim=0)


# =====================================================================
# 3. Baselines and metrics
# =====================================================================


def predict_heuristic(name, occ, start_px, end_px, cfg=None, batch=256):
    """Run one registered geometric push heuristic on the same transitions.

    The push models take occ as (B, Nx, Ny) and the action as (x, y) pixels
    (they index rho[y, x] after permuting), so the dataset's (H, W) = (y, x)
    grid is transposed on the way in and back on the way out.
    """
    from model.eulerian_wrapper import build_push_model

    model = build_push_model({"heuristic_type": name, **(cfg or {})})
    outs = []
    for i in range(0, occ.shape[0], batch):
        sl = slice(i, i + batch)
        o = occ[sl].permute(0, 2, 1).contiguous()
        z = torch.zeros((o.shape[0], 1))
        s3 = torch.cat([start_px[sl], z], dim=-1)
        e3 = torch.cat([end_px[sl], z], dim=-1)
        with torch.no_grad():
            p = model(o, s3, e3)
        outs.append(p.permute(0, 2, 1).clamp_(0.0, 1.0))
    return torch.cat(outs, dim=0)


def swept_region_mask(start_px, end_px, grid_res, half_width_px, pad_px):
    """Pixels the push can plausibly affect: the swept rectangle plus a pad.

    A whole-image error is a bad instrument here. The push touches a few
    percent of the grid, persistence is EXACT on the untouched remainder by
    construction, and `blend_push_prediction` additionally forces every model
    to be persistence outside the warp's validity mask. So a whole-image rms
    is ~95% a measurement of agreement about pixels nothing could have
    changed, and it will rank persistence first almost regardless of how good
    a model is inside the swept band. This mask is where the models actually
    differ.
    """
    B = start_px.shape[0]
    H, W = int(grid_res[0]), int(grid_res[1])
    dev = start_px.device
    rows = torch.arange(H, device=dev, dtype=torch.float32)
    cols = torch.arange(W, device=dev, dtype=torch.float32)
    gr, gc = torch.meshgrid(rows, cols, indexing="ij")
    # start/end are (col, row); the grid point is (gc, gr).
    p0c, p0r = start_px[:, 0, None, None], start_px[:, 1, None, None]
    p1c, p1r = end_px[:, 0, None, None], end_px[:, 1, None, None]
    dc, dr = p1c - p0c, p1r - p0r
    L2 = (dc * dc + dr * dr).clamp_min(1e-9)
    t = (((gc[None] - p0c) * dc + (gr[None] - p0r) * dr) / L2).clamp(0.0, 1.0)
    projc, projr = p0c + t * dc, p0r + t * dr
    dist = ((gc[None] - projc) ** 2 + (gr[None] - projr) ** 2).sqrt()
    # Pad forward along the push so the deposit zone ahead of the blade is in.
    ahead = (((gc[None] - p1c) * dc + (gr[None] - p1r) * dr) / L2.sqrt())
    return ((dist <= half_width_px) & (ahead <= pad_px)).float()


def metrics(pred, truth, occ_prev, region=None):
    """Resolution-comparable metrics plus the paper's own raw Frobenius.

    ||.||_F sums over N^2 pixels, so it is NOT comparable across resolutions;
    it is reported only to sit alongside the paper's Table 1 at 32x32. rms and
    the mass-normalised L1 are the numbers to compare across settings.
    """
    n = pred.shape[0]
    if region is None:
        region = torch.ones_like(pred)
    w = region.reshape(n, -1)
    npix = w.sum(dim=1).clamp_min(1.0)
    d = (pred - truth) * region
    flat = d.reshape(n, -1)
    pr, tr_, pv = (x.reshape(n, -1) * w for x in (pred, truth, occ_prev))
    mass = tr_.sum(dim=1).clamp_min(1e-6)
    inter = torch.minimum(pr, tr_).sum(dim=1)
    union = torch.maximum(pr, tr_).sum(dim=1).clamp_min(1e-6)
    return {
        "frobenius": float(flat.norm(dim=1).mean()),
        "rms": float((flat.pow(2).sum(dim=1) / npix).sqrt().mean()),
        "l1_per_mass": float(flat.abs().sum(dim=1).div(mass).mean()),
        "soft_iou": float((inter / union).mean()),
        # Fraction of the change the model actually explains: 1 means perfect,
        # 0 means no better than predicting no change at all.
        "explained": float(1.0 - flat.norm(dim=1).mean()
                           / (tr_ - pv).norm(dim=1).mean().clamp_min(1e-9)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_cfg")
    ap.add_argument("--res", type=int, default=32,
                    help="canonical-frame fit resolution (32 = the paper's)")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="Gaussian sigma (px) applied to the occupancy of BOTH "
                         "frames before anything else, so every model incl. "
                         "persistence sees the same field. The raw occupancy of "
                         "5 mm cubes on a 64x64 grid has features at the pixel "
                         "scale (|laplacian|/|occ| ~ 3), which is the worst case "
                         "for the bilinear SE(2) rotation the method requires: "
                         "measured, the warp round trip alone then costs more "
                         "than one push changes. At sigma >= 1 that cost goes to "
                         "zero. The paper's thresholded diced-carrot images at "
                         "32x32 were effectively already in this regime.")
    ap.add_argument("--native-res", type=int, default=None,
                    help="downsample the DATA (and every baseline, including "
                         "persistence) to this grid before anything else. With "
                         "--native-res == --res the warp is rotation-only and "
                         "no model pays a resampling penalty another avoids — "
                         "the true parity test against the paper, whose native "
                         "grid was 32x32.")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--ridge-sweep", default="1e-2",
                    help="comma-separated ridge lambdas to fit and compare")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--estimators", default="persistence,identity,ridge,nonneg",
                    help="comma-separated: persistence,ols,ridge,nonneg")
    ap.add_argument("--heuristics", default="spread,cumulative")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"=== linear visual foresight: pixel operator @ {args.res}x{args.res} ===")
    t0 = time.time()
    data = load_transition_arrays(args.dataset_cfg, split="train",
                                  max_samples=args.max_samples)
    data.cfg_path, data.split = args.dataset_cfg, "train"
    H, W = data.occ_t.shape[-2:]
    print(f"loaded {data.occ_t.shape[0]} transitions, grid {H}x{W}, "
          f"{time.time() - t0:.1f}s")

    if args.blur > 0:
        sig = args.blur
        k = int(2 * round(3 * sig) + 1)
        ax = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-ax ** 2 / (2 * sig * sig))
        g = (g / g.sum())
        def _blur(x):
            x = torch.nn.functional.conv2d(x.unsqueeze(1), g.view(1, 1, 1, -1),
                                           padding=(0, k // 2))
            return torch.nn.functional.conv2d(x, g.view(1, 1, -1, 1),
                                              padding=(k // 2, 0)).squeeze(1)
        data.occ_t, data.occ_t1 = _blur(data.occ_t), _blur(data.occ_t1)
        print(f"  blurred both frames with sigma={sig} px (all models, "
              f"persistence included, see the same field)")

    if args.native_res and args.native_res != H:
        nr = args.native_res
        pool = torch.nn.functional.adaptive_avg_pool2d
        data.occ_t = pool(data.occ_t.unsqueeze(1), (nr, nr)).squeeze(1)
        data.occ_t1 = pool(data.occ_t1.unsqueeze(1), (nr, nr)).squeeze(1)
        H = W = nr
        print(f"  downsampled the DATA to {nr}x{nr}: every model, persistence "
              f"included, now lives on this grid")

    start_px, end_px = actions_to_pixels(
        data.actions, data.workspace_min, data.workspace_max, (H, W))
    lengths = (end_px - start_px).norm(dim=-1)
    print(f"push length in pixels: mean {float(lengths.mean()):.2f} "
          f"std {float(lengths.std()):.3f}  (a single-operator dataset wants "
          f"std ~ 0)")

    # Verify against the raw dataset's own grid, which the downsample above
    # does not change; scale our pixels to it.
    raw_H = raw_W = 64 if args.native_res else H
    scale = raw_W / W
    d_mid, d_start = verify_pixel_mapping(
        data, start_px * scale, end_px * scale, (raw_H, raw_W))
    # The action channel's centroid sits near the push MIDPOINT. A few pixels
    # of residual is expected (it rasterises the plate body, not the push
    # segment); the check is that this is small next to the ~10 px separating
    # the competing orientation hypotheses.
    print(f"pixel-mapping check vs the dataset's own plate channel: "
          f"median {d_mid:.2f} px from midpoint, {d_start:.2f} px from start "
          f"(a transposed or flipped mapping scores >= 9.5 px here)")
    if not (d_mid < 6.0):
        print("  !! mapping looks wrong — aborting rather than fitting a "
              "transposed operator")
        return

    m_tr, m_te = split_by_episode(data, holdout_frac=args.holdout_frac)
    n_runs = int(data.episode_ids.unique().numel())
    print(f"split: {int(m_tr.sum())} train / {int(m_te.sum())} test "
          f"(whole runs held out; {n_runs} runs total)")

    dev = args.device
    occ_tr, occ1_tr = data.occ_t[m_tr].to(dev), data.occ_t1[m_tr].to(dev)
    occ_te, occ1_te = data.occ_t[m_te].to(dev), data.occ_t1[m_te].to(dev)
    s_tr, e_tr = start_px[m_tr].to(dev), end_px[m_tr].to(dev)
    s_te, e_te = start_px[m_te].to(dev), end_px[m_te].to(dev)

    # ---- mass diagnostic: the paper's convexity argument needs ||I||_1 const
    m0, m1 = occ_tr.sum(dim=(1, 2)), occ1_tr.sum(dim=(1, 2))
    ratio = (m1 / m0.clamp_min(1e-6))
    print(f"\nmass ||I_k||_1 -> ||I_k+1||_1 : ratio mean {float(ratio.mean()):.4f} "
          f"median {float(ratio.median()):.4f} p05 {float(ratio.quantile(0.05)):.4f} "
          f"p95 {float(ratio.quantile(0.95)):.4f}")
    print("  (their V is mass-normalised and their convexity argument assumes "
          "this is ~1)")

    R = args.res
    print(f"\ncanonicalising ({R}x{R}) ...")
    Y0 = canonicalise(occ_tr, s_tr, e_tr, R).reshape(occ_tr.shape[0], -1).T
    Y1 = canonicalise(occ1_tr, s_tr, e_tr, R).reshape(occ_tr.shape[0], -1).T
    D, M = Y0.shape
    print(f"  operator is {D}x{D} = {D * D:,} params from M={M} pairs "
          f"({'OVER' if M > D else 'UNDER'}determined per row: M={M} vs D={D})")

    # Swept band: plate half-length in px, plus a forward pad for the deposit
    # zone (their transport model deposits just ahead of the rectangle).
    plate_px = 0.04 / 0.128 * W          # plate is 40 mm on a 128 mm box
    region = swept_region_mask(s_te, e_te, (H, W),
                               half_width_px=0.5 * plate_px + 2.0,
                               pad_px=0.5 * plate_px)
    print(f"swept-region mask covers {100 * float(region.mean()):.1f}% of the grid")

    results, results_sw = {}, {}
    wanted = [s.strip() for s in args.estimators.split(",") if s.strip()]

    def record(name, pred):
        results[name] = metrics(pred, occ1_te, occ_te)
        results_sw[name] = metrics(pred, occ1_te, occ_te, region=region)

    if "persistence" in wanted:
        record("persistence", occ_te)

    if "identity" in wanted:
        # A = I: warp down to `res`, do nothing, warp back, blend. Isolates
        # what the PIPELINE costs from what the OPERATOR costs -- without it,
        # a fitted operator is being charged for resampling loss that
        # persistence never pays, and the comparison is not a fair one.
        eye = torch.eye(R * R, device=dev)
        record("identity(pipeline)", predict_world(eye, occ_te, s_te, e_te, R, (H, W)))

    fits = [("ols", lambda: fit_operator(Y0, Y1, 0.0)),
            ("nonneg", lambda: fit_operator_nonneg(Y0, Y1))]
    for lam in [float(x) for x in args.ridge_sweep.split(",") if x.strip()]:
        fits.append((f"ridge{lam:g}", lambda lam=lam: fit_operator(Y0, Y1, lam)))

    for name, fit in fits:
        if name not in wanted and not (name.startswith("ridge")
                                       and "ridge" in wanted):
            continue
        t = time.time()
        A = fit()
        t_fit = time.time() - t
        record(f"linear-{name}", predict_world(A, occ_te, s_te, e_te, R, (H, W)))
        print(f"  fitted {name} in {t_fit:.1f}s")

    for h in [x.strip() for x in args.heuristics.split(",") if x.strip()]:
        try:
            pred = predict_heuristic(h, occ_te.cpu(), s_te.cpu(), e_te.cpu())
            record(f"heuristic-{h}", pred.to(dev))
        except Exception as exc:                      # noqa: BLE001
            print(f"  heuristic {h} failed: {exc}")

    for title, res in [("WHOLE IMAGE", results),
                       ("SWEPT REGION ONLY", results_sw)]:
        print(f"\n=== held-out one-step error, {title} "
              f"({occ_te.shape[0]} transitions) ===")
        hdr = (f"{'model':22s} {'rms':>9s} {'frob':>8s} {'L1/mass':>9s} "
               f"{'softIoU':>8s} {'explained':>10s}")
        print(hdr)
        print("-" * len(hdr))
        for k, v in sorted(res.items(), key=lambda kv: kv[1]["rms"]):
            print(f"{k:22s} {v['rms']:9.5f} {v['frobenius']:8.3f} "
                  f"{v['l1_per_mass']:9.4f} {v['soft_iou']:8.4f} "
                  f"{v['explained']:10.4f}")
    print("\nexplained = 1 - ||pred-truth|| / ||I_k+1 - I_k||; <= 0 means the "
          "model is no better than predicting no change.")


if __name__ == "__main__":
    main()
