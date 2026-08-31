"""Is the operator good enough to CONTROL with, even if its images are poor?

Motivation. Suh & Tedrake's headline result is a control result: greedy descent
on an image-space Lyapunov function drives the pile into a target set. Their
prediction evidence is comparative and narrow -- Table 1 gives 1.858 for the
linear model against 2.062 for the deep one, ~10% apart -- and they never report
a persistence baseline at all. So "the operator's per-pixel error is 96% of the
change" does not by itself contradict the paper. A greedy controller never needs
an accurate image; it needs the *ordering* of candidate actions to be right.

This measures exactly that, on the same held-out transitions:

    V(I) = d^T y / ||y||_1        (their eq. 4: distance-transform-weighted mass)
    dV_true = V(I_1) - V(I_0)     what the push actually achieved
    dV_pred = V(I_hat_1) - V(I_0) what the model expected

and reports, for each model:

  * correlation between predicted and actual dV -- can it rank actions?
  * sign agreement -- can it tell a helpful push from a harmful one?
  * regret of picking the model's best action out of a random slate, in units of
    the achievable dV spread, against an oracle that knows dV_true.

Persistence predicts dV_pred = 0 for every action, so it cannot rank at all: it
is the floor here by construction, which makes this a fairer test of the
operator than one-step image error was.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dmdc_baseline import load_transition_arrays, split_by_episode
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator_nonneg, predict_heuristic,
    predict_world,
)
from loro_foresight import gaussian_blur


def lyapunov_weights(grid_res, goal, device):
    """Distance transform `d` for a target set, normalised to [0, 1].

    `goal` is one of:
      center  -- a centred square covering a quarter of the tray area
      corner  -- a square in one corner (their non-convex-ish harder case)
      stripe  -- a central band, so the cost only rewards one axis
    """
    from scipy.ndimage import distance_transform_edt

    H, W = grid_res
    mask = np.zeros((H, W), dtype=bool)
    if goal == "center":
        a, b = H // 4, 3 * H // 4
        mask[a:b, a:b] = True
    elif goal == "corner":
        mask[: H // 2, : W // 2] = True
    elif goal == "stripe":
        mask[H // 2 - H // 8: H // 2 + H // 8, :] = True
    else:
        raise ValueError(goal)
    d = distance_transform_edt(~mask).astype(np.float32)
    d /= max(float(d.max()), 1e-6)
    return torch.from_numpy(d).to(device)


def lyapunov(occ, d, eps=1e-6):
    """V = d^T y / ||y||_1 — mass-normalised mean distance to the target."""
    n = occ.shape[0]
    flat = occ.reshape(n, -1)
    return (flat * d.reshape(1, -1)).sum(1) / flat.sum(1).clamp_min(eps)


def _partial(x, y, z):
    """Correlation of x and y after linearly removing z from both.

    The slate test below draws its candidates from DIFFERENT states, because the
    data holds one action per state. That conflates "this action is good" with
    "this state was easy", and a model that only tracked state difficulty would
    score well while being useless for choosing between actions. Removing the
    state's own V_0 (and its contact score, when available) leaves the part of
    the ranking that is about the action.
    """
    def resid(v):
        Z = torch.stack([torch.ones_like(z[0])] + list(z), dim=-1)
        beta = torch.linalg.lstsq(Z, v.unsqueeze(-1)).solution
        return v - (Z @ beta).squeeze(-1)
    a, b = resid(x), resid(y)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).mean() / (a.std().clamp_min(1e-9) * b.std().clamp_min(1e-9)))


def rank_metrics(dv_pred, dv_true, control=None):
    """Correlations, sign agreement, and a slate-selection regret."""
    p = dv_pred - dv_pred.mean()
    t = dv_true - dv_true.mean()
    pear = float((p * t).mean() / (p.std().clamp_min(1e-9) * t.std().clamp_min(1e-9)))
    rp = dv_pred.argsort().argsort().float()
    rt = dv_true.argsort().argsort().float()
    rp = (rp - rp.mean()) / rp.std().clamp_min(1e-9)
    rt = (rt - rt.mean()) / rt.std().clamp_min(1e-9)
    spear = float((rp * rt).mean())
    # Sign agreement only counts pushes that did something either way.
    live = dv_true.abs() > 1e-4
    sign = float((torch.sign(dv_pred[live]) == torch.sign(dv_true[live])).float().mean())

    # Slate regret: form random slates of K candidates, pick the one the model
    # says is best, and compare the dV actually obtained against the best in the
    # slate. Reported as a fraction of the oracle's advantage over a random
    # pick, so 1.0 = as good as the oracle and 0.0 = no better than random.
    g = torch.Generator(device='cpu').manual_seed(0)
    n = dv_true.shape[0]
    out = {}
    for K in (4, 16):
        idx = torch.randint(0, n, (2000, K), generator=g).to(dv_true.device)
        cand_true = dv_true[idx]
        cand_pred = dv_pred[idx]
        chosen = cand_true.gather(1, cand_pred.argmin(dim=1, keepdim=True)).squeeze(1)
        oracle = cand_true.min(dim=1).values
        rand = cand_true.mean(dim=1)
        denom = (rand - oracle).clamp_min(1e-9)
        out[K] = float(((rand - chosen) / denom).mean())
    part = _partial(dv_pred, dv_true, control) if control else float('nan')
    return pear, spear, sign, out, part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="configs/dataset/genesis_foresight_pile30.yaml")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--crop", type=float, default=0.5)
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--goals", default="center,corner,stripe")
    args = ap.parse_args()

    d = load_transition_arrays(args.dataset, "train")
    H, W = d.occ_t.shape[-2:]
    d.occ_t, d.occ_t1 = (gaussian_blur(d.occ_t, args.blur),
                         gaussian_blur(d.occ_t1, args.blur))
    s, e = actions_to_pixels(d.actions, d.workspace_min, d.workspace_max, (H, W))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    o0, o1, s, e = d.occ_t.to(dev), d.occ_t1.to(dev), s.to(dev), e.to(dev)

    torch.manual_seed(0)
    mtr, mte = split_by_episode(d, 0.25)
    R, CR = args.res, args.crop
    Y0 = canonicalise(o0[mtr], s[mtr], e[mtr], R, CR).reshape(int(mtr.sum()), -1).T
    Y1 = canonicalise(o1[mtr], s[mtr], e[mtr], R, CR).reshape(int(mtr.sum()), -1).T
    A = fit_operator_nonneg(Y0, Y1, max_iter=args.iters, ridge=1.0)

    ote, o1te, ste, ete = o0[mte], o1[mte], s[mte], e[mte]
    preds = {
        "persistence": ote,
        "linear operator": predict_world(A, ote, ste, ete, R, (H, W), CR),
        "heur-cumulative": predict_heuristic(
            "cumulative", ote.cpu(), ste.cpu(), ete.cpu()).to(dev),
    }
    print(f"n_train={int(mtr.sum())}  n_test={int(mte.sum())}  "
          f"operator {R*R}x{R*R}, crop {CR}, blur {args.blur}")

    for goal in [g.strip() for g in args.goals.split(",") if g.strip()]:
        dw = lyapunov_weights((H, W), goal, dev)
        v0, v1 = lyapunov(ote, dw), lyapunov(o1te, dw)
        dv_true = v1 - v0
        # Controls for the state: its own cost, and how much pile the blade
        # meets (both known before acting, so a controller has them too).
        from fit_linear_foresight import contact_score, swept_region_mask
        _plate = 0.04 / 0.128 * W
        control = [v0, contact_score(ote, ste, ete, (H, W), _plate)]
        print(f"\n=== goal '{goal}' ===")
        print(f"  actual dV: mean {float(dv_true.mean()):+.5f}  sd {float(dv_true.std()):.5f}  "
              f"helpful pushes (dV<0): {100 * float((dv_true < 0).float().mean()):.0f}%")
        print(f"  {'model':18s} {'pearson':>8s} {'partial':>8s} {'spearman':>9s} "
              f"{'sign ok':>8s} {'slate4':>8s} {'slate16':>8s}")
        for nm, p in preds.items():
            dv_pred = lyapunov(p, dw) - v0
            if float(dv_pred.abs().max()) < 1e-9:
                print(f"  {nm:18s} {'--':>8s} {'--':>8s} {'--':>9s} {'--':>8s} "
                      f"{0.0:8.3f} {0.0:8.3f}   (predicts dV=0 always: cannot rank)")
                continue
            pe, sp, sg, reg, part = rank_metrics(dv_pred, dv_true, control)
            print(f"  {nm:18s} {pe:8.3f} {part:8.3f} {sp:9.3f} "
                  f"{100 * sg:7.0f}% {reg[4]:8.3f} {reg[16]:8.3f}")
    print("\nslateK = fraction of the oracle's advantage over a random pick that "
          "the model captures\nwhen choosing from K candidates (1.0 = oracle, "
          "0.0 = no better than random).\nThis is what a greedy controller "
          "actually needs; per-pixel accuracy is not.")
    print("partial = correlation after removing the state's own cost and contact "
          "score, i.e.\nthe part of the ranking that is about the ACTION rather "
          "than which state it hit.\nCandidates come from different states "
          "(the data holds one action per state), so\nthe unpartialled columns "
          "overstate what a same-state slate would give.")


if __name__ == "__main__":
    main()
