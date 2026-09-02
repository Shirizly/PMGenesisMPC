"""
DMDc falsification test for the latent-linear (Koopman) pile-dynamics hypothesis.

The question this answers, with zero training loops: are per-action LINEAR maps
over cheap ANALYTIC descriptors of the occupancy grid already predictive of pile
evolution? (Suh & Tedrake 2020, arXiv:2002.09093, switched-linear visual
foresight — here at descriptor level instead of raw 32x32 pixels.)

Pipeline:
    occupancy -> phi(occupancy)  analytic descriptors (mass, COM, moments, DFT)
    push action [sx,sy,ex,ey] -> discrete bin u
    fit A_u : phi_t -> phi_{t+1} per bin by ridge least squares (closed form)
    evaluate one-step and multi-step prediction vs the persistence baseline

Interpretation:
    - Clearly beats persistence on held-out rollouts  -> hypothesis has legs;
      every learned encoder/operator must beat THIS number.
    - Fails -> the per-group error report says WHAT is not linearly predictable
      (e.g. moments fine, high-freq DFT dead), i.e. what a learned encoder must add.

Fully implemented: descriptors, binning, ridge fit, one-step eval, rollout eval,
greedy planner sanity check.
Stubbed (TODO): data loading from the existing registry (see load_transition_arrays),
episode grouping, action canonicalization into the tool frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

# =====================================================================
# 1. Analytic descriptors  phi: occupancy [B,H,W] -> [B,D]
# =====================================================================
#
# Layout (slices exposed for per-group error reporting):
#   [ const 1 | mass | com (2) | central 2nd moments (3) | DFT real | DFT imag ]
#
# The constant dim makes every fitted linear operator effectively AFFINE
# (A_u can move the "origin"), which is required for structure creation from
# a spread state — see EDMD convention of including the constant observable.
#
# Grid convention: dim0 = world_y, dim1 = world_x (matches draw_plate_soft in
# transforms/functional.py). Coordinates normalized to [0,1].


def descriptor_slices(n_fourier: int) -> dict[str, slice]:
    nf = n_fourier * (n_fourier // 2 + 1)  # rfft2 low block, flattened
    s: dict[str, slice] = {}
    i = 0

    def take(name: str, n: int) -> None:
        nonlocal i
        s[name] = slice(i, i + n)
        i += n

    take("const", 1)
    take("mass", 1)
    take("com", 2)
    take("moments2", 3)
    take("dft_real", nf)
    take("dft_imag", nf)
    s["_total"] = slice(0, i)
    return s


def occupancy_descriptors(occ: torch.Tensor, n_fourier: int = 8) -> torch.Tensor:
    """occ: [B,H,W] in [0,1]  ->  phi: [B,D] per descriptor_slices layout."""
    assert occ.dim() == 3
    B, H, W = occ.shape
    eps = 1e-8

    mass = occ.sum(dim=(-2, -1))
    m = mass.clamp_min(eps)

    ys = torch.linspace(0.0, 1.0, H, device=occ.device)
    xs = torch.linspace(0.0, 1.0, W, device=occ.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    com_y = (occ * gy).sum(dim=(-2, -1)) / m
    com_x = (occ * gx).sum(dim=(-2, -1)) / m

    dy = gy.unsqueeze(0) - com_y[:, None, None]
    dx = gx.unsqueeze(0) - com_x[:, None, None]
    mu_yy = (occ * dy * dy).sum(dim=(-2, -1)) / m
    mu_xx = (occ * dx * dx).sum(dim=(-2, -1)) / m
    mu_xy = (occ * dy * dx).sum(dim=(-2, -1)) / m

    # Low-frequency 2D DFT block. norm="forward" keeps coefficients O(mean
    # occupancy) so no per-group rescaling is needed before the ridge fit.
    # F[:,0,0].real duplicates mass/(H*W); harmless under ridge, kept for
    # layout simplicity.
    F = torch.fft.rfft2(occ, norm="forward")
    block = F[:, :n_fourier, : n_fourier // 2 + 1]

    return torch.cat(
        [
            torch.ones(B, 1, device=occ.device),
            (mass / (H * W)).unsqueeze(1),
            torch.stack([com_y, com_x], dim=1),
            torch.stack([mu_yy, mu_xx, mu_xy], dim=1),
            block.real.flatten(1),
            block.imag.flatten(1),
        ],
        dim=1,
    )


# =====================================================================
# 2. Action binning
# =====================================================================
#
# Actions are world-frame push segments [sx, sy, ex, ey] (the convention
# consumed by transforms.functional.genesis_action_to_cam3d).
#
# Baseline scheme: coarse start-cell grid x push-angle bins. This is the
# no-canonicalization variant (like Suh & Tedrake's action discretization);
# it needs enough data per bin but sidesteps the fence problem entirely.
#
# TODO(canonicalization): the data-efficient variant instead rotates/translates
# the occupancy into the tool frame, collapsing bins to ~(push length) only.
# Requires the regime split for the fence: canonicalize interior pushes fully;
# near-wall pushes keep wall-relative bins (see discussion in review).


@dataclass
class ActionBinner:
    workspace_min: tuple[float, float]  # (x_min, y_min) world units
    workspace_max: tuple[float, float]
    n_start_bins: int = 4  # per axis -> n_start_bins^2 cells
    n_angle_bins: int = 8

    @property
    def n_bins(self) -> int:
        return self.n_start_bins**2 * self.n_angle_bins

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [B,4] = sx,sy,ex,ey  ->  bin indices [B] long."""
        sx, sy, ex, ey = actions.unbind(dim=1)
        x0, y0 = self.workspace_min
        x1, y1 = self.workspace_max

        fx = ((sx - x0) / (x1 - x0)).clamp(0, 1 - 1e-6)
        fy = ((sy - y0) / (y1 - y0)).clamp(0, 1 - 1e-6)
        cell = (fy * self.n_start_bins).long() * self.n_start_bins + (
            fx * self.n_start_bins
        ).long()

        angle = torch.atan2(ey - sy, ex - sx)  # (-pi, pi]
        frac = (angle + torch.pi) / (2 * torch.pi)
        abin = (frac * self.n_angle_bins).long().clamp(max=self.n_angle_bins - 1)

        return cell * self.n_angle_bins + abin


# =====================================================================
# 3. Per-bin ridge fit (DMDc, closed form)
# =====================================================================


def fit_per_action_operators(
    phi_t: torch.Tensor,  # [N,D]
    phi_t1: torch.Tensor,  # [N,D]
    bins: torch.Tensor,  # [N] long
    n_bins: int,
    lam: float = 1e-4,
    prior_A: torch.Tensor | None = None,  # [n_bins,D,D] shrink target
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each bin b, ridge-regularised least squares toward a prior operator:

        A_b = argmin ||A phi_t - phi_t1||^2 + lam ||A - A0_b||^2
            = (Y X^T + lam A0_b)(X X^T + lam I)^{-1}

    ``prior_A`` supplies the shrink target A0 (the transfer-learning prior — e.g.
    operators fitted on a source domain). When ``prior_A is None`` the target is
    0 (plain ridge, the original behaviour). Empty target bins fall back to the
    prior operator when one is given, else to identity.

    Returns A: [n_bins, D, D], counts: [n_bins].
    """
    N, D = phi_t.shape
    if prior_A is not None:
        A = prior_A.clone()  # empty/unseen bins keep the source operator
    else:
        A = torch.eye(D).expand(n_bins, D, D).clone()
    counts = torch.zeros(n_bins, dtype=torch.long)
    I = torch.eye(D, dtype=torch.float64)

    for b in range(n_bins):
        mask = bins == b
        nb = int(mask.sum())
        counts[b] = nb
        if nb == 0:
            continue
        X = phi_t[mask].T.double()  # [D, nb]
        Y = phi_t1[mask].T.double()  # [D, nb]
        # A^T solves (X X^T + lam I) A^T = X Y^T + lam A0^T
        rhs = X @ Y.T
        if prior_A is not None:
            rhs = rhs + lam * prior_A[b].T.double()
        At = torch.linalg.solve(X @ X.T + lam * I, rhs)
        A[b] = At.T.float()

    under = int((counts.gt(0) & counts.lt(D)).sum())
    if under:
        print(
            f"[fit] {under}/{n_bins} bins have fewer samples than D={D} "
            "(ridge regularizes, but treat those operators as low-confidence; "
            "consider coarser bins or canonicalization)"
        )
    return A, counts


def apply_operators(A: torch.Tensor, phi: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """phi: [N,D], bins: [N] -> A_{bin} phi : [N,D]."""
    return torch.bmm(A[bins], phi.unsqueeze(-1)).squeeze(-1)


# =====================================================================
# 4. Evaluation
# =====================================================================


def one_step_report(
    A: torch.Tensor,
    phi_t: torch.Tensor,
    phi_t1: torch.Tensor,
    bins: torch.Tensor,
    slices: dict[str, slice],
) -> dict[str, tuple[float, float]]:
    """
    Per descriptor group: (model MSE, persistence MSE). Persistence — predicting
    phi_{t+1} = phi_t — is the bar to clear; a ratio < 1 means signal.
    """
    pred = apply_operators(A, phi_t, bins)
    report: dict[str, tuple[float, float]] = {}
    for name, sl in slices.items():
        if name.startswith("_") or name == "const":
            continue
        mse = (pred[:, sl] - phi_t1[:, sl]).pow(2).mean().item()
        mse_persist = (phi_t[:, sl] - phi_t1[:, sl]).pow(2).mean().item()
        report[name] = (mse, mse_persist)
    return report


def rollout(A: torch.Tensor, phi0: torch.Tensor, bin_seq: torch.Tensor) -> torch.Tensor:
    """phi0: [D], bin_seq: [T] -> predicted descriptors [T,D] (open loop)."""
    out, phi = [], phi0
    for b in bin_seq:
        phi = A[b] @ phi
        out.append(phi)
    return torch.stack(out)


def multistep_report(
    A: torch.Tensor,
    episodes: list[dict[str, torch.Tensor]],
    slices: dict[str, slice],
    horizon: int,
) -> dict[str, list[tuple[float, float]]]:
    """
    episodes: list of {"phi": [T+1,D], "bins": [T]} per contiguous rollout.
    Returns per-group [(model MSE, persistence MSE)] indexed by rollout step.
    """
    groups = [n for n in slices if not n.startswith("_") and n != "const"]
    errs = {n: [[] for _ in range(horizon)] for n in groups}
    errs_p = {n: [[] for _ in range(horizon)] for n in groups}

    for ep in episodes:
        phi, bins = ep["phi"], ep["bins"]
        T = bins.shape[0]
        for t0 in range(T - horizon + 1):
            pred = rollout(A, phi[t0], bins[t0 : t0 + horizon])
            for k in range(horizon):
                truth = phi[t0 + k + 1]
                for n in groups:
                    sl = slices[n]
                    errs[n][k].append((pred[k, sl] - truth[sl]).pow(2).mean().item())
                    errs_p[n][k].append((phi[t0, sl] - truth[sl]).pow(2).mean().item())

    return {
        n: [
            (float(torch.tensor(errs[n][k]).mean()), float(torch.tensor(errs_p[n][k]).mean()))
            for k in range(horizon)
            if errs[n][k]
        ]
        for n in groups
    }


# =====================================================================
# 5. Data plumbing  (TODO — the only part needing codebase integration)
# =====================================================================


@dataclass
class TransitionArrays:
    occ_t: torch.Tensor  # [N,H,W] occupancy before push
    occ_t1: torch.Tensor  # [N,H,W] occupancy after push
    actions: torch.Tensor  # [N,4] raw world push [sx,sy,ex,ey]
    episode_ids: torch.Tensor  # [N] long; contiguous t within an episode share an id
    workspace_min: tuple[float, float] = (0.0, 0.0)
    workspace_max: tuple[float, float] = (1.0, 1.0)


def load_transition_arrays(
    dataset_cfg_path: str,
    split: str = "train",
    max_samples: int | None = None,
) -> TransitionArrays:
    """
    Bridge to the existing dataset registry.

    Builds a ``genesis`` dataset (Genesis/training/dataset.py PileSweepData,
    wrapped by registry.dataset_registry) for the requested ``split`` and
    extracts, per transition:

    - occ_t   : ``input[0]`` — current-occupancy channel (channel 1 is the
                rasterised plate/action and is deliberately excluded).
    - occ_t1  : ``target``   — next occupancy.
    - actions : the RAW world-frame push ``[sx,sy,ex,ey]`` (metres), recovered
                via ``PileSweepData.get_raw_action`` (it is not present in the
                rasterised batch).
    - episode_ids : the run index (``PileSweepData.get_run_index``). One run =
                one data file; samples in a run share nominal physics and
                particle geometry. NB: for these datasets the samples within a
                run are *independent* single-push transitions, not a contiguous
                rollout — see the contiguity diagnosis in ``main()``.

    Workspace bounds come from the box volume (``PileSweepData.workspace_bounds``),
    in the same world-metre frame as the raw actions.

    Macro-step convention: one transition = one full push (matches the MPC
    action space), NOT sim micro-steps.
    """
    import yaml as _yaml

    from registry.dataset_registry import build_dataset

    cfg = _yaml.safe_load(open(dataset_cfg_path).read())
    wrapper = build_dataset(cfg, split)
    raw = wrapper.raw_dataset  # PileSweepData

    n = len(raw)
    if max_samples is not None:
        n = min(n, int(max_samples))

    occ_t = torch.empty((n, *raw._output_grid.shape), dtype=torch.float32)
    occ_t1 = torch.empty_like(occ_t)
    actions = torch.empty((n, 4), dtype=torch.float32)
    episode_ids = torch.empty((n,), dtype=torch.long)

    for i in range(n):
        (input_grid, _physics), target = raw[i]
        occ_t[i] = input_grid[0]
        occ_t1[i] = target
        actions[i] = raw.get_raw_action(i)
        episode_ids[i] = raw.get_run_index(i)

    ws_min, ws_max = raw.workspace_bounds
    return TransitionArrays(
        occ_t=occ_t,
        occ_t1=occ_t1,
        actions=actions,
        episode_ids=episode_ids,
        workspace_min=ws_min,
        workspace_max=ws_max,
    )


def split_by_episode(
    data: TransitionArrays, holdout_frac: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean train/test masks over transitions, split at EPISODE granularity
    (transition-level splits leak: adjacent frames are near-duplicates).

    NB: ``main()`` uses the dataset registry's own physics-group split instead
    of this helper; it is kept for callers that load a single pooled
    ``TransitionArrays`` and want to sub-split it."""
    ids = data.episode_ids.unique()
    n_hold = max(1, int(len(ids) * holdout_frac))
    hold = set(ids[torch.randperm(len(ids))[:n_hold]].tolist())
    test = torch.tensor([int(e) in hold for e in data.episode_ids])
    return ~test, test


def diagnose_contiguity(
    phi_t: torch.Tensor, phi_t1: torch.Tensor, episode_ids: torch.Tensor
) -> float:
    """
    Are the per-run samples a contiguous rollout (phi_t1[i] == phi_t[i+1]),
    or independent single-push transitions?

    Returns the ratio  mean||phi_t1[i]-phi_t[i+1]||^2 / mean||phi_t1[i]-phi_t[i]||^2
    over consecutive same-run pairs. ~0 => contiguous trajectories; ~1 => the
    "successor" is no closer than the pre-state, i.e. independent transitions.
    Returns inf if there are no consecutive same-run pairs to compare.
    """
    succ, change = [], []
    for i in range(len(episode_ids) - 1):
        if episode_ids[i] != episode_ids[i + 1]:
            continue
        succ.append((phi_t1[i] - phi_t[i + 1]).pow(2).mean())
        change.append((phi_t1[i] - phi_t[i]).pow(2).mean())
    if not succ:
        return float("inf")
    denom = torch.stack(change).mean()
    if denom <= 0:
        return float("inf")
    return float((torch.stack(succ).mean() / denom))


def build_episodes(
    phi_t: torch.Tensor,
    phi_t1: torch.Tensor,
    bins: torch.Tensor,
    episode_ids: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    """
    Group contiguous same-run transitions into rollouts for the multi-step
    report. Only meaningful when the data is actually contiguous (see
    ``diagnose_contiguity``): the state sequence is
    ``[phi_t[first], phi_t1[first], phi_t1[second], ...]``.
    """
    episodes: list[dict[str, torch.Tensor]] = []
    ids = episode_ids.tolist()
    start = 0
    for i in range(len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            if i - start >= 1:
                idx = torch.arange(start, i)
                phi_seq = torch.cat([phi_t[idx[0]].unsqueeze(0), phi_t1[idx]], dim=0)
                episodes.append({"phi": phi_seq, "bins": bins[idx]})
            start = i
    return episodes


# =====================================================================
# 6. Greedy planner sanity check (optional; Suh & Tedrake-style descent)
# =====================================================================


def greedy_action_bin(
    A: torch.Tensor,
    phi_cur: torch.Tensor,
    phi_goal: torch.Tensor,
    counts: torch.Tensor,
    weight: torch.Tensor | None = None,  # [D] per-dim cost weights
    min_count: int = 10,
) -> int:
    """argmin over trained bins of || A_b phi_cur - phi_goal ||_W^2.

    A real closed loop needs bin -> representative action + re-encoding the
    observed next state (TODO: run inside run_experiments.py against the sim
    or the heuristic push models as surrogate ground truth)."""
    w = weight if weight is not None else torch.ones_like(phi_goal)
    pred = A @ phi_cur  # [n_bins, D]
    cost = ((pred - phi_goal) * w).pow(2).sum(dim=1)
    cost[counts < min_count] = float("inf")
    return int(cost.argmin())


# =====================================================================
# main
# =====================================================================


def _encode(data: TransitionArrays, n_fourier: int, binner: ActionBinner):
    phi_t = occupancy_descriptors(data.occ_t, n_fourier)
    phi_t1 = occupancy_descriptors(data.occ_t1, n_fourier)
    bins = binner(data.actions)
    return phi_t, phi_t1, bins


def _ratios(A, phi_te, phi1_te, bins_te, slices) -> dict[str, float]:
    """group -> model/persistence MSE ratio on held-out test."""
    return {
        name: (mse / mp if mp > 0 else float("nan"))
        for name, (mse, mp) in one_step_report(A, phi_te, phi1_te, bins_te, slices).items()
    }


def _run_transfer(args, slices, binner, target_train, target_test) -> None:
    """
    Transfer-learning comparison, source (``--transfer-from``) -> target.

    Reports, on the SAME target held-out test, three regimes:
      in-domain : operators fitted only on target train (ridge -> 0)
      zero-shot : source operators applied directly to target (no target fit)
      transfer  : target fit with a ridge PRIOR toward the source operators
                  (strength --transfer-weight)
    Optionally caps target train via --target-max-train to expose the
    data-scarcity regime where a prior actually helps.

    ``target_train``/``target_test`` are the encodings from the SINGLE split
    made in main(), so train and eval stay disjoint (no leakage).
    """
    phi_t_te, phi_t1_te, bins_te = target_test
    pt_t, pt_t1, pt_b = target_train

    # --- source operators (fit on the transfer-from dataset's train split) ---
    src_train, _src_test, src_mode = _load_train_test(
        args.transfer_from, "auto", args.holdout_frac, args.max_samples
    )
    print(f"\n[transfer] source={args.transfer_from} split={src_mode} "
          f"train={len(src_train.episode_ids)}")
    ps_t, ps_t1, ps_b = _encode(src_train, args.n_fourier, binner)
    A_src, _ = fit_per_action_operators(ps_t, ps_t1, ps_b, binner.n_bins, lam=args.lam)

    # --- target train (optionally subsampled to simulate scarce target data) ---
    n_tgt = pt_t.shape[0]
    if args.target_max_train is not None and args.target_max_train < n_tgt:
        keep = torch.randperm(n_tgt)[: args.target_max_train]
        pt_t, pt_t1, pt_b = pt_t[keep], pt_t1[keep], pt_b[keep]
    print(f"[transfer] target train used = {pt_t.shape[0]} "
          f"(of {n_tgt}), weight={args.transfer_weight}")

    A_in, _ = fit_per_action_operators(pt_t, pt_t1, pt_b, binner.n_bins, lam=args.lam)
    A_xf, _ = fit_per_action_operators(
        pt_t, pt_t1, pt_b, binner.n_bins, lam=args.transfer_weight, prior_A=A_src
    )

    r_in = _ratios(A_in, phi_t_te, phi_t1_te, bins_te, slices)
    r_zs = _ratios(A_src, phi_t_te, phi_t1_te, bins_te, slices)
    r_xf = _ratios(A_xf, phi_t_te, phi_t1_te, bins_te, slices)

    print("\n== transfer comparison (ratio = model MSE / persistence MSE, <1 = signal) ==")
    print(f"  {'group':10s}  {'in-domain':>10s}  {'zero-shot':>10s}  {'transfer':>10s}")
    for name in r_in:
        print(f"  {name:10s}  {r_in[name]:10.3f}  {r_zs[name]:10.3f}  {r_xf[name]:10.3f}")
    better = sum(r_xf[n] < r_in[n] - 1e-3 for n in r_in)
    print(f"\ntransfer beats in-domain on {better}/{len(r_in)} groups "
          f"(lower is better).")


def _load_train_test(
    cfg_path: str, split_mode: str, holdout_frac: float, max_samples: int | None
) -> tuple[TransitionArrays, TransitionArrays, str]:
    """
    Return (train_data, test_data, mode_used).

    - "registry": the genesis physics-group split (proper when several physics
      groups exist — keeps equivalent runs off both sides).
    - "by-file":  pool everything and hold out whole runs/files. Correct for
      single-physics datasets (one physics group => registry leaves test empty),
      where files are independent random-pile collections so a file-level
      holdout leaks nothing.
    - "auto": try registry; if it yields no test transitions, fall back to by-file.
    """
    def _registry():
        train = load_transition_arrays(cfg_path, "train", max_samples)
        try:
            test = load_transition_arrays(cfg_path, "test", max_samples)
        except (ValueError, FileNotFoundError):
            test = None
        return train, test

    if split_mode in ("registry", "auto"):
        train, test = _registry()
        if test is not None and len(test.episode_ids) > 0:
            return train, test, "registry"
        if split_mode == "registry":
            raise SystemExit(
                "registry split produced no test set (single physics group?). "
                "Re-run with --split by-file."
            )

    # by-file: pool all data (config should set test_pct/val_pct = 0 so the whole
    # set lands in "train"), then hold out whole files by episode id.
    pooled = load_transition_arrays(cfg_path, "train", max_samples)
    tr_mask, te_mask = split_by_episode(pooled, holdout_frac=holdout_frac)

    def _subset(mask: torch.Tensor) -> TransitionArrays:
        return TransitionArrays(
            occ_t=pooled.occ_t[mask],
            occ_t1=pooled.occ_t1[mask],
            actions=pooled.actions[mask],
            episode_ids=pooled.episode_ids[mask],
            workspace_min=pooled.workspace_min,
            workspace_max=pooled.workspace_max,
        )

    return _subset(tr_mask), _subset(te_mask), "by-file"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_cfg", help="configs/dataset/*.yaml")
    p.add_argument("--n-fourier", type=int, default=8)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n-start-bins", type=int, default=4,
                   help="start-cell bins per axis (n^2 cells); coarser for small data")
    p.add_argument("--n-angle-bins", type=int, default=8, help="push-angle bins")
    p.add_argument("--split", choices=("auto", "registry", "by-file"), default="auto",
                   help="auto (registry, else file-level holdout) | registry | by-file")
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="fraction of files held out for test in by-file mode")
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap transitions per split (debugging)")
    p.add_argument("--transfer-from", default=None,
                   help="source dataset config; fit operators there and use them "
                        "as a ridge prior for this (target) dataset")
    p.add_argument("--transfer-weight", type=float, default=1e-2,
                   help="ridge strength toward the source operators (transfer mode)")
    p.add_argument("--target-max-train", type=int, default=None,
                   help="cap target train transitions (transfer mode) to expose "
                        "the data-scarcity regime")
    args = p.parse_args()

    slices = descriptor_slices(args.n_fourier)
    D = slices["_total"].stop
    print(f"descriptor dim D = {D}  (n_fourier={args.n_fourier})")

    train_data, test_data, mode = _load_train_test(
        args.dataset_cfg, args.split, args.holdout_frac, args.max_samples
    )
    print(f"split={mode}  transitions: train={len(train_data.episode_ids)} "
          f"test={len(test_data.episode_ids)}  grid={tuple(train_data.occ_t.shape[1:])}")
    print(f"workspace (m): {train_data.workspace_min} .. {train_data.workspace_max}")

    binner = ActionBinner(
        train_data.workspace_min, train_data.workspace_max,
        n_start_bins=args.n_start_bins, n_angle_bins=args.n_angle_bins,
    )
    print(f"bins: {binner.n_bins}  (target samples/bin >> D={D} for a reliable fit)")

    phi_t, phi_t1, bins = _encode(train_data, args.n_fourier, binner)
    phi_t_te, phi_t1_te, bins_te = _encode(test_data, args.n_fourier, binner)

    if args.transfer_from:
        _run_transfer(
            args, slices, binner,
            (phi_t, phi_t1, bins), (phi_t_te, phi_t1_te, bins_te),
        )
        ratio = diagnose_contiguity(phi_t_te, phi_t1_te, test_data.episode_ids)
        print(f"\ncontiguity ratio (test) = {ratio:.3f}")
        return

    A, counts = fit_per_action_operators(
        phi_t, phi_t1, bins, binner.n_bins, lam=args.lam
    )
    pop = counts[counts > 0]
    print(f"bins populated: {int(counts.gt(0).sum())}/{binner.n_bins}, "
          f"median samples/bin: {int(pop.median().item()) if len(pop) else 0}")

    print("\n== one-step, held-out (model MSE / persistence MSE, ratio<1 = signal) ==")
    for name, (mse, mse_p) in one_step_report(
        A, phi_t_te, phi_t1_te, bins_te, slices
    ).items():
        ratio = mse / mse_p if mse_p > 0 else float("nan")
        flag = "  <-- beats persistence" if ratio < 1 else ""
        print(f"  {name:10s}  {mse:.3e} / {mse_p:.3e}   ratio {ratio:.3f}{flag}")

    # Multi-step only makes sense on contiguous rollouts. These datasets store
    # independent single-push transitions per run, so diagnose before running.
    ratio = diagnose_contiguity(phi_t_te, phi_t1_te, test_data.episode_ids)
    print(f"\ncontiguity ratio (test) = {ratio:.3f}  "
          f"(~0 => trajectories, ~1 => independent transitions)")
    if ratio < 0.25:
        episodes = build_episodes(phi_t_te, phi_t1_te, bins_te, test_data.episode_ids)
        print(f"\n== multi-step, held-out ({len(episodes)} episodes) ==")
        for name, steps in multistep_report(A, episodes, slices, args.horizon).items():
            line = "  ".join(
                f"k{k+1}:{m/mp if mp > 0 else float('nan'):.2f}"
                for k, (m, mp) in enumerate(steps)
            )
            print(f"  {name:10s}  {line}")
    else:
        print("multi-step report: SKIPPED — dataset has no contiguous rollouts "
              "(independent single-push transitions); one-step is the falsification test.")


if __name__ == "__main__":
    main()
