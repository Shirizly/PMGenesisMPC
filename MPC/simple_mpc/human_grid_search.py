"""
Local grid-search refinement of a human-provided oracle-MPC action.

``simple_mpc.oracle_mpc``'s CEM/MPPI optimizers explore the whole action
space via random sampling because they have no prior on where a good action
is. Here the search space is instead a small neighborhood around an action a
human just drew — a human demonstrator's input is assumed to already be in
roughly the right place, so a plain grid covering that neighborhood is both
cheaper to reason about and sufficient; grid search just nudges the drawn
action toward the locally best point the oracle simulator (the same
Genesis-as-model rollout ``run_oracle_mpc`` uses) predicts.

Actions are the 5D convention ``GenesisOracleEnv`` understands
(``transforms.functional.action_to_pose``):
``[sx, sy, ex, ey, angle_norm]``, with ``angle_norm`` in ``[0, 1)`` mapping to
the plate's pi-periodic yaw, independent of push direction. ``n_ahead`` is
always 1 here — a human demonstrator makes one decision at a time, not a
multi-step plan (unlike the automated optimizers' ``n_look_ahead``).

See ``docs/human_demo_design.md`` for the full subsystem design.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import torch

ACTION_DIM = 5   # [sx, sy, ex, ey, angle_norm]


def build_action_grid(
    center: np.ndarray,
    grid_n: int,
    delta: float | np.ndarray,
    clip_lo: np.ndarray,
    clip_hi: np.ndarray,
) -> np.ndarray:
    """Build the full ``grid_n ** 5`` Cartesian grid of candidates around ``center``.

    ``delta`` is a *normalized* half-width: dimension ``i``'s grid spans
    ``center[i] +/- delta_i * (clip_hi[i] - clip_lo[i])``, then clips to
    ``[clip_lo[i], clip_hi[i]]``. A single scalar broadcasts to all 5
    dimensions — this is what makes one number meaningful across dimensions
    with very different native units (metres for sx/sy/ex/ey, a ``[0, 1)``
    fraction of pi for the angle); pass a length-5 array to vary the
    fraction per dimension instead.

    ``grid_n == 1`` degenerates to the single center point (no exploration —
    useful for evaluating a drawn action without search, without a special
    case at the call site).

    Returns ``(grid_n ** 5, 5)`` float32 array.
    """
    center  = np.asarray(center, dtype=np.float32)
    clip_lo = np.asarray(clip_lo, dtype=np.float32)
    clip_hi = np.asarray(clip_hi, dtype=np.float32)
    if center.shape != (ACTION_DIM,) or clip_lo.shape != (ACTION_DIM,) or clip_hi.shape != (ACTION_DIM,):
        raise ValueError(
            f"center/clip_lo/clip_hi must all be shape ({ACTION_DIM},); "
            f"got {center.shape}, {clip_lo.shape}, {clip_hi.shape}")

    delta_arr = (np.full(ACTION_DIM, float(delta), dtype=np.float32)
                 if np.isscalar(delta) else np.asarray(delta, dtype=np.float32))
    if delta_arr.shape != (ACTION_DIM,):
        raise ValueError(f"delta must be a scalar or length-{ACTION_DIM} array, got shape {delta_arr.shape}")

    half_width = delta_arr * (clip_hi - clip_lo)
    axes = []
    for i in range(ACTION_DIM):
        if grid_n <= 1:
            axes.append(np.array([center[i]], dtype=np.float32))
        else:
            axes.append(np.linspace(center[i] - half_width[i],
                                     center[i] + half_width[i],
                                     grid_n, dtype=np.float32))

    grid = np.array(list(product(*axes)), dtype=np.float32)   # (grid_n**5, 5)
    return np.clip(grid, clip_lo, clip_hi)


def grid_search_refine(
    env,
    snapshot: dict,
    center_action: np.ndarray,
    occ_cur: torch.Tensor,
    goal_mask: torch.Tensor,
    score_tensor: torch.Tensor,
    loss_fn,
    grid_bounds: dict,
    grid_res: tuple,
    footprint_r: float,
    grid_n: int,
    delta: float | np.ndarray,
    clip_lo: np.ndarray,
    clip_hi: np.ndarray,
    device: str,
) -> dict:
    """Evaluate a 5D grid of candidates around ``center_action`` and return the best.

    Two rollout passes, mirroring ``run_oracle_mpc``'s own optimize-then-
    re-roll structure (see its docstring for the full rationale):

      1. Every grid candidate at reduced (``rollout_*``) fidelity, batched in
         chunks of ``env.n_envs`` — this is the search itself. Every
         candidate is recorded (``rollout_candidates``' default
         ``record=True``) as free additional training data, same as
         automated CEM/MPPI candidates.
      2. The single winning candidate re-rolled at full fidelity (the same
         step budgets the real executed step will use, ``record=False``
         since the real step about to follow records it for real) — so the
         reported ``predicted_reward`` isn't confounded by a fidelity gap on
         top of the genuine "will the real step match the plan" question.

    Parameters
    ----------
    env : GenesisOracleEnv
    snapshot : dict from ``env.snapshot_particles()`` — the frozen state
        every candidate rolls out from.
    center_action : (5,) — the human-drawn action to search around.
    occ_cur, goal_mask, score_tensor, loss_fn, grid_bounds, grid_res,
    footprint_r : same objects ``run_oracle_mpc`` builds once per episode —
        pass them straight through unchanged.
    grid_n, delta : see ``build_action_grid``.
    clip_lo, clip_hi : (5,) workspace/orientation bounds.

    Returns
    -------
    dict with ``best_action`` (5,), ``predicted_reward`` (float,
    full-fidelity), ``occ_pred`` ``(*grid_res,)`` numpy (full-fidelity),
    ``n_candidates`` (int, excludes any batch-padding), ``best_cost`` /
    ``mean_cost`` (float, from pass 1 — diagnostics only; a different
    fidelity/metric than ``predicted_reward``, not directly comparable to it).
    """
    # Deferred: simple_mpc.oracle_mpc transitively imports genesis_oracle ->
    # genesis, so importing it lazily here (rather than at module level)
    # keeps build_action_grid importable/testable without genesis installed.
    from simple_mpc.oracle_mpc import _occupancy_reward, _per_sample_cost

    grid = build_action_grid(center_action, grid_n, delta, clip_lo, clip_hi)   # (M, 5)
    n_real = grid.shape[0]

    n_envs = env.n_envs
    n_pad = (-n_real) % n_envs
    if n_pad:
        grid = np.concatenate([grid, np.repeat(grid[-1:], n_pad, axis=0)], axis=0)
    n_batches = grid.shape[0] // n_envs

    best_cost   = math.inf
    best_action = np.asarray(center_action, dtype=np.float32).copy()
    cost_chunks = []
    for b in range(n_batches):
        batch = grid[b * n_envs:(b + 1) * n_envs]                                # (n_envs, 5)
        cand  = torch.as_tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)  # (n_envs, 1, 5)
        pos   = env.rollout_candidates(cand, snapshot)                           # reduced fidelity
        occ   = env.particles_to_occ(pos, grid_bounds, grid_res, footprint_r)
        cost  = _per_sample_cost(occ, occ_cur, goal_mask, score_tensor, loss_fn)  # (n_envs,)

        valid   = min(n_envs, n_real - b * n_envs)
        cost_np = cost[:valid].detach().cpu().numpy()
        cost_chunks.append(cost_np)
        local_best = int(np.argmin(cost_np))
        if cost_np[local_best] < best_cost:
            best_cost   = float(cost_np[local_best])
            best_action = batch[local_best].copy()

    all_costs = np.concatenate(cost_chunks)

    # Full-fidelity re-roll of the winner only — see docstring above.
    winner       = torch.as_tensor(best_action, dtype=torch.float32, device=device)
    winner_batch = winner.view(1, 1, ACTION_DIM).expand(n_envs, 1, ACTION_DIM).contiguous()
    pos_full = env.rollout_candidates(
        winner_batch, snapshot, use_rollout_fidelity=False, record=False)
    occ_full = env.particles_to_occ(pos_full[0:1], grid_bounds, grid_res, footprint_r)
    predicted_reward = _occupancy_reward(occ_full, score_tensor)

    return {
        'best_action':      best_action,
        'predicted_reward': predicted_reward,
        'occ_pred':         occ_full[0].detach().cpu().numpy(),
        'n_candidates':     n_real,
        'best_cost':        float(all_costs.min()),
        'mean_cost':        float(all_costs.mean()),
    }
