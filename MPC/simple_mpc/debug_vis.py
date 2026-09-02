"""Debug visualization helpers for simple_mpc.

Pure numpy/cv2 — no dependency on visualize_*.py or the simulation environment.
"""

import os
import cv2
import numpy as np


# ── primitive drawing helpers ─────────────────────────────────────────────────

def heatmap_bgr(
    arr: np.ndarray,                      # (Nx, Ny) float
    size: tuple,                          # (H, W)
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Render a 2-D grid as a false-colour BGR image (x=right, y=down)."""
    t  = arr.T   # (Ny, Nx)
    lo, hi = t.min(), t.max()
    u8 = ((t - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return cv2.resize(cv2.applyColorMap(u8, colormap),
                      (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def diff_heatmap_bgr(diff: np.ndarray, size: tuple) -> np.ndarray:
    """Occupancy change: green = material arrived, red = material left."""
    H, W = size
    gain = cv2.resize((np.clip( diff, 0, 1).T * 255).astype(np.uint8),
                      (W, H), interpolation=cv2.INTER_NEAREST)
    loss = cv2.resize((np.clip(-diff, 0, 1).T * 255).astype(np.uint8),
                      (W, H), interpolation=cv2.INTER_NEAREST)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 1] = gain   # green channel → gain  (BGR)
    img[:, :, 2] = loss   # red   channel → loss  (BGR)
    return img


def stamp(img: np.ndarray, lines) -> np.ndarray:
    """Stamp text lines at the top-left corner of a copy of img."""
    out = img.copy()
    for k, text in enumerate(lines):
        y = 15 + k * 15
        cv2.putText(out, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0,   0,   0), 1, cv2.LINE_AA)
    return out


# ── composite debug panels ───────────────────────────────────────────────────

def save_debug_candidates(
    path:              str,
    occ_init_np:       np.ndarray,        # (Nx, Ny)
    topk_occ_nps:      list,              # K × (Nx, Ny)
    topk_acts:         np.ndarray,        # (K, 4)  world-2D
    topk_rews:         np.ndarray,        # (K,)
    score_np:          np.ndarray,        # (Nx, Ny)  goal score map
    step: int,
    it:   int,
    tile: int = 160,
    topk_start_grids:  np.ndarray = None, # (K, 2)  grid-index coords
    topk_end_grids:    np.ndarray = None, # (K, 2)
) -> None:
    """
    Save a vertical stack of top-K candidate tiles.
    Each row: [input occ | predicted occ + action arrow | pred*score | score map]

    Columns:
      col 0 – input occupancy (same for all rows)
      col 1 – predicted occupancy after the push, with a cyan arrow showing
               where the push action falls IN GRID SPACE.  If the arrow
               misses the occupied region the optimizer has nothing to push.
      col 2 – pred * goal_score (reward density): WHERE reward comes from.
               Compare with col 3 to verify alignment with the goal.
      col 3 – goal score map (same for all rows, for reference).
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    Nx, Ny    = occ_init_np.shape
    inp_base  = heatmap_bgr(occ_init_np, (tile, tile))
    score_lbl = stamp(heatmap_bgr(score_np, (tile, tile), cv2.COLORMAP_HOT),
                      ['score map (goal)'])
    rows = []
    for rank, (occ_p, act, rew) in enumerate(
            zip(topk_occ_nps, topk_acts, topk_rews)):
        inp_l    = stamp(inp_base.copy(),
                         ['input occ'] if rank == 0 else ['  (same)'])
        pred_img = heatmap_bgr(occ_p, (tile, tile))
        # -- draw push arrow in grid space (cyan = start, tip = end) ----------
        # Grid dim 0 (cam-x) → image column; dim 1 (cam-y) → image row.
        if topk_start_grids is not None and topk_end_grids is not None:
            sx_g = float(topk_start_grids[rank, 0])
            sy_g = float(topk_start_grids[rank, 1])
            ex_g = float(topk_end_grids[rank, 0])
            ey_g = float(topk_end_grids[rank, 1])
            sc   = (tile - 1) / max(Nx - 1, 1)
            sp   = (int(np.clip(sx_g * sc, 0, tile - 1)),
                    int(np.clip(sy_g * sc, 0, tile - 1)))  # (col, row) for cv2
            ep   = (int(np.clip(ex_g * sc, 0, tile - 1)),
                    int(np.clip(ey_g * sc, 0, tile - 1)))
            cv2.arrowedLine(pred_img, sp, ep, (0, 255, 255), 2,
                            cv2.LINE_AA, tipLength=0.3)
            cv2.circle(pred_img, sp, 4, (0, 180, 255), -1, cv2.LINE_AA)
        pred_l = stamp(pred_img,
                       [f'rank{rank+1}  r={rew:.3f}',
                        f's({act[0]:.2f},{act[1]:.2f})',
                        f'e({act[2]:.2f},{act[3]:.2f})'])
        # Use clamped occ for reward density — matches optimizer objective.
        rew_l  = stamp(heatmap_bgr(np.clip(occ_p, 0.0, 1.0) * score_np,
                                   (tile, tile), cv2.COLORMAP_HOT),
                       ['pred*score (clamped)'])
        rows.append(np.hstack([inp_l, pred_l, rew_l, score_lbl]))
    hdr = np.zeros((20, rows[0].shape[1], 3), dtype=np.uint8)
    cv2.putText(hdr, f'Step {step}  iter {it}  top-{len(rows)} candidates',
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.imwrite(path, np.vstack([hdr] + rows))


def save_predicted_trajectory_video(
    path:        str,
    occ_init_np: np.ndarray,       # (Nx, Ny)
    occ_seq:     list,             # list of (Nx, Ny) per look-ahead step
    score_np:    np.ndarray,       # (Nx, Ny)
    step:        int,
    tile:        int = 320,
    fps:         int = 4,
    hold_seconds:  float = 2.0,
    final_seconds: float = 5.0,
) -> None:
    """Save a video showing the predicted occupancy trajectory for one MPC step."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    all_occ = [occ_init_np] + list(occ_seq)
    frames = []
    for t, occ in enumerate(all_occ):
        occ_img = heatmap_bgr(occ, (tile, tile))
        rew_img = heatmap_bgr(np.clip(occ, 0, 1) * score_np,
                              (tile, tile), cv2.COLORMAP_HOT)
        score_img = heatmap_bgr(score_np, (tile, tile), cv2.COLORMAP_HOT)
        r = float((np.clip(occ, 0, 1) * score_np).sum())
        lbl = f'Step {step}  t={t} (init)' if t == 0 else f'Step {step}  t={t}  r={r:.3f}'
        occ_img   = stamp(occ_img,   [lbl])
        rew_img   = stamp(rew_img,   ['pred*score'])
        score_img = stamp(score_img, ['goal score'])
        frames.append(np.hstack([occ_img, rew_img, score_img]))
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (W, H))
    hold  = max(1, int(fps * hold_seconds))
    final = max(1, int(fps * final_seconds))
    for idx, f in enumerate(frames):
        n = final if idx == len(frames) - 1 else hold
        for _ in range(n):
            writer.write(f)
    writer.release()


def save_debug_winner(
    path:        str,
    occ_init_np: np.ndarray,   # (Nx, Ny)
    occ_pred_np: np.ndarray,   # (Nx, Ny)
    score_np:    np.ndarray,   # (Nx, Ny)
    best_act:    np.ndarray,   # (4,)
    predicted_r: float,        # best reward from optimizer
    actual_r:    float,        # real Eulerian reward after env.step
    step:        int,
    tile:        int = 240,
) -> None:
    """
    6-tile winner panel (2 rows x 3 cols):
      Row 0: [input occ | predicted occ | goal score map]
      Row 1: [input*score  | pred*score  | occ change (pred - input)]

    Reading guide:
      - Compare 'input*score' vs 'pred*score': is the reward density
        expected to increase?  If yes but act_r drops, the PUSH MODEL is
        the problem (it predicts material movement that doesn't happen).
      - Compare 'pred*score' shape with 'goal score map': if material is
        predicted to land in dark (low-score) regions, the SCORING is wrong.
      - 'act_gain' printed in the diff tile: negative means the environment
        moved material AWAY from the goal even though the MPC predicted gain.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    T       = tile
    cur_r   = float((occ_init_np * score_np).sum())
    pred_gain = predicted_r - cur_r
    act_gain  = actual_r    - cur_r

    r0 = np.hstack([
        stamp(heatmap_bgr(occ_init_np, (T, T)),
              ['input occ', f'r={cur_r:.3f}']),
        stamp(heatmap_bgr(occ_pred_np, (T, T)),
              ['predicted occ',
               f's({best_act[0]:.2f},{best_act[1]:.2f})',
               f'e({best_act[2]:.2f},{best_act[3]:.2f})']),
        stamp(heatmap_bgr(score_np, (T, T), cv2.COLORMAP_HOT),
              ['goal score map']),
    ])
    r1 = np.hstack([
        stamp(heatmap_bgr(occ_init_np * score_np, (T, T), cv2.COLORMAP_HOT),
              ['cur rew density', f'sum={cur_r:.3f}']),
        # Use clamped occ for reward density — matches optimizer objective.
        stamp(heatmap_bgr(np.clip(occ_pred_np, 0.0, 1.0) * score_np,
                          (T, T), cv2.COLORMAP_HOT),
              ['pred*score (clamped)',
               f'pred_r={predicted_r:.3f}',
               f'pred_gain={pred_gain:+.3f}']),
        stamp(diff_heatmap_bgr(occ_pred_np - occ_init_np, (T, T)),
              ['occ change (pred-input)',
               'green=gain  red=loss',
               f'act_r={actual_r:.3f}',
               f'act_gain={act_gain:+.3f}']),
    ])
    title_h = 24
    title = np.zeros((title_h, r0.shape[1], 3), dtype=np.uint8)
    cv2.putText(title,
                (f'Step {step} | pred_r={predicted_r:.4f}  act_r={actual_r:.4f} | '
                 f'pred_gain={pred_gain:+.3f}  act_gain={act_gain:+.3f}'),
                (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
    cv2.imwrite(path, np.vstack([title, r0, r1]))
