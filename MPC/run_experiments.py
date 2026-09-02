#!/usr/bin/env python3
"""
run_experiments.py — Batch MPC experiment runner for model / hyperparameter comparison.

Usage
-----
    python run_experiments.py simple_mpc/config/experiments/compare_models.yaml
    python run_experiments.py simple_mpc/config/experiments/compare_models.yaml --dry-run
    python run_experiments.py simple_mpc/config/experiments/compare_models.yaml \\
            --only gnn_baseline gnn_more_samples

Output layout
-------------
    outputs/experiments/<experiment_name>_<timestamp>/
        experiment_suite.yaml           copy of the config used
        overall_summary.json            cross-experiment statistics
        comparison.png                  bar charts: reward gain, success rate
        trajectories.png                mean reward curves overlaid
        <experiment.name>/
            summary.json                aggregated statistics over all episodes
            rewards_all.npy             (n_episodes, n_mpc+1) stacked reward arrays
            rewards_summary.png         mean ± std reward curve
            episode_NNN/
                metrics.json            scalar stats for this episode
                rewards.npy             (n_mpc+1,)
                actions.npy             (n_mpc, 5)  — [sx, sy, ex, ey, angle_rad]
                episode_data.npz        rewards, actions, rew_means, pred_rewards
                rewards.png             reward-vs-step plot
"""

import os
import sys
import copy
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _cv2 = None
    _CV2_OK = False

import genesis_path  # noqa: F401  (puts Genesis/ and MPC/ on sys.path)

from utils import (
    load_yaml,
    gen_subgoal,
    gen_goal_shape,
    set_seed,
    get_current_YYYY_MM_DD_hh_mm_ss_ms,
    scale_subgoal_to_material_pixels,
    pcd2pix,
)
from env.genesis_env import GenesisEnv
from model.model_card import load_model_from_card
from simple_mpc import run_simple_mpc
from simple_mpc.benchmark import benchmark_push_throughput



# ── config helpers ─────────────────────────────────────────────────────────────

def _deep_set(cfg: dict, dotkey: str, value) -> None:
    """Set a nested config value using a dot-notation key in-place."""
    keys = dotkey.split('.')
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def apply_overrides(base_cfg: dict, overrides: dict) -> dict:
    """
    Return a deep copy of *base_cfg* with all *overrides* applied.

    Keys in *overrides* use dot-notation (e.g. ``"mpc.n_sample": 1024``),
    which is translated to nested dict access before setting.
    """
    cfg = copy.deepcopy(base_cfg)
    for dotkey, value in overrides.items():
        _deep_set(cfg, dotkey, value)
    return cfg


# ── model loading (with in-process cache) ─────────────────────────────────────

_model_cache: dict = {}   # cache key includes all model-selection knobs → model


def _cache_key(model_spec: dict, card_path: str | None = None) -> tuple:
    """Build a cache key for model-card or heuristic loading."""
    mode = "heuristic" if card_path is None else "model_card"
    return (
        mode,
        str(card_path or ""),
        model_spec.get('type', '').lower(),
        str(model_spec.get('heuristic_type', '')).lower(),
        int(model_spec.get('grid_n', -1)),
        json.dumps(model_spec.get('inference_overrides', {}), sort_keys=True),
    )


def _resolve_model_card_path(model_spec: dict) -> str:
    """Resolve model card path from explicit card path or training config."""
    if "model_card" in model_spec:
        return str(model_spec["model_card"])

    training_cfg_path = model_spec.get("training_config")
    if training_cfg_path is None:
        raise ValueError(
            "Model spec must include either 'model_card' or 'training_config'. "
            "For non-trained baselines use type='eulerian-heuristic'."
        )

    from model_training.trainer import load_config

    train_cfg = load_config(Path(training_cfg_path))
    for dotkey, value in model_spec.get("training_overrides", {}).items():
        _deep_set(train_cfg, dotkey, value)

    log_dir = Path(train_cfg["output"]["log_dir"])
    return str(log_dir / "model_card.yaml")


def _encode_override_value(value) -> str:
    """Encode a Python value into a single CLI-safe scalar for KEY=VALUE."""
    return json.dumps(value)


def _ensure_model_card_ready(model_spec: dict) -> str:
    """Resolve model_card path and optionally train if missing."""
    card_path = _resolve_model_card_path(model_spec)
    if Path(card_path).exists():
        return card_path

    if not bool(model_spec.get("train_if_missing", False)):
        raise FileNotFoundError(
            f"Model card not found: {card_path}. "
            "Provide an existing model_card."
        )
    return card_path

def _build_eulerian_heuristic(model_spec: dict, cfg: dict, env):
    """Create an explicit heuristic Eulerian model (no learned weights)."""
    from model.eulerian_wrapper import EulerianModelWrapper, build_push_model

    if env is None:
        raise ValueError(
            "Eulerian heuristic model requires env for cam_extrinsic "
            "(needed by forward pass to convert world-space actions to grid coords)"
        )

    # heuristic_type: 'splat' | 'spread' | 'spread2' | 'cumulative' | 'fluid'
    # (see model/eulerian_wrapper.py's push-model registry). Per-heuristic
    # hyperparameters (width, sigma, decay, ...) may also be set in model_spec
    # and are forwarded as-is; unset ones fall back to the registry defaults.
    push_model = build_push_model(model_spec)

    bounds = EulerianModelWrapper.default_bounds(cfg, convention="genesis")
    grid_n = int(model_spec.get("grid_n", 64))
    cam_extrinsic = env.get_cam_extrinsics()
    global_scale = cfg["dataset"]["global_scale"]
    model = EulerianModelWrapper(
        push_model,
        bounds,
        (grid_n, grid_n),
        cam_extrinsic,
        global_scale,
        action_convention="genesis",
    )
    return model.cuda() if hasattr(model, "cuda") else model


def load_model(model_spec: dict, cfg: dict, env=None, force_reload: bool = False):
    """
    Load (or retrieve from cache) the dynamics model described by *model_spec*.

    model_spec keys
    ---------------
    type               : 'eulerian-heuristic' for non-trained baselines.
                         Otherwise model-card loading is used.
    model_card         : explicit path to model_card.yaml (preferred).
    training_config    : path to configs/training/*.yaml; resolves to
                         output.log_dir/model_card.yaml.
    training_overrides : dot-key overrides used both for card-path resolution
                         and optional train-if-missing bootstrap.
    train_if_missing   : if true and model_card does not exist, runs
                         `python -m training.train <training_config> ...`.
    inference_overrides: merged into cfg['dataset'] before load_model_from_card.
    heuristic_type     : for eulerian-heuristic, one of the names registered in
                         model.eulerian_wrapper's push-model registry:
                         'splat'|'spread'|'spread2'|'cumulative'|'fluid'.
                         Defaults to 'spread'.
    grid_n             : for eulerian-heuristic, occupancy grid size.
    (heuristic hyperparameters, e.g. width/sigma/decay/n_steps, may also be
    set directly on model_spec; unset ones fall back to registry defaults —
    see model/eulerian_wrapper.py's build_push_model)
    """
    mtype = model_spec.get("type", "").lower()
    is_heuristic = (
        mtype == "eulerian-heuristic"
        or (mtype == "eulerian" and not model_spec.get("need_weights", True) and "model_card" not in model_spec and "training_config" not in model_spec)
    )

    if is_heuristic:
        key = _cache_key(model_spec, card_path=None)
        if not force_reload and key in _model_cache:
            print(f"    [model cache hit]  {key}")
            return _model_cache[key]
        model = _build_eulerian_heuristic(model_spec, cfg, env)
        _model_cache[key] = model
        return model

    card_path = _ensure_model_card_ready(model_spec)
    key = _cache_key(model_spec, card_path=card_path)
    if not force_reload and key in _model_cache:
        print(f"    [model cache hit]  {key}")
        return _model_cache[key]

    print(f"    Loading model from card: {card_path}")
    inference_overrides = dict(cfg.get("dataset", {}))
    inference_overrides.update(model_spec.get("inference_overrides", {}))
    model = load_model_from_card(card_path, env=env, mpc_cfg=inference_overrides)

    _model_cache[key] = model
    return model


# ── goal / subgoal construction ────────────────────────────────────────────────

def build_goal(cfg: dict, env):
    """Return ``(subgoal, goal_img_rgb)`` from the task config."""
    task  = cfg['mpc']['task']
    H, W  = env.screenHeight, env.screenWidth
    ttype = task['type']

    if ttype == 'target_control':
        subgoal, mask = gen_subgoal(
            task['goal_row'], task['goal_col'], task['goal_r'], h=H, w=W)
        goal_img = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

    elif ttype == 'target_shape':
        subgoal, goal_img = gen_goal_shape(task['target_char'], h=H, w=W)

    else:
        raise NotImplementedError(f"Unknown task type: '{ttype}'")

    return subgoal, goal_img


# ── per-episode metrics ────────────────────────────────────────────────────────

def compute_episode_metrics(result: dict, success_threshold: float = 0.0) -> dict:
    """
    Extract scalar and array metrics from a ``run_simple_mpc`` result dict.

    Returns a JSON-serialisable dict.  Arrays are converted to lists.

    Prediction gap
    --------------
    For each step i, the model predicted a reward of ``best_rewards_per_step[i]``
    when choosing the best action.  The environment then measured
    ``rewards[i+1]``.  The *prediction gap* is
        best_predicted_reward − actual_reward_after_step
    A large positive gap means the model is over-optimistic.
    """
    rewards    = result['rewards']        # (n_mpc+1,) — may be shorter on early stop
    n          = len(rewards)             # actual steps completed + 1
    best_preds = result.get('best_rewards_per_step', [])

    step_gains  = (rewards[1:] - rewards[:-1]).tolist()
    pred_gaps   = []
    for bp, r_actual in zip(best_preds, rewards[1:]):
        if bp is not None:
            pred_gaps.append(float(bp) - float(r_actual))

    compute_t = float(result.get('rollout_time', 0) + result.get('optim_time', 0))
    n_steps   = max(n - 1, 1)

    metrics = {
        'n_steps_completed':       n - 1,
        'reward_initial':          float(rewards[0]),
        'reward_final':            float(rewards[n - 1]),
        'reward_gain':             float(rewards[n - 1] - rewards[0]),
        'reward_max':              float(rewards.max()),
        'step_gains':              step_gains,
        'best_predicted_rewards':  [float(b) for b in best_preds if b is not None],
        'prediction_reward_gaps':  pred_gaps,
        'mean_prediction_gap':     float(np.mean(pred_gaps)) if pred_gaps else None,
        'compute_time_s':          compute_t,
        'compute_time_per_step_s': compute_t / n_steps,
        'total_time_s':            float(result.get('total_time', 0)),
        'success':                 bool(rewards[n - 1] - rewards[0] > success_threshold),
    }

    occ_rews = result.get('occ_rewards')
    if occ_rews is not None:
        metrics['occ_reward_initial'] = float(occ_rews[0])
        metrics['occ_reward_final']   = float(occ_rews[n - 1])
        metrics['occ_reward_gain']    = float(occ_rews[n - 1] - occ_rews[0])

    return metrics


# ── plotting helpers ───────────────────────────────────────────────────────────

def _plot_reward_episode(rewards: np.ndarray, out_path: str) -> None:
    """Save a reward-vs-step line plot for a single episode."""
    fig, ax = plt.subplots(figsize=(6, 3))
    steps = np.arange(len(rewards))
    ax.plot(steps, rewards, 'o-', lw=1.5, color='steelblue')
    ax.fill_between(steps, rewards[0], rewards, alpha=0.12, color='steelblue')
    ax.axhline(rewards[0], color='gray', lw=0.8, linestyle='--', label='initial')
    ax.set_xlabel('MPC step')
    ax.set_ylabel('Reward')
    ax.set_title(f'Reward  (gain={rewards[-1]-rewards[0]:+.4f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _plot_reward_experiment(rewards_mat: np.ndarray, name: str,
                            out_path: str) -> None:
    """Save mean ± std reward curve across all episodes of one experiment."""
    mean  = rewards_mat.mean(axis=0)
    std   = rewards_mat.std(axis=0)
    steps = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rewards_mat:
        ax.plot(steps, row, 'k-', alpha=0.15, lw=0.7)
    ax.plot(steps, mean, 'o-', lw=2, color='steelblue',
            label=f'mean  (n={len(rewards_mat)})')
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25,
                    color='steelblue', label='± 1 std')
    ax.set_xlabel('MPC step')
    ax.set_ylabel('Reward')
    ax.set_title(f'{name} — reward over steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── prediction video helpers ──────────────────────────────────────────────────

def _draw_ptcls(img_bgr, pts_3d, cam_params, color_bgr,
                radius: int = 5, alpha: float = 0.75):
    """Project 3-D particles to pixel space and draw circles on img_bgr."""
    if pts_3d is None or len(pts_3d) == 0:
        return img_bgr.copy()
    pix = pcd2pix(pts_3d, cam_params)
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    for r, c in pix:
        if 0 <= r < H and 0 <= c < W:
            _cv2.circle(overlay, (int(c), int(r)), radius, color_bgr, -1, _cv2.LINE_AA)
    return _cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)


def _lbl(img_bgr, text: str):
    out = img_bgr.copy()
    _cv2.putText(out, text, (8, 24), _cv2.FONT_HERSHEY_SIMPLEX,
                 0.6, (255, 255, 255), 2, _cv2.LINE_AA)
    _cv2.putText(out, text, (8, 24), _cv2.FONT_HERSHEY_SIMPLEX,
                 0.6, (0, 0, 0), 1, _cv2.LINE_AA)
    return out


def _save_prediction_video(
    result: dict,
    ep_dir: str,
    cam_params,
    goal_img_rgb: np.ndarray,
    fps: int = 2,
) -> None:
    """
    Write a per-episode prediction-vs-actual comparison video.

    Layout per frame (mirrors visualize_prediction.py make_panel):
      LEFT  – current state,  green  = actual particles
      RIGHT – next-state,     cyan   = model-predicted particles
                               red    = actual particles after action
      (optional third tile)  – goal image
    """
    if not _CV2_OK:
        print('  Warning: cv2 not available — skipping prediction video')
        return

    raw_obs     = result.get('raw_obs', [])
    states      = result.get('states', [])
    states_pred = result.get('states_pred', [])
    rewards     = result.get('rewards')

    if (raw_obs is None or states is None or states_pred is None
            or len(raw_obs) == 0 or len(states) == 0 or len(states_pred) == 0):
        print('  Warning: raw_obs/states/states_pred missing — skipping prediction video')
        return

    n_steps = min(len(states) - 1, len(states_pred))
    if n_steps < 1:
        return

    obs0    = np.array(raw_obs[0])
    H, W    = obs0.shape[:2]
    n_tiles = 3 if goal_img_rgb is not None else 2
    frame_w = W * n_tiles
    frame_h = H + 28   # +28 for legend bar

    vid_path = os.path.join(ep_dir, 'prediction_video.avi')
    fourcc   = _cv2.VideoWriter_fourcc(*'MJPG')
    writer   = _cv2.VideoWriter(vid_path, fourcc, fps, (frame_w, frame_h))

    GREEN = (50,  200,  0)
    CYAN  = (200, 180,  0)
    RED   = (0,    50, 220)

    for step in range(n_steps):
        img_cur_rgb  = np.array(raw_obs[step])[..., :3].astype(np.uint8)
        img_next_rgb = np.array(raw_obs[step + 1])[..., :3].astype(np.uint8)

        img_cur_bgr  = _cv2.cvtColor(img_cur_rgb,  _cv2.COLOR_RGB2BGR)
        img_next_bgr = _cv2.cvtColor(img_next_rgb, _cv2.COLOR_RGB2BGR)

        tile_cur  = _draw_ptcls(img_cur_bgr,  states[step],      cam_params, GREEN)
        tile_next = _draw_ptcls(img_next_bgr, states_pred[step], cam_params, CYAN)
        tile_next = _draw_ptcls(tile_next,    states[step + 1],  cam_params, RED, radius=3)

        r_cur  = float(rewards[step])     if rewards is not None else 0.0
        r_next = float(rewards[step + 1]) if rewards is not None else 0.0

        tile_cur  = _lbl(tile_cur,  f"Step {step}  r={r_cur:.3f}")
        tile_next = _lbl(tile_next, f"After action  r={r_next:.3f}")

        tiles = [tile_cur, tile_next]
        if goal_img_rgb is not None:
            goal_bgr = _cv2.cvtColor(goal_img_rgb.astype(np.uint8), _cv2.COLOR_RGB2BGR)
            goal_bgr = _cv2.resize(goal_bgr, (W, H))
            goal_bgr = _lbl(goal_bgr, 'Goal')
            tiles.append(goal_bgr)

        main_row = np.hstack(tiles)

        legend = np.zeros((28, frame_w, 3), dtype=np.uint8)
        for x, col, txt in [(8,   GREEN, "current particles"),
                             (220, CYAN,  "predicted (model)"),
                             (430, RED,   "actual (sim)")]:
            _cv2.circle(legend, (x + 6, 14), 6, col, -1)
            _cv2.putText(legend, txt, (x + 16, 19),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)

        writer.write(np.vstack([main_row, legend]))

    writer.release()
    print(f'    Saved prediction video → {vid_path}')


# ── Eulerian occupancy overlay helpers (mirrors visualize_prediction_eulerian) ─

def _draw_occ_overlay(img_bgr, occ_np, grid_bounds, grid_res, cam_params,
                      color_bgr, radius=4, alpha=0.75, threshold=0.3):
    """Project occupied voxels of occ_np to pixel space and draw circles."""
    Nx, Ny = grid_res
    x_min, x_max = grid_bounds['x_min'], grid_bounds['x_max']
    y_min, y_max = grid_bounds['y_min'], grid_bounds['y_max']
    z_mid = 0.5 * (grid_bounds.get('z_min', 0.70) + grid_bounds.get('z_max', 0.80))
    ix = np.arange(Nx, dtype=np.float32)
    iy = np.arange(Ny, dtype=np.float32)
    X  = x_min + ix * (x_max - x_min) / max(Nx - 1, 1)
    Y  = y_min + iy * (y_max - y_min) / max(Ny - 1, 1)
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    mask   = occ_np >= threshold
    pts = np.zeros((mask.sum(), 3), dtype=np.float32)
    pts[:, 0] = XX[mask]
    pts[:, 1] = YY[mask]
    pts[:, 2] = z_mid
    vals = occ_np[mask]
    if pts.shape[0] == 0:
        return img_bgr.copy()
    pix = pcd2pix(pts, cam_params)
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    for (r, c), v in zip(pix, vals):
        if 0 <= r < H and 0 <= c < W:
            _cv2.circle(overlay, (int(c), int(r)), max(1, int(radius * v)),
                        color_bgr, -1, _cv2.LINE_AA)
    return _cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)


def _occ_to_heatmap_bgr(occ_np, size):
    """(Nx, Ny) occupancy → false-colour BGR image of given (H, W) size."""
    grid_u8 = (np.clip(occ_np.T, 0.0, 1.0) * 255).astype(np.uint8)
    colored = _cv2.applyColorMap(grid_u8, _cv2.COLORMAP_VIRIDIS)
    return _cv2.resize(colored, (size[1], size[0]), interpolation=_cv2.INTER_NEAREST)


def _occ_to_dual_heatmap_bgr(occ_pred, occ_actual, size):
    """Cyan = predicted, Red = actual occupancy, as a single BGR image."""
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pred_u8   = _cv2.resize((occ_pred.T   * 255).astype(np.uint8), (W, H),
                             interpolation=_cv2.INTER_NEAREST)
    actual_u8 = _cv2.resize((occ_actual.T * 255).astype(np.uint8), (W, H),
                             interpolation=_cv2.INTER_NEAREST)
    img[:, :, 0] = np.maximum(img[:, :, 0], pred_u8)
    img[:, :, 1] = np.maximum(img[:, :, 1], pred_u8)
    img[:, :, 2] = np.maximum(img[:, :, 2], actual_u8)
    return img


def _save_prediction_video_eulerian(
    result: dict,
    ep_dir: str,
    model,              # EulerianModelWrapper — provides grid_bounds, grid_res,
                        # and initial_occ_from_particles()
    cam_params,
    goal_img_rgb: np.ndarray,
    fps: int = 2,
) -> None:
    """
    Write a per-episode Eulerian prediction comparison video.

    Each frame has two rows (mirroring visualize_prediction_eulerian.py):
      Top row  — rendered images with occupancy overlays
                  LEFT : current image + input occ (green)
                  RIGHT: next image + predicted occ (cyan) + actual occ (red)
      Bottom row — raw 2-D occupancy heatmaps
    """
    if not _CV2_OK:
        print('  Warning: cv2 not available — skipping Eulerian prediction video')
        return

    raw_obs     = result.get('raw_obs', [])
    states      = result.get('states', [])       # list of (N,3) raw particles
    states_pred = result.get('states_pred', [])  # list of (n_ahead,Nx,Ny) arrays
    rewards     = result.get('rewards')

    if (raw_obs is None or states is None or states_pred is None
            or len(raw_obs) == 0 or len(states) == 0 or len(states_pred) == 0):
        print('  Warning: raw_obs/states/states_pred missing — skipping Eulerian prediction video')
        return

    grid_bounds = model.grid_bounds
    grid_res    = tuple(model.grid_res)

    def _pts_to_occ(pts_np):
        """(N,3) particle cloud → (Nx,Ny) occupancy numpy array."""
        import torch as _torch
        pts_t = _torch.from_numpy(pts_np.astype(np.float32)).unsqueeze(0)
        with _torch.no_grad():
            occ = model.initial_occ_from_particles(pts_t)
        return occ[0].cpu().numpy()  # (Nx, Ny)

    n_steps = min(len(states) - 1, len(states_pred))
    if n_steps < 1:
        return

    obs0    = np.array(raw_obs[0])
    H, W    = obs0.shape[:2]
    HMAP_H, HMAP_W = H // 2, W // 2
    n_tiles = 3 if goal_img_rgb is not None else 2
    # Total frame width must accommodate both rows
    frame_w = W * n_tiles
    # Bottom row: 4 heatmap tiles at half-size, padded to frame_w
    # Total frame height = H (top) + HMAP_H (bottom) + 28 (legend)
    frame_h = H + HMAP_H + 28

    vid_path = os.path.join(ep_dir, 'prediction_video_eulerian.avi')
    fourcc   = _cv2.VideoWriter_fourcc(*'MJPG')
    writer   = _cv2.VideoWriter(vid_path, fourcc, fps, (frame_w, frame_h))

    GREEN  = (50,  200,  0)
    CYAN   = (200, 180,  0)
    RED    = (0,    50, 220)

    for step in range(n_steps):
        img_cur_rgb  = np.array(raw_obs[step])[..., :3].astype(np.uint8)
        img_next_rgb = np.array(raw_obs[step + 1])[..., :3].astype(np.uint8)

        img_cur_bgr  = _cv2.cvtColor(img_cur_rgb,  _cv2.COLOR_RGB2BGR)
        img_next_bgr = _cv2.cvtColor(img_next_rgb, _cv2.COLOR_RGB2BGR)

        occ_input  = _pts_to_occ(states[step])       # (Nx, Ny)
        occ_actual = _pts_to_occ(states[step + 1])   # (Nx, Ny)
        occ_pred   = states_pred[step][0]             # (Nx, Ny) — first look-ahead step

        r_cur  = float(rewards[step])     if rewards is not None else 0.0
        r_next = float(rewards[step + 1]) if rewards is not None else 0.0

        # ── top row ──────────────────────────────────────────────────────────
        tile_cur  = _draw_occ_overlay(img_cur_bgr,  occ_input,  grid_bounds,
                                       grid_res, cam_params, GREEN)
        tile_cur  = _lbl(tile_cur,  f"Step {step}  r={r_cur:.3f}  (input occ)")

        tile_next = _draw_occ_overlay(img_next_bgr, occ_pred,   grid_bounds,
                                       grid_res, cam_params, CYAN)
        tile_next = _draw_occ_overlay(tile_next,    occ_actual, grid_bounds,
                                       grid_res, cam_params, RED, radius=3)
        tile_next = _lbl(tile_next, f"After action  r={r_next:.3f}")

        top_tiles = [tile_cur, tile_next]
        if goal_img_rgb is not None:
            goal_bgr = _cv2.cvtColor(goal_img_rgb.astype(np.uint8), _cv2.COLOR_RGB2BGR)
            goal_bgr = _cv2.resize(goal_bgr, (W, H))
            top_tiles.append(_lbl(goal_bgr, 'Goal'))
        top_row = np.hstack(top_tiles)

        # ── heatmap row ───────────────────────────────────────────────────────
        hmap_in  = _lbl(_occ_to_heatmap_bgr(occ_input,  (HMAP_H, HMAP_W)), 'Input occ')
        hmap_pr  = _lbl(_occ_to_heatmap_bgr(occ_pred,   (HMAP_H, HMAP_W)), 'Predicted occ')
        hmap_ac  = _lbl(_occ_to_heatmap_bgr(occ_actual, (HMAP_H, HMAP_W)), 'Actual occ')
        hmap_du  = _lbl(_occ_to_dual_heatmap_bgr(occ_pred, occ_actual,
                                                  (HMAP_H, HMAP_W)), 'Cyan=pred / Red=actual')
        hmap_row = np.hstack([hmap_in, hmap_pr, hmap_ac, hmap_du])
        # Pad or crop to match frame_w
        if hmap_row.shape[1] < frame_w:
            pad = np.zeros((HMAP_H, frame_w - hmap_row.shape[1], 3), dtype=np.uint8)
            hmap_row = np.hstack([hmap_row, pad])
        else:
            hmap_row = hmap_row[:, :frame_w]

        # ── legend ────────────────────────────────────────────────────────────
        legend = np.zeros((28, frame_w, 3), dtype=np.uint8)
        for x, col, txt in [(8,   GREEN, 'input occ'),
                             (160, CYAN,  'predicted occ'),
                             (310, RED,   'actual occ')]:
            _cv2.circle(legend, (x + 6, 14), 6, col, -1)
            _cv2.putText(legend, txt, (x + 16, 19),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1)

        writer.write(np.vstack([top_row, hmap_row, legend]))

    writer.release()
    print(f'    Saved Eulerian prediction video → {vid_path}')


def _plot_comparison(all_summaries: list, out_dir: str) -> None:
    """
    Save two comparison charts across all experiments:
      1. Bar chart: occupancy reward gain ± std + compute time per MPC step
      2. Reward trajectory overlays (mean ± std)

    Both charts use the occupancy-based reporting reward (occ_rewards_all.npy /
    occ_reward_gain) as the primary metric for model-invariant comparison.
    Falls back to the native per-model reward only when occupancy data is absent.
    """
    names = [s['name'] for s in all_summaries]
    x     = np.arange(len(names))

    # Primary: occ_reward_gain — occupancy-based, model-invariant reporting metric.
    # Fallback: reward_gain (native per-model reward) when occupancy stats are absent.
    def _gain_stats(s):
        og = s.get('occ_reward_gain', {})
        if og and og.get('mean') is not None:
            return og['mean'], og.get('std', 0.0)
        rg = s.get('reward_gain', {})
        if rg and rg.get('mean') is not None:
            return rg['mean'], rg.get('std', 0.0)
        return 0.0, 0.0

    gain_data  = [_gain_stats(s) for s in all_summaries]
    gain_means = [g[0] for g in gain_data]
    gain_stds  = [g[1] for g in gain_data]

    # Compute time per MPC step
    ctimes     = [s.get('compute_time_per_step_s', {}).get('mean') for s in all_summaries]
    has_ctime  = all(c is not None for c in ctimes)

    # Prediction gap — commented out; may be useful for diagnosing model overconfidence
    # pred_gaps = [s['mean_prediction_gap']['mean']
    #              for s in all_summaries
    #              if s['mean_prediction_gap']['mean'] is not None]
    # has_pred  = len(pred_gaps) == len(all_summaries)

    n_cols = 2 if has_ctime else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # ── reward gain ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(x, gain_means, yerr=gain_stds, capsize=5,
           color='steelblue', edgecolor='black', width=0.6)
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Reporting reward gain  (final − initial)')
    ax.set_title('Mean reporting reward gain ± std')
    ax.grid(True, axis='y', alpha=0.3)

    # ── compute time per MPC step ─────────────────────────────────────────────
    if has_ctime:
        ax = axes[1]
        ax.bar(x, ctimes, color='goldenrod', edgecolor='black', width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Mean compute time per step (s)')
        ax.set_title('Compute time per MPC step')
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Experiment comparison', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'comparison.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)

    # ── trajectory overlay ────────────────────────────────────────────────────
    # Primary: occ_rewards_all.npy — occupancy-based reporting reward (model-invariant).
    # Fallback: rewards_all.npy (native per-model reward) when occupancy arrays are absent.
    traj = {}
    for s in all_summaries:
        res_dir = s.get('results_dir', '')
        p_occ = os.path.join(res_dir, 'occ_rewards_all.npy')
        p     = os.path.join(res_dir, 'rewards_all.npy')
        if os.path.exists(p_occ):
            traj[s['name']] = np.load(p_occ)
        elif os.path.exists(p):
            traj[s['name']] = np.load(p)

    if len(traj) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(traj)))
    for (nm, mat), color in zip(traj.items(), colors):
        mean  = mat.mean(axis=0)
        std   = mat.std(axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, 'o-', lw=2, color=color, label=nm)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel('MPC step')
    ax.set_ylabel('Reporting reward')
    ax.set_title('Reward trajectories — all experiments (mean ± std)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'trajectories.png'), dpi=130)
    plt.close(fig)


# ── env config sync ───────────────────────────────────────────────────────────

def _max_n_particles_in_suite(all_exp_specs: list, base_cfg: dict) -> int:
    """Return the maximum particle count across all experiments + the base config.

    Used to build the shared GenesisEnv with enough particles for every
    experiment; extras are parked outside camera view when not needed.
    """
    ds = base_cfg.get('dataset', {})
    base_n = int(ds.get('num_objects', ds.get('n_particles', 20)))
    max_n = base_n
    for exp in all_exp_specs:
        for k in ('dataset.num_objects', 'dataset.n_particles'):
            v = exp.get('overrides', {}).get(k)
            if v is not None:
                max_n = max(max_n, int(v))
    return max_n


def _sync_env_config(env, cfg: dict) -> None:
    """Propagate per-experiment config overrides to the shared GenesisEnv instance.

    The env is created once (from base_cfg) and reused across all experiments.
    When an experiment overrides ``dataset.*`` keys (e.g. ``dataset.obj``,
    ``dataset.num_objects``, ``dataset.global_scale``) those values must be
    pushed onto the env's Python attributes so that the next ``env.reset()``
    call picks them up.

    Genesis simulation is NOT re-initialised — only Python-level attributes are
    updated here.  Physics params (friction, density) are applied live via
    ``set_material_properties()``.  Particle count is applied by updating
    ``_n_active`` (takes effect on the next ``env.reset()``).
    """
    ds = cfg['dataset']

    env.obj           = ds.get('obj',           env.obj)
    env.wkspc_w       = ds.get('wkspc_w',        env.wkspc_w)
    env.init_pos      = ds.get('init_pos',       env.init_pos)
    env.num_objects_override = ds.get('num_objects', env.num_objects_override)

    new_scale = ds.get('global_scale', env.global_scale)
    scale_changed = (new_scale != env.global_scale)
    env.global_scale  = new_scale

    env.fast_mode                   = bool(ds.get('fast_mode',                   env.fast_mode))
    env.action_step_size            = float(ds.get('action_step_size',           env.action_step_size))
    env.settle_steps                = int(ds.get('settle_steps',                 env.settle_steps))
    env.reset_warmup_steps          = int(ds.get('reset_warmup_steps',           env.reset_warmup_steps))
    env.render_step_before_capture  = bool(ds.get('render_step_before_capture',  env.render_step_before_capture))

    # NOTE: GenesisEnv uses a fixed overhead camera; no per-scale camera recomputation needed.
    # (PyFleX camPos/camAngle block removed — Genesis camera is set at build time.)

    # ── live physics parameter sync ────────────────────────────────────────────
    # particle_friction / particle_density / box_friction can be changed at any
    # time via set_material_properties() — no sim rebuild required.
    physics_keys = {'particle_friction', 'particle_density', 'box_friction'}
    if any(k in ds for k in physics_keys):
        cur_mat = env._sim._material_params
        cur_box = env._sim._box_params
        env._sim.set_material_properties({
            'particle_friction':         ds.get('particle_friction',
                                                cur_mat.get('friction', 0.5) or 0.5),
            'particle_density':          ds.get('particle_density',
                                                cur_mat.get('density', 1000.0) or 1000.0),
            'box_friction':              ds.get('box_friction',
                                                cur_box.get('friction', 0.5) or 0.5),
            'sampled_particle_friction': None,
            'sampled_particle_density':  None,
        })

    # ── particle count sync ────────────────────────────────────────────────────
    # n_particles is fixed in the Genesis scene topology, but we can "park"
    # excess particles outside camera view.  Takes effect on next env.reset().
    new_n = ds.get('num_objects', ds.get('n_particles', None))
    if new_n is not None:
        env.set_active_particles(int(new_n))

    # ── timing knob sync ───────────────────────────────────────────────────────
    # settle_steps / reset_warmup_steps are forwarded live by GenesisEnv.step()
    # and reset() via env.settle_steps / env.reset_warmup_steps (already synced
    # above).  pos_ctrl_steps controls lowering/lifting and must also be pushed
    # to the sim's internal tensors immediately.
    if 'pos_ctrl_steps' in ds:
        n = int(ds['pos_ctrl_steps'])
        env._sim._pos_ctrl_steps = n
        env._sim._steps_0to1 = torch.linspace(
            0, 1, n, device=env._sim._steps_0to1.device)

    # Keep env.config in sync so adapters and reset() read correct values.
    env.config = cfg


# ── single episode ────────────────────────────────────────────────────────────

def run_episode(
    env,
    model,
    cfg: dict,
    ep_idx: int,
    seed_base: int,
    fixed_initial_pos,        # np.ndarray | None
    ep_dir: str,
    output_cfg: dict,
    model_for_video=None,     # passed through to Eulerian video function
) -> dict:
    """Reset the env, run one MPC episode, save outputs, return metrics dict."""
    seed = seed_base + ep_idx
    set_seed(seed)

    env.reset()
    if fixed_initial_pos is not None:
        env.set_positions(fixed_initial_pos)

    # Tag every real step's incremental transition flush (see
    # env.genesis_env.GenesisEnv.step()/set_recording_context) with this
    # episode's identity, set BEFORE the episode runs — reward/success
    # aren't known yet, so they aren't included here; they're saved
    # separately below (metrics.json) and joinable by source + episode_idx.
    if output_cfg.get('save_mpc_transitions', True) and hasattr(env, 'set_recording_context'):
        env.set_recording_context({
            'source':      'simple_mpc.mpc',
            'episode_idx': ep_idx,
            'seed':        seed,
            'model_type':  type(model).__name__,
        })

    subgoal, goal_img = build_goal(cfg, env)

    # Scale goal so its pixel footprint matches the current material extent
    obs_init = env.render()
    subgoal  = scale_subgoal_to_material_pixels(
        subgoal, obs_init[..., -1], cfg['dataset']['global_scale'])

    need_video = bool(output_cfg.get('save_prediction_videos', False))
    need_raw_obs = bool(output_cfg.get('save_raw_obs', False)) or need_video
    need_states = bool(output_cfg.get('save_states', False)) or need_video
    need_states_pred = need_video

    result  = run_simple_mpc(
        env,
        model,
        subgoal,
        cfg,
        collect_raw_obs=need_raw_obs,
        collect_states=need_states,
        collect_states_pred=need_states_pred,
    )
    metrics = compute_episode_metrics(
        result, success_threshold=output_cfg.get('success_threshold', 0.0))

    os.makedirs(ep_dir, exist_ok=True)

    with open(os.path.join(ep_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    if output_cfg.get('save_rewards', True):
        np.save(os.path.join(ep_dir, 'rewards.npy'), result['rewards'])
        if 'occ_rewards' in result:
            np.save(os.path.join(ep_dir, 'occ_rewards.npy'), result['occ_rewards'])

    if output_cfg.get('save_actions', True):
        np.save(os.path.join(ep_dir, 'actions.npy'), result['actions'])

    if output_cfg.get('save_raw_obs', False):
        np.save(os.path.join(ep_dir, 'raw_obs.npy'), result['raw_obs'])

    if output_cfg.get('save_states', False):
        arr = np.empty(len(result['states']), dtype=object)
        for k, s in enumerate(result['states']):
            arr[k] = s
        np.save(os.path.join(ep_dir, 'states.npy'), arr, allow_pickle=True)

    if output_cfg.get('save_per_episode_npz', True):
        npz_data = {
            'rewards':   result['rewards'],
            'actions':   result['actions'],
            'rew_means': result['rew_means'],
            'rew_stds':  result['rew_stds'],
        }
        bps = result.get('best_rewards_per_step', [])
        if bps:
            npz_data['best_rewards_per_step'] = np.array(
                [b if b is not None else np.nan for b in bps], dtype=np.float32)
        np.savez(os.path.join(ep_dir, 'episode_data.npz'), **npz_data)

    if output_cfg.get('save_reward_plot', True):
        plot_rewards = result.get('occ_rewards', result['rewards'])
        _plot_reward_episode(plot_rewards, os.path.join(ep_dir, 'rewards.png'))

    if output_cfg.get('save_prediction_videos', False):
        cam_params = env.get_cam_params()
        fps        = output_cfg.get('prediction_video_fps', 2)
        from model.eulerian_wrapper import EulerianModelWrapper as _EW
        if isinstance(model, _EW):
            _save_prediction_video_eulerian(
                result, ep_dir, model, cam_params, goal_img, fps=fps)
        else:
            _save_prediction_video(
                result, ep_dir, cam_params, goal_img, fps=fps)

    # Real-step transitions (before/after particle state + action) were
    # already flushed to disk incrementally during the episode — see
    # set_recording_context() above and
    # env.genesis_env.GenesisEnv.step()/push_and_record
    # (docs/ARCHITECTURE.md). Nothing left to do here; this episode's
    # outcome (metrics, above) is joinable with its transitions by
    # source + episode_idx.

    return metrics


# ── single experiment (N episodes) ────────────────────────────────────────────

def run_experiment(
    exp_spec: dict,
    base_cfg: dict,
    env,
    output_cfg: dict,
    episodes_cfg: dict,
    exp_dir: str,
) -> dict:
    """Run all episodes for one experiment entry.  Returns a summary dict."""
    name       = exp_spec['name']
    desc       = exp_spec.get('description', '')
    overrides  = exp_spec.get('overrides', {})
    n_episodes = exp_spec.get('n_episodes', episodes_cfg.get('n_episodes', 5))

    # Build the effective config for this experiment
    cfg = apply_overrides(base_cfg, overrides)

    # Model loading is driven by model cards (or explicit eulerian heuristics).
    model_spec = exp_spec.get('model', {})

    os.makedirs(exp_dir, exist_ok=True)

    # ── sync env attributes from per-experiment cfg ───────────────────────────
    # Must happen before load_model so the Eulerian model gets the right
    # cam_extrinsic and global_scale when dataset.* keys are overridden.
    _sync_env_config(env, cfg)

    # ── header ────────────────────────────────────────────────────────────────
    sep = '=' * 65
    print(f'\n{sep}')
    print(f'Experiment: {name}')
    if desc:
        print(f'  {desc}')
    if overrides:
        for k, v in overrides.items():
            print(f'  override  {k} = {v}')
    print(f'  n_episodes={n_episodes}  output: {exp_dir}')
    print(sep)

    seed_base   = episodes_cfg.get('random_seed_base', 42)
    fpos_file   = episodes_cfg.get('fixed_initial_pos_file', None)
    fixed_pos   = np.load(fpos_file) if fpos_file else None

    # Force-reload models when dataset.* keys are overridden because inference
    # geometry and/or normalisation can be embedded in wrapper construction.
    dataset_override = any(k.startswith('dataset.') for k in overrides)
    force_reload = dataset_override
    model       = load_model(model_spec, cfg, env=env, force_reload=force_reload)
    
    # ── benchmark model throughput (once per experiment) ──────────────────────
    if output_cfg.get('benchmark_throughput', False):
        wkspc_w = cfg['dataset']['wkspc_w']
        benchmark_push_throughput(model, wkspc_w=wkspc_w)
    
    all_metrics = []
    t_exp_start = time.time()

    for ep in range(n_episodes):
        print(f'\n  --- Episode {ep + 1}/{n_episodes} ---')
        ep_dir_ep = os.path.join(exp_dir, f'episode_{ep:03d}')
        metrics = run_episode(
            env, model, cfg, ep,
            seed_base=seed_base,
            fixed_initial_pos=fixed_pos,
            ep_dir=ep_dir_ep,
            output_cfg=output_cfg,
        )
        all_metrics.append(metrics)
        print(f'  → gain={metrics["reward_gain"]:+.4f}  '
              f'final={metrics["reward_final"]:.4f}  '
              f'success={metrics["success"]}  '
              f't={metrics["total_time_s"]:.1f}s')

    t_exp = time.time() - t_exp_start

    # ── aggregate stats ───────────────────────────────────────────────────────
    gains   = [m['reward_gain']    for m in all_metrics]
    finals  = [m['reward_final']   for m in all_metrics]
    inits   = [m['reward_initial'] for m in all_metrics]
    success = [m['success']        for m in all_metrics]
    ctimes      = [m['compute_time_s'] for m in all_metrics]
    ctimes_step = [m['compute_time_per_step_s'] for m in all_metrics
                   if m.get('compute_time_per_step_s') is not None]
    pgaps       = [m['mean_prediction_gap'] for m in all_metrics
                   if m.get('mean_prediction_gap') is not None]
    occ_gains   = [m['occ_reward_gain']    for m in all_metrics if 'occ_reward_gain'    in m]
    occ_finals  = [m['occ_reward_final']   for m in all_metrics if 'occ_reward_final'   in m]
    occ_inits   = [m['occ_reward_initial'] for m in all_metrics if 'occ_reward_initial' in m]

    def _stats(arr):
        a = np.array(arr, dtype=float)
        return {'mean': float(a.mean()), 'std': float(a.std()),
                'min':  float(a.min()),  'max': float(a.max()),
                'median': float(np.median(a))}

    summary = {
        'name':                   name,
        'description':            desc,
        'n_episodes':             n_episodes,
        'overrides':              overrides,
        'model_spec':             model_spec,
        'results_dir':            exp_dir,
        'reward_gain':            _stats(gains),
        'reward_final':           _stats(finals),
        'reward_initial':         _stats(inits),
        'success_rate':           float(np.mean(success)),
        'compute_time_s':              _stats(ctimes),
        'compute_time_per_step_s':     _stats(ctimes_step) if ctimes_step
                                       else {'mean': None, 'std': None, 'min': None,
                                             'max': None, 'median': None},
        'occ_reward_gain':             _stats(occ_gains)  if occ_gains
                                       else {'mean': None, 'std': None, 'min': None,
                                             'max': None, 'median': None},
        'occ_reward_final':            _stats(occ_finals) if occ_finals
                                       else {'mean': None, 'std': None, 'min': None,
                                             'max': None, 'median': None},
        'occ_reward_initial':          _stats(occ_inits)  if occ_inits
                                       else {'mean': None, 'std': None, 'min': None,
                                             'max': None, 'median': None},
        'mean_prediction_gap':         _stats(pgaps) if pgaps else {'mean': None, 'std': None,
                                                                     'min': None, 'max': None,
                                                                     'median': None},
        'total_experiment_time_s': t_exp,
        'episodes':               all_metrics,
    }

    with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── stack reward arrays (if same length) → rewards_all.npy / occ_rewards_all.npy ─
    reward_arrs     = []
    occ_reward_arrs = []
    for ep in range(n_episodes):
        ep_path = os.path.join(exp_dir, f'episode_{ep:03d}')
        p = os.path.join(ep_path, 'rewards.npy')
        if os.path.exists(p):
            reward_arrs.append(np.load(p))
        p_occ = os.path.join(ep_path, 'occ_rewards.npy')
        if os.path.exists(p_occ):
            occ_reward_arrs.append(np.load(p_occ))

    if reward_arrs and all(len(r) == len(reward_arrs[0]) for r in reward_arrs):
        mat = np.stack(reward_arrs)    # (n_episodes, n_mpc+1)
        np.save(os.path.join(exp_dir, 'rewards_all.npy'), mat)

    if occ_reward_arrs and all(len(r) == len(occ_reward_arrs[0]) for r in occ_reward_arrs):
        occ_mat = np.stack(occ_reward_arrs)
        np.save(os.path.join(exp_dir, 'occ_rewards_all.npy'), occ_mat)
        if output_cfg.get('save_reward_plot', True):
            _plot_reward_experiment(
                occ_mat, name,
                os.path.join(exp_dir, 'rewards_summary.png'))
    elif reward_arrs and all(len(r) == len(reward_arrs[0]) for r in reward_arrs):
        if output_cfg.get('save_reward_plot', True):
            _plot_reward_experiment(
                mat, name,
                os.path.join(exp_dir, 'rewards_summary.png'))

    print(f'\n  [{name}] done in {t_exp:.0f}s'
          f' | gain {np.mean(gains):+.4f}±{np.std(gains):.4f}'
          f' | success {np.mean(success)*100:.0f}%')

    return summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run a batch of MPC experiments from a YAML suite config.')
    parser.add_argument(
        'suite_config',
        help='Path to experiment suite YAML (e.g. config/experiments/compare_models.yaml)')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print experiment plan without running anything')
    parser.add_argument(
        '--only', nargs='+', metavar='NAME',
        help='Run only the named experiments (space-separated)')
    args = parser.parse_args()

    suite = load_yaml(args.suite_config)

    base_cfg_path = suite.get('base_config', 'simple_mpc/config/config_simple.yaml')
    base_cfg      = load_yaml(base_cfg_path)
    output_cfg    = suite.get('output', {})
    episodes_cfg  = suite.get('episodes', {})
    all_exp_specs = suite['experiments']

    # ── filter by --only ──────────────────────────────────────────────────────
    if args.only:
        known = {e['name'] for e in all_exp_specs}
        unknown = set(args.only) - known
        if unknown:
            print(f"ERROR: unknown experiment name(s): {sorted(unknown)}", file=sys.stderr)
            print(f"  Available: {sorted(known)}", file=sys.stderr)
            sys.exit(1)
        all_exp_specs = [e for e in all_exp_specs if e['name'] in args.only]

    # ── output directories ────────────────────────────────────────────────────
    suite_name  = output_cfg.get('experiment_name', 'experiment_suite')
    timestamp   = get_current_YYYY_MM_DD_hh_mm_ss_ms()
    base_root   = output_cfg.get('root_dir', 'outputs/experiments')
    compare_dir = os.path.join(base_root, f'{suite_name}_{timestamp}')
    os.makedirs(compare_dir, exist_ok=True)

    # Copy config for reproducibility
    shutil.copy(args.suite_config,
                os.path.join(compare_dir, 'experiment_suite.yaml'))

    # ── dry-run: just print the plan ─────────────────────────────────────────
    if args.dry_run:
        print(f'\nExperiment suite: {args.suite_config}')
        print(f'Base config:      {base_cfg_path}')
        print(f'Compare dir:      {compare_dir}')
        print(f'Per-experiment:   {base_root}/<name>/{timestamp}/')
        print(f'\nPlan ({len(all_exp_specs)} experiments):')
        total_episodes = 0
        for i, exp in enumerate(all_exp_specs):
            n   = exp.get('n_episodes', episodes_cfg.get('n_episodes', 5))
            mcfg = exp.get('model', {})
            mt  = mcfg.get('type', 'model-card')
            if mt in ('eulerian-heuristic',) or (mt == 'eulerian' and not mcfg.get('need_weights', True) and 'model_card' not in mcfg and 'training_config' not in mcfg):
                msrc = f"heuristic:{mcfg.get('heuristic_type', 'spread')}"
            elif 'model_card' in mcfg:
                msrc = mcfg['model_card']
            else:
                msrc = mcfg.get('training_config', '?')
            ovr = exp.get('overrides', {})
            total_episodes += n
            print(f'  {i+1:2d}. {exp["name"]:<30}  '
                  f'{n:3d} ep  model={mt}/{msrc}  '
                  f'overrides={list(ovr.keys())}')
        print(f'\n  Total: {total_episodes} episodes across {len(all_exp_specs)} experiments')
        return

    # ── create env (shared across all experiments; reset between episodes) ────
    # Build with the largest particle count seen in any experiment so that
    # per-experiment particle counts can be honoured without a sim rebuild.
    max_n = _max_n_particles_in_suite(all_exp_specs, base_cfg)
    base_ds = base_cfg.setdefault('dataset', {})
    base_n  = int(base_ds.get('num_objects', base_ds.get('n_particles', 20)))
    if max_n > base_n:
        print(f'  Pre-allocating {max_n} particles (max across experiments; '
              f'base config has {base_n})')
        base_ds['num_objects'] = max_n
    print(f'\nCreating environment …')
    env = GenesisEnv(base_cfg)

    all_summaries = []
    t_total_start = time.time()

    for exp_spec in all_exp_specs:
        exp_dir = os.path.join(base_root, exp_spec['name'], timestamp)
        summary = run_experiment(
            exp_spec    = exp_spec,
            base_cfg    = base_cfg,
            env         = env,
            output_cfg  = output_cfg,
            episodes_cfg= episodes_cfg,
            exp_dir     = exp_dir,
        )
        all_summaries.append(summary)

    t_total = time.time() - t_total_start

    # ── overall summary JSON ──────────────────────────────────────────────────
    # Ensure compare_dir exists (defensive; should already exist from earlier creation)
    os.makedirs(compare_dir, exist_ok=True)
    
    overall = {
        'suite_config':   args.suite_config,
        'base_config':    base_cfg_path,
        'total_time_s':   t_total,
        'n_experiments':  len(all_summaries),
        'experiments': [
            {k: v for k, v in s.items() if k != 'episodes'}
            for s in all_summaries
        ],
    }
    with open(os.path.join(compare_dir, 'overall_summary.json'), 'w') as f:
        json.dump(overall, f, indent=2)

    # ── comparison plots ──────────────────────────────────────────────────────
    try:
        _plot_comparison(all_summaries, compare_dir)
    except Exception as e:
        print(f'  Warning: comparison plots failed: {e}')

    # ── summary table ─────────────────────────────────────────────────────────
    sep = '=' * 75
    print(f'\n{sep}')
    print(f'All {len(all_summaries)} experiments done in {t_total:.0f}s')
    print(f'Compare dir:      {compare_dir}')
    print(f'Per-experiment:   {base_root}/<name>/{timestamp}/')
    print(sep)
    col = f"{'Experiment':<32} {'Gain mean':>10} {'±std':>8} {'Final':>8} {'Success':>9}"
    print(f'\n{col}')
    print('-' * len(col))
    for s in all_summaries:
        print(f"{s['name']:<32} "
              f"{s['reward_gain']['mean']:>+10.4f} "
              f"{s['reward_gain']['std']:>8.4f} "
              f"{s['reward_final']['mean']:>8.4f} "
              f"{s['success_rate']*100:>8.0f}%")
    print()


if __name__ == '__main__':
    main()
