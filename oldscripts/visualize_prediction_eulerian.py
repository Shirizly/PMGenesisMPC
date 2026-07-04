"""
Visualize Eulerian model input representation and prediction quality during MPC.

For each MPC step produces a composite panel with two rows:

  Top row (camera-projected overlay on rendered images):
    [A] Current rendered image  + input occupancy overlay (green  = model input)
    [B] Actual next image       + predicted occ (cyan) + actual occ (red)
    [C] Goal image (first step only)

  Bottom row (raw 2-D occupancy heatmaps for direct inspection):
    [A'] Input occupancy heatmap
    [B'] Predicted occupancy heatmap
    [C'] Actual-next occupancy heatmap

This layout lets you directly compare what the Eulerian model sees (2-D
occupancy grid) with the corresponding rendered simulator images, and judge
how well the model's prediction matches reality.

All MPC and model settings are read from config/mpc/config.yaml and the
constants block at the top of this file (mirrored from visualize_mpc.py).

Usage:
    python visualize_prediction_eulerian.py
"""

import os
import cv2
import numpy as np
import torch
from typing import Optional, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from env.genesis_env import GenesisEnv
from model.eulerian_wrapper import (
    EulerianModelWrapper, SplatPushModel, FluidPushModel, SpreadPushModel,
    UNetFiLMPushModel, _action_to_cam_3d_genesis,
)
from utils import (load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms,
                   pcd2pix, gen_goal_shape, gen_subgoal,
                   scale_subgoal_to_material_pixels)


# ── Eulerian model configuration (mirrors visualize_mpc.py) ──────────────────
# EULERIAN_MODEL_TYPE and GRID_RES are overridden at runtime by
# config['mpc']['model']['type'] when USE_SIMPLE_MPC is True.
EULERIAN_MODEL_TYPE = 'spread'     # fallback default
GRID_RES = (64, 64)

# ── Simple MPC switch ─────────────────────────────────────────────────────────
# Set USE_SIMPLE_MPC = True to call simple_mpc.run_simple_mpc() instead of
# env.step_subgoal_ptcl().  The config is then loaded from SIMPLE_MPC_CONFIG
# (which replaces config/mpc/config.yaml for all MPC hyperparameters).
USE_SIMPLE_MPC    = True
SIMPLE_MPC_CONFIG = 'simple_mpc/config/config_simple.yaml'

TOOL_WIDTH              = 5.0
ACTION_SIGMA              = 0.5
SPLAT_REDISTRIBUTE       = False
SPLAT_REDISTRIBUTE_ITERS = 10

FLUID_WIDTH              = 5.0
FLUID_SIGMA              = 1.5
FLUID_N_STEPS            = 20
FLUID_DECAY              = 0.95
FLUID_MEDIA_SHARPNESS    = 5.0
FLUID_BLUR_SIGMA         = 1.0
FLUID_CORRECT_DIVERGENCE = False
FLUID_REDISTRIBUTE       = False
FLUID_REDISTRIBUTE_ITERS = 10

# ─────────────────────────────────────────────────────────── occ helpers ─────

def _occ_grid_to_3d(
    occ_np: np.ndarray,         # (Nx, Ny) float [0..1]
    grid_bounds: Dict[str, float],
    grid_res: Tuple[int, int],
    threshold: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D occupancy grid to 3-D points in normalized camera space.

    Grid axis 0 → camera x (column direction); axis 1 → camera y (row direction).
    Depth (camera z) is set to z_mid (mid of z bounds).

    Returns
    -------
    pts   : (M, 3) float32  – point cloud in normalized cam coords
    vals  : (M,)   float32  – occupancy value of each occupied cell [threshold..1]
    """
    Nx, Ny = grid_res
    x_min, x_max = grid_bounds['x_min'], grid_bounds['x_max']
    y_min, y_max = grid_bounds['y_min'], grid_bounds['y_max']
    z_mid = 0.5 * (grid_bounds.get('z_min', 0.70) + grid_bounds.get('z_max', 0.80))

    # Build all cell-centre positions at once (vectorised)
    ix = np.arange(Nx, dtype=np.float32)
    iy = np.arange(Ny, dtype=np.float32)
    X = x_min + ix * (x_max - x_min) / max(Nx - 1, 1)   # (Nx,)
    Y = y_min + iy * (y_max - y_min) / max(Ny - 1, 1)   # (Ny,)
    XX, YY = np.meshgrid(X, Y, indexing='ij')             # (Nx, Ny)

    mask = occ_np >= threshold
    pts_x = XX[mask]
    pts_y = YY[mask]
    vals  = occ_np[mask]

    n = pts_x.shape[0]
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = pts_x         # camera x
    pts[:, 1] = pts_y         # camera y
    pts[:, 2] = z_mid         # camera z (depth)
    return pts, vals.astype(np.float32)


def _draw_occ_overlay(
    img_bgr: np.ndarray,
    occ_np: np.ndarray,         # (Nx, Ny)
    grid_bounds: Dict[str, float],
    grid_res: Tuple[int, int],
    cam_params,
    color_bgr: tuple,
    radius: int = 4,
    alpha: float = 0.75,
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Project occupied voxels of *occ_np* to pixel space and draw coloured
    circles on a copy of *img_bgr*.  Returns modified BGR image.
    """
    pts, vals = _occ_grid_to_3d(occ_np, grid_bounds, grid_res, threshold)
    if pts.shape[0] == 0:
        return img_bgr.copy()

    pix = pcd2pix(pts, cam_params)   # (M, 2) [row, col]
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    for (r, c), v in zip(pix, vals):
        if 0 <= r < H and 0 <= c < W:
            # scale radius by occupancy strength
            r_draw = max(1, int(radius * v))
            cv2.circle(overlay, (int(c), int(r)), r_draw, color_bgr, -1, cv2.LINE_AA)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)


def _occ_to_heatmap_bgr(
    occ_np: np.ndarray,         # (Nx, Ny) float [0..1]
    size: Tuple[int, int],      # (H, W) output size
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """
    Convert an occupancy grid to a false-colour BGR heatmap image.

    occ_np is transposed so that axis 0 (camera-x / columns) maps to
    the horizontal image axis, matching the projected overlay.
    """
    # Transpose: grid[ix, iy] → image rows=iy cols=ix
    grid_disp = occ_np.T                               # (Ny, Nx)
    grid_u8   = (np.clip(grid_disp, 0.0, 1.0) * 255).astype(np.uint8)
    colored   = cv2.applyColorMap(grid_u8, colormap)  # (Ny, Nx, 3) BGR
    return cv2.resize(colored, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def _occ_to_dual_heatmap_bgr(
    occ_pred: np.ndarray,       # (Nx, Ny)
    occ_actual: np.ndarray,     # (Nx, Ny)
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Overlay predicted (cyan) and actual (red) occupancies as a dual heatmap.

    Output is a BGR image of shape (H, W, 3) with:
      - cyan  channel weighted by occ_pred
      - red   channel weighted by occ_actual
    """
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    Ny, Nx = occ_pred.T.shape

    pred_u8   = cv2.resize((occ_pred.T  * 255).astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST)
    actual_u8 = cv2.resize((occ_actual.T * 255).astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST)

    # BGR: cyan = (B, G, 0), red = (0, 0, R)
    img[:, :, 0] = np.maximum(img[:, :, 0], pred_u8)    # B → cyan
    img[:, :, 1] = np.maximum(img[:, :, 1], pred_u8)    # G → cyan
    img[:, :, 2] = np.maximum(img[:, :, 2], actual_u8)  # R → red
    return img

# ──────────────────────────────────────── action-arrow helpers ───────────────

def _action_world_to_pixels(
    action: np.ndarray,          # (4,) [sx, sy, ex, ey] world 2-D
    cam_extrinsic: np.ndarray,   # (4, 4)
    global_scale: float,
    cam_params,                  # [fx, fy, cx, cy]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a world-2D push action to two (row, col) pixel coordinates
    for the start and end points.
    """
    act_t = torch.tensor(action[None], dtype=torch.float32)  # (1, 4)
    s_cam, e_cam = _action_to_cam_3d_genesis(act_t, global_scale)
    s_pix = pcd2pix(s_cam.detach().cpu().numpy(), cam_params)  # (1, 2) [row, col]
    e_pix = pcd2pix(e_cam.detach().cpu().numpy(), cam_params)
    return s_pix[0], e_pix[0]   # each (2,) [row, col]


def _draw_action_arrow(
    img_bgr: np.ndarray,
    action: np.ndarray,
    cam_extrinsic: np.ndarray,
    global_scale: float,
    cam_params,
    color_bgr: tuple = (50, 255, 255),   # yellow
    thickness: int = 2,
) -> np.ndarray:
    """Draw the push action as an arrow on img_bgr."""
    s_pix, e_pix = _action_world_to_pixels(
        action, cam_extrinsic, global_scale, cam_params)
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()
    sp = (int(np.clip(s_pix[1], 0, W - 1)),
          int(np.clip(s_pix[0], 0, H - 1)))   # (col, row) for cv2
    ep = (int(np.clip(e_pix[1], 0, W - 1)),
          int(np.clip(e_pix[0], 0, H - 1)))
    cv2.arrowedLine(out, sp, ep, color_bgr, thickness, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(out, sp, 5, color_bgr, -1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────── score-map helpers ────────────

def _score_to_heatmap_bgr(
    score_np: np.ndarray,       # (Nx, Ny)
    size: Tuple[int, int],
) -> np.ndarray:
    """Render the goal score map as a BGR heatmap (same conventions as _occ_to_heatmap_bgr)."""
    grid_disp = score_np.T
    vmin, vmax = grid_disp.min(), grid_disp.max()
    if vmax > vmin:
        norm = (grid_disp - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(grid_disp)
    u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_HOT)
    return cv2.resize(colored, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def _draw_score_overlay(
    img_bgr: np.ndarray,
    score_np: np.ndarray,       # (Nx, Ny)  — from prepare_goal_reward
    grid_bounds: Dict[str, float],
    grid_res: Tuple[int, int],
    cam_params,
    radius: int = 4,
    alpha: float = 0.60,
) -> np.ndarray:
    """
    Overlay the goal score map on img_bgr.  High-score voxels are shown
    as bright dots (warm colour), low-score voxels are cool/dark.
    """
    Nx, Ny = grid_res
    x_min, x_max = grid_bounds['x_min'], grid_bounds['x_max']
    y_min, y_max = grid_bounds['y_min'], grid_bounds['y_max']
    z_mid = 0.5 * (grid_bounds.get('z_min', 0.70) + grid_bounds.get('z_max', 0.80))

    ix = np.arange(Nx, dtype=np.float32)
    iy = np.arange(Ny, dtype=np.float32)
    X = x_min + ix * (x_max - x_min) / max(Nx - 1, 1)
    Y = y_min + iy * (y_max - y_min) / max(Ny - 1, 1)
    XX, YY = np.meshgrid(X, Y, indexing='ij')   # (Nx, Ny)

    flat_x  = XX.ravel()
    flat_y  = YY.ravel()
    flat_s  = score_np.ravel()
    pts = np.stack([flat_x, flat_y,
                    np.full(len(flat_x), z_mid)], axis=1).astype(np.float32)

    # normalise scores to [0, 1] for colour mapping
    smin, smax = flat_s.min(), flat_s.max()
    flat_sn = (flat_s - smin) / (smax - smin + 1e-8)

    pix = pcd2pix(pts, cam_params)   # (N, 2) [row, col]
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    cmap = plt.cm.hot
    for (r, c), sn in zip(pix, flat_sn):
        if sn < 0.15:        # skip very-low-score voxels to avoid clutter
            continue
        if not (0 <= r < H and 0 <= c < W):
            continue
        col = tuple(int(v * 255) for v in cmap(float(sn))[2::-1])  # RGB→BGR
        cv2.circle(overlay, (int(c), int(r)), radius, col, -1, cv2.LINE_AA)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, overlay, alpha, 0)

# ────────────────────────────────────────────── panel-building helpers ────────

def _label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0,   0,   0), 1, cv2.LINE_AA)
    return out


def make_eulerian_panel(
    img_cur_rgb:  np.ndarray,       # (H, W, 3) current rendered image
    occ_input:    np.ndarray,       # (Nx, Ny) occupancy from current particles
    img_next_rgb: np.ndarray,       # (H, W, 3) actual next rendered image
    occ_pred:     np.ndarray,       # (Nx, Ny) model-predicted next occupancy
    occ_actual:   np.ndarray,       # (Nx, Ny) actual next occupancy
    grid_bounds:  Dict[str, float],
    grid_res:     Tuple[int, int],
    cam_params,
    step: int,
    reward_cur:   float,
    reward_next:  float,
    action:       Optional[np.ndarray] = None,  # (4,) [sx,sy,ex,ey] world 2-D
    cam_extrinsic: Optional[np.ndarray] = None,
    global_scale:  float = 24.0,
    score_np:     Optional[np.ndarray] = None,  # (Nx, Ny) goal score map
    goal_img_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a two-row composite panel for one MPC step.

    Row 0 – camera-projected overlays on rendered images
    Row 1 – raw 2-D occupancy grid heatmaps
    """
    H, W = img_cur_rgb.shape[:2]
    HMAP_H = H // 2   # heatmap row is half-height to keep panel compact
    HMAP_W = W // 2

    GREEN = (50,  200,  0)
    CYAN  = (200, 180,  0)
    RED   = (0,    50, 220)

    img_cur_bgr  = cv2.cvtColor(img_cur_rgb.astype(np.uint8),  cv2.COLOR_RGB2BGR)
    img_next_bgr = cv2.cvtColor(img_next_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    YELLOW = (50,  255, 255)

    # ── top row ──────────────────────────────────────────────────────────────
    tile_cur  = _draw_occ_overlay(img_cur_bgr,  occ_input,  grid_bounds, grid_res,
                                   cam_params, GREEN, radius=5, alpha=0.75)
    # Optionally overlay the goal score map on the current image
    if score_np is not None:
        tile_cur = _draw_score_overlay(tile_cur, score_np, grid_bounds, grid_res,
                                       cam_params, radius=3, alpha=0.45)
    # Draw the chosen action arrow on the current-state tile
    if action is not None and cam_extrinsic is not None:
        tile_cur = _draw_action_arrow(tile_cur, action, cam_extrinsic, global_scale,
                                      cam_params, color_bgr=YELLOW)
    tile_cur  = _label(tile_cur,  f"Step {step}  r={reward_cur:.3f}  (input occ)")

    tile_next = _draw_occ_overlay(img_next_bgr, occ_pred,   grid_bounds, grid_res,
                                   cam_params, CYAN,  radius=5, alpha=0.65)
    tile_next = _draw_occ_overlay(tile_next,    occ_actual, grid_bounds, grid_res,
                                   cam_params, RED,   radius=3, alpha=0.75)
    if action is not None and cam_extrinsic is not None:
        tile_next = _draw_action_arrow(tile_next, action, cam_extrinsic, global_scale,
                                       cam_params, color_bgr=YELLOW)
    tile_next = _label(tile_next, f"After action  r={reward_next:.3f}")

    top_tiles = [tile_cur, tile_next]
    if goal_img_rgb is not None:
        goal_bgr = cv2.cvtColor(goal_img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        goal_bgr = cv2.resize(goal_bgr, (W, H))
        top_tiles.append(_label(goal_bgr, "Goal"))

    top_row = np.hstack(top_tiles)

    # ── heatmap row ───────────────────────────────────────────────────────────
    hmap_input  = _occ_to_heatmap_bgr(occ_input,  (HMAP_H, HMAP_W))
    hmap_input  = _label(hmap_input,  "Input occ")
    hmap_pred   = _occ_to_heatmap_bgr(occ_pred,   (HMAP_H, HMAP_W))
    hmap_pred   = _label(hmap_pred,   "Predicted occ")
    hmap_actual = _occ_to_heatmap_bgr(occ_actual,  (HMAP_H, HMAP_W))
    hmap_actual = _label(hmap_actual, "Actual occ")
    hmap_diff   = _occ_to_dual_heatmap_bgr(occ_pred, occ_actual, (HMAP_H, HMAP_W))
    hmap_diff   = _label(hmap_diff,   "Cyan=pred / Red=actual")
    hmap_tiles  = [hmap_input, hmap_pred, hmap_actual, hmap_diff]

    # Score-map heatmap (goal landscape in grid space)
    if score_np is not None:
        hmap_score = _score_to_heatmap_bgr(score_np, (HMAP_H, HMAP_W))
        hmap_score = _label(hmap_score, "Goal score map")
        hmap_tiles.append(hmap_score)

    if goal_img_rgb is not None:
        hmap_blank = np.zeros((HMAP_H, W, 3), dtype=np.uint8)
        hmap_tiles.append(hmap_blank)

    # Ensure total heatmap row width matches top row
    hmap_row_raw = np.hstack(hmap_tiles)
    if hmap_row_raw.shape[1] < top_row.shape[1]:
        pad = np.zeros((HMAP_H, top_row.shape[1] - hmap_row_raw.shape[1], 3),
                       dtype=np.uint8)
        hmap_row_raw = np.hstack([hmap_row_raw, pad])
    else:
        hmap_row_raw = hmap_row_raw[:, :top_row.shape[1]]

    # ── legend strip ──────────────────────────────────────────────────────────
    total_w = top_row.shape[1]
    leg_h = 28
    legend = np.zeros((leg_h, total_w, 3), dtype=np.uint8)
    items = [(8,   GREEN,  "input occ"),
             (160, CYAN,   "predicted occ"),
             (310, RED,    "actual occ"),
             (440, YELLOW, "chosen action")]
    for x, col, txt in items:
        cv2.circle(legend, (x + 6, leg_h // 2), 6, col, -1)
        cv2.putText(legend, txt, (x + 16, leg_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1)

    return np.vstack([top_row, hmap_row_raw, legend])


# ─────────────────────────────────────────────────────────────── main ─────────

def main():
    config = load_yaml(SIMPLE_MPC_CONFIG)

    # ── prediction_viz overrides (fall back to mpc-level values) ─────────────
    pv            = config['mpc'].get('prediction_viz', {})
    n_mpc         = pv.get('n_mpc',         config['mpc']['n_mpc'])
    n_look_ahead  = pv.get('n_look_ahead',  config['mpc']['n_look_ahead'])
    n_sample      = pv.get('n_sample',      config['mpc']['n_sample'])
    n_update_iter = pv.get('n_update_iter', config['mpc']['n_update_iter'])
    gd_loop       = pv.get('gd_loop',       config['mpc']['gd_loop'])
    mpc_type      = pv.get('mpc_type',      config['mpc']['mpc_type'])
    time_lim      = pv.get('time_lim',      config['mpc'].get('time_lim', float('inf')))
    output_base   = pv.get('output_dir',    'outputs/prediction_viz_eulerian')
    video_fps     = pv.get('video_fps',     2)

    # ── build environment + goal (mirrors visualize_mpc.py) ──────────────────
    env = GenesisEnv(config)
    screenH = env.screenHeight
    screenW = env.screenWidth

    task_type = config['mpc']['task']['type']
    if task_type == 'target_control':
        subgoal, goal_mask = gen_subgoal(config['mpc']['task']['goal_row'],
                                         config['mpc']['task']['goal_col'],
                                         config['mpc']['task']['goal_r'],
                                         h=screenH, w=screenW)
        goal_img_rgb = np.stack([goal_mask * 255] * 3, axis=-1).astype(np.uint8)
    elif task_type == 'target_shape':
        subgoal, goal_img_rgb = gen_goal_shape(config['mpc']['task']['target_char'],
                                               h=screenH, w=screenW)
    else:
        raise NotImplementedError(f"Unknown task type: {task_type}")

    env.reset()

    # ── build Eulerian model (mirrors visualize_mpc.py) ──────────────────────
    cam_extrinsic = env.get_cam_extrinsics()
    global_scale  = config['dataset']['global_scale']

    # Allow the YAML to select the model type (overrides the module constant).
    model_cfg = config['mpc'].get('model', {})
    eulerian_model_type = model_cfg.get('type', EULERIAN_MODEL_TYPE)

    if eulerian_model_type.lower() == 'unetfilm':
        from GranularDynamics2.myClasses.UNetModels_conditioned import UNetFiLM

        weights_path = model_cfg.get('weights_path', '')
        if not weights_path:
            raise ValueError(
                "config.mpc.model.weights_path is required when model.type = 'UNetFiLM'"
            )

        # Physics from dataset config
        physics_vec = torch.tensor([
            config['dataset'].get('particle_friction', 1.0),
            config['dataset'].get('particle_density', 1000.0),
            config['dataset'].get('box_friction', 0.5),
        ], dtype=torch.float32)

        # Grid resolution: 1 px/mm, matching the training dataset
        box_full_m  = config['dataset']['wkspc_w'] * 2
        grid_n      = round(box_full_m * 1000)
        grid_res    = (grid_n, grid_n)

        plate_L_m  = config['dataset'].get('plate_length', 0.04)
        plate_W_m  = config['dataset'].get('plate_width',  0.002)
        px_per_m   = grid_n / box_full_m

        physics_dim = model_cfg.get('physics_dim', 3)
        unet = UNetFiLM(in_channels=2, physics_dim=physics_dim, out_channels=1)
        print(f"Loading UNetFiLM weights from {weights_path}")
        unet.load_state_dict(torch.load(weights_path, map_location='cpu'))
        unet.eval()

        push_model = UNetFiLMPushModel(
            unet_film=unet,
            physics=physics_vec,
            grid_size=grid_res,
            plate_length_px=plate_L_m * px_per_m,
            plate_width_px=plate_W_m * px_per_m,
        )
        bounds = UNetFiLMPushModel.default_bounds(config)

    else:
        bounds   = EulerianModelWrapper.default_bounds(config, convention='genesis')
        grid_res = GRID_RES

        if eulerian_model_type == 'splat':
            push_model = SplatPushModel(
                width=TOOL_WIDTH, sigma=ACTION_SIGMA,
                redistribute=SPLAT_REDISTRIBUTE,
                redistribute_iters=SPLAT_REDISTRIBUTE_ITERS,
            )
        elif eulerian_model_type == 'fluid':
            push_model = FluidPushModel(
                width=FLUID_WIDTH, sigma=FLUID_SIGMA,
                n_steps=FLUID_N_STEPS, decay=FLUID_DECAY,
                media_sharpness=FLUID_MEDIA_SHARPNESS,
                blur_sigma=FLUID_BLUR_SIGMA,
                correct_divergence=FLUID_CORRECT_DIVERGENCE,
                redistribute=FLUID_REDISTRIBUTE,
                redistribute_iters=FLUID_REDISTRIBUTE_ITERS,
            )
        elif eulerian_model_type == 'spread':
            push_model = SpreadPushModel(width=TOOL_WIDTH, sigma=ACTION_SIGMA)
        else:
            raise ValueError(f"Unknown model type: {eulerian_model_type!r}")

    model_dy = EulerianModelWrapper(
        push_model, bounds, grid_res, cam_extrinsic, global_scale,
        action_convention='genesis',
    ).cuda()
    print(f"Eulerian model: {eulerian_model_type}, grid {grid_res}")

    # ── pre-compute goal score map (diagnostic: verify goal is correctly captured) ──
    cam_params = env.get_cam_params()

    # Scale goal shape so its pixel footprint matches the actual material area.
    obs_init = env.render()
    subgoal  = scale_subgoal_to_material_pixels(
        subgoal, obs_init[..., -1], global_scale)

    score_tensor = model_dy.prepare_goal_reward(subgoal, cam_params, device='cpu')
    score_np     = score_tensor.numpy()   # (Nx, Ny)
    print(f"Score map: min={score_np.min():.3f}  max={score_np.max():.3f}  "
          f"nonzero={np.count_nonzero(score_np > 0)}/{score_np.size}")

    # ── output directory ──────────────────────────────────────────────────────
    timestamp = get_current_YYYY_MM_DD_hh_mm_ss_ms()
    run_dir = os.path.join(output_base, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving to: {run_dir}\n")

    if not USE_SIMPLE_MPC:
        funnel_dist       = np.zeros_like(subgoal)
        action_seq_init   = np.load(f'init_action/init_action_{n_sample}.npy')[np.newaxis, ...]
        action_label_init = np.zeros(1)

    # ── save score map visualisations once (goal alignment check) ──────────────
    _save_score_map_panel(
        score_np, subgoal, goal_img_rgb, bounds, grid_res,
        cam_params, cam_extrinsic, global_scale, run_dir)

    # ── run MPC ───────────────────────────────────────────────────────────────
    print(f"Running MPC: {n_mpc} steps, {n_sample} samples, "
          f"{n_update_iter} update iters, model={eulerian_model_type}\n")
    if USE_SIMPLE_MPC:
        from simple_mpc import run_simple_mpc
        subg_output = run_simple_mpc(env, model_dy, subgoal, config)
    else:
        subg_output = env.step_subgoal_ptcl(
            subgoal,
            model_dy,
            None,
            n_mpc=n_mpc,
            n_look_ahead=n_look_ahead,
            n_sample=n_sample,
            n_update_iter=n_update_iter,
            mpc_type=mpc_type,
            gd_loop=gd_loop,
            particle_num=-1,
            funnel_dist=funnel_dist,
            action_seq_mpc_init=action_seq_init,
            action_label_seq_mpc_init=action_label_init,
            time_lim=time_lim,
            auto_particle_r=True,
        )


    # ── unpack results ────────────────────────────────────────────────────────
    rewards     = subg_output['rewards']      # (n_mpc+1,)
    raw_obs     = subg_output['raw_obs']      # (n_mpc+1, H, W, 5)
    states      = subg_output['states']       # list[n_mpc+1], each (particle_num, 3)
    states_pred = subg_output['states_pred']  # list[n_mpc],   each (n_look_ahead, Nx, Ny)
    actions     = subg_output['actions']      # (n_mpc, 4)
    rew_means   = subg_output['rew_means']    # (n_mpc, 1, n_update_iter*gd_loop)

    # ── build per-step panels ─────────────────────────────────────────────────
    panels = []
    for i in range(len(states_pred)):
        img_cur_rgb  = raw_obs[i][..., :3]
        img_next_rgb = raw_obs[i + 1][..., :3]

        # -- input occupancy: convert current particles → occ ------------------
        pts_cur = torch.from_numpy(states[i]).float().cuda().unsqueeze(0)  # (1, N, 3)
        with torch.no_grad():
            occ_input = model_dy.initial_occ_from_particles(pts_cur)       # (1, Nx, Ny)
        occ_input_np = occ_input[0].cpu().numpy()                          # (Nx, Ny)

        # -- predicted occupancy (Eulerian model output) -----------------------
        # states_pred[i] has shape (n_look_ahead, Nx, Ny); take first look-ahead
        occ_pred_np = states_pred[i][0]        # (Nx, Ny)

        # -- actual next occupancy: convert next particles → occ ---------------
        pts_next = torch.from_numpy(states[i + 1]).float().cuda().unsqueeze(0)
        with torch.no_grad():
            occ_actual = model_dy.initial_occ_from_particles(pts_next)     # (1, Nx, Ny)
        occ_actual_np = occ_actual[0].cpu().numpy()                        # (Nx, Ny)

        action_i = actions[i]   # (4,)

        # ── diagnostic prints ─────────────────────────────────────────────
        s_pix, e_pix = _action_world_to_pixels(
            action_i, cam_extrinsic, global_scale, cam_params)
        occ_err = float(np.abs(occ_pred_np - occ_actual_np).mean())
        rew_i  = rew_means[i, 0, :]         # (n_iters,)
        valid  = rew_i[rew_i != 0]
        rew_rng = f"{valid.min():.3f}→{valid.max():.3f}" if len(valid) > 1 else "n/a"
        print(f"  action: [{action_i[0]:.3f},{action_i[1]:.3f}→"
              f"{action_i[2]:.3f},{action_i[3]:.3f}]  "
              f"pixel: ({int(s_pix[0])},{int(s_pix[1])})→({int(e_pix[0])},{int(e_pix[1])})")
        print(f"  mean |occ_pred - occ_actual|: {occ_err:.4f}")
        print(f"  reward within step: {rew_rng}  "
              f"(std={float(rew_means[i,0,:].std()):.4f})")

        goal_for_panel = goal_img_rgb if i == 0 else None
        panel = make_eulerian_panel(
            img_cur_rgb,
            occ_input_np,
            img_next_rgb,
            occ_pred_np,
            occ_actual_np,
            bounds,
            grid_res,
            cam_params,
            step=i + 1,
            reward_cur=float(rewards[i]),
            reward_next=float(rewards[i + 1]),
            action=action_i,
            cam_extrinsic=cam_extrinsic,
            global_scale=global_scale,
            score_np=score_np,
            goal_img_rgb=goal_for_panel,
        )

        out_path = os.path.join(run_dir, f'step_{i + 1:03d}.png')
        cv2.imwrite(out_path, panel)
        print(f"  step {i + 1:3d}  r_cur={rewards[i]:.4f}  r_next={rewards[i+1]:.4f}"
              f"  → {out_path}")
        panels.append(panel)

    # ── save video ────────────────────────────────────────────────────────────
    if panels:
        # Ensure all frames are the same size (first step panel is wider due to goal tile)
        max_w = max(p.shape[1] for p in panels)
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            ph, pw = p.shape[:2]
            if ph < max_h or pw < max_w:
                canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                canvas[:ph, :pw] = p
                padded.append(canvas)
            else:
                padded.append(p)

        vid_path = os.path.join(run_dir, 'prediction_comparison.avi')
        fps = max(1, video_fps)
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps, (max_w, max_h))
        hold_frames = fps * 2        # 2 seconds per transition
        final_frames = fps * 5       # 5 seconds on last frame
        for idx, f in enumerate(padded):
            n = final_frames if idx == len(padded) - 1 else hold_frames
            for _ in range(n):
                writer.write(f)
        writer.release()
        print(f"\nSaved comparison video → {vid_path}")

    # ── reward plot (across MPC steps) ───────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, marker='o')
    plt.xlabel('MPC step')
    plt.ylabel('Reward')
    plt.title(f'Reward over MPC steps ({EULERIAN_MODEL_TYPE} model)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'rewards.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved reward plot → {plot_path}")

    # ── per-step optimizer convergence plot ───────────────────────────────────
    # Shows rew_mean across Adam iterations for each MPC step.
    # If curves are flat the gradient is zero (model output not differentiable
    # w.r.t. action) or all samples are stuck at the same point.
    # If curves rise but actions don't help, the score map is misaligned.
    n_steps_ran = rew_means.shape[0]
    n_cols = min(n_steps_ran, 5)
    n_rows = int(np.ceil(n_steps_ran / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for i in range(n_steps_ran):
        ax = axes[i // n_cols][i % n_cols]
        curve = rew_means[i, 0, :]   # (n_iters,)
        ax.plot(curve)
        ax.set_title(f"Step {i+1}  Δr={curve[-1]-curve[0]:.3f}", fontsize=9)
        ax.set_xlabel('iter', fontsize=8)
        ax.set_ylabel('mean reward', fontsize=8)
        ax.grid(True, alpha=0.3)
    for i in range(n_steps_ran, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)
    fig.suptitle(f'Optimizer convergence per MPC step ({EULERIAN_MODEL_TYPE})',
                 fontsize=11)
    fig.tight_layout()
    conv_path = os.path.join(run_dir, 'optimizer_convergence.png')
    fig.savefig(conv_path, dpi=150)
    plt.close(fig)
    print(f"Saved optimizer convergence plot → {conv_path}")

    print(f"\nDone. All outputs in: {run_dir}")


def _save_score_map_panel(
    score_np:     np.ndarray,
    subgoal:      np.ndarray,
    goal_img_rgb: np.ndarray,
    grid_bounds:  Dict[str, float],
    grid_res:     Tuple[int, int],
    cam_params,
    cam_extrinsic: np.ndarray,
    global_scale:  float,
    run_dir:      str,
) -> None:
    """
    Save a stand-alone 'score_map.png' showing the goal reward landscape
    both as a raw heatmap and projected onto a blank canvas.
    """
    H, W = 720, 720
    HMAP_H, HMAP_W = H // 2, W // 2

    # raw heatmap
    hmap = _score_to_heatmap_bgr(score_np, (HMAP_H, HMAP_W))
    hmap = _label(hmap, "Goal score (grid)")

    # projected overlay on blank background
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    projected = _draw_score_overlay(blank, score_np, grid_bounds, grid_res,
                                     cam_params, radius=4, alpha=0.9)
    projected = _label(projected, "Goal score (projected)")

    # goal image
    goal_bgr = cv2.cvtColor(goal_img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    goal_bgr = cv2.resize(goal_bgr, (W, H))
    goal_bgr = _label(goal_bgr, "Goal image")

    # goal subgoal heatmap (raw distance-transform before reward scaling)
    sub_u8 = np.clip(subgoal / (subgoal.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    sub_colored = cv2.applyColorMap(sub_u8, cv2.COLORMAP_VIRIDIS)
    sub_colored = cv2.resize(sub_colored, (W // 2, H // 2))
    sub_colored = _label(sub_colored, "Raw subgoal dist-transform")

    top_row  = np.hstack([goal_bgr, projected])
    bot_left = np.hstack([hmap, sub_colored])
    bot_row  = np.hstack([bot_left,
                           np.zeros((HMAP_H, top_row.shape[1] - bot_left.shape[1], 3),
                                    dtype=np.uint8)])
    panel = np.vstack([top_row, bot_row])

    path = os.path.join(run_dir, 'score_map.png')
    cv2.imwrite(path, panel)
    print(f"Saved score map → {path}")


if __name__ == "__main__":
    main()
