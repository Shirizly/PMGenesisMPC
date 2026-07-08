#!/usr/bin/env python3
from __future__ import annotations

"""
Interactive MPC debugger GUI.

Layout
------
  Row 0 – five heatmap tiles (static image labels):
    Current occ | Score map | Predicted occ + arrow | Pred×Score | Δocc

  Row 1 – draggable workspace canvas:
    Top-down view of the workspace.  Drag the ORANGE circle to set the push
    start (sx, sy) and the CYAN circle to set the end (ex, ey).  Releasing
    the mouse auto-evaluates.

  Row 2 – action text fields  [sx] [sy] [ex] [ey]  and  [lr] (editable)

  Row 3 – reward / gradient info bar

  Row 4 – status bar

  Row 5 – buttons: Evaluate | GD Step | Reset GD | Submit | Reset Env

Notes
-----
  * The environment is launched in headless mode (off-screen rendering) so
    PyFlex does not try to open its own OpenGL window, which caused crashes
    when running alongside Tkinter.

  * Dragging on the canvas updates the text fields live and triggers an
    automatic Evaluate on mouse release.

  * The LR field applies immediately on the next GD Step press; changing it
    rebuilds the Adam optimizer (resets momentum) so the new rate takes
    effect cleanly.

Usage
-----
    python debug_mpc_gui.py
"""

import tkinter as tk
import numpy as np
import torch
import torch.optim as optim
import cv2
from PIL import Image, ImageTk

from utils import load_yaml, gen_goal_shape, gen_subgoal
from env.genesis_env import GenesisEnv
from model.eulerian_wrapper import _action_to_cam_3d_genesis
from simple_mpc import load_simple_config
from run_experiments import load_model

# ── Configuration ────────────────────────────────────────────────────────────
SIMPLE_MPC_CONFIG = 'simple_mpc/config/config_simple.yaml'

# ── Model selection ───────────────────────────────────────────────────────────
# Set MODEL_TYPE to one of:
#   'eulerian'         — heuristic Eulerian push model (no learned weights)
#   'unetfilm'         — learned NFDUNetFiLM model
#   'unetfilm-shallow' — lightweight NFDUNetFiLM variant
MODEL_TYPE = 'unetfilm'

# For MODEL_TYPE = 'eulerian': one of 'splat', 'spread', 'fluid'
EULERIAN_HEURISTIC    = 'spread'
TOOL_WIDTH            = 8.0
ACTION_SIGMA          = 1.5

# For MODEL_TYPE = 'unetfilm' or 'unetfilm-shallow': path to a model_card.yaml
# written by the Trainer (run_experiments.load_model loads via model cards).
UNETFILM_MODEL_CARD = 'runs_cubes/unetfilm_corl_limited_100e/model_card.yaml'

TILE_SIZE    = 300   # px – size of each heatmap tile
CANVAS_SIZE  = 380   # px – draggable workspace canvas
FONT_MONO    = ('Courier', 13)
FONT_MONO_B  = ('Courier', 14, 'bold')
FONT_LABEL   = ('Courier', 12)
FONT_HDR     = ('Courier', 11)
# ─────────────────────────────────────────────────────────────────────────────


# ── Image helpers ─────────────────────────────────────────────────────────────

def _heatmap_bgr(arr: np.ndarray, size: int,
                 colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    t = arr.T
    lo, hi = t.min(), t.max()
    u8 = ((t - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return cv2.resize(cv2.applyColorMap(u8, colormap),
                      (size, size), interpolation=cv2.INTER_NEAREST)


def _diff_heatmap_bgr(diff: np.ndarray, size: int) -> np.ndarray:
    gain = cv2.resize((np.clip( diff, 0, 1).T * 255).astype(np.uint8),
                      (size, size), interpolation=cv2.INTER_NEAREST)
    loss = cv2.resize((np.clip(-diff, 0, 1).T * 255).astype(np.uint8),
                      (size, size), interpolation=cv2.INTER_NEAREST)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 1] = gain
    img[:, :, 2] = loss
    return img


def _stamp(img: np.ndarray, lines) -> np.ndarray:
    out = img.copy()
    for k, text in enumerate(lines):
        y = 26 + k * 22
        cv2.putText(out, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0,   0,   0), 1, cv2.LINE_AA)
    return out


def _draw_arrow_on_tile(img: np.ndarray,
                        sg_xy: np.ndarray, eg_xy: np.ndarray,
                        size: int, grid_res: tuple) -> np.ndarray:
    Nx = grid_res[0]
    sc = (size - 1) / max(Nx - 1, 1)
    sp = (int(np.clip(sg_xy[0] * sc, 0, size - 1)),
          int(np.clip(sg_xy[1] * sc, 0, size - 1)))
    ep = (int(np.clip(eg_xy[0] * sc, 0, size - 1)),
          int(np.clip(eg_xy[1] * sc, 0, size - 1)))
    out = img.copy()
    cv2.arrowedLine(out, sp, ep, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(out, sp, 6, (0, 140, 255), -1, cv2.LINE_AA)
    return out


def _bgr_to_photo(arr_bgr: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb).resize((w, h), Image.NEAREST))


# ── GUI ───────────────────────────────────────────────────────────────────────

class MPCDebugGUI:
    # Radius of draggable handle circles on the workspace canvas (pixels)
    _HANDLE_R = 10

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MPC Debug GUI")
        self.root.configure(bg='#1a1a1a')

        self._status_var = tk.StringVar(value="Starting up...")
        root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._status("Launching GenesisEnv (headless)...")
        root.update()

        # ── Config ───────────────────────────────────────────────────────────
        self.cfg     = load_simple_config(SIMPLE_MPC_CONFIG)
        self.wkspc_w = float(self.cfg['dataset']['wkspc_w'])
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ── Environment ───────────────────────────────────────────────────────
        self.env          = GenesisEnv(self.cfg)
        self.env.reset()
        self.cam_params   = self.env.get_cam_params()
        self.global_scale = float(self.cfg['dataset']['global_scale'])

        # ── Dynamics model ────────────────────────────────────────────────────
        self._status("Loading dynamics model...")
        root.update()
        if MODEL_TYPE == 'eulerian':
            model_spec = {
                'type': 'eulerian',
                'need_weights': False,
                'heuristic_type': EULERIAN_HEURISTIC,
                'grid_n': 64,
            }
        elif MODEL_TYPE in ('unetfilm', 'unetfilm-shallow'):
            if not UNETFILM_MODEL_CARD:
                raise ValueError(
                    "UNETFILM_MODEL_CARD must be set at the top of debug_mpc_gui.py "
                    "when MODEL_TYPE is 'unetfilm' or 'unetfilm-shallow'."
                )
            model_spec = {
                'type': MODEL_TYPE,
                'need_weights': True,
                'model_card': UNETFILM_MODEL_CARD,
            }
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE!r}")

        self.model_dy = load_model(model_spec, self.cfg, self.env)
        self.grid_res = self.model_dy.grid_res

        # ── Initial render (needed before goal scaling) ───────────────────────
        # We need occ_init to measure the material footprint so we can scale
        # the goal shape to match it.  score_tensor is not yet set here so we
        # compute occ_init inline without calling _refresh_env_state().
        self._status("Rendering initial state for goal scaling...")
        root.update()
        self._act_tensor   = None
        self._optimizer    = None
        self._occ_pred_np  = None
        self._r_pred       = None
        obs = self.env.render()
        with torch.no_grad():
            self.occ_init = self.model_dy.obs_to_occupancy(
                obs, self.cam_params).to(self.device).detach()

        # ── Goal + score map ──────────────────────────────────────────────────
        self._status("Building goal score map (auto-scaling to material area)...")
        root.update()
        task = self.cfg['mpc']['task']
        if task['type'] == 'target_shape':
            raw_subgoal, _ = gen_goal_shape(task['target_char'], h=720, w=720)
        else:
            raw_subgoal, _ = gen_subgoal(task['goal_row'], task['goal_col'],
                                         task['goal_r'], h=720, w=720)

        self.subgoal = self._scale_subgoal_to_material(raw_subgoal)

        empty_penalty = float(self.cfg['mpc'].get('reward', {}).get('empty_penalty', 0.0))
        self.score_tensor = self.model_dy.prepare_goal_reward(
            self.subgoal, self.cam_params,
            device=self.device, empty_penalty=empty_penalty)
        self.score_np = self.score_tensor.cpu().numpy()

        # current_reward can now be computed (score_tensor is ready)
        self.current_reward = float(
            (self.occ_init[0] * self.score_tensor).sum().item())


        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()
        self._update_canvas_handles()
        self._redraw(self._get_action(), occ_pred_np=None, r_pred=None)
        self._status("Ready - drag handles or enter action, then press Evaluate.")

    # ─────────────────────────── env helpers ─────────────────────────────────

    def _refresh_env_state(self):
        obs = self.env.render()
        with torch.no_grad():
            self.occ_init = self.model_dy.obs_to_occupancy(
                obs, self.cam_params).to(self.device).detach()
            self.current_reward = float(
                (self.occ_init[0] * self.score_tensor).sum().item())
        self._clear_gd_state()

    def _scale_subgoal_to_material(self, subgoal_dist: np.ndarray) -> np.ndarray:
        """Scale the goal shape in pixel space so its projected voxel footprint
        on the occupancy grid roughly matches the material's occupied area.

        Works by:
          1. Back-projecting the raw goal pixels onto the 64x64 grid to count
             goal voxels.
          2. Counting occupied voxels in occ_init (the actual material).
          3. Computing scale = sqrt(n_material / n_goal) — since area ∝ scale².
          4. Warping the binary goal mask around the image centre by that scale
             and recomputing the distance transform.
        """
        H, W = subgoal_dist.shape

        # Quick area estimate of the unscaled goal on the grid
        occ_goal_raw = self.model_dy.subgoal_mask_to_occupancy(
            subgoal_dist, self.cam_params)          # (1, *grid_res)
        n_goal = float((occ_goal_raw[0] > 0.1).sum().item())
        n_mat  = float((self.occ_init[0]  > 0.1).sum().item())

        if n_goal < 1 or n_mat < 1:
            print('[goal scale] cannot estimate areas; skipping scaling.')
            return subgoal_dist

        scale = (n_mat / n_goal) ** 0.5
        self.goal_area_scale = scale          # expose for display in UI
        print(f'[goal scale] goal_vox={n_goal:.0f}  mat_vox={n_mat:.0f}  '
              f'scale={scale:.3f}')

        if abs(scale - 1.0) < 0.02:          # already close enough
            return subgoal_dist

        # Warp binary mask around image centre
        goal_mask = (subgoal_dist < 0.5).astype(np.uint8)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), 0.0, float(scale))
        scaled_mask = cv2.warpAffine(goal_mask, M, (W, H),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        scaled_mask = (scaled_mask > 0.5).astype(np.uint8)

        scaled_dist = np.minimum(
            cv2.distanceTransform(1 - scaled_mask, cv2.DIST_L2, 5), 1e4)
        return scaled_dist.astype(np.float32)


    def _clear_gd_state(self):
        self._act_tensor = None
        self._optimizer  = None

    # ─────────────────────────── prediction ──────────────────────────────────

    def _predict(self, act_np: np.ndarray):
        act_t = torch.tensor(act_np[None], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            occ_pred = self.model_dy.predict_one_step_occ(
                self.occ_init.clone(), act_t)
        occ_pred_np = occ_pred[0].cpu().numpy()
        r_pred = float((occ_pred[0].clamp(0, 1) * self.score_tensor).sum().item())
        return occ_pred_np, r_pred

    # ─────────────────────────── coordinate helpers ───────────────────────────

    def _world_to_canvas(self, wx: float, wy: float) -> tuple:
        """World-2D [-w, w]^2 -> canvas pixel (cx, cy)."""
        C = CANVAS_SIZE
        w = self.wkspc_w
        cx = int((wx + w) / (2 * w) * (C - 1))
        cy = int((C - 1) - (wy + w) / (2 * w) * (C - 1))  # flip y: world-up = canvas-up
        return cx, cy

    def _canvas_to_world(self, cx: int, cy: int) -> tuple:
        """Canvas pixel -> world-2D [-w, w]^2."""
        C = CANVAS_SIZE
        w = self.wkspc_w
        wx = cx / (C - 1) * (2 * w) - w
        wy = ((C - 1) - cy) / (C - 1) * (2 * w) - w  # flip y
        return float(wx), float(wy)

    def _action_to_grid_coords(self, act_np: np.ndarray):
        """Return start/end in grid-index coords for arrow drawing."""
        act_t = torch.tensor(act_np[None], dtype=torch.float32)
        s_cam, e_cam = _action_to_cam_3d_genesis(act_t, self.global_scale)
        sg = self.model_dy._cam3d_to_grid(s_cam.to(self.device)).cpu().numpy()[0]
        eg = self.model_dy._cam3d_to_grid(e_cam.to(self.device)).cpu().numpy()[0]
        return sg[:2], eg[:2]

    # ─────────────────────────── tile rendering ───────────────────────────────

    def _build_tiles(self, act_np: np.ndarray,
                     occ_pred_np: np.ndarray | None) -> list:
        T           = TILE_SIZE
        occ_init_np = self.occ_init[0].cpu().numpy()
        sg, eg      = self._action_to_grid_coords(act_np)

        tile0 = _stamp(_heatmap_bgr(occ_init_np, T),
                       [f"Current occ",
                        f"r={self.current_reward:.5f}"])

        tile1 = _stamp(_heatmap_bgr(self.score_np, T, cv2.COLORMAP_HOT),
                       [f"Score map",
                        f"[{self.score_np.min():.2f}, {self.score_np.max():.2f}]"])

        blank = np.zeros((T, T, 3), dtype=np.uint8)
        if occ_pred_np is None:
            tile2 = _stamp(blank, ["Predicted occ", "(press Evaluate)"])
            tile3 = _stamp(blank, ["Pred x Score"])
            tile4 = _stamp(blank, ["Occ change"])
        else:
            tile2 = _draw_arrow_on_tile(
                _stamp(_heatmap_bgr(occ_pred_np, T), ["Predicted occ"]),
                sg, eg, T, self.grid_res)
            tile3 = _stamp(
                _heatmap_bgr(np.clip(occ_pred_np, 0, 1) * self.score_np,
                             T, cv2.COLORMAP_HOT),
                ["Pred x Score (clamped)"])
            tile4 = _stamp(
                _diff_heatmap_bgr(occ_pred_np - occ_init_np, T),
                ["Docc: green=gain red=loss"])

        return [tile0, tile1, tile2, tile3, tile4]

    # ─────────────────────────── UI builder ───────────────────────────────────

    def _build_ui(self):
        T = TILE_SIZE
        C = CANVAS_SIZE
        BG  = '#1a1a1a'
        BG2 = '#222222'
        BG3 = '#2c2c2c'

        # ── Row 0: heatmap tiles ──────────────────────────────────────────────
        tile_frame = tk.Frame(self.root, bg=BG)
        tile_frame.pack(padx=10, pady=(10, 4))

        col_labels = [
            "Current occupancy",
            "Goal score map",
            "Predicted occ\n+ push arrow",
            "Pred x Score\n(reward density)",
            "Delta occ\ngreen=gained  red=lost",
        ]
        self._img_widgets = []
        self._photo_refs  = [None] * len(col_labels)
        for col, lbl in enumerate(col_labels):
            f = tk.Frame(tile_frame, bg=BG)
            f.grid(row=0, column=col, padx=5)
            tk.Label(f, text=lbl, bg=BG, fg='#aaaaaa',
                     font=FONT_HDR, justify='center').pack()
            w = tk.Label(f, bg=BG2, width=T, height=T)
            w.pack()
            self._img_widgets.append(w)

        # ── Row 1: draggable workspace canvas ─────────────────────────────────
        row1 = tk.Frame(self.root, bg=BG)
        row1.pack(padx=10, pady=4, fill=tk.X)

        canvas_frame = tk.LabelFrame(
            row1,
            text="  Workspace  (drag ORANGE=start, CYAN=end -- release auto-evaluates)  ",
            fg='#888888', bg=BG2, font=FONT_HDR)
        canvas_frame.pack(side=tk.LEFT, padx=(0, 10))

        self._canvas = tk.Canvas(canvas_frame, width=C, height=C,
                                 bg='#111111', highlightthickness=0)
        self._canvas.pack(padx=4, pady=4)

        # Draw workspace border
        self._canvas.create_rectangle(0, 0, C - 1, C - 1,
                                       outline='#444444', width=1)
        # Centre cross-hair
        self._canvas.create_line(C // 2, 0, C // 2, C,
                                  fill='#333333', dash=(3, 6))
        self._canvas.create_line(0, C // 2, C, C // 2,
                                  fill='#333333', dash=(3, 6))

        # Axis labels
        wkspc = self.wkspc_w
        for v in [-wkspc, 0, wkspc]:
            cx, _ = self._world_to_canvas(v, 0)
            _, cy = self._world_to_canvas(0, v)
            self._canvas.create_text(cx, C - 10, text=f"{v:.1f}",
                                      fill='#555555', font=('Courier', 8))
            self._canvas.create_text(8, cy, text=f"{v:.1f}",
                                      fill='#555555', font=('Courier', 8))

        # Action arrow and handles (updated dynamically)
        self._cv_line  = self._canvas.create_line(
            0, 0, 1, 1, fill='#88ff88', width=2,
            arrow=tk.LAST, arrowshape=(12, 14, 5))
        self._cv_s_dot = self._canvas.create_oval(
            0, 0, 1, 1, fill='#ff8800', outline='white', width=2)
        self._cv_e_dot = self._canvas.create_oval(
            0, 0, 1, 1, fill='#00ddff', outline='white', width=2)
        self._cv_s_lbl = self._canvas.create_text(
            0, 0, text='S', fill='white', font=('Courier', 10, 'bold'))
        self._cv_e_lbl = self._canvas.create_text(
            0, 0, text='E', fill='white', font=('Courier', 10, 'bold'))

        self._dragging = None   # 's' or 'e' or None
        self._canvas.tag_bind(self._cv_s_dot, '<ButtonPress-1>',
                              lambda e: self._drag_start(e, 's'))
        self._canvas.tag_bind(self._cv_s_lbl, '<ButtonPress-1>',
                              lambda e: self._drag_start(e, 's'))
        self._canvas.tag_bind(self._cv_e_dot, '<ButtonPress-1>',
                              lambda e: self._drag_start(e, 'e'))
        self._canvas.tag_bind(self._cv_e_lbl, '<ButtonPress-1>',
                              lambda e: self._drag_start(e, 'e'))
        self._canvas.bind('<B1-Motion>',       self._drag_move)
        self._canvas.bind('<ButtonRelease-1>', self._drag_release)

        # ── Legend / hint panel (to the right of the canvas) ─────────────────
        hint_frame = tk.Frame(row1, bg=BG)
        hint_frame.pack(side=tk.LEFT, padx=4, anchor='n')
        hints = [
            ("● ORANGE -- push start (sx, sy)", '#ff8800'),
            ("● CYAN   -- push end   (ex, ey)", '#00ccdd'),
            ("",                                 '#888888'),
            ("Drag handles to set action.",      '#888888'),
            ("Release -> auto-evaluates.",       '#888888'),
            ("",                                 '#888888'),
            ("Workspace bounds:",                '#888888'),
            (f"  x, y in [{-wkspc:.1f}, {wkspc:.1f}]", '#888888'),
        ]
        for text, color in hints:
            tk.Label(hint_frame, text=text, bg=BG, fg=color,
                     font=FONT_MONO, anchor='w').pack(anchor='w')

        # ── Row 2: action fields + editable LR ───────────────────────────────
        act_frame = tk.LabelFrame(
            self.root,
            text="  Action  [sx sy ex ey]  world-2D  and  learning rate  ",
            fg='#888888', bg=BG2, font=FONT_HDR)
        act_frame.pack(fill=tk.X, padx=10, pady=4)

        half_w = self.wkspc_w * 0.4
        self._vars = {}
        defaults   = {'sx': -half_w, 'sy': -half_w,
                      'ex':  half_w, 'ey':  half_w}
        for col, key in enumerate(['sx', 'sy', 'ex', 'ey']):
            self._vars[key] = tk.StringVar(value=f"{defaults[key]:.4f}")
            tk.Label(act_frame, text=f"  {key}:", bg=BG2, fg='#cccccc',
                     font=FONT_MONO_B).grid(row=0, column=col * 2, sticky='e')
            e = tk.Entry(act_frame, textvariable=self._vars[key], width=10,
                         bg='#333333', fg='#ffffff', insertbackground='white',
                         font=FONT_MONO)
            e.grid(row=0, column=col * 2 + 1, padx=4, pady=8)
            e.bind('<Return>',    lambda _: self.on_evaluate())
            e.bind('<KeyRelease>', lambda _: self._update_canvas_handles())

        # Editable LR field
        self._var_lr = tk.StringVar(
            value=f"{self.cfg['mpc']['gd']['lr']:.5f}")
        tk.Label(act_frame, text="  lr:", bg=BG2, fg='#aaaaaa',
                 font=FONT_MONO_B).grid(row=0, column=8, sticky='e')
        lr_entry = tk.Entry(act_frame, textvariable=self._var_lr, width=9,
                            bg='#333344', fg='#aaddff', insertbackground='white',
                            font=FONT_MONO)
        lr_entry.grid(row=0, column=9, padx=4, pady=8)
        tk.Label(act_frame,
                 text="(applies on next GD step; resets momentum)",
                 bg=BG2, fg='#555555', font=('Courier', 10)).grid(
                     row=0, column=10, padx=8)

        # ── Row 3: info bar ───────────────────────────────────────────────────
        info_frame = tk.Frame(self.root, bg=BG)
        info_frame.pack(fill=tk.X, padx=10, pady=2)
        scale_str = f"{getattr(self, 'goal_area_scale', 1.0):.3f}"
        self._var_r_cur  = tk.StringVar(value="Current r:     --")
        self._var_r_pred = tk.StringVar(value="Predicted r:   --")
        self._var_gain   = tk.StringVar(value="Gain:          --")
        self._var_grad   = tk.StringVar(value="Grad norm:     --")
        self._var_scale  = tk.StringVar(value=f"Goal scale:    {scale_str}")
        for v in (self._var_r_cur, self._var_r_pred,
                  self._var_gain, self._var_grad, self._var_scale):
            tk.Label(info_frame, textvariable=v, bg=BG, fg='#eeeeee',
                     font=FONT_MONO, width=24, anchor='w').pack(
                         side=tk.LEFT, padx=10)

        # ── Row 4: status bar ─────────────────────────────────────────────────
        tk.Label(self.root, textvariable=self._status_var,
                 bg=BG, fg='#88ff88', font=FONT_MONO, anchor='w').pack(
                     fill=tk.X, padx=12, pady=2)

        # ── Row 5: buttons ────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=10)
        btn_cfg = dict(font=FONT_MONO_B, width=15, pady=6)
        buttons = [
            ("Evaluate",      self.on_evaluate,  '#1a4a7a'),
            ("GD Step",       self.on_gd_step,   '#4a3a7a'),
            ("Reset GD",      self.on_reset_gd,  '#3a3a3a'),
            ("Submit Action", self.on_submit,     '#2a6a1a'),
            ("Reset Env",     self.on_reset_env,  '#7a1a1a'),
        ]
        for label, cmd, bg in buttons:
            tk.Button(btn_frame, text=label, command=cmd,
                      bg=bg, fg='white', **btn_cfg).pack(side=tk.LEFT, padx=6)

    # ─────────────────────────── canvas drag ──────────────────────────────────

    def _update_canvas_handles(self):
        """Redraw canvas arrow + handles from current field values."""
        act = self._get_action()
        R   = self._HANDLE_R

        sx_c, sy_c = self._world_to_canvas(act[0], act[1])
        ex_c, ey_c = self._world_to_canvas(act[2], act[3])

        self._canvas.coords(self._cv_line, sx_c, sy_c, ex_c, ey_c)
        self._canvas.coords(self._cv_s_dot,
                            sx_c - R, sy_c - R, sx_c + R, sy_c + R)
        self._canvas.coords(self._cv_e_dot,
                            ex_c - R, ey_c - R, ex_c + R, ey_c + R)
        self._canvas.coords(self._cv_s_lbl, sx_c, sy_c)
        self._canvas.coords(self._cv_e_lbl, ex_c, ey_c)

        # Keep handle items on top so they remain clickable
        self._canvas.tag_raise(self._cv_s_dot)
        self._canvas.tag_raise(self._cv_e_dot)
        self._canvas.tag_raise(self._cv_s_lbl)
        self._canvas.tag_raise(self._cv_e_lbl)

    def _drag_start(self, event: tk.Event, which: str):
        self._dragging = which

    def _drag_move(self, event: tk.Event):
        if self._dragging is None:
            return
        C  = CANVAS_SIZE
        cx = max(0, min(C - 1, event.x))
        cy = max(0, min(C - 1, event.y))
        wx, wy = self._canvas_to_world(cx, cy)
        ww = self.wkspc_w
        if self._dragging == 's':
            self._vars['sx'].set(f"{np.clip(wx, -ww, ww):.4f}")
            self._vars['sy'].set(f"{np.clip(wy, -ww, ww):.4f}")
        else:
            self._vars['ex'].set(f"{np.clip(wx, -ww * 0.85, ww * 0.85):.4f}")
            self._vars['ey'].set(f"{np.clip(wy, -ww * 0.85, ww * 0.85):.4f}")
        self._update_canvas_handles()

    def _drag_release(self, event: tk.Event):
        if self._dragging is not None:
            self._dragging = None
            self.on_evaluate()   # auto-evaluate on mouse release

    # ─────────────────────────── display update ───────────────────────────────

    def _redraw(self, act_np: np.ndarray,
                occ_pred_np: np.ndarray | None,
                r_pred: float | None):
        T     = TILE_SIZE
        tiles = self._build_tiles(act_np, occ_pred_np)
        for i, (widget, tile) in enumerate(zip(self._img_widgets, tiles)):
            photo = _bgr_to_photo(tile, T, T)
            widget.configure(image=photo)
            self._photo_refs[i] = photo

        self._var_r_cur.set(f"Current r:     {self.current_reward:.6f}")
        if r_pred is not None:
            gain = r_pred - self.current_reward
            self._var_r_pred.set(f"Predicted r:   {r_pred:.6f}")
            self._var_gain.set(  f"Gain:          {gain:+.6f}")
        else:
            self._var_r_pred.set("Predicted r:   --")
            self._var_gain.set(  "Gain:          --")

    def _status(self, msg: str):
        self._status_var.set(msg)
        print(msg)
        self.root.update_idletasks()

    # ─────────────────────────── action helpers ───────────────────────────────

    def _get_action(self) -> np.ndarray:
        try:
            return np.array([float(self._vars[k].get())
                             for k in ('sx', 'sy', 'ex', 'ey')],
                            dtype=np.float32)
        except (ValueError, KeyError):
            return np.zeros(4, dtype=np.float32)

    def _set_action(self, act: np.ndarray):
        for key, val in zip(('sx', 'sy', 'ex', 'ey'), act):
            self._vars[key].set(f"{val:.4f}")
        self._update_canvas_handles()

    def _get_lr(self) -> float:
        try:
            return max(1e-6, float(self._var_lr.get()))
        except ValueError:
            return float(self.cfg['mpc']['gd']['lr'])

    # ─────────────────────────── button callbacks ─────────────────────────────

    def on_evaluate(self):
        act_np = self._get_action()
        self._clear_gd_state()
        occ_pred_np, r_pred = self._predict(act_np)
        self._occ_pred_np = occ_pred_np
        self._r_pred      = r_pred
        self._redraw(act_np, occ_pred_np, r_pred)
        self._update_canvas_handles()
        self._var_grad.set("Grad norm:     -- (press GD Step)")
        gain = r_pred - self.current_reward
        self._status(f"Evaluated.  pred_r={r_pred:.6f}  gain={gain:+.6f}")

    def on_gd_step(self):
        act_np = self._get_action()
        lr     = self._get_lr()

        # Rebuild optimizer if LR changed or state is stale
        lr_changed = (self._optimizer is not None and
                      self._optimizer.param_groups[0]['lr'] != lr)
        if self._act_tensor is None or lr_changed:
            self._act_tensor = torch.tensor(
                act_np[None, None, :],   # (1, 1, 4)
                device=self.device, dtype=torch.float32,
                requires_grad=True)
            self._optimizer = optim.Adam(
                [self._act_tensor], lr=lr, betas=(0.9, 0.999))
        else:
            with torch.no_grad():
                self._act_tensor.data[:] = torch.tensor(
                    act_np[None, None, :],
                    device=self.device, dtype=torch.float32)

        # Forward + backward
        occ = self.occ_init.expand(1, *self.grid_res).clone()
        occ_pred = self.model_dy.predict_one_step_occ(occ, self._act_tensor[:, 0, :])
        r = (occ_pred[0].clamp(0, 1) * self.score_tensor).sum()
        self._optimizer.zero_grad()
        (-r).backward()
        grad_norm = float(self._act_tensor.grad.abs().mean().item())
        torch.nn.utils.clip_grad_norm_([self._act_tensor], max_norm=1.0)
        self._optimizer.step()

        # Clip to workspace bounds
        ww = self.wkspc_w
        with torch.no_grad():
            self._act_tensor.data[:, :, 0].clamp_(-ww,        ww)
            self._act_tensor.data[:, :, 1].clamp_(-ww,        ww)
            self._act_tensor.data[:, :, 2].clamp_(-ww * 0.85, ww * 0.85)
            self._act_tensor.data[:, :, 3].clamp_(-ww * 0.85, ww * 0.85)

        new_act_np = self._act_tensor[0, 0].detach().cpu().numpy()
        self._set_action(new_act_np)

        occ_pred_np, r_after = self._predict(new_act_np)
        self._occ_pred_np = occ_pred_np
        self._r_pred      = r_after
        self._redraw(new_act_np, occ_pred_np, r_after)
        self._var_grad.set(f"Grad norm:     {grad_norm:.6f}")
        gain = r_after - self.current_reward
        self._status(
            f"GD step (lr={lr}).  grad={grad_norm:.6f}  "
            f"r_before={float(r):.6f}  r_after={r_after:.6f}  "
            f"env_gain={gain:+.6f}")

    def on_reset_gd(self):
        self._clear_gd_state()
        self._var_grad.set("Grad norm:     -- (GD reset)")
        self._status("Adam momentum cleared. Next GD Step starts fresh.")

    def on_submit(self):
        act_np = self._get_action()
        self._status("Executing action in simulator...")

        obs_next = self.env.step(act_np)
        if obs_next is None:
            self._status("WARNING: simulator returned None -- episode may have exploded.")
            return

        with torch.no_grad():
            occ_next = self.model_dy.obs_to_occupancy(
                obs_next, self.cam_params).to(self.device)
            r_actual = float((occ_next[0] * self.score_tensor).sum().item())

        gain = r_actual - self.current_reward
        prev = self.current_reward
        self.occ_init       = occ_next.detach()
        self.current_reward = r_actual
        self._clear_gd_state()
        self._occ_pred_np = None
        self._r_pred      = None
        self._redraw(act_np, occ_pred_np=None, r_pred=None)
        self._var_r_pred.set("Predicted r:   -- (Evaluate for new state)")
        self._var_gain.set(  f"Last gain:     {gain:+.6f}  (actual)")
        self._status(
            f"Submitted.  actual_r={r_actual:.6f}  gain={gain:+.6f}  "
            f"(prev={prev:.6f})")

    def on_reset_env(self):
        self._status("Resetting environment...")
        self.env.reset()
        self._refresh_env_state()
        self._occ_pred_np = None
        self._r_pred      = None
        self._redraw(self._get_action(), occ_pred_np=None, r_pred=None)
        self._var_r_pred.set("Predicted r:   --")
        self._var_gain.set(  "Gain:          --")
        self._var_grad.set(  "Grad norm:     --")
        self._status("Environment reset.")

    def _on_close(self):
        self._status("Shutting down...")
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    MPCDebugGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
