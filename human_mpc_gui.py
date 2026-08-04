#!/usr/bin/env python3
from __future__ import annotations

"""
Human-demonstration GUI for the Genesis-oracle ceiling baseline.

Lets a person pilot the pusher plate directly against the real Genesis
simulator (not a learned model), to measure the reward ceiling achievable on
a task independent of any MPC sampling optimizer. See
docs/human_demo_design.md for the full design and
simple_mpc/human_mpc.py / simple_mpc/human_grid_search.py for the underlying
session + grid-search-refinement logic this GUI drives.

Deliberately modeled on debug_mpc_gui.py's canvas-drag interaction (same
draggable start/end handles, same tile-image helpers, reused directly from
that module) — but the "model" here is the real simulator, so every
Refine/Submit click costs real physics compute (tens of candidates rolled
out), unlike debug_mpc_gui.py's near-instant learned-model forward pass.
Two consequences follow directly from that:

  * There is no auto-evaluate-on-drag-release — dragging only updates the
    drawn arrow (cheap); a rollout only happens on explicit Refine/Submit.
  * Only two tiles are shown (current occupancy, goal score map) — no
    predicted-occupancy heatmap tile. Refine still reports a predicted
    reward NUMBER (from the grid search's own full-fidelity re-roll), just
    not rendered as a heatmap.

Action is 5D — [sx, sy, ex, ey, angle_norm] — the plate yaw is a free 5th
input (an "Auto" button resets it to the usual perpendicular-to-travel
default; see transforms.functional.action_to_pose), not derived from travel
direction like every automated sampler/optimizer in this codebase.

Workflow per real step: drag START/END, optionally set the angle slider,
then either click **Submit** directly (executes exactly the drawn action,
no grid search) or click **Refine** first (runs the grid search around your
drawn action — slow, prints a predicted reward and updates the fields to the
refined action) and *then* Submit. Refinement is strictly optional — Submit
always executes whatever action is currently shown in the fields, regardless
of whether Refine was ever clicked.

Usage
-----
    python human_mpc_gui.py [--config simple_mpc/config/config_human_demo.yaml]
"""

import argparse
import math
import os
import time

import numpy as np
import tkinter as tk

from utils import (
    load_yaml,
    set_seed,
    scale_subgoal_to_material_pixels,
    get_current_YYYY_MM_DD_hh_mm_ss_ms,
)
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.human_mpc import HumanDemoSession, save_episode
from run_oracle_mpc import build_goal

# Reused as-is — pure image/tile helpers with no side effects at import time.
from debug_mpc_gui import (
    _heatmap_bgr,
    _stamp,
    _bgr_to_photo,
    TILE_SIZE,
    CANVAS_SIZE,
    FONT_MONO,
    FONT_MONO_B,
    FONT_HDR,
)

import cv2


DEFAULT_CONFIG = 'simple_mpc/config/config_human_demo.yaml'


class HumanDemoGUI:
    _HANDLE_R = 10

    def __init__(self, root: tk.Tk, cfg: dict):
        self.root = root
        self.root.title("Human Demo GUI — Genesis Oracle")
        self.root.configure(bg='#1a1a1a')

        self._status_var = tk.StringVar(value="Starting up...")
        root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._status("Building GenesisOracleEnv (this can take a while)...")
        root.update()

        self.cfg     = cfg
        self.wkspc_w = float(cfg['dataset']['wkspc_w'])
        n_envs       = int(cfg['mpc']['n_envs'])

        self.env = GenesisOracleEnv(cfg, n_envs=n_envs)

        self.run_dir = os.path.join(
            cfg['mpc'].get('output_dir', 'outputs/human_demo'),
            get_current_YYYY_MM_DD_hh_mm_ss_ms())
        os.makedirs(self.run_dir, exist_ok=True)
        # Same convention as run_oracle_mpc.py: save the exact config this
        # run used, so a saved episode's data is reproducible/inspectable
        # without having to know which config file (or CLI override) produced it.
        import yaml
        with open(os.path.join(self.run_dir, 'run_config.yaml'), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        self.ep_idx    = 0
        self.seed_base = int(cfg.get('episodes', {}).get('random_seed_base', 0))
        self.session: HumanDemoSession | None = None
        self._angle_user_set = False   # False => angle slider auto-follows travel direction
        self._proposed_key = None      # last action5 tuple a propose() succeeded for

        self._build_ui()
        self._start_new_episode()
        self._status("Ready — drag handles, set angle, then Submit (or Refine first, optionally).")

    # ─────────────────────────── episode lifecycle ───────────────────────

    def _start_new_episode(self):
        seed = self.seed_base + self.ep_idx
        self._status(f"Starting episode {self.ep_idx} (seed={seed})...")
        self.root.update_idletasks()

        set_seed(seed)
        self.env.reset()
        self.env.set_recording_context({
            'source':      'human_demo',
            'episode_idx': self.ep_idx,
            'seed':        seed,
            'optimizer':   'human_grid_search',
        })

        subgoal  = build_goal(self.cfg, self.env)
        obs_init = self.env.render()
        subgoal  = scale_subgoal_to_material_pixels(
            subgoal, obs_init[..., -1], self.cfg['dataset']['global_scale'])

        self.session   = HumanDemoSession(self.env, subgoal, self.cfg)
        self._ep_t0    = time.time()
        self._proposed_key = None

        half_w = self.wkspc_w * 0.4
        self._set_action_fields(np.array(
            [-half_w, -half_w, half_w, half_w, 0.5], dtype=np.float32))
        self._angle_user_set = False
        self._sync_angle_to_default()

        self._refresh_state_tiles()
        self._var_pred.set("Predicted r:   -- (Refine is optional)")
        self._var_gain.set("Gain:          --")
        self._status(f"Episode {self.ep_idx} ready. step 0/{self.session.n_mpc_cap}")

    def _finish_episode(self, reason: str):
        if self.session is None:
            return
        elapsed = time.time() - self._ep_t0
        result  = self.session.finalize()
        ep_dir  = os.path.join(self.run_dir, f'episode_{self.ep_idx:03d}')
        seed    = self.seed_base + self.ep_idx
        metrics = save_episode(result, ep_dir, self.ep_idx, seed, elapsed)
        self._status(
            f"Episode {self.ep_idx} finished ({reason}). "
            f"steps={metrics['n_steps']} reward {metrics['reward_init']:.4f} -> "
            f"{metrics['reward_final']:.4f}  saved to {ep_dir}")
        self.session = None
        self.ep_idx += 1

    # ─────────────────────────── action helpers ───────────────────────────

    def _get_action5(self) -> np.ndarray:
        try:
            sx, sy, ex, ey = (float(self._vars[k].get()) for k in ('sx', 'sy', 'ex', 'ey'))
            angle_deg = float(self._var_angle.get())
        except (ValueError, KeyError):
            return np.zeros(5, dtype=np.float32)
        angle_norm = (angle_deg % 180.0) / 180.0
        return np.array([sx, sy, ex, ey, angle_norm], dtype=np.float32)

    def _set_action_fields(self, act5: np.ndarray):
        for key, val in zip(('sx', 'sy', 'ex', 'ey'), act5[:4]):
            self._vars[key].set(f"{val:.4f}")
        self._var_angle.set(float(np.clip(act5[4] * 180.0, 0.0, 179.999)))
        self._update_canvas_handles()

    def _sync_angle_to_default(self):
        if self._angle_user_set or self.session is None:
            return
        sx, sy, ex, ey = (float(self._vars[k].get()) for k in ('sx', 'sy', 'ex', 'ey'))
        norm = HumanDemoSession.default_angle_norm(sx, sy, ex, ey)
        self._var_angle.set(round(norm * 180.0, 1))

    def _on_auto_angle(self):
        self._angle_user_set = False
        self._sync_angle_to_default()
        self._update_canvas_handles()

    def _on_angle_slider(self, _value=None):
        self._angle_user_set = True
        self._update_canvas_handles()

    def _invalidate_proposal(self):
        self._proposed_key = None

    # ─────────────────────────── tiles / display ───────────────────────────

    def _refresh_state_tiles(self):
        occ_np = self.session.occ_cur_opt[0].detach().cpu().numpy()
        score_np = self.session.score_np
        T = TILE_SIZE

        tile0 = _stamp(_heatmap_bgr(occ_np, T),
                        [f"Current occ", f"r={self.session.current_reward:.5f}"])
        tile1 = _stamp(_heatmap_bgr(score_np, T, cv2.COLORMAP_HOT),
                        [f"Score map", f"[{score_np.min():.2f}, {score_np.max():.2f}]"])

        for i, tile in enumerate([tile0, tile1]):
            photo = _bgr_to_photo(tile, T, T)
            self._img_widgets[i].configure(image=photo)
            self._photo_refs[i] = photo

        self._var_r_cur.set(f"Current r:     {self.session.current_reward:.6f}")
        step_lbl = f"{self.session.step_idx}/{self.session.n_mpc_cap}"
        self._var_step.set(f"Episode {self.ep_idx}  step {step_lbl}")

    # ─────────────────────────── UI builder ────────────────────────────────

    def _build_ui(self):
        T, C = TILE_SIZE, CANVAS_SIZE
        BG, BG2 = '#1a1a1a', '#222222'

        tile_frame = tk.Frame(self.root, bg=BG)
        tile_frame.pack(padx=10, pady=(10, 4))
        self._img_widgets, self._photo_refs = [], [None, None]
        for col, lbl in enumerate(["Current occupancy", "Goal score map"]):
            f = tk.Frame(tile_frame, bg=BG)
            f.grid(row=0, column=col, padx=5)
            tk.Label(f, text=lbl, bg=BG, fg='#aaaaaa', font=FONT_HDR).pack()
            w = tk.Label(f, bg=BG2, width=T, height=T)
            w.pack()
            self._img_widgets.append(w)

        row1 = tk.Frame(self.root, bg=BG)
        row1.pack(padx=10, pady=4, fill=tk.X)

        canvas_frame = tk.LabelFrame(
            row1,
            text="  Workspace  (drag ORANGE=start, CYAN=end)  ",
            fg='#888888', bg=BG2, font=FONT_HDR)
        canvas_frame.pack(side=tk.LEFT, padx=(0, 10))

        self._canvas = tk.Canvas(canvas_frame, width=C, height=C,
                                  bg='#111111', highlightthickness=0)
        self._canvas.pack(padx=4, pady=4)
        self._canvas.create_rectangle(0, 0, C - 1, C - 1, outline='#444444', width=1)
        self._canvas.create_line(C // 2, 0, C // 2, C, fill='#333333', dash=(3, 6))
        self._canvas.create_line(0, C // 2, C, C // 2, fill='#333333', dash=(3, 6))

        wkspc = self.wkspc_w
        for v in [-wkspc, 0, wkspc]:
            cx, _ = self._world_to_canvas(v, 0)
            _, cy = self._world_to_canvas(0, v)
            self._canvas.create_text(cx, C - 10, text=f"{v:.1f}", fill='#555555', font=('Courier', 8))
            self._canvas.create_text(8, cy, text=f"{v:.1f}", fill='#555555', font=('Courier', 8))

        self._cv_line = self._canvas.create_line(
            0, 0, 1, 1, fill='#88ff88', width=2, arrow=tk.LAST, arrowshape=(12, 14, 5))
        self._cv_orient = self._canvas.create_line(0, 0, 1, 1, fill='#ffee00', width=3)
        self._cv_s_dot = self._canvas.create_oval(0, 0, 1, 1, fill='#ff8800', outline='white', width=2)
        self._cv_e_dot = self._canvas.create_oval(0, 0, 1, 1, fill='#00ddff', outline='white', width=2)
        self._cv_s_lbl = self._canvas.create_text(0, 0, text='S', fill='white', font=('Courier', 10, 'bold'))
        self._cv_e_lbl = self._canvas.create_text(0, 0, text='E', fill='white', font=('Courier', 10, 'bold'))

        self._dragging = None
        for item in (self._cv_s_dot, self._cv_s_lbl):
            self._canvas.tag_bind(item, '<ButtonPress-1>', lambda e: self._drag_start(e, 's'))
        for item in (self._cv_e_dot, self._cv_e_lbl):
            self._canvas.tag_bind(item, '<ButtonPress-1>', lambda e: self._drag_start(e, 'e'))
        self._canvas.bind('<B1-Motion>', self._drag_move)
        self._canvas.bind('<ButtonRelease-1>', self._drag_release)

        hint_frame = tk.Frame(row1, bg=BG)
        hint_frame.pack(side=tk.LEFT, padx=4, anchor='n')
        for text, color in [
            ("● ORANGE -- push start", '#ff8800'),
            ("● CYAN   -- push end", '#00ccdd'),
            ("● YELLOW -- plate orientation", '#ffee00'),
            ("", '#888888'),
            ("Dragging never runs the sim.", '#888888'),
            ("Refine/Submit are explicit.", '#888888'),
        ]:
            tk.Label(hint_frame, text=text, bg=BG, fg=color, font=FONT_MONO, anchor='w').pack(anchor='w')

        act_frame = tk.LabelFrame(
            self.root, text="  Action  [sx sy ex ey]  world-2D,  angle in degrees  ",
            fg='#888888', bg=BG2, font=FONT_HDR)
        act_frame.pack(fill=tk.X, padx=10, pady=4)

        self._vars = {}
        for col, key in enumerate(['sx', 'sy', 'ex', 'ey']):
            self._vars[key] = tk.StringVar(value="0.0000")
            tk.Label(act_frame, text=f"  {key}:", bg=BG2, fg='#cccccc', font=FONT_MONO_B).grid(
                row=0, column=col * 2, sticky='e')
            e = tk.Entry(act_frame, textvariable=self._vars[key], width=10,
                         bg='#333333', fg='#ffffff', insertbackground='white', font=FONT_MONO)
            e.grid(row=0, column=col * 2 + 1, padx=4, pady=8)
            e.bind('<Return>', lambda _: self._on_field_edited())
            e.bind('<KeyRelease>', lambda _: self._on_field_edited())

        tk.Label(act_frame, text="  angle:", bg=BG2, fg='#ffee88', font=FONT_MONO_B).grid(
            row=0, column=8, sticky='e')
        self._var_angle = tk.DoubleVar(value=0.0)
        angle_scale = tk.Scale(act_frame, variable=self._var_angle, from_=0, to=179.9,
                                resolution=0.5, orient=tk.HORIZONTAL, length=180,
                                bg='#333344', fg='#aaddff', troughcolor='#222222',
                                highlightthickness=0, command=self._on_angle_slider)
        angle_scale.grid(row=0, column=9, padx=4, pady=4)
        tk.Button(act_frame, text="Auto", command=self._on_auto_angle,
                  bg='#3a3a3a', fg='white', font=FONT_MONO).grid(row=0, column=10, padx=6)

        info_frame = tk.Frame(self.root, bg=BG)
        info_frame.pack(fill=tk.X, padx=10, pady=2)
        self._var_step   = tk.StringVar(value="Episode --  step --")
        self._var_r_cur  = tk.StringVar(value="Current r:     --")
        self._var_pred   = tk.StringVar(value="Predicted r:   --")
        self._var_gain   = tk.StringVar(value="Gain:          --")
        for v in (self._var_step, self._var_r_cur, self._var_pred, self._var_gain):
            tk.Label(info_frame, textvariable=v, bg=BG, fg='#eeeeee', font=FONT_MONO,
                     width=26, anchor='w').pack(side=tk.LEFT, padx=10)

        tk.Label(self.root, textvariable=self._status_var, bg=BG, fg='#88ff88',
                 font=FONT_MONO, anchor='w').pack(fill=tk.X, padx=12, pady=2)

        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=10)
        btn_cfg = dict(font=FONT_MONO_B, width=16, pady=6)
        for label, cmd, bg in [
            ("Refine",             self.on_refine,          '#1a4a7a'),
            ("Submit",             self.on_submit,          '#2a6a1a'),
            ("Finish Episode",     self.on_finish_episode,  '#7a5a1a'),
            ("New Episode",        self.on_new_episode,     '#3a3a3a'),
        ]:
            tk.Button(btn_frame, text=label, command=cmd, bg=bg, fg='white', **btn_cfg).pack(
                side=tk.LEFT, padx=6)

    # ─────────────────────────── coordinate helpers ────────────────────────

    def _world_to_canvas(self, wx: float, wy: float) -> tuple:
        C, w = CANVAS_SIZE, self.wkspc_w
        cx = int((wx + w) / (2 * w) * (C - 1))
        cy = int((C - 1) - (wy + w) / (2 * w) * (C - 1))
        return cx, cy

    def _canvas_to_world(self, cx: int, cy: int) -> tuple:
        C, w = CANVAS_SIZE, self.wkspc_w
        wx = cx / (C - 1) * (2 * w) - w
        wy = ((C - 1) - cy) / (C - 1) * (2 * w) - w
        return float(wx), float(wy)

    # ─────────────────────────── canvas drag ───────────────────────────────

    def _update_canvas_handles(self):
        act = self._get_action5()
        R = self._HANDLE_R
        sx_c, sy_c = self._world_to_canvas(act[0], act[1])
        ex_c, ey_c = self._world_to_canvas(act[2], act[3])

        self._canvas.coords(self._cv_line, sx_c, sy_c, ex_c, ey_c)
        self._canvas.coords(self._cv_s_dot, sx_c - R, sy_c - R, sx_c + R, sy_c + R)
        self._canvas.coords(self._cv_e_dot, ex_c - R, ey_c - R, ex_c + R, ey_c + R)
        self._canvas.coords(self._cv_s_lbl, sx_c, sy_c)
        self._canvas.coords(self._cv_e_lbl, ex_c, ey_c)

        # Orientation tick: a short yellow segment through the start point at
        # the plate's yaw (act[4] normalized [0,1) -> [0, pi) radians) — purely
        # visual feedback, independent of the travel (green) arrow.
        angle_rad = float(act[4]) * math.pi
        tick_len_px = self._HANDLE_R * 2.5
        dx, dy = math.cos(angle_rad), math.sin(angle_rad)
        self._canvas.coords(
            self._cv_orient,
            sx_c - dx * tick_len_px, sy_c + dy * tick_len_px,
            sx_c + dx * tick_len_px, sy_c - dy * tick_len_px)

        for item in (self._cv_s_dot, self._cv_e_dot, self._cv_s_lbl, self._cv_e_lbl):
            self._canvas.tag_raise(item)

    def _drag_start(self, event, which):
        self._dragging = which

    def _drag_move(self, event):
        if self._dragging is None:
            return
        C = CANVAS_SIZE
        cx, cy = max(0, min(C - 1, event.x)), max(0, min(C - 1, event.y))
        wx, wy = self._canvas_to_world(cx, cy)
        lo, hi = self.session.clip_lo, self.session.clip_hi
        if self._dragging == 's':
            self._vars['sx'].set(f"{np.clip(wx, lo[0], hi[0]):.4f}")
            self._vars['sy'].set(f"{np.clip(wy, lo[1], hi[1]):.4f}")
        else:
            self._vars['ex'].set(f"{np.clip(wx, lo[2], hi[2]):.4f}")
            self._vars['ey'].set(f"{np.clip(wy, lo[3], hi[3]):.4f}")
        self._sync_angle_to_default()
        self._update_canvas_handles()
        self._invalidate_proposal()

    def _drag_release(self, event):
        self._dragging = None

    def _on_field_edited(self):
        self._sync_angle_to_default()
        self._update_canvas_handles()
        self._invalidate_proposal()

    # ─────────────────────────── button callbacks ──────────────────────────

    def on_refine(self):
        if self.session is None:
            self._status("No active episode — click New Episode first.")
            return
        act5 = self._get_action5()
        n_cand = self.session.grid_n ** 5
        self._status(f"Running grid search ({n_cand} candidates)... this uses real physics, please wait.")
        self.root.update_idletasks()

        result = self.session.propose(act5)
        self._set_action_fields(result['best_action'])
        self._angle_user_set = True   # refined angle is now authoritative until the user drags again
        self._proposed_key = tuple(np.round(result['best_action'], 6))

        gain = result['predicted_reward'] - self.session.current_reward
        self._var_pred.set(f"Predicted r:   {result['predicted_reward']:.6f}")
        self._var_gain.set(f"Gain:          {gain:+.6f}")
        self._status(
            f"Refined over {result['n_candidates']} candidates. "
            f"pred_r={result['predicted_reward']:.6f}  gain={gain:+.6f}  "
            f"(fields updated to the refined action — Submit executes this)")

    def on_submit(self):
        if self.session is None:
            self._status("No active episode — click New Episode first.")
            return
        act5 = self._get_action5()
        # Refinement is optional: Submit always executes exactly the action
        # currently shown in the fields, whether or not Refine was clicked —
        # if it was (and fields weren't edited since), that's the refined
        # action; otherwise it's exactly what was drawn.
        was_refined = (self._proposed_key is not None
                       and tuple(np.round(act5, 6)) == self._proposed_key)

        self._status("Executing action in simulator...")
        self.root.update_idletasks()
        out = self.session.commit(act5)
        self._invalidate_proposal()
        self._var_pred.set("Predicted r:   -- (Refine for next step, optional)")
        self._var_gain.set(f"Last gain:     {out['gain']:+.6f}  (actual)")
        self._refresh_state_tiles()
        self._status(
            f"Submitted step {self.session.step_idx} "
            f"({'refined' if was_refined else 'as drawn'}).  "
            f"actual_r={out['reward']:.6f}  gain={out['gain']:+.6f}")

        if self.session.finished():
            self._finish_episode("reached step cap")

    def on_finish_episode(self):
        self._finish_episode("manual stop")
        self._var_pred.set("Predicted r:   --")
        self._var_gain.set("Gain:          --")

    def on_new_episode(self):
        self._start_new_episode()

    # ─────────────────────────── misc ──────────────────────────────────────

    def _status(self, msg: str):
        self._status_var.set(msg)
        print(msg)
        self.root.update_idletasks()

    def _on_close(self):
        self._status("Shutting down...")
        if self.session is not None:
            self._finish_episode("window closed")
        self.env.destroy()
        self.root.destroy()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    root = tk.Tk()
    HumanDemoGUI(root, cfg)
    root.mainloop()


if __name__ == '__main__':
    main()
