"""
GenesisOracleEnv — batched Genesis wrapper for ``simple_mpc.oracle_mpc``.

Genesis only allows ``gs.init()`` once per process, and a scene's ``n_envs``
is fixed at ``build()``, so a separate 1-env "real" sim plus a K-env
"planning" sim is not possible. Instead this builds **one**
``SandboxManipulation(n_envs=K)`` scene (mirroring
``Genesis/data_collection_clean.py``'s multi-env batching):

  - Env 0 is the canonical "real" environment. Its particle state is ground
    truth, and the overhead camera is bound to it via
    ``scene.add_camera(env_idx=0)`` + ``VisOptions(rendered_envs_idx=[0])``,
    so ``render()`` only ever sees env 0.
  - All K envs (including env 0) serve as rollout workers during planning:
    ``rollout_candidates`` first broadcasts env 0's particle state to every
    env (``SandboxManipulation.broadcast_state_from_env``), then each env
    executes a *different* candidate action sequence via the already-batched
    ``execute_action(p_start[K,3], p_stop[K,3], angle[K])`` path.
  - Executing the winning action (``step()``) broadcasts it to *all* K envs
    and steps them in lockstep, so every env stays in an identical state
    between MPC steps — the realized outcome should exactly match the
    winning rollout, which doubles as a state-sync correctness check.

See docs/oracle_mpc_plan.md §1.1–1.2 for the full design rationale.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Genesis.sandbox_manipulation_clean import SandboxManipulation
from env.genesis_env import GenesisEnv
from transforms.functional import (
    genesis_particles_to_cam3d,
    footprint_radius_voxels,
    particles_to_occupancy,
)


class GenesisOracleEnv:
    """
    Batched analogue of ``env.genesis_env.GenesisEnv`` with ``n_envs`` copies:
    env 0 is "reality" (rendered, stepped for real); all envs are rollout
    workers during planning.

    Exposes the same real-env surface as ``GenesisEnv`` (``render``, ``step``,
    ``reset``, ``get_cam_params``, ``get_cam_extrinsics``) plus a planning
    API (``rollout_candidates``, ``particles_to_occ``) used by
    ``simple_mpc.oracle_mpc.run_oracle_mpc``.
    """

    def __init__(self, cfg: dict, n_envs: int):
        ds = cfg.get('dataset', cfg)   # accept both wrapped and flat forms

        self._cam_h   = float(ds.get('cam_height', 0.3))
        self._cam_fov = float(ds.get('cam_fov', 45.0))
        self.screenWidth  = int(ds.get('render_width',  720))
        self.screenHeight = int(ds.get('render_height', 720))

        self.global_scale = 2.0 * self._cam_h
        self.wkspc_w      = float(ds.get('wkspc_w', 0.064))
        self.settle_steps       = int(ds.get('settle_steps', 100))
        self.reset_warmup_steps = int(ds.get('reset_warmup_steps', 0))
        self.config = cfg
        self.n_envs = int(n_envs)
        if self.n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")

        # Build the Genesis sim config via GenesisEnv's existing builder (no
        # duplication — it is a pure @staticmethod), then restrict rendering
        # to env 0 so the overhead camera never sees the other rollout
        # workers laid out alongside it.
        genesis_cfg = GenesisEnv._build_genesis_config(ds)
        genesis_cfg['simulation']['rendered_envs_idx'] = [0]

        headless    = bool(ds.get('headless', True))
        viewer_type = None if headless else 'bird'
        self._sim = SandboxManipulation(genesis_cfg, n_envs=self.n_envs, viewer_type=viewer_type)

        mat = genesis_cfg['material']
        psize = mat['particle_size']
        self._particle_size_m = float(psize) if isinstance(psize, (int, float)) else float(np.mean(psize))

        # Overhead camera bound to env 0 only (env_idx=0 offsets the camera
        # pose by envs_offset[0]; rendered_envs_idx=[0] above keeps the
        # rasterizer from drawing the other K-1 rollout-worker copies).
        self._cam = self._sim._scene.add_camera(
            res=(self.screenWidth, self.screenHeight),
            pos=(0.0, 0.0, self._cam_h),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=self._cam_fov,
            GUI=False,
            env_idx=0,
        )

        self._sim.build()
        self._sim.shuffle_particles()
        self._sim.update_material_state()
        self._sim.broadcast_state_from_env(0)

        self._mat_idxc: np.ndarray | None = None

        # Real-step vs. rollout-phase step budgets (docs/oracle_mpc_plan.md
        # open question #2: halve every phase except the sweep — sweep step
        # count is already the physical minimum given plate speed/distance,
        # not an independent "steps" knob).
        self._real_settle_steps    = self.settle_steps
        self._real_clearance_steps = self._sim._clearance_ctrl_steps
        mpc_cfg = cfg.get('mpc', {})
        self._rollout_settle_steps = int(mpc_cfg.get(
            'rollout_settle_steps', max(1, self._real_settle_steps // 2)))
        self._rollout_clearance_steps = max(1, self._real_clearance_steps // 2)

    # ------------------------------------------------------------------ #
    #  Material segmentation helpers (identical to GenesisEnv)             #
    # ------------------------------------------------------------------ #

    def _build_material_idxc(self) -> None:
        idxc_map: dict = self._sim._scene.visualizer.segmentation_idx_dict
        mat_seg_keys = {(p.idx, p.link_start) for p in self._sim.material}
        self._mat_idxc = np.array(
            [idxc for idxc, seg_key in idxc_map.items()
             if seg_key in mat_seg_keys],
            dtype=np.int64,
        )

    def _material_mask(self, seg_arr: np.ndarray) -> np.ndarray:
        if self._mat_idxc is None:
            self._build_material_idxc()
        return np.isin(seg_arr, self._mat_idxc)

    # ------------------------------------------------------------------ #
    #  Real-env API (env 0) — signature-compatible with GenesisEnv         #
    # ------------------------------------------------------------------ #

    def render(self) -> np.ndarray:
        """Render env 0. Returns (H, W, 5) float32 — same convention as GenesisEnv.render()."""
        rgb_arr, _, seg_arr, _ = self._cam.render(rgb=True, depth=False, segmentation=True)

        rgb_np = np.asarray(rgb_arr, dtype=np.uint8)
        seg_np = np.asarray(seg_arr, dtype=np.int64)
        is_mat = self._material_mask(seg_np)

        obs = np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
        obs[..., :3] = rgb_np[..., :3].astype(np.float32) / 255.0
        obs[..., 3]  = is_mat.astype(np.float32)
        obs[..., 4]  = np.where(is_mat, self._cam_h, 2.0 * self._cam_h)
        return obs

    def step(self, action: np.ndarray, video_recorder=None) -> np.ndarray:
        """Execute a push, broadcast to ALL envs, and step them in lockstep.

        Because every env receives the identical action from an identical
        (post-broadcast) state, all K envs end this call in the same state —
        keeping them synchronized for the next planning phase.
        """
        import genesis as gs

        sx, sy, ex, ey = (float(action[0]), float(action[1]),
                          float(action[2]), float(action[3]))
        dxy   = math.hypot(ex - sx, ey - sy)
        angle = math.atan2(ey - sy, ex - sx) + math.pi / 2 if dxy > 1e-6 else 0.0

        z_op = float(self._sim._operation_height)
        p_start = torch.tensor([sx, sy, z_op], dtype=torch.float32, device=gs.device
                                ).unsqueeze(0).expand(self.n_envs, 3).contiguous()
        p_stop  = torch.tensor([ex, ey, z_op], dtype=torch.float32, device=gs.device
                                ).unsqueeze(0).expand(self.n_envs, 3).contiguous()
        angle_t = torch.full((self.n_envs,), angle, dtype=torch.float32, device=gs.device)

        self._sim.execute_action(p_start, p_stop, angle_t)
        self._sim._settle_steps         = self._real_settle_steps
        self._sim._clearance_ctrl_steps = self._real_clearance_steps
        self._sim.update_material_state()
        obs = self.render()

        if video_recorder is not None:
            writer = video_recorder[0] if isinstance(video_recorder, list) else video_recorder
            bgr = np.ascontiguousarray(
                (np.clip(obs[..., :3], 0.0, 1.0) * 255).astype(np.uint8)[..., ::-1]
            )
            writer.write(bgr)

        return obs

    def reset(self) -> None:
        """Shuffle particles (all envs), then broadcast env 0's post-settle
        state to every env so all K copies start identical."""
        self._sim._settle_steps = self.reset_warmup_steps
        self._sim.shuffle_particles()
        self._sim.update_material_state()
        self._sim.broadcast_state_from_env(0)
        self._sim._settle_steps = self._real_settle_steps
        return None

    def set_save_render_mode(self) -> None:
        pass

    def restore_native_render_mode(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Camera intrinsics / extrinsics (identical to GenesisEnv)            #
    # ------------------------------------------------------------------ #

    def get_cam_params(self) -> list:
        f  = float(self._cam.f)
        cx = float(self._cam.cx)
        cy = float(self._cam.cy)
        return [f, f, cx, cy]

    def get_cam_extrinsics(self) -> np.ndarray:
        return np.eye(4, dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  Planning API — used by simple_mpc.oracle_mpc                        #
    # ------------------------------------------------------------------ #

    def rollout_candidates(
        self,
        act_seqs: torch.Tensor,
        collect_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Roll out one candidate action sequence per env, in parallel.

        Parameters
        ----------
        act_seqs : (n_envs, n_ahead, 4) tensor — [sx, sy, ex, ey] per step,
            one independent candidate sequence per env.
        collect_intermediate : bool
            If True, also return the post-push particle positions after
            *every* horizon step (needed for ``cost_mode='discounted'`` in
            ``oracle_mpc``), not just the terminal state.

        Returns
        -------
        terminal_pos : (n_envs, n_particles, 3) world-frame particle
            positions after the full sequence, using the reduced-fidelity
            ``rollout_*`` step budgets.
        step_positions : list[(n_envs, n_particles, 3)], length n_ahead —
            only returned when ``collect_intermediate=True``.

        Restores env 0's state onto all envs first, so this is safe to call
        repeatedly across optimizer iterations without executing any real
        step.
        """
        import genesis as gs

        n_envs, n_ahead, _ = act_seqs.shape
        if n_envs != self.n_envs:
            raise ValueError(f"act_seqs batch dim {n_envs} != n_envs {self.n_envs}")

        self._sim.broadcast_state_from_env(0)
        self._sim._settle_steps         = self._rollout_settle_steps
        self._sim._clearance_ctrl_steps = self._rollout_clearance_steps

        z_op = float(self._sim._operation_height)
        step_positions = []
        for s in range(n_ahead):
            a = act_seqs[:, s, :].to(device=gs.device, dtype=torch.float32)
            sx, sy, ex, ey = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            dxy   = torch.hypot(ex - sx, ey - sy)
            angle = torch.where(
                dxy > 1e-6,
                torch.atan2(ey - sy, ex - sx) + math.pi / 2,
                torch.zeros_like(dxy),
            )
            z_col   = torch.full_like(sx, z_op)
            p_start = torch.stack([sx, sy, z_col], dim=1)
            p_stop  = torch.stack([ex, ey, z_col], dim=1)

            self._sim.execute_action(p_start, p_stop, angle)
            self._sim.update_material_state()
            if collect_intermediate:
                step_positions.append(self._sim._particle_state[:, :, 0:3].clone())

        self._sim._settle_steps         = self._real_settle_steps
        self._sim._clearance_ctrl_steps = self._real_clearance_steps

        terminal_pos = self._sim._particle_state[:, :, 0:3].clone()
        if collect_intermediate:
            return terminal_pos, step_positions
        return terminal_pos

    def current_particles_world(self) -> torch.Tensor:
        """Env 0's current world-frame particle positions — (1, n_particles, 3).

        Use with ``particles_to_occ`` to get a footprint-splatted "current
        occupancy" that's representation-consistent with rollout predictions
        (both sparse/footprint-splatted), as opposed to the dense
        depth-derived occupancy used for reward *reporting*
        (``oracle_mpc._report_occupancy_from_obs``), which intentionally
        matches the learned-model MPC convention instead.
        """
        return self._sim._particle_state[0:1, :, 0:3].clone()

    def particles_to_occ(
        self,
        pos_world: torch.Tensor,
        grid_bounds: dict,
        grid_res: tuple,
        footprint_radius: float | None = None,
    ) -> torch.Tensor:
        """(n_envs, n_particles, 3) world positions → (n_envs, *grid_res) occupancy.

        Uses the same normalized-camera-coordinate convention as
        ``EulerianAdapter`` / ``OccupancyReward``
        (``transforms.functional.genesis_particles_to_cam3d``), so the
        result is directly comparable to / usable with the existing goal
        score maps and loss registry.
        """
        if footprint_radius is None:
            footprint_radius = self.default_footprint_radius_voxels(grid_bounds, grid_res)
        pts_cam = genesis_particles_to_cam3d(pos_world, self.global_scale)
        return particles_to_occupancy(pts_cam, grid_bounds, grid_res,
                                       footprint_radius=footprint_radius)

    def default_footprint_radius_voxels(self, grid_bounds: dict, grid_res: tuple) -> float:
        """Particle-footprint voxel radius for ``particles_to_occ``, derived
        from the configured (scalar) particle size."""
        return footprint_radius_voxels(
            self._particle_size_m, self.global_scale, grid_bounds, grid_res)

    def destroy(self) -> None:
        self._sim.destroy()
