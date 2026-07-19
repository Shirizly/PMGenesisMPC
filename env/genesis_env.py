"""
GenesisEnv — thin wrapper around SandboxManipulation for the simple_mpc pipeline.

Coordinate conventions
----------------------
  World  : x/y horizontal, z up (meters).
  Camera : positioned directly overhead at (0, 0, cam_h), lookat=(0,0,0),
           up=(0,1,0).  Camera-X = world-X, camera-Y = −world-Y.
  obs    : (H, W, 5) float32 — channels [R, G, B, is_material, depth].
           depth channel is set to ``cam_h`` for material pixels so that
           ``depth / global_scale = 0.5 < _FG_DEPTH_THRESHOLD (≈0.749) → fg``,
           and to ``2*cam_h`` for background so that
           ``depth / global_scale = 1.0 > threshold → bg``.
           ``global_scale = 2 * cam_h``.

Action convention
-----------------
  ``step([sx, sy, ex, ey])`` accepts x/y in world-frame metres.
  For eulerian_wrapper, use ``action_convention='genesis'`` so the wrapper
  converts actions via ``_action_to_cam_3d_genesis`` (no OpenGL matrix needed).
"""

import math
import os
import sys
import numpy as np
import torch

# Ensure project root is on sys.path so Genesis utilities resolve correctly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Genesis.sandbox_manipulation_clean import SandboxManipulation
from utils import write_video_frame


class GenesisEnv:
    """
    Thin wrapper around :class:`SandboxManipulation` that exposes the same
    interface expected by ``simple_mpc`` (``run_experiments.py``, ``mpc.py``,
    ``adapters.py``).
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: dict):
        """
        Parameters
        ----------
        cfg : dict
            Full MPC config as loaded from ``config_simple.yaml``.
            Reads fields from ``cfg['dataset']``.
        """
        ds = cfg.get('dataset', cfg)   # accept both wrapped and flat forms

        # Camera / image dimensions
        self._cam_h   = float(ds.get('cam_height', 0.3))
        self._cam_fov = float(ds.get('cam_fov', 45.0))
        self.screenWidth  = int(ds.get('render_width',  720))
        self.screenHeight = int(ds.get('render_height', 720))

        # Attributes read (and written) by run_experiments.py / _sync_env_config
        self.global_scale               = 2.0 * self._cam_h
        self.wkspc_w                    = float(ds.get('wkspc_w', 0.064))
        self.obj                        = ds.get('obj', 'chickpeas')
        self.init_pos                   = ds.get('init_pos', None)
        self.fast_mode                  = bool(ds.get('fast_mode', False))
        self.action_step_size           = float(ds.get('action_step_size', 0.01))
        self.settle_steps               = int(ds.get('settle_steps', 100))
        self.reset_warmup_steps         = int(ds.get('reset_warmup_steps', 0))
        self.render_step_before_capture = bool(ds.get('render_step_before_capture', False))
        self.num_objects_override       = ds.get('num_objects_override',
                                                 ds.get('num_objects', None))
        self.config = cfg

        # Build Genesis simulation config and launch
        genesis_cfg = self._build_genesis_config(ds)
        headless    = bool(ds.get('headless', True))
        viewer_type = None if headless else 'bird'
        self._sim = SandboxManipulation(genesis_cfg, n_envs=1, viewer_type=viewer_type)

        # Add overhead camera BEFORE build()
        self._cam = self._sim._scene.add_camera(
            res=(self.screenWidth, self.screenHeight),
            pos=(0.0, 0.0, self._cam_h),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=self._cam_fov,
            GUI=False,
        )

        self._sim.build()
        self._sim.shuffle_particles()
        self._sim.update_material_state()

        # Lazily built after first render (once seg_idxc_map is populated)
        self._mat_idxc: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Genesis config builder                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_genesis_config(ds: dict) -> dict:
        """
        Build a Genesis sim config dict from the MPC ``dataset`` config.

        Uses safe defaults for any missing fields.  The caller should provide
        ``wkspc_w`` (half-width in metres) for correct box dimensions.
        """
        wkspc_w  = float(ds.get('wkspc_w', 0.064))
        box_side = wkspc_w * 2          # full box width/depth in metres
        box_h    = float(ds.get('box_height', 0.04))
        box_wall = float(ds.get('box_wall_thickness', 0.02))

        mat   = ds.get('material', {})
        plate = ds.get('plate', {})
        sim   = ds.get('simulation', {})

        n_particles   = int(mat.get('n_particles',
                                    ds.get('n_particles',
                                           ds.get('num_objects', 20))))
        particle_size = mat.get('particle_size', ds.get('particle_size', 0.01))
        shape         = mat.get('shape', ds.get('material_shape', 'sphere'))

        return {
            'simulation': {
                'dt':               float(sim.get('dt', 4e-3)),
                'substeps':         int(sim.get('substeps', 5)),
                'backend':          sim.get('backend', 'gpu'),
                'precision':        str(sim.get('precision', '32')),
                'performance_mode': bool(sim.get('performance_mode', True)),
                # timing knobs — read by SandboxManipulation for settle/ctrl steps
                'settle_steps':     int(ds.get('settle_steps',   100)),
                'pos_ctrl_steps':   int(ds.get('pos_ctrl_steps', 100)),
            },
            'rigid_options': {
                'iterations':         int(ds.get('rigid_iterations', 10)),
                'ls_iterations':      int(ds.get('rigid_ls_iterations', 10)),
                'tolerance':          float(ds.get('rigid_tolerance', 1e-4)),
                'ls_tolerance':       float(ds.get('rigid_ls_tolerance', 0.05)),
                'box_box_detection':  bool(ds.get('box_box_detection', True)),
                'use_contact_island': bool(ds.get('use_contact_island', True)),
                'use_hibernation':    bool(ds.get('use_hibernation', False)),
            },
            'box': {
                'vol':            [box_side, box_side, box_h],
                'wall_thickness': box_wall,
                'friction':       ds.get('box_friction', None),
            },
            'material': {
                'vol':           mat.get('vol',
                                        [box_side * 0.99, box_side * 0.99, 0.05]),
                'shape':         shape,
                'particle_size': particle_size,
                'n_particles':   n_particles,
                'density':       mat.get('density',
                                        ds.get('particle_density', 1000.0)),
                'friction':      mat.get('friction',
                                        ds.get('particle_friction', 0.5)),
            },
            'plate': {
                'speed': float(plate.get('speed',
                                        ds.get('plate_speed', 0.125))),
                'size':  plate.get('size',
                                   ds.get('plate_size', [0.04, 0.002, 0.01])),
            },
            'data_collection': {'sampled': {}},
        }

    # ------------------------------------------------------------------ #
    #  Material segmentation helpers                                       #
    # ------------------------------------------------------------------ #

    def _build_material_idxc(self) -> None:
        """
        Populate ``self._mat_idxc`` with the seg idxc values that belong to
        material particles.

        ``scene.visualizer.segmentation_idx_dict`` returns ``{idxc: seg_key}``
        where (with the default segmentation_level='link'):
          - background → seg_key = -1
          - rigid objects → seg_key = (entity.idx, link.idx)  [a 2-tuple]

        For single-link particle entities, entity.idx == link.idx == link_start,
        so we build a set of (p.idx, p.link_start) tuples and filter by
        membership.  Confirmed by test: p.idx and p.link_start are identical for
        sphere particles (verified 2026-05-25 via _test_genesis_api.py).
        """
        idxc_map: dict = self._sim._scene.visualizer.segmentation_idx_dict
        mat_seg_keys = {(p.idx, p.link_start) for p in self._sim.material}
        self._mat_idxc = np.array(
            [idxc for idxc, seg_key in idxc_map.items()
             if seg_key in mat_seg_keys],
            dtype=np.int64,
        )

    def _material_mask(self, seg_arr: np.ndarray) -> np.ndarray:
        """Return (H, W) bool mask — True where pixel belongs to a material particle."""
        if self._mat_idxc is None:
            self._build_material_idxc()
        return np.isin(seg_arr, self._mat_idxc)

    # ------------------------------------------------------------------ #
    #  Core environment API                                                #
    # ------------------------------------------------------------------ #

    def render(self) -> np.ndarray:
        """
        Render the current state.

        Returns
        -------
        obs : (H, W, 5) float32
            Channels: [R, G, B, is_material_float, depth]

            The depth channel encodes material vs background:
              - material pixel → depth = cam_h   (depth/global_scale = 0.5)
              - background     → depth = 2*cam_h (depth/global_scale = 1.0)
        """
        rgb_arr, _, seg_arr, _ = self._cam.render(
            rgb=True, depth=False, segmentation=True)

        rgb_np = np.asarray(rgb_arr, dtype=np.uint8)
        seg_np = np.asarray(seg_arr, dtype=np.int64)

        is_mat = self._material_mask(seg_np)

        obs = np.zeros((self.screenHeight, self.screenWidth, 5), dtype=np.float32)
        obs[..., :3] = rgb_np[..., :3].astype(np.float32) / 255.0
        obs[..., 3]  = is_mat.astype(np.float32)
        obs[..., 4]  = np.where(is_mat, self._cam_h, 2.0 * self._cam_h)
        return obs

    def step(self, action: np.ndarray, video_recorder=None) -> np.ndarray:
        """
        Execute a push action and return the next observation.

        Parameters
        ----------
        action : (4,) array-like
            [sx, sy, ex, ey] in world-frame metres (x/y horizontal plane).
        video_recorder : list[cv2.VideoWriter] | None
            If provided, three RGB frames are written as BGR uint8 images:
            right after the plate reaches the push start (about to sweep),
            right after the sweep ends (about to lift), and the final
            post-action state — so the action taken is visible in the video,
            not just the before/after box state.

        Returns
        -------
        obs : (H, W, 5) float32  — same as ``render()``.
        """
        import genesis as gs  # already initialised inside SandboxManipulation

        sx, sy, ex, ey = (float(action[0]), float(action[1]),
                          float(action[2]), float(action[3]))
        dxy   = math.hypot(ex - sx, ey - sy)
        # Plate yaw must be PERPENDICULAR to the direction of travel so the
        # 4 cm face (plate size[0]) sweeps material.  atan2 gives the travel
        # direction; adding π/2 rotates the plate 90° so its long axis is
        # orthogonal to motion (like a bulldozer blade).
        # Using the parallel angle (= travel direction) would push with the
        # 2 mm edge (size[1]) instead, moving virtually no material.
        angle = math.atan2(ey - sy, ex - sx) + math.pi / 2 if dxy > 1e-6 else 0.0

        z_op    = float(self._sim._operation_height)
        p_start = torch.tensor([[sx, sy, z_op]], dtype=torch.float32,
                                device=gs.device)
        p_stop  = torch.tensor([[ex, ey, z_op]], dtype=torch.float32,
                                device=gs.device)
        angle_t = torch.tensor([angle],          dtype=torch.float32,
                                device=gs.device)

        def _on_phase(phase: str) -> None:
            if video_recorder is not None and phase in ('post_lower', 'post_sweep'):
                write_video_frame(self.render(), video_recorder)

        self._sim.execute_action(p_start, p_stop, angle_t, on_phase=_on_phase)
        self._sim._settle_steps = self.settle_steps   # honour per-experiment override
        self._sim.update_material_state()
        obs = self.render()
        write_video_frame(obs, video_recorder)

        return obs

    def reset(self) -> None:
        """Shuffle particles to a new random configuration."""
        # Use reset_warmup_steps for the post-shuffle settle (particles need
        # time to fall under gravity from their newly placed positions).
        self._sim._settle_steps = self.reset_warmup_steps
        self._sim.shuffle_particles()
        self._sim.update_material_state()
        # Restore the per-step settle budget for subsequent step() calls.
        self._sim._settle_steps = self.settle_steps
        return None

    def set_active_particles(self, n: int) -> None:
        """Set the number of particles placed inside the box on the next reset().

        Particles above this count are parked outside camera view.  The change
        takes effect on the next ``reset()`` call.  ``n`` must be ≤ the number
        of particles the sim was built with.
        """
        n_total = len(self._sim.material)
        n = max(0, min(n, n_total))
        self._sim.set_n_active(n)

    def set_save_render_mode(self) -> None:
        """No-op for Genesis: camera resolution is fixed at construction time."""
        pass

    def restore_native_render_mode(self) -> None:
        """No-op for Genesis: camera resolution is fixed at construction time."""
        pass

    def set_positions(self, positions) -> None:
        """
        TBD: direct particle placement is not yet implemented for GenesisEnv.

        In the old FlexEnv, this used pyflex.set_positions().  A Genesis
        equivalent would set rigid-solver link positions directly.
        """
        raise NotImplementedError(
            "GenesisEnv.set_positions() is TBD — needs explicit particle "
            "placement via Genesis rigid solver (link_start / set_pos)."
        )

    # ------------------------------------------------------------------ #
    #  Camera intrinsics / extrinsics                                      #
    # ------------------------------------------------------------------ #

    def get_cam_params(self) -> list:
        """Return ``[fx, fy, cx, cy]`` (pixels) from Genesis camera intrinsics."""
        f  = float(self._cam.f)
        cx = float(self._cam.cx)
        cy = float(self._cam.cy)
        return [f, f, cx, cy]

    def get_cam_extrinsics(self) -> np.ndarray:
        """
        Return identity 4×4 matrix.

        GenesisEnv actions are converted to camera-space via
        ``_action_to_cam_3d_genesis`` in ``eulerian_wrapper.py``, which does not
        require an extrinsic matrix.  The identity is returned here so that any
        code that blindly uses it will at least not crash.
        """
        return np.eye(4, dtype=np.float64)
