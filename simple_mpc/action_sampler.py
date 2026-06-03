"""
Action sampling strategies for MPC candidate generation.

Different sampling strategies (random, learned priors, importance sampling, etc.)
can be implemented as subclasses of ActionSampler and plugged into run_simple_mpc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import cv2
import torch


class ActionSampler(ABC):
    """
    Base class for MPC action sampling strategies.

    Each MPC step samples n_sample candidate action sequences of length n_ahead.
    Different samplers can implement different strategies (random, learned,
    importance-weighted, etc.).
    """

    @abstractmethod
    def sample(
        self,
        n_sample: int,
        n_ahead: int,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate a batch of action sequences.

        Parameters
        ----------
        n_sample : int
            Number of independent candidates to sample
        n_ahead : int
            Planning horizon (rollout length)
        act_lo : np.ndarray (4,)
            Lower bounds for actions [sx, sy, ex, ey]
        act_hi : np.ndarray (4,)
            Upper bounds for actions [sx, sy, ex, ey]
        device : str
            PyTorch device ('cuda' or 'cpu')

        Returns
        -------
        act_seqs : torch.Tensor (n_sample, n_ahead, 4)
            Sampled action sequences, requires_grad=True for optimization
        """
        pass


class RandomUniformSampler(ActionSampler):
    """
    Sample actions uniformly at random from the workspace bounds.

    This is the default / baseline strategy: each action component is drawn
    independently from the specified bounds. Fast and simple.
    """

    def sample(
        self,
        n_sample: int,
        n_ahead: int,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate uniformly random action sequences.

        Each of the n_sample candidates independently samples each action
        component uniformly from [act_lo, act_hi].
        """
        act_np = np.random.uniform(
            act_lo, act_hi, (n_sample, n_ahead, 4)
        ).astype(np.float32)
        act_seqs = torch.tensor(act_np, device=device, requires_grad=True)
        return act_seqs


class PhysicsAwareActionSampler(ActionSampler):
    """
    Sample actions so the pusher plate fits entirely inside the workspace box
    at both the start and stop positions, for *any* travel direction.

    In ``GenesisEnv.step`` the plate yaw is derived as::

        yaw = atan2(ey - sy, ex - sx) + π/2

    so the plate is always perpendicular to its travel direction.  For a plate
    of size ``[L, W, H]`` the axis-aligned bounding box (AABB) at yaw α is::

        AABB_x = |cos α| * L/2 + |sin α| * W/2  ≤  L/2
        AABB_y = |sin α| * L/2 + |cos α| * W/2  ≤  L/2

    The maximum over all orientations is ``L/2`` in either axis.  Using the
    conservative half-range::

        v = workspace_half  −  L/2  −  safety_margin

    for all four action coordinates guarantees the plate stays inside the box
    regardless of which direction the optimiser pushes the action during GD.
    """

    def __init__(
        self,
        plate_length: float  = 0.04,
        plate_width:  float  = 0.002,
        safety_margin: float = 0.01,
    ):
        self.plate_length  = plate_length
        self.plate_width   = plate_width
        self.safety_margin = safety_margin

    def sample(
        self,
        n_sample: int,
        n_ahead: int,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Sample start and stop positions within the plate-aware valid region.

        ``act_lo`` / ``act_hi`` are the raw workspace bounds; the plate margin
        is subtracted internally so the caller does not need to adjust them.
        """
        # Workspace centre and half-widths (from start-coord bounds dims 0, 1)
        half_x = (act_hi[0] - act_lo[0]) / 2.0
        half_y = (act_hi[1] - act_lo[1]) / 2.0
        cx     = (act_hi[0] + act_lo[0]) / 2.0
        cy     = (act_hi[1] + act_lo[1]) / 2.0

        # Conservative valid half-range: plate fits for ANY travel direction
        vx = max(half_x - self.plate_length / 2.0 - self.safety_margin, 0.0)
        vy = max(half_y - self.plate_length / 2.0 - self.safety_margin, 0.0)

        # Sample (sx, sy, ex, ey) independently within [-v, v] + centre
        rng = np.random.uniform(size=(n_sample, n_ahead, 4)).astype(np.float32)
        sx = cx + rng[:, :, 0] * (2.0 * vx) - vx
        sy = cy + rng[:, :, 1] * (2.0 * vy) - vy
        ex = cx + rng[:, :, 2] * (2.0 * vx) - vx
        ey = cy + rng[:, :, 3] * (2.0 * vy) - vy

        acts = np.stack([sx, sy, ex, ey], axis=-1)   # (n_sample, n_ahead, 4)
        return torch.tensor(acts, device=device, requires_grad=True)


# ---------------------------------------------------------------------------
# Plate collision checking helpers
# ---------------------------------------------------------------------------

def _make_plate_kernel(L_px: int, W_px: int, theta_rad: float) -> np.ndarray:
    """Return a binary uint8 kernel shaped as the filled plate footprint at yaw theta_rad."""
    theta_deg = np.degrees(theta_rad)
    c, s = abs(np.cos(theta_rad)), abs(np.sin(theta_rad))
    # Exact AABB of the rotated rectangle (+3 px for sub-pixel rounding; ensure odd)
    bbox_h = int(np.ceil(L_px * s + W_px * c)) + 3
    bbox_w = int(np.ceil(L_px * c + W_px * s)) + 3
    bbox_h += (bbox_h % 2 == 0)
    bbox_w += (bbox_w % 2 == 0)
    center = (bbox_w // 2, bbox_h // 2)
    kernel = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
    pts = np.int32(cv2.boxPoints((center, (L_px, W_px), theta_deg)))
    cv2.fillPoly(kernel, [pts], 1)
    return kernel


class PlateCollisionChecker:
    """Pre-computed plate-shaped dilation kernels for multiple push orientations.

    Maintains per-orientation valid-start masks so that both
    ``CollisionAwareActionSampler`` and ``OTGuidedActionSampler`` can guarantee
    that sampled push-start positions are free of material.

    Call ``update(occupancy_grid)`` once per MPC step after observing the new
    state.  Then call ``get_valid_pts(k)`` to retrieve (col, row) arrays of
    valid plate-centre positions for orientation index k.
    """

    def __init__(
        self,
        grid_size       : int,
        wkspc_w         : float,
        plate_length_m  : float = 0.04,
        plate_width_m   : float = 0.002,
        n_angles        : int   = 8,
        safety_margin_m : float = 0.01,
    ):
        self._n     = grid_size
        self._n_ang = n_angles
        self._scale = (2.0 * wkspc_w) / grid_size      # metres per pixel
        ppm         = 1.0 / self._scale                 # pixels per metre

        L_px = int(round(plate_length_m * ppm))
        W_px = max(1, int(round(plate_width_m * ppm)))

        # Plate-footprint kernels — one per orientation
        self._kernels: list[np.ndarray] = [
            _make_plate_kernel(L_px, W_px, k * np.pi / n_angles)
            for k in range(n_angles)
        ]

        # Per-orientation workspace masks (valid plate-centre region, no material)
        # Uses exact per-angle AABB from sandbox_manipulation_clean.py
        half_x  = wkspc_w
        half_y  = wkspc_w
        safety_px = safety_margin_m * ppm
        L_half    = plate_length_m / 2.0 * ppm
        W_half    = plate_width_m  / 2.0 * ppm
        cx = cy   = grid_size / 2.0

        cols_g = np.arange(grid_size, dtype=float)
        rows_g = np.arange(grid_size, dtype=float)
        C, R   = np.meshgrid(cols_g, rows_g)           # (H, W)

        self._workspace_masks: list[np.ndarray] = []
        for k in range(n_angles):
            theta_k = k * np.pi / n_angles
            ck = abs(np.cos(theta_k))
            sk = abs(np.sin(theta_k))
            vhx = max(0.0, half_x / self._scale - (ck * L_half + sk * W_half + safety_px))
            vhy = max(0.0, half_y / self._scale - (sk * L_half + ck * W_half + safety_px))
            mask = (C >= cx - vhx) & (C < cx + vhx) & (R >= cy - vhy) & (R < cy + vhy)
            self._workspace_masks.append(mask.astype(bool))

        # Cache workspace-only pts (fallback when grid is entirely clear)
        self._workspace_pts: list[np.ndarray] = []
        for k in range(n_angles):
            rows_w, cols_w = np.where(self._workspace_masks[k])
            self._workspace_pts.append(
                np.stack([cols_w, rows_w], axis=1).astype(np.int32)
            )

        # valid_pts[k] populated by update()
        self._valid_pts: list[np.ndarray] = list(self._workspace_pts)  # shallow copy

    def update(self, occupancy_grid: np.ndarray) -> None:
        """Recompute valid_pts[k] from the current (H, W) binary occupancy grid."""
        occ = (occupancy_grid > 0).astype(np.uint8)
        for k in range(self._n_ang):
            dilated          = cv2.dilate(occ, self._kernels[k])
            valid            = (dilated == 0) & self._workspace_masks[k]
            rows_v, cols_v   = np.where(valid)
            if len(rows_v) > 0:
                self._valid_pts[k] = np.stack([cols_v, rows_v], axis=1).astype(np.int32)
            else:
                # No collision-free positions; fall back to workspace bounds
                self._valid_pts[k] = self._workspace_pts[k]

    def get_valid_pts(self, k: int) -> np.ndarray:
        """Return (M, 2) array of valid (col, row) start positions for orientation k."""
        return self._valid_pts[k]

    @property
    def n_angles(self) -> int:
        return self._n_ang

    @property
    def scale(self) -> float:
        """Metres per pixel."""
        return self._scale


# ---------------------------------------------------------------------------
# Collision-aware sampler
# ---------------------------------------------------------------------------

class CollisionAwareActionSampler(ActionSampler):
    """Sample push actions whose start positions are free of material.

    Uses morphological dilation to compute forbidden-start zones per plate
    orientation, then samples starts from the valid complement.  Push directions
    are assigned round-robin across ``n_angles`` orientations.

    Call ``update_state(source_grid)`` once per MPC step (before ``sample``).
    """

    def __init__(
        self,
        grid_size     : int,
        wkspc_w       : float,
        plate_length  : float = 0.04,
        plate_width   : float = 0.002,
        safety_margin : float = 0.01,
        n_angles      : int   = 8,
        d_min         : float = None,
        d_max         : float = None,
    ):
        self._grid_size = grid_size
        self._wkspc_w   = wkspc_w
        self._n_angles  = n_angles
        self._checker   = PlateCollisionChecker(
            grid_size, wkspc_w, plate_length, plate_width, n_angles, safety_margin
        )
        self._d_min = d_min if d_min is not None else plate_length / 2.0
        self._d_max = d_max if d_max is not None else 2.0 * np.sqrt(2.0) * wkspc_w

        # Per-orientation workspace half-widths for end-position clipping
        L_half = plate_length / 2.0
        W_half = plate_width  / 2.0
        self._phy_vx = np.zeros(n_angles)
        self._phy_vy = np.zeros(n_angles)
        for k in range(n_angles):
            theta_k = k * np.pi / n_angles
            ck = abs(np.cos(theta_k))
            sk = abs(np.sin(theta_k))
            self._phy_vx[k] = max(0.0, wkspc_w - (ck * L_half + sk * W_half + safety_margin))
            self._phy_vy[k] = max(0.0, wkspc_w - (sk * L_half + ck * W_half + safety_margin))

    def update_state(self, source_grid: np.ndarray, goal_grid: np.ndarray = None) -> None:
        """Update valid-start masks from the current occupancy grid."""
        self._checker.update(source_grid)

    def sample(
        self,
        n_sample : int,
        n_ahead  : int,
        act_lo   : np.ndarray,
        act_hi   : np.ndarray,
        device   : str = 'cuda',
    ) -> torch.Tensor:
        n     = self._grid_size
        scale = self._checker.scale

        k_assign = np.arange(n_sample) % self._n_angles
        acts     = np.zeros((n_sample, n_ahead, 4), dtype=np.float32)

        for k in range(self._n_angles):
            bucket = np.where(k_assign == k)[0]
            if len(bucket) == 0:
                continue
            valid_pts = self._checker.get_valid_pts(k)   # (M_k, 2) col/row

            theta_k      = k * np.pi / self._n_angles
            travel_angle = theta_k - np.pi / 2.0
            tvec         = np.array([np.cos(travel_angle), np.sin(travel_angle)])

            # Sample start positions (same mask for all look-ahead steps)
            idxs = np.random.randint(0, len(valid_pts), size=(len(bucket), n_ahead))
            sel  = valid_pts[idxs]                        # (b, n_ahead, 2)
            sx   = (sel[..., 0] - n / 2.0) * scale       # world_x
            sy   = (sel[..., 1] - n / 2.0) * scale       # world_y

            # Push distance → end position
            d  = np.random.uniform(self._d_min, self._d_max, size=(len(bucket), n_ahead))
            vx = self._phy_vx[k]
            vy = self._phy_vy[k]
            ex = np.clip(sx + d * tvec[0], -vx, vx)
            ey = np.clip(sy + d * tvec[1], -vy, vy)

            acts[bucket] = np.stack([sx, sy, ex, ey], axis=-1)

        return torch.tensor(acts, device=device, requires_grad=True)


# ---------------------------------------------------------------------------
# OT-guided sampler
# ---------------------------------------------------------------------------

class OTGuidedActionSampler(ActionSampler):
    """Seeds MPC candidates from OT coherent-flow regions + collision-aware random fill.

    A fraction ``ot_fraction`` of the candidate pool is seeded from push actions
    aligned to OT displacement vectors in low-divergence (coherent) regions.
    The remainder uses ``PlateCollisionChecker`` orientation-stratified sampling
    to guarantee collision-free starts.

    Call ``update_state(source_grid, goal_grid)`` once per MPC step before
    calling ``sample()``.
    """

    def __init__(
        self,
        grid_size     : int,
        wkspc_w       : float,
        reg           : float = 0.002,
        ot_fraction   : float = 0.7,
        noise_std_m   : float = 0.005,
        plate_length  : float = 0.04,
        plate_width   : float = 0.002,
        safety_margin : float = 0.01,
        div_percentile: float = 30.0,
        n_ot_seeds    : int   = 8,
        n_angles      : int   = 8,
    ):
        from simple_mpc.ot_planner import OTPlannerSparse
        self._planner       = OTPlannerSparse(grid_size, reg=reg)
        self._checker       = PlateCollisionChecker(
            grid_size, wkspc_w, plate_length, plate_width, n_angles, safety_margin
        )
        self._grid_size     = grid_size
        self._wkspc_w       = wkspc_w
        self._ot_fraction   = ot_fraction
        self._noise_std_m   = noise_std_m
        self._div_percentile = div_percentile
        self._n_ot_seeds    = n_ot_seeds
        self._n_angles      = n_angles
        self._candidates: list = []
        self._d_min = plate_length / 2.0
        self._d_max = 2.0 * np.sqrt(2.0) * wkspc_w

        # Per-orientation workspace half-widths for end-position clipping
        L_half = plate_length / 2.0
        W_half = plate_width  / 2.0
        self._phy_vx = np.zeros(n_angles)
        self._phy_vy = np.zeros(n_angles)
        for k in range(n_angles):
            theta_k = k * np.pi / n_angles
            ck = abs(np.cos(theta_k))
            sk = abs(np.sin(theta_k))
            self._phy_vx[k] = max(0.0, wkspc_w - (ck * L_half + sk * W_half + safety_margin))
            self._phy_vy[k] = max(0.0, wkspc_w - (sk * L_half + ck * W_half + safety_margin))

    def update_state(self, source_grid: np.ndarray, goal_grid: np.ndarray) -> None:
        """Re-solve the OT plan and refresh collision masks."""
        self._checker.update(source_grid)
        result = self._planner.solve(source_grid, goal_grid)
        self._candidates = self._planner.extract_push_candidates(
            result, self._wkspc_w,
            n_candidates   = self._n_ot_seeds,
            div_percentile = self._div_percentile,
        )

    def sample(
        self,
        n_sample : int,
        n_ahead  : int,
        act_lo   : np.ndarray,
        act_hi   : np.ndarray,
        device   : str = 'cuda',
    ) -> torch.Tensor:
        n_ot   = int(n_sample * self._ot_fraction) if self._candidates else 0
        n_rand = n_sample - n_ot
        acts   = np.zeros((n_sample, n_ahead, 4), dtype=np.float32)

        # ---- OT-guided candidates -----------------------------------------
        if n_ot > 0:
            n_cands = len(self._candidates)
            for i in range(n_ot):
                cand = self._candidates[i % n_cands]
                sw   = cand['start_world'] + np.random.randn(2) * self._noise_std_m
                ew   = cand['end_world']   + np.random.randn(2) * self._noise_std_m
                lim  = self._wkspc_w
                sw   = np.clip(sw, -lim, lim)
                ew   = np.clip(ew, -lim, lim)
                acts[i, 0, :] = [sw[0], sw[1], ew[0], ew[1]]
                if n_ahead > 1:
                    acts[i, 1:, :] = self._sample_rand_acts(1, n_ahead - 1)[0]

        # ---- Collision-aware random fill -----------------------------------
        if n_rand > 0:
            acts[n_ot:] = self._sample_rand_acts(n_rand, n_ahead)

        return torch.tensor(acts, device=device, requires_grad=True)

    def _sample_rand_acts(self, n: int, n_ahead: int) -> np.ndarray:
        """Orientation-stratified collision-aware random actions."""
        scale    = self._checker.scale
        grid_n   = self._grid_size
        k_assign = np.arange(n) % self._n_angles
        out      = np.zeros((n, n_ahead, 4), dtype=np.float32)

        for k in range(self._n_angles):
            bucket = np.where(k_assign == k)[0]
            if len(bucket) == 0:
                continue
            valid_pts    = self._checker.get_valid_pts(k)
            theta_k      = k * np.pi / self._n_angles
            travel_angle = theta_k - np.pi / 2.0
            tvec         = np.array([np.cos(travel_angle), np.sin(travel_angle)])

            idxs = np.random.randint(0, len(valid_pts), size=(len(bucket), n_ahead))
            sel  = valid_pts[idxs]
            sx   = (sel[..., 0] - grid_n / 2.0) * scale
            sy   = (sel[..., 1] - grid_n / 2.0) * scale
            d    = np.random.uniform(self._d_min, self._d_max, size=(len(bucket), n_ahead))
            vx   = self._phy_vx[k]
            vy   = self._phy_vy[k]
            ex   = np.clip(sx + d * tvec[0], -vx, vx)
            ey   = np.clip(sy + d * tvec[1], -vy, vy)
            out[bucket] = np.stack([sx, sy, ex, ey], axis=-1)

        return out


# Factory function for easy selection
def make_action_sampler(sampler_type: str = 'physics_aware', **kwargs) -> ActionSampler:
    """
    Create an action sampler by name.

    Parameters
    ----------
    sampler_type : str
        Name of the sampler: ``'physics_aware'`` (default) or ``'uniform'``.
    **kwargs
        Extra keyword arguments forwarded to the sampler constructor.
        ``PhysicsAwareActionSampler`` accepts ``plate_length``,
        ``plate_width``, and ``safety_margin``.

    Returns
    -------
    sampler : ActionSampler
        Instantiated sampler ready for use

    Examples
    --------
    >>> sampler = make_action_sampler('physics_aware', plate_length=0.04, safety_margin=0.01)
    >>> acts = sampler.sample(n_sample=512, n_ahead=1, ...)
    """
    if sampler_type == 'uniform':
        return RandomUniformSampler()
    elif sampler_type == 'physics_aware':
        valid = {'plate_length', 'plate_width', 'safety_margin'}
        return PhysicsAwareActionSampler(
            **{k: v for k, v in kwargs.items() if k in valid}
        )
    elif sampler_type == 'collision_aware':
        valid = {'grid_size', 'wkspc_w', 'plate_length', 'plate_width',
                 'safety_margin', 'n_angles', 'd_min', 'd_max'}
        return CollisionAwareActionSampler(
            **{k: v for k, v in kwargs.items() if k in valid}
        )
    elif sampler_type == 'ot_guided':
        valid = {
            'grid_size', 'wkspc_w', 'reg', 'ot_fraction', 'noise_std_m',
            'plate_length', 'plate_width', 'safety_margin',
            'div_percentile', 'n_ot_seeds', 'n_angles',
        }
        return OTGuidedActionSampler(
            **{k: v for k, v in kwargs.items() if k in valid}
        )
    else:
        raise ValueError(
            f"Unknown sampler '{sampler_type}'. "
            f"Available: {sorted(['uniform', 'physics_aware', 'collision_aware', 'ot_guided'])}"
        )
