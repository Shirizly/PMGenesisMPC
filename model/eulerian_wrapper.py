"""
EulerianModelWrapper
====================
Adapts a user-supplied Eulerian (occupancy-field) dynamics model so that it can
be dropped into the existing particle-based MPC pipeline in place of
PropNetDiffDenModel.

Pipeline position
-----------------
The MPC planner calls the model inside ``PlannerGD.ptcl_model_rollout``, which
will dispatch to this wrapper whenever ``isinstance(model_dy, EulerianModelWrapper)``
is true.  The planner passes the raw 4-D action ``[sx, sy, ex, ey]`` directly to
``predict_one_step``, so the wrapper never has to reconstruct it from s_delta.

User model contract
-------------------
The model passed as ``user_model`` must implement::

    forward(occ_grid, action_start, action_end) -> occ_grid_pred

where:
    occ_grid      : torch.Tensor  (B, *grid_shape)  – occupancy field [0..1]
    action_start  : torch.Tensor  (B, 3)            – tool start in *grid coordinates*
                                                       (voxel indices, float)
    action_end    : torch.Tensor  (B, 3)            – tool end   in *grid coordinates*
    returns:
    occ_grid_pred : torch.Tensor  (B, *grid_shape)  – predicted occupancy field [0..1]

Grid coordinates are defined as (ix, iy, iz) in [0, grid_res-1] along each axis,
mapping linearly from the supplied ``grid_bounds``.  For a 2-D grid supply a
``grid_res`` of length 2; the wrapper will operate in the x-y plane of normalized
camera space (which corresponds to the horizontal table plane for a top-down camera).

Coordinate conventions
-----------------------
*Normalized camera space* – the coordinate system used for particle positions
throughout the existing code – is obtained by:

1. Unprojecting each depth pixel with the pinhole camera model (``depth2fgpcd``).
2. Dividing by ``global_scale``.

The x-z axes span the horizontal plane (the table surface); y points upward.
For the default top-down camera the table is at approximately
``z_cam ≈ 0.745`` in these units.

The 4-D MPC action ``[sx, sy, ex, ey]`` lives in *world 2-D* coordinates
(the horizontal plane, range ≈ [-wkspc_w, wkspc_w]).  The wrapper converts it
to normalized camera 3-D using the same ``world2cam`` logic as the planner's
``gen_s_delta``.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical implementations live in transforms.functional; the aliases keep
# this module's historical private names working for existing callers.
from transforms.functional import (
    particles_to_occupancy as _particles_to_occupancy,
    get_grid_axes as _get_axes,
    grid_axis_indices as _axis_indices,
)


def _occupancy_to_particles(
    occ: torch.Tensor,         # (B, *grid_res)  float, in [0, 1]
    n_particles: int,
    grid_bounds: Dict[str, float],
    grid_res: Tuple[int, ...],
    thresh: float = 0.5,
) -> torch.Tensor:
    """
    Convert occupancy grids back to a fixed-size particle set via FPS.

    Steps
    -----
    1. Threshold at ``thresh`` to get occupied voxel centres.
    2. Farthest-point-sample ``n_particles`` from those centres.
       If fewer occupied voxels than n_particles, repeat voxel centres.
    3. Return particles in normalised camera coordinates.

    Returns
    -------
    s_pred : (B, n_particles, 3) float32 – in normalized camera coords.
             The 'missing' axis (y for 2-D grids) is set to the mid-point of
             its bound.
    """
    B = occ.shape[0]
    ndim = len(grid_res)
    device = occ.device
    dtype = occ.dtype

    axes = _get_axes(ndim)
    lo  = np.array([grid_bounds[f'{a}_min'] for a in axes])
    hi  = np.array([grid_bounds[f'{a}_max'] for a in axes])
    res = np.array(grid_res)
    voxel_size = (hi - lo) / (res - 1)  # (ndim,)

    # Depth (z) fill value for 2-D grids: all particles sit at ~constant table depth.
    if ndim == 2:
        z_mid = 0.5 * (grid_bounds.get('z_min', 0.745) + grid_bounds.get('z_max', 0.745))

    s_pred = torch.zeros(B, n_particles, 3, device=device, dtype=dtype)

    for b in range(B):
        occ_b = occ[b]  # (*grid_res)

        # Voxel indices of occupied cells
        occ_np = occ_b.detach().cpu().numpy()
        mask = occ_np >= thresh
        occupied_idx = np.argwhere(mask)  # (M, ndim)

        if occupied_idx.shape[0] == 0:
            # Fallback: use all voxel centres
            occupied_idx = np.argwhere(np.ones_like(occ_np, dtype=bool))

        # Convert voxel indices → world coordinates
        voxel_coords = lo + occupied_idx * voxel_size  # (M, ndim)

        # Reconstruct 3-D from 2-D / 3-D grid
        pts_3d = _to_3d(voxel_coords, axes, z_mid if ndim == 2 else None)  # (M, 3)

        # FPS to n_particles
        pts_3d = _fps_np(pts_3d, n_particles)  # (n_particles, 3)

        s_pred[b] = torch.from_numpy(pts_3d).to(device=device, dtype=dtype)

    return s_pred


def _action_to_cam_3d(
    action: torch.Tensor,      # (B, 4) [sx, sy, ex, ey] in world 2-D
    cam_extrinsic: np.ndarray, # (4, 4) OpenGL view matrix
    global_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a 4-D planar action to 3-D normalized camera-space start/end.

    Mirrors the logic of ``PlannerGD.world2cam`` and ``gen_s_delta``.

    Returns
    -------
    s_3d_cam, e_3d_cam : each (B, 3)
    """
    device = action.device
    dtype  = action.dtype
    B = action.shape[0]

    h = torch.zeros(B, 1, device=device, dtype=dtype)  # height = 0 (table plane)

    s_world = torch.cat([action[:, 0:1], h, -action[:, 1:2]], dim=1)  # (B, 3)
    e_world = torch.cat([action[:, 2:3], h, -action[:, 3:4]], dim=1)  # (B, 3)

    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0,-1, 0, 0],
                                [0, 0,-1, 0],
                                [0, 0, 0, 1]], dtype=np.float64)
    opencv_T_world     = np.matmul(np.linalg.inv(cam_extrinsic), opencv_T_opengl)
    opencv_T_world_inv = np.linalg.inv(opencv_T_world)
    M = torch.tensor(opencv_T_world_inv, device=device, dtype=dtype)  # (4, 4)

    def _transform(pts):
        ones = torch.ones(B, 1, device=device, dtype=dtype)
        homog = torch.cat([pts, ones], dim=1)  # (B, 4)
        out = (M @ homog.T).T                  # (B, 4)
        return out[:, :3] / global_scale

    return _transform(s_world), _transform(e_world)


def _action_to_cam_3d_genesis(
    action: torch.Tensor,   # (B, 4) [sx, sy, ex, ey] world metres
    global_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a 4-D planar action to 3-D normalised camera-space start/end
    for the Genesis overhead camera convention.

    Camera is at (0, 0, cam_h) looking straight down with up=(0,1,0):
      camera-X = world-X,  camera-Y = −world-Y.
    Normalised coords divide by global_scale (= 2*cam_h), so the table
    plane maps to z_norm = cam_h / global_scale = 0.5.

    Returns
    -------
    s_3d_cam, e_3d_cam : each (B, 3)  in normalised camera space.
    """
    gs = global_scale
    sx, sy = action[:, 0:1] / gs, action[:, 1:2] / gs   # normalised world x, y
    ex, ey = action[:, 2:3] / gs, action[:, 3:4] / gs
    half   = torch.full_like(sx, 0.5)                   # z_norm = 0.5 = table
    # camera-Y = −world-Y
    s_3d_cam = torch.cat([ sx, -sy, half], dim=1)       # (B, 3)
    e_3d_cam = torch.cat([ ex, -ey, half], dim=1)       # (B, 3)
    return s_3d_cam, e_3d_cam


# ----- small numpy FPS (avoids importing the utils module) -----------------

def _fps_np(pts: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point sample *n* points from pts (M, d).  Repeats if M < n."""
    M = pts.shape[0]
    if M == 0:
        return np.zeros((n, pts.shape[1]), dtype=pts.dtype)
    if M <= n:
        reps = int(np.ceil(n / M))
        pts = np.tile(pts, (reps, 1))
    rand_idx = np.random.randint(pts.shape[0])
    selected = [pts[rand_idx]]
    dist = np.linalg.norm(pts - selected[0], axis=1)
    while len(selected) < n:
        farthest = pts[dist.argmax()]
        selected.append(farthest)
        dist = np.minimum(dist, np.linalg.norm(pts - farthest, axis=1))
    return np.stack(selected[:n])


# ----- coordinate bookkeeping helpers --------------------------------------

def _to_3d(voxel_coords: np.ndarray, axes, depth_fill: float | None) -> np.ndarray:
    """Promote a (M, ndim) grid-coordinate array to (M, 3).

    For 2-D grids ``depth_fill`` is written into the depth axis (dim 2, camera z)
    so that returned points can be used as normalized camera-space coordinates.
    """
    M = voxel_coords.shape[0]
    pts = np.zeros((M, 3), dtype=voxel_coords.dtype)
    for k, a in enumerate(axes):
        idx = {'x': 0, 'y': 1, 'z': 2}[a]
        pts[:, idx] = voxel_coords[:, k]
    if len(axes) == 2 and depth_fill is not None:
        pts[:, 2] = depth_fill   # fill the depth (camera z) axis
    return pts


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class EulerianModelWrapper(nn.Module):
    """
    Wraps a user-supplied Eulerian dynamics model for the particle-based MPC
    pipeline.

    Parameters
    ----------
    user_model : nn.Module
        Must implement ``forward(occ_grid, action_start, action_end) -> occ_grid_pred``
        as documented at the top of this file.
    grid_bounds : dict
        World-extent of the occupancy grid in *normalized camera coordinates*.
        Required keys for a 2-D grid: ``x_min, x_max, z_min, z_max``.
        Also requires ``y_min, y_max`` (used only to reconstruct the y coordinate
        when converting back to 3-D particles).
        For a 3-D grid all six keys must be present.
        Use ``EulerianModelWrapper.default_bounds(config)`` to get a sensible
        default derived from the environment config.
    grid_res : tuple[int, ...]
        Number of voxels along each grid axis, e.g. ``(64, 64)`` (2-D) or
        ``(32, 32, 16)`` (3-D).
    cam_extrinsic : np.ndarray  (4, 4)
        OpenGL view matrix from ``env.get_cam_extrinsics()``.
    global_scale : float
        From ``config['dataset']['global_scale']``.
    splat_sigma : float, optional
        If > 0, use a Gaussian splat when converting particles → occupancy
        (differentiable but slower).  Default 0 → hard voxel assignment.
    occ_threshold : float, optional
        Occupancy threshold used when converting the predicted grid back to
        particles.  Default 0.5.
    """

    def __init__(
        self,
        user_model: nn.Module,
        grid_bounds: Dict[str, float],
        grid_res: Sequence[int],
        cam_extrinsic: np.ndarray,
        global_scale: float,
        splat_sigma: float = 0.0,
        occ_threshold: float = 0.5,
        action_convention: str = 'flex',
    ):
        super().__init__()
        self.user_model       = user_model
        self.grid_bounds      = dict(grid_bounds)
        self.grid_res         = tuple(grid_res)
        self.cam_extrinsic    = cam_extrinsic.copy() if cam_extrinsic is not None else None
        self.global_scale     = float(global_scale)
        self.splat_sigma      = splat_sigma
        self.occ_threshold    = occ_threshold
        self.action_convention = action_convention   # 'flex' or 'genesis'

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    @staticmethod
    def default_bounds(config: dict, convention: str = 'flex') -> Dict[str, float]:
        """
        Return grid bounds that cover the workspace in normalised camera coords.

        Parameters
        ----------
        config : full MPC config dict.
        convention : 'flex' (PyFleX/OpenGL, y-up) or 'genesis' (overhead camera,
                     z-up).  Controls the z_table depth value — for Genesis the
                     table plane normalises to z_norm = 0.5.
        """
        gs_val = config['dataset']['global_scale']
        w      = config['dataset']['wkspc_w']   # world units
        w_n    = w / gs_val                      # normalised

        if convention == 'genesis':
            # Overhead camera at cam_h, global_scale = 2*cam_h
            # → table (cam_h below camera) normalises to 0.5
            z_table = 0.5
        else:
            # PyFleX / FlexEnv: cam_height ≈ 6/8 of global_scale
            z_table = 6.0 / 8.0

        z_margin = 0.05
        return {
            'x_min': -w_n * 1.2, 'x_max':  w_n * 1.2,
            'y_min': -w_n * 1.2, 'y_max':  w_n * 1.2,
            'z_min': z_table - z_margin, 'z_max': z_table + z_margin,
        }

    # ------------------------------------------------------------------
    # Core interface – called by ptcl_model_rollout
    # ------------------------------------------------------------------

    def predict_one_step(
        self,
        s_cur: torch.Tensor,    # (B, N, 3)  particles in normalized cam coords
        action: torch.Tensor,   # (B, 4)     [sx, sy, ex, ey] in world 2-D
    ) -> torch.Tensor:
        """
        Predict the next particle state using the Eulerian model.

        Steps
        -----
        1. Convert particles to an occupancy grid  (B, *grid_res).
        2. Convert raw action [sx, sy, ex, ey] to 3-D camera-space start/end,
           then to grid-coordinate start/end (B, 3).
        3. Run ``user_model(occ, start_grid, end_grid)`` → predicted occupancy.
        4. Convert the predicted occupancy back to (B, N, 3) particles.

        Returns
        -------
        s_pred : (B, N, 3)  predicted particle positions in normalized cam coords.
        """
        B, N, _ = s_cur.shape
        occ = _particles_to_occupancy(
            s_cur, self.grid_bounds, self.grid_res, sigma=self.splat_sigma)
        occ_pred = self.predict_one_step_occ(occ, action)
        return _occupancy_to_particles(
            occ_pred, n_particles=N,
            grid_bounds=self.grid_bounds, grid_res=self.grid_res,
            thresh=self.occ_threshold)

    def predict_one_step_occ(
        self,
        occ_cur: torch.Tensor,   # (B, *grid_res)  current occupancy field
        action:  torch.Tensor,   # (B, 4)           [sx, sy, ex, ey] world 2-D
    ) -> torch.Tensor:
        """
        Single prediction step that stays entirely in occupancy space.

        This is the method used by the Eulerian MPC optimizer for multi-step
        rollouts.  It avoids the lossy ``occ → FPS-particles → occ``
        round-trip that ``predict_one_step`` performs for interface
        compatibility.

        The gradient path is:
            act_seqs_tensor → action_to_cam_3d → cam3d_to_grid
                            → user_model → occ_pred → reward

        Returns
        -------
        occ_pred : (B, *grid_res)  next predicted occupancy field.
        """
        if self.action_convention == 'genesis':
            s_3d_cam, e_3d_cam = _action_to_cam_3d_genesis(
                action, self.global_scale)
        else:
            s_3d_cam, e_3d_cam = _action_to_cam_3d(
                action, self.cam_extrinsic, self.global_scale)
        start_grid = self._cam3d_to_grid(s_3d_cam)  # (B, 3)
        end_grid   = self._cam3d_to_grid(e_3d_cam)  # (B, 3)
        return self.user_model(occ_cur, start_grid, end_grid)

    def initial_occ_from_particles(
        self,
        s_cur: torch.Tensor,   # (B, N, 3)  particles in normalized cam coords
    ) -> torch.Tensor:
        """
        Convert a batch of particle observations to occupancy grids.

        Call this once at the start of each MPC step to seed the Eulerian
        optimizer from the current environment observation.

        Returns
        -------
        occ : (B, *grid_res)  float32, detached (not part of the grad graph).
        """
        with torch.no_grad():
            return _particles_to_occupancy(
                s_cur, self.grid_bounds, self.grid_res, sigma=self.splat_sigma)

    def prepare_goal_reward(
        self,
        subgoal: np.ndarray,   # (H, W)  0 = object should be here
        cam_params,            # [fx, fy, cx, cy]  from env.get_cam_params()
        device: str = 'cuda',
        empty_penalty: float = 0.0,
    ) -> torch.Tensor:
        """
        Precompute a (*grid_res) score tensor from the pixel-space subgoal.

        Call this **once** before the MPC optimizer loop; the tensor is then
        used inside the loop as a fixed reward landscape:

            reward_per_sample = (occ_pred * score_tensor).sum()

        Parameters
        ----------
        empty_penalty : float, optional
            Controls the reward assigned to voxels that should be empty.

            0.0 (default, backward-compatible)
                ``score -= score.min()`` is applied so all values are ≥ 0.
                Empty voxels carry 0 reward — the optimizer has no incentive
                to move material *away* from non-goal regions.

            > 0.0
                ``dist_from_goal`` is normalized to [0, 1] and subtracted
                with weight ``empty_penalty``.  Score ∈ [-empty_penalty, +1]:
                goal voxels ≈ +1, the farthest empty voxel ≈ -empty_penalty.
                The optimizer is now penalized for placing material in regions
                that should be empty, which discourages minimalist no-op
                actions (pushing material nowhere useful still hurts).

        Returns
        -------
        score : torch.Tensor (*grid_res) on ``device``, higher = better.
        """
        from scipy.ndimage import distance_transform_edt
        device = device if torch.cuda.is_available() else 'cpu'
        occ_goal = self.subgoal_mask_to_occupancy(subgoal, cam_params)  # (1, *grid_res)
        occ_goal_np = occ_goal[0].numpy()                               # (*grid_res)

        occupied = (occ_goal_np > 0.5)
        # Zero on occupied voxels, positive elsewhere.
        dist_from_goal = distance_transform_edt(~occupied).astype(np.float32)

        if empty_penalty > 0.0:
            # Normalize distance to [0, 1] so the penalty is bounded regardless
            # of grid size.  Score: goal ≈ +1, farthest empty ≈ -empty_penalty.
            max_dist = dist_from_goal.max()
            dist_norm = dist_from_goal / max_dist if max_dist > 0 else dist_from_goal
            score = occ_goal_np.astype(np.float32) - empty_penalty * dist_norm
        else:
            score = occ_goal_np.astype(np.float32) - dist_from_goal
            score -= score.min()   # shift: 0 = farthest from goal, max = at goal

        return torch.from_numpy(score).to(device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def obs_to_occupancy(
        self,
        obs: np.ndarray,
        cam_params,
        depth_thresh: float = 0.599 / 0.8,
    ) -> torch.Tensor:
        """
        Convert a raw simulation observation (H, W, 5) directly to an
        occupancy grid, bypassing the particle intermediate representation.

        This is useful for computing the *goal* occupancy if the subgoal is
        given as a binary pixel mask (the existing ``subgoal`` array has value
        0 where the object should be and 1 elsewhere).

        Parameters
        ----------
        obs : np.ndarray (H, W, 5)
            Raw observation from ``env.render()`` (or a goal image similarly
            structured).
        cam_params : [fx, fy, cx, cy]
            From ``env.get_cam_params()``.
        depth_thresh : float
            Normalized depth threshold below which pixels are foreground.

        Returns
        -------
        occ : torch.Tensor (1, *grid_res)
        """
        from utils import depth2fgpcd
        depth = obs[..., -1] / self.global_scale
        mask  = depth < depth_thresh
        fgpcd = depth2fgpcd(depth, mask, cam_params)   # (M, 3)
        s_cur = torch.from_numpy(fgpcd).float().unsqueeze(0)  # (1, M, 3)
        return _particles_to_occupancy(s_cur, self.grid_bounds, self.grid_res,
                                       sigma=self.splat_sigma)

    def subgoal_mask_to_occupancy(
        self,
        subgoal: np.ndarray,   # (H, W)  0 = object should be here, 1 = background
        cam_params,
    ) -> torch.Tensor:
        """
        Convert the binary pixel-space subgoal mask to a goal occupancy grid.

        The convention in the existing code is that ``subgoal < 0.5`` marks
        the target region.  This method back-projects those pixels into 3-D
        at a fixed depth (the mean of ``z_min`` and ``z_max`` in grid_bounds)
        and then voxelizes them.

        Returns
        -------
        occ_goal : torch.Tensor (1, *grid_res)
        """
        H, W = subgoal.shape
        fx, fy, cx, cy = cam_params

        # Pixels where the goal region is active
        ys, xs = np.where(subgoal < 0.5)   # pixel rows and columns

        # Use a fixed depth equal to the z-extent midpoint
        z_mid = 0.5 * (self.grid_bounds.get('z_min', 0.7)
                       + self.grid_bounds.get('z_max', 0.8))
        depth_val = z_mid * self.global_scale   # un-normalize for projection

        # Mirrors depth2fgpcd: particle.x = (col - cx) * depth_norm / fx
        X = (xs - cx) * z_mid / fx
        Y = (ys - cy) * z_mid / fy
        Z = np.full_like(X, z_mid, dtype=np.float32)
        fgpcd = np.stack([X, Y, Z], axis=1).astype(np.float32)  # (M, 3)

        if fgpcd.shape[0] == 0:
            return torch.zeros([1] + list(self.grid_res))

        s_goal = torch.from_numpy(fgpcd).float().unsqueeze(0)
        return _particles_to_occupancy(s_goal, self.grid_bounds, self.grid_res,
                                       sigma=self.splat_sigma)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cam3d_to_grid(self, pts_cam: torch.Tensor) -> torch.Tensor:
        """
        Map (B, 3) normalized camera coords to (B, 3) grid-index coords.

        The mapping is linear from [bound_min, bound_max] to [0, grid_res-1].
        The output uses the same axis ordering as the grid (x, [y,] z).
        """
        axes = _get_axes(len(self.grid_res))
        axis_idx = _axis_indices(axes)
        lo  = torch.tensor([self.grid_bounds[f'{a}_min'] for a in axes],
                           device=pts_cam.device, dtype=pts_cam.dtype)
        hi  = torch.tensor([self.grid_bounds[f'{a}_max'] for a in axes],
                           device=pts_cam.device, dtype=pts_cam.dtype)
        res = torch.tensor(self.grid_res,
                           device=pts_cam.device, dtype=pts_cam.dtype)
        pts_sel = pts_cam[:, axis_idx]                         # (B, ndim)
        grid_coords = (pts_sel - lo) / (hi - lo) * (res - 1)  # (B, ndim)

        # Return as 3-D; fill the missing axis with 0 for 2-D grids
        out = torch.zeros(pts_cam.shape[0], 3, device=pts_cam.device, dtype=pts_cam.dtype)
        for k, a in enumerate(axes):
            dim3d = {'x': 0, 'y': 1, 'z': 2}[a]
            out[:, dim3d] = grid_coords[:, k]
        return out


# ---------------------------------------------------------------------------
# Built-in push models for use with EulerianModelWrapper
# ---------------------------------------------------------------------------

class SplatPushModel(nn.Module):
    """
    Differentiable push model based on bilinear splatting.

    Wraps ``differentiable_push_splat`` (and optionally
    ``differentiable_redistribute``) from ``model/diff_mass_push.py`` so that
    it satisfies the ``EulerianModelWrapper`` user-model contract::

        forward(occ, action_start, action_end) -> occ_pred

    Usage
    -----
    >>> model = SplatPushModel(width=5, sigma=1.5, redistribute=True)
    >>> wrapper = EulerianModelWrapper(model, bounds, grid_res, cam_ext, gs)

    Coordinate note
    ---------------
    ``EulerianModelWrapper`` stores the grid as ``(B, Nx, Ny)`` where dim 1
    indexes camera-x (column direction) and dim 2 indexes camera-y (row
    direction).  The ``diff_mass_push`` functions use the standard image
    convention ``(H=rows, W=cols)`` = ``(cam_y, cam_x)``.  This class
    handles the transpose internally; callers do not need to worry about it.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).  Smaller → sharper edges
        but weaker gradients near the boundary.
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push to diffuse excess
        occupancy forward along the push direction.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 3.0,
        sigma: float = 1.0,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords (cam_x, cam_y, ...)
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import differentiable_push_splat_batch

        # Transpose entire batch at once: (B, Nx, Ny) → (B, Ny, Nx) so that
        # dim 1 = rows = cam_y and dim 2 = cols = cam_x, matching the (H, W)
        # image convention expected by the push function.  This replaces the
        # previous Python ``for b in range(B)`` loop (one serial GPU call per
        # sample) with a single fully-vectorised kernel call.
        rho = occ.permute(0, 2, 1)             # (B, Ny, Nx) = (B, H, W)
        p0  = action_start[:, :2]              # (B, 2) [cam_x, cam_y]
        p1  = action_end[:, :2]

        rho_new, _ = differentiable_push_splat_batch(
            rho, p0, p1, width=self.width, sigma=self.sigma)

        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute
            d      = p1 - p0                   # (B, 2)
            d_norm = d.norm(dim=-1)            # (B,)
            # redistribute operates on single frames — loop is unavoidable here;
            # this path is only taken when redistribute=True (off by default)
            for b in range(occ.shape[0]):
                if d_norm[b].item() > 1e-6:
                    rho_new[b] = differentiable_redistribute(
                        rho_new[b], d[b] / d_norm[b],
                        max_iters=self.redistribute_iters)

        return rho_new.permute(0, 2, 1)        # (B, Nx, Ny)


class FluidPushModel(nn.Module):
    """
    Differentiable push model using a velocity-field (fluid) approach.

    Wraps ``fluid_push`` (and optionally ``differentiable_redistribute``) from
    ``model/diff_mass_push.py`` so that it satisfies the
    ``EulerianModelWrapper`` user-model contract::

        forward(occ, action_start, action_end) -> occ_pred

    Usage
    -----
    >>> model = FluidPushModel(width=5, n_steps=20)
    >>> wrapper = EulerianModelWrapper(model, bounds, grid_res, cam_ext, gs)

    See ``SplatPushModel`` for notes on the coordinate-convention transpose.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).
    n_steps : int
        Number of velocity propagation iterations (controls influence radius).
    decay : float
        Per-step velocity attenuation factor (0–1).
    media_sharpness : float
        Steepness of the soft media-presence gate; higher → velocity stays
        more confined to already-occupied regions.
    blur_sigma : float
        Gaussian σ for the propagation blur kernel (voxels).
    correct_divergence : bool
        Apply the Jacobian divergence correction for approximate mass
        conservation during the advection step.
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 5.0,
        sigma: float = 1.0,
        n_steps: int = 20,
        decay: float = 0.95,
        media_sharpness: float = 5.0,
        blur_sigma: float = 1.0,
        correct_divergence: bool = False,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.n_steps = n_steps
        self.decay = decay
        self.media_sharpness = media_sharpness
        self.blur_sigma = blur_sigma
        self.correct_divergence = correct_divergence
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import fluid_push
        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute

        B = occ.shape[0]
        results = []
        for b in range(B):
            rho = occ[b].T              # (Ny, Nx) – image (rows, cols) convention
            p0  = action_start[b, :2]
            p1  = action_end[b, :2]

            rho_new, _, _ = fluid_push(
                rho, p0, p1,
                width=self.width,
                sigma=self.sigma,
                n_steps=self.n_steps,
                decay=self.decay,
                media_sharpness=self.media_sharpness,
                blur_sigma=self.blur_sigma,
                correct_divergence=self.correct_divergence,
            )

            if self.redistribute:
                d = p1 - p0
                d_norm = d.norm()
                if d_norm.item() > 1e-6:
                    rho_new = differentiable_redistribute(
                        rho_new, d / d_norm,
                        max_iters=self.redistribute_iters)

            results.append(rho_new.T)   # back to (Nx, Ny)
        return torch.stack(results, dim=0)


class SpreadPushModel(nn.Module):
    """
    Differentiable push with linear proportional spread (Approach B).

    Instead of depositing all swept mass at a single line (like
    ``SplatPushModel``), spreads the pile forward past the tool end: material
    originally near the start of the sweep is deposited farthest from p1,
    material near p1 stays close.  Pile length equals total_mass / (2*width),
    which is exact for uniform-density input.

    Wraps ``differentiable_push_spread_batch`` from ``model/diff_mass_push.py``.

    See ``SplatPushModel`` for notes on the coordinate-convention transpose.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 3.0,
        sigma: float = 1.0,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords (cam_x, cam_y, ...)
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import differentiable_push_spread_batch

        rho = occ.permute(0, 2, 1)             # (B, Ny, Nx) = (B, H, W)
        p0  = action_start[:, :2]
        p1  = action_end[:, :2]

        rho_new, _ = differentiable_push_spread_batch(
            rho, p0, p1, width=self.width, sigma=self.sigma)

        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute
            d      = p1 - p0
            d_norm = d.norm(dim=-1)
            for b in range(occ.shape[0]):
                if d_norm[b].item() > 1e-6:
                    rho_new[b] = differentiable_redistribute(
                        rho_new[b], d[b] / d_norm[b],
                        max_iters=self.redistribute_iters)

        return rho_new.permute(0, 2, 1)        # (B, Nx, Ny)


class SplatPushModel2(nn.Module):
    """
    Destination-aware spread push with optional isotropic blur.

    Extends ``SpreadPushModel`` to handle the case where the deposit region
    already contains material (which would otherwise cause stacking).

    Two anti-stacking mechanisms:

    1.  **Destination extension** – the landing zone (rectangle past p1,
        width = tool width, length = pile_depth) is probed for existing
        occupancy.  The deposit band is extended forward by
        ``extra_depth = existing_mass / (2 * width)``, so swept material
        leapfrogs pre-existing material.

    2.  **Isotropic blur** (optional, ``blur_sigma > 0``) – a mild Gaussian
        blur is applied *only* to the deposited mass before it is added
        back to the cleared field.  Simulates granular scatter and smooths
        residual peaks the first-order extension misses.

    Wraps ``differentiable_push_spread2_batch`` from
    ``model/diff_mass_push.py``.

    See ``SplatPushModel`` for notes on the coordinate-convention transpose.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).
    blur_sigma : float
        If > 0, Gaussian σ (voxels) for post-deposit isotropic blur on the
        deposited mass only.  Default 0 (off).
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 3.0,
        sigma: float = 1.0,
        blur_sigma: float = 0.0,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.blur_sigma = blur_sigma
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords (cam_x, cam_y, ...)
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import differentiable_push_spread2_batch

        rho = occ.permute(0, 2, 1)             # (B, Ny, Nx) = (B, H, W)
        p0  = action_start[:, :2]
        p1  = action_end[:, :2]

        rho_new, _ = differentiable_push_spread2_batch(
            rho, p0, p1, width=self.width, sigma=self.sigma,
            blur_sigma=self.blur_sigma)

        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute
            d      = p1 - p0
            d_norm = d.norm(dim=-1)
            for b in range(occ.shape[0]):
                if d_norm[b].item() > 1e-6:
                    rho_new[b] = differentiable_redistribute(
                        rho_new[b], d[b] / d_norm[b],
                        max_iters=self.redistribute_iters)

        return rho_new.permute(0, 2, 1)        # (B, Nx, Ny)


class CumulativePushModel(nn.Module):
    """
    Differentiable push with cumulative-mass forward spread (Approach A).

    Instead of depositing all swept mass at a single line (like
    ``SplatPushModel``), computes the cumulative mass ahead of each swept
    pixel along the push direction and deposits it at
    ``p1 + cum_ahead * d_unit``.  This implements the snow-plow formula and
    is exact for arbitrary (non-uniform) density distributions.

    The cumulative scan uses K sequential ``grid_sample`` calls where
    K ≈ push length in pixels, so it is more expensive than
    ``SpreadPushModel`` but more physically accurate for non-uniform input.

    Wraps ``differentiable_push_cumulative_batch`` from
    ``model/diff_mass_push.py``.

    See ``SplatPushModel`` for notes on the coordinate-convention transpose.

    Parameters
    ----------
    width : float
        Half-width of the tool in grid voxels.
    sigma : float
        Edge softness of the swept mask (voxels).
    redistribute : bool
        Apply ``differentiable_redistribute`` after the push.
    redistribute_iters : int
        Maximum spread distance in voxels (used only when
        ``redistribute=True``).
    """

    def __init__(
        self,
        width: float = 3.0,
        sigma: float = 1.0,
        redistribute: bool = False,
        redistribute_iters: int = 10,
    ):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.redistribute = redistribute
        self.redistribute_iters = redistribute_iters

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)
        action_start: torch.Tensor,  # (B, 3)  grid coords (cam_x, cam_y, ...)
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)
        from model.diff_mass_push import differentiable_push_cumulative_batch

        rho = occ.permute(0, 2, 1)             # (B, Ny, Nx) = (B, H, W)
        p0  = action_start[:, :2]
        p1  = action_end[:, :2]

        rho_new, _ = differentiable_push_cumulative_batch(
            rho, p0, p1, width=self.width, sigma=self.sigma)

        if self.redistribute:
            from model.diff_mass_push import differentiable_redistribute
            d      = p1 - p0
            d_norm = d.norm(dim=-1)
            for b in range(occ.shape[0]):
                if d_norm[b].item() > 1e-6:
                    rho_new[b] = differentiable_redistribute(
                        rho_new[b], d[b] / d_norm[b],
                        max_iters=self.redistribute_iters)

        return rho_new.permute(0, 2, 1)        # (B, Nx, Ny)


class UNetFiLMPushModel(nn.Module):
    """
    Adapter that wraps a trained ``UNetFiLM`` model for use as the *push model*
    inside ``EulerianModelWrapper``.

    ``EulerianModelWrapper`` calls ``user_model.forward(occ, start_grid, end_grid)``
    where ``occ`` is ``(B, Nx, Ny)`` in *camera* convention (dim 0 = camera-x,
    dim 1 = camera-y = −world-y) and start/end are grid-index coordinates.

    ``UNetFiLM`` was trained with the ``PileSweepData`` dataset convention:

    - The grid is ``(Nx, Ny)`` in *world* coordinates
      (dim 0 = world-x, dim 1 = world-y).
    - Channel 0: current particle occupancy.
    - Channel 1: action channel — plate drawn at start position (intensity 0.5)
      and end position (intensity 1.0).

    This wrapper converts between the two conventions and renders the action
    channel using a differentiable soft-edged rectangle, so that gradient
    descent in the MPC optimizer can propagate through the action input.

    Parameters
    ----------
    unet_film : nn.Module
        Trained ``UNetFiLM`` instance.
    physics : torch.Tensor  shape (3,)
        Fixed physics parameters: [particle_friction, particle_density,
        box_friction].  Read from the sim config; held constant during MPC.
    grid_size : (int, int)
        Occupancy grid size ``(Nx, Ny)``.  Must match the dataset training
        resolution (e.g. (128, 128) for a 128 mm × 128 mm box at 1 px/mm).
    plate_length_px : float
        Plate long-axis in grid pixels (= plate_length_m × pixels_per_metre).
        Default 40 (= 0.04 m × 1000 px/m for a 128×128 px grid).
    plate_width_px : float
        Plate short-axis in grid pixels.  Default 2.
    sigma : float
        Softness (pixels) of the plate boundary in the differentiable mask.
        Larger → smoother gradients; smaller → sharper edges.  Default 1.5.
    """

    def __init__(
        self,
        unet_film: nn.Module,
        physics: torch.Tensor,        # (3,) [particle_friction, density, box_friction]
        grid_size: Tuple[int, int],
        plate_length_px: float = 40.0,
        plate_width_px: float  =  2.0,
        sigma: float           =  1.5,
    ):
        super().__init__()
        self.unet_film = unet_film
        self.register_buffer('_physics', physics.view(1, -1).float())
        self.Nx, self.Ny = grid_size
        self.plate_L = float(plate_length_px)
        self.plate_W = float(plate_width_px)
        self.sigma   = float(sigma)

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    @staticmethod
    def default_bounds(config: dict) -> Dict[str, float]:
        """
        Return grid bounds that exactly match the training dataset's box.

        For UNetFiLM the grid must align with the 1 px/mm dataset convention,
        so this uses the *exact* box half-width (no 1.2× expansion used by the
        heuristic models).
        """
        gs_val = config['dataset']['global_scale']
        w      = config['dataset']['wkspc_w']   # half-width in world metres
        w_n    = w / gs_val                      # normalised half-width
        z_table = 0.5                            # Genesis: table at z_norm = 0.5
        z_margin = 0.05
        return {
            'x_min': -w_n, 'x_max':  w_n,
            'y_min': -w_n, 'y_max':  w_n,
            'z_min': z_table - z_margin, 'z_max': z_table + z_margin,
        }

    # ------------------------------------------------------------------
    # Differentiable plate rendering
    # ------------------------------------------------------------------

    def _draw_plate_soft(
        self,
        center: torch.Tensor,  # (B, 2) [iy_world_y, ix_world_x] in dataset grid coords
        angle: torch.Tensor,   # (B,)   plate draw-angle (see convention below)
        intensity: float,
    ) -> torch.Tensor:         # (B, Nx, Ny)
        """
        Render a soft-edged plate rectangle differentiably.

        The gradient flows through ``center``, enabling the MPC Adam optimizer
        to improve start/end positions via backpropagation.

        Convention (training/dataset layout, after the flip+transpose):
            grid[dim0, dim1] = grid[world_y_idx, world_x_idx]
            Dim 0 = world-y, Dim 1 = world-x.

        Angle convention:
            angle = atan2(Δworld_y, Δworld_x) — direction of travel.
            At angle=0 (travel along +world_x): plate_L is along dim 0 (world_y),
            i.e. the plate is perpendicular to the direction of travel ✓.
            This differs from the physical plate yaw (angle_sim = angle + π/2).
        """
        device = center.device

        # Grid coordinate arrays for the two spatial dims
        ix = torch.arange(self.Nx, device=device, dtype=torch.float32)
        iy = torch.arange(self.Ny, device=device, dtype=torch.float32)
        GX, GY = torch.meshgrid(ix, iy, indexing='ij')  # (Nx, Ny) each

        cx = center[:, 0:1, None]   # (B, 1, 1) — along world-y (dim 0)
        cy = center[:, 1:2, None]   # (B, 1, 1) — along world-x (dim 1)

        cos_a = torch.cos(angle)[:, None, None]  # (B, 1, 1)
        sin_a = torch.sin(angle)[:, None, None]

        dx = GX[None] - cx   # (B, Nx, Ny) — displacement along dim 0 (world-y)
        dy = GY[None] - cy   # (B, Nx, Ny) — displacement along dim 1 (world-x)

        # Rotate into plate-local frame:
        #   rl = along length,  rw = along width
        # At angle=0 (travel in +world_x): rl=dx (length along world_y ⊥ to travel) ✓
        rl = cos_a * dx + sin_a * dy
        rw = -sin_a * dx + cos_a * dy

        mask_l = torch.sigmoid((self.plate_L / 2.0 - rl.abs()) / self.sigma)
        mask_w = torch.sigmoid((self.plate_W / 2.0 - rw.abs()) / self.sigma)
        return mask_l * mask_w * intensity   # (B, Nx, Ny)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        occ: torch.Tensor,           # (B, Nx, Ny)  EulerianWrapper camera convention
        action_start: torch.Tensor,  # (B, 3) [ix_cam_x, iy_cam_y, iz]  grid coords
        action_end: torch.Tensor,    # (B, 3)
    ) -> torch.Tensor:               # (B, Nx, Ny)  EulerianWrapper convention
        """
        Predict the next occupancy given current occupancy and action.

        Coordinate conversion
        ---------------------
        ``EulerianModelWrapper`` stores grids with:
            dim 0 = camera-x = world-x,  dim 1 = camera-y = −world-y

        ``UNetFiLM`` was trained with the ``PileSweepData`` dataset convention
        (cv2 image layout, rows × cols):
            dim 0 = world-y (rows ↑ = world_y ↑),  dim 1 = world-x (cols)

        This method applies **flip(dim 1) + transpose(-2,-1)** on input and
        the inverse **transpose(-2,-1) + flip(dim 1)** on output.

        Action channel
        --------------
        Rendered by drawing a soft plate rectangle at the start position
        (intensity 0.5) and end position (intensity 1.0), with the plate
        orientation perpendicular to the direction of travel — matching
        the ``genesis_env.step()`` convention.
        """
        Ny = self.Ny

        # ── 1. Convert EulerianWrapper → dataset convention ──────────────────
        # EulerianWrapper: (B, dim0=cam_x=world_x, dim1=cam_y=−world_y)
        # Dataset (cv2):   (B, dim0=world_y,        dim1=world_x)
        # Step 1: flip dim1 so cam_y → world_y  →  (world_x, world_y)
        # Step 2: transpose so (world_x, world_y) → (world_y, world_x)
        occ_ds = occ.flip(dims=[-1]).transpose(-2, -1)   # (B, Ny_world, Nx_world)

        # ── 2. Convert action y-indices to dataset convention ────────────────
        # EulerianWrapper: iy_cam = grid-y index; cam_y = −world_y
        # Dataset:         iy_ds  = (Ny−1) − iy_cam
        iy_s_cam = action_start[:, 1]
        iy_e_cam = action_end[:, 1]
        iy_s_ds  = (Ny - 1) - iy_s_cam
        iy_e_ds  = (Ny - 1) - iy_e_cam

        # ── 3. Plate draw-angle in dataset convention ─────────────────────────
        # Dataset convention after flip+transpose: dim0=world_y, dim1=world_x.
        # In _draw_plate_soft at angle=0: plate_L is along dim0=world_y.
        # Plate should be perpendicular to the direction of travel, so:
        #   angle_draw = atan2(Δworld_y, Δworld_x) = direction of travel
        # (The physical plate yaw angle_sim = angle_draw + π/2 is NOT used here;
        #  the +π/2 and the axis-swap of the transpose cancel exactly.)
        dx    = action_end[:, 0] - action_start[:, 0]
        dy_ds = iy_e_ds - iy_s_ds            # world-y direction in grid px

        dxy   = torch.hypot(dx, dy_ds)
        # Draw-angle = direction of travel (no +π/2 compared to physical yaw)
        angle = torch.where(
            dxy > 1e-4,
            torch.atan2(dy_ds, dx),
            torch.zeros_like(dxy),
        )

        # ── 4. Draw action channel in dataset convention ──────────────────────
        # Center = (world_y_idx, world_x_idx) — dim0 first, matching dataset.
        # Use .detach() for the angle only; gradients still flow through centers.
        start_center = torch.stack([iy_s_ds, action_start[:, 0]], dim=1)  # (B,2) (world_y, world_x)
        end_center   = torch.stack([iy_e_ds, action_end[:, 0]],   dim=1)

        act_start = self._draw_plate_soft(start_center, angle.detach(), 0.5)
        act_end   = self._draw_plate_soft(end_center,   angle.detach(), 1.0)
        act_ch    = torch.maximum(act_start, act_end)   # (B, Nx, Ny)

        # ── 5. Build 2-channel input [occupancy, action] ─────────────────────
        x = torch.stack([occ_ds, act_ch], dim=1)        # (B, 2, Nx, Ny)

        # ── 6. Physics vector (fixed, broadcast to batch) ────────────────────
        phys = self._physics.expand(occ.shape[0], -1)   # (B, 3)

        # ── 7. UNetFiLM forward ───────────────────────────────────────────────
        occ_pred_ds = self.unet_film(x, phys).squeeze(1)  # (B, Nx, Ny)

        # ── 8. Convert dataset → EulerianWrapper convention ──────────────────
        # Apply sigmoid: model was trained with MSE(sigmoid(logit), target),
        # so sigmoid gives the occupancy probability in [0,1].  This is
        # essential for:
        #   (a) reward gradients — clamp(0,1) has zero gradient when its input
        #       is ≥ 1 or ≤ 0; sigmoid output is always in (0,1) so gradient
        #       always flows;
        #   (b) multi-step rollouts — subsequent steps receive [0,1]
        #       occupancy, matching the training input distribution.
        # Inverse of step 1: transpose(-2,-1) then flip(dim1)
        return torch.sigmoid(occ_pred_ds).transpose(-2, -1).flip(dims=[-1])
