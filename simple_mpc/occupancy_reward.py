"""
Model-agnostic occupancy-based reward computation.

This module provides a single utility class for converting a pixel-space goal
mask to an occupancy grid and computing a reward map. Both Eulerian and GNN
models can use this to ensure consistent, model-invariant reward calculation.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict

from model.eulerian_wrapper import _particles_to_occupancy


class OccupancyReward:
    """
    Compute occupancy-based goal reward maps.

    This class handles the conversion of a pixel-space goal mask (subgoal)
    to a reward tensor compatible with occupancy grid scoring. It is used
    by both Eulerian and GNN adapters to ensure model-invariant rewards.

    Parameters
    ----------
    grid_bounds : dict
        Grid extent in normalized camera coords.
        Required keys: x_min, x_max, y_min, y_max, z_min, z_max
    grid_res : tuple[int, ...]
        Number of voxels per axis, e.g., (64, 64)
    global_scale : float
        Normalization factor for camera-space coordinates
    cam_params : tuple[float, float, float, float]
        Camera intrinsics [fx, fy, cx, cy] from env.get_cam_params()
    """

    def __init__(
        self,
        grid_bounds: Dict[str, float],
        grid_res: tuple,
        global_scale: float,
        cam_params: tuple,
    ):
        self.grid_bounds = dict(grid_bounds)
        self.grid_res = tuple(grid_res)
        self.global_scale = float(global_scale)
        self.cam_params = cam_params

    def compute_score_tensor(
        self,
        subgoal: np.ndarray,     # (H, W)  0 = inside goal
        device: str = 'cuda',
        empty_penalty: float = 0.0,
    ) -> torch.Tensor:
        """
        Convert a binary subgoal mask to a reward tensor.

        The subgoal is back-projected to 3D, converted to an occupancy grid,
        and scored using distance transform to assign gradient: high reward
        in goal region, lower reward outside.

        Parameters
        ----------
        subgoal : np.ndarray (H, W)
            Binary distance transform; 0 = inside goal, 1 = outside
        device : str
            PyTorch device ('cuda' or 'cpu')
        empty_penalty : float
            Controls reward in non-goal regions (see details below)

        Returns
        -------
        score : torch.Tensor (*grid_res)
            Occupancy-based reward map on ``device``

        Notes
        -----
        empty_penalty behavior:
            0.0 (default): Empty voxels get 0 reward; no penalty for wrong regions
            > 0.0: Penalizes material in non-goal regions proportionally to distance
        """
        from scipy.ndimage import distance_transform_edt

        # Step 1: Convert pixel-space goal to occupancy grid
        occ_goal = self._subgoal_mask_to_occupancy(subgoal)  # (1, *grid_res)
        occ_goal_np = occ_goal[0].numpy()                    # (*grid_res)

        # Step 2: Compute distance map from goal region
        occupied = (occ_goal_np > 0.5)
        # Zero on occupied voxels, positive elsewhere.
        dist_from_goal = distance_transform_edt(~occupied).astype(np.float32)

        # Step 3: Apply empty_penalty logic
        if empty_penalty > 0.0:
            # Normalize distance to [0, 1] so the penalty is bounded regardless
            # of grid size.  Score: goal ≈ +1, farthest empty ≈ -empty_penalty.
            max_dist = dist_from_goal.max()
            dist_norm = (dist_from_goal / max_dist) if max_dist > 0 else dist_from_goal
            score = occ_goal_np.astype(np.float32) - empty_penalty * dist_norm
        else:
            score = occ_goal_np.astype(np.float32) - dist_from_goal
            score -= score.min()   # shift: 0 = farthest from goal, max = at goal

        return torch.from_numpy(score).to(device=device, dtype=torch.float32)

    def _subgoal_mask_to_occupancy(self, subgoal: np.ndarray) -> torch.Tensor:
        """
        Convert pixel-space binary mask to occupancy grid.

        Back-projects goal pixels to 3D at a fixed depth, then voxelizes.

        Parameters
        ----------
        subgoal : np.ndarray (H, W)
            Binary mask; 0 = object should be here, 1 = background

        Returns
        -------
        occ_goal : torch.Tensor (1, *grid_res)
        """
        H, W = subgoal.shape
        fx, fy, cx, cy = self.cam_params

        # Pixels where goal region is active
        ys, xs = np.where(subgoal < 0.5)   # rows and columns

        # Fixed depth at z-extent midpoint
        z_mid = 0.5 * (
            self.grid_bounds.get('z_min', 0.7)
            + self.grid_bounds.get('z_max', 0.8)
        )

        # Back-project to 3D normalized camera coords (mirrors depth2fgpcd)
        X = (xs - cx) * z_mid / fx
        Y = (ys - cy) * z_mid / fy
        Z = np.full_like(X, z_mid, dtype=np.float32)
        fgpcd = np.stack([X, Y, Z], axis=1).astype(np.float32)  # (M, 3)

        if fgpcd.shape[0] == 0:
            return torch.zeros([1] + list(self.grid_res))

        s_goal = torch.from_numpy(fgpcd).float().unsqueeze(0)
        return _particles_to_occupancy(
            s_goal, self.grid_bounds, self.grid_res, sigma=0.0
        )
