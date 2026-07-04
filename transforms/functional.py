"""Pure functional transform utilities shared across datasets and MPC paths."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from model.eulerian_wrapper import _particles_to_occupancy


def particles_to_occupancy(
    particles: torch.Tensor,
    bounds: Dict[str, float],
    resolution: Tuple[int, ...],
    sigma: float = 0.0,
) -> torch.Tensor:
    """Convert particle clouds (B,N,3) to occupancy grids using shared wrapper logic."""
    return _particles_to_occupancy(particles, bounds, resolution, sigma=sigma)


def genesis_action_to_cam3d(
    action: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map world 2-D push actions [sx,sy,ex,ey] to normalised Genesis camera 3-D."""
    sx, sy = action[:, 0:1] / scale, action[:, 1:2] / scale
    ex, ey = action[:, 2:3] / scale, action[:, 3:4] / scale
    half = torch.full_like(sx, 0.5)
    s_3d_cam = torch.cat([sx, -sy, half], dim=1)
    e_3d_cam = torch.cat([ex, -ey, half], dim=1)
    return s_3d_cam, e_3d_cam


def _point_to_segment_distance_xy(
    points_xy: torch.Tensor,
    seg_start_xy: torch.Tensor,
    seg_end_xy: torch.Tensor,
) -> torch.Tensor:
    vec = seg_end_xy - seg_start_xy
    vec_norm_sq = torch.dot(vec, vec).clamp_min(1e-12)
    rel = points_xy - seg_start_xy[None, :]
    t = (rel * vec[None, :]).sum(dim=1) / vec_norm_sq
    t = t.clamp(0.0, 1.0)
    proj = seg_start_xy[None, :] + t[:, None] * vec[None, :]
    return (points_xy - proj).norm(dim=1)


def build_action_delta(
    s_cur_xyz: torch.Tensor,
    p_start_xyz: torch.Tensor,
    p_stop_xyz: torch.Tensor,
    sigma_m: float,
) -> torch.Tensor:
    """Build differentiable per-particle action displacement used by PropNet training."""
    action_vec = p_stop_xyz - p_start_xyz
    action_xy = action_vec[:2]
    points_xy = s_cur_xyz[:, :2]

    dist = _point_to_segment_distance_xy(points_xy, p_start_xyz[:2], p_stop_xyz[:2])
    influence = torch.exp(-(dist * dist) / (2.0 * sigma_m * sigma_m))

    s_delta = torch.zeros_like(s_cur_xyz)
    s_delta[:, :2] = influence[:, None] * action_xy[None, :]
    return s_delta


def draw_plate_soft(
    center: torch.Tensor,
    angle: torch.Tensor,
    grid_size: Tuple[int, int],
    plate_length_px: float,
    plate_width_px: float,
    intensity: float,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Differentiable soft plate rasterizer in dataset convention (dim0=world_y, dim1=world_x)."""
    device = center.device
    nx, ny = grid_size

    ix = torch.arange(nx, device=device, dtype=torch.float32)
    iy = torch.arange(ny, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(ix, iy, indexing="ij")

    cx = center[:, 0:1, None]
    cy = center[:, 1:2, None]

    cos_a = torch.cos(angle)[:, None, None]
    sin_a = torch.sin(angle)[:, None, None]

    dx = gx[None] - cx
    dy = gy[None] - cy

    rl = cos_a * dx + sin_a * dy
    rw = -sin_a * dx + cos_a * dy

    mask_l = torch.sigmoid((plate_length_px / 2.0 - rl.abs()) / sigma)
    mask_w = torch.sigmoid((plate_width_px / 2.0 - rw.abs()) / sigma)
    return mask_l * mask_w * float(intensity)
