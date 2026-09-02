"""Pure functional transform utilities shared across datasets and MPC paths."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


# ----- coordinate bookkeeping helpers --------------------------------------

def get_grid_axes(ndim: int):
    """Return the particle-coordinate axes used by the grid.

    For a top-down camera ``depth2fgpcd`` places:
      dim 0 = camera x  (horizontal, proportional to pixel column)
      dim 1 = camera y  (horizontal, proportional to pixel row)
      dim 2 = camera z  (depth, ≈ constant 0.75 for all table particles)
    So a 2-D top-down grid should span dims 0 and 1, not 0 and 2.
    """
    if ndim == 2:
        return ('x', 'y')   # camera x and y span the horizontal table plane
    elif ndim == 3:
        return ('x', 'y', 'z')
    else:
        raise ValueError(f"grid_res must have 2 or 3 elements, got {ndim}")


def grid_axis_indices(axes):
    """Map axis names to indices in the 3-D particle coordinate vector."""
    mapping = {'x': 0, 'y': 1, 'z': 2}
    return [mapping[a] for a in axes]


def _ravel_idx(idx: torch.Tensor, grid_res: Tuple[int, ...]) -> torch.Tensor:
    """Ravel multi-dim indices (N, ndim) to flat index (N,) for a C-order grid."""
    strides = torch.tensor(
        [math.prod(grid_res[k + 1:]) for k in range(len(grid_res))],
        device=idx.device, dtype=torch.long)
    return (idx * strides).sum(-1)


def _make_grid_coords(grid_res: Tuple[int, ...], device) -> torch.Tensor:
    """Return a grid of voxel-index coordinates (*grid_res, ndim)."""
    ranges = [torch.arange(r, device=device, dtype=torch.float32) for r in grid_res]
    mesh = torch.meshgrid(*ranges, indexing='ij')  # each: (*grid_res)
    return torch.stack(mesh, dim=-1)               # (*grid_res, ndim)


def particles_to_occupancy(
    particles: torch.Tensor,              # (B, N, 3) normalized cam coords
    bounds: Dict[str, float],
    resolution: Tuple[int, ...],
    sigma: float = 0.0,                   # >0 → soft Gaussian splat; 0 → hard voxel
    footprint_radius: float = 0.0,        # >0 → hard disk splat of this voxel radius
) -> torch.Tensor:
    """
    Convert a batch of particle point-clouds to an occupancy grid.

    ``footprint_radius`` (in voxel units) fills every cell within that radius
    of each particle center, instead of just the single nearest voxel. This
    matters when particle centers are sparse relative to the grid (e.g. a
    Lagrangian/particle-based rollout compared against a dense depth-derived
    occupancy grid) — see ``simple_mpc.genesis_oracle``. Mutually exclusive
    with ``sigma`` (footprint_radius takes precedence if both are set).

    Returns
    -------
    occ : (B, *resolution)  float32, values in [0, 1].
    """
    B, N, _ = particles.shape
    device = particles.device
    ndim = len(resolution)

    # Axis names and order: x (grid dim 0), then y / z depending on ndim
    axes = get_grid_axes(ndim)
    lo  = torch.tensor([bounds[f'{a}_min'] for a in axes], device=device, dtype=particles.dtype)
    hi  = torch.tensor([bounds[f'{a}_max'] for a in axes], device=device, dtype=particles.dtype)
    res = torch.tensor(resolution, device=device, dtype=particles.dtype)

    # Compute the axis index (dim) in the 3-D particle coordinate for each grid axis
    axis_idx = torch.tensor(grid_axis_indices(axes), device=device, dtype=torch.long)

    # Particle coords projected onto grid axes  (B, N, ndim)
    pts = particles[..., axis_idx]  # (B, N, ndim)

    # Normalise to [0, resolution-1]
    pts_norm = (pts - lo) / (hi - lo) * (res - 1)  # (B, N, ndim)

    occ = torch.zeros([B] + list(resolution), device=device, dtype=torch.float32)

    if footprint_radius > 0.0:
        # Hard disk splat: fill every voxel within footprint_radius of a particle.
        grid_pts = _make_grid_coords(resolution, device)  # (*resolution, ndim)
        r2 = footprint_radius ** 2
        for b in range(B):
            diff  = pts_norm[b].unsqueeze(1).unsqueeze(1) - grid_pts.unsqueeze(0)  # (N, *resolution, ndim)
            dist2 = (diff ** 2).sum(-1)                    # (N, *resolution)
            occ[b] = (dist2 <= r2).any(dim=0).float()
    elif sigma <= 0.0:
        # Hard voxel: scatter-add a '1' to each occupied cell
        idx = pts_norm.round().long().clamp(
            torch.zeros(ndim, device=device, dtype=torch.long),
            (res.long() - 1))
        for b in range(B):
            flat_idx = _ravel_idx(idx[b], resolution)  # (N,)
            occ[b].view(-1).scatter_add_(0, flat_idx, torch.ones(N, device=device))
        # Clamp to [0, 1]
        occ = occ.clamp(0.0, 1.0)
    else:
        # Soft Gaussian splat – expensive but differentiable
        # Build grid of voxel centres  (*resolution, ndim)
        grid_pts = _make_grid_coords(resolution, device)  # (*resolution, ndim)  in voxel idx
        for b in range(B):
            # pts_norm[b]: (N, ndim), grid_pts: (*resolution, ndim)
            # dist2: (N, *resolution)
            diff = pts_norm[b].unsqueeze(1).unsqueeze(1) - grid_pts.unsqueeze(0)  # (N, *resolution, ndim)
            dist2 = (diff ** 2).sum(-1)             # (N, *resolution)
            contrib = torch.exp(-dist2 / (2 * sigma ** 2)).sum(0)  # (*resolution)
            occ[b] = contrib.clamp(0.0, 1.0)

    return occ  # (B, *resolution)


def footprint_radius_voxels(
    particle_size_m: float,
    global_scale: float,
    bounds: Dict[str, float],
    resolution: Tuple[int, ...],
    shape_factor: float = 1.0,
) -> float:
    """
    Convert a particle's real-world footprint size (metres) into a voxel
    radius suitable for ``particles_to_occupancy(..., footprint_radius=...)``.

    ``bounds`` are expressed in normalised camera coords (world metres /
    global_scale, matching ``particles_to_occupancy``'s convention), so
    ``particle_size_m`` is normalised the same way before converting to
    voxels. Uses the grid's first axis scale (the grid is square in
    practice: same span/resolution on x and y).

    ``shape_factor`` scales the base (sphere-equivalent) radius —
    ``particle_size_m / 2``, correct as-is for a sphere of that diameter —
    to account for non-spherical particles. For a cube of edge length
    ``particle_size_m`` viewed from directly overhead, pass
    ``shape_factor=sqrt(2)`` to get the *circumscribed* radius
    (half-diagonal) rather than the inscribed one (half-edge): the
    inscribed radius makes two face-touching cubes' disks exactly tangent
    (zero overlap margin) and under-covers a yaw-rotated cube's corners, so
    once a push clusters particles tightly, the disk union develops gaps a
    true solid silhouette wouldn't have — the resulting occupancy
    under-count grows precisely as clustering (reward) improves. The
    circumscribed radius covers rotated corners under any yaw and gives
    face-touching neighbours a comfortable overlap margin
    (2 * r_circumscribed ≈ 1.41 * edge > edge). Default 1.0 (sphere / no
    adjustment).
    """
    axes = get_grid_axes(len(resolution))
    span = bounds[f'{axes[0]}_max'] - bounds[f'{axes[0]}_min']
    voxels_per_unit = (resolution[0] - 1) / span
    radius_norm = 0.5 * particle_size_m / global_scale * shape_factor
    return radius_norm * voxels_per_unit


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


def action_to_pose(act: torch.Tensor):
    """(..., 4) or (..., 5) push action -> (sx, sy, ex, ey, angle_rad), each (...,).

    4-component actions (``[sx, sy, ex, ey]``) derive the plate yaw
    perpendicular to the push direction — the convention every automated
    sampler/optimizer in this codebase uses (``simple_mpc.action_sampler``,
    ``simple_mpc.sampling_optimizers``, ``env.genesis_env.GenesisEnv``).
    5-component actions carry an explicit 5th component, the plate yaw as a
    normalized ``[0, 1)`` fraction of its pi-periodic orientation range (the
    plate is symmetric under a 180-degree rotation, matching
    ``simple_mpc.action_sampler.PlateCollisionChecker``'s
    ``k * pi / n_angles`` convention), denormalized to radians here. This is
    used by ``simple_mpc.genesis_oracle.GenesisOracleEnv`` (real steps and
    planning rollouts alike) and the human-demo grid search / GUI
    (``simple_mpc.human_mpc``, ``simple_mpc.human_grid_search``,
    ``human_mpc_gui.py``) to decouple tool orientation from travel
    direction — something ``SandboxManipulation.execute_action``'s own
    ``angle`` parameter already supports natively; every other caller in
    this codebase just happens to always derive it from direction instead
    of setting it independently.
    """
    sx, sy, ex, ey = act[..., 0], act[..., 1], act[..., 2], act[..., 3]
    if act.shape[-1] >= 5:
        angle = act[..., 4] * math.pi
    else:
        dxy = torch.hypot(ex - sx, ey - sy)
        angle = torch.where(dxy > 1e-6,
                             torch.atan2(ey - sy, ex - sx) + math.pi / 2,
                             torch.zeros_like(dxy))
    return sx, sy, ex, ey, angle


def genesis_particles_to_cam3d(
    pos_world: torch.Tensor,   # (..., 3) world-frame [x, y, z] metres
    scale: float,
) -> torch.Tensor:
    """Map world-frame particle positions to normalised Genesis camera coords.

    Same convention as ``genesis_action_to_cam3d`` (overhead camera, x/y
    horizontal plane): x_n = x/scale, y_n = -y/scale. The table-plane z is
    fixed at 0.5 since the 2-D occupancy grid (``get_grid_axes(2)``) only
    ever uses the x/y projection — z is carried along only to keep the
    3-vector shape ``particles_to_occupancy`` expects.
    """
    x = pos_world[..., 0:1] / scale
    y = -pos_world[..., 1:2] / scale
    z = torch.full_like(x, 0.5)
    return torch.cat([x, y, z], dim=-1)


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
