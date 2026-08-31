"""Pure functional transform utilities shared across datasets and MPC paths."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

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


# ---------------------------------------------------------------------------
# SE(2) push-frame warp  (Suh & Tedrake 2020, Fig. 4)
# ---------------------------------------------------------------------------
#
# The switched-linear visual-foresight baseline predicts in a canonical frame
# in which every push looks the same: origin at the push midpoint, push
# direction along +x. That collapses the action space from (start, angle,
# length) to (length) alone, which is what lets one operator per length bin
# cover the whole action space.
#
# Conventions, fixed here once so every downstream user reads them from one
# place:
#   * occupancy is (B, H, W) with H = rows, W = cols;
#   * push endpoints are pixel coordinates in (col, row) order — the same
#     order grid_sample's grid uses, and the same order the heuristic push
#     models already receive from `_cam3d_to_grid` (they index rho[y, x] with
#     p0 = (x, y));
#   * grids must be SQUARE, because normalized [-1, 1] coordinates are only
#     isotropic when H == W and a rotation in anisotropic normalized space is
#     a rotation plus an unwanted shear;
#   * align_corners=False throughout, so pixel i's center is at normalized
#     (2i + 1)/N - 1.


def _to_normalized(px: torch.Tensor, n: int) -> torch.Tensor:
    """Pixel coordinate -> align_corners=False normalized coordinate."""
    return (2.0 * px + 1.0) / n - 1.0


def push_frame_transform(start_px: torch.Tensor, end_px: torch.Tensor,
                         grid_res: Tuple[int, int],
                         scale: float = 1.0) -> torch.Tensor:
    """SE(2) transform taking the canonical push frame to image coordinates.

    Returns ``(B, 2, 3)`` suitable for ``torch.nn.functional.affine_grid``,
    which wants the output->input map: a canonical-frame pixel at normalized
    ``(x, y)`` reads the image at ``R(phi) @ (x, y) + m``, where ``phi`` is the
    push direction and ``m`` the normalized push midpoint. So canonical (0, 0)
    is the push midpoint and canonical +x is the push direction.

    Parameters
    ----------
    start_px, end_px : (B, 2) push endpoints in pixels, (col, row) order.
    grid_res         : (H, W), must be square.
    scale            : fraction of the image the canonical window spans.
                       1.0 warps the whole image (what the paper does).
                       Smaller CROPS to a window around the push, which is
                       usually what you want: the push only affects a
                       neighbourhood of itself, so a full-image operator spends
                       almost all of its parameters modelling the identity.
                       Cropping shrinks the operator by 1/scale**4 and is what
                       makes a well-determined fit possible from modest data.
    """
    H, W = int(grid_res[0]), int(grid_res[1])
    if H != W:
        raise ValueError(f"push_frame_transform needs a square grid, got {(H, W)}")

    mid = 0.5 * (start_px + end_px)
    delta = end_px - start_px
    phi = torch.atan2(delta[:, 1], delta[:, 0])          # (col, row) frame
    cos, sin = torch.cos(phi), torch.sin(phi)

    mx = _to_normalized(mid[:, 0], W)
    my = _to_normalized(mid[:, 1], H)

    f = float(scale)
    theta = torch.stack([
        torch.stack([f * cos, -f * sin, mx], dim=-1),
        torch.stack([f * sin,  f * cos, my], dim=-1),
    ], dim=-2)                                            # (B, 2, 3)
    return theta


def invert_affine(theta: torch.Tensor) -> torch.Tensor:
    """Inverse of a ``(B, 2, 3)`` affine transform: [R|t] -> [R^-1 | -R^-1 t].

    A true inverse rather than the transpose shortcut, because
    `push_frame_transform(scale=...)` makes R a scaled rotation, for which
    R^-1 != R^T.
    """
    R, t = theta[:, :, :2], theta[:, :, 2:]
    Ri = torch.linalg.inv(R)
    return torch.cat([Ri, -Ri @ t], dim=-1)


def warp_affine_occ(occ: torch.Tensor, theta: torch.Tensor,
                    out_res: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Resample ``occ`` (B, H, W) through the output->input map ``theta``.

    Differentiable in both `occ` and `theta` (plain ``grid_sample``), so a
    model built on this stays usable by the gradient-descent MPC even though
    the paper's controller only enumerates.
    """
    B, H, W = occ.shape
    oh, ow = (H, W) if out_res is None else (int(out_res[0]), int(out_res[1]))
    grid = torch.nn.functional.affine_grid(
        theta, (B, 1, oh, ow), align_corners=False)
    out = torch.nn.functional.grid_sample(
        occ.unsqueeze(1), grid, mode='bilinear',
        padding_mode='zeros', align_corners=False)
    return out.squeeze(1)


def to_push_frame(occ: torch.Tensor, start_px: torch.Tensor,
                  end_px: torch.Tensor,
                  out_res: Optional[Tuple[int, int]] = None,
                  scale: float = 1.0) -> torch.Tensor:
    """Warp occupancy into the canonical push frame (their ``T(I)``)."""
    theta = push_frame_transform(start_px, end_px, occ.shape[-2:], scale)
    return warp_affine_occ(occ, theta, out_res)


def from_push_frame(occ_canon: torch.Tensor, start_px: torch.Tensor,
                    end_px: torch.Tensor,
                    out_res: Tuple[int, int],
                    scale: float = 1.0) -> torch.Tensor:
    """Warp a canonical-frame image back to image coordinates (``T^-1``)."""
    theta = push_frame_transform(start_px, end_px, out_res, scale)
    return warp_affine_occ(occ_canon, invert_affine(theta), out_res)


def push_frame_validity_mask(start_px: torch.Tensor, end_px: torch.Tensor,
                             grid_res: Tuple[int, int],
                             canon_res: Optional[Tuple[int, int]] = None,
                             scale: float = 1.0) -> torch.Tensor:
    """Their ``M = T^-1(T(1))`` — where a round trip preserves information.

    Rotating a square image inside a same-sized square loses the corners, so
    the prediction is only trustworthy where this mask is ~1; elsewhere the
    caller keeps the original image (see `blend_push_prediction`).
    """
    B = start_px.shape[0]
    H, W = int(grid_res[0]), int(grid_res[1])
    ones = torch.ones((B, H, W), dtype=start_px.dtype, device=start_px.device)
    canon = to_push_frame(ones, start_px, end_px, canon_res, scale)
    return from_push_frame(canon, start_px, end_px, (H, W), scale)


def blend_push_prediction(pred: torch.Tensor, occ: torch.Tensor,
                          mask: torch.Tensor,
                          threshold: float = 0.5) -> torch.Tensor:
    """Recombine a warped prediction with the original image (their Fig. 4).

    A hard threshold on the validity mask rather than a soft alpha blend: the
    mask is ~1 in the interior and ~0 outside, and soft-blending its thin
    bilinear ramp would darken a one-pixel ring at the boundary on every
    single step — which compounds over a closed-loop episode.
    """
    keep = (mask >= threshold).to(pred.dtype)
    return keep * pred + (1.0 - keep) * occ
