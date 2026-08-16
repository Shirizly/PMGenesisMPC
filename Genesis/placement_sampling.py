"""
Genesis/placement_sampling.py — configuration-space sampling of tool placements
that do not collide with the pile at touchdown.

The problem
-----------
``execute_action`` lowers the plate onto ``p_start`` before sweeping. The blind
sampler in ``generate_action_samples`` picks ``p_start`` from the box interior
without looking at where the particles are, so the plate routinely descends
*into* a cube. The solver resolves that overlap by ejecting the particle, which
is not a push — it is an artifact, and it is recorded as though it were a
normal transition.

The approach
------------
Work in the tool's configuration space. The plate is a rectangle
(``length x width``) at yaw ``theta``; a placement ``(x, y, theta)`` is safe iff
that rectangle, inflated by a clearance margin, contains no particle centre.
Equivalently: rasterize the particles into an occupancy grid, then for each
discretized ``theta`` dilate the occupancy by the *rotated tool rectangle* —
the Minkowski sum that turns "does the tool overlap an obstacle" into "is the
tool's centre inside a forbidden region". What remains is exactly the free
placement set for that orientation, and any cell in it can be sampled directly.

The Euclidean distance transform of the occupancy serves two purposes: it gives
a cheap orientation-independent pre-filter (a placement whose centre is farther
from every obstacle than the tool's circumscribed radius is safe at *any* yaw),
and it provides a clearance value used to bias sampling toward roomier spots
rather than ones that merely squeak past.

Graceful degradation is the point
---------------------------------
As the pile grows, the free set shrinks and eventually empties — at 200
particles there may be no collision-free placement at all for some
orientations. Every entry point here reports which samples it could satisfy and
leaves the rest to the caller, so ``generate_action_samples`` can fall back to
its blind behaviour per-sample rather than failing or looping forever.
"""

from __future__ import annotations

import math

import numpy as np
import torch

try:                                     # SciPy's EDT is exact and fast enough
    from scipy.ndimage import distance_transform_edt as _edt
except ImportError:                      # pragma: no cover
    _edt = None


def build_occupancy(positions: torch.Tensor, half_extents_xy: torch.Tensor,
                    box_xy: tuple[float, float], resolution: float,
                    active: int | None = None) -> tuple[torch.Tensor, dict]:
    """Rasterize particle footprints into a per-env occupancy grid.

    positions       : (n_envs, n_particles, 2 or 3) particle centres, metres
    half_extents_xy : (n_particles, 2) half footprint, metres
    box_xy          : interior (width, depth) in metres
    resolution      : cell size in metres

    Returns (occupancy [n_envs, H, W] bool, grid metadata).

    Footprints are painted as axis-aligned boxes of the particle's *conservative*
    xy half-extent. That is deliberately pessimistic for a rotated cube, which
    matches how ``shuffle_particles`` reasons about overlap and keeps the free
    set on the safe side.
    """
    width, depth = box_xy
    n_envs, n_particles = positions.shape[0], positions.shape[1]
    if active is not None:
        n_particles = min(n_particles, active)
    pos = positions[:, :n_particles, :2]
    half = half_extents_xy[:n_particles].to(pos.device)

    W = max(1, int(round(width / resolution)))
    H = max(1, int(round(depth / resolution)))
    grid = torch.zeros((n_envs, H, W), dtype=torch.bool, device=pos.device)

    # cell centres
    xs = (torch.arange(W, device=pos.device, dtype=pos.dtype) + 0.5) * resolution - width / 2
    ys = (torch.arange(H, device=pos.device, dtype=pos.dtype) + 0.5) * resolution - depth / 2

    # A particle occupies cells whose centre lies within its half-extent.
    # Done per particle to keep the peak allocation at (n_envs, H, W) rather
    # than (n_envs, n_particles, H, W), which would be ~200x larger at n=200.
    for i in range(n_particles):
        dx = (xs.view(1, 1, W) - pos[:, i, 0].view(n_envs, 1, 1)).abs()
        dy = (ys.view(1, H, 1) - pos[:, i, 1].view(n_envs, 1, 1)).abs()
        grid |= (dx <= half[i, 0]) & (dy <= half[i, 1])

    meta = {"resolution": resolution, "H": H, "W": W,
            "width": width, "depth": depth}
    return grid, meta


def clearance_map(occupancy: torch.Tensor, meta: dict) -> torch.Tensor:
    """Euclidean distance (metres) from each free cell to the nearest obstacle.

    Cells that are themselves occupied get distance 0. Uses SciPy's exact EDT
    per env — the grids are small (128x128 at 1 mm) and this runs once per
    action-sampling call, not per simulation step, so the host round-trip is
    immaterial next to a settle.
    """
    occ = occupancy.detach().cpu().numpy()
    out = np.zeros(occ.shape, dtype=np.float32)
    for b in range(occ.shape[0]):
        if occ[b].all():
            continue                      # fully blocked -> all zeros
        if _edt is None:                  # pragma: no cover
            out[b] = np.where(occ[b], 0.0, np.inf)
        else:
            out[b] = _edt(~occ[b], sampling=meta["resolution"])
    return torch.from_numpy(out).to(occupancy.device)


def _rotated_rect_kernel(length: float, width: float, angle: float,
                         resolution: float, device) -> torch.Tensor:
    """Binary structuring element for a rectangle of given size at ``angle``."""
    half_l, half_w = length / 2.0, width / 2.0
    reach = math.hypot(half_l, half_w)
    r = max(1, int(math.ceil(reach / resolution)))
    n = 2 * r + 1
    coords = (torch.arange(n, device=device, dtype=torch.float32) - r) * resolution
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    c, s = math.cos(angle), math.sin(angle)
    # rotate the query point into the rectangle's own frame
    u = c * xx + s * yy
    v = -s * xx + c * yy
    return ((u.abs() <= half_l) & (v.abs() <= half_w)).float()


def free_placements(occupancy: torch.Tensor, meta: dict, angles: torch.Tensor,
                    tool_length: float, tool_width: float,
                    clearance: float = 0.0,
                    wall_margin: float = 0.0) -> torch.Tensor:
    """Free tool-centre positions per orientation.

    Returns (n_envs, n_angles, H, W) bool: True where the tool, at that yaw,
    can be lowered without its footprint touching any particle.

    Implemented as a dilation of the occupancy by the rotated tool rectangle
    (Minkowski sum), which is what turns an overlap test into a point-in-set
    test in configuration space.

    ``clearance`` inflates the tool against PARTICLES; ``wall_margin`` keeps it
    clear of the WALLS. They are separate because they answer different
    questions, and because the blind sampler in ``generate_action_samples``
    applies a wall margin of its own — leaving this at 0 makes placement-aware
    sampling draw touchdowns closer to the rim than the blind sampler ever
    would, which is a distribution shift rather than a refinement.
    """
    n_envs = occupancy.shape[0]
    occ = occupancy.float().unsqueeze(1)              # (n_envs, 1, H, W)
    masks = []
    for angle in angles.tolist():
        kernel = _rotated_rect_kernel(
            tool_length + 2 * clearance, tool_width + 2 * clearance,
            angle, meta["resolution"], occupancy.device)
        pad = kernel.shape[-1] // 2
        # max-filter == binary dilation for a 0/1 kernel
        dilated = torch.nn.functional.conv2d(
            occ, kernel.view(1, 1, *kernel.shape), padding=pad)
        masks.append(dilated[:, 0] <= 0.0)
    free = torch.stack(masks, dim=1)                  # (n_envs, n_angles, H, W)

    # The tool must also stay inside the box: forbid centres whose rectangle
    # would poke through a wall. Done by clearing a margin equal to the tool's
    # extent along each axis at that yaw.
    H, W, res = meta["H"], meta["W"], meta["resolution"]
    for a_i, angle in enumerate(angles.tolist()):
        ext_x = (abs(math.cos(angle)) * tool_length / 2
                 + abs(math.sin(angle)) * tool_width / 2 + wall_margin)
        ext_y = (abs(math.sin(angle)) * tool_length / 2
                 + abs(math.cos(angle)) * tool_width / 2 + wall_margin)
        mx, my = int(math.ceil(ext_x / res)), int(math.ceil(ext_y / res))
        if mx:
            free[:, a_i, :, :mx] = False
            free[:, a_i, :, W - mx:] = False
        if my:
            free[:, a_i, :my, :] = False
            free[:, a_i, H - my:, :] = False
    return free


def sample_free_placements(free: torch.Tensor, meta: dict, angles: torch.Tensor,
                           n_samples: int, generator: torch.Generator | None = None,
                           clearance: torch.Tensor | None = None,
                           clearance_bias: float = 0.0):
    """Draw ``n_samples`` placements per env uniformly from the free set.

    Returns (xy [n_envs, n_samples, 2], yaw [n_envs, n_samples],
             ok [n_envs, n_samples] bool). Entries with ok=False had no free
    placement available and must be filled in by the caller — this is the
    documented degradation path for dense piles.

    ``clearance_bias`` > 0 weights the draw by clearance**bias, preferring
    roomier placements over ones that only just fit.
    """
    n_envs, n_angles, H, W = free.shape
    flat = free.reshape(n_envs, -1).float()

    if clearance_bias > 0.0 and clearance is not None:
        w = clearance.clamp(min=0.0).unsqueeze(1).expand(n_envs, n_angles, H, W)
        flat = flat * (w.reshape(n_envs, -1) ** clearance_bias + 1e-9)

    totals = flat.sum(dim=1)
    ok_env = totals > 0
    # torch.multinomial rejects all-zero rows, so give blocked envs a dummy
    # uniform row and mask their draws out afterwards.
    safe = torch.where(ok_env.view(-1, 1), flat, torch.ones_like(flat))
    idx = torch.multinomial(safe, n_samples, replacement=True, generator=generator)

    a_i = idx // (H * W)
    rem = idx % (H * W)
    row, col = rem // W, rem % W

    res = meta["resolution"]
    # jitter within the chosen cell so placements are continuous, not snapped
    jitter = torch.rand((n_envs, n_samples, 2), device=free.device,
                        generator=generator) - 0.5
    x = (col.float() + 0.5) * res - meta["width"] / 2 + jitter[..., 0] * res
    y = (row.float() + 0.5) * res - meta["depth"] / 2 + jitter[..., 1] * res

    yaw = angles.to(free.device)[a_i]
    ok = ok_env.view(-1, 1).expand(n_envs, n_samples)
    return torch.stack((x, y), dim=-1), yaw, ok
