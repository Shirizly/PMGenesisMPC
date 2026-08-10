"""Tests for Genesis/placement_sampling.py — configuration-space sampling of
collision-free tool touchdown poses.

Genesis-free: the module depends only on torch/numpy/scipy.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from placement_sampling import (  # noqa: E402
    build_occupancy, clearance_map, free_placements, sample_free_placements,
    _rotated_rect_kernel,
)

BOX = (0.128, 0.128)
RES = 0.002
TOOL_L, TOOL_W = 0.04, 0.002


def _occupancy(positions, half=0.004, n_envs=1):
    pos = torch.tensor(positions, dtype=torch.float32).view(n_envs, -1, 2)
    half_xy = torch.full((pos.shape[1], 2), half)
    return build_occupancy(pos, half_xy, BOX, RES)


def test_empty_box_is_entirely_free():
    # one particle parked far outside the tray
    occ, meta = _occupancy([[10.0, 10.0]])
    assert not occ.any()

    angles = torch.tensor([0.0])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)
    # everything except the wall margin should be available
    assert free.float().mean() > 0.5


def test_particle_blocks_placements_around_itself():
    occ, meta = _occupancy([[0.0, 0.0]])
    assert occ.any()

    angles = torch.tensor([0.0])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)

    centre_row, centre_col = meta["H"] // 2, meta["W"] // 2
    assert not free[0, 0, centre_row, centre_col], "tool centre on a particle must be blocked"
    # a horizontal tool is blocked far away in x but not in y
    dx = int(round(0.015 / RES))
    dy = int(round(0.015 / RES))
    assert not free[0, 0, centre_row, centre_col + dx], "blocked along the blade"
    assert free[0, 0, centre_row + dy, centre_col], "free perpendicular to the blade"


def test_orientation_changes_the_free_set():
    """A long thin tool must be orientation-sensitive, or the C-space is wrong."""
    occ, meta = _occupancy([[0.0, 0.0]])
    angles = torch.tensor([0.0, math.pi / 2])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)

    r, c = meta["H"] // 2, meta["W"] // 2
    off = int(round(0.015 / RES))
    # offset along +x: blocked when the blade lies along x, free when across
    assert not free[0, 0, r, c + off]
    assert free[0, 1, r, c + off]


def test_wall_margin_excludes_poses_that_poke_through():
    occ, meta = _occupancy([[10.0, 10.0]])          # empty tray
    angles = torch.tensor([0.0])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)

    # with the blade along x, its half-length is 0.02 -> the outer 0.02 m of
    # the x range must be unusable
    margin_cells = int(math.ceil((TOOL_L / 2) / RES))
    assert not free[0, 0, :, :margin_cells].any()
    assert not free[0, 0, :, meta["W"] - margin_cells:].any()


def test_clearance_is_zero_on_obstacles_and_positive_in_free_space():
    occ, meta = _occupancy([[0.0, 0.0]])
    dist = clearance_map(occ, meta)

    r, c = meta["H"] // 2, meta["W"] // 2
    assert dist[0, r, c] == pytest.approx(0.0)
    assert dist[0, 0, 0] > 0.0
    # a cell ~10 mm from a 4 mm-half particle should read roughly that far
    probe = dist[0, r, c + int(round(0.02 / RES))]
    assert 0.010 < float(probe) < 0.020


def test_fully_blocked_env_reports_not_ok():
    """The documented degradation path: dense pile -> caller falls back."""
    occ, meta = _occupancy([[10.0, 10.0]])
    occ[:] = True                                   # nothing is free anywhere
    angles = torch.tensor([0.0])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)
    assert not free.any()

    xy, yaw, ok = sample_free_placements(free, meta, angles, n_samples=5)
    assert ok.shape == (1, 5)
    assert not ok.any(), "a blocked env must report ok=False, not raise"
    assert torch.isfinite(xy).all(), "outputs must stay finite so callers can mask"


def test_sampled_placements_land_in_free_space():
    occ, meta = _occupancy([[0.0, 0.0], [0.03, 0.02], [-0.025, 0.03]])
    angles = torch.tensor([0.0, math.pi / 4, math.pi / 2])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)

    xy, yaw, ok = sample_free_placements(free, meta, angles, n_samples=64)
    assert ok.all()

    # every sample must map back to a free cell of its own orientation
    col = ((xy[..., 0] + BOX[0] / 2) / RES).long().clamp(0, meta["W"] - 1)
    row = ((xy[..., 1] + BOX[1] / 2) / RES).long().clamp(0, meta["H"] - 1)
    a_i = torch.stack([(angles - y).abs().argmin() for y in yaw.flatten()]).view(yaw.shape)
    assert free[0, a_i[0], row[0], col[0]].all()


def test_sampled_yaws_come_from_the_requested_bins():
    occ, meta = _occupancy([[0.0, 0.0]])
    angles = torch.tensor([-0.5, 0.0, 0.7])
    free = free_placements(occ, meta, angles, TOOL_L, TOOL_W)
    _, yaw, ok = sample_free_placements(free, meta, angles, n_samples=32)
    assert torch.isin(yaw[ok], angles).all()


def test_rotated_kernel_covers_the_tool_area():
    kernel = _rotated_rect_kernel(TOOL_L, TOOL_W, 0.0, RES, torch.device("cpu"))
    assert kernel.sum() > 0
    # rotating by 90 deg transposes the footprint
    k90 = _rotated_rect_kernel(TOOL_L, TOOL_W, math.pi / 2, RES, torch.device("cpu"))
    assert torch.allclose(kernel.sum(), k90.sum(), rtol=0.15)
    assert kernel.shape == k90.shape


def test_multi_env_grids_are_independent():
    pos = torch.tensor([[[0.0, 0.0]], [[10.0, 10.0]]], dtype=torch.float32)
    half_xy = torch.full((1, 2), 0.004)
    occ, meta = build_occupancy(pos, half_xy, BOX, RES)
    assert occ[0].any() and not occ[1].any()


def test_active_limits_which_particles_are_rasterized():
    """Parked particles must not be treated as obstacles inside the tray."""
    pos = torch.tensor([[[0.0, 0.0], [0.02, 0.02]]], dtype=torch.float32)
    half_xy = torch.full((2, 2), 0.004)
    full, _ = build_occupancy(pos, half_xy, BOX, RES)
    limited, _ = build_occupancy(pos, half_xy, BOX, RES, active=1)
    assert full.sum() > limited.sum()
