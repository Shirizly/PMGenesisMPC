"""
Fast, data-free unit tests for the footprint-radius particle splatting added
to transforms.functional for simple_mpc.genesis_oracle
(docs/oracle_mpc_plan.md §1.3 occupancy-density caveat).
"""

import torch

from transforms.functional import (
    particles_to_occupancy,
    footprint_radius_voxels,
    genesis_particles_to_cam3d,
)

BOUNDS = {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0,
          "z_min": 0.0, "z_max": 1.0}
RES = (32, 32)


def test_footprint_radius_zero_matches_hard_voxel_path():
    pts = torch.tensor([[[0.1, 0.2, 0.5], [-0.3, 0.4, 0.5]]])   # (1, 2, 3)
    occ_hard = particles_to_occupancy(pts, BOUNDS, RES, sigma=0.0)
    occ_footprint_zero = particles_to_occupancy(pts, BOUNDS, RES, footprint_radius=0.0)
    assert torch.equal(occ_hard, occ_footprint_zero)


def test_footprint_radius_fills_more_voxels_than_hard_voxel():
    pts = torch.tensor([[[0.0, 0.0, 0.5]]])   # (1, 1, 3) single particle at center
    occ_hard = particles_to_occupancy(pts, BOUNDS, RES, sigma=0.0)
    occ_footprint = particles_to_occupancy(pts, BOUNDS, RES, footprint_radius=3.0)

    assert occ_hard.sum().item() == 1.0
    assert occ_footprint.sum().item() > occ_hard.sum().item()
    # every filled hard voxel must also be filled in the footprint version
    assert bool(((occ_hard > 0) & (occ_footprint == 0)).sum().item() == 0)


def test_footprint_values_are_binary():
    pts = torch.rand(3, 5, 3) * 2.0 - 1.0
    occ = particles_to_occupancy(pts, BOUNDS, RES, footprint_radius=2.0)
    assert set(torch.unique(occ).tolist()) <= {0.0, 1.0}


def test_footprint_radius_voxels_scales_with_particle_size():
    global_scale = 0.6
    r_small = footprint_radius_voxels(0.005, global_scale, BOUNDS, RES)
    r_large = footprint_radius_voxels(0.010, global_scale, BOUNDS, RES)
    assert r_large > r_small > 0.0
    assert abs(r_large - 2.0 * r_small) < 1e-6


def test_genesis_particles_to_cam3d_matches_action_convention():
    from transforms.functional import genesis_action_to_cam3d

    scale = 0.6
    action = torch.tensor([[0.05, -0.02, 0.03, 0.01]])   # (1, 4) [sx,sy,ex,ey]
    s_cam, e_cam = genesis_action_to_cam3d(action, scale)

    pos_world = torch.tensor([[[0.05, -0.02, 0.02]]])    # (1, 1, 3) matches sx,sy
    pos_cam = genesis_particles_to_cam3d(pos_world, scale)

    assert torch.allclose(pos_cam[0, 0, :2], s_cam[0, :2], atol=1e-6)
    assert pos_cam[0, 0, 2].item() == 0.5
