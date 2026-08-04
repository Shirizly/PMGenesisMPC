"""
Fast, Genesis-free unit tests for simple_mpc.human_grid_search.build_action_grid.

grid_search_refine itself needs a real GenesisOracleEnv and is exercised by
hand/smoke-tested instead (see docs/human_demo_design.md) — only the pure
grid-construction logic is covered here.
"""

import numpy as np

from simple_mpc.human_grid_search import build_action_grid, ACTION_DIM

CLIP_LO = np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
CLIP_HI = np.array([ 1.0,  1.0,  1.0,  1.0, 1.0], dtype=np.float32)
CENTER  = np.array([0.2, -0.1, 0.3, 0.0, 0.5], dtype=np.float32)


def test_grid_shape_and_dim():
    grid = build_action_grid(CENTER, grid_n=3, delta=0.1, clip_lo=CLIP_LO, clip_hi=CLIP_HI)
    assert grid.shape == (3 ** ACTION_DIM, ACTION_DIM)


def test_grid_n_one_degenerates_to_center_point():
    grid = build_action_grid(CENTER, grid_n=1, delta=0.5, clip_lo=CLIP_LO, clip_hi=CLIP_HI)
    assert grid.shape == (1, ACTION_DIM)
    assert np.allclose(grid[0], CENTER)


def test_grid_centered_on_input_action():
    grid = build_action_grid(CENTER, grid_n=5, delta=0.1, clip_lo=CLIP_LO, clip_hi=CLIP_HI)
    # the grid is symmetric around center per-dimension, so the mean of each
    # axis' distinct values should recover the center (no clipping active here)
    for i in range(ACTION_DIM):
        assert abs(grid[:, i].mean() - CENTER[i]) < 1e-5


def test_grid_respects_bounds_even_with_large_delta():
    grid = build_action_grid(CENTER, grid_n=7, delta=5.0, clip_lo=CLIP_LO, clip_hi=CLIP_HI)
    assert (grid >= CLIP_LO - 1e-6).all()
    assert (grid <= CLIP_HI + 1e-6).all()


def test_per_dimension_delta_widens_only_that_axis():
    delta = np.array([0.05, 0.05, 0.05, 0.05, 0.4], dtype=np.float32)
    grid = build_action_grid(CENTER, grid_n=5, delta=delta, clip_lo=CLIP_LO, clip_hi=CLIP_HI)
    spread = grid.max(axis=0) - grid.min(axis=0)
    assert spread[4] > spread[0]   # angle dim spans much more than position dims


def test_scalar_delta_broadcasts_to_all_dims():
    grid_scalar = build_action_grid(CENTER, grid_n=3, delta=0.2, clip_lo=CLIP_LO, clip_hi=CLIP_HI)
    grid_vector = build_action_grid(
        CENTER, grid_n=3, delta=np.full(ACTION_DIM, 0.2, dtype=np.float32),
        clip_lo=CLIP_LO, clip_hi=CLIP_HI)
    assert np.allclose(grid_scalar, grid_vector)


def test_invalid_shapes_raise():
    import pytest
    with pytest.raises(ValueError):
        build_action_grid(np.zeros(4), grid_n=3, delta=0.1, clip_lo=CLIP_LO, clip_hi=CLIP_HI)
    with pytest.raises(ValueError):
        build_action_grid(CENTER, grid_n=3, delta=np.zeros(3), clip_lo=CLIP_LO, clip_hi=CLIP_HI)
