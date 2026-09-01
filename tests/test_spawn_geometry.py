"""Tests for Genesis/spawn_geometry.py — pyramid spawn layouts. Genesis-free."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from spawn_geometry import pyramid_layer_plan, pyramid_positions  # noqa: E402

SIZE = 0.005


def test_every_cube_is_placed():
    """Parking the remainder was the first design and it wrecked the
    measurement — parked cubes either fall through the floor or register as a
    spurious bottom layer. So the plan must account for all n."""
    for n in (1, 5, 14, 30, 50, 80, 137):
        assert sum(pyramid_layer_plan(n)) == n
        pos, _ = pyramid_positions(n, SIZE)
        assert pos.shape == (n, 3)


def test_layer_plan_is_a_complete_pyramid_with_the_remainder_in_the_base():
    assert pyramid_layer_plan(14) == [9, 4, 1]
    assert pyramid_layer_plan(30) == [16, 9, 4, 1]
    assert pyramid_layer_plan(50) == [36, 9, 4, 1]     # 30 complete + 20 in base
    assert pyramid_layer_plan(55) == [25, 16, 9, 4, 1]


def test_fifty_cubes_gives_four_layers():
    """50 was asked about specifically: it comfortably supports real depth."""
    pos, n_layers = pyramid_positions(50, SIZE)
    assert n_layers == 4
    assert len(torch.unique(torch.round(pos[:, 2] / SIZE))) == 4


def test_layer_count_is_monotone_in_n():
    """Caught a real wart: choosing the base from the smallest full-pyramid sum
    that REACHES n and truncating gave 5 layers at n=55 and only 2 at n=56."""
    prev = 0
    for n in range(1, 200):
        _, k = pyramid_positions(n, SIZE)
        assert k >= prev, f"layer count fell from {prev} to {k} at n={n}"
        prev = k


def test_bottom_layer_rests_on_the_floor():
    floor = 0.01
    pos, _ = pyramid_positions(30, SIZE, floor_z=floor)
    lowest = float(pos[:, 2].min())
    assert lowest == pytest.approx(floor + SIZE / 2, abs=1e-4), (
        "bottom cube centres must sit half a cube above the floor, or they "
        "interpenetrate it and get ejected")


def test_no_two_cubes_overlap():
    pos, _ = pyramid_positions(50, SIZE)
    d = (pos[:, None, :] - pos[None, :, :]).abs()
    same = torch.eye(len(pos), dtype=torch.bool)
    # Non-overlapping means separated by >= one cube edge on some axis.
    clear = (d >= SIZE * 0.99).any(dim=-1)
    assert bool((clear | same).all()), "cubes are born interpenetrating"


def test_each_layer_is_centred():
    pos, _ = pyramid_positions(30, SIZE, centre=(0.02, -0.01))
    for z in torch.unique(pos[:, 2]):
        layer = pos[pos[:, 2] == z]
        assert torch.allclose(layer[:, :2].mean(0),
                              torch.tensor([0.02, -0.01]), atol=1e-6)


def test_is_taller_and_narrower_than_a_flat_layer():
    """The whole point: depth in a small footprint."""
    pos, k = pyramid_positions(50, SIZE)
    span_z = float(pos[:, 2].max() - pos[:, 2].min())
    span_xy = float(pos[:, :2].max(0).values.sub(pos[:, :2].min(0).values).max())
    assert span_z == pytest.approx((k - 1) * SIZE, rel=0.05)
    # 50 cubes in one layer need a ~7x7 footprint; the pyramid's base is 6x6.
    assert span_xy < 7 * SIZE


def test_empty_input():
    assert pyramid_layer_plan(0) == []
