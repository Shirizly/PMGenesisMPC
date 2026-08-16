"""Tests for Genesis/action_sampling.py — batch-aware action shaping.

Genesis-free: the module is pure torch geometry.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from action_sampling import (  # noqa: E402
    equalize_travel_distance, shared_batch_distance,
)

LOW = torch.tensor([-0.05, -0.05])
HIGH = torch.tensor([0.05, 0.05])


def _bounds(shape):
    return LOW.expand(*shape, 2).clone(), HIGH.expand(*shape, 2).clone()


def test_equalized_pushes_all_travel_the_target_distance():
    starts = torch.tensor([[0.0, 0.0], [-0.01, 0.02], [0.03, -0.03]])
    stops = torch.tensor([[0.01, 0.0], [-0.01, 0.03], [0.01, -0.03]])
    low, high = _bounds((3,))
    target = torch.full((3, 1), 0.02)

    new_stops, clipped = equalize_travel_distance(starts, stops, low, high, target)

    travelled = (new_stops - starts).norm(dim=-1)
    assert torch.allclose(travelled, torch.full((3,), 0.02), atol=1e-6)
    assert not clipped.any()


def test_direction_is_preserved():
    starts = torch.tensor([[0.0, 0.0]])
    stops = torch.tensor([[0.006, 0.008]])          # direction (0.6, 0.8)
    low, high = _bounds((1,))

    new_stops, _ = equalize_travel_distance(
        starts, stops, low, high, torch.tensor([[0.02]]))

    unit_before = (stops - starts) / (stops - starts).norm()
    unit_after = (new_stops - starts) / (new_stops - starts).norm()
    assert torch.allclose(unit_before, unit_after, atol=1e-6)


def test_pushes_stay_inside_their_box():
    """A target longer than the box must truncate at the boundary, not escape."""
    starts = torch.tensor([[0.04, 0.0]])            # near the +x wall
    stops = torch.tensor([[0.045, 0.0]])            # heading further +x
    low, high = _bounds((1,))

    new_stops, clipped = equalize_travel_distance(
        starts, stops, low, high, torch.tensor([[0.5]]))

    assert clipped.all(), "an unreachable target must be reported as clipped"
    assert new_stops[0, 0] <= 0.05 + 1e-9
    assert torch.allclose(new_stops, torch.tensor([[0.05, 0.0]]), atol=1e-6)


def test_shrinking_is_always_possible():
    """A target shorter than the original never needs clipping."""
    g = torch.Generator().manual_seed(0)
    starts = (torch.rand(64, 2, generator=g) - 0.5) * 0.09
    stops = (torch.rand(64, 2, generator=g) - 0.5) * 0.09
    low, high = _bounds((64,))
    dist = (stops - starts).norm(dim=-1, keepdim=True)

    _, clipped = equalize_travel_distance(starts, stops, low, high, dist * 0.5)
    assert not clipped.any()


def test_shared_batch_distance_takes_one_envs_draw():
    """A summary statistic would collapse between-batch variation as envs grow."""
    dist = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])  # (envs, samples, 1)
    shared = shared_batch_distance(dist)

    assert shared.shape == (1, 2, 1)
    assert torch.allclose(shared[0], dist[0]), "must be env 0's own draw"
    # broadcasting it gives every env the same per-sample distance
    assert torch.allclose(shared.expand_as(dist)[:, 0, 0],
                          torch.full((3,), 1.0))


def test_equalization_makes_the_batch_maximum_equal_the_target():
    """The point of the exercise: sweep_steps follows the batch maximum."""
    g = torch.Generator().manual_seed(1)
    starts = (torch.rand(16, 2, generator=g) - 0.5) * 0.05
    stops = (torch.rand(16, 2, generator=g) - 0.5) * 0.05
    low, high = _bounds((16,))
    dist = (stops - starts).norm(dim=-1, keepdim=True)

    before_spread = float(dist.max() - dist.min())
    target = torch.full_like(dist, float(dist.min()))
    new_stops, _ = equalize_travel_distance(starts, stops, low, high, target)
    after = (new_stops - starts).norm(dim=-1)

    assert before_spread > 1e-3, "fixture should have varied distances"
    assert float(after.max() - after.min()) < 1e-6
