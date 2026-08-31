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


# ---------------------------------------------------------------------------
# Action-space restriction: perpendicular pushes and fixed push length
# ---------------------------------------------------------------------------

from action_sampling import (  # noqa: E402
    blade_normal, constrain_push, relative_blade_angle, sampling_box,
)

# The tray/blade the collection configs actually use (Genesis/configs/basic.yaml).
VOL = [0.27, 0.27, 0.1]
TOOL_L, TOOL_W, MARGIN = 0.04, 0.002, 0.02


def _box(angles):
    return sampling_box(angles, VOL, TOOL_L, TOOL_W, MARGIN)


def _random_angles(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (-torch.pi / 2) + torch.rand(n, generator=g) * torch.pi


def _random_starts(angles, seed=1):
    """Uniform in each entry's own yaw-dependent box, as the sampler draws."""
    low, high = _box(angles)
    g = torch.Generator().manual_seed(seed)
    return low + (high - low) * torch.rand(angles.shape + (2,), generator=g)


def test_sampling_box_matches_the_original_inline_formula():
    """The extracted helper must not move any existing dataset's bounds."""
    angles = _random_angles(32)
    low, high = _box(angles)

    space_x = VOL[0] / 2 - (torch.cos(angles) * TOOL_L / 2
                            + abs(torch.sin(angles)) * TOOL_W / 2 + MARGIN)
    space_y = VOL[1] / 2 - (abs(torch.sin(angles)) * TOOL_L / 2
                            + torch.cos(angles) * TOOL_W / 2 + MARGIN)
    assert torch.allclose(high, torch.stack([space_x, space_y], dim=1), atol=1e-9)
    assert torch.allclose(low, -high, atol=1e-9)


def test_blade_normal_is_perpendicular_to_the_blade_axis():
    angles = _random_angles(16)
    axis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    n_hat = blade_normal(angles)

    assert torch.allclose((axis * n_hat).sum(-1), torch.zeros(16), atol=1e-6)
    assert torch.allclose(n_hat.norm(dim=-1), torch.ones(16), atol=1e-6)


def test_perpendicular_pushes_are_perpendicular():
    angles = _random_angles(256)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=2)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high, perpendicular=True)

    assert relative_blade_angle(push.starts_xy, push.stops_xy, angles).max() < 1e-6


def test_perpendicular_preserves_the_push_length_distribution():
    """Only the direction is replaced — v1 datasets stay length-comparable."""
    angles = _random_angles(256)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=3)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high, perpendicular=True)

    before = (stops - starts).norm(dim=-1)
    after = (push.stops_xy - push.starts_xy).norm(dim=-1)
    # The start may be nudged to make the drawn length fit along the normal;
    # what must survive is the LENGTH, so v1 datasets stay comparable to the
    # unrestricted ones already collected.
    keep = ~push.truncated
    assert torch.allclose(before[keep], after[keep], atol=1e-6)
    assert keep.float().mean() > 0.9, (
        "nudging the start should rescue nearly every draw; only lengths "
        "exceeding the tray extent along the normal may truncate")


def test_perpendicular_pushes_go_both_ways():
    """Yaw is drawn from (-pi/2, pi/2), so cos(theta) > 0 always: without a
    random sign every push would travel into the +y half-plane."""
    angles = _random_angles(512)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=4)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high, perpendicular=True,
                          generator=torch.Generator().manual_seed(7))

    dy = (push.stops_xy - push.starts_xy)[:, 1]
    assert (dy > 0).float().mean() > 0.35
    assert (dy < 0).float().mean() > 0.35


def test_fixed_length_pushes_all_travel_that_length():
    angles = _random_angles(512)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=5)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=0.04)

    travelled = (push.stops_xy - push.starts_xy).norm(dim=-1)
    assert not push.truncated.any(), "40 mm fits the 270 mm tray from anywhere"
    assert torch.allclose(travelled, torch.full((512,), 0.04), atol=1e-6)
    assert relative_blade_angle(push.starts_xy, push.stops_xy, angles).max() < 1e-6


def test_fixed_length_pushes_stay_inside_the_box():
    angles = _random_angles(512)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=6)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=0.04)

    for pts in (push.starts_xy, push.stops_xy):
        assert (pts >= low - 1e-6).all()
        assert (pts <= high + 1e-6).all()


def test_a_push_blocked_by_the_wall_flips_instead_of_truncating():
    """The +/- choice is free, so it is spent on reaching the target length."""
    angles = torch.tensor([0.0])                 # normal is +y
    low, high = _box(angles)
    starts = torch.stack([torch.zeros(1), high[:, 1] - 0.005], dim=-1)  # 5 mm from the +y wall
    stops = starts + torch.tensor([[0.0, 0.001]])

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=0.04)

    assert not push.truncated.any()
    assert not push.starts_moved.any(), "flipping should suffice; no nudge needed"
    assert torch.equal(push.starts_xy, starts), "start must be preserved"
    assert (push.stops_xy - starts)[0, 1] < 0, "should have flipped to -y"
    assert torch.allclose((push.stops_xy - starts).norm(dim=-1),
                          torch.tensor([0.04]), atol=1e-6)


def test_an_unreachable_length_is_reported_not_silently_shortened():
    """A truncated push is not in the requested length bin — it must be loud."""
    angles = torch.tensor([0.0])
    low, high = _box(angles)
    starts = torch.zeros(1, 2)
    stops = starts + torch.tensor([[0.0, 0.001]])

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=10.0)

    assert push.truncated.all()


def test_no_restriction_requested_is_a_no_op():
    angles = _random_angles(64)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=8)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high)

    assert torch.equal(push.stops_xy, stops)
    assert torch.equal(push.starts_xy, starts)
    assert not push.truncated.any()
    assert not push.starts_moved.any()


def test_fixed_length_without_perpendicular_keeps_the_drawn_direction():
    angles = _random_angles(64)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=9)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high, length=0.03)

    keep = ~push.truncated
    unit_before = (stops - starts) / (stops - starts).norm(dim=-1, keepdim=True)
    step = push.stops_xy - push.starts_xy
    unit_after = step / step.norm(dim=-1, keepdim=True)
    assert torch.allclose(unit_before[keep], unit_after[keep], atol=1e-5)
    assert torch.allclose(step.norm(dim=-1)[keep],
                          torch.full((int(keep.sum()),), 0.03), atol=1e-6)


def test_restriction_works_on_batched_env_sample_shapes():
    """The sampler applies this to (n_envs, n_samples, ...) tensors."""
    angles = _random_angles(24).reshape(4, 6)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=10)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=0.04)

    assert push.stops_xy.shape == (4, 6, 2)
    assert push.truncated.shape == (4, 6)
    assert relative_blade_angle(push.starts_xy, push.stops_xy, angles).max() < 1e-6


def test_relative_blade_angle_spans_plow_to_shear():
    angles = torch.tensor([0.0, 0.0])
    starts = torch.zeros(2, 2)
    stops = torch.tensor([[0.0, 0.04],     # along the normal -> plow
                          [0.04, 0.0]])    # along the blade axis -> shear
    rel = relative_blade_angle(starts, stops, angles)

    assert abs(float(rel[0])) < 1e-6
    assert abs(float(rel[1]) - torch.pi / 2) < 1e-6


# ---------------------------------------------------------------------------
# Pile-aware action sampling
# ---------------------------------------------------------------------------

from action_sampling import pile_contact_starts  # noqa: E402


def _pile(n=30, extent=0.015, seed=0, batch=8):
    """A compact heap at the origin, as the piled spawn produces."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, n, 2, generator=g) - 0.5) * 2 * extent


def _headings(batch=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, generator=g) * 2 * torch.pi


def _along(pts, headings):
    u = torch.stack([torch.cos(headings), torch.sin(headings)], dim=-1)
    return (pts * u).sum(-1) if pts.dim() == headings.dim() + 1 else \
        (pts * u.unsqueeze(-2)).sum(-1)


def test_start_sits_one_clearance_behind_the_pile():
    """The whole point: no sweep distance is spent reaching the pile."""
    p, h = _pile(), _headings()
    starts, n_in, ok = pile_contact_starts(
        p, h, blade_half_length=0.02, clearance=0.005, min_swath=3,
        generator=torch.Generator().manual_seed(2))

    gap = _along(p, h).min(dim=-1).values - _along(starts, h)
    # The nearest particle IN THE SWATH is exactly `clearance` ahead; the
    # nearest particle overall can only be nearer still, never further.
    assert (gap <= 0.005 + 1e-6).all()
    assert (gap > 0).all(), "the blade must start behind the pile, not inside it"
    assert ok.all()


def test_swath_actually_contains_material():
    p, h = _pile(), _headings()
    _, n_in, ok = pile_contact_starts(
        p, h, blade_half_length=0.02, clearance=0.005, min_swath=5,
        generator=torch.Generator().manual_seed(3))
    assert ok.all()
    assert (n_in >= 5).all()


def test_a_wider_blade_sweeps_more_of_the_pile():
    p, h = _pile(), _headings()
    g = lambda: torch.Generator().manual_seed(4)
    _, narrow, _ = pile_contact_starts(p, h, blade_half_length=0.004,
                                       clearance=0.005, generator=g())
    _, wide, _ = pile_contact_starts(p, h, blade_half_length=0.04,
                                     clearance=0.005, generator=g())
    assert float(wide.float().mean()) > float(narrow.float().mean())


def test_unreachable_min_swath_is_reported_not_hidden():
    """A single isolated particle cannot fill a swath; the caller must know."""
    p = torch.zeros(4, 1, 2)
    _, n_in, ok = pile_contact_starts(p, _headings(batch=4), blade_half_length=0.02,
                                      clearance=0.005, min_swath=5)
    assert not ok.any()
    assert (n_in == 1).all()


def test_start_is_finite_even_when_min_swath_fails():
    p = torch.zeros(4, 1, 2)
    starts, _, ok = pile_contact_starts(p, _headings(batch=4), blade_half_length=0.02,
                                        clearance=0.005, min_swath=5)
    assert torch.isfinite(starts).all(), "must fall back, not return inf"
    assert not ok.any()


def test_push_direction_points_into_the_pile():
    """Every particle in the swath must lie AHEAD of the start, so the sweep
    travels through the pile rather than away from it."""
    p, h = _pile(), _headings()
    starts, _, _ = pile_contact_starts(
        p, h, blade_half_length=0.02, clearance=0.005,
        generator=torch.Generator().manual_seed(5))
    a_p, a_s = _along(p, h), _along(starts, h)
    assert (a_p.min(dim=-1).values > a_s).all()
    # and the pile's far side is further ahead still, i.e. there is depth to sweep
    assert (a_p.max(dim=-1).values > a_p.min(dim=-1).values).all()


def test_clearance_is_respected_exactly():
    p, h = _pile(batch=32), _headings(batch=32)
    for clr in (0.002, 0.005, 0.01):
        starts, n_in, ok = pile_contact_starts(
            p, h, blade_half_length=0.02, clearance=clr, min_swath=3,
            generator=torch.Generator().manual_seed(6))
        # Distance from start to the nearest swath particle == clearance.
        u = torch.stack([torch.cos(h), torch.sin(h)], dim=-1)
        nv = torch.stack([-torch.sin(h), torch.cos(h)], dim=-1)
        lat = (p * nv.unsqueeze(-2)).sum(-1)
        c = (starts * nv).sum(-1)
        in_sw = (lat - c.unsqueeze(-1)).abs() <= 0.02
        a = (p * u.unsqueeze(-2)).sum(-1)
        near = torch.where(in_sw, a, torch.full_like(a, 1e3)).min(dim=-1).values
        assert torch.allclose(near - (starts * u).sum(-1),
                              torch.full_like(near, clr), atol=1e-6)


def test_batched_env_sample_shape():
    p = _pile(batch=4).unsqueeze(1).expand(-1, 6, -1, -1)
    h = torch.rand(4, 6, generator=torch.Generator().manual_seed(7)) * 2 * torch.pi
    starts, n_in, ok = pile_contact_starts(p, h, blade_half_length=0.02,
                                           clearance=0.005)
    assert starts.shape == (4, 6, 2)
    assert n_in.shape == (4, 6) and ok.shape == (4, 6)


def test_pile_contact_start_can_be_far_outside_a_small_box():
    """Documents WHY SandboxManipulation clamps the start.

    `pile_contact_starts` places the blade behind the pile's near face; it knows
    nothing about the workspace box. For a pile whose radius approaches the box
    half-extent, that start is outside the box — measured on real data, 35.8% of
    the time, and a start outside the box has nowhere to travel (3.3% of pushes
    came out at ~0 mm). The clamp lives in the caller, so this test pins the
    property the caller has to defend against rather than asserting it away.
    """
    # A spread pile: radius ~35 mm, like a settled-and-pushed 30-cube heap.
    g = torch.Generator().manual_seed(11)
    ang = torch.rand(256, generator=g) * 2 * torch.pi
    rad = 0.030 + 0.005 * torch.rand(256, generator=g)
    p = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], dim=-1)
    p = p.unsqueeze(0).expand(64, -1, -1)
    h = _headings(batch=64, seed=12)

    starts, _, ok = pile_contact_starts(p, h, blade_half_length=0.02,
                                        clearance=0.005, min_swath=3)
    assert ok.all()
    # The tightest blade box in the 127 mm tray is ~23.5 mm half-extent.
    outside = (starts.abs() > 0.0235).any(dim=-1)
    assert outside.any(), (
        "fixture should reproduce the out-of-box condition the clamp exists for")
