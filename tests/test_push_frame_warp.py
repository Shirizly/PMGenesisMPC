"""Tests for the SE(2) push-frame warp in transforms/functional.py.

The canonical frame of Suh & Tedrake 2020 Fig. 4: origin at the push midpoint,
push direction along +x. Genesis-free.
"""

import math

import pytest
import torch

from transforms.functional import (
    blend_push_prediction, from_push_frame, invert_affine,
    push_frame_transform, push_frame_validity_mask, to_push_frame,
)

N = 64


def _blob(cx=30.0, cy=34.0, sigma=6.0, n=N):
    yy, xx = torch.meshgrid(torch.arange(n).float(), torch.arange(n).float(),
                            indexing='ij')
    return torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).unsqueeze(0)


def _push(sx, sy, ex, ey):
    return torch.tensor([[sx, sy]]), torch.tensor([[ex, ey]])


def test_inverse_undoes_the_transform():
    s, e = _push(10., 32., 30., 32.)
    theta = push_frame_transform(s, e, (N, N))
    inv = invert_affine(theta)

    R = theta[:, :, :2] @ inv[:, :, :2]
    assert torch.allclose(R[0], torch.eye(2), atol=1e-6)
    # Composing the full affine maps must give the identity too.
    t = theta[:, :, :2] @ inv[:, :, 2:] + theta[:, :, 2:]
    assert torch.allclose(t, torch.zeros_like(t), atol=1e-6)


def test_non_square_grid_is_rejected():
    """Normalized coords are only isotropic when H == W; a rotation in
    anisotropic normalized space silently shears."""
    s, e = _push(10., 32., 30., 32.)
    with pytest.raises(ValueError, match="square"):
        push_frame_transform(s, e, (32, 64))


@pytest.mark.parametrize("push", [
    (10., 32., 30., 32.),      # along +x
    (32., 10., 32., 30.),      # along +y
    (20., 20., 40., 40.),      # 45 degrees
    (44., 40., 24., 20.),      # 225 degrees
])
def test_roundtrip_preserves_a_smooth_image(push):
    """Sharp edges lose to bilinear resampling; the transform itself must not."""
    img = _blob()
    s, e = _push(*push)
    mask = push_frame_validity_mask(s, e, (N, N))
    inside = mask >= 0.99

    back = from_push_frame(to_push_frame(img, s, e), s, e, (N, N))
    assert float((back - img).abs()[inside].max()) < 0.02


@pytest.mark.parametrize("push", [
    (10., 32., 30., 32.),
    (20., 20., 40., 40.),
])
def test_roundtrip_preserves_mass(push):
    """Mass conservation through the warp, so any mass drift measured in a
    closed-loop episode is attributable to the physics or the operator, never
    to the resampling."""
    img = _blob()
    s, e = _push(*push)
    back = from_push_frame(to_push_frame(img, s, e), s, e, (N, N))
    assert float(back.sum()) == pytest.approx(float(img.sum()), rel=1e-3)


def test_push_maps_to_the_canonical_axis():
    """The whole point of the frame: whatever the push, in canonical
    coordinates it starts left of center and runs along +x."""
    n = 64
    for sx, sy, ex, ey in [(10., 32., 30., 32.), (32., 10., 32., 30.),
                           (20., 20., 40., 40.), (44., 40., 24., 20.)]:
        img = torch.zeros(1, n, n)
        # Mark the push start with a single bright pixel.
        img[0, int(sy), int(sx)] = 1.0
        s, e = _push(sx, sy, ex, ey)
        canon = to_push_frame(img, s, e)

        idx = int(canon.argmax())
        row, col = idx // n, idx % n
        length = math.hypot(ex - sx, ey - sy)
        # Start sits on the canonical x axis (row = center), half the push
        # length to the LEFT of center.
        assert abs(row - n / 2) <= 1.5, (sx, sy, ex, ey, row)
        assert col < n / 2, (sx, sy, ex, ey, col)
        assert abs((n / 2 - col) - length / 2) <= 1.5, (sx, sy, ex, ey, col)


def test_validity_mask_marks_the_lost_corners():
    """Rotating a square inside a same-sized square must lose area, and the
    mask has to say where."""
    s, e = _push(20., 20., 40., 40.)          # 45 deg: worst case
    mask = push_frame_validity_mask(s, e, (N, N))

    assert 0.5 < float(mask.mean()) < 0.95, "a rotated square loses corners"
    assert float(mask[0, N // 2, N // 2]) > 0.99, "the center is always valid"
    assert float(mask[0, 0, 0]) < 0.5, "a corner is not"


def test_axis_aligned_push_keeps_almost_everything():
    """A pure translation loses only the strip shifted in from outside."""
    s, e = _push(28., 32., 36., 32.)
    mask = push_frame_validity_mask(s, e, (N, N))
    assert float(mask.mean()) > 0.85


def test_blend_keeps_the_original_outside_the_mask():
    pred = torch.full((1, N, N), 0.25)
    occ = torch.full((1, N, N), 0.75)
    mask = torch.zeros(1, N, N)
    mask[0, 10:20, 10:20] = 1.0

    out = blend_push_prediction(pred, occ, mask)
    assert torch.allclose(out[0, 10:20, 10:20], torch.full((10, 10), 0.25))
    assert torch.allclose(out[0, 0, 0], torch.tensor(0.75))


def test_identity_operator_reconstructs_the_input():
    """With A = I the whole pipeline (warp, apply, unwarp, blend) must be a
    no-op up to resampling — the reference check for any fitted operator."""
    img = _blob()
    s, e = _push(20., 20., 40., 40.)
    canon = to_push_frame(img, s, e)                    # A = identity
    back = from_push_frame(canon, s, e, (N, N))
    mask = push_frame_validity_mask(s, e, (N, N))
    out = blend_push_prediction(back, img, mask)

    assert float((out - img).abs().max()) < 0.05


def test_batched_pushes_are_independent():
    img = torch.cat([_blob(20., 20.), _blob(44., 44.)], dim=0)
    s = torch.tensor([[10., 32.], [20., 20.]])
    e = torch.tensor([[30., 32.], [40., 40.]])

    canon = to_push_frame(img, s, e)
    assert canon.shape == (2, N, N)
    for i in range(2):
        one = to_push_frame(img[i:i + 1], s[i:i + 1], e[i:i + 1])
        assert torch.allclose(canon[i], one[0], atol=1e-6)


def test_warp_is_differentiable():
    """Keeps the gradient-descent MPC usable on a warp-based model."""
    img = _blob().requires_grad_(True)
    s, e = _push(20., 20., 40., 40.)
    to_push_frame(img, s, e).sum().backward()
    assert img.grad is not None and float(img.grad.abs().sum()) > 0
