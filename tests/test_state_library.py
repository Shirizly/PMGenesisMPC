"""Tests for Genesis/state_library.py — the settled-state bank used to reset
environments without paying for a re-settle.

Genesis-free: the module only depends on torch, so these run in the fast suite.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from state_library import (  # noqa: E402
    StateLibrary, apply_symmetry, box_symmetries, _mirror_quat, _quat_mul,
)

SQUARE_BOX = [0.128, 0.128, 0.04]
RECT_BOX = [0.128, 0.200, 0.04]


def _rand_quats(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def _rotmat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _pdist(a):
    """Exact pairwise distances.

    Deliberately not torch.cdist: its ||a||^2 + ||b||^2 - 2ab expansion loses
    ~1e-3 in float32 on spread-out points, which is far larger than the error
    these tests are trying to detect.
    """
    return (a[:, None, :] - a[None, :, :]).norm(dim=-1)


def test_square_box_gets_full_dihedral_group():
    assert len(box_symmetries(SQUARE_BOX)) == 8


def test_rectangular_box_only_gets_half_turns_and_mirrors():
    # a 90 deg rotation would not fit back inside a non-square tray
    assert len(box_symmetries(RECT_BOX)) == 4


def test_mirror_maps_rotations_to_rotations():
    """A reflection is improper, but the conjugation M R M is a rotation."""
    q = _rand_quats(200)
    M = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    R, R_mirrored = _rotmat(q), _rotmat(_mirror_quat(q))
    assert torch.allclose(torch.linalg.det(R_mirrored), torch.ones(200), atol=1e-5)
    assert torch.allclose(R_mirrored, M @ R @ M, atol=1e-5)


def test_mirror_negates_yaw():
    t = 0.7
    q = torch.tensor([[math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]])
    expected = torch.tensor([[math.cos(t / 2), 0.0, 0.0, -math.sin(t / 2)]])
    assert torch.allclose(_mirror_quat(q), expected, atol=1e-6)


def test_quat_mul_identity():
    q = _rand_quats(16, seed=3)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(q)
    assert torch.allclose(_quat_mul(identity, q), q, atol=1e-6)


@pytest.mark.parametrize("yaw,mirror", box_symmetries(SQUARE_BOX))
def test_symmetry_is_rigid_and_z_preserving(yaw, mirror):
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(1, 40, 3, generator=g)
    quat = _rand_quats(40, seed=2).unsqueeze(0)

    moved, rotated = apply_symmetry(pos, quat, yaw, mirror)

    assert torch.allclose(_pdist(pos[0, :, :2]), _pdist(moved[0, :, :2]), atol=1e-5)
    assert torch.allclose(moved[..., 2], pos[..., 2])
    assert torch.allclose(rotated.norm(dim=-1), torch.ones(1, 40), atol=1e-5)


def test_symmetries_produce_distinct_arrangements():
    """The whole point of augmenting is more distinct states, not copies."""
    g = torch.Generator().manual_seed(4)
    pos = torch.randn(1, 40, 3, generator=g)
    quat = _rand_quats(40, seed=5).unsqueeze(0)

    images = {tuple(torch.round(apply_symmetry(pos, quat, y, m)[0].flatten() * 1e4).tolist())
              for y, m in box_symmetries(SQUARE_BOX)}
    assert len(images) == 8


def test_symmetry_keeps_particles_inside_a_square_tray():
    half = SQUARE_BOX[0] / 2
    g = torch.Generator().manual_seed(6)
    pos = (torch.rand(1, 60, 3, generator=g) - 0.5) * 2 * (half - 0.005)
    quat = _rand_quats(60, seed=7).unsqueeze(0)
    for yaw, mirror in box_symmetries(SQUARE_BOX):
        moved, _ = apply_symmetry(pos, quat, yaw, mirror)
        assert moved[..., :2].abs().max() <= half + 1e-6


def test_library_roundtrip(tmp_path):
    lib = StateLibrary(torch.randn(24, 40, 7), {"n_settles": 3})
    path = lib.save(tmp_path)
    back = StateLibrary.load(path)

    assert torch.allclose(back.states, lib.states)
    assert back.meta["n_settles"] == 3
    assert len(back) == 24
    assert back.n_particles == 40


def test_library_rejects_wrong_shape():
    with pytest.raises(ValueError):
        StateLibrary(torch.randn(24, 40))


def test_sample_index_in_range():
    lib = StateLibrary(torch.randn(7, 5, 7))
    assert all(0 <= lib.sample_index() < 7 for _ in range(50))
