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


def test_default_library_path_matches_dataset_layout():
    """The library must be findable where a collection run wrote it."""
    from state_library import default_library_path, STATE_LIBRARY_FILENAME

    p = default_library_path("data/dry_run", "cube", 200, 0.005)
    assert p.as_posix() == f"data/dry_run/cube/n200/size0.005/{STATE_LIBRARY_FILENAME}"


class _FakeSim:
    """Minimal stand-in for SandboxManipulation's per-env state interface."""

    def __init__(self, n_envs, n_particles):
        self._n_envs = n_envs
        self.material = [object()] * n_particles
        self._particle_state = torch.zeros(n_envs, n_particles, 7)
        self.written = None

    def set_particle_state(self, pos, quat):
        self.written = (pos, quat)


def test_apply_per_env_gives_each_env_its_own_state():
    lib = StateLibrary(torch.randn(16, 12, 7))
    sim = _FakeSim(n_envs=4, n_particles=12)

    idx = lib.apply_per_env(sim, indices=[0, 5, 5, 9])

    assert idx == [0, 5, 5, 9]
    pos, quat = sim.written
    assert pos.shape == (4, 12, 3) and quat.shape == (4, 12, 4)
    # env 0 and env 1 drew different library entries -> different poses
    assert not torch.allclose(pos[0], pos[1])
    # envs 1 and 2 drew the same entry -> identical poses
    assert torch.allclose(pos[1], pos[2])
    assert torch.allclose(pos[3], lib.states[9, :, 0:3])


def test_apply_per_env_rejects_wrong_index_count():
    lib = StateLibrary(torch.randn(16, 12, 7))
    with pytest.raises(ValueError):
        lib.apply_per_env(_FakeSim(4, 12), indices=[0, 1])


def test_apply_per_env_rejects_particle_count_mismatch():
    lib = StateLibrary(torch.randn(16, 12, 7))
    with pytest.raises(ValueError):
        lib.apply_per_env(_FakeSim(2, 40), indices=[0, 1])


def test_next_index_draws_without_replacement():
    """Every state is used once before any is used twice."""
    lib = StateLibrary(torch.randn(8, 5, 7))
    drawn = [lib.next_index() for _ in range(8)]
    assert sorted(drawn) == list(range(8))
    assert lib.refills == 0


def test_next_index_refills_after_exhaustion():
    lib = StateLibrary(torch.randn(4, 5, 7))
    first = [lib.next_index() for _ in range(4)]
    assert lib.draws_remaining == 0

    second = [lib.next_index() for _ in range(4)]
    assert sorted(first) == sorted(second) == list(range(4))
    assert lib.refills == 1, "wrapping must be reported, it signals a short library"


def test_draws_remaining_counts_down():
    lib = StateLibrary(torch.randn(5, 5, 7))
    assert lib.draws_remaining == 5
    lib.next_index()
    assert lib.draws_remaining == 4


def test_apply_broadcasts_one_state_to_every_env():
    """Collection deliberately shares one initial state across envs."""
    lib = StateLibrary(torch.randn(6, 9, 7))
    sim = _FakeSim(n_envs=4, n_particles=9)
    sim._particle_state = torch.zeros(4, 9, 7)

    idx = lib.apply(sim, index=2)

    assert idx == 2
    pos, quat = sim.written
    # a single state, left for set_particle_state to broadcast
    assert pos.shape[0] == 1 and quat.shape[0] == 1
    assert torch.allclose(pos[0], lib.states[2, :, 0:3])


def test_apply_without_index_consumes_the_permutation():
    lib = StateLibrary(torch.randn(3, 9, 7))
    sim = _FakeSim(n_envs=2, n_particles=9)
    used = {lib.apply(sim) for _ in range(3)}
    assert used == {0, 1, 2}, "successive batches must get distinct states"
