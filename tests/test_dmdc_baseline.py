"""Fast, data-free unit tests for the DMDc falsification baseline math."""

import torch

from dmdc_baseline import (
    ActionBinner,
    apply_operators,
    build_episodes,
    descriptor_slices,
    diagnose_contiguity,
    fit_per_action_operators,
    occupancy_descriptors,
    rollout,
)


def test_descriptor_slices_layout():
    nf = 8
    s = descriptor_slices(nf)
    nfb = nf * (nf // 2 + 1)
    assert s["const"] == slice(0, 1)
    assert (s["dft_real"].stop - s["dft_real"].start) == nfb
    assert (s["dft_imag"].stop - s["dft_imag"].start) == nfb
    # groups are contiguous and cover [0, D)
    assert s["_total"].start == 0
    assert s["_total"].stop == 1 + 1 + 2 + 3 + nfb + nfb


def test_occupancy_descriptors_shape_and_dim():
    B, H, W = 5, 32, 32
    occ = (torch.rand(B, H, W) > 0.7).float()
    for nf in (4, 8):
        phi = occupancy_descriptors(occ, nf)
        assert phi.shape == (B, descriptor_slices(nf)["_total"].stop)
        assert torch.isfinite(phi).all()


def test_action_binner_in_range():
    binner = ActionBinner((-1.0, -1.0), (1.0, 1.0), n_start_bins=4, n_angle_bins=8)
    actions = torch.tensor([
        [-0.9, -0.9, 0.9, 0.9],
        [0.9, 0.9, -0.9, -0.9],
        [0.0, 0.0, 0.0, 1.0],
    ])
    bins = binner(actions)
    assert bins.dtype == torch.long
    assert int(bins.min()) >= 0
    assert int(bins.max()) < binner.n_bins


def test_fit_recovers_linear_map():
    torch.manual_seed(0)
    N, D = 400, 12
    A_true = torch.randn(D, D) * 0.3
    phi_t = torch.randn(N, D)
    phi_t1 = phi_t @ A_true.T
    bins = torch.zeros(N, dtype=torch.long)
    A, counts = fit_per_action_operators(phi_t, phi_t1, bins, n_bins=1, lam=1e-8)
    assert counts[0] == N
    pred = apply_operators(A, phi_t, bins)
    assert (pred - phi_t1).pow(2).mean() < 1e-6


def test_prior_used_for_empty_bins():
    D = 4
    prior = torch.randn(3, D, D)
    phi_t = torch.randn(5, D)
    phi_t1 = torch.randn(5, D)
    bins = torch.zeros(5, dtype=torch.long)  # only bin 0 has data
    A, counts = fit_per_action_operators(
        phi_t, phi_t1, bins, n_bins=3, lam=1e-4, prior_A=prior
    )
    # unseen bins keep the source (prior) operator, not identity
    assert torch.allclose(A[1], prior[1])
    assert torch.allclose(A[2], prior[2])


def test_prior_pulls_estimate_toward_source():
    torch.manual_seed(1)
    N, D = 60, 5
    phi_t = torch.randn(N, D)
    phi_t1 = torch.randn(N, D)  # unrelated -> data alone has no strong preference
    bins = torch.zeros(N, dtype=torch.long)
    prior = torch.randn(1, D, D)
    # strong prior weight -> operator close to the source prior
    A_strong, _ = fit_per_action_operators(
        phi_t, phi_t1, bins, n_bins=1, lam=1e6, prior_A=prior
    )
    # weak prior weight -> operator close to the data least-squares solution
    A_weak, _ = fit_per_action_operators(
        phi_t, phi_t1, bins, n_bins=1, lam=1e-6, prior_A=prior
    )
    assert (A_strong[0] - prior[0]).abs().mean() < (A_weak[0] - prior[0]).abs().mean()


def test_empty_bin_is_identity():
    N, D = 10, 4
    phi_t = torch.randn(N, D)
    phi_t1 = torch.randn(N, D)
    bins = torch.zeros(N, dtype=torch.long)  # only bin 0 used
    A, counts = fit_per_action_operators(phi_t, phi_t1, bins, n_bins=3, lam=1e-4)
    assert counts[1] == 0 and counts[2] == 0
    assert torch.equal(A[1], torch.eye(D))
    assert torch.equal(A[2], torch.eye(D))


def test_diagnose_contiguity_distinguishes_regimes():
    D = 6
    ep = torch.zeros(8, dtype=torch.long)  # one run
    # Contiguous: phi_t1[i] == phi_t[i+1]
    phi_t = torch.randn(8, D)
    phi_t1 = torch.randn(8, D)
    phi_t1[:-1] = phi_t[1:]
    assert diagnose_contiguity(phi_t, phi_t1, ep) < 1e-6
    # Independent: successor no closer than pre-state -> ratio ~ 1
    phi_t = torch.randn(8, D)
    phi_t1 = torch.randn(8, D)
    r = diagnose_contiguity(phi_t, phi_t1, ep)
    assert 0.3 < r < 3.0


def test_build_episodes_grouping_and_rollout():
    D = 5
    phi_t = torch.randn(6, D)
    phi_t1 = torch.randn(6, D)
    bins = torch.arange(6) % 3
    episode_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    eps = build_episodes(phi_t, phi_t1, bins, episode_ids)
    assert len(eps) == 2
    for e in eps:
        assert e["phi"].shape == (4, D)  # T+1 = 3+1
        assert e["bins"].shape == (3,)
    # rollout length matches the bin sequence
    A = torch.eye(D).expand(3, D, D).clone()
    out = rollout(A, eps[0]["phi"][0], eps[0]["bins"])
    assert out.shape == (3, D)
