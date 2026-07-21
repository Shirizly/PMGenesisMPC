"""
Fast, data-free unit tests for the per-sample loss reduction mode added for
simple_mpc.oracle_mpc (see docs/oracle_mpc_design.md "Cost" and
docs/INTERFACES.md §3.5), and for the new score_map_weighted loss.
"""

import torch

from training.losses import build_loss
from training.types import ModelOutput


def _random_batch(B=6, H=16, W=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits  = torch.randn(B, H, W, generator=g)
    targets = (torch.rand(B, H, W, generator=g) > 0.5).float()
    current = (torch.rand(B, H, W, generator=g) > 0.5).float()
    return logits, targets, current


def test_eulerian_combined_per_sample_mean_matches_scalar_mode():
    logits, targets, current = _random_batch()
    prediction = ModelOutput(logits=logits)
    batch = {"input": current.unsqueeze(1), "target": targets,
             "current_occupancy": current, "target_occupancy": targets}

    cfg = {"mse": 1.0, "dice": 0.5, "bce": 0.3, "sharpness": 0.1,
           "tv": 0.2, "mass": 0.4, "add": 0.1, "remove": 0.1}

    scalar_loss_fn = build_loss({**cfg, "type": "eulerian_combined", "per_sample": False})
    total_scalar, comps_scalar = scalar_loss_fn(prediction, batch)
    assert total_scalar.shape == ()

    per_sample_loss_fn = build_loss({**cfg, "type": "eulerian_combined", "per_sample": True})
    total_ps, comps_ps = per_sample_loss_fn(prediction, batch)
    assert total_ps.shape == (logits.shape[0],)

    # Numerically identical: per-sample mean must equal the scalar-mode total,
    # since H, W are fixed across the batch (see training/losses.py comment).
    assert torch.allclose(total_ps.mean(), total_scalar, atol=1e-5)
    for k in comps_scalar:
        assert abs(comps_scalar[k] - comps_ps[k]) < 1e-5


def test_eulerian_combined_default_is_scalar():
    logits, targets, current = _random_batch()
    prediction = ModelOutput(logits=logits)
    batch = {"input": current.unsqueeze(1), "target": targets}
    loss_fn = build_loss({"type": "eulerian_combined", "mse": 1.0})
    total, _ = loss_fn(prediction, batch)
    assert total.shape == ()


def test_score_map_weighted_per_sample_matches_reward_formula():
    B, H, W = 4, 8, 8
    logits = torch.randn(B, H, W)
    score_map = torch.randn(H, W)

    loss_fn = build_loss({"type": "score_map_weighted", "per_sample": True})
    total, comps = loss_fn(ModelOutput(logits=logits), {"score_map": score_map})
    assert total.shape == (B,)

    probs = torch.sigmoid(logits)
    expected_reward = (probs.clamp(0.0, 1.0) * score_map).reshape(B, -1).sum(dim=-1)
    assert torch.allclose(-total, expected_reward, atol=1e-5)
    assert abs(comps["score_map_reward"] - expected_reward.mean().item()) < 1e-5


def test_score_map_weighted_scalar_mode_is_mean_of_per_sample():
    B, H, W = 5, 8, 8
    logits = torch.randn(B, H, W)
    score_map = torch.randn(H, W)

    ps_loss_fn = build_loss({"type": "score_map_weighted", "per_sample": True})
    total_ps, _ = ps_loss_fn(ModelOutput(logits=logits), {"score_map": score_map})

    scalar_loss_fn = build_loss({"type": "score_map_weighted"})
    total_scalar, _ = scalar_loss_fn(ModelOutput(logits=logits), {"score_map": score_map})

    assert total_scalar.shape == ()
    assert torch.allclose(total_ps.mean(), total_scalar, atol=1e-5)


def test_unknown_loss_type_raises():
    import pytest
    with pytest.raises(KeyError):
        build_loss({"type": "not_a_real_loss"})
