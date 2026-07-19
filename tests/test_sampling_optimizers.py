"""
Fast, Genesis-free unit tests for simple_mpc.sampling_optimizers (CEM / MPPI).

Optimizers are exercised against a known quadratic cost over the action box
so convergence can be checked without any simulator.
"""

import numpy as np
import torch

from simple_mpc.sampling_optimizers import (
    CEMOptimizer,
    MPPIOptimizer,
    make_sampling_optimizer,
)

ACT_LO = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
ACT_HI = np.array([ 1.0,  1.0,  1.0,  1.0], dtype=np.float32)
TARGET = torch.tensor([0.3, -0.2, 0.5, -0.4])


def _quadratic_cost(candidates: torch.Tensor) -> torch.Tensor:
    """(n, n_ahead, 4) -> (n,) squared distance to TARGET, summed over horizon."""
    diff = candidates - TARGET.view(1, 1, 4)
    return (diff ** 2).sum(dim=(1, 2))


def _run_optimizer(opt, n_iters=25, n_ask=64):
    init_mean = torch.zeros(opt.n_ahead, 4)
    opt.reset(init_mean)
    for _ in range(n_iters):
        cand = opt.ask(n_ask)
        cost = _quadratic_cost(cand)
        opt.tell(cand, cost)
    return opt


def test_cem_converges_toward_target():
    opt = CEMOptimizer(n_ahead=1, act_lo=ACT_LO, act_hi=ACT_HI, device='cpu',
                        n_elite=8, momentum=0.3, std_floor=0.001)
    _run_optimizer(opt)
    best = opt.best()
    assert best.shape == (1, 4)
    assert torch.allclose(best[0], TARGET, atol=0.15)


def test_cem_std_never_below_floor():
    opt = CEMOptimizer(n_ahead=2, act_lo=ACT_LO, act_hi=ACT_HI, device='cpu',
                        n_elite=4, momentum=0.0, std_floor=0.02)
    opt.reset(torch.zeros(2, 4))
    for _ in range(15):
        cand = opt.ask(32)
        cost = _quadratic_cost(cand)
        opt.tell(cand, cost)
        assert (opt.std >= 0.02 - 1e-6).all()


def test_cem_candidates_respect_bounds():
    opt = CEMOptimizer(n_ahead=1, act_lo=ACT_LO, act_hi=ACT_HI, device='cpu')
    opt.reset(torch.zeros(1, 4))
    cand = opt.ask(256)
    assert (cand >= torch.as_tensor(ACT_LO) - 1e-6).all()
    assert (cand <= torch.as_tensor(ACT_HI) + 1e-6).all()


def test_mppi_converges_toward_target():
    opt = MPPIOptimizer(n_ahead=1, act_lo=ACT_LO, act_hi=ACT_HI, device='cpu',
                         lambda_=0.05, sigma=0.3, beta_filter=0.0)
    _run_optimizer(opt, n_iters=40, n_ask=128)
    best = opt.best()
    assert torch.allclose(best[0], TARGET, atol=0.2)


def test_mppi_weights_sum_to_one():
    opt = MPPIOptimizer(n_ahead=1, act_lo=ACT_LO, act_hi=ACT_HI, device='cpu', lambda_=0.1)
    costs = torch.tensor([1.0, 2.0, 0.5, 3.0])
    c = costs - costs.min()
    weights = torch.softmax(-c / opt.lambda_, dim=0)
    assert abs(weights.sum().item() - 1.0) < 1e-6


def test_mppi_beta_filter_blends_across_reset_calls():
    opt = MPPIOptimizer(n_ahead=1, act_lo=ACT_LO, act_hi=ACT_HI, device='cpu', beta_filter=0.5)
    opt.reset(torch.full((1, 4), 1.0))
    cand = opt.ask(16)
    opt.tell(cand, _quadratic_cost(cand))
    mean_after_first_tell = opt.mean.clone()

    opt.reset(torch.zeros(1, 4))
    expected = 0.5 * mean_after_first_tell + 0.5 * torch.zeros(1, 4)
    assert torch.allclose(opt.mean, expected, atol=1e-6)


def test_warm_start_mean_shifts_and_stays_in_bounds():
    opt = CEMOptimizer(n_ahead=3, act_lo=ACT_LO, act_hi=ACT_HI, device='cpu')
    seq = torch.tensor([[0.1, 0.1, 0.1, 0.1],
                        [0.2, 0.2, 0.2, 0.2],
                        [0.3, 0.3, 0.3, 0.3]])
    opt.reset(seq)
    shifted = opt.warm_start_mean()
    assert shifted.shape == (3, 4)
    # first two rows are the old rows [1] and [2] (dropped row [0], shifted forward)
    assert torch.allclose(shifted[0], seq[1])
    assert torch.allclose(shifted[1], seq[2])
    assert (shifted[2] >= torch.as_tensor(ACT_LO)).all()
    assert (shifted[2] <= torch.as_tensor(ACT_HI)).all()


def test_make_sampling_optimizer_factory():
    cem = make_sampling_optimizer('cem', 1, ACT_LO, ACT_HI, {}, device='cpu')
    assert isinstance(cem, CEMOptimizer)
    mppi = make_sampling_optimizer('mppi', 1, ACT_LO, ACT_HI, {}, device='cpu')
    assert isinstance(mppi, MPPIOptimizer)

    import pytest
    with pytest.raises(ValueError):
        make_sampling_optimizer('not_an_optimizer', 1, ACT_LO, ACT_HI, {}, device='cpu')


def test_best_raises_before_any_tell():
    import pytest
    opt = CEMOptimizer(n_ahead=1, act_lo=ACT_LO, act_hi=ACT_HI, device='cpu')
    opt.reset(torch.zeros(1, 4))
    with pytest.raises(RuntimeError):
        opt.best()
