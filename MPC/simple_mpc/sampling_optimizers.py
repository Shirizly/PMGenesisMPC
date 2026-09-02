"""
Sampling-based (gradient-free) trajectory optimizers for ``oracle_mpc``.

The Genesis simulator is not differentiable end-to-end through
``execute_action``, so ``simple_mpc.mpc.run_simple_mpc``'s Adam/backprop loop
is not usable there. CEM and MPPI both replace it with the same skeleton:

    reset(init_mean)                      # seed the sampling distribution
    for it in range(n_opt_iter):
        candidates = ask(n)               # sample action sequences
        costs      = <external rollout + loss evaluation>
        tell(candidates, costs)           # update the distribution
    best_seq = best()

They differ only in the ``tell`` update rule (~15 lines each), which is why
both are implemented here rather than picking one (see
docs/oracle_mpc_design.md "Sampling optimizers").
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class SamplingOptimizer(ABC):
    """Base class for sampling-based MPC trajectory optimizers.

    Candidates are ``(n_ahead, 4)`` action sequences (``[sx, sy, ex, ey]``
    per step), sampled from and clipped to ``[act_lo, act_hi]`` — the same
    physics-aware workspace bounds ``simple_mpc.mpc.run_simple_mpc`` uses for
    gradient clipping.
    """

    def __init__(self, n_ahead: int, act_lo: np.ndarray, act_hi: np.ndarray,
                 device: str = 'cuda'):
        self.n_ahead = n_ahead
        self.device  = device
        self.act_lo  = torch.as_tensor(act_lo, dtype=torch.float32, device=device)
        self.act_hi  = torch.as_tensor(act_hi, dtype=torch.float32, device=device)
        self.mean: torch.Tensor | None = None    # (n_ahead, 4)
        self._best_seq:  torch.Tensor | None = None
        self._best_cost: float = float('inf')

    @abstractmethod
    def reset(self, init_mean: torch.Tensor) -> None:
        """Seed the sampling distribution for a fresh MPC step.

        Parameters
        ----------
        init_mean : (n_ahead, 4) tensor — typically the warm-started mean
            from the previous MPC step (see ``warm_start_mean``), or a fresh
            random sample on the very first step.
        """

    @abstractmethod
    def ask(self, n: int) -> torch.Tensor:
        """Sample ``n`` candidate action sequences, clipped to bounds.

        Returns (n, n_ahead, 4).
        """

    @abstractmethod
    def tell(self, candidates: torch.Tensor, costs: torch.Tensor) -> None:
        """Update the sampling distribution from evaluated candidates.

        Parameters
        ----------
        candidates : (n, n_ahead, 4)
        costs      : (n,) — LOWER is better (this is a cost, not a reward).
        """

    def best(self) -> torch.Tensor:
        """Return the lowest-cost action sequence seen since the last ``reset()``."""
        if self._best_seq is None:
            raise RuntimeError("best() called before any tell()")
        return self._best_seq

    def warm_start_mean(self) -> torch.Tensor:
        """Time-shift the mean by one step for the next MPC step's ``reset()``.

        Standard receding-horizon warm start: drop the first action and
        shift the rest forward. The freed-up tail slot is re-randomized
        within bounds (rather than repeating the last action), so warm-start
        doesn't bias the optimizer toward a fixed action near the previous
        step's choice. For ``n_ahead == 1`` there is nothing to shift, so
        this correctly reduces to a fully fresh random restart every step.
        """
        shifted = self.mean.roll(-1, dims=0).clone()
        shifted[-1] = self.act_lo + torch.rand(4, device=self.device) * (self.act_hi - self.act_lo)
        return shifted

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(x, self.act_hi), self.act_lo)

    def _track_best(self, candidates: torch.Tensor, costs: torch.Tensor) -> None:
        i = int(torch.argmin(costs).item())
        if float(costs[i].item()) < self._best_cost:
            self._best_cost = float(costs[i].item())
            self._best_seq  = candidates[i].detach().clone()


class CEMOptimizer(SamplingOptimizer):
    """Cross-Entropy Method.

    Each iteration: sample from a diagonal Gaussian, keep the ``n_elite``
    lowest-cost candidates, refit (mean, std) from the elites with momentum
    smoothing, floor the std to avoid premature collapse.
    """

    def __init__(self, n_ahead: int, act_lo: np.ndarray, act_hi: np.ndarray,
                 device: str = 'cuda', n_elite: int = 8, momentum: float = 0.5,
                 std_floor: float = 0.005, init_std_frac: float = 0.5):
        super().__init__(n_ahead, act_lo, act_hi, device)
        self.n_elite    = n_elite
        self.momentum   = momentum
        self.std_floor  = std_floor
        self._init_std  = init_std_frac * (self.act_hi - self.act_lo)   # (4,)
        self.std: torch.Tensor | None = None

    def reset(self, init_mean: torch.Tensor) -> None:
        self.mean = init_mean.to(self.device).clone()
        self.std  = self._init_std.unsqueeze(0).expand(self.n_ahead, -1).clone()
        self._best_seq  = None
        self._best_cost = float('inf')

    def ask(self, n: int) -> torch.Tensor:
        noise = torch.randn(n, self.n_ahead, 4, device=self.device)
        cand  = self.mean.unsqueeze(0) + noise * self.std.unsqueeze(0)
        return self._clip(cand)

    def tell(self, candidates: torch.Tensor, costs: torch.Tensor) -> None:
        self._track_best(candidates, costs)
        k = min(self.n_elite, candidates.shape[0])
        elite_idx = torch.topk(costs, k, largest=False).indices
        elite = candidates[elite_idx]   # (k, n_ahead, 4)

        new_mean = elite.mean(dim=0)
        new_std  = elite.std(dim=0, unbiased=False).clamp_min(self.std_floor)

        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * new_mean
        self.std  = self.momentum * self.std  + (1.0 - self.momentum) * new_std


class MPPIOptimizer(SamplingOptimizer):
    """Model Predictive Path Integral control.

    Each iteration: sample from a fixed-sigma Gaussian around the mean,
    reweight candidates by ``softmax(-cost / lambda)``, set the mean to the
    weighted average. ``beta_filter`` low-pass-filters the mean *across MPC
    steps* (applied in ``reset()``, blending the previous step's converged
    mean with the freshly warm-started one) to damp step-to-step jitter.
    """

    def __init__(self, n_ahead: int, act_lo: np.ndarray, act_hi: np.ndarray,
                 device: str = 'cuda', lambda_: float = 0.1, sigma: float = 0.02,
                 beta_filter: float = 0.7):
        super().__init__(n_ahead, act_lo, act_hi, device)
        self.lambda_     = lambda_
        self.sigma       = sigma
        self.beta_filter = beta_filter
        self._prev_mean: torch.Tensor | None = None

    def reset(self, init_mean: torch.Tensor) -> None:
        init_mean = init_mean.to(self.device)
        if self._prev_mean is not None and self.beta_filter > 0.0:
            self.mean = (self.beta_filter * self._prev_mean
                         + (1.0 - self.beta_filter) * init_mean)
        else:
            self.mean = init_mean.clone()
        self._best_seq  = None
        self._best_cost = float('inf')

    def ask(self, n: int) -> torch.Tensor:
        noise = torch.randn(n, self.n_ahead, 4, device=self.device) * self.sigma
        cand  = self.mean.unsqueeze(0) + noise
        return self._clip(cand)

    def tell(self, candidates: torch.Tensor, costs: torch.Tensor) -> None:
        self._track_best(candidates, costs)
        # Numerically stable softmax(-cost / lambda): subtract the min cost.
        c = costs - costs.min()
        weights = torch.softmax(-c / self.lambda_, dim=0)   # (n,)
        self.mean = (weights.view(-1, 1, 1) * candidates).sum(dim=0)
        self._prev_mean = self.mean.detach().clone()


def make_sampling_optimizer(
    name: str,
    n_ahead: int,
    act_lo: np.ndarray,
    act_hi: np.ndarray,
    cfg: dict,
    device: str = 'cuda',
) -> SamplingOptimizer:
    """Factory: build a ``SamplingOptimizer`` by name ('cem' | 'mppi').

    ``cfg`` is the ``mpc`` config sub-dict; optimizer-specific weights are
    read from ``cfg['cem']`` / ``cfg['mppi']``.
    """
    key = name.lower()
    if key == 'cem':
        c = cfg.get('cem', {})
        return CEMOptimizer(
            n_ahead, act_lo, act_hi, device=device,
            n_elite=int(c.get('n_elite', 8)),
            momentum=float(c.get('momentum', 0.5)),
            std_floor=float(c.get('std_floor', 0.005)),
            init_std_frac=float(c.get('init_std_frac', 0.5)),
        )
    if key == 'mppi':
        c = cfg.get('mppi', {})
        return MPPIOptimizer(
            n_ahead, act_lo, act_hi, device=device,
            lambda_=float(c.get('lambda', 0.1)),
            sigma=float(c.get('sigma', 0.02)),
            beta_filter=float(c.get('beta_filter', 0.7)),
        )
    raise ValueError(f"Unknown optimizer {name!r}. Expected 'cem' or 'mppi'.")
