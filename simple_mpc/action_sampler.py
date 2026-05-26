"""
Action sampling strategies for MPC candidate generation.

Different sampling strategies (random, learned priors, importance sampling, etc.)
can be implemented as subclasses of ActionSampler and plugged into run_simple_mpc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import torch


class ActionSampler(ABC):
    """
    Base class for MPC action sampling strategies.

    Each MPC step samples n_sample candidate action sequences of length n_ahead.
    Different samplers can implement different strategies (random, learned,
    importance-weighted, etc.).
    """

    @abstractmethod
    def sample(
        self,
        n_sample: int,
        n_ahead: int,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate a batch of action sequences.

        Parameters
        ----------
        n_sample : int
            Number of independent candidates to sample
        n_ahead : int
            Planning horizon (rollout length)
        act_lo : np.ndarray (4,)
            Lower bounds for actions [sx, sy, ex, ey]
        act_hi : np.ndarray (4,)
            Upper bounds for actions [sx, sy, ex, ey]
        device : str
            PyTorch device ('cuda' or 'cpu')

        Returns
        -------
        act_seqs : torch.Tensor (n_sample, n_ahead, 4)
            Sampled action sequences, requires_grad=True for optimization
        """
        pass


class RandomUniformSampler(ActionSampler):
    """
    Sample actions uniformly at random from the workspace bounds.

    This is the default / baseline strategy: each action component is drawn
    independently from the specified bounds. Fast and simple.
    """

    def sample(
        self,
        n_sample: int,
        n_ahead: int,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Generate uniformly random action sequences.

        Each of the n_sample candidates independently samples each action
        component uniformly from [act_lo, act_hi].
        """
        act_np = np.random.uniform(
            act_lo, act_hi, (n_sample, n_ahead, 4)
        ).astype(np.float32)
        act_seqs = torch.tensor(act_np, device=device, requires_grad=True)
        return act_seqs


class PhysicsAwareActionSampler(ActionSampler):
    """
    Sample actions so the pusher plate fits entirely inside the workspace box
    at both the start and stop positions, for *any* travel direction.

    In ``GenesisEnv.step`` the plate yaw is derived as::

        yaw = atan2(ey - sy, ex - sx) + π/2

    so the plate is always perpendicular to its travel direction.  For a plate
    of size ``[L, W, H]`` the axis-aligned bounding box (AABB) at yaw α is::

        AABB_x = |cos α| * L/2 + |sin α| * W/2  ≤  L/2
        AABB_y = |sin α| * L/2 + |cos α| * W/2  ≤  L/2

    The maximum over all orientations is ``L/2`` in either axis.  Using the
    conservative half-range::

        v = workspace_half  −  L/2  −  safety_margin

    for all four action coordinates guarantees the plate stays inside the box
    regardless of which direction the optimiser pushes the action during GD.
    """

    def __init__(
        self,
        plate_length: float  = 0.04,
        plate_width:  float  = 0.002,
        safety_margin: float = 0.01,
    ):
        self.plate_length  = plate_length
        self.plate_width   = plate_width
        self.safety_margin = safety_margin

    def sample(
        self,
        n_sample: int,
        n_ahead: int,
        act_lo: np.ndarray,
        act_hi: np.ndarray,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Sample start and stop positions within the plate-aware valid region.

        ``act_lo`` / ``act_hi`` are the raw workspace bounds; the plate margin
        is subtracted internally so the caller does not need to adjust them.
        """
        # Workspace centre and half-widths (from start-coord bounds dims 0, 1)
        half_x = (act_hi[0] - act_lo[0]) / 2.0
        half_y = (act_hi[1] - act_lo[1]) / 2.0
        cx     = (act_hi[0] + act_lo[0]) / 2.0
        cy     = (act_hi[1] + act_lo[1]) / 2.0

        # Conservative valid half-range: plate fits for ANY travel direction
        vx = max(half_x - self.plate_length / 2.0 - self.safety_margin, 0.0)
        vy = max(half_y - self.plate_length / 2.0 - self.safety_margin, 0.0)

        # Sample (sx, sy, ex, ey) independently within [-v, v] + centre
        rng = np.random.uniform(size=(n_sample, n_ahead, 4)).astype(np.float32)
        sx = cx + rng[:, :, 0] * (2.0 * vx) - vx
        sy = cy + rng[:, :, 1] * (2.0 * vy) - vy
        ex = cx + rng[:, :, 2] * (2.0 * vx) - vx
        ey = cy + rng[:, :, 3] * (2.0 * vy) - vy

        acts = np.stack([sx, sy, ex, ey], axis=-1)   # (n_sample, n_ahead, 4)
        return torch.tensor(acts, device=device, requires_grad=True)


# Factory function for easy selection
def make_action_sampler(sampler_type: str = 'physics_aware', **kwargs) -> ActionSampler:
    """
    Create an action sampler by name.

    Parameters
    ----------
    sampler_type : str
        Name of the sampler: ``'physics_aware'`` (default) or ``'uniform'``.
    **kwargs
        Extra keyword arguments forwarded to the sampler constructor.
        ``PhysicsAwareActionSampler`` accepts ``plate_length``,
        ``plate_width``, and ``safety_margin``.

    Returns
    -------
    sampler : ActionSampler
        Instantiated sampler ready for use

    Examples
    --------
    >>> sampler = make_action_sampler('physics_aware', plate_length=0.04, safety_margin=0.01)
    >>> acts = sampler.sample(n_sample=512, n_ahead=1, ...)
    """
    if sampler_type == 'uniform':
        return RandomUniformSampler()
    elif sampler_type == 'physics_aware':
        valid = {'plate_length', 'plate_width', 'safety_margin'}
        return PhysicsAwareActionSampler(
            **{k: v for k, v in kwargs.items() if k in valid}
        )
    else:
        raise ValueError(
            f"Unknown sampler '{sampler_type}'. "
            f"Available: {sorted(['uniform', 'physics_aware'])}"
        )
