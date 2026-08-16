"""
Genesis/action_sampling.py — action-distribution shaping that has to be aware of
how a batch is simulated, not just of what a single action should look like.

Why this exists
---------------
Environments in a batch step in lockstep, so a batch costs what its *worst*
member costs — twice over:

  step count      ``plate_velocity_translation`` derives ``sweep_steps`` from
                  the LONGEST travel distance in the batch, so every env runs
                  for as long as the longest one, however short its own push.
  per-step cost   each step costs what the densest contact graph in the batch
                  costs.

Sampling each env's action independently therefore makes batching much worse
than it needs to be. Measured at 100 objects (see
tests/scaling_investigation/probe_action_coupling.py), going from 1 to 8 envs:

  identical action in every env    13.47 s -> 15.21 s   (1.13x for 8x the work)
  independently sampled actions    13.34 s -> 40.17 s   (3.01x)

with the gap decomposing into 1.54x from step count and 1.72x from contact
complexity. The step-count half is removable at almost no cost to the data:
share one travel *distance* across the batch while every env keeps its own start
point, direction and blade yaw. Distance is one of five action dimensions and it
still varies fully from batch to batch — only its within-batch variance is given
up, and that variance was buying nothing except a longer sweep for everybody.
"""

from __future__ import annotations

import torch


def equalize_travel_distance(starts_xy: torch.Tensor, stops_xy: torch.Tensor,
                             low: torch.Tensor, high: torch.Tensor,
                             target: torch.Tensor):
    """Rescale each env's push to a shared travel distance, staying in bounds.

    Each env keeps its own start point and direction; only the distance along
    that direction changes. Where the shared distance would take a push outside
    its sampling box, it is truncated at the boundary — those envs travel less,
    which is the safe direction to err since the batch's step count follows the
    longest push, not the shortest.

    Parameters
    ----------
    starts_xy, stops_xy : (..., 2) push endpoints.
    low, high           : (..., 2) per-entry sampling bounds (they depend on the
                          blade yaw, so they are not a single global box).
    target              : (..., 1) desired travel distance, broadcastable.

    Returns
    -------
    (new_stops_xy, clipped_mask) — the mask marks entries that could not reach
    the target distance without leaving their box.
    """
    eps = 1e-12
    delta = stops_xy - starts_xy
    dist = delta.norm(dim=-1, keepdim=True)
    direction = delta / (dist + eps)

    # Ray-box: how far can we travel along `direction` before leaving [low, high]?
    # Per axis the limiting bound is `high` when moving positively along it and
    # `low` when moving negatively; an axis we are not moving along cannot limit
    # us, hence the infinities.
    inf = torch.full_like(direction, float("inf"))
    t_axis = torch.where(
        direction > eps, (high - starts_xy) / torch.where(direction > eps, direction, inf),
        torch.where(
            direction < -eps, (low - starts_xy) / torch.where(direction < -eps, direction, inf),
            inf,
        ),
    )
    t_max = t_axis.min(dim=-1, keepdim=True).values.clamp(min=0.0)

    t = torch.minimum(target, t_max)
    clipped = (target > t_max + 1e-9).squeeze(-1)
    return starts_xy + direction * t, clipped


def shared_batch_distance(dist: torch.Tensor, env_dim: int = 0) -> torch.Tensor:
    """Pick the travel distance a whole batch will share, per sample.

    Takes one environment's own draw rather than a summary statistic. A median
    or mean over environments would concentrate as the batch grows — with many
    envs it would converge to the population median and the *between-batch*
    variation in push length would quietly disappear. One env's draw is an exact
    sample from the same distribution a single-env run would see, so the
    marginal distribution of push lengths across batches is unchanged.
    """
    return dist.select(env_dim, 0).unsqueeze(env_dim)
