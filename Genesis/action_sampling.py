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

Action-space *restriction* also lives here
--------------------------------------------
The second group of helpers restricts which actions are drawn at all, rather
than reshaping their cost. Both restrictions exist for the switched-linear
visual-foresight baseline (docs/linear_visual_foresight_baseline.md §7), which
needs one operator per push length in a frame where the blade face is normal to
the push:

  perpendicular    the push travels along the blade's face normal, so the
                   5-DOF action (start, direction, yaw) collapses to the 4-DOF
                   action that planar-pushing work assumes (Mason 1986).
  fixed length     every push travels the same distance, so a whole dataset
                   supports a single transition operator.

They are plain torch geometry with no Genesis dependency, so they are unit
testable without a GPU (tests/test_action_sampling.py).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

_EPS = 1e-12


def ray_box_max_travel(starts_xy: torch.Tensor, direction: torch.Tensor,
                       low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """How far can we travel along `direction` before leaving [low, high]?

    Per axis the limiting bound is `high` when moving positively along it and
    `low` when moving negatively; an axis we are not moving along cannot limit
    us, hence the infinities.

    Parameters
    ----------
    starts_xy : (..., 2) ray origins.
    direction : (..., 2) unit (or at least non-zero) directions.
    low, high : (..., 2) per-entry box bounds.

    Returns
    -------
    (..., 1) non-negative maximum travel distance.
    """
    inf = torch.full_like(direction, float("inf"))
    t_axis = torch.where(
        direction > _EPS, (high - starts_xy) / torch.where(direction > _EPS, direction, inf),
        torch.where(
            direction < -_EPS, (low - starts_xy) / torch.where(direction < -_EPS, direction, inf),
            inf,
        ),
    )
    return t_axis.min(dim=-1, keepdim=True).values.clamp(min=0.0)


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
    delta = stops_xy - starts_xy
    dist = delta.norm(dim=-1, keepdim=True)
    direction = delta / (dist + _EPS)

    t_max = ray_box_max_travel(starts_xy, direction, low, high)

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


# ---------------------------------------------------------------------------
# Action-space restriction: perpendicular pushes and fixed push length
# ---------------------------------------------------------------------------


def blade_normal(angles: torch.Tensor) -> torch.Tensor:
    """Unit normal of the blade face, for blade yaw `angles`.

    The blade's long axis is ``(cos θ, sin θ)`` — read off how
    ``SandboxManipulation.generate_action_samples`` shrinks its sampling box,
    which subtracts ``cos(θ)·tool_length/2`` from the x half-extent — so the
    face normal, the only direction a *perpendicular* push can travel, is
    ``(−sin θ, cos θ)``.

    Returns (..., 2). Note the sign is arbitrary: a blade at yaw θ can push
    along either ``+n̂`` or ``−n̂``, and callers must choose (see
    `constrain_push`, which randomizes it). Because yaw is drawn from
    ``(−π/2, π/2)``, ``cos θ > 0`` always, so always taking ``+n̂`` would send
    every push into the ``+y`` half-plane — a badly skewed dataset that no
    aggregate statistic would reveal.
    """
    return torch.stack([-torch.sin(angles), torch.cos(angles)], dim=-1)


def sampling_box(angles: torch.Tensor, granular_vol, tool_length: float,
                 tool_width: float, safety_margin: float):
    """Per-yaw bounds for the blade centre, as (low, high) each (..., 2).

    The box shrinks with yaw because the blade's axis-aligned footprint does:
    at θ=0 its length eats into x and its width into y, and they swap at
    θ=±π/2. Extracted so the bounds can be *recomputed* after
    placement-aware sampling replaces the yaw — the box that was correct for
    the drawn yaw is stale for the replacement.
    """
    cos, sin = torch.cos(angles), torch.sin(angles).abs()
    half_x = granular_vol[0] / 2 - (cos * tool_length / 2
                                    + sin * tool_width / 2 + safety_margin)
    half_y = granular_vol[1] / 2 - (sin * tool_length / 2
                                    + cos * tool_width / 2 + safety_margin)
    high = torch.stack([half_x, half_y], dim=-1)
    return -high, high


class ConstrainedPush(NamedTuple):
    """Result of `constrain_push`.

    starts_xy, stops_xy : the push, after restriction.
    starts_moved        : starts nudged to make a fixed-length push fit (see
                          `constrain_push`); the nudge is at most `length`.
    truncated           : pushes that could NOT be made to travel the requested
                          length. These are not in the requested length bin and
                          must not be fitted as though they were.
    """
    starts_xy: torch.Tensor
    stops_xy: torch.Tensor
    starts_moved: torch.Tensor
    truncated: torch.Tensor


def constrain_push(starts_xy: torch.Tensor, stops_xy: torch.Tensor,
                   angles: torch.Tensor, low: torch.Tensor, high: torch.Tensor,
                   *, perpendicular: bool = False, length: float | None = None,
                   generator: torch.Generator | None = None) -> ConstrainedPush:
    """Restrict pushes to the blade normal and/or to a fixed travel distance.

    Blade yaw is never changed, so this composes with placement-aware start
    sampling (which chooses start *and* yaw) as long as it runs *after* it —
    running before would let the yaw be replaced underneath and silently break
    perpendicularity.

    Parameters
    ----------
    starts_xy, stops_xy : (..., 2) push endpoints as drawn.
    angles              : (...,)   blade yaw, radians.
    low, high           : (..., 2) sampling bounds *for these angles*.
    perpendicular       : send the push along ±`blade_normal(angles)` instead
                          of along the drawn start→stop direction.
    length              : if given, every push travels exactly this distance
                          (metres) instead of the drawn one, so the whole
                          dataset supports one transition operator.
    generator           : optional torch.Generator for the ± sign draw.

    Returns
    -------
    A `ConstrainedPush`. See the notes below on how the tray boundary is
    handled, which differs between the two restrictions on purpose.
    """
    no_flag = torch.zeros(starts_xy.shape[:-1], dtype=torch.bool,
                          device=starts_xy.device)
    if not perpendicular and length is None:
        return ConstrainedPush(starts_xy, stops_xy, no_flag, no_flag)

    delta = stops_xy - starts_xy
    drawn_dist = delta.norm(dim=-1, keepdim=True)
    target = (torch.full_like(drawn_dist, float(length))
              if length is not None else drawn_dist)

    if perpendicular:
        n_hat = blade_normal(angles)
        sign = torch.randint(0, 2, starts_xy.shape[:-1], generator=generator,
                             device=starts_xy.device,
                             dtype=starts_xy.dtype).mul_(2).sub_(1)
        direction = sign.unsqueeze(-1) * n_hat

        # A push blocked by the tray wall on the drawn side is retried on the
        # other: the ± choice is free, so spend it on reaching the target
        # length rather than fighting the boundary.
        reach = ray_box_max_travel(starts_xy, direction, low, high)
        flipped = ray_box_max_travel(starts_xy, -direction, low, high)
        direction = torch.where((reach < target) & (flipped > reach),
                                -direction, direction)
    else:
        direction = delta / (drawn_dist + _EPS)

    # Where the requested travel does not fit inside the tray from the start it
    # was drawn at, move the START rather than shortening the push. Shortening
    # would be the obvious fix and it is the wrong one twice over: a shortened
    # fixed-length push silently leaves the requested bin and contaminates a
    # single-operator fit, and a shortened free-length push distorts the very
    # length distribution this restriction is supposed to leave alone (measured
    # on 20k draws: shortening truncates 22.5% and drags the mean push from
    # 105 mm to 94 mm). Nudging the start keeps yaw, direction and length all
    # exact and costs a start displacement of at most the push length.
    #
    # Feasible starts satisfy both `low <= start <= high` and
    # `low <= start + d <= high`, i.e. `low - d <= start <= high - d`.
    # Intersecting: [low + relu(-d), high - relu(d)].
    #
    # Note this is not a rare corner case even in a tray many times the push
    # length: along an OBLIQUE normal the box extent through a start near a
    # corner can be well under 2*length, because the binding constraint is
    # whichever axis the diagonal reaches first. Measured in the 270 mm tray
    # over 20k draws — fixed 40 mm: 2.5% nudged, 0% truncated; fixed 100 mm:
    # 15.2% nudged, 0% truncated; free length: 22.2% nudged, 0.4% truncated
    # with the mean push length preserved to 4 decimal places.
    d = direction * target
    lo_feasible = low + (-d).clamp(min=0.0)
    hi_feasible = high - d.clamp(min=0.0)

    # Genuinely infeasible only when the travel exceeds the box extent along
    # `direction` — i.e. no start point in the tray admits this push at all.
    # Unreachable for a sane fixed length; common for a free length drawn as
    # the distance between two points in the box, which can exceed the box's
    # extent along any single axis.
    infeasible = (lo_feasible > hi_feasible).any(dim=-1)
    hi_feasible = torch.maximum(hi_feasible, lo_feasible)

    new_starts = starts_xy.clamp(min=lo_feasible, max=hi_feasible)
    moved = ((new_starts - starts_xy).abs().amax(dim=-1) > 1e-9) & ~infeasible

    if infeasible.any():
        # Nothing else to do for these but travel as far as the tray allows.
        t_max = ray_box_max_travel(new_starts, direction, low, high)
        d = torch.where(infeasible.unsqueeze(-1), direction * t_max, d)

    return ConstrainedPush(new_starts, new_starts + d, moved, infeasible)


def relative_blade_angle(starts_xy: torch.Tensor, stops_xy: torch.Tensor,
                         angles: torch.Tensor) -> torch.Tensor:
    """Angle between the push direction and the blade face normal, in [0, π/2].

    Zero means a perpendicular push (the blade plows), π/2 means the push runs
    along the blade's own axis (it shears). The diagnostic that says whether
    restricting to perpendicular pushes gives anything up — see
    docs/linear_visual_foresight_baseline.md §7.5.

    Computed as atan2(|cross|, |dot|) rather than acos(|dot|): acos is
    ill-conditioned exactly where this is used most, near zero, where float32
    rounding in the dot product turns an exactly-perpendicular push into a
    5e-4 rad "error". atan2 is stable across the whole range.
    """
    delta = stops_xy - starts_xy
    direction = delta / (delta.norm(dim=-1, keepdim=True) + _EPS)
    n_hat = blade_normal(angles)
    dot = (direction * n_hat).sum(dim=-1)
    cross = direction[..., 0] * n_hat[..., 1] - direction[..., 1] * n_hat[..., 0]
    return torch.atan2(cross.abs(), dot.abs())
