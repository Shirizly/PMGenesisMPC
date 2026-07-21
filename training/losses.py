"""
Loss functions for training push-dynamics models.

Registry
--------
``register_loss(name)``   — decorator to add a LossFn class to the registry.
``build_loss(loss_cfg)``  — instantiate the right LossFn from a config dict.

Loss function contract
----------------------
All registered loss classes must be callable as::

    total_loss, components = loss_fn(prediction, batch)

where:

    prediction  : Tensor        — model output (dtype float32; shape depends on
                                  representation)
    batch       : dict          — standard batch dict (same keys produced by
                                  DatasetWrapper.__getitem__)
    total_loss  : Tensor[()]    — scalar, differentiable
    components  : dict[str, float]  — named sub-losses for TensorBoard logging
                                       (detached floats, not tensors)

Some losses (``EulerianCombinedLoss``, ``ScoreMapWeightedLoss``) accept a
``per_sample: true`` config key; when set, ``total_loss`` is instead an
unreduced ``Tensor[B]`` (one cost per batch item). This is used by
sampling-based MPC (``simple_mpc.oracle_mpc``) to rank candidates, not by the
training loop.

Adding a new loss function
--------------------------
1.  Subclass LossFn, implement ``__init__(cfg)`` and
    ``__call__(prediction, batch)``.
2.  Decorate with ``@register_loss("your_name")``.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from training.types import (
    EulerianBatch,
    LagrangianBatch,
    ModelOutput,
    TrainingBatch,
    prediction_to_logits,
)

_LOSS_REGISTRY: dict[str, Callable[[dict], "LossFn"]] = {}


def register_loss(name: str):
    """Decorator: ``@register_loss("eulerian_combined")``."""
    def decorator(cls):
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def build_loss(loss_cfg: dict) -> "LossFn":
    """
    Instantiate a LossFn from a loss config dict.

    Parameters
    ----------
    loss_cfg : dict — must contain ``type`` key (defaults to
               "eulerian_combined" if absent); remaining keys are
               loss-specific weight scalars.
    """
    ltype = loss_cfg.get("type", "eulerian_combined")
    if ltype not in _LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss type {ltype!r}. Registered: {sorted(_LOSS_REGISTRY)}"
        )
    return _LOSS_REGISTRY[ltype](loss_cfg)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class LossFn:
    """
    Abstract base for all loss functions.

    __call__(prediction, batch) → (total_loss: Tensor, components: dict[str, float])
    """

    def __call__(
        self, prediction: torch.Tensor | ModelOutput, batch: TrainingBatch
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Eulerian combined loss
# ---------------------------------------------------------------------------

@register_loss("eulerian_combined")
class EulerianCombinedLoss(LossFn):
    """
    Weighted combination of loss terms for Eulerian (occupancy-grid) models.

    Inputs
    ------
    prediction : Tensor[B, 1, H, W]
        Raw logit from model.forward() (float32; sigmoid NOT yet applied).
    batch : dict
        "input":  Tensor[B, C, H, W] — channel 0 is current occupancy
        "target": Tensor[B, H, W]    — binary target occupancy after push

    Outputs
    -------
    total_loss : Tensor[()]          — weighted sum, differentiable
    components : dict[str, float]    — each active term's value (for logging)

    Config keys (float weights; set to 0.0 to disable a term):
        mse        1.0  — MSE between sigmoid(logit) and target
        mass       0.0  — mean absolute mass-conservation error (normalised by area)
        dice       0.0  — soft Dice loss
        bce        0.0  — BCEWithLogitsLoss (uses pos_weight if provided)
        sharpness  0.0  — mean entropy of sigmoid output (penalises uncertainty)
        tv         0.0  — total variation on sigmoid output
        add        0.0  — MSE on the added-material map (pred > current)
        remove     0.0  — MSE on the removed-material map (current > pred)
        bce_pos_weight  1.0  — positive class weight for BCEWithLogitsLoss
        per_sample      False — if True, ``total_loss`` is returned unreduced
                                 as a ``(B,)`` tensor (one cost per batch item)
                                 instead of a scalar. Used by sampling-based
                                 MPC (simple_mpc.oracle_mpc), which needs a
                                 per-candidate cost. ``components`` are always
                                 batch-mean scalars regardless of this flag.
    """

    def __init__(self, cfg: dict):
        self.w_mse       = float(cfg.get("mse",        1.0))
        self.w_mass      = float(cfg.get("mass",        0.0))
        self.w_dice      = float(cfg.get("dice",        0.0))
        self.w_bce       = float(cfg.get("bce",         0.0))
        self.w_sharpness = float(cfg.get("sharpness",   0.0))
        self.w_tv        = float(cfg.get("tv",          0.0))
        self.w_add       = float(cfg.get("add",         0.0))
        self.w_remove    = float(cfg.get("remove",      0.0))
        pos_weight_val   = float(cfg.get("bce_pos_weight", 1.0))
        self._bce_pw     = pos_weight_val
        self.per_sample  = bool(cfg.get("per_sample", False))

    def __call__(
        self,
        prediction: torch.Tensor | ModelOutput,
        batch: EulerianBatch,
    ) -> tuple[torch.Tensor, dict]:
        logits = prediction_to_logits(prediction).float()
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        targets = batch.get("target_occupancy", batch["target"]).float()
        if "current_occupancy" in batch:
            current = batch["current_occupancy"].float()
        else:
            current = batch["input"][:, 0].float()
        dev     = logits.device

        probs = torch.sigmoid(logits)

        # --- individual terms, computed per-sample (B,) then optionally
        # reduced to a scalar below; numerically identical to the old
        # full-batch-mean formulas when per_sample=False, since H,W are
        # fixed across the batch. ---
        mse_ps = F.mse_loss(probs, targets, reduction="none").mean(dim=(1, 2))

        pw  = torch.tensor([self._bce_pw], dtype=torch.float32, device=dev)
        bce_ps = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none").mean(dim=(1, 2))

        dice_ps = _soft_dice_loss(logits, targets, reduce=False)

        sharpness_ps = (probs * (1.0 - probs)).mean(dim=(1, 2))

        tv_ps = (
            (probs[:, 1:, :] - probs[:, :-1, :]).abs().mean(dim=(1, 2))
            + (probs[:, :, 1:] - probs[:, :, :-1]).abs().mean(dim=(1, 2))
        )

        n_px  = float(targets[0].numel())
        mass_ps = (probs.sum(dim=(1, 2)) - targets.sum(dim=(1, 2))).abs() / n_px

        pred_add   = (probs   - current).clamp_min(0.0)
        target_add = (targets - current).clamp_min(0.0)
        pred_rem   = (current - probs  ).clamp_min(0.0)
        target_rem = (current - targets).clamp_min(0.0)
        add_ps    = F.mse_loss(pred_add, target_add, reduction="none").mean(dim=(1, 2))
        remove_ps = F.mse_loss(pred_rem, target_rem, reduction="none").mean(dim=(1, 2))

        total_ps = (
            self.w_mse       * mse_ps
            + self.w_bce       * bce_ps
            + self.w_dice      * dice_ps
            + self.w_sharpness * sharpness_ps
            + self.w_tv        * tv_ps
            + self.w_mass      * mass_ps
            + self.w_add       * add_ps
            + self.w_remove    * remove_ps
        )

        components = {
            "mse":       mse_ps.mean().item(),
            "bce":       bce_ps.mean().item(),
            "dice":      dice_ps.mean().item(),
            "sharpness": sharpness_ps.mean().item(),
            "tv":        tv_ps.mean().item(),
            "mass":      mass_ps.mean().item(),
            "add":       add_ps.mean().item(),
            "remove":    remove_ps.mean().item(),
        }
        total = total_ps if self.per_sample else total_ps.mean()
        return total, components


@register_loss("lagrangian_mse")
class LagrangianMSELoss(LossFn):
    """Masked position MSE for particle-based dynamics models."""

    def __call__(
        self,
        prediction: torch.Tensor | ModelOutput,
        batch: LagrangianBatch,
    ) -> tuple[torch.Tensor, dict]:
        pred = prediction_to_logits(prediction).float()  # (B, N, 3)
        target = batch.get("target_particles", batch["target"]).float()

        if "particle_nums" in batch:
            n_max = pred.shape[1]
            mask = (
                torch.arange(n_max, device=pred.device)[None, :]
                < batch["particle_nums"].long()[:, None]
            )
            sq = (pred - target).pow(2).sum(dim=-1)
            denom = (mask.sum().clamp_min(1) * pred.shape[-1]).float()
            mse = (sq * mask).sum() / denom
        else:
            mse = F.mse_loss(pred, target)

        return mse, {"mse": float(mse.item())}


def _soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    reduce: bool = True,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims  = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    per_sample = 1.0 - (2.0 * inter + eps) / (denom + eps)
    return per_sample.mean() if reduce else per_sample


# ---------------------------------------------------------------------------
# Score-map-weighted loss — literally the simple_mpc occupancy reward,
# registered as a loss so oracle MPC optimization and reporting can share the
# exact same objective (see docs/oracle_mpc_design.md "Cost").
# ---------------------------------------------------------------------------

@register_loss("score_map_weighted")
class ScoreMapWeightedLoss(LossFn):
    """
    Loss = -sum(occupancy_probs * score_map), i.e. the negative of the
    occupancy reward used elsewhere in simple_mpc
    (``EulerianAdapter._reward_default`` / ``OccupancyReward``).

    Inputs
    ------
    prediction : Tensor[B, H, W] or Tensor[B, 1, H, W] (logits) | ModelOutput
    batch : dict
        "score_map": Tensor[H, W] or Tensor[B, H, W] — fixed goal reward
            landscape (e.g. from ``OccupancyReward.compute_score_tensor``).

    Config keys:
        per_sample  False — see ``EulerianCombinedLoss``.
    """

    def __init__(self, cfg: dict):
        self.per_sample = bool(cfg.get("per_sample", False))

    def __call__(
        self,
        prediction: torch.Tensor | ModelOutput,
        batch: EulerianBatch,
    ) -> tuple[torch.Tensor, dict]:
        logits = prediction_to_logits(prediction).float()
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)

        score_map = batch["score_map"].float()
        if score_map.ndim == 2:
            score_map = score_map.unsqueeze(0)

        reward_ps = (probs.clamp(0.0, 1.0) * score_map).reshape(probs.shape[0], -1).sum(dim=-1)
        loss_ps   = -reward_ps

        total = loss_ps if self.per_sample else loss_ps.mean()
        return total, {"score_map_reward": reward_ps.mean().item()}
