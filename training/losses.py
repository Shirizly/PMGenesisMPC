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
        self, prediction: torch.Tensor, batch: dict
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

    def __call__(
        self,
        prediction: torch.Tensor,   # (B, 1, H, W) raw logit, float32
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        prediction = prediction.float()
        logits  = prediction.squeeze(1)               # (B, H, W)
        targets = batch["target"].float()             # (B, H, W)
        current = batch["input"][:, 0].float()        # (B, H, W)
        dev     = logits.device

        probs = torch.sigmoid(logits)

        # --- individual terms ---
        mse = F.mse_loss(probs, targets)

        pw  = torch.tensor([self._bce_pw], dtype=torch.float32, device=dev)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

        dice = _soft_dice_loss(logits, targets)

        sharpness = (probs * (1.0 - probs)).mean()

        tv = (
            (probs[:, 1:, :] - probs[:, :-1, :]).abs().mean()
            + (probs[:, :, 1:] - probs[:, :, :-1]).abs().mean()
        )

        n_px  = float(targets[0].numel())
        mass  = (probs.sum(dim=(1, 2)) - targets.sum(dim=(1, 2))).abs().mean() / n_px

        pred_add   = (probs   - current).clamp_min(0.0)
        target_add = (targets - current).clamp_min(0.0)
        pred_rem   = (current - probs  ).clamp_min(0.0)
        target_rem = (current - targets).clamp_min(0.0)
        add_loss    = F.mse_loss(pred_add,   target_add)
        remove_loss = F.mse_loss(pred_rem,   target_rem)

        total = (
            self.w_mse       * mse
            + self.w_bce       * bce
            + self.w_dice      * dice
            + self.w_sharpness * sharpness
            + self.w_tv        * tv
            + self.w_mass      * mass
            + self.w_add       * add_loss
            + self.w_remove    * remove_loss
        )

        components = {
            "mse":       mse.item(),
            "bce":       bce.item(),
            "dice":      dice.item(),
            "sharpness": sharpness.item(),
            "tv":        tv.item(),
            "mass":      mass.item(),
            "add":       add_loss.item(),
            "remove":    remove_loss.item(),
        }
        return total, components


def _soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims  = tuple(range(1, probs.ndim))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()
