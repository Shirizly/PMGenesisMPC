"""
Per-epoch training metrics for push-dynamics models.

All metric classes share the same three-method interface::

    metrics.update(prediction, batch)   — accumulate stats for one batch
    metrics.compute() → dict[str, float] — aggregate over epoch, reset state
    metrics.reset()                     — clear accumulated state

The ``update`` method is decorated with ``@torch.no_grad()`` and is safe to
call inside an evaluation loop.

EulerianMetrics
---------------
Inputs to update():
    prediction : Tensor[B, 1, H, W]
        Raw logit from model.forward() (sigmoid applied internally).
    batch : dict
        "input":  Tensor[B, C, H, W] — channel 0 is current occupancy
        "target": Tensor[B, H, W]    — binary target occupancy after push

Output keys from compute() (all float):
    prob_mse           — MSE(sigmoid(logit), target)
    hard_iou           — mean per-sample IoU at 0.5 threshold
    hard_dice          — mean per-sample Dice at 0.5 threshold
    copy_mse           — MSE(current_occ, target)  [copy-input baseline]
    zero_mse           — MSE(zeros, target)         [zero-output baseline]
    changed_mse        — MSE on pixels where |target − current| > threshold
    changed_copy_mse   — copy-input MSE on the same changed pixels
    changed_pixel_frac — fraction of pixels that actually changed
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class EulerianMetrics:
    """
    Accumulates per-batch Eulerian occupancy metrics over an epoch.

    See module docstring for full input/output specification.
    """

    CHANGE_THRESHOLD = 1e-3  # pixels with |target - current| above this are "changed"

    def __init__(self):
        self.reset()

    def reset(self):
        self._s = dict(
            prob_mse_sum=0.0,
            zero_mse_sum=0.0,
            copy_mse_sum=0.0,
            iou_sum=0.0,
            dice_sum=0.0,
            changed_pred_sse=0.0,
            changed_copy_sse=0.0,
            changed_pixels=0,
            total_pixels=0,
            n=0,
        )

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, batch: dict):
        """
        Accumulate metrics for one batch.

        Parameters
        ----------
        prediction : Tensor[B, 1, H, W]  — raw logit
        batch      : dict with "input" Tensor[B, C, H, W] and "target" Tensor[B, H, W]
        """
        logits  = prediction.squeeze(1).float()  # (B, H, W)
        targets = batch["target"].float()         # (B, H, W)
        current = batch["input"][:, 0].float()    # (B, H, W)

        probs       = torch.sigmoid(logits)
        pred_mask   = probs   > 0.5
        target_mask = targets > 0.5
        changed     = (targets - current).abs() > self.CHANGE_THRESHOLD

        B     = logits.size(0)
        inter = (pred_mask & target_mask).float().sum(dim=(1, 2))
        pa    = pred_mask.float().sum(dim=(1, 2))
        ta    = target_mask.float().sum(dim=(1, 2))

        s = self._s
        s["prob_mse_sum"] += F.mse_loss(probs, targets).item() * B
        s["zero_mse_sum"] += F.mse_loss(torch.zeros_like(targets), targets).item() * B
        s["copy_mse_sum"] += F.mse_loss(current, targets).item() * B
        s["iou_sum"]      += ((inter + 1e-6) / (pa + ta - inter + 1e-6)).sum().item()
        s["dice_sum"]     += ((2.0 * inter + 1e-6) / (pa + ta + 1e-6)).sum().item()

        c_count = int(changed.sum().item())
        if c_count > 0:
            s["changed_pred_sse"] += ((probs - targets).pow(2) * changed).sum().item()
            s["changed_copy_sse"] += ((current - targets).pow(2) * changed).sum().item()
            s["changed_pixels"]   += c_count

        s["total_pixels"] += int(changed.numel())
        s["n"] += B

    def compute(self) -> dict[str, float]:
        """
        Return aggregated metrics for the epoch and reset internal state.

        Returns
        -------
        dict[str, float]  — see module docstring for key names.
        """
        s  = self._s
        n  = max(1, s["n"])
        cp = max(1, s["changed_pixels"])
        result = {
            "prob_mse":           s["prob_mse_sum"]     / n,
            "zero_mse":           s["zero_mse_sum"]     / n,
            "copy_mse":           s["copy_mse_sum"]     / n,
            "hard_iou":           s["iou_sum"]           / n,
            "hard_dice":          s["dice_sum"]          / n,
            "changed_mse":        s["changed_pred_sse"]  / cp,
            "changed_copy_mse":   s["changed_copy_sse"]  / cp,
            "changed_pixel_frac": s["changed_pixels"]    / max(1, s["total_pixels"]),
        }
        self.reset()
        return result
