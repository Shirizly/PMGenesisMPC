"""Typed contracts shared across training, evaluation, and MPC-facing wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import torch


class BaseBatch(TypedDict, total=False):
    """Fields shared across representations."""

    representation: Literal["eulerian", "lagrangian"]
    physics: torch.Tensor
    metadata: dict


class EulerianBatch(BaseBatch, total=False):
    """
    Eulerian occupancy-grid batch.

    Common keys:
    - input: Tensor[B, C, H, W]
    - target: Tensor[B, H, W]

    Optional keys:
    - current_occupancy: Tensor[B, H, W]
    - target_occupancy: Tensor[B, H, W]
    """

    input: torch.Tensor
    target: torch.Tensor
    current_occupancy: torch.Tensor
    target_occupancy: torch.Tensor


class LagrangianBatch(BaseBatch, total=False):
    """
    Particle-based batch for GNN-like models.

    Common keys:
    - s_cur: Tensor[B, N, 3]

    Optional keys:
    - target_particles: Tensor[B, N, 3]
    - a_cur: Tensor[B, N]
    - s_delta: Tensor[B, N, 3]
    - particle_dens: Tensor[B]
    - particle_nums: Tensor[B]
    - particle_mask: Tensor[B, N]
    """

    # Generic aliases
    particles: torch.Tensor
    target_particles: torch.Tensor
    action: torch.Tensor

    # Current GNN/PropNet keys
    a_cur: torch.Tensor
    s_cur: torch.Tensor
    s_delta: torch.Tensor
    particle_dens: torch.Tensor
    particle_num: torch.Tensor
    particle_nums: torch.Tensor

    particle_mask: torch.Tensor


# Backward-compatible broad alias used by training modules.
TrainingBatch = EulerianBatch | LagrangianBatch


@dataclass(frozen=True)
class ModelOutput:
    """
    Structured prediction contract.

    `logits` is required and remains the canonical training output.
    `probabilities` is optional. If absent, callers derive it as sigmoid(logits).
    """

    logits: torch.Tensor
    probabilities: torch.Tensor | None = None


def prediction_to_logits(prediction: torch.Tensor | ModelOutput) -> torch.Tensor:
    """Extract raw logits from either a tensor prediction or a ModelOutput."""
    if isinstance(prediction, ModelOutput):
        return prediction.logits
    return prediction


def prediction_to_probabilities(prediction: torch.Tensor | ModelOutput) -> torch.Tensor:
    """Extract probabilities from prediction, deriving from logits when needed."""
    if isinstance(prediction, ModelOutput):
        if prediction.probabilities is not None:
            return prediction.probabilities
        return torch.sigmoid(prediction.logits)
    return torch.sigmoid(prediction)
