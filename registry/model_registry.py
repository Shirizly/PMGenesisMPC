"""
Model registry and training-time forward wrappers.

Registry
--------
``register_model(name)``  — decorator that adds a model factory to the registry.
``build_model(cfg)``      — instantiate a ModelTrainingWrapper from a config dict.

Training-time wrappers
----------------------
A ModelTrainingWrapper is an nn.Module whose ``forward(batch)`` accepts a
**batch dict** and returns a **prediction tensor**.  This decouples the Trainer
from the calling convention of each underlying model.

Batch dict keys consumed by EulerianTrainingWrapper:

    "input":   Tensor[B, C, H, W]  — stacked channels (occupancy + action, etc.)
    "physics": Tensor[B, P]        — normalised [0, 1] physics vector
                                     (key absent if uses_physics=False)

Returns:

    Tensor[B, 1, H, W]  — raw logit; sigmoid NOT applied (loss/metrics handle this)

Adding a new model
------------------
1.  Write a factory function that accepts ``cfg: dict`` and returns a
    ``ModelTrainingWrapper`` instance.
2.  Decorate it with ``@register_model("your_name")``.
3.  Add a ``configs/model/your_name.yaml`` for the canonical defaults.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from training.types import TrainingBatch, ModelOutput

_REGISTRY: dict[str, Callable[[dict], "ModelTrainingWrapper"]] = {}


def register_model(name: str):
    """Decorator: ``@register_model("unetfilm")``."""
    def decorator(factory_fn: Callable[[dict], "ModelTrainingWrapper"]):
        _REGISTRY[name] = factory_fn
        return factory_fn
    return decorator


def build_model(cfg: dict) -> "ModelTrainingWrapper":
    """
    Instantiate a ModelTrainingWrapper from a model config dict.

    Parameters
    ----------
    cfg : dict
        Must contain ``type`` (str); remaining keys are model-specific.

    Returns
    -------
    ModelTrainingWrapper  ready for use in Trainer.
    """
    mtype = cfg["type"]
    if mtype not in _REGISTRY:
        raise KeyError(
            f"Unknown model type {mtype!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[mtype](cfg)


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------

class ModelTrainingWrapper(nn.Module):
    """
    Abstract base for all training-time model wrappers.

    Wraps a concrete ``nn.Module`` and exposes a uniform ``forward(batch)``
    interface so the Trainer, loss functions, and metrics never import the
    underlying model class.

    Checkpoint saving/loading operates on ``self.model.state_dict()`` so that
    the saved ``.pth`` files are compatible with direct model loading
    (e.g. loading into ``UNetFiLMPushModel`` for MPC inference).

    Attributes
    ----------
    model        : nn.Module — the underlying model
    uses_physics : bool      — whether this model consumes a "physics" batch key
    """

    def __init__(self, model: nn.Module, uses_physics: bool):
        super().__init__()
        self.model = model
        self.uses_physics = uses_physics

    def forward(self, batch: TrainingBatch) -> torch.Tensor | ModelOutput:
        raise NotImplementedError

    def state_dict(self, **kwargs):
        return self.model.state_dict(**kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        return self.model.load_state_dict(state_dict, strict=strict, **kwargs)


# ---------------------------------------------------------------------------
# Eulerian (occupancy-grid) wrapper
# ---------------------------------------------------------------------------

class EulerianTrainingWrapper(ModelTrainingWrapper):
    """
    Wraps an Eulerian occupancy model (e.g. NFDUNetFiLM) for training.

    forward(batch) → Tensor[B, 1, H, W]

    Batch keys consumed:
        "input":   Tensor[B, C, H, W]  — occupancy + action channels
        "physics": Tensor[B, P]        — normalised physics
                                         (required only if uses_physics=True;
                                          raises ValueError if absent when needed)

    Returns raw logit.  Sigmoid is NOT applied here — the loss function and
    metrics apply it as needed so that BCEWithLogitsLoss receives the raw logit.
    """

    def forward(self, batch: TrainingBatch) -> torch.Tensor | ModelOutput:
        x = batch["input"]
        if self.uses_physics:
            physics = batch.get("physics")
            if physics is None:
                raise ValueError(
                    "Model requires 'physics' key in batch but it is absent.  "
                    "Set uses_physics: false in the model config to disable, "
                    "or ensure the dataset config has include_physics: true."
                )
            return self.model(x, physics)
        return self.model(x)


class LagrangianTrainingWrapper(ModelTrainingWrapper):
    """Wraps particle-based models (e.g. PropNetDiffDenModel) for training."""

    def __init__(self, model: nn.Module):
        super().__init__(model, uses_physics=False)

    def forward(self, batch: TrainingBatch) -> torch.Tensor | ModelOutput:
        a_cur = batch["a_cur"]
        s_cur = batch["s_cur"]
        s_delta = batch["s_delta"]
        particle_dens = batch["particle_dens"]
        particle_nums = batch.get("particle_nums")
        return self.model.predict_one_step(
            a_cur,
            s_cur,
            s_delta,
            particle_dens,
            particle_nums=particle_nums,
        )


# ---------------------------------------------------------------------------
# Registered factories
# ---------------------------------------------------------------------------

@register_model("unetfilm")
def _build_unetfilm(cfg: dict) -> EulerianTrainingWrapper:
    """
    Full-depth NFDUNetFiLM (FiLM-conditioned U-Net).

    Config keys (all optional, defaults shown):
        in_channels:      2        — input channels (3 if sweep-removed modes)
        cond_dim:         3        — physics conditioning dimension
        uses_physics:     true     — whether to consume "physics" from the batch
        input_mode:       standard — standard | sweep-removed-input |
                                     sweep-removed-residual
        residual_channel: 0        — which input channel to use as residual skip
        depth:            3        — number of pooling levels (default is 3)
    """
    from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLM
    input_mode = cfg.get("input_mode", "standard")
    in_ch  = 3 if input_mode in ("sweep-removed-input", "sweep-removed-residual") else int(cfg.get("in_channels", 2))
    res_ch = 2 if input_mode == "sweep-removed-residual" else int(cfg.get("residual_channel", 0))
    depth = int(cfg.get("depth", 3)) # Use depth from config, default to 3
    model = NFDUNetFiLM(
        in_channels=in_ch,
        out_channels=1,
        cond_dim=int(cfg.get("cond_dim", 3)),
        depth=depth, # Pass the configurable depth
        residual_channel=res_ch,
    )
    return EulerianTrainingWrapper(model, uses_physics=bool(cfg.get("uses_physics", True)))


@register_model("unetfilm-shallow")
def _build_unetfilm_shallow(cfg: dict) -> EulerianTrainingWrapper:
    """
    Lightweight NFDUNetFiLMShallow.  Same config keys as ``unetfilm``.

    Config keys (all optional, defaults shown):
        in_channels:      2        — input channels (3 if sweep-removed modes)
        cond_dim:         3        — physics conditioning dimension
        uses_physics:     true     — whether to consume "physics" from the batch
        input_mode:       standard — standard | sweep-removed-input |
                                     sweep-removed-residual
        residual_channel: 0        — which input channel to use as residual skip
        depth:            2        — number of pooling levels (default is 2, matching original shallow depth)
    """
    from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLMShallow
    input_mode = cfg.get("input_mode", "standard")
    in_ch  = 3 if input_mode in ("sweep-removed-input", "sweep-removed-residual") else int(cfg.get("in_channels", 2))
    res_ch = 2 if input_mode == "sweep-removed-residual" else int(cfg.get("residual_channel", 0))
    depth = int(cfg.get("depth", 2)) # Use depth from config, default to 2 (shallow)
    model = NFDUNetFiLMShallow(
        in_channels=in_ch,
        out_channels=1,
        cond_dim=int(cfg.get("cond_dim", 3)),
        depth=depth, # Pass the configurable depth
        residual_channel=res_ch,
    )
    return EulerianTrainingWrapper(model, uses_physics=bool(cfg.get("uses_physics", True)))


@register_model("gnn-propnet")
def _build_gnn_propnet(cfg: dict) -> LagrangianTrainingWrapper:
    """PropNetDiffDenModel for Lagrangian particle dynamics training."""
    from model.gnn_dyn import PropNetDiffDenModel

    model_cfg = {
        "train": {
            "particle": {
                "nf_effect": int(cfg.get("nf_effect", 150)),
                "add_delta": bool(cfg.get("add_delta", False)),
                "adj_thresh": float(cfg.get("adj_thresh", 0.08)),
            }
        }
    }
    model = PropNetDiffDenModel(model_cfg, use_gpu=torch.cuda.is_available())
    return LagrangianTrainingWrapper(model)
