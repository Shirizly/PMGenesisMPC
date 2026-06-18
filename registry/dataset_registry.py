"""
Dataset registry and wrappers.

All datasets produced by this registry return a **standard batch dict** so that
the Trainer, loss functions, and metrics are fully decoupled from the concrete
dataset class.

Standard Eulerian batch dict:

    "input":   Tensor[C, H, W]  — model input (occupancy + action channels, etc.)
    "target":  Tensor[H, W]     — prediction target (occupancy after push)
    "physics": Tensor[P]        — normalised [0, 1] physics vector
                                   (key ABSENT — not None — when not applicable)

The ``"physics"`` key is absent rather than None so that callers can use
``"physics" in batch`` as a clean check.  A batch collated from samples that
all lack ``"physics"`` will also lack the key.

Adding a new dataset
--------------------
1.  Write a ``build_<name>_dataset(cfg, split)`` function.
2.  Register it: ``_DATASET_REGISTRY["myname"] = build_<name>_dataset``.
3.  Return an ``EulerianDatasetWrapper`` (or write a custom wrapper that
    produces the same batch dict keys).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset

_DATASET_REGISTRY: dict[str, Callable[[dict, str], Dataset]] = {}


def build_dataset(cfg: dict, split: str) -> Dataset:
    """
    Instantiate a dataset wrapper from a dataset config dict.

    Parameters
    ----------
    cfg   : dict — must contain ``type`` key; remaining keys are dataset-specific.
    split : str  — "train", "val", "test", or None (all data).

    Returns
    -------
    Dataset whose ``__getitem__`` returns a standard batch dict.
    """
    dtype = cfg["type"]
    if dtype not in _DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset type {dtype!r}. Registered: {sorted(_DATASET_REGISTRY)}"
        )
    return _DATASET_REGISTRY[dtype](cfg, split)


# ---------------------------------------------------------------------------
# Eulerian wrapper
# ---------------------------------------------------------------------------

class EulerianDatasetWrapper(Dataset):
    """
    Wraps PileSweepData (or any dataset with the same output format) to produce
    standard batch dicts.

    Expected raw dataset ``__getitem__`` format:
        ((input_grid: Tensor[C, H, W], physics: Tensor[P]), target: Tensor[H, W])

    Produces:
        {
            "input":   Tensor[C, H, W],
            "target":  Tensor[H, W],
            "physics": Tensor[P],       ← only present if include_physics=True
        }

    Parameters
    ----------
    raw_dataset     : Dataset — PileSweepData or compatible.
    include_physics : bool    — whether to include the "physics" key.
    """

    def __init__(self, raw_dataset: Dataset, include_physics: bool = True):
        self.raw_dataset = raw_dataset
        self.include_physics = include_physics

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> dict:
        (input_grid, physics), target = self.raw_dataset[idx]
        sample: dict = {"input": input_grid, "target": target}
        if self.include_physics:
            sample["physics"] = physics
        return sample


# ---------------------------------------------------------------------------
# Registered factories
# ---------------------------------------------------------------------------

def _build_genesis_dataset(cfg: dict, split: str) -> EulerianDatasetWrapper:
    """
    Wrap PileSweepData (Genesis simulation data).

    Config keys:
        paths:                 list[str]  — paths relative to Genesis/data/
        val_pct:               int        (default 5)
        test_pct:              int        (default 5)
        resolution_scale:      float      (default 1.0)
        include_sweep_removed: bool       (default false)
        include_physics:       bool       (default true)
        physics.normalization: dict       — bounds used by PileSweepData._det_physics;
                                            passed to the dataset so normalisation
                                            is config-driven rather than hardcoded.
    """
    from Genesis.training.dataset import PileSweepData
    from physics.normalization import PhysicsBounds

    bounds = (
        PhysicsBounds.from_config(cfg["physics"])
        if "physics" in cfg
        else PhysicsBounds.default()
    )
    raw = PileSweepData(
        paths=cfg["paths"],
        split=split,
        val_pct=int(cfg.get("val_pct", 5)),
        test_pct=int(cfg.get("test_pct", 5)),
        resolution_scale=float(cfg.get("resolution_scale", 1.0)),
        include_sweep_removed=bool(cfg.get("include_sweep_removed", False)),
        physics_bounds=bounds,
    )
    return EulerianDatasetWrapper(raw, include_physics=bool(cfg.get("include_physics", True)))


_DATASET_REGISTRY["genesis"] = _build_genesis_dataset


def _build_real_dataset(cfg: dict, split: str) -> EulerianDatasetWrapper:
    """
    Wrap RealPileSweepData (real-world camera data).

    RealPileSweepData returns raw (un-normalised) physics values; this wrapper
    normalises them using the bounds from the dataset config.

    Config keys:
        data_root:         str        — root directory
        paths:             list[str]  — subdirectories under data_root
        val_pct:           int        (default 10)
        test_pct:          int        (default 10)
        default_physics:   list[float] (default [0, 0, 0])
        include_physics:   bool       (default true)
        physics.normalization: dict   — normalisation bounds
    """
    from RealData.dataset import RealPileSweepData
    from physics.normalization import PhysicsBounds

    bounds = (
        PhysicsBounds.from_config(cfg["physics"])
        if "physics" in cfg
        else PhysicsBounds.default()
    )
    raw = RealPileSweepData(
        data_root=cfg["data_root"],
        paths=cfg["paths"],
        split=split,
        default_physics=cfg.get("default_physics", [0.0, 0.0, 0.0]),
        val_pct=int(cfg.get("val_pct", 10)),
        test_pct=int(cfg.get("test_pct", 10)),
    )
    include_physics = bool(cfg.get("include_physics", True))
    return _RealEulerianDatasetWrapper(raw, bounds, include_physics)


class _RealEulerianDatasetWrapper(EulerianDatasetWrapper):
    """
    EulerianDatasetWrapper that normalises raw physics from RealPileSweepData.
    RealPileSweepData returns raw physical units; we apply bounds.normalize() here.
    """

    def __init__(self, raw_dataset: Dataset, bounds, include_physics: bool):
        super().__init__(raw_dataset, include_physics)
        self.bounds = bounds

    def __getitem__(self, idx: int) -> dict:
        (input_grid, raw_physics), target = self.raw_dataset[idx]
        sample: dict = {"input": input_grid, "target": target}
        if self.include_physics:
            sample["physics"] = self.bounds.normalize(raw_physics)
        return sample


_DATASET_REGISTRY["real"] = _build_real_dataset
