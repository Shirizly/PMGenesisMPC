"""
Dataset registry and wrappers.

All datasets produced by this registry return a **batch dict contract** so that
the Trainer, loss functions, and metrics stay decoupled from concrete dataset
classes. Batch keys are representation-specific.

Example Eulerian batch dict:

    "input":   Tensor[C, H, W]  — occupancy/action channels
    "target":  Tensor[H, W]     — next occupancy target
    "physics": Tensor[P]        — optional normalized physics vector

Future Lagrangian/particle wrappers should return particle-centric keys
(``particles``, ``target_particles``, etc.) while keeping the same outer dict
contract pattern.

The ``"physics"`` key is absent rather than None so callers can use
``"physics" in batch`` as a clean check.

Adding a new dataset
--------------------
1.  Write a ``build_<name>_dataset(cfg, split)`` function.
2.  Decorate it with ``@register_dataset("myname")``.
3.  Return a representation wrapper (e.g., ``EulerianDatasetWrapper``) or
    implement a custom wrapper that produces a documented batch dict.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch
import yaml
from torch.utils.data import Dataset

from transforms.functional import build_action_delta
from transforms.representation import (
    EnsureRepresentation,
    EulerianOccupancyAliases,
    LagrangianAliases,
    build_transforms,
)

_DATASET_REGISTRY: dict[str, Callable[[dict, str], Dataset]] = {}


def register_dataset(name: str):
    """Decorator: ``@register_dataset("genesis")``."""
    def decorator(factory_fn: Callable[[dict, str], Dataset]):
        _DATASET_REGISTRY[name] = factory_fn
        return factory_fn
    return decorator


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

    def __init__(
        self,
        raw_dataset: Dataset,
        include_physics: bool = True,
        transforms_cfg: list[dict] | None = None,
    ):
        self.raw_dataset = raw_dataset
        self.include_physics = include_physics
        self.transforms = build_transforms(
            transforms_cfg,
            defaults=[EnsureRepresentation("eulerian"), EulerianOccupancyAliases()],
        )

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> dict:
        (input_grid, physics), target = self.raw_dataset[idx]
        sample: dict = {"input": input_grid, "target": target}
        if self.include_physics:
            sample["physics"] = physics
        return self.transforms(sample)


class LagrangianDatasetWrapper(Dataset):
    """
    Wraps particle-transition datasets for GNN-like training.

    Expected raw item format:
        {
            "a_cur": Tensor[N],
            "s_cur": Tensor[N, 3],
            "s_delta": Tensor[N, 3],
            "target_particles": Tensor[N, 3],
            "particle_dens": Tensor[],
            "particle_num": Tensor[],
        }
    """

    def __init__(
        self,
        raw_dataset: Dataset,
        include_physics: bool = False,
        transforms_cfg: list[dict] | None = None,
    ):
        self.raw_dataset = raw_dataset
        self.include_physics = include_physics
        self.transforms = build_transforms(
            transforms_cfg,
            defaults=[EnsureRepresentation("lagrangian"), LagrangianAliases()],
        )

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.raw_dataset[idx]
        sample = {
            "a_cur": item["a_cur"],
            "s_cur": item["s_cur"],
            "s_delta": item["s_delta"],
            "target": item["target_particles"],
            "target_particles": item["target_particles"],
            "particle_dens": item["particle_dens"],
            "particle_num": item["particle_num"],
        }
        if self.include_physics and "physics" in item:
            sample["physics"] = item["physics"]
        return self.transforms(sample)

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        max_n = max(int(item["particle_num"].item()) for item in batch)

        a_cur = torch.zeros((batch_size, max_n), dtype=torch.float32)
        s_cur = torch.zeros((batch_size, max_n, 3), dtype=torch.float32)
        s_delta = torch.zeros((batch_size, max_n, 3), dtype=torch.float32)
        target = torch.zeros((batch_size, max_n, 3), dtype=torch.float32)
        particle_dens = torch.zeros((batch_size,), dtype=torch.float32)
        particle_nums = torch.zeros((batch_size,), dtype=torch.long)

        for b, item in enumerate(batch):
            n = int(item["particle_num"].item())
            particle_nums[b] = n
            particle_dens[b] = item["particle_dens"]
            a_cur[b, :n] = item["a_cur"]
            s_cur[b, :n] = item["s_cur"]
            s_delta[b, :n] = item["s_delta"]
            target[b, :n] = item["target_particles"]

        return {
            "a_cur": a_cur,
            "s_cur": s_cur,
            "s_delta": s_delta,
            "target": target,
            "target_particles": target,
            "particle_dens": particle_dens,
            "particle_nums": particle_nums,
        }


def _resolve_data_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent.parent / "Genesis" / "data" / path


def _collect_particle_run_paths(root: Path) -> list[tuple[Path, Path]]:
    run_paths: list[tuple[Path, Path]] = []
    for data_file in sorted(root.rglob("*_data.pt")):
        config_file = data_file.with_name(f"{data_file.stem.replace('_data', '')}_config.yaml")
        if config_file.exists():
            run_paths.append((data_file, config_file))
    return run_paths


def _assign_split(data_file: Path, val_pct: int, test_pct: int) -> str:
    bucket = int(hashlib.md5(str(data_file).encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < (test_pct + val_pct):
        return "val"
    return "train"


class GenesisParticlePushDataset(Dataset):
    """Particle transition dataset used by PropNetDiffDenModel training."""

    def __init__(
        self,
        paths: list[str],
        split: str,
        val_pct: int,
        test_pct: int,
        max_samples: int | None = None,
        action_sigma: float | None = None,
    ):
        assert split in ("train", "val", "test")
        self.action_sigma = action_sigma
        self.runs: list[dict] = []
        self.configs: list[dict] = []
        self.index: list[tuple[int, int]] = []

        run_entries: list[tuple[Path, Path]] = []
        for path in paths:
            root = _resolve_data_path(path)
            if not root.exists():
                raise FileNotFoundError(f"Data folder not found: {root}")
            run_entries.extend(_collect_particle_run_paths(root))

        if not run_entries:
            raise FileNotFoundError("No *_data.pt / *_config.yaml pairs found.")

        split_entries = [
            (d, c)
            for d, c in run_entries
            if _assign_split(d, val_pct=val_pct, test_pct=test_pct) == split
        ]
        if not split_entries and split == "train":
            split_entries = run_entries

        for run_idx, (data_file, config_file) in enumerate(split_entries):
            run_data = torch.load(data_file, map_location="cpu")
            cfg = yaml.full_load(config_file.read_text())
            states = torch.as_tensor(run_data["states"], dtype=torch.float32)
            states_next = torch.as_tensor(run_data["states_"], dtype=torch.float32)
            if states.shape != states_next.shape:
                raise ValueError(f"Mismatched states/states_ in {data_file}")

            self.runs.append(run_data)
            self.configs.append(cfg)
            for sample_idx in range(states.shape[0]):
                self.index.append((run_idx, sample_idx))

        if max_samples is not None:
            self.index = self.index[: max(0, int(max_samples))]
        if not self.index:
            raise ValueError(f"No samples found for split={split}.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        run_idx, sample_idx = self.index[idx]
        run_data = self.runs[run_idx]
        cfg = self.configs[run_idx]

        s_cur = torch.as_tensor(run_data["states"][sample_idx], dtype=torch.float32)[:, :3]
        s_next = torch.as_tensor(run_data["states_"][sample_idx], dtype=torch.float32)[:, :3]
        p_start = torch.as_tensor(run_data["p_starts"][sample_idx], dtype=torch.float32)
        p_stop = torch.as_tensor(run_data["p_stops"][sample_idx], dtype=torch.float32)

        particle_num = s_cur.shape[0]
        a_cur = torch.ones((particle_num,), dtype=torch.float32)

        sigma_m = self.action_sigma
        if sigma_m is None:
            plate_size = cfg.get("plate", {}).get("size", [0.04, 0.002, 0.01])
            sigma_m = max(float(plate_size[0]) * 0.5, 0.005)

        s_delta = build_action_delta(s_cur, p_start, p_stop, sigma_m=sigma_m)
        density = float(cfg.get("material", {}).get("density", 750.0))

        return {
            "a_cur": a_cur,
            "s_cur": s_cur,
            "s_delta": s_delta,
            "target_particles": s_next,
            "particle_dens": torch.tensor(density, dtype=torch.float32),
            "particle_num": torch.tensor(particle_num, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Registered factories
# ---------------------------------------------------------------------------

@register_dataset("genesis")
def _build_genesis_dataset(cfg: dict, split: str) -> EulerianDatasetWrapper:
    """
    Wrap PileSweepData (Genesis simulation data).

    Config keys:
        paths:                 list[str]  — paths relative to Genesis/data/
        val_pct:               int        (default 5)
        test_pct:              int        (default 5)
        resolution_scale:      float      (default 1.0)
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
        physics_bounds=bounds,
    )
    return EulerianDatasetWrapper(
        raw,
        include_physics=bool(cfg.get("include_physics", True)),
        transforms_cfg=cfg.get("transforms"),
    )


@register_dataset("real")
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
    return _RealEulerianDatasetWrapper(
        raw,
        bounds,
        include_physics,
        transforms_cfg=cfg.get("transforms"),
    )


class _RealEulerianDatasetWrapper(EulerianDatasetWrapper):
    """
    EulerianDatasetWrapper that normalises raw physics from RealPileSweepData.
    RealPileSweepData returns raw physical units; we apply bounds.normalize() here.
    """

    def __init__(
        self,
        raw_dataset: Dataset,
        bounds,
        include_physics: bool,
        transforms_cfg: list[dict] | None = None,
    ):
        super().__init__(raw_dataset, include_physics, transforms_cfg=transforms_cfg)
        self.bounds = bounds

    def __getitem__(self, idx: int) -> dict:
        (input_grid, raw_physics), target = self.raw_dataset[idx]
        sample: dict = {"input": input_grid, "target": target}
        if self.include_physics:
            sample["physics"] = self.bounds.normalize(raw_physics)
        return self.transforms(sample)


@register_dataset("genesis-particles")
def _build_genesis_particles_dataset(cfg: dict, split: str) -> LagrangianDatasetWrapper:
    """Build Lagrangian particle-transition dataset for GNN training."""
    raw = GenesisParticlePushDataset(
        paths=cfg["paths"],
        split=split,
        val_pct=int(cfg.get("val_pct", 10)),
        test_pct=int(cfg.get("test_pct", 10)),
        max_samples=cfg.get(f"max_{split}_samples", None),
        action_sigma=cfg.get("action_sigma", None),
    )
    return LagrangianDatasetWrapper(
        raw,
        include_physics=bool(cfg.get("include_physics", False)),
        transforms_cfg=cfg.get("transforms"),
    )
