"""
CLI entry point for training and evaluation.

Usage
-----
Train::

    python -m training.train configs/training/unetfilm_genesis.yaml
    python -m training.train configs/training/unetfilm_genesis.yaml --no-resume
    python -m training.train configs/training/unetfilm_genesis.yaml \\
        --override training.epochs=200 output.log_dir=runs_cubes/my_run

Evaluate only (loads ``unet_best.pth`` from output.log_dir by default)::

    python -m training.train configs/training/unetfilm_genesis.yaml --eval-only
    python -m training.train configs/training/unetfilm_genesis.yaml \\
        --eval-only --checkpoint runs_cubes/my_run/unet_epoch_50.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation key=value overrides to a config dict in-place.

    Values are parsed as YAML scalars so integers, floats, booleans, and
    strings are all handled correctly::

        training.epochs=200
        output.log_dir=runs_cubes/my_run
        training.augmentation=false
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got: {override!r}")
        key, _, raw_value = override.partition("=")
        value  = yaml.safe_load(raw_value)
        parts  = key.split(".")
        d      = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return cfg


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate a push-dynamics model.")
    parser.add_argument("config", type=Path, help="Path to training config YAML.")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Evaluate on test set without training.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start from scratch even if a checkpoint already exists in log_dir.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Path to a specific checkpoint for --eval-only (default: unet_best.pth).",
    )
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config keys, e.g. --override training.epochs=200.",
    )
    args = parser.parse_args(argv)

    from training.trainer import load_config, DEVICE
    cfg = load_config(args.config)
    _apply_overrides(cfg, args.override or [])

    print(f"Config : {args.config}")
    print(f"Device : {DEVICE}")

    if args.eval_only:
        _run_eval_only(cfg, args)
        return

    from training.trainer import Trainer
    trainer = Trainer.from_config(args.config, resume=not args.no_resume)
    _apply_overrides(trainer.cfg, args.override or [])
    trainer.run()


def _run_eval_only(cfg: dict, args) -> None:
    from registry.model_registry import build_model
    from registry.dataset_registry import build_dataset
    from training.losses import build_loss
    from training.metrics import build_metrics
    from training.trainer import _to_device, _batch_size, DEVICE

    model_wrapper = build_model(cfg["model"])

    ckpt_path = args.checkpoint or (
        Path(cfg["output"]["log_dir"]) / "unet_best.pth"
    )
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model_wrapper.load_state_dict(state["model_state_dict"])
    else:
        model_wrapper.load_state_dict(state)
    model_wrapper = model_wrapper.to(DEVICE)
    model_wrapper.eval()

    test_ds  = build_dataset(cfg["dataset"], "test")
    default_loss_type = "lagrangian_mse" if cfg["model"]["type"] == "gnn-propnet" else "eulerian_combined"
    loss_fn  = build_loss(cfg["training"].get("loss", {"type": default_loss_type}))
    metrics  = build_metrics(cfg["model"]["type"])

    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=int(cfg["training"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=(DEVICE == "cuda"),
    )

    loss_sum: float = 0.0
    comp_sum: dict  = {}
    n = 0

    with torch.no_grad():
        for batch in loader:
            batch      = _to_device(batch, DEVICE)
            prediction = model_wrapper(batch)
            loss, comps = loss_fn(prediction, batch)
            bsz = _batch_size(batch)
            loss_sum += loss.item() * bsz
            for k, v in comps.items():
                comp_sum[k] = comp_sum.get(k, 0.0) + v * bsz
            n += bsz
            metrics.update(prediction, batch)

    mean_loss    = loss_sum / max(1, n)
    test_metrics = metrics.compute()

    print(f"\n=== Test results  checkpoint: {ckpt_path} ===")
    print(f"  loss: {mean_loss:.6f}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
