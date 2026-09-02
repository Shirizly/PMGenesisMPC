"""
Model-agnostic training loop.

The Trainer is responsible for:
    dataset/loader construction, model instantiation, optimizer + scheduler
    + AMP GradScaler, augmentation dispatch, TensorBoard logging, early
    stopping, checkpoint saving, and model-card writing.

The Trainer is NOT responsible for:
    loss arithmetic, metric computation, model architecture — those are
    delegated to ``loss_fn``, ``metrics``, and ``model_wrapper``.

Config file path references
---------------------------
If a top-level key's value is a string ending in ``.yaml``, the referenced
file is loaded and its content replaces the string.  Paths are resolved
against the project root first, then against the config file's directory::

    model:   configs/model/unetfilm.yaml    ← loaded and inlined
    dataset: configs/dataset/genesis_cube.yaml

This allows composable configs without YAML extension libraries.

Checkpoint format
-----------------
Each checkpoint is a plain ``state_dict`` saved with ``torch.save``.
A ``model_card.yaml`` sidecar is always written alongside the best
checkpoint.  To save epoch/optimiser state use the ``save_full_state``
training flag (not yet implemented — the checkpoint is raw state_dict only).

Deduplication
-------------
``_get_log_dir()`` returns ``output.log_dir`` unchanged unless a completed
run (one that has written ``unet.pth``) already exists there — in that case
it appends ``_2``, ``_3``, etc.  An in-progress run (checkpoint exists but
no ``unet.pth``) is always resumed in-place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from registry.model_registry import build_model, ModelTrainingWrapper
from registry.dataset_registry import build_dataset
from model_training.losses import build_loss
from model_training.metrics import build_metrics
from model.model_card import ModelCard
from model_training.types import TrainingBatch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """
    Load a YAML config file and resolve any path-referenced sub-configs.

    Parameters
    ----------
    path : str | Path — path to a training config YAML.

    Returns
    -------
    dict — fully resolved config (sub-configs inlined under their keys).
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    return _resolve_path_includes(cfg, path.parent)


def _resolve_path_includes(cfg: dict, base_dir: Path) -> dict:
    """
    Replace any top-level string value ending in ``.yaml`` with the loaded
    sub-config.  Paths are resolved first relative to the working directory
    (project root), then relative to the config file's directory as a fallback.
    Only one level of resolution is applied.
    """
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, str) and value.endswith(".yaml"):
            # Try project root first (most natural for configs/ paths)
            from_root = Path(value)
            from_dir  = base_dir / value
            if from_root.exists():
                sub = from_root
            elif from_dir.exists():
                sub = from_dir
            else:
                raise FileNotFoundError(
                    f"Sub-config not found: tried '{from_root}' and '{from_dir}'"
                )
            resolved[key] = yaml.safe_load(sub.read_text())
        else:
            resolved[key] = value
    return resolved


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

def _to_device(batch: TrainingBatch, device: str) -> TrainingBatch:
    """Move all tensor values in a batch dict to ``device``."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _augment_eulerian_batch(batch: TrainingBatch) -> TrainingBatch:
    """
    Spatial ×8 augmentation for Eulerian batches: 4 rotations × 2 horizontal flips.

    Operates on:
        "input":   Tensor[B, C, H, W]  — spatial dims are the last two
        "target":  Tensor[B, H, W]
        "physics": Tensor[B, P]        — replicated ×8, not spatially modified

    Returns a new batch dict with batch dimension ×8.

    Note: this augmentation assumes spatial symmetry of the occupancy grid.
    It is only appropriate for Eulerian (grid-based) representations.
    For GNN/Lagrangian datasets, set ``augmentation: false`` in the training
    config.
    """
    x       = batch["input"]    # (B, C, H, W)
    targets = batch["target"]   # (B, H, W)

    xs, ts = [], []
    for k in range(4):
        xr = torch.rot90(x,       k, dims=(-2, -1))
        xm = torch.flip(xr, dims=[-1])
        tr = torch.rot90(targets, k, dims=(-2, -1))
        tm = torch.flip(tr, dims=[-1])
        xs.extend([xr, xm])
        ts.extend([tr, tm])

    new_batch: TrainingBatch = {
        "input":  torch.cat(xs, dim=0),
        "target": torch.cat(ts, dim=0),
    }
    if "physics" in batch:
        new_batch["physics"] = batch["physics"].repeat(8, 1)
    return new_batch


def _is_eulerian_batch(batch: TrainingBatch) -> bool:
    return ("input" in batch) and ("target" in batch)


def _batch_size(batch: TrainingBatch) -> int:
    """Infer batch size from the first present tensor key."""
    for key in (
        "input",
        "target",
        "s_cur",
        "target_particles",
        "particles",
        "a_cur",
        "particle_nums",
    ):
        tensor = batch.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0:
            return int(tensor.shape[0])
    raise KeyError("Cannot infer batch size from batch keys.")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Model-agnostic training loop for push-dynamics models.

    Parameters
    ----------
    model_wrapper : ModelTrainingWrapper
        Wraps the underlying nn.Module; forward(batch) → prediction tensor.
    train_ds / val_ds / test_ds : Dataset
        Each returns a standard batch dict (see registry/dataset_registry.py).
    loss_fn  : LossFn callable
        (prediction, batch) → (total_loss: Tensor, components: dict[str, float])
    metrics  : EulerianMetrics or compatible
        update / compute / reset interface.
    cfg : dict
        Fully resolved training config dict (has "model", "dataset",
        "training", "inference", "output" sections).
    """

    def __init__(
        self,
        model_wrapper: ModelTrainingWrapper,
        train_ds,
        val_ds,
        test_ds,
        loss_fn,
        metrics,
        cfg: dict,
    ):
        self.model_wrapper = model_wrapper.to(DEVICE)
        self.train_ds  = train_ds
        self.val_ds    = val_ds
        self.test_ds   = test_ds
        self.loss_fn   = loss_fn
        self.metrics   = metrics
        self.cfg       = cfg
        self._tcfg     = cfg.get("training", {})
        self._resumed  = False

    @classmethod
    def from_config(cls, config_path: str | Path, resume: bool = True) -> "Trainer":
        """
        Build a Trainer from a training config YAML file.

        Parameters
        ----------
        config_path : str | Path — path to training config YAML.
        resume      : bool       — if True, load checkpoint from log_dir before
                                   training (weights only; epoch counter resets).

        Required config sections: model, dataset, training
        Optional config sections: inference, output
        """
        cfg = load_config(config_path)
        cfg["_config_path"] = str(Path(config_path).resolve())

        model_wrapper = build_model(cfg["model"])
        default_loss_type = "lagrangian_mse" if cfg["model"]["type"] == "gnn-propnet" else "eulerian_combined"
        loss_cfg      = cfg["training"].get("loss", {"type": default_loss_type})
        loss_fn       = build_loss(loss_cfg)
        metrics       = build_metrics(cfg["model"]["type"])

        train_ds = build_dataset(cfg["dataset"], "train")
        val_ds   = build_dataset(cfg["dataset"], "val")
        test_ds  = build_dataset(cfg["dataset"], "test")

        trainer = cls(model_wrapper, train_ds, val_ds, test_ds, loss_fn, metrics, cfg)
        if resume:
            trainer._try_resume()
        return trainer

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full training loop, then evaluate on the test set."""
        tcfg    = self._tcfg
        log_dir = self._get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

        # Write config for reproducibility
        with open(log_dir / "run_config.yaml", "w") as f:
            yaml.dump(self.cfg, f, sort_keys=False)

        # Prepare model card (written after each best checkpoint)
        self._model_card = ModelCard.from_training_config(
            self.cfg, log_dir, "unet_best.pth"
        )

        writer     = SummaryWriter(log_dir=log_dir)
        optimizer, scheduler, scaler = self._build_optimizer()

        batch_size  = int(tcfg.get("batch_size", 64))
        augment     = bool(tcfg.get("augmentation", True))
        num_workers = int(tcfg.get("num_workers", 4))

        # Augmentation ×8: reduce loader batch size so augmented batch = batch_size
        loader_bs = max(1, batch_size // 8) if augment else batch_size

        train_loader = self._make_loader(self.train_ds, loader_bs, shuffle=True,  num_workers=num_workers)
        val_loader   = self._make_loader(self.val_ds,   loader_bs, shuffle=False, num_workers=num_workers)

        epochs     = int(tcfg.get("epochs",               100))
        patience   = int(tcfg.get("patience",             100))
        save_every = int(tcfg.get("save_every_n_epochs",   10))

        best_val_loss = float("inf")
        best_epoch    = 0
        no_improve    = 0

        print(
            f"Training on {DEVICE}, epochs 0→{epochs}, log_dir={log_dir}\n"
            f"Train: {len(self.train_ds)}  Val: {len(self.val_ds)}  Test: {len(self.test_ds)}"
        )

        with trange(epochs, desc="Epochs") as tbar:
            for epoch in tbar:

                # ── Train ────────────────────────────────────────────────────
                self.model_wrapper.train()
                train_loss_sum = 0.0
                train_comp_sum: dict[str, float] = {}
                train_n = 0

                for batch in train_loader:
                    batch = _to_device(batch, DEVICE)
                    if augment and _is_eulerian_batch(batch):
                        batch = _augment_eulerian_batch(batch)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                        prediction = self.model_wrapper(batch)
                        loss, components = self.loss_fn(prediction, batch)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model_wrapper.model.parameters(), grad_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()

                    bsz = _batch_size(batch)
                    train_loss_sum += loss.item() * bsz
                    for k, v in components.items():
                        train_comp_sum[k] = train_comp_sum.get(k, 0.0) + v * bsz
                    train_n += bsz

                scheduler.step()
                train_loss  = train_loss_sum / max(1, train_n)
                train_comps = {k: v / max(1, train_n) for k, v in train_comp_sum.items()}

                # ── Validate ─────────────────────────────────────────────────
                val_loss, val_comps, val_metrics = self._evaluate(val_loader)

                # ── Checkpoint / early stopping ───────────────────────────────
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch    = epoch + 1
                    no_improve    = 0
                    self._save_checkpoint(log_dir / "unet_best.pth")
                    self._model_card.save()
                else:
                    no_improve += 1

                if (epoch + 1) % save_every == 0:
                    self._save_checkpoint(log_dir / f"unet_epoch_{epoch + 1}.pth")

                # ── Logging ───────────────────────────────────────────────────
                self._log(
                    writer, epoch,
                    train_loss, train_comps,
                    val_loss, val_comps, val_metrics,
                    scheduler.get_last_lr()[0], best_epoch, no_improve,
                )
                tbar.set_postfix({
                    "trn":  f"{train_loss:.4f}",
                    "val":  f"{val_loss:.4f}",
                    "iou":  f"{val_metrics.get('hard_iou', 0):.3f}",
                    "best": best_epoch,
                    "ni":   no_improve,
                })
                print(
                    f"Epoch {epoch+1:4d}: trn={train_loss:.6f}  val={val_loss:.6f}  "
                    f"iou={val_metrics.get('hard_iou', 0):.4f}  "
                    f"chg_mse={val_metrics.get('changed_mse', 0):.6f}  "
                    f"best={best_epoch}"
                )

                if no_improve >= patience:
                    print(f"Early stopping after {patience} epochs without improvement.")
                    break

        # Write the final "completed" checkpoint; dedup uses this as sentinel
        self._save_checkpoint(log_dir / "unet.pth")
        writer.close()

        # ── Test ─────────────────────────────────────────────────────────────
        test_loader = self._make_loader(
            self.test_ds, loader_bs, shuffle=False, num_workers=num_workers
        )
        _, _, test_metrics = self._evaluate(test_loader)
        print("\n=== Test metrics ===")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")
        print(
            f"\n=== Training complete ===\n"
            f"  Best val loss : {best_val_loss:.6f}  (epoch {best_epoch})\n"
            f"  Model card    : {self._model_card.path}\n"
            f"  Log dir       : {log_dir}"
        )

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _evaluate(self, loader) -> tuple[float, dict, dict]:
        """Run one evaluation pass. Returns (mean_loss, loss_components, metrics)."""
        self.model_wrapper.eval()
        self.metrics.reset()
        loss_sum: float = 0.0
        comp_sum: dict[str, float] = {}
        n = 0

        with torch.no_grad():
            for batch in loader:
                batch      = _to_device(batch, DEVICE)
                prediction = self.model_wrapper(batch)
                loss, comps = self.loss_fn(prediction, batch)
                bsz = _batch_size(batch)
                loss_sum += loss.item() * bsz
                for k, v in comps.items():
                    comp_sum[k] = comp_sum.get(k, 0.0) + v * bsz
                n += bsz
                self.metrics.update(prediction, batch)

        mean_loss  = loss_sum / max(1, n)
        mean_comps = {k: v / max(1, n) for k, v in comp_sum.items()}
        return mean_loss, mean_comps, self.metrics.compute()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_log_dir(self) -> Path:
        """
        Return the log directory, deduplicating if a completed run already lives there.
        Completed = ``unet.pth`` present (written at the very end of run()).
        An in-progress run (has checkpoints but no unet.pth) is resumed in-place.
        """
        base = Path(self.cfg.get("output", {}).get("log_dir", "runs/training"))
        if self._resumed:
            return base   # resume in the same directory
        if (base / "unet.pth").exists():
            counter = 2
            while True:
                cand = base.with_name(f"{base.name}_{counter}")
                if not (cand / "unet.pth").exists():
                    return cand
                counter += 1
        return base

    def _build_optimizer(self):
        tcfg = self._tcfg
        lr   = float(tcfg.get("lr", 1e-4))

        optimizer = torch.optim.Adam(
            self.model_wrapper.model.parameters(), lr=lr
        )
        sched_cfg  = tcfg.get("lr_scheduler", {})
        step_size  = int(sched_cfg.get("step_size", 100))
        gamma      = float(sched_cfg.get("gamma", 0.75))
        scheduler  = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        use_amp = bool(tcfg.get("mixed_precision", True)) and (DEVICE == "cuda")
        scaler  = torch.amp.GradScaler(enabled=use_amp)
        return optimizer, scheduler, scaler

    def _make_loader(self, ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
        collate_fn = getattr(ds, "collate_fn", None)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(DEVICE == "cuda"),
            collate_fn=collate_fn,
        )

    def _save_checkpoint(self, path: Path) -> None:
        torch.save(self.model_wrapper.state_dict(), path)

    def _try_resume(self) -> None:
        """
        If a checkpoint exists in output.log_dir, load it (weights only).
        Sets self._resumed = True so _get_log_dir() keeps the same directory.
        """
        log_dir = Path(self.cfg.get("output", {}).get("log_dir", "runs/training"))
        if not log_dir.exists():
            return

        # Prefer the most recent epoch checkpoint, fall back to unet_best.pth
        candidates = []
        for p in log_dir.glob("unet_epoch_*.pth"):
            try:
                candidates.append((int(p.stem.split("_")[-1]), p))
            except ValueError:
                pass
        if candidates:
            _, latest = max(candidates)
        elif (log_dir / "unet_best.pth").exists():
            latest = log_dir / "unet_best.pth"
        else:
            return

        state = torch.load(latest, map_location=DEVICE, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model_wrapper.load_state_dict(state["model_state_dict"])
        else:
            self.model_wrapper.load_state_dict(state)

        self._resumed = True
        print(f"Resumed weights from {latest}")

    def _log(
        self, writer, epoch,
        train_loss, train_comps,
        val_loss, val_comps, val_metrics,
        lr, best_epoch, no_improve,
    ) -> None:
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val",   val_loss,   epoch)
        writer.add_scalar("LR",         lr,         epoch)
        writer.add_scalar("Convergence/BestEpoch",            best_epoch, epoch)
        writer.add_scalar("Convergence/EpochsNoImprovement",  no_improve, epoch)
        for k, v in train_comps.items():
            writer.add_scalar(f"TrainComponent/{k}", v, epoch)
        for k, v in val_comps.items():
            writer.add_scalar(f"ValComponent/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"ValMetric/{k}", v, epoch)
