#############################################################
# THIS SCRIPT IS USED TO TRAIN A GNN WITH THE GENESIS DATA #
#############################################################
import argparse
import hashlib
import math
import re
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from model.gnn_dyn import PropNetDiffDenModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 5e-3
DEFAULT_SCHEDULER_STEP_SIZE = 100
DEFAULT_SCHEDULER_GAMMA = 0.75
DEFAULT_PATIENCE = 200
DEFAULT_DATA_FOLDERS = ["ignore/cube/n10"]
DEFAULT_LOG_DIR = Path("runs_cubes/gnn_mse_smalldata_n10")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate PropNetDiffDenModel on Genesis particle transitions.")

    parser.add_argument("--eval-only", action="store_true", help="Load checkpoint and only evaluate test split.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for --eval-only. Defaults to <log-dir>/gnn.pth.")

    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--data-folders", nargs="+", default=DEFAULT_DATA_FOLDERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fresh-start", action="store_true", help="Do not resume from existing checkpoint.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Checkpoint path for training resume.")
    parser.add_argument("--start-epoch", type=int, default=None, help="Override resumed epoch index.")

    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--scheduler-step-size", type=int, default=DEFAULT_SCHEDULER_STEP_SIZE)
    parser.add_argument("--scheduler-gamma", type=float, default=DEFAULT_SCHEDULER_GAMMA)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    parser.add_argument("--val-pct", type=int, default=10)
    parser.add_argument("--test-pct", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--nf-effect", type=int, default=150)
    parser.add_argument("--adj-thresh", type=float, default=0.08)
    parser.add_argument("--add-delta", action="store_true")

    parser.add_argument(
        "--action-sigma",
        type=float,
        default=None,
        help="Gaussian influence sigma (meters) for building s_delta from push action.",
    )
    parser.add_argument("--save-every", type=int, default=10)

    return parser.parse_args()


def checkpoint_epoch(path: Path) -> int | None:
    match = re.search(r"epoch_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def latest_epoch_checkpoint(log_dir: Path) -> Path | None:
    checkpoints = []
    for path in log_dir.glob("gnn_epoch_*.pth"):
        epoch = checkpoint_epoch(path)
        if epoch is not None:
            checkpoints.append((epoch, path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def default_resume_checkpoint(log_dir: Path) -> Path | None:
    latest = latest_epoch_checkpoint(log_dir)
    if latest is not None:
        return latest

    best = log_dir / "gnn_best.pth"
    if best.exists():
        return best

    final = log_dir / "gnn.pth"
    if final.exists():
        return final

    return None


def unique_log_dir(base: Path) -> Path:
    if not base.exists() or not (base / "run_config.yaml").exists():
        return base
    counter = 2
    while True:
        candidate = base.with_name(f"{base.name}_{counter}")
        if not candidate.exists() or not (candidate / "run_config.yaml").exists():
            return candidate
        counter += 1


def _physics_key(cfg: dict) -> str:
    material = cfg.get("material", {})
    box = cfg.get("box", {})
    key_parts = [
        str(material.get("friction", "na")),
        str(material.get("density", "na")),
        str(box.get("friction", "na")),
        str(material.get("shape", "na")),
        str(material.get("n_particles", "na")),
        str(material.get("particle_size", "na")),
    ]
    return "|".join(key_parts)


def _assign_group_splits(num_groups: int, val_pct: int, test_pct: int) -> list[str]:
    if num_groups <= 0:
        return []

    test_count = round(num_groups * test_pct / 100)
    val_count = round(num_groups * val_pct / 100)

    if test_pct > 0 and test_count == 0 and num_groups >= 3:
        test_count = 1
    if val_pct > 0 and val_count == 0 and num_groups - test_count >= 2:
        val_count = 1

    if test_count + val_count >= num_groups:
        overflow = test_count + val_count - (num_groups - 1)
        val_count = max(0, val_count - overflow)
        overflow = test_count + val_count - (num_groups - 1)
        test_count = max(0, test_count - overflow)

    return ["test"] * test_count + ["val"] * val_count + ["train"] * (num_groups - test_count - val_count)


def _resolve_data_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).parent / "Genesis" / "data" / path


def _collect_run_paths(root: Path, run: int | None = None) -> list[tuple[Path, Path]]:
    run_paths: list[tuple[Path, Path]] = []

    if run is not None:
        run_name = str(run)
        expected_data = root / f"{run_name}_data.pt"
        expected_config = root / f"{run_name}_config.yaml"
        _expected_data = root / f"_{run_name}_data.pt"
        _expected_config = root / f"_{run_name}_config.yaml"

        if expected_data.exists() and expected_config.exists():
            return [(expected_data, expected_config)]
        if _expected_data.exists() and _expected_config.exists():
            return [(_expected_data, _expected_config)]

        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            data_file = subdir / f"{run_name}_data.pt"
            config_file = subdir / f"{run_name}_config.yaml"
            _data_file = subdir / f"_{run_name}_data.pt"
            _config_file = subdir / f"_{run_name}_config.yaml"
            if data_file.exists() and config_file.exists():
                return [(data_file, config_file)]
            if _data_file.exists() and _config_file.exists():
                return [(_data_file, _config_file)]
        return []

    for data_file in sorted(root.rglob("*_data.pt")):
        config_file = data_file.with_name(f"{data_file.stem.replace('_data', '')}_config.yaml")
        if config_file.exists():
            run_paths.append((data_file, config_file))

    return run_paths


def _filter_split(
    run_files: list[tuple[Path, Path]],
    split: str,
    val_pct: int,
    test_pct: int,
) -> list[tuple[Path, Path]]:
    split_by_file: dict[Path, str] = {}
    folder_groups: dict[Path, dict[str, list[tuple[Path, Path]]]] = {}

    for data_file, config_file in run_files:
        cfg = yaml.full_load(config_file.read_text())
        physics_key = _physics_key(cfg)
        folder_groups.setdefault(data_file.parent, {}).setdefault(physics_key, []).append((data_file, config_file))

    for physics_groups in folder_groups.values():
        groups = sorted(physics_groups.items(), key=lambda item: hashlib.md5(item[0].encode()).hexdigest())
        assignments = _assign_group_splits(len(groups), val_pct, test_pct)
        for (_, group), assigned_split in zip(groups, assignments):
            for data_file, _ in group:
                split_by_file[data_file] = assigned_split

    return [
        (data_file, config_file)
        for data_file, config_file in run_files
        if split_by_file.get(data_file) == split
    ]


def collect_run_paths(paths: list[str], split: str | None, val_pct: int, test_pct: int) -> list[tuple[Path, Path]]:
    assert split in (None, "train", "val", "test"), f"Invalid split: {split!r}"
    run_files: list[tuple[Path, Path]] = []

    for path in paths:
        root = _resolve_data_path(path)
        if not root.exists():
            raise FileNotFoundError(f"Data folder not found: {root}")

        path_run_files = _collect_run_paths(root)
        if not path_run_files:
            raise FileNotFoundError(f"No data runs found in path: {root}")
        run_files.extend(path_run_files)

    if not run_files:
        raise FileNotFoundError("No *_data.pt / *_config.yaml pairs found for the provided data-folders.")

    if split is None:
        return run_files

    split_run_files = _filter_split(run_files, split, val_pct, test_pct)
    if split_run_files:
        return split_run_files

    # Fallback for tiny datasets where a split may become empty after grouping.
    fallback_bucket = []
    for data_file, config_file in run_files:
        bucket = int(hashlib.md5(str(data_file).encode()).hexdigest(), 16) % 100
        if split == "test" and bucket < test_pct:
            fallback_bucket.append((data_file, config_file))
        elif split == "val" and test_pct <= bucket < (test_pct + val_pct):
            fallback_bucket.append((data_file, config_file))
        elif split == "train" and bucket >= (test_pct + val_pct):
            fallback_bucket.append((data_file, config_file))

    if fallback_bucket:
        return fallback_bucket

    return run_files if split == "train" else []


def _point_to_segment_distance_xy(points_xy: torch.Tensor, seg_start_xy: torch.Tensor, seg_end_xy: torch.Tensor) -> torch.Tensor:
    vec = seg_end_xy - seg_start_xy
    vec_norm_sq = torch.dot(vec, vec).clamp_min(1e-12)
    rel = points_xy - seg_start_xy[None, :]
    t = (rel * vec[None, :]).sum(dim=1) / vec_norm_sq
    t = t.clamp(0.0, 1.0)
    proj = seg_start_xy[None, :] + t[:, None] * vec[None, :]
    return (points_xy - proj).norm(dim=1)


def build_action_delta(
    s_cur_xyz: torch.Tensor,
    p_start_xyz: torch.Tensor,
    p_stop_xyz: torch.Tensor,
    sigma_m: float,
) -> torch.Tensor:
    action_vec = p_stop_xyz - p_start_xyz
    action_xy = action_vec[:2]
    points_xy = s_cur_xyz[:, :2]

    dist = _point_to_segment_distance_xy(points_xy, p_start_xyz[:2], p_stop_xyz[:2])
    influence = torch.exp(-(dist * dist) / (2.0 * sigma_m * sigma_m))

    s_delta = torch.zeros_like(s_cur_xyz)
    s_delta[:, :2] = influence[:, None] * action_xy[None, :]
    return s_delta


class GenesisParticlePushDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        split: str,
        val_pct: int,
        test_pct: int,
        max_samples: int | None = None,
        action_sigma: float | None = None,
    ):
        self.entries: list[tuple[Path, Path]] = collect_run_paths(paths, split, val_pct, test_pct)
        self.runs: list[dict] = []
        self.configs: list[dict] = []
        self.index: list[tuple[int, int]] = []
        self.action_sigma = action_sigma

        for run_idx, (data_file, config_file) in enumerate(self.entries):
            run_data = torch.load(data_file, map_location="cpu")
            cfg = yaml.full_load(config_file.read_text())

            states = torch.as_tensor(run_data["states"], dtype=torch.float32)
            states_next = torch.as_tensor(run_data["states_"], dtype=torch.float32)
            if states.shape != states_next.shape:
                raise ValueError(f"Mismatched states/states_ in {data_file}")

            sample_count = states.shape[0]
            self.runs.append(run_data)
            self.configs.append(cfg)
            for sample_idx in range(sample_count):
                self.index.append((run_idx, sample_idx))

        if max_samples is not None:
            self.index = self.index[: max(0, int(max_samples))]

        if not self.index:
            raise ValueError(f"No samples found for split={split}.")

    def __len__(self):
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
            "target": s_next,
            "particle_dens": torch.tensor(density, dtype=torch.float32),
            "particle_num": torch.tensor(particle_num, dtype=torch.long),
        }


def particle_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
        target[b, :n] = item["target"]

    return {
        "a_cur": a_cur,
        "s_cur": s_cur,
        "s_delta": s_delta,
        "target": target,
        "particle_dens": particle_dens,
        "particle_nums": particle_nums,
    }


def build_model_config(args: argparse.Namespace) -> dict:
    return {
        "train": {
            "particle": {
                "nf_effect": int(args.nf_effect),
                "add_delta": bool(args.add_delta),
                "adj_thresh": float(args.adj_thresh),
            }
        }
    }


def masked_position_mse(pred: torch.Tensor, target: torch.Tensor, particle_nums: torch.Tensor) -> torch.Tensor:
    # pred/target: (B, N, 3), particle_nums: (B,)
    bsz, n_max, dim = pred.shape
    mask = torch.arange(n_max, device=pred.device)[None, :] < particle_nums[:, None]
    sq = (pred - target).pow(2).sum(dim=-1)

    denom = (mask.sum().clamp_min(1) * dim).float()
    return (sq * mask).sum() / denom


def masked_changed_mse(
    pred: torch.Tensor,
    s_cur: torch.Tensor,
    target: torch.Tensor,
    particle_nums: torch.Tensor,
    move_threshold: float = 1e-4,
) -> tuple[float, float, float, float]:
    n_max = pred.shape[1]
    valid_mask = torch.arange(n_max, device=pred.device)[None, :] < particle_nums[:, None]
    changed = ((target - s_cur).norm(dim=-1) > move_threshold) & valid_mask

    changed_count = changed.sum().item()
    if changed_count == 0:
        return 0.0, 0.0, 0.0, 0.0

    pred_sse = ((pred - target).pow(2).sum(dim=-1) * changed).sum().item()
    copy_sse = ((s_cur - target).pow(2).sum(dim=-1) * changed).sum().item()

    changed_mse = pred_sse / (changed_count * pred.shape[-1])
    changed_copy_mse = copy_sse / (changed_count * pred.shape[-1])
    changed_frac = changed_count / valid_mask.sum().item()
    return changed_mse, changed_copy_mse, float(changed_count), changed_frac


def evaluate(model: PropNetDiffDenModel, loader: DataLoader) -> dict[str, float]:
    model.eval()

    totals = {
        "loss": 0.0,
        "copy_mse": 0.0,
        "changed_mse": 0.0,
        "changed_copy_mse": 0.0,
        "changed_frac": 0.0,
        "n": 0,
    }

    with torch.no_grad():
        for batch in loader:
            a_cur = batch["a_cur"].to(DEVICE)
            s_cur = batch["s_cur"].to(DEVICE)
            s_delta = batch["s_delta"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            particle_dens = batch["particle_dens"].to(DEVICE)
            particle_nums = batch["particle_nums"].to(DEVICE)

            pred = model.predict_one_step(a_cur, s_cur, s_delta, particle_dens, particle_nums=particle_nums)

            loss = masked_position_mse(pred, target, particle_nums)
            copy_loss = masked_position_mse(s_cur, target, particle_nums)
            changed_mse, changed_copy_mse, _, changed_frac = masked_changed_mse(pred, s_cur, target, particle_nums)

            bsz = a_cur.size(0)
            totals["loss"] += loss.item() * bsz
            totals["copy_mse"] += copy_loss.item() * bsz
            totals["changed_mse"] += changed_mse * bsz
            totals["changed_copy_mse"] += changed_copy_mse * bsz
            totals["changed_frac"] += changed_frac * bsz
            totals["n"] += bsz

    n = max(1, totals["n"])
    return {
        "loss": totals["loss"] / n,
        "copy_mse": totals["copy_mse"] / n,
        "changed_mse": totals["changed_mse"] / n,
        "changed_copy_mse": totals["changed_copy_mse"] / n,
        "changed_frac": totals["changed_frac"] / n,
    }


def save_checkpoint(
    path: Path,
    model: PropNetDiffDenModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
    model_cfg: dict,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_config": model_cfg,
    }
    torch.save(payload, path)


def load_model_or_checkpoint(path: Path, model: PropNetDiffDenModel):
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt

    model.load_state_dict(ckpt)
    return None


def dataloader_for(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=particle_collate,
    )


def main():
    args = parse_args()

    log_dir = args.log_dir
    if not args.eval_only:
        log_dir = unique_log_dir(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = build_model_config(args)

    if not args.eval_only:
        run_config = {
            "script": "train_GNN_genesis.py",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "device": DEVICE,
            "log_dir": str(log_dir),
            "model": {
                "type": "PropNetDiffDenModel",
                "config": model_cfg,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lr_scheduler": f"StepLR(step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma})",
                "patience": args.patience,
                "save_every": args.save_every,
            },
            "data": {
                "folders": list(args.data_folders),
                "val_pct": args.val_pct,
                "test_pct": args.test_pct,
                "action_sigma": args.action_sigma,
                "split_strategy": "deterministic stratified over per-folder physics groups",
            },
            "features": {
                "target": "states_[..., :3]",
                "input_state": "states[..., :3]",
                "particle_attr": "constant 1 per particle",
                "s_delta": "distance-weighted action displacement from p_starts/p_stops",
                "particle_density": "material.density from run config",
            },
        }
        with open(log_dir / "run_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, sort_keys=False)

    print(f"Device: {DEVICE}")
    print(f"Log dir: {log_dir}")
    print(f"Model config: {model_cfg}")

    test_dataset = GenesisParticlePushDataset(
        args.data_folders,
        split="test",
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        max_samples=args.max_test_samples,
        action_sigma=args.action_sigma,
    )
    test_loader = dataloader_for(test_dataset, args, shuffle=False)

    model = PropNetDiffDenModel(model_cfg, use_gpu=(DEVICE == "cuda")).to(DEVICE)

    if args.eval_only:
        checkpoint = args.checkpoint if args.checkpoint is not None else log_dir / "gnn.pth"
        _ = load_model_or_checkpoint(checkpoint, model)
        print(f"Evaluating checkpoint: {checkpoint}")
        test_metrics = evaluate(model, test_loader)
        print(
            f"Test MSE: {test_metrics['loss']:.6f}, "
            f"Test Copy MSE: {test_metrics['copy_mse']:.6f}, "
            f"Test Changed MSE: {test_metrics['changed_mse']:.6f}, "
            f"Test Changed Copy MSE: {test_metrics['changed_copy_mse']:.6f}, "
            f"Test Changed Frac: {test_metrics['changed_frac']:.6f}"
        )
        return

    train_dataset = GenesisParticlePushDataset(
        args.data_folders,
        split="train",
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        max_samples=args.max_train_samples,
        action_sigma=args.action_sigma,
    )
    val_dataset = GenesisParticlePushDataset(
        args.data_folders,
        split="val",
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        max_samples=args.max_val_samples,
        action_sigma=args.action_sigma,
    )

    train_loader = dataloader_for(train_dataset, args, shuffle=True)
    val_loader = dataloader_for(val_dataset, args, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
    )

    writer = SummaryWriter(log_dir=log_dir)

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if not args.fresh_start:
        resume_checkpoint = args.resume_checkpoint or default_resume_checkpoint(log_dir)
        if resume_checkpoint is not None and resume_checkpoint.exists():
            loaded = load_model_or_checkpoint(resume_checkpoint, model)
            if loaded is not None:
                if "optimizer_state_dict" in loaded:
                    optimizer.load_state_dict(loaded["optimizer_state_dict"])
                if "scheduler_state_dict" in loaded:
                    scheduler.load_state_dict(loaded["scheduler_state_dict"])
                if args.start_epoch is not None:
                    start_epoch = int(args.start_epoch)
                else:
                    start_epoch = int(loaded.get("epoch", checkpoint_epoch(resume_checkpoint) or 0))
                best_val_loss = float(loaded.get("best_val_loss", best_val_loss))
            else:
                start_epoch = args.start_epoch if args.start_epoch is not None else (checkpoint_epoch(resume_checkpoint) or 0)

            print(f"Resuming from {resume_checkpoint} at epoch {start_epoch}. Training to epoch {args.epochs}.")
        else:
            print("No resume checkpoint found; starting from scratch.")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    with trange(start_epoch, args.epochs, desc="Training Epochs") as tbar:
        for epoch in tbar:
            model.train()
            train_loss_sum = 0.0
            train_copy_sum = 0.0
            train_n = 0

            for batch in train_loader:
                a_cur = batch["a_cur"].to(DEVICE)
                s_cur = batch["s_cur"].to(DEVICE)
                s_delta = batch["s_delta"].to(DEVICE)
                target = batch["target"].to(DEVICE)
                particle_dens = batch["particle_dens"].to(DEVICE)
                particle_nums = batch["particle_nums"].to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                pred = model.predict_one_step(a_cur, s_cur, s_delta, particle_dens, particle_nums=particle_nums)
                loss = masked_position_mse(pred, target, particle_nums)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    copy_loss = masked_position_mse(s_cur, target, particle_nums)

                bsz = a_cur.size(0)
                train_loss_sum += loss.item() * bsz
                train_copy_sum += copy_loss.item() * bsz
                train_n += bsz

            scheduler.step()

            train_loss = train_loss_sum / max(1, train_n)
            train_copy = train_copy_sum / max(1, train_n)

            val_metrics = evaluate(model, val_loader)
            val_loss = val_metrics["loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                save_checkpoint(log_dir / "gnn_best.pth", model, optimizer, scheduler, epoch + 1, best_val_loss, model_cfg)
            else:
                epochs_without_improvement += 1

            writer.add_scalar("Loss/TrainMSE", train_loss, epoch)
            writer.add_scalar("Loss/TrainCopyMSE", train_copy, epoch)
            writer.add_scalar("Loss/ValMSE", val_metrics["loss"], epoch)
            writer.add_scalar("Loss/ValCopyMSE", val_metrics["copy_mse"], epoch)
            writer.add_scalar("Loss/ValChangedMSE", val_metrics["changed_mse"], epoch)
            writer.add_scalar("Loss/ValChangedCopyMSE", val_metrics["changed_copy_mse"], epoch)
            writer.add_scalar("Metric/ValChangedFrac", val_metrics["changed_frac"], epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Convergence/BestEpoch", best_epoch, epoch)
            writer.add_scalar("Convergence/EpochsWithoutImprovement", epochs_without_improvement, epoch)

            tbar.set_postfix(
                {
                    "Train MSE": f"{train_loss:.5f}",
                    "Val MSE": f"{val_loss:.5f}",
                    "Val Copy": f"{val_metrics['copy_mse']:.5f}",
                    "Best": best_epoch,
                    "No Improv": epochs_without_improvement,
                }
            )

            print(
                f"Epoch {epoch + 1}: "
                f"train MSE={train_loss:.6f}, train copy MSE={train_copy:.6f}, "
                f"val MSE={val_metrics['loss']:.6f}, val copy MSE={val_metrics['copy_mse']:.6f}, "
                f"val changed MSE={val_metrics['changed_mse']:.6f}, "
                f"val changed copy MSE={val_metrics['changed_copy_mse']:.6f}, "
                f"val changed frac={val_metrics['changed_frac']:.6f}"
            )

            if (epoch + 1) % max(1, args.save_every) == 0:
                save_checkpoint(
                    log_dir / f"gnn_epoch_{epoch + 1}.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    best_val_loss,
                    model_cfg,
                )

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation improvement.")
                break

    writer.close()

    save_checkpoint(log_dir / "gnn.pth", model, optimizer, scheduler, epoch + 1, best_val_loss, model_cfg)

    print(
        f"\n=== Convergence summary ==="
        f"\n  Best val loss:   {best_val_loss:.6f} (epoch {best_epoch})"
        f"\n  Suggested budget:{best_epoch + args.patience} epochs (best + patience)"
    )

    test_metrics = evaluate(model, test_loader)
    print(
        f"Test MSE: {test_metrics['loss']:.6f}, "
        f"Test Copy MSE: {test_metrics['copy_mse']:.6f}, "
        f"Test Changed MSE: {test_metrics['changed_mse']:.6f}, "
        f"Test Changed Copy MSE: {test_metrics['changed_copy_mse']:.6f}, "
        f"Test Changed Frac: {test_metrics['changed_frac']:.6f}"
    )


if __name__ == "__main__":
    main()