import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.gnn_dyn import PropNetDiffDenModel
from train_GNN_genesis import GenesisParticlePushDataset, masked_changed_mse, masked_position_mse


# Global settings (defaults formerly provided by parse_args)
CHECKPOINT = Path("runs_cubes/gnn_mse_smalldata_n10_2/gnn_best.pth")
DATA_FOLDERS = ["ignore/cube/n10"]
VAL_PCT = 10
TEST_PCT = 10
ACTION_SIGMA = None
RESOLUTION_SCALE = 1.0
NF_EFFECT = 150
ADJ_THRESH = 0.08
ADD_DELTA = False
MAX_SAMPLES = None
NUM_PLOTS = 8
START = 0
SAVE_DIR = Path("outputs/model_visualization/gnn_n10")
SHOW = False
CPU = False


def get_settings():
    return SimpleNamespace(
        checkpoint=CHECKPOINT,
        data_folders=DATA_FOLDERS,
        val_pct=VAL_PCT,
        test_pct=TEST_PCT,
        action_sigma=ACTION_SIGMA,
        resolution_scale=RESOLUTION_SCALE,
        nf_effect=NF_EFFECT,
        adj_thresh=ADJ_THRESH,
        add_delta=ADD_DELTA,
        max_samples=MAX_SAMPLES,
        num_plots=NUM_PLOTS,
        start=START,
        save_dir=SAVE_DIR,
        show=SHOW,
        cpu=CPU,
    )


def _grid_spec(config: dict, resolution_scale: float) -> tuple[float, int, int, torch.Tensor]:
    to_pxl = 1000.0 * float(resolution_scale)
    x_dim, y_dim, _ = config["box"]["vol"]
    x_pxl = max(1, int(round(x_dim * to_pxl)))
    y_pxl = max(1, int(round(y_dim * to_pxl)))
    ctr = torch.tensor((round(x_pxl / 2), round(y_pxl / 2), 0), dtype=torch.float32)
    return to_pxl, x_pxl, y_pxl, ctr


def _particle_diameters(config: dict, n_particles: int) -> list[float]:
    sampled = config.get("data_collection", {}).get("sampled", {}).get("particle_sizes")
    if sampled is not None and len(sampled) >= n_particles:
        return [float(sampled[i][0]) for i in range(n_particles)]
    fallback = float(config.get("material", {}).get("particle_size", 0.005))
    return [fallback] * n_particles


def particles_to_occ(states_xyz_m: torch.Tensor, config: dict, resolution_scale: float) -> torch.Tensor:
    """Render particle centers to an Eulerian occupancy grid using cv2 circles."""
    to_pxl, x_pxl, y_pxl, ctr = _grid_spec(config, resolution_scale)
    grid_np = np.zeros((x_pxl, y_pxl), dtype=np.float32)
    states_pxl = states_xyz_m * to_pxl + ctr[None, :]
    diameters = _particle_diameters(config, states_pxl.shape[0])

    for i in range(states_pxl.shape[0]):
        center_x = float(states_pxl[i, 0])
        center_y = float(states_pxl[i, 1])
        radius = max(1, int(round(diameters[i] * to_pxl * 0.5)))
        cv2.circle(
            grid_np,
            (int(round(center_x)), int(round(center_y))),
            radius,
            color=1,
            thickness=-1,
        )
    return torch.from_numpy(grid_np)


def draw_action_grid(p_start_m: torch.Tensor, p_stop_m: torch.Tensor, angle: torch.Tensor, config: dict, resolution_scale: float) -> torch.Tensor:
    """Render start/end tool poses into the action channel (0.5 start, 1.0 end)."""
    to_pxl, x_pxl, y_pxl, ctr = _grid_spec(config, resolution_scale)
    grid_np = np.zeros((x_pxl, y_pxl), dtype=np.float32)

    start_pos = p_start_m * to_pxl + ctr
    stop_pos = p_stop_m * to_pxl + ctr

    plate_dim_x, plate_dim_y, _ = config["plate"]["size"]
    plate_dim_x *= to_pxl
    plate_dim_y *= to_pxl
    angle_rad = float(angle)

    def draw_box(center, density):
        rotated_rect = (
            (int(center[0]), int(center[1])),
            (int(plate_dim_x), int(plate_dim_y)),
            int(angle_rad * 180 / np.pi),
        )
        box = cv2.boxPoints(rotated_rect)
        box = np.int32(box)
        cv2.fillPoly(grid_np, [box], float(density))

    draw_box(start_pos[:2], 0.5)
    draw_box(stop_pos[:2], 1.0)
    return torch.from_numpy(grid_np)

def plot_prediction(
    input_grid,
    action_grid,
    ground_truth,
    prediction,
    sample_idx,
    mse,
    copy_mse,
    save_dir,
    show,
):
    """Show the model input, target, prediction, and signed error for one sample."""
    error = prediction - ground_truth

    fig, axs = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(
        f"Test sample {sample_idx} | model MSE={mse:.6f} | copy-input MSE={copy_mse:.6f}"
    )

    images = [
        (input_grid, "Input particles", "gray", 0.0, 1.0),
        (action_grid, "Sweep action", "magma", 0.0, 1.0),
        (ground_truth, "Ground truth", "gray", 0.0, 1.0),
        (prediction, "Prediction", "viridis", None, None),
        (error, "Prediction - ground truth", "coolwarm", -1.0, 1.0),
        (prediction.clamp(0.0, 1.0), "Prediction clamped [0, 1]", "gray", 0.0, 1.0),
    ]

    for ax, (grid, title, cmap, vmin, vmax) in zip(axs.flat, images):
        im = ax.imshow(
            grid.detach().cpu().numpy(),
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"test_sample_{sample_idx:06d}.png"
        fig.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def load_model(checkpoint_path, device, args):
    model_cfg = {
        "train": {
            "particle": {
                "nf_effect": int(args.nf_effect),
                "add_delta": bool(args.add_delta),
                "adj_thresh": float(args.adj_thresh),
            }
        }
    }
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        model_cfg = checkpoint["model_config"]

    model = PropNetDiffDenModel(model_cfg, use_gpu=(device == "cuda")).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def inspect_test_set(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    dataset = GenesisParticlePushDataset(
        args.data_folders,
        split="test",
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        max_samples=args.max_samples,
        action_sigma=args.action_sigma,
    )

    model = load_model(args.checkpoint, device, args)

    total_model_loss = 0.0
    total_zero_loss = 0.0
    total_copy_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0

    # Lagrangian metrics (same family used in train/val pipeline)
    total_lagr_mse = 0.0
    total_lagr_copy_mse = 0.0
    total_lagr_changed_mse = 0.0
    total_lagr_changed_copy_mse = 0.0
    total_lagr_changed_frac = 0.0

    num_samples = 0
    plotted = 0

    with torch.no_grad():
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            run_idx, local_idx = dataset.index[sample_idx]
            config = dataset.configs[run_idx]
            run_data = dataset.runs[run_idx]

            a_cur = sample["a_cur"].unsqueeze(0).to(device)
            s_cur = sample["s_cur"].unsqueeze(0).to(device)
            s_delta = sample["s_delta"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            particle_dens = sample["particle_dens"].unsqueeze(0).to(device)
            particle_nums = sample["particle_num"].unsqueeze(0).to(device)

            pred_states = model.predict_one_step(
                a_cur,
                s_cur,
                s_delta,
                particle_dens,
                particle_nums=particle_nums,
            )[0].detach().cpu()

            # Match training loss space (Lagrangian positions).
            pred_states_b = pred_states.unsqueeze(0)
            s_cur_b = s_cur.detach().cpu()
            target_b = target.detach().cpu()
            particle_nums_b = particle_nums.detach().cpu()

            lagr_mse = masked_position_mse(pred_states_b, target_b, particle_nums_b)
            lagr_copy_mse = masked_position_mse(s_cur_b, target_b, particle_nums_b)
            lagr_changed_mse, lagr_changed_copy_mse, _, lagr_changed_frac = masked_changed_mse(
                pred_states_b,
                s_cur_b,
                target_b,
                particle_nums_b,
            )

            input_occ = particles_to_occ(sample["s_cur"], config, args.resolution_scale)
            gt_occ = particles_to_occ(sample["target"], config, args.resolution_scale)
            pred_occ = particles_to_occ(pred_states, config, args.resolution_scale)

            p_start = torch.as_tensor(run_data["p_starts"][local_idx], dtype=torch.float32)
            p_stop = torch.as_tensor(run_data["p_stops"][local_idx], dtype=torch.float32)
            angle = torch.as_tensor(run_data["angles"][local_idx], dtype=torch.float32)
            action_grid = draw_action_grid(p_start, p_stop, angle, config, args.resolution_scale)

            pred_mask = pred_occ > 0.5
            target_mask = gt_occ > 0.5
            intersection = (pred_mask & target_mask).sum().float()
            pred_area = pred_mask.sum().float()
            target_area = target_mask.sum().float()
            union = pred_area + target_area - intersection

            model_loss = F.mse_loss(pred_occ, gt_occ)
            zero_loss = F.mse_loss(torch.zeros_like(gt_occ), gt_occ)
            copy_loss = F.mse_loss(input_occ, gt_occ)
            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2.0 * intersection + 1e-6) / (pred_area + target_area + 1e-6)

            total_model_loss += model_loss.item()
            total_zero_loss += zero_loss.item()
            total_copy_loss += copy_loss.item()
            total_iou += iou.item()
            total_dice += dice.item()

            total_lagr_mse += lagr_mse.item()
            total_lagr_copy_mse += lagr_copy_mse.item()
            total_lagr_changed_mse += lagr_changed_mse
            total_lagr_changed_copy_mse += lagr_changed_copy_mse
            total_lagr_changed_frac += lagr_changed_frac

            num_samples += 1

            if sample_idx >= args.start and plotted < args.num_plots:
                plot_prediction(
                    input_occ,
                    action_grid,
                    gt_occ,
                    torch.where(pred_mask, pred_occ, torch.zeros_like(pred_occ)),
                    sample_idx,
                    model_loss.item(),
                    copy_loss.item(),
                    args.save_dir,
                    args.show,
                )
                plotted += 1

    print(f"Test samples: {num_samples}")
    print(f"Model MSE:      {total_model_loss / num_samples:.6f}")
    print(f"All-zero MSE:   {total_zero_loss / num_samples:.6f}")
    print(f"Copy input MSE: {total_copy_loss / num_samples:.6f}")
    print(f"Hard IoU @0.5:  {total_iou / num_samples:.4f}")
    print(f"Hard Dice @0.5: {total_dice / num_samples:.4f}")

    print("Lagrangian (train-style) metrics:")
    print(f"  Position MSE:            {total_lagr_mse / num_samples:.6f}")
    print(f"  Copy Position MSE:       {total_lagr_copy_mse / num_samples:.6f}")
    print(f"  Changed Position MSE:    {total_lagr_changed_mse / num_samples:.6f}")
    print(f"  Changed Copy Pos MSE:    {total_lagr_changed_copy_mse / num_samples:.6f}")
    print(f"  Changed Particle Frac:   {total_lagr_changed_frac / num_samples:.6f}")

    if args.save_dir is not None and plotted > 0:
        print(f"Saved plots to: {args.save_dir}")


if __name__ == "__main__":
    inspect_test_set(get_settings())
