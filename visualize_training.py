import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLM


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect NFDUNetFiLM predictions on the Genesis test split.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs_cubes/nfu_mse_mass2_addremove/unet_best.pth"), help="Path to the trained NFDUNetFiLM state dict.",)

    # +++ MODEL SETTINGS +++
    parser.add_argument("--data-folders", nargs="+", default=["corl/cube"])
    parser.add_argument("--model-variant", choices=["full", "shallow", "lowres", "shallow-lowres"], default="full", help=(
        "full: UNet of depth 3 with 128x128 grid input;"
        "shallow: UNet of depth 2 with 128x128 grid input;"
        "lowres: UNet of depth 3 with 32x32 grid input;"
        "shallow-lowres: UNet of depth 2 with 32x32 grid input;"
        ),
    )
    parser.add_argument("--input-mode", choices=["standard", "sweep-removed-input", "sweep-removed-residual"], default="standard", help=(
            "standard: 2 channels, residual to current occupancy; "
            "sweep-removed-input: 3 channels, residual to current occupancy; "
            "sweep-removed-residual: 3 channels, residual to sweep-removed occupancy."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64, help="To adjust if custom test set")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-plots", type=int, default=8)
    parser.add_argument("--start", type=int, default=0, help="First test-set sample index to plot.")
    parser.add_argument("--save-dir", type=Path, default="", help="Directory for saved inspection PNGs. Use --save-dir '' to disable saving.",)
    parser.add_argument("--show", action="store_true", help="Also call plt.show() for interactive backends.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args()
    
    if args.save_dir == Path(""):
        args.save_dir = None
    elif args.save_dir == "":
        args.save_dir = Path(args.checkpoint.parent) / "plots"
    return args

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


def load_model(checkpoint_path, device):
    model = NFDUNetFiLM().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def inspect_test_set(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    resolution_scale = 0.25 if args.model_variant in ("lowres", "shallow-lowres") else 1.0 # low-res models have 0.25 resolution of the original size
    include_sweep_removed = args.input_mode in ("sweep-removed-input", "sweep-removed-residual")
    
    dataset = PileSweepData(
        args.data_folders,
        split="test",
        resolution_scale=resolution_scale,
        include_sweep_removed=include_sweep_removed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = load_model(args.checkpoint, device)

    total_model_loss = 0.0
    total_zero_loss = 0.0
    total_copy_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0
    plotted = 0

    with torch.no_grad():
        for inputs_, outputs in loader:
            inputs, physics = inputs_
            inputs = inputs.to(device)
            physics = physics.to(device)
            outputs = outputs.to(device)

            logits = model(inputs, physics).squeeze(1).float()
            predictions = torch.sigmoid(logits)
            pred_mask = predictions > 0.5
            target_mask = outputs > 0.5
            intersection = (pred_mask & target_mask).sum(dim=(1, 2)).float()
            pred_area = pred_mask.sum(dim=(1, 2)).float()
            target_area = target_mask.sum(dim=(1, 2)).float()
            union = pred_area + target_area - intersection
            batch_model_loss = F.mse_loss(predictions, outputs, reduction="none").mean(dim=(1, 2))
            batch_zero_loss = F.mse_loss(torch.zeros_like(outputs), outputs, reduction="none").mean(dim=(1, 2))
            batch_copy_loss = F.mse_loss(inputs[:, 0], outputs, reduction="none").mean(dim=(1, 2))
            batch_iou = (intersection + 1e-6) / (union + 1e-6)
            batch_dice = (2.0 * intersection + 1e-6) / (pred_area + target_area + 1e-6)

            batch_size = inputs.size(0)
            total_model_loss += batch_model_loss.sum().item()
            total_zero_loss += batch_zero_loss.sum().item()
            total_copy_loss += batch_copy_loss.sum().item()
            total_iou += batch_iou.sum().item()
            total_dice += batch_dice.sum().item()

            for i in range(batch_size):
                sample_idx = num_samples + i
                if sample_idx < args.start:
                    continue
                if plotted >= args.num_plots:
                    continue

                plot_prediction(
                    inputs[i, 0],
                    inputs[i, 1],
                    outputs[i],
                    torch.where(pred_mask[i], predictions[i], torch.zeros_like(predictions[i])),
                    sample_idx,
                    batch_model_loss[i].item(),
                    batch_copy_loss[i].item(),
                    args.save_dir,
                    args.show,
                )
                plotted += 1

            num_samples += batch_size

    print(f"Test samples: {num_samples}")
    print(f"Model MSE:      {total_model_loss / num_samples:.6f}")
    print(f"All-zero MSE:   {total_zero_loss / num_samples:.6f}")
    print(f"Copy input MSE: {total_copy_loss / num_samples:.6f}")
    print(f"Hard IoU @0.5:  {total_iou / num_samples:.4f}")
    print(f"Hard Dice @0.5: {total_dice / num_samples:.4f}")
    if args.save_dir is not None and plotted > 0:
        print(f"Saved plots to: {args.save_dir}")


if __name__ == "__main__":
    inspect_test_set(parse_args())
