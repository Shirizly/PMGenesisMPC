###########################################################################
# sysid_unet_cg.py
#
# System identification using a trained UNetFiLM.
#
# Given real experiment data (before-mask + action → after-mask), finds
# the physics vector p* = [friction, density, box_friction] that minimises
#
#     Σ_i || UNetFiLM(x_i, p) - y_i ||²
#
# via gradient descent through the frozen model.
#
# Modes:
#   global   — one shared p* for all provided data.
#   per_run  — a separate p* per .pt file; reports mean ± std at the end.
#
# Physics are parameterised as  p_j = lo_j + σ(θ_j)·(hi_j − lo_j)
# so θ ∈ ℝ³ is unconstrained. θ=0 initialises at the midpoint of each range.
###########################################################################

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import trange

from GranularDynamics2.myClasses.UNetModels_conditioned import UNetFiLM
from RealData.dataset import RealPileSweepData, SingleRunData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Physics bounds: [min, max] for [friction, density, box_friction]
# Matching the ranges used during simulation data generation.
PHYSICS_BOUNDS = torch.tensor(
    [
        [0.05,    0.50  ],   # friction
        [750.0, 5000.0  ],   # density  (kg/m³)
        [0.05,    0.50  ],   # box_friction
    ],
    dtype=torch.float32,
    device=DEVICE,
)

PHYSICS_NAMES = ["friction", "density", "box_friction"]


# ---------------------------------------------------------------------------
# Physics parameterisation helpers
# ---------------------------------------------------------------------------

def raw_to_physics(raw: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """
    Map unconstrained raw parameters θ → valid physics values via sigmoid.

        p_j = lo_j + σ(θ_j) · (hi_j − lo_j)

    Parameters
    ----------
    raw    : Tensor[3]   unconstrained parameters (requires_grad=True for sysid)
    bounds : Tensor[3,2] [[lo_0,hi_0], [lo_1,hi_1], [lo_2,hi_2]]

    Returns
    -------
    Tensor[3]  physics values in [lo_j, hi_j]
    """
    return bounds[:, 0] + torch.sigmoid(raw) * (bounds[:, 1] - bounds[:, 0])


def physics_to_raw(physics: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """
    Inverse map: known physics values → unconstrained θ (for warm-starting).

        θ_j = logit( (p_j − lo_j) / (hi_j − lo_j) )
    """
    p = (physics - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    p = p.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Core optimisation routine
# ---------------------------------------------------------------------------

def run_sysid(
    model: nn.Module,
    dataloader: DataLoader,
    n_epochs: int = 200,
    lr: float = 0.05,
    init_physics: torch.Tensor | None = None,
) -> dict:
    """
    Optimise a 3-element physics vector to minimise prediction MSE.

    The model must already be frozen (all param.requires_grad = False) and
    in eval mode before calling this function.

    Parameters
    ----------
    model       : frozen UNetFiLM
    dataloader  : DataLoader yielding ((input_grid, _), output_grid)
                  (the dataset-provided physics vector is ignored)
    n_epochs    : number of full passes over the dataloader
    lr          : Adam learning rate for the raw physics parameters
    init_physics: Tensor[3] optional warm-start in physics space;
                  None → initialise at midpoint of each range (raw = 0)

    Returns
    -------
    dict with keys:
        "physics"       : Tensor[3]  identified physics values (CPU)
        "loss_history"  : list[float] per-epoch mean MSE
    """
    criterion = nn.MSELoss()
    bounds = PHYSICS_BOUNDS.to(DEVICE)

    if init_physics is not None:
        raw = physics_to_raw(init_physics.to(DEVICE), bounds).detach().clone()
    else:
        # raw = 0  →  sigmoid(0) = 0.5  →  midpoint of every range
        raw = torch.zeros(3, device=DEVICE)
    raw = raw.requires_grad_(True)

    optimizer = torch.optim.Adam([raw], lr=lr)
    loss_history: list[float] = []

    with trange(n_epochs, desc="SysID epochs") as pbar:
        for _ in pbar:
            epoch_loss = 0.0
            n_samples  = 0

            for (inputs, _), outputs in dataloader:
                inputs  = inputs.to(DEVICE)
                outputs = outputs.to(DEVICE)

                # Expand the current physics estimate to match batch size
                physics = raw_to_physics(raw, bounds)
                physics_batch = physics.unsqueeze(0).expand(inputs.size(0), -1)

                pred = model(inputs, physics_batch)
                loss = criterion(pred.squeeze(1).float(), outputs)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                n_samples  += inputs.size(0)

            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)

            with torch.no_grad():
                current = raw_to_physics(raw, bounds)
            pbar.set_postfix(
                {"loss": f"{avg_loss:.5f}"}
                | {n: f"{v:.4f}" for n, v in zip(PHYSICS_NAMES, current.tolist())}
            )

    with torch.no_grad():
        identified = raw_to_physics(raw, bounds).cpu()

    return {"physics": identified, "loss_history": loss_history}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_run_file_pairs(data_root: Path, paths: list[str]) -> list[tuple[Path, Path]]:
    """Gather all (_data.pt, _config.yaml) pairs under the given paths."""
    pairs: list[tuple[Path, Path]] = []
    for path in paths:
        for df in sorted((data_root / path).rglob("*_data.pt")):
            cf = df.with_name(df.stem.replace("_data", "") + "_config.yaml")
            if cf.exists():
                pairs.append((df, cf))
    return pairs


def _save_results(results: dict, save_path: str | Path) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Results saved to {save_path}")


def _plot_loss(loss_history: list[float], out_path: str | Path, title: str = "SysID Loss") -> None:
    try:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=100)
        plt.close()
    except Exception as e:
        print(f"[warn] Could not save loss plot: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Configuration -------------------------------------------------------
    model_path  = "runs/unetfilm/unet.pth"  # path to trained UNetFiLM weights
    data_root   = "."
    data_paths  = ["real_data"]             # subdirectory(ies) under data_root

    mode        = "global"   # "global" | "per_run"
    n_epochs    = 200
    lr          = 0.05
    batch_size  = 32
    log_dir     = "runs/sysid"
    save_path   = os.path.join(log_dir, "identified_physics.yaml")

    # Optional warm-start: provide known / estimated physics to start near the
    # right region of the search space.  Set to None to start at mid-range.
    #   init_physics = torch.tensor([0.2, 2000.0, 0.15])
    init_physics: torch.Tensor | None = None

    # ---- Load model ----------------------------------------------------------
    model = UNetFiLM(physics_dim=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    os.makedirs(log_dir, exist_ok=True)

    # ==========================================================================
    # GLOBAL mode — single physics vector over all data
    # ==========================================================================
    if mode == "global":
        dataset = RealPileSweepData(data_root, data_paths)
        loader  = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        result  = run_sysid(model, loader, n_epochs=n_epochs, lr=lr, init_physics=init_physics)
        physics = result["physics"]

        print("\n=== Identified Physics (global) ===")
        out_dict: dict = {}
        for name, val in zip(PHYSICS_NAMES, physics.tolist()):
            print(f"  {name:15s}: {val:.4f}")
            out_dict[name] = float(val)

        _save_results(out_dict, save_path)
        _plot_loss(result["loss_history"], Path(log_dir) / "loss_curve.png")

    # ==========================================================================
    # PER-RUN mode — separate physics vector per .pt file
    # ==========================================================================
    elif mode == "per_run":
        data_root_path = Path(data_root)
        run_pairs = _collect_run_file_pairs(data_root_path, data_paths)

        if not run_pairs:
            raise FileNotFoundError(f"No run files found under {data_root}/{data_paths}")

        all_results: dict = {}

        for data_file, config_file in run_pairs:
            run_name = data_file.stem   # e.g. "_0_data"
            print(f"\n--- Run: {run_name} ---")

            run_ds = SingleRunData(data_file, config_file)
            # num_workers=0 avoids fork overhead for small single-run datasets
            loader = DataLoader(run_ds, batch_size=min(batch_size, len(run_ds)), shuffle=True, num_workers=0)

            result = run_sysid(model, loader, n_epochs=n_epochs, lr=lr, init_physics=init_physics)

            phys_dict = {n: float(v) for n, v in zip(PHYSICS_NAMES, result["physics"].tolist())}
            all_results[run_name] = {
                "physics":    phys_dict,
                "final_loss": float(result["loss_history"][-1]),
            }
            print(f"  physics    : {phys_dict}")
            print(f"  final loss : {result['loss_history'][-1]:.5f}")

            _plot_loss(
                result["loss_history"],
                Path(log_dir) / f"loss_{run_name}.png",
                title=f"SysID loss — {run_name}",
            )

        _save_results(all_results, save_path)

        # Summary statistics across runs
        print("\n=== Summary (per-run) ===")
        for name in PHYSICS_NAMES:
            vals = [all_results[r]["physics"][name] for r in all_results]
            print(f"  {name:15s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'global' or 'per_run'.")
