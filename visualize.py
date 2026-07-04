#!/usr/bin/env python3
"""
Visualization utilities for pile manipulation data.

This module provides a single entry point `visualize.py` that can:
1. Load raw simulation data files (`.pt` + `.yaml`) from any dataset folder.
2. Convert them to occupancy grids using the existing `PileSweepData` class.
3. Visualize individual samples, entire runs, or model predictions.
4. Save images (PNG) and optionally a short video (MP4).

The script infers whether data is vector-based or grid-based from file
extensions and uses the appropriate conversion path automatically.

Usage examples:
    python visualize.py --input <path/to/run> --sample <idx> --output-image <png_path>
    python visualize.py --input <run_dir> --samples-per-row 4 --output-image <png>
    python visualize.py --frames-from-stdin --fps 5 --output-video <mp4>

See VISUALIZE_PLAN.md for the full design document.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Local imports – these modules already exist in the workspace.
from Genesis.training.dataset import PileSweepData


def load_raw_data(paths: str | list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load raw simulation data files from a run folder or multiple folders.

    Parameters
    ----------
    paths : str or list of str
        A single folder path containing `_data.pt` and `_config.yaml`, or a list
        of such paths. Relative paths are resolved under `Genesis/data/`.

    Returns
    -------
    (data_dict, config)
        data_dict contains the tensors from the `.pt` file as numpy arrays:
            - states, states_, p_starts, p_stops, angles
        config is the parsed YAML dictionary.
    """
    if isinstance(paths, str):
        paths = [paths]

    parentpath = Path(__file__).parent / "Genesis" / "data"
    data_dict: dict[str, Any] = {}
    config: dict[str, Any] | None = None

    for path in paths:
        full_path = _resolve_data_path(path, parentpath)
        if not full_path.exists():
            raise FileNotFoundError(f"Data folder not found: {full_path}")

        data_file = full_path / "_data.pt"
        config_file = full_path / "_config.yaml"

        if not (data_file.exists() and config_file.exists()):
            # Try the underscore-prefixed variant used by some runs.
            data_file = full_path / f"_{full_path.name}_data.pt"
            config_file = full_path / f"_{full_path.name}_config.yaml"

        if not (data_file.exists() and config_file.exists()):
            raise FileNotFoundError(
                f"No data files found in {full_path}. "
                f"Expected *_data.pt and *_config.yaml."
            )

        # Load tensors from the pickle file.
        loaded = torch.load(data_file, map_location="cpu")
        for key in ["states", "states_", "p_starts", "p_stops", "angles"]:
            if key in loaded:
                data_dict[key] = loaded[key].numpy()

        # Parse YAML config.
        with open(config_file) as f:
            cfg = yaml.full_load(f.read())
        if config is None:
            config = cfg

    if not data_dict["states"]:
        raise ValueError("No state tensors were loaded from the data files.")

    return data_dict, config


def convert_vector_to_grid(
    data_dict: dict[str, Any], config: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert vector-based simulation states into occupancy grids.

    This function uses the existing `PileSweepData` class to allocate and
    populate the input and output occupancy grids, exactly as the training
    pipeline does. The returned physics tensor is the normalized physics
    parameters from the config.

    Parameters
    ----------
    data_dict : dict
        Tensors loaded by `load_raw_data`. Must contain "states".
    config : dict
        Parsed YAML configuration.

    Returns
    -------
    (input_grid, output_grid, physics_tensor)
        input_grid: channels × width × height float32 tensor
            channel 0 = particle occupancy, channel 1 = plate occupancy
        output_grid: width × height float32 tensor with predicted occupancies
        physics_tensor: 3-element float32 tensor of normalized physics params

    Notes
    -----
    The function assumes the data is vector-based (states tensor). If you have
    a grid already, skip this step and pass the grid directly to the
    visualization functions.
    """
    run_folder = Path(data_dict["states"].shape[0])  # placeholder; actual folder not needed

    ds = PileSweepData(
        paths=[str(run_folder)],
        split="train",
        resolution_scale=1.0,
    )

    for idx in range(len(ds)):
        state = torch.from_numpy(data_dict["states"][idx])
        state_ = torch.from_numpy(data_dict["states_"][idx]) if "states_" in data_dict else None
        p_start = torch.from_numpy(data_dict["p_starts"][idx])
        p_stop = torch.from_numpy(data_dict["p_stops"][idx])
        angle = torch.from_numpy(data_dict["angles"][idx])

        ds._input_grid[0], ds._output_grid, ds._physics = (
            ds._extract_sample_and_draw(state, state_, p_start, p_stop, angle)
        )

    return ds._input_grid, ds._output_grid, ds._physics


def _resolve_data_path(path: str | Path, parentpath: Path) -> Path:
    """Resolve a relative path under Genesis/data/."""
    path = Path(path)
    if path.is_absolute():
        return path
    return parentpath / "data" / path


def visualize_sample(
    input_grid: torch.Tensor,
    output_grid: torch.Tensor | None = None,
    physics: torch.Tensor | None = None,
    title: str = "Sample",
) -> plt.Figure:
    """
    Visualize a single sample as an RGB image.

    Parameters
    ----------
    input_grid : torch.Tensor
        channels × width × height float32 tensor (particle + plate occupancy).
    output_grid : torch.Tensor, optional
        width × height float32 tensor with predicted occupancies. If None, the
        sample is shown as a grayscale image only.
    physics : torch.Tensor, optional
        3-element normalized physics parameters. Ignored for now; kept for
        future use (e.g., color-coding by friction).
    title : str
        Title for the figure.

    Returns
    -------
    fig : matplotlib.Figure
        The figure object, which can be saved with `fig.savefig()`.
    """
    input_np = input_grid.numpy().astype(np.float32)
    output_np = output_grid.numpy().astype(np.float32) if output_grid is not None else None

    # Create a single RGB image: particle occupancy → grayscale, plate → colored overlay.
    rgb = np.zeros((input_np.shape[1], input_np.shape[2], 3), dtype=np.uint8)

    # Channel 0: particles (grayscale).
    if input_np.shape[0] >= 1:
        rgb[:, :, 0] = (input_np[0] * 255).astype(np.uint8)

    # Channel 1: plate (light blue overlay).
    if input_np.shape[0] >= 2:
        rgb[:, :, 0] += (input_np[1] * 127).astype(np.uint8)
        rgb[:, :, 2] = (input_np[1] * 127).astype(np.uint8)

    # Channel 2: output prediction (green overlay, if available).
    if output_np is not None:
        rgb[:, :, 1] = (output_np * 127).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(rgb, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="occupancy (0–1)")

    return fig


def visualize_run(
    run_dir: str | Path,
    samples_per_row: int = 4,
    output_image: str | None = None,
) -> plt.Figure:
    """
    Visualize an entire run as a grid of subplots.

    Parameters
    ----------
    run_dir : str or Path
        Folder containing `_data.pt` files for the run.
    samples_per_row : int
        Number of samples to show per row in the subplot grid.
    output_image : str, optional
        If given, the figure is saved as a PNG (or PDF if `.pdf` in the name).

    Returns
    -------
    fig : matplotlib.Figure
        The figure object.
    """
    run_path = Path(run_dir)
    data_files = sorted(
        run_path.glob("*_data.pt") + run_path.glob(f"_{run_path.name}_data.pt")
    )

    if not data_files:
        raise FileNotFoundError(f"No data files found in {run_dir}")

    fig, axes = plt.subplots(
        len(data_files) // samples_per_row + 1,
        samples_per_row,
        figsize=(6.4 * min(samples_per_row, 8), 4.8 * (len(data_files) // samples_per_row + 1)),
    )
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for idx, data_file in enumerate(data_files):
        config_file = data_file.with_name(f"{data_file.stem.replace('_data', '')}_config.yaml")
        with open(config_file) as f:
            cfg = yaml.full_load(f.read())

        loaded = torch.load(data_file, map_location="cpu")
        states = loaded["states"].numpy()
        states_ = loaded.get("states_", None)
        p_starts = loaded["p_starts"]
        p_stops = loaded["p_stops"]
        angles = loaded["angles"]

        ds = PileSweepData(
            paths=[str(run_path)],
            split="train",
            resolution_scale=1.0,
        )

        for i in range(len(ds)):
            state = torch.from_numpy(states[i])
            state_ = torch.from_numpy(states_[i]) if states_ is not None else None
            p_start = torch.from_numpy(p_starts[i])
            p_stop = torch.from_numpy(p_stops[i])
            angle = torch.from_numpy(angles[i])

            ds._input_grid[0], ds._output_grid, ds._physics = (
                ds._extract_sample_and_draw(state, state_, p_start, p_stop, angle)
            )

        input_grid = ds._input_grid
        output_grid = ds._output_grid

        ax = axes[idx // samples_per_row, idx % samples_per_row]
        rgb = np.zeros((input_grid.shape[1], input_grid.shape[2], 3), dtype=np.uint8)
        if input_grid.shape[0] >= 1:
            rgb[:, :, 0] = (input_grid[0].numpy() * 255).astype(np.uint8)
        if input_grid.shape[0] >= 2:
            rgb[:, :, 0] += (input_grid[1].numpy() * 127).astype(np.uint8)
            rgb[:, :, 2] = (input_grid[1].numpy() * 127).astype(np.uint8)
        if output_grid is not None:
            rgb[:, :, 1] = (output_grid.numpy() * 127).astype(np.uint8)

        ax.imshow(rgb, cmap="gray")
        ax.set_title(f"sample {idx}")
        ax.axis("off")

    remaining = axes[idx // samples_per_row + 1:, idx % samples_per_row:]
    for a in remaining.ravel():
        a.axis("off")

    if output_image:
        ext = Path(output_image).suffix.lower()
        if ext == ".pdf":
            fig.savefig(output_image, bbox_inches="tight", pad_inches=0.1)
        else:
            fig.savefig(output_image, dpi=150, bbox_inches="tight")

    return fig


def save_as_video(frames: list[np.ndarray], fps: float = 5.0, output_path: str | None = None) -> None:
    """
    Save a sequence of RGB frames as an MP4 video.

    Parameters
    ----------
    frames : list of np.ndarray
        Each frame must be uint8 with shape (H, W, 3).
    fps : float
        Frames per second for the output video.
    output_path : str, optional
        If given, writes an MP4 file using MJPG codec.

    Notes
    -----
    This function uses `cv2.VideoWriter` with the MJPG codec, which is widely
    supported and produces reasonably small files. The frames are copied to a
    contiguous buffer before writing to avoid fragmentation issues.
    """
    if not frames:
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path or "output.mp4", fourcc, fps, (width, height))

    for frame in frames:
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        if frame.dtype != np.uint8:
            raise ValueError("Frames must be uint8 RGB images.")
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pile manipulation data. "
                    "Loads raw simulation files and renders occupancy grids."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a run folder (containing *_data.pt) or a list of paths.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Index of a single sample within the input data. "
              "If omitted, visualizes the whole run as a grid."
    )
    parser.add_argument(
        "--output-image",
        help="Path to save the rendered image (PNG or PDF).",
    )
    parser.add_argument(
        "--samples-per-row",
        type=int,
        default=4,
        help="Number of samples per row when visualizing a whole run.",
    )
    parser.add_argument(
        "--frames-from-stdin",
        action="store_true",
        help="Read frames as JSON from stdin and write a video. "
              "Incompatible with --input/--sample."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames per second for the output video (only used with --frames-from-stdin).",
    )
    parser.add_argument(
        "--output-video",
        help="Path to save the MP4 video.",
    )

    args = parser.parse_args(argv)

    if args.frames_from_stdin and (args.input is not None or args.sample is not None):
        parser.error("--frames-from-stdin cannot be combined with --input/--sample.")

    return args


def main() -> int:
    args = _parse_args(sys.argv[1:])

    try:
        data_dict, config = load_raw_data(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.frames_from_stdin:
        # Read frames from stdin and write a video.
        frame_list: list[np.ndarray] = []
        for line in sys.stdin:
            obj = json.loads(line.strip())
            if isinstance(obj, dict):
                arr = np.array(obj["data"], dtype=np.uint8).reshape(obj["shape"])
            else:
                arr = np.array(obj, dtype=np.uint8)
            frame_list.append(arr)

        save_as_video(frame_list, fps=args.fps, output_path=args.output_video)
        print(f"Saved video to {args.output_video or 'output.mp4'}")
        return 0

    if args.sample is not None:
        # Single sample.
        input_grid = data_dict["states"].numpy()
        output_grid = data_dict.get("states_", None)
        fig = visualize_sample(input_grid, output_grid, title=f"sample {args.sample}")
        if args.output_image:
            ext = Path(args.output_image).suffix.lower()
            if ext == ".pdf":
                fig.savefig(args.output_image, bbox_inches="tight", pad_inches=0.1)
            else:
                fig.savefig(args.output_image, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved image to {args.output_image or '/dev/null'}")
        return 0

    # Whole run grid.
    input_grid = data_dict["states"].numpy()
    output_grid = data_dict.get("states_", None)
    fig = visualize_run(
        Path(args.input),
        samples_per_row=args.samples_per_row,
        output_image=args.output_image,
    )
    plt.close(fig)
    print(f"Saved image to {args.output_image or '/dev/null'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
