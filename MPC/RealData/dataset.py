"""
RealData/dataset.py

PyTorch Dataset for real-world pile manipulation experiments.
Mirrors the interface of Genesis/training/dataset.py (PileSweepData) so that
training and rollout scripts can swap datasets without any other changes.

Expected on-disk format per run:
    _{id}_data.pt      — torch.save'd dict (see keys below)
    _{id}_config.yaml  — grid / tool / physics metadata

data.pt keys:
    masks_before  (N, H, W)  float32  binary occupancy before action  [0,1]
    masks_after   (N, H, W)  float32  binary occupancy after  action  [0,1]
    p_starts_px   (N, 2)     float32  tool centre at start  (x_col, y_row) px
    p_stops_px    (N, 2)     float32  tool centre at end    (x_col, y_row) px
    angles        (N,)       float32  tool yaw in radians

config.yaml (minimal required fields):
    grid:
      height: 128
      width:  128
    tool:
      size_px: [40, 2]     # [width_px, height_px]
    physics:               # null entries fall back to default_physics
      friction:     null
      density:      null
      box_friction: null

__getitem__ returns:
    ((input_grid: Tensor[2, H, W], physics: Tensor[3]), output_grid: Tensor[H, W])

which is identical to PileSweepData's output format.
"""

import cv2
import hashlib
import math

import numpy as np
import torch
import yaml
from pathlib import Path
from torch.utils.data import Dataset


class RealPileSweepData(Dataset):
    """
    Dataset for real-world pile-manipulation experiments.

    Returns the same ((input_grid, physics), output_grid) format as the
    simulation-based PileSweepData so that training scripts are interchangeable.

    Parameters
    ----------
    data_root : str | Path
        Root directory; all ``paths`` are resolved relative to this.
    paths : list[str] | str
        Subdirectory path(s) under ``data_root`` to search for run files.
    run : int | None
        If given, load only that specific run ID.
    split : "train" | "val" | "test" | None
        Deterministic split by hashing the run file path. None = all data.
    val_pct : int
        Percentage of runs assigned to validation (default 10).
    test_pct : int
        Percentage of runs assigned to test (default 10).
    default_physics : list[float] | None
        Fallback [friction, density, box_friction] used when config values are
        null / missing.  Defaults to [0.0, 0.0, 0.0].
    """

    def __init__(
        self,
        data_root: str | Path,
        paths: list[str] | str,
        run: int | None = None,
        split: str | None = None,
        val_pct: int = 10,
        test_pct: int = 10,
        default_physics: list[float] | None = None,
    ):
        assert split in (None, "train", "val", "test"), f"Invalid split: {split!r}"
        self.data_root = Path(data_root)
        self._default_physics = default_physics if default_physics is not None else [0.0, 0.0, 0.0]

        self.runs: list[dict] = []
        self.configs: list[dict] = []
        self._run_lengths: list[int] = []

        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            full_path = self.data_root / path
            if not full_path.exists():
                raise FileNotFoundError(f"Data folder not found: {full_path}")

            run_files = self._collect_run_paths(full_path, run)
            if not run_files:
                raise FileNotFoundError(f"No run files found in: {full_path}")

            if split is not None:
                run_files = [
                    (df, cf) for df, cf in run_files
                    if self._assign_split(df, val_pct, test_pct) == split
                ]

            for data_file, config_file in run_files:
                run_data = torch.load(data_file, map_location="cpu", weights_only=True)
                self.runs.append(run_data)
                self.configs.append(yaml.full_load(config_file.read_text()))
                self._run_lengths.append(int(run_data["masks_before"].shape[0]))

        if not self.runs:
            raise ValueError("No data samples found for dataset.")

        # Build flat index lookup table (same pattern as PileSweepData)
        self._run_lookup: list[int] = []
        self._offsets: list[int] = [0]
        for r, n in enumerate(self._run_lengths):
            self._run_lookup.extend([r] * n)
            self._offsets.append(self._offsets[-1] + n)

        # Pre-allocate reusable grid tensors (safe with num_workers > 0 because
        # each worker process gets its own copy after fork)
        cfg0 = self.configs[0]
        self.H: int = cfg0["grid"]["height"]
        self.W: int = cfg0["grid"]["width"]
        self._input_grid = torch.zeros((2, self.H, self.W), dtype=torch.float32)
        self._output_grid = torch.zeros((self.H, self.W), dtype=torch.float32)

    # ------------------------------------------------------------------
    # Index / path helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._run_lookup)

    def _collect_run_paths(self, root: Path, run: int | None) -> list[tuple[Path, Path]]:
        if run is not None:
            for prefix in (f"_{run}", str(run)):
                d = root / f"{prefix}_data.pt"
                c = root / f"{prefix}_config.yaml"
                if d.exists() and c.exists():
                    return [(d, c)]
            return []

        pairs: list[tuple[Path, Path]] = []
        for data_file in sorted(root.rglob("*_data.pt")):
            config_file = data_file.with_name(
                data_file.stem.replace("_data", "") + "_config.yaml"
            )
            if config_file.exists():
                pairs.append((data_file, config_file))
        return pairs

    @staticmethod
    def _assign_split(data_file: Path, val_pct: int, test_pct: int) -> str:
        """Deterministic split via MD5 hash of the run file path."""
        h = int(hashlib.md5(str(data_file).encode()).hexdigest(), 16) % 100
        if h < test_pct:
            return "test"
        elif h < test_pct + val_pct:
            return "val"
        return "train"

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_tool_channel(
        self,
        p_start_px: torch.Tensor,
        p_stop_px: torch.Tensor,
        angle: torch.Tensor,
        tool_size_px: list,
        grid: torch.Tensor,
    ) -> None:
        """
        Draw tool start (value 0.5) and end (value 1.0) onto the grid.

        Mirrors _draw_plate_cv2 in PileSweepData: same rotated-rectangle
        drawing, same value encoding, same cv2 in-place modification.

        Parameters
        ----------
        p_start_px : Tensor[2]  (x_col, y_row) in pixels
        p_stop_px  : Tensor[2]  (x_col, y_row) in pixels
        angle      : Tensor scalar, yaw in radians
        tool_size_px : [width_px, height_px]
        grid       : Tensor[H, W] — modified in-place via shared numpy memory
        """
        grid_np = grid.numpy()  # shares memory — cv2 writes directly into the tensor
        tw = float(tool_size_px[0])
        th = float(tool_size_px[1])
        angle_deg = float(angle) * 180.0 / math.pi

        def draw_rect(cx: float, cy: float, val: float) -> None:
            rect = ((int(cx), int(cy)), (int(tw), int(th)), angle_deg)
            box = cv2.boxPoints(rect)
            cv2.fillPoly(grid_np, [np.int32(box)], val)

        draw_rect(float(p_start_px[0]), float(p_start_px[1]), 0.5)
        draw_rect(float(p_stop_px[0]),  float(p_stop_px[1]),  1.0)

    def _get_physics(self, config: dict) -> torch.Tensor:
        """Return [friction, density, box_friction]; fall back to defaults for null values."""
        phys = config.get("physics") or {}
        f  = phys.get("friction")     if phys.get("friction")     is not None else self._default_physics[0]
        d  = phys.get("density")      if phys.get("density")      is not None else self._default_physics[1]
        bf = phys.get("box_friction") if phys.get("box_friction") is not None else self._default_physics[2]
        return torch.tensor([float(f), float(d), float(bf)], dtype=torch.float32)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        self._input_grid.zero_()
        self._output_grid.zero_()

        run_idx    = self._run_lookup[idx]
        run        = self.runs[run_idx]
        config     = self.configs[run_idx]
        sample_idx = idx - self._offsets[run_idx]

        # Channel 0: before occupancy mask (already at target resolution)
        self._input_grid[0].copy_(run["masks_before"][sample_idx])

        # Channel 1: tool action channel
        self._render_tool_channel(
            p_start_px   = run["p_starts_px"][sample_idx],
            p_stop_px    = run["p_stops_px"][sample_idx],
            angle        = run["angles"][sample_idx],
            tool_size_px = config["tool"]["size_px"],
            grid         = self._input_grid[1],
        )

        # Output label: after occupancy mask
        self._output_grid.copy_(run["masks_after"][sample_idx])

        physics = self._get_physics(config)
        return (self._input_grid.clone(), physics), self._output_grid.clone()

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def plot_input_and_output(
        self,
        input_grid: torch.Tensor,
        label_grid: torch.Tensor,
        title: str = "",
    ) -> None:
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(title)
        for ax, data, t, cmap in zip(
            axes,
            [input_grid[0], input_grid[1], label_grid],
            ["Before mask (ch 0)", "Tool channel (ch 1)", "After mask (label)"],
            ["Blues", "Greens", "Blues"],
        ):
            ax.imshow(data.numpy(), cmap=cmap, origin="lower", vmin=0, vmax=1)
            ax.set_title(t)
            ax.axis("off")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Convenience: create a minimal single-run dataset directly from file paths.
# Used by sysid_unet_cg.py for per-run optimisation without needing a
# directory structure that matches the full dataset conventions.
# ---------------------------------------------------------------------------

class SingleRunData(Dataset):
    """
    Lightweight dataset wrapping one (_data.pt, _config.yaml) pair.
    Returns the same ((input_grid, physics), output_grid) format.
    """

    def __init__(self, data_file: str | Path, config_file: str | Path):
        self.run    = torch.load(Path(data_file),   map_location="cpu", weights_only=True)
        self.config = yaml.full_load(Path(config_file).read_text())
        self.H      = self.config["grid"]["height"]
        self.W      = self.config["grid"]["width"]
        self.n      = int(self.run["masks_before"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        input_grid = torch.zeros((2, self.H, self.W), dtype=torch.float32)

        input_grid[0] = self.run["masks_before"][idx]

        grid_np   = input_grid[1].numpy()
        p_start   = self.run["p_starts_px"][idx]
        p_stop    = self.run["p_stops_px"][idx]
        angle_deg = float(self.run["angles"][idx]) * 180.0 / math.pi
        tw, th    = self.config["tool"]["size_px"]

        for (cx, cy), val in [
            (p_start, 0.5),
            (p_stop,  1.0),
        ]:
            rect = ((int(float(cx)), int(float(cy))), (int(tw), int(th)), angle_deg)
            box  = cv2.boxPoints(rect)
            cv2.fillPoly(grid_np, [np.int32(box)], val)

        output_grid = self.run["masks_after"][idx].clone()

        # Physics vector is a placeholder here — sysid replaces it with the
        # optimised parameter at runtime.
        physics = torch.zeros(3, dtype=torch.float32)
        return (input_grid, physics), output_grid


if __name__ == "__main__":
    import sys
    # Quick sanity check: python -m RealData.dataset <data_root> <path>
    root  = sys.argv[1] if len(sys.argv) > 1 else "."
    path  = sys.argv[2] if len(sys.argv) > 2 else "real_data"
    ds    = RealPileSweepData(root, path)
    print(f"Dataset length: {len(ds)}")
    (inp, phys), out = ds[0]
    print(f"  input_grid : {inp.shape}  {inp.dtype}  range [{inp.min():.2f}, {inp.max():.2f}]")
    print(f"  physics    : {phys}")
    print(f"  output_grid: {out.shape}  {out.dtype}  range [{out.min():.2f}, {out.max():.2f}]")
    ds.plot_input_and_output(inp, out, title="sample 0")
