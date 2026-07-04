from torch.utils.data import Dataset
from pathlib import Path
import hashlib
import yaml
import torch
from physics.normalization import PhysicsBounds
import torch.nn.functional as F
import os
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from transforms.functional import draw_plate_soft

TO_PXL = 1e3


class PileSweepData(Dataset):

    def __init__(
            self,
            paths: list[str] | str,
            split: str,
            val_pct: int = 5,
            test_pct: int = 5,
            resolution_scale: float = 1.0,
            physics_bounds: PhysicsBounds | None = None,
        ):
        """
        Initialize dataset with either a folder containing data or a specific run.

            @param paths: list of folder paths or a single folder path containing data files.
                Relative paths are resolved under Genesis/data; absolute paths are used as-is.
            @param split: one of "train", "val", "test".
                Splits are deterministic and stratified per leaf data folder, so each
                physical geometry group contributes to train/val/test when possible.
                Whole runs with the same nominal physics params stay in the same split.
            @param val_pct: percentage of physics groups assigned to validation
            @param test_pct: percentage of physics groups assigned to test
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split!r}"
        if val_pct < 0 or test_pct < 0 or val_pct + test_pct >= 100:
            raise ValueError("val_pct and test_pct must be non-negative and sum to less than 100.")
        if resolution_scale <= 0:
            raise ValueError("resolution_scale must be positive.")
        self.runs = []
        self.configs = []
        self._run_lengths = []
        self._plate_cache = {}
        self._physics = torch.zeros((3,), dtype=torch.float32)
        self._physics_bounds = physics_bounds or PhysicsBounds.default()
        self.resolution_scale = float(resolution_scale)
        self.to_pxl = TO_PXL * self.resolution_scale


        parentpath = Path(__file__).parent.parent
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            full_path = self._resolve_data_path(path, parentpath)
            if not full_path.exists():
                raise FileNotFoundError(f"Data folder not found: {full_path}")

            run_files = self._collect_run_paths(full_path)
            if not run_files:
                raise FileNotFoundError(
                    f"No data runs found in path: {full_path}"
                )

            run_files = self._filter_split(run_files, split, val_pct, test_pct)

            for data_file, config_file in run_files:
                self.runs.append(torch.load(data_file, map_location="cpu"))
                self.configs.append(yaml.full_load(config_file.read_text()))
                self._run_lengths.append(self._count_samples_in_run(self.runs[-1]))
        
        if not self.configs:
            raise ValueError("No configs found for dataset.")

        if not self.runs:    
            raise ValueError("No data samples found for dataset.")

        # create lookup table
        self._run_lookup = []
        self._offsets = [0]
        for r, num_samples in enumerate(self._run_lengths):
            self._run_lookup.extend([r] * num_samples)
            self._offsets.append(self._offsets[-1] + num_samples)

        self._create_grids(self.configs[0])
        
    def __len__(self):
        return len(self._run_lookup)

    def _create_grids(
            self, 
            config
        ) -> None:
        """
        Allocates tensors for all occupancy grids.

            @param config: A config to initialize grids.
        """

        # Box grid, dimension, and center
        x_dim, y_dim, _ = config["box"]["vol"]
        x_pxl = max(1, int(round(x_dim * self.to_pxl)))
        y_pxl = max(1, int(round(y_dim * self.to_pxl)))
        self.ctr_in_PXL = torch.tensor((round(x_pxl / 2), round(y_pxl / 2), 0))

        self._input_grid = torch.zeros((2, x_pxl, y_pxl), dtype=torch.float32)
        self._output_grid = torch.zeros((x_pxl, y_pxl), dtype=torch.float32)

        # Plate grid and dimension
        x_dim_plt, y_dim_plt, _ = self._get_plate_dims(config)
        self._plt_dim_in_m = (x_dim_plt, y_dim_plt)

    def _count_samples_in_run(self, run):
        return run["states"].shape[0]

    @staticmethod
    def _resolve_data_path(path: str | Path, parentpath: Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return parentpath / "data" / path

    def _collect_run_paths(self, root: Path):
        run_paths = []
        for data_file in sorted(root.rglob("*_data.pt")):
            config_file = data_file.with_name(
                f"{data_file.stem.replace('_data', '')}_config.yaml"
            )
            if config_file.exists():
                run_paths.append((data_file, config_file))

        return run_paths

    @classmethod
    def _filter_split(
        cls,
        run_files: list[tuple[Path, Path]],
        split: str,
        val_pct: int,
        test_pct: int,
    ) -> list[tuple[Path, Path]]:
        split_by_file = {}
        folder_groups: dict[Path, dict[str, list[tuple[Path, Path]]]] = {}

        for data_file, config_file in run_files:
            cfg = yaml.full_load(config_file.read_text())
            physics_key = cls._physics_key(cfg)
            folder_groups.setdefault(data_file.parent, {}).setdefault(
                physics_key, []
            ).append((data_file, config_file))

        for physics_groups in folder_groups.values():
            groups = sorted(
                physics_groups.items(),
                key=lambda item: hashlib.md5(item[0].encode()).hexdigest(),
            )
            assignments = cls._assign_group_splits(len(groups), val_pct, test_pct)
            for (_, group), assigned_split in zip(groups, assignments):
                for data_file, _ in group:
                    split_by_file[data_file] = assigned_split

        return [
            (data_file, config_file)
            for data_file, config_file in run_files
            if split_by_file[data_file] == split
        ]

    @staticmethod
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

        return (
            ["test"] * test_count
            + ["val"] * val_count
            + ["train"] * (num_groups - test_count - val_count)
        )

    @staticmethod
    def _physics_key(cfg: dict) -> str:
        """Nominal physics identity used to keep equivalent runs in one split."""
        return "%s|%d|%.6f|%.6f|%.6f|%.6f" % (
            cfg["material"]["shape"],
            cfg["material"]["n_particles"],
            cfg["material"]["particle_size"],
            cfg["material"]["friction"],
            cfg["material"]["density"],
            cfg["box"]["friction"],
        )

    def _get_plate_dims(self, config):
        return config["plate"]["size"]


    def _extract_sample_in_pxl(self, run, index):
        """
        Returns the sample at given index. Positions are converted from meters to pixels.

            @param run: data run.
            @param index: index of sample in run
        """
        particles = run["states"][index].clone()
        particles_ = run["states_"][index].clone()

        particles[:, :3] = particles[:, :3] * self.to_pxl + self.ctr_in_PXL
        particles_[:, :3] = particles_[:, :3] * self.to_pxl + self.ctr_in_PXL
        plate_pos = run["p_starts"][index] * self.to_pxl + self.ctr_in_PXL
        plate_pos_ = run["p_stops"][index] * self.to_pxl + self.ctr_in_PXL
        angle = run["angles"][index]

        return particles, particles_, plate_pos, plate_pos_, angle

    def _draw_particle_grid(self, particle_states, grid, config):

        num_particles = config["material"]["n_particles"]
        shape = config["material"]["shape"]
        particle_sizes = config["data_collection"]["sampled"]["particle_sizes"]
        grid_np = grid.numpy()

        def draw_box_points(grid, center, box_dim, angle, density=1):
            rotated_rect = (
                (int(center[0]), int(center[1])),
                (int(box_dim[0]), int(box_dim[1])), 
                int(angle * 180 / math.pi)
            )
            
            box = cv2.boxPoints(rotated_rect)
            box = np.int32(box)
            cv2.fillPoly(grid, [box], density)
        
        def quaternion_to_yaw(q):
            w, x, y, z = q
            siny_cosp = 2 * (w * z + x * y)
            cos_y_cosp = 1 - 2 * (y * y + z * z)
            return math.atan2(siny_cosp, cos_y_cosp)
        

        for idx in range(num_particles):
            particle_state = particle_states[idx]
            dimensions = particle_sizes[idx]
            center_x = float(particle_state[0])
            center_y = float(particle_state[1])
            upright_cylinder = False

            if shape == "cylinder":
                def cylinder_is_standing(quat):
                    local_up = np.array([0, 0, 1])
                    world_up = np.array([0, 0, 1])

                    rot = R.from_quat(quat)
                    
                    rotated_axis = rot.apply(local_up)
                
                    alignment = abs(np.dot(rotated_axis, world_up))

                    return alignment >= 0.5
                
                if cylinder_is_standing(particle_state[3:].numpy()):
                    upright_cylinder = True

            if shape == "sphere" or (upright_cylinder and shape=="cylinder"):
                diameter, _, _ = dimensions
                cv2.circle(
                    grid_np,
                    (int(round(center_x)), int(round(center_y))),
                    max(1, int(round(diameter * self.to_pxl * 0.5))),
                    color=1,
                    thickness=-1,
                )
                continue
            
            draw_box_points(
                grid_np,
                (center_x, center_y),
                (float(dimensions[0]) * self.to_pxl, float(dimensions[1]) * self.to_pxl),
                quaternion_to_yaw(particle_state[3:]),
                1
            )
    
    def _draw_plate(self, start_pos, end_pos, angle, config):
        plate_dim_x, plate_dim_y, _ = self._get_plate_dims(config)
        plate_dim_x *= self.to_pxl
        plate_dim_y *= self.to_pxl
        sigma = max(0.5, 1.5 * self.resolution_scale)
        angle_tensor = torch.as_tensor([angle], dtype=torch.float32)
        start_center = start_pos[:2].to(torch.float32).unsqueeze(0)
        end_center = end_pos[:2].to(torch.float32).unsqueeze(0)
        grid_size = (self._input_grid.shape[1], self._input_grid.shape[2])

        occ1 = draw_plate_soft(
            start_center,
            angle_tensor,
            grid_size,
            plate_dim_x,
            plate_dim_y,
            intensity=0.5,
            sigma=sigma,
        )[0]
        occ2 = draw_plate_soft(
            end_center,
            angle_tensor,
            grid_size,
            plate_dim_x,
            plate_dim_y,
            intensity=1.0,
            sigma=sigma,
        )[0]
        self._input_grid[1] = 1 - (1 - occ1) * (1 - occ2)
        
    def plot_grid(self, grid: torch.Tensor, title: str = "", ontop: bool = False) -> None:
        """Visualize the grid as an image"""
        
        from matplotlib import pyplot as plt

        plt.imshow(grid, interpolation="nearest", origin="lower")
        plt.title(title)
        plt.show()

    def plot_input_and_output(
            self,
            input_grid: torch.Tensor,
            label_grid: torch.Tensor,
            title: str = "",
            save_path: str | Path | None = None,
        ) -> Path:

        from matplotlib import pyplot as plt

        if save_path is None:
            filename = "input_output_plot.png"
            if title:
                safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title.strip())
                filename = f"{safe_title or 'input_output_plot'}.png"
            save_path = Path(__file__).with_name(filename)
        else:
            save_path = Path(save_path)

        channel_names = ["Current occupancy", "Action projection"]
        if input_grid.shape[0] > 2:
            channel_names.append("Sweep-removed occupancy")
        channel_names.extend(
            f"Input channel {idx}"
            for idx in range(len(channel_names), input_grid.shape[0])
        )

        n_cols = input_grid.shape[0] + 2
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        if title:
            fig.suptitle(title)

        for idx, name in enumerate(channel_names):
            axes[idx].imshow(
                input_grid[idx],
                cmap="gray",
                origin="lower",
                vmin=0,
                vmax=1,
            )
            axes[idx].set_title(name)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

        axes[-2].imshow(
            input_grid[0],
            cmap="Reds",
            alpha=0.5,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[-2].imshow(
            label_grid,
            cmap="Blues",
            alpha=0.5,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[-2].set_title("Current + target")
        axes[-2].set_xticks([])
        axes[-2].set_yticks([])

        axes[-1].imshow(
            input_grid[1],
            cmap="Reds",
            alpha=0.8,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[-1].imshow(
            label_grid,
            cmap="Blues",
            alpha=0.5,
            origin="lower",
            vmin=0,
            vmax=1,
        )
        axes[-1].set_title("Action + target")
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])

        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    def _clear_grids(self):
        self._input_grid.zero_()
        self._output_grid.zero_()

    def _det_physics(self, config):
        raw = torch.tensor([
            float(config["material"]["friction"]),
            float(config["material"]["density"]),
            float(config["box"]["friction"]),
        ], dtype=torch.float32)
        self._physics[:] = self._physics_bounds.normalize(raw)

    def __getitem__(self, idx: int):
        self._clear_grids()

        # look up sample and config in table
        run_idx = self._run_lookup[idx]
        run = self.runs[run_idx]
        config = self.configs[run_idx]
        sample_index = idx - self._offsets[run_idx]

        # extract sample at given index
        particles, particles_, plate_pos, plate_pos_, angle = self._extract_sample_in_pxl(run, sample_index)

        self._draw_particle_grid(particles, self._input_grid[0], config)
        self._draw_particle_grid(particles_, self._output_grid, config)
        self._draw_plate(plate_pos, plate_pos_, angle, config)
        self._det_physics(config)

        return (self._input_grid.clone(), self._physics.clone()), self._output_grid.clone()

def main():
    dataset = PileSweepData("corl/cube/n40", split="train")

    for i in range(10):
        inputs, label = dataset[i+10]
        input_, physics = inputs
        
        dataset.plot_input_and_output(input_, label, title=f"particles and plate {i}")
        
        
        

if __name__ == "__main__":
    main()
        
        