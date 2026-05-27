from torch.utils.data import Dataset
from pathlib import Path
import hashlib
import yaml
import torch
import torch.nn.functional as F
import os
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

TO_PXL = 1e3


class PileSweepData(Dataset):

    def __init__(
            self,
            paths: list[str] | str,
            run: int | None = None,
            split: str | None = None,
            val_pct: int = 5,
            test_pct: int = 5,
            resolution_scale: float = 1.0,
            include_sweep_removed: bool = False,
        ):
        """
        Initialize dataset with either a folder containing data or a specific run.

            @param paths: list of folder paths or a single folder path containing data files.
                Relative paths are resolved under Genesis/data; absolute paths are used as-is.
            @param run: number of a specific run
            @param split: one of "train", "val", "test", or None (all data).
                Splits are deterministic and stratified per leaf data folder, so each
                physical geometry group contributes to train/val/test when possible.
                Whole runs with the same nominal physics params stay in the same split.
            @param val_pct: percentage of physics groups assigned to validation
            @param test_pct: percentage of physics groups assigned to test
        """
        assert split in (None, "train", "val", "test"), f"Invalid split: {split!r}"
        if val_pct < 0 or test_pct < 0 or val_pct + test_pct >= 100:
            raise ValueError("val_pct and test_pct must be non-negative and sum to less than 100.")
        if resolution_scale <= 0:
            raise ValueError("resolution_scale must be positive.")
        self.runs = []
        self.configs = []
        self._run_lengths = []
        self._plate_cache = {}
        self._physics = torch.zeros((3,), dtype=torch.float32)
        self.resolution_scale = float(resolution_scale)
        self.to_pxl = TO_PXL * self.resolution_scale
        self.include_sweep_removed = bool(include_sweep_removed)


        parentpath = Path(__file__).parent.parent
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            full_path = self._resolve_data_path(path, parentpath)
            if not full_path.exists():
                raise FileNotFoundError(f"Data folder not found: {full_path}")

            run_files = self._collect_run_paths(full_path, run)
            if not run_files:
                raise FileNotFoundError(
                    f"No data runs found in path: {full_path}"
                )

            if split is not None:
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

        input_channels = 3 if self.include_sweep_removed else 2
        self._input_grid = torch.zeros((input_channels, x_pxl, y_pxl), dtype=torch.float32)
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

    def _collect_run_paths(self, root: Path, run: int | None):
        run_paths = []
        if run is not None:
            run_name = str(run)
            expected_data = root / f"{run_name}_data.pt"
            expected_config = root / f"{run_name}_config.yaml"
            _expected_data = root / f"_{run_name}_data.pt"
            _expected_config = root / f"_{run_name}_config.yaml"
            if expected_data.exists() and expected_config.exists():
                return [(expected_data, expected_config)]
            elif _expected_data.exists() and _expected_config.exists():
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
                elif _data_file.exists() and _config_file.exists():
                    return [(_data_file, _config_file)]

            return []

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

    @classmethod
    def _assign_split(cls, data_file: Path, root: Path, val_pct: int, test_pct: int) -> str:
        """Backward-compatible split helper for a single run file."""
        config_file = data_file.with_name(
            data_file.stem.replace("_data", "") + "_config.yaml"
        )
        cfg = yaml.full_load(config_file.read_text())
        key = cls._physics_key(cfg)
        h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
        if h < test_pct:
            return "test"
        elif h < test_pct + val_pct:
            return "val"
        return "train"

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
    
    def _draw_plate(self, start_pos, end_pos, angle, config, binary=False):
        plate_dim_x, plate_dim_y, _ = self._get_plate_dims(config)
        plate_dim_x *= self.to_pxl
        plate_dim_y *= self.to_pxl

        if not binary:
            occ1 = self._rectangle_occupancy(
                start_pos,
                angle,
                plate_dim_x,
                plate_dim_y
            )
            occ2 = self._rectangle_occupancy(
                end_pos,
                angle,
                plate_dim_x,
                plate_dim_y
            )
            self._input_grid[1] = 1 - (1 - occ1*0.5) * (1 - occ2)
        else:
            grid_np = self._input_grid[1].numpy()
            def draw_box_points(grid, center, box_dim, angle, density=1):
                rotated_rect = (
                    (int(center[0]), int(center[1])),
                    (int(box_dim[0]), int(box_dim[1])), 
                    int(angle * 180 / math.pi)
                )
                
                box = cv2.boxPoints(rotated_rect)
                box = np.int32(box)
                cv2.fillPoly(grid, [box], density)
            
            # Draw start position
            draw_box_points(
                grid_np,
                start_pos[:2],
                (plate_dim_x, plate_dim_y),
                angle,
                0.5
            )

            # Draw end position
            draw_box_points(
                grid_np,
                end_pos[:2],
                (plate_dim_x, plate_dim_y),
                angle,
                1
            )

    def _rectangle_occupancy(self, pos, theta, rect_w, rect_h, sigma=None):
        """
        Creates a soft occupancy grid for a rotated rectangle.
        Coordinates and dimensions are in pixels.
        """
        if sigma is None:
            sigma = max(0.5, 1.5 * self.resolution_scale)
        _, H, W = self._input_grid.shape
        cx, cy = pos[:2]
        ys = torch.linspace(0, H - 1, H)
        xs = torch.linspace(0, W - 1, W)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        x = xx - cx
        y = yy - cy

        c = torch.cos(theta)
        s = torch.sin(theta)
        xr = c * x + s * y
        yr = -s * x + c * y

        hx = rect_w / 2.0
        hy = rect_h / 2.0
        qx = torch.abs(xr) - hx
        qy = torch.abs(yr) - hy
        dx = torch.clamp(qx, min=0.0)
        dy = torch.clamp(qy, min=0.0)
        outside_dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        inside_dist = torch.clamp(torch.maximum(qx, qy), max=0.0)
        sdf = outside_dist + inside_dist
        return torch.sigmoid(-sdf / sigma)

    def _swept_plate_occupancy(self, start_pos, end_pos, angle, config):
        """
        Soft occupancy of the continuous tool sweep volume.

        The tool orientation is fixed during the sweep, so this accumulates
        rectangle occupancies along the straight-line path between start and end.
        """
        plate_dim_x, plate_dim_y, _ = self._get_plate_dims(config)
        plate_dim_x *= self.to_pxl
        plate_dim_y *= self.to_pxl
        distance = torch.norm(end_pos[:2] - start_pos[:2]).item()
        n_steps = max(2, int(math.ceil(distance)) + 1)
        mask = torch.zeros_like(self._output_grid)
        for alpha in torch.linspace(0.0, 1.0, n_steps):
            pos = start_pos + alpha * (end_pos - start_pos)
            mask = torch.maximum(
                mask,
                self._rectangle_occupancy(pos, angle, plate_dim_x, plate_dim_y),
            )
        return mask
        
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

    def _color_grid(
            self,
            grid: torch.Tensor,
            x: float,
            y: float,
            size: tuple[float, float],
            drawing: float | torch.Tensor = 1,
        ) -> None:
        
        h, w = grid.shape[:2]
        x_size = int(round(float(size[0])))
        y_size = int(round(float(size[1])))

        x0 = int(round(float(x - x_size / 2)))
        y0 = int(round(float(y - y_size / 2)))
        x1 = x0 + x_size
        y1 = y0 + y_size

        gx0 = max(0, x0)
        gy0 = max(0, y0)
        gx1 = min(h, x1)
        gy1 = min(w, y1)
        if gx0 >= gx1 or gy0 >= gy1:
            return

        target = grid[gx0:gx1, gy0:gy1]
        mask = target == 0

        if isinstance(drawing, torch.Tensor):
            dx0 = gx0 - x0
            dy0 = gy0 - y0
            dx1 = dx0 + (gx1 - gx0)
            dy1 = dy0 + (gy1 - gy0)
            # source = drawing[dx0:dx1, dy0:dy1]
            source = drawing[dx0:dx1, dy0:dy1]
            target[mask] = source[mask].to(torch.float32)
        else:
            target[mask] = float(drawing)

    def _det_physics(self, config):
        self._physics[0] = (config["material"]["friction"] - 0.05) / (0.5  - 0.05)
        self._physics[1] = (config["material"]["density"]  -  750) / (5000 -  750)
        self._physics[2] = (config["box"]["friction"]      - 0.05) / (0.5  - 0.05)
        # self._physics[3] = config["plate"]["speed"]

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
        if self.include_sweep_removed:
            sweep_mask = (self._swept_plate_occupancy(plate_pos, plate_pos_, angle, config) >= 0.5).to(torch.float32)
            self._input_grid[2] = self._input_grid[0] * (1.0 - sweep_mask)
        self._det_physics(config)

        return (self._input_grid.clone(), self._physics.clone()), self._output_grid.clone()

def main():
    dataset = PileSweepData("corl/cube/n40", include_sweep_removed=True)

    for i in range(10):
        inputs, label = dataset[i+10]
        input_, physics = inputs
        
        dataset.plot_input_and_output(input_, label, title=f"particles and plate {i}")
        
        
        

if __name__ == "__main__":
    main()
        
        