import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from .utilities.materials import *
from pathlib import Path
import os
import math
import torch


class SandboxManipulation:

    def __init__(
        self,
        config: dict | str | Path,
        n_envs: int = 1,
        debug : bool = False,
        viewer_type: str | None = None,
    ):
        """
        Initialize sandbox manipulation with multi-environment support.
        
        Args:
            config: Configuration dict or path to YAML file
            n_envs: Number of parallel environments within a single scene (default: 1)
        """
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, (str, Path)):
            full_path = Path(__file__).parent / config
            with open(full_path) as stream:
                self._config = yaml.safe_load(stream)
        else:
            raise TypeError("config must be dict or a path to a YAML file")
    
        # extract subdicts from config
        self._sim_params = self._config["simulation"]
        self._box_params = self._config["box"]
        self._plate_params = self._config["plate"] 
        self._material_params = self._config["material"]
        self._config.setdefault("data_collection", {})
        self._config["data_collection"].setdefault("sampled", {})
        self._sampled_params = self._config["data_collection"]["sampled"]
        
        self._rigid_options = self._config.get("rigid_options", {})
        
        # Init simulation
        gs.init(
            backend=getattr(gs, self._sim_params.get('backend', 'gpu')),
            precision=self._sim_params.get('precision', '32'),
            performance_mode=self._sim_params.get('performance_mode', True),  # Enable for multi-env
        )

        # PARAMETERS FOR TRAINING
        self._wall_thickness = self._box_params.get('wall_thickness', 0.02)
        self._granular_vol = self._material_params.get('vol', [0.27, 0.27, 0.1])

        # Configurable timing — read from simulation section so they can be
        # tuned without rebuilding the scene.
        self._settle_steps   = int(self._sim_params.get('settle_steps',   100))
        self._goal_threshold = 0.001
        
        self._debug = debug
        self._viewer_type = viewer_type
        
        # Multi-environment settings
        self._n_envs = n_envs

        self._init_scene()
        self._add_entities()

        # Active particle count — may be reduced per-experiment via set_n_active()
        # to "park" excess particles outside the camera's field of view.
        self._n_active = self._material_params["n_particles"]
        # Parking position: far from box, above ground plane (outside camera FOV)
        _bw = self._box_params["vol"][0]
        self._park_pos = [_bw * 15.0, 0.0, self._wall_thickness * 0.5 + 0.005]

        ###########
        # HELPERS #
        ###########
        
        # operation height
        particle_size = self._material_params["particle_size"]
        p_height = particle_size/2 if isinstance(particle_size, float) else min(particle_size)/4
        self._operation_height = self._wall_thickness/2 + p_height + self._plate_params["size"][2]/2
        
        # lift height for plate
        lift_height = self._box_params["vol"][2]
        self._lift_height_tensor = torch.tensor([0, 0, lift_height], device=gs.device).expand(self._n_envs, -1)
        
        # used to create path for position control (lower/lift plate)
        self._pos_ctrl_steps = int(self._sim_params.get('pos_ctrl_steps', 100))
        self._steps_0to1 = torch.linspace(0, 1, self._pos_ctrl_steps, device=gs.device)

        # Clearance height for teleport-assisted lower/lift.
        # The plate is teleported to this height above the operating height
        # before simulating only the short final descent (and the first short
        # ascent before teleporting away).  2 × particle_size clears the top of
        # even a two-layer pile; 8 mm is the minimum to avoid spawning inside
        # a cube corner.
        _ps  = particle_size if isinstance(particle_size, float) else max(particle_size)
        _lift_h = self._box_params['vol'][2]          # full lift = box interior height
        self._clearance_height     = max(0.008, 2.0 * _ps)
        self._clearance_ctrl_steps = max(10, int(round(
            self._pos_ctrl_steps * self._clearance_height / _lift_h)))
        self._clearance_offset = torch.zeros((self._n_envs, 3), device=gs.device)
        self._clearance_offset[:, 2] = self._clearance_height

        # helpers to fix all dofs except z during lowering and lifting
        self._vertical_dofs_local = [0, 1, 3, 4, 5] 
        self._vertical_dof_fix = torch.zeros((self._n_envs, 5), device=gs.device)

        # helpers to fix all dofs except x, y during sweeping
        self._horizontal_dofs_local = [2, 3, 4, 5] 
        self._horizontal_dof_fix = torch.zeros((self._n_envs, 4), device=gs.device)
        self._horizontal_dof_fix[:, 0] = self._operation_height

        self._particle_state = torch.empty((self._n_envs, self._material_params["n_particles"], 7), device=gs.device)
        self._particle_state_ = torch.empty((self._n_envs, self._material_params["n_particles"], 7), device=gs.device)
        
        self._zero_n_envsx3 = torch.zeros((self._n_envs, 3), device=gs.device)

        # pre-allocated freeze buffer for reached-goal envs in the sweep loop
        # layout: [x, y, z=operation_height, roll=0, pitch=0, yaw]
        self._freeze_dofs_buf = torch.zeros((self._n_envs, 6), device=gs.device)
        self._freeze_dofs_buf[:, 2] = self._operation_height


    def _log(self, message: str):
        print(message, flush=True)

    def _step_scene(self):
        _show = self._debug or self._viewer_type is not None
        self._scene.step(
            update_visualizer=_show,
            refresh_visualizer=_show,
        )

    def _init_scene(self):
        v_x, _, v_z = self._box_params["vol"]
        resolution = (1280, 1280)
        
        if self._viewer_type == "observer":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = [3 * v_x, 0.0, 3*v_z],
                camera_lookat = [0.0, 0.0, v_z/2],
                res           = resolution,
            )
        elif self._viewer_type == "bird":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = [0, 0, 10*v_z],
                camera_lookat = [0.0, 0.0, 0.0],
                res           = resolution,
            )
        elif self._viewer_type == "leveled":
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = [1.5, 0, v_z],
                camera_lookat = [0.0, 0.0, v_z],
                res           = resolution,
            )
        else:
            # No viewer --> Training mode
            viewer_options = None

        rigid_cfg = self._config.get("rigid_options", {})
        # rendered_envs_idx: restrict which parallel envs the (non-batched)
        # rasterizer considers, e.g. [0] so an overhead camera bound to env 0
        # (via add_camera(env_idx=0)) only ever sees that env's geometry —
        # needed by GenesisOracleEnv, which runs n_envs>1 rollout workers
        # alongside a single "real" env. None (default) renders all envs,
        # matching prior single-env behaviour.
        rendered_envs_idx = self._sim_params.get('rendered_envs_idx', None)
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt       = self._config["simulation"].get('dt', 4e3),
                substeps = self._config["simulation"].get('substeps', 1),
            ),
            rigid_options=gs.options.RigidOptions(
                iterations=rigid_cfg.get("iterations", 50),
                ls_iterations=rigid_cfg.get("ls_iterations", 50),
                tolerance=rigid_cfg.get("tolerance", 1e-6),
                ls_tolerance=rigid_cfg.get("ls_tolerance", 0.01),
                box_box_detection=rigid_cfg.get("box_box_detection", False),
                use_contact_island=rigid_cfg.get("use_contact_island", False),
                use_hibernation=rigid_cfg.get("use_hibernation", False),
                max_collision_pairs=rigid_cfg.get("max_collision_pairs", 150),
                enable_multi_contact=rigid_cfg.get("enable_multi_contact", True),
            ),
            viewer_options = viewer_options,
            vis_options=gs.options.VisOptions(
                show_link_frame=self._debug and self._viewer_type == "observer",
                rendered_envs_idx=rendered_envs_idx,
            ),
            show_viewer=self._debug or self._viewer_type is not None
        )
        self._scene.profiling_options.show_FPS=False
    
    def _add_entities(self):
        width, depth, height = self._box_params["vol"]

        def add_box_entity(pos, size):
            box = gs.morphs.Box(pos=pos, size=size, fixed=True)
            surface = gs.surfaces.Default(color=[0, 0, 0])
            return self._scene.add_entity(morph=box, surface=surface)
        
        # floor        
        self.plane = self._scene.add_entity(gs.morphs.Plane())

        # add container
        self.box_parts = {
            "ground_plate": add_box_entity(
                pos=(0, 0, 0),
                size=(width, depth, self._wall_thickness),
            ),
            "front_wall" : add_box_entity(
                pos=(-(width+self._wall_thickness)/2, 0, (height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
            ),
            "back_wall" : add_box_entity(
                pos=((width+self._wall_thickness)/2, 0, (height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
            ),
            "left_wall" : add_box_entity(
                pos=(0, (depth+self._wall_thickness)/2, (height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
            ),
            "right_wall" : add_box_entity(
                pos=(0, -(depth+self._wall_thickness)/2, (height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
            ),
        }
        
        # add tool
        self.plate = self._scene.add_entity(
            material=gs.materials.Rigid(
                rho=3000,
            ),
            morph=gs.morphs.Box(
                pos=(0, 0, height * 2),
                size=self._plate_params["size"]
            ),
            surface=gs.surfaces.Default(color=[0, 1, 0])
        )
        
        # add granular
        self._safety_margin = 0.02

        particle_size = self._sampled_params.get(
            "particle_size",
            self._material_params["particle_size"],
        )
        self.material, particle_sizes = random_sequential_addition(
            scene=self._scene,
            granular_vol=self._granular_vol,
            shape=self._material_params["shape"],
            num_particles=self._material_params["n_particles"],
            particle_size=particle_size,
            wall_thickness=self._wall_thickness,
        )      
        self._config["data_collection"]["sampled"].update({"particle_sizes": particle_sizes})

    def _save_data(self, path : str | Path, num : int, flat_success_mask : torch.Tensor, max_samples : int):
        """
        Save data efficiently using torch.save (binary format).
        
        Avoids per-sample cloning and per-element pickling. Supports both
        successful and failed samples. ~2-10x faster than pickle list-of-dicts.
        """
        path = Path(path)
        
        # Split into valid (successful) and failed samples
        valid_states = self._collection_buffers["states"].reshape(max_samples, len(self.material), 7)[flat_success_mask]
        valid_states_ = self._collection_buffers["states_"].reshape(max_samples, len(self.material), 7)[flat_success_mask]
        valid_p_starts = self._collection_buffers["p_starts"].reshape(max_samples, 3)[flat_success_mask]
        valid_p_stops = self._collection_buffers["p_stops"].reshape(max_samples, 3)[flat_success_mask]
        valid_angles = self._collection_buffers["sample_angles"].reshape(max_samples)[flat_success_mask]
        
        failed_states = self._collection_buffers["states"].reshape(max_samples, len(self.material), 7)[~flat_success_mask]
        failed_states_ = self._collection_buffers["states_"].reshape(max_samples, len(self.material), 7)[~flat_success_mask]
        failed_p_starts = self._collection_buffers["p_starts"].reshape(max_samples, 3)[~flat_success_mask]
        failed_p_stops = self._collection_buffers["p_stops"].reshape(max_samples, 3)[~flat_success_mask]
        failed_angles = self._collection_buffers["sample_angles"].reshape(max_samples)[~flat_success_mask]

        # Check if any tensor is on GPU
        use_non_blocking = any(
            tensor.is_cuda
            for tensor in (valid_states, valid_states_, valid_p_starts, valid_p_stops, valid_angles)
        )

        # Transfer all tensors to CPU in bulk (GPU → CPU DMA)
        valid_data = {
            "states": valid_states.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "states_": valid_states_.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_starts": valid_p_starts.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_stops": valid_p_stops.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "angles": valid_angles.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
        }
        
        failed_data = {
            "states": failed_states.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "states_": failed_states_.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_starts": failed_p_starts.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_stops": failed_p_stops.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "angles": failed_angles.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
        }

        # Ensure GPU→CPU transfers complete before I/O
        if use_non_blocking:
            torch.cuda.synchronize()

        # Save as torch binary format (faster and preserves dtype/shape)
        torch.save(valid_data, str(path / f"_{num}_data.pt"))
        torch.save(failed_data, str(path / f"_{num}_failed.pt"))

    def _save_config(
            self,
            path : str | Path,
            num : int
        ):
        path = path / (f"_{num}_config.yaml")
        with open(path, 'w') as outfile:
            yaml.dump(self._config, outfile, default_flow_style=False)

    def _allocate_collection_buffers(self, n_samples: int):
        """Allocate persistent GPU buffers for repeated data collection."""
        state_dim = 7
        self._collection_buffers = {
            "states" : torch.empty((n_samples, self._n_envs, len(self.material), state_dim), device=gs.device),
            "states_" : torch.empty((n_samples, self._n_envs, len(self.material), state_dim), device=gs.device),
            "p_starts" : torch.empty((n_samples, self._n_envs, 3), device=gs.device),
            "p_stops" : torch.empty((n_samples, self._n_envs, 3), device=gs.device),
            "sample_angles" : torch.empty((n_samples, self._n_envs), device=gs.device),
            "success_mask" : torch.empty((n_samples, self._n_envs), dtype=torch.bool, device=gs.device),
        }

    @staticmethod
    def load_data(path: str | Path, split: str = "valid"):
        """
        Load saved data from torch.save format (replaces old pickle loader).
        
        Args:
            path: Can be one of:
                - Full path to .pt file: "/path/to/0_data.pt"
                - Base path without extension: "/path/to/0_data"
                - Run directory with number: "/path/to/training" (looks for "0_data.pt")
            split: "valid" for successful samples, "failed" for failed samples (ignored if path has extension)
        
        Returns:
            Dict with keys: "states", "states_", "p_starts", "p_stops", "angles"
            Each is a CPU-side tensor ready for training.
        
        Example:
            # Full path
            data = SandboxManipulation.load_data("/path/to/0_data.pt")
            
            # Base path with split
            data = SandboxManipulation.load_data("/path/to/0_data", split="valid")
            data = SandboxManipulation.load_data("/path/to/0", split="valid")
        """
        path = Path(path)
        
        # If path has .pt extension, use it directly
        if path.suffix == ".pt":
            file_path = path
        else:
            # Construct filename based on split
            if split == "valid":
                suffix = "_data.pt"
            elif split == "failed":
                suffix = "_failed.pt"
            else:
                raise ValueError("split must be 'valid' or 'failed'")
            
            # Handle case where path ends with _data or _failed already
            path_str = str(path)
            if path_str.endswith("_data"):
                file_path = Path(path_str.replace("_data", suffix))
            elif path_str.endswith("_failed"):
                file_path = Path(path_str.replace("_failed", suffix))
            else:
                file_path = path.parent / (path.name + suffix)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        return torch.load(file_path, weights_only=False)
       
    def build(self):
        """Build the scene with multiple environments"""
        self._scene.build(
            n_envs=self._n_envs,
            env_spacing=(self._box_params["vol"][0]*2 , self._box_params["vol"][1]*2)
            )  # Adjust env_spacing as needed
        
        dofs_idx = [0, 1, 2, 3, 4, 5]
        self.plate.set_dofs_kp((0.8,) * 6, dofs_idx)
        self.plate.set_dofs_kv((1.0,) * 6, dofs_idx)

        self._cache_particle_idx()

    def _cache_particle_idx(self):
        links_idx = []
        dofs_idx = []
        for i, particle in enumerate(self.material):
            links_idx.append(particle.link_start)
            if particle.n_dofs == 6:
                dofs_idx.extend(range(particle.dof_start, particle.dof_end))
                
        self._particle_links_idx = torch.tensor(links_idx, dtype=gs.tc_int, device=gs.device)
        self._particle_dofs_idx = torch.tensor(dofs_idx, dtype=gs.tc_int, device=gs.device)

    def _sample_particle_property(self, value, *, min_value: float | None = None):
        n_particles = len(self.material)
        if isinstance(value, (int, float)):
            values = np.full(n_particles, float(value), dtype=np.float32)
        else:
            if len(value) >= n_particles:
                values = np.asarray(value[:n_particles], dtype=np.float32)
            else:
                raise ValueError(
                    "Particle property must be a scalar or a list with the same length as the number of particles"
                )
        if min_value is not None:
            values = np.maximum(values, min_value)
        return values

    def _set_particle_density_value(self, particle, density: float):
        old_density = getattr(particle.material, "rho", None)
        particle.material.rho = float(density)
        if getattr(self._scene, "is_built", False) and old_density is not None and old_density > 0:
            particle.set_mass(particle.get_mass() * (float(density) / float(old_density)))

    def set_material_properties(self, setting):
        """
        Set one material configuration shared by all parallel environments.

        This keeps Genesis on the fast shared link-info path. Density changes
        are applied as scalar entity mass updates, not per-environment masses.
        """
        particle_friction = (
            setting["sampled_particle_friction"]
            if setting.get("sampled_particle_friction") is not None
            else setting["particle_friction"]
        )
        particle_density = (
            setting["sampled_particle_density"]
            if setting.get("sampled_particle_density") is not None
            else setting["particle_density"]
        )
        particle_frictions = self._sample_particle_property(particle_friction, min_value=1e-2)
        particle_densities = self._sample_particle_property(particle_density, min_value=gs.EPS)
        box_friction = max(float(setting["box_friction"]), 1e-2)

        for particle_idx, particle in enumerate(self.material):
            particle.set_friction(float(particle_frictions[particle_idx]))
            self._set_particle_density_value(particle, float(particle_densities[particle_idx]))

        for part in self.box_parts.values():
            part.set_friction(box_friction)

        # save to config dict
        self._material_params["friction"] = setting["particle_friction"]
        self._material_params["density"] = setting["particle_density"]
        self._box_params["friction"] = setting["box_friction"]
        self._sampled_params.pop("friction", None)
        self._sampled_params.pop("density", None)
        if setting.get("sampled_particle_friction") is not None:
            self._sampled_params["friction"] = particle_frictions.tolist()
        if setting.get("sampled_particle_density") is not None:
            self._sampled_params["density"] = particle_densities.tolist()
        self._sampled_params["box_friction"] = box_friction

    def set_n_active(self, n: int) -> None:
        """Set how many particles are active (placed inside the box) on reset.

        Particles with indices ``[n, len(material))`` are moved to a parking
        position outside the camera's field of view on the next call to
        ``shuffle_particles()``.  The change takes effect on the next reset.
        """
        n_total = len(self.material)
        if not (0 <= n <= n_total):
            raise ValueError(f"n must be in [0, {n_total}], got {n}")
        self._n_active = n

    def shuffle_particles(self):
        n_particles = len(self.material)
        if n_particles == 0:
            return
        n_active = getattr(self, '_n_active', n_particles)

        max_retries = 10
        for attempt in range(max_retries):
            try:
                size_values = self._sampled_params.get("particle_sizes", None)
                if size_values is None:
                    size_values = [
                        particle.morph.size if hasattr(particle.morph, "size")
                        else (particle.morph.radius * 2,) * 3
                        for particle in self.material
                    ]
                sizes = torch.as_tensor(size_values, dtype=torch.float32, device=gs.device)
                half_extents = sizes * 0.5

                # For cubes, a random yaw rotation up to 45° increases the xy footprint by up to sqrt(2).
                # Use conservative collision extents so placed cubes don't overlap after rotation is applied.
                is_cube = torch.tensor(
                    [hasattr(p.morph, "size") for p in self.material],
                    dtype=torch.float32, device=gs.device,
                )
                xy_scale = 1.0 + (math.sqrt(2) - 1.0) * is_cube  # sqrt(2) for cubes, 1.0 for others
                collision_half_extents = half_extents.clone()
                collision_half_extents[:, :2] = half_extents[:, :2] * xy_scale.unsqueeze(1)

                width, depth, height = self._box_params["vol"]
                wall = float(self._wall_thickness)
                inner_min = torch.tensor([-width / 2, -depth / 2, wall / 2], device=gs.device)
                inner_max = torch.tensor([width / 2, depth / 2, height - wall / 2], device=gs.device)
                lower = inner_min + collision_half_extents
                upper = inner_max - collision_half_extents

                positions = torch.empty((self._n_envs, n_particles, 3), device=gs.device)
                placed = torch.zeros(n_particles, dtype=torch.bool, device=gs.device)
                order = torch.argsort(torch.prod(half_extents, dim=1), descending=True)
                order = order[order < n_active]  # only place the first n_active particles
                candidate_batch = max(1024, min(4096, 64 * max(n_active, 1)))
                min_gap = 1e-3

                for particle_idx_tensor in order:
                    particle_idx = int(particle_idx_tensor.item())
                    active = torch.ones(self._n_envs, dtype=torch.bool, device=gs.device)
                    span_xy = upper[particle_idx, :2] - lower[particle_idx, :2]
                    z_pos = inner_min[2] + half_extents[particle_idx, 2] + min_gap
                    for _ in range(128):
                        active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
                        if active_idx.numel() == 0:
                            break
                        candidate_xy = (
                            torch.rand((active_idx.numel(), candidate_batch, 2), device=gs.device)
                            * span_xy
                            + lower[particle_idx, :2]
                        )
                        placed_idx = torch.nonzero(placed, as_tuple=False).squeeze(1)
                        if placed_idx.numel() == 0:
                            valid = torch.ones((active_idx.numel(), candidate_batch), dtype=torch.bool, device=gs.device)
                        else:
                            delta = candidate_xy.unsqueeze(2) - positions[active_idx][:, placed_idx, :2].unsqueeze(1)
                            min_sep = collision_half_extents[particle_idx, :2] + collision_half_extents[placed_idx, :2] + min_gap
                            valid = (torch.abs(delta) >= min_sep.view(1, 1, -1, 2)).any(dim=3).all(dim=2)
                        has_valid = valid.any(dim=1)
                        accepted = active_idx[has_valid]
                        first_valid = valid[has_valid].to(torch.int64).argmax(dim=1)
                        positions[accepted, particle_idx, :2] = candidate_xy[has_valid, first_valid]
                        positions[accepted, particle_idx, 2] = z_pos
                        active[accepted] = False
                    if active.any():
                        raise RuntimeError("placement_failed")
                    placed[particle_idx] = True

                envs_idx = torch.arange(self._n_envs, device=gs.device)
                # Move parked (inactive) particles to a fixed spot outside the box
                if n_active < n_particles:
                    park = torch.tensor(self._park_pos, dtype=torch.float32, device=gs.device)
                    positions[:, n_active:, :] = park.view(1, 1, 3).expand(
                        self._n_envs, n_particles - n_active, 3)
                for particle_idx, particle in enumerate(self.material):
                    particle.set_pos(positions[:, particle_idx, :], envs_idx=envs_idx)
                    particle.set_quat(self._random_particle_quats(particle, self._n_envs), envs_idx=envs_idx)
                if self._particle_dofs_idx.numel() > 0:
                    self._scene.rigid_solver.set_dofs_velocity(
                        torch.zeros((self._n_envs, self._particle_dofs_idx.numel()), device=gs.device),
                        dofs_idx=self._particle_dofs_idx,
                        skip_forward=True,
                    )
                # Success, break out of retry loop
                break
            except RuntimeError as e:
                if str(e) == "placement_failed":
                    print(f"Placement of particles failed due to overlap, retrying {attempt+1}/{max_retries}...")
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Could not randomly shuffle particles without overlap after {max_retries} attempts. "
                            "Try a smaller particle size or fewer particles."
                        )
                    # else, try again
                    continue
                else:
                    raise

    def _random_particle_quats(self, particle, n_envs: int) -> torch.Tensor:
        if not hasattr(particle.morph, "size") and not hasattr(particle.morph, "height"):
            return torch.tensor((1.0, 0.0, 0.0, 0.0), device=gs.device).repeat(n_envs, 1)

        roll = torch.zeros(n_envs, device=gs.device)
        pitch = torch.zeros(n_envs, device=gs.device)
        yaw = torch.rand(n_envs, device=gs.device) * math.tau

        cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
        cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        return torch.stack(
            (
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ),
            dim=1,
        )

    def _get_particle_positions(self):
        return self._scene.rigid_solver.get_links_pos(links_idx=self._particle_links_idx)

    def _get_particle_quats(self):
        return self._scene.rigid_solver.get_links_quat(links_idx=self._particle_links_idx)

    def update_material_state(self, store_other=False):
        """
        Returns particle state (positions and sizes) for all environments.
        Optimized for GPU processing.

        Returns:
            Tensor of shape [n_envs, n_particles, 4] with (x, y, z, size)
        """

        # Hold plate still
        self.plate.set_pos(self.plate.get_pos())
        self.plate.control_dofs_position_velocity(
            self.plate.get_pos(),
            self._zero_n_envsx3,
            dofs_idx_local=[0, 1, 2]
        )

        frozen_plate_dofs = self.plate.get_dofs_position()
        for _ in range(self._settle_steps):
            self.plate.set_dofs_position(frozen_plate_dofs)
            self._step_scene()

        self._particle_state[:, :, 0:3] = self._get_particle_positions()
        self._particle_state[:, :, 3:] = self._get_particle_quats()

    def broadcast_state_from_env(self, src_env: int = 0) -> None:
        """
        Copy particle pose (position + quaternion) from environment
        ``src_env`` to every environment, and zero particle velocities.

        Used by multi-candidate rollout planners (see
        ``simple_mpc.genesis_oracle.GenesisOracleEnv``) to reset all
        ``n_envs`` copies to a common starting state before evaluating a new
        batch of candidate pushes. Requires ``self._particle_state`` to be
        current (i.e. ``update_material_state()`` was called after the last
        push). Does not touch the plate: ``execute_action`` teleports the
        plate to its start pose on every call, so no explicit plate reset is
        needed between rollouts.
        """
        envs_idx = torch.arange(self._n_envs, device=gs.device)
        src_pos  = self._particle_state[src_env:src_env + 1, :, 0:3]   # (1, n_p, 3)
        src_quat = self._particle_state[src_env:src_env + 1, :, 3:7]   # (1, n_p, 4)
        for particle_idx, particle in enumerate(self.material):
            particle.set_pos(
                src_pos[:, particle_idx, :].expand(self._n_envs, 3), envs_idx=envs_idx)
            particle.set_quat(
                src_quat[:, particle_idx, :].expand(self._n_envs, 4), envs_idx=envs_idx)
        if self._particle_dofs_idx.numel() > 0:
            self._scene.rigid_solver.set_dofs_velocity(
                torch.zeros((self._n_envs, self._particle_dofs_idx.numel()), device=gs.device),
                dofs_idx=self._particle_dofs_idx,
                skip_forward=True,
            )
        self._particle_state[:] = self._particle_state[src_env:src_env + 1].expand(
            self._n_envs, -1, -1)

    def plate_velocity_translation(
            self,
            p_start,
            p_end,
            angle,
            debug=False,
        ):
        """
        Move plates with velocity control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            angle: Rotation angle (scalar)
        Returns:
            reached_goal : Mask of environments that reached the goal
        """
        
        if debug:
            self._scene.clear_debug_objects()
            T_start = gu.trans_to_T(p_start[0])
            T_end = gu.trans_to_T(p_end[0])
            self._scene.draw_debug_frame(T_start, axis_length=0.05, origin_size=0.001, axis_radius=0.001)
            self._scene.draw_debug_frame(T_end, axis_length=0.05, origin_size=0.001, axis_radius=0.001)
        
        # Horizontal movement
        self._horizontal_dof_fix[:, -1] = angle 

        # Calculate velocity vector for each environment
        delta = p_end - p_start  # [n_envs, 3]
        dist = torch.linalg.norm(delta, axis=1, keepdim=True)  # [n_envs, 1]  
        direction = delta / (dist + 1e-8)
        v = direction * self._plate_params["speed"]  # [n_envs, 3]

        # Set initial position, velocity and goal for all plates in all environments
        self.plate.set_pos(p_start)
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
        
        max_sweep_distance = float(dist.max().item())
        sweep_steps = max(1, math.ceil(max_sweep_distance / (self._plate_params["speed"] * self._scene.dt) * 1.7))
        
        reached_goal = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
        best_dist = torch.full((self._n_envs,), torch.inf, device=gs.device)
        frozen_pos = self.plate.get_pos()
        
        n_reached = 0
        for step in range(sweep_steps):
            if n_reached > 0:
                reached_envs_idx = reached_goal.nonzero().squeeze(dim=1)
                self.plate.set_pos(
                    frozen_pos[reached_goal],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )
                # write only varying columns in-place; z/roll/pitch stay constant
                self._freeze_dofs_buf[reached_envs_idx, 0] = frozen_pos[reached_envs_idx, 0]
                self._freeze_dofs_buf[reached_envs_idx, 1] = frozen_pos[reached_envs_idx, 1]
                self._freeze_dofs_buf[reached_envs_idx, 5] = angle[reached_envs_idx]
                self.plate.set_dofs_position(
                    self._freeze_dofs_buf[reached_envs_idx],
                    dofs_idx_local=[0, 1, 2, 3, 4, 5],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )

            self.plate.set_dofs_position(
                self._horizontal_dof_fix,
                dofs_idx_local=self._horizontal_dofs_local,
            )
            self._step_scene()

            cur_pos = self.plate.get_pos()
            cur_dist = torch.linalg.norm(cur_pos[:, :2] - p_end[:, :2], axis=1)
            improved = cur_dist < best_dist
            best_dist = torch.where(improved, cur_dist, best_dist)
            newly_reached = (cur_dist < self._goal_threshold) & ~reached_goal
            frozen_pos = torch.where(newly_reached[:, None], cur_pos, frozen_pos)
            reached_goal |= newly_reached
            
            n_reached = int(reached_goal.sum().item())
            if n_reached == self._n_envs:
                if self._debug:
                    print(f"All environments reached target at step {step + 1}")
                break

        final_pos = torch.where(reached_goal[:, None], frozen_pos, self.plate.get_pos())

        if self._debug:
            print(
                f" > Goal reached : {int(reached_goal.sum().item())}/{self._n_envs}; "
                f" > Best distance range {float(best_dist.min().item()):.4f}-"
                f"{float(best_dist.max().item()):.4f}m"
            )

        return reached_goal, final_pos
    
    def plate_position_translation(self, p_start, p_end, n_steps: int | None = None):
        """
        Move plates with position control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            n_steps: Override step count (defaults to self._pos_ctrl_steps)
        """
        n = n_steps if n_steps is not None else self._pos_ctrl_steps
        steps_0to1 = (self._steps_0to1 if n_steps is None
                      else torch.linspace(0, 1, n, device=gs.device))
        path = (1 - steps_0to1[:, None, None]) * p_start[None, :, :] + steps_0to1[:, None, None] * p_end[None, :, :]

        self.plate.set_pos(p_start)
        for i in range(n):
            self.plate.set_pos(pos=path[i])
            self.plate.set_dofs_position(
                position=self._vertical_dof_fix,
                dofs_idx_local=self._vertical_dofs_local
            )
            self._step_scene()

    def generate_action_samples(
            self,
            n_samples: int,
        ):
        """
        Generate random action samples for all environments.
        
        Returns:
            Tuple of (action_starts, action_stops, angles) each of shape [n_envs * n_samples, 3/1]
        """
        tool_length, tool_width, _ = self._plate_params["size"]

        # Generate samples for each environment
        n_total = self._n_envs * n_samples
        angles = (-torch.pi/2) + torch.rand(n_total, device=gs.device) * torch.pi
        
        # Sampling dimensions in x and y from box center
        sample_space_x = self._granular_vol[0]/2 - (torch.cos(angles) * tool_length/2 + abs(torch.sin(angles)) * tool_width/2 + self._safety_margin)
        sample_space_y = self._granular_vol[1]/2 - (abs(torch.sin(angles)) * tool_length/2 + torch.cos(angles) * tool_width/2 + self._safety_margin)

        # Min and max coordinates
        low = torch.stack([-sample_space_x, -sample_space_y], axis=1)
        high = torch.stack([sample_space_x, sample_space_y], axis=1)
        
        # Sample start and end positions
        start_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        stop_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        _z = torch.ones((n_total, 1), device=gs.device) * self._operation_height
        
        action_starts = torch.cat((start_samples, _z), axis=1)
        action_stops = torch.cat((stop_samples, _z), axis=1)
        
        # Reshape to [n_envs, n_samples, ...]
        action_starts = action_starts.reshape(self._n_envs, n_samples, 3)
        action_stops = action_stops.reshape(self._n_envs, n_samples, 3)
        angles = angles.reshape(self._n_envs, n_samples)

        return action_starts, action_stops, angles

    def execute_action(
            self,
            p_start,
            p_stop,
            angle,
        ):
        """
        Execute action (lower, sweep, lift) for all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3]
            p_stop: Stopping positions [n_envs, 3]
            angle: Angles [n_envs]
            lift_height: Lift height [n_envs, 3]
        
        Returns:
            Tensor of shape [n_envs] with success status
        """

        # Lower: teleport to clearance height, then simulate only the short
        # final descent into operating position.  This skips simulating the
        # approach from the full lift height above.
        self._vertical_dof_fix[:, 0] = p_start[:, 0]
        self._vertical_dof_fix[:, 1] = p_start[:, 1]
        self._vertical_dof_fix[:, 4] = angle
        lower_start = p_start + self._clearance_offset
        self.plate.set_pos(lower_start, zero_velocity=True)
        self.plate_position_translation(lower_start, p_start, self._clearance_ctrl_steps)

        # Sweep
        reached_goal, final_pos = self.plate_velocity_translation(
            p_start,
            p_stop,
            angle,
        )

        # Lift: simulate only the short ascent to clearance height, then
        # teleport the plate out of the way.  Particles are already below
        # clearance height so there is no contact after this point.
        self._vertical_dof_fix[:, 0] = final_pos[:, 0]
        self._vertical_dof_fix[:, 1] = final_pos[:, 1]
        self.plate_position_translation(
            final_pos, final_pos + self._clearance_offset, self._clearance_ctrl_steps)
        self.plate.set_pos(final_pos + self._lift_height_tensor, zero_velocity=True)

        return reached_goal, final_pos

    def collect_data_samples(
            self,
            n_samples: int = 200,
            path : str | Path = "training",
        ):
        """
        Collect data samples from all environments efficiently.
        Optimized for GPU processing and memory efficiency.

        Args:
            n_samples: Number of samples to collect per environment
            path: Output path for data
        """
        max_samples = n_samples * self._n_envs

        self._config["data_collection"].update({
            "n_envs": self._n_envs,
            "samples_per_env": n_samples,
            "goal_threshold": self._goal_threshold,
        })

        # Allocate once or reuse if same size
        if (not hasattr(self, '_collection_buffers') or 
            self._collection_buffers['states'].shape[0] != n_samples or
            self._collection_buffers['states'].shape[1] != self._n_envs):
            self._allocate_collection_buffers(n_samples)
        
        # Clear data buffer
        for buf in self._collection_buffers.values():
            buf.zero_()
        
        # Generate random action samples per env
        action_starts, action_stops, angles = self.generate_action_samples(n_samples)

        self.update_material_state()
        for sample_idx in range(n_samples):
            print(f" > sample {sample_idx + 1}/{n_samples}")


            self._collection_buffers["states"][sample_idx].copy_(self._particle_state)

            p_start = action_starts[:, sample_idx, :]  # [n_envs, 3]
            p_stop = action_stops[:, sample_idx, :]    # [n_envs, 3]
            angle = angles[:, sample_idx]              # [n_envs]

            reached_goal, p_stop = self.execute_action(
                p_start,
                p_stop,
                angle,
            )
            
            self.update_material_state()

            self._collection_buffers["states_"][sample_idx].copy_(self._particle_state)   
            self._collection_buffers["p_starts"][sample_idx] = p_start
            self._collection_buffers["p_stops"][sample_idx] = p_stop
            self._collection_buffers["sample_angles"][sample_idx] = angle
            self._collection_buffers["success_mask"][sample_idx] = reached_goal
            if self._debug and torch.equal(self._collection_buffers["states"][sample_idx], self._collection_buffers["states_"][sample_idx]):
                print("State did not change")
            
        # Number of collected samples
        flat_success_mask = self._collection_buffers["success_mask"].reshape(max_samples)
        num_collected_samples = int(flat_success_mask.sum().item())

        # Print statistics
        print("\nStatistics (Multi-Environment Collection)")
        print("=" * 50)
        print(f">> Number of environments   : {self._n_envs}")
        print(f">> Samples per environment  : {n_samples}")
        print(f">> Total samples collected  : {num_collected_samples}")
        print(f">> Number of failed samples : {max_samples - num_collected_samples}")

        self._config["statistics"] = {
            "n_envs"   : self._n_envs,
            "samples_per_env"  : n_samples,
            "total_samples_collected"  : num_collected_samples,
            "number_of_failed_samples" : max_samples - num_collected_samples,
        }

        base_dir = Path(__file__).parent
        full_path = base_dir / path
        Path.mkdir(full_path, parents=True, exist_ok=True)

        # look for number of runs in existing dict
        n_runs = int(len([name for name in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, name))])/3)
        
        self._save_config(full_path, n_runs)
        self._save_data(full_path, n_runs, flat_success_mask, max_samples)
        self._log(f"Material batch finished. Run {n_runs} saved to {full_path}.")

    def destroy(self):
        gs.destroy()

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        for _ in range(horizon):
            self._step_scene()
