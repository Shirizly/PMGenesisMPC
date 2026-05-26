import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from .utilities.materials import *
import quaternion as qu
from pathlib import Path
import pickle
import os
import math
import copy
import torch


class SandboxManipulation:

    def __init__(self, config, n_envs: int = 1):
        """
        Initialize sandbox manipulation with multi-environment support.
        
        Args:
            config: Configuration dict or path to YAML file
            n_envs: Number of parallel environments within a single scene (default: 1)
        """
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, (str, Path)):
            base_dir = Path(__file__).parent
            full_path = base_dir / config
            with open(full_path) as stream:
                try:
                    self._config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            raise TypeError("config must be dict or a path to a YAML file")
        
        # Initialize Genesis Environment
        gs.init(
            backend=getattr(gs, self._config["simulation"].get('backend', 'gpu')),
            precision=self._config["simulation"].get('precision', '32'),
            performance_mode=self._config["simulation"].get('performance_mode', True),  # Enable for multi-env
        )

        # PARAMETERS FOR TRAINING
        self._box_pos = self._config["sandbox"]["box"].get('pos', [0.0, 0.0, 0.0])
        self._box_vol = self._config["sandbox"]["box"].get('vol', [0.3, 0.3, 0.1])
        self._wall_thickness = self._config["sandbox"]["box"].get('wall_thickness', 0.02)
        self._particle_size = self._config["sandbox"]["material"]["properties"].get('particle_size', 0.01)
        self._granular_vol = self._config["sandbox"]["material"].get('vol', [0.27, 0.27, 0.1])
        self._material_type = self._config["sandbox"]["material"].get('type', 'rsa')
        collection_cfg = self._config.get("data_collection", {})
        self._settle_steps = collection_cfg.get("settle_steps", 100)
        self._settle_min_steps = collection_cfg.get("settle_min_steps", 10)
        self._settle_check_interval = collection_cfg.get("settle_check_interval", 20)
        self._settle_threshold = collection_cfg.get("settle_threshold", 0.01)
        self._lower_steps = collection_cfg.get("lower_steps", 100)
        self._lift_steps = collection_cfg.get("lift_steps", 100)
        self._goal_threshold = collection_cfg.get("goal_threshold", 0.001)
        self._progress = collection_cfg.get("progress", True)
        self._sample_progress_interval = max(1, collection_cfg.get("sample_progress_interval", 1))
        self._phase_progress_interval = max(1, collection_cfg.get("phase_progress_interval", 100))
        self._trace_scene_steps = collection_cfg.get("trace_scene_steps", False)
        self._update_visualizer = collection_cfg.get("update_visualizer", False)
        self._settle_stabilization = collection_cfg.get("settle_stabilization", True)
        self._settle_angular_damping = collection_cfg.get("settle_angular_damping", 0.2)
        self._settle_linear_damping = collection_cfg.get("settle_linear_damping", 1.0)
        self._settle_sleep_threshold = collection_cfg.get("settle_sleep_threshold", 0.01)
        self._freeze_reached_envs = collection_cfg.get("freeze_reached_envs", False)
        
        # Multi-environment settings
        self._n_envs = n_envs

        self._init_scene()
        self._add_entities()

        self._n_aborted_down = torch.zeros(n_envs, device=gs.device)
        self._n_aborted_action = torch.zeros(n_envs, device=gs.device)
        self._particle_sizes = None
        self._particle_links_idx = None
        self._particle_dofs_idx = None
        self._particle_linear_dofs_idx = None
        self._particle_angular_dofs_idx = None
        self._particle_sizes_expanded = None
        self._env_index_offset = 0

    def _log(self, message: str):
        if self._progress:
            print(message, flush=True)

    def _log_step_progress(self, label: str | None, step: int, total_steps: int):
        if not label or total_steps <= 0:
            return
        # if step == 1 or step == total_steps or step % self._phase_progress_interval == 0:
        #     self._log(f"  {label}: step {step}/{total_steps}")

    def _step_scene(self, label: str | None = None, step: int | None = None, total_steps: int | None = None):
        if self._trace_scene_steps and label and step is not None and total_steps is not None:
            should_log_step = (
                step == 1
                or step == total_steps
                or step % self._phase_progress_interval == 0
            )
            self._scene.step(
                update_visualizer=self._update_visualizer,
                refresh_visualizer=self._update_visualizer,
            )
            if should_log_step:
                self._log(f"  {label}: scene.step {step}/{total_steps}")
        else:
            self._scene.step(
                update_visualizer=self._update_visualizer,
                refresh_visualizer=self._update_visualizer,
            )

    def _init_scene(self):
        viewer_settings = self._config["simulation"].get('viewer_options', dict())
        viz_settings = self._config["simulation"].get('viz_options', dict())
        c_fov = viewer_settings.get('camera_fov', 30)
        max_fps = viewer_settings.get('max_FPS', 60)
        resolution = viewer_settings.get('resolution', [1280, 1280])

        b_x, b_y, b_z = self._box_pos   
        v_x, v_y, v_z = self._box_vol
        l_bound = (b_x-2*v_x, b_y-2*v_y, b_z-2*v_z)
        u_bound = (b_x+2*v_x, b_y+2*v_y, b_z+2*v_z+self._wall_thickness)

        viewer_type = viewer_settings.get('viewer_type', None)
        
        if viewer_type == "observer":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [3 * v_x, 0.0, 10*v_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.0, 0.0, v_z/2]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
                res           = resolution,
            )
        elif viewer_type == "bird":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [b_x, b_y, 10*v_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.0, 0.0, 0.0]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
                res           = resolution,
            )
        elif viewer_type == "leveled":
            self._viewer_options = gs.options.ViewerOptions(
                camera_pos    = viewer_settings.get('camera_pos', [b_x+1.5, b_y, b_z]),
                camera_lookat = viewer_settings.get('camera_lookat', [0.5, 0.0, 0.2]),
                camera_fov    = c_fov,
                max_FPS       = max_fps,
                res           = resolution,
            )
        else:
            # No viewer --> Training mode
            self._viewer_options = None

        rigid_cfg = self._config.get("rigid_options", {})

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
                batch_links_info=rigid_cfg.get("batch_links_info", False),
                box_box_detection=rigid_cfg.get("box_box_detection", False),
                use_contact_island=rigid_cfg.get("use_contact_island", False),
                use_hibernation=rigid_cfg.get("use_hibernation", False),
                max_collision_pairs=rigid_cfg.get("max_collision_pairs", 150),
                enable_multi_contact=rigid_cfg.get("enable_multi_contact", True),
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ) if self._material_type in ("sand", "liquid") else None,
            sph_options=gs.options.SPHOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ) if self._material_type == "liquid" else None,
            pbd_options=gs.options.PBDOptions(
                lower_bound = l_bound,
                upper_bound = u_bound,
                particle_size = self._particle_size,
            ) if self._material_type == "liquid" else None,
            viewer_options = self._viewer_options,
            vis_options=gs.options.VisOptions(
                show_link_frame=viz_settings.get('show_link_frame', False),
            ),
            show_viewer=viewer_settings.get('show_viewer', False)
        )
        self._scene.profiling_options.show_FPS = viz_settings.get('show_FPS', False)
    
    def _add_entities(self):

        self.plane = self._scene.add_entity(
            gs.morphs.Plane()
        )

        x, y, z = self._box_pos
        _, _, box_height = self._box_vol

        self._plate_size = self._config["plate"].get("size", [0.1, 0.005, 0.06])
        self.plate = self._scene.add_entity(
            material=gs.materials.Rigid(
                rho=3000,
            ),
            morph=gs.morphs.Box(
                    pos=(x, y, z + (self._wall_thickness + self._granular_vol[2])/2 + box_height),
                    size=self._plate_size, 
                ),    
            surface=gs.surfaces.Default(
                color = self._config["plate"].get("color", [0.0, 1.0, 0.0]),
            ),
        )

        if not self._config["sandbox"]["box"].get('omit', False):
            self._add_box()
        
        self._add_material()

    def _add_box(self):
        x, y, z = self._box_pos
        width, depth, height = self._box_vol
        box_color = self._config["sandbox"]["box"].get('color', [0.0, 0.0, 0.0])
        friction = self._config["sandbox"]["box"]["properties"].get('friction', 1)

        self.box_parts = {}
        self.box_parts["ground_plate"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=self._box_pos,
                size=(width, depth, self._wall_thickness),
                fixed=True
            ),     
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_parts["front_wall"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=(x-(width+self._wall_thickness)/2, y, z+(height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_parts["back_wall"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=(x+(width+self._wall_thickness)/2, y, z+(height-self._wall_thickness)/2),
                size=(self._wall_thickness, depth, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_parts["left_wall"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=(x, y+(depth+self._wall_thickness)/2, z+(height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )
        
        self.box_parts["right_wall"] = self._scene.add_entity(
            material=gs.materials.Rigid(
                friction=friction,
            ),    
            morph=gs.morphs.Box(
                pos=(x, y-(depth+self._wall_thickness)/2, z+(height-self._wall_thickness)/2),
                size=(width, self._wall_thickness, height),
                fixed=True
            ),
            surface=gs.surfaces.Default(
                color = box_color,
            ),
        )

    def _add_material(self):
        material_properties = self._config["sandbox"]["material"].get('properties', {})
        granular_color = self._config["sandbox"]["material"].get('color', [1.0, 1.0, 0.0])
        self._safety_margin = self._config["sandbox"].get('safety_margin', 0.02)


        if (self._granular_vol[0] > self._box_vol[0]-self._safety_margin or self._granular_vol[1] > self._box_vol[1]-self._safety_margin):
            raise ValueError(
                f"Safety margin of {self._safety_margin} exceeded. Box volume is x={self._box_vol[0]}, y={self._box_vol[1]}, but granular volume is x={self._granular_vol[0]}, y={self._granular_vol[1]}.")

        granular_touch_height = self._granular_vol[2]/2
        if self._material_type == "rsa":
            shape = material_properties.get("shape", None)
            if shape is None:
                shape = "cube" if material_properties.get("cubes", False) else "sphere"
            self.material, self._rsa_particle_sizes = random_sequential_addition(
                scene=self._scene,
                granular_vol=self._granular_vol,
                shape=shape,
                num_particles=material_properties.get("n_particles", 1000),
                particle_size=material_properties["particle_size"],
                wall_thickness=self._wall_thickness,
            )                
            granular_touch_height = self._get_rsa_particle_touch_height(material_properties)
        
        self.material = random_sequential_addition(
            scene=self._scene,
            box_pos=self._box_pos,
            granular_vol=self._granular_vol,
            material_properties=material_properties,
            wall_thickness=self._wall_thickness,
            color=granular_color
        )                
        
        self._operation_height = self._box_pos[2] + self._wall_thickness/2 + self._get_rsa_particle_touch_height(material_properties)
        

    def _get_rsa_particle_touch_height(self, material_properties):
        shape = material_properties.get("shape", None)
        if shape is None:
            shape = "cube" if material_properties.get("cubes", False) else "sphere"
        shape = shape.lower()

        particle_size = material_properties.get("particle_size", self._particle_size)
        default_size = float(particle_size if isinstance(particle_size, (int, float)) else max(particle_size))

        def max_size(name, default):
            value = material_properties.get(name, default)
            if isinstance(value, (int, float)):
                return float(value)
            return float(max(value))

        if shape in ("rectangle", "rectangular_cube", "box", "cylinder"):
            return max_size("particle_height", default_size) * 0.5
        return default_size * 0.5

    def _save_data(
            self,
            path : str | Path,
            env_idx: int | None = None,
    ):
        """
        Save data in the legacy list-of-dicts pickle format.

        Each row is cloned before pickling. Indexing a large tensor produces a
        view, and pickling many views can serialize much more backing storage
        than the row itself needs.
        """
        path = Path(path)
        use_non_blocking = any(
            tensor.is_cuda
            for tensor in (
                self.valid_states,
                self.valid_states_,
                self.valid_p_starts,
                self.valid_p_stops,
                self.valid_angles,
            )
        )

        states = self.valid_states.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        states_ = self.valid_states_.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        p_starts = self.valid_p_starts.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        p_stops = self.valid_p_stops.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        angles = self.valid_angles.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        env_indices = None
        if hasattr(self, "valid_env_indices"):
            env_indices = self.valid_env_indices.detach().to('cpu', non_blocking=use_non_blocking).contiguous()
        if env_idx is not None:
            if env_indices is None:
                row_mask = torch.zeros((states.shape[0],), dtype=torch.bool)
            else:
                row_mask = env_indices == env_idx
            states = states[row_mask]
            states_ = states_[row_mask]
            p_starts = p_starts[row_mask]
            p_stops = p_stops[row_mask]
            angles = angles[row_mask]
            if env_indices is not None:
                env_indices = env_indices[row_mask]
        metadata = copy.deepcopy(self._config.get("data_collection_property_sweep", {}))
        metadata.update({
            "state_columns": [
                "x", "y", "z",
                "rot_x", "rot_y", "rot_z",
                "size_x", "size_y", "size_z",
            ],
            "action_columns": ["start_position", "end_position", "angle"],
            "particle_friction": copy.deepcopy(
                self._config["sandbox"]["material"].get("properties", {}).get("sampled_friction")
            ),
            "particle_density": copy.deepcopy(
                self._config["sandbox"]["material"].get("properties", {}).get("sampled_density")
            ),
            "table_friction": self._config["sandbox"]["box"].get("properties", {}).get("friction"),
            "plate_size": copy.deepcopy(self._plate_size),
            "box_pos": copy.deepcopy(self._box_pos),
            "box_vol": copy.deepcopy(self._box_vol),
            "wall_thickness": self._wall_thickness,
        })

        # Ensure transfers are complete before pickling
        if use_non_blocking:
            torch.cuda.synchronize()

        raw_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (states, states_, p_starts, p_stops, angles)
        )
        self._log(
            f"Saving {states.shape[0]} samples to {path.name} "
            f"({raw_bytes / 1024**2:.1f} MiB raw tensors)"
        )

        data = [
            {
                "state" : states[i].clone(),
                "state_" : states_[i].clone(),
                "action" : (p_starts[i].clone(), p_stops[i].clone(), angles[i].clone()),
                "metadata": self._sample_metadata(metadata, int(env_indices[i].item()) if env_indices is not None else None),
            } for i in range(states.shape[0])
        ]
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._log(f"Saved {path.name} ({path.stat().st_size / 1024**2:.1f} MiB on disk)")

    def _valid_samples_to_cpu(self):
        use_non_blocking = any(
            tensor.is_cuda
            for tensor in (
                self.valid_states,
                self.valid_states_,
                self.valid_p_starts,
                self.valid_p_stops,
                self.valid_angles,
            )
        )
        tensors = {
            "states": self.valid_states.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "states_": self.valid_states_.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_starts": self.valid_p_starts.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "p_stops": self.valid_p_stops.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "angles": self.valid_angles.detach().to('cpu', non_blocking=use_non_blocking).contiguous(),
            "env_indices": None,
        }
        if hasattr(self, "valid_env_indices"):
            tensors["env_indices"] = self.valid_env_indices.detach().to(
                'cpu', non_blocking=use_non_blocking
            ).contiguous()
        if use_non_blocking:
            torch.cuda.synchronize()
        return tensors

    def _base_sample_metadata(self):
        metadata = copy.deepcopy(self._config.get("data_collection_property_sweep", {}))
        metadata.update({
            "state_columns": [
                "x", "y", "z",
                "rot_x", "rot_y", "rot_z",
                "size_x", "size_y", "size_z",
            ],
            "action_columns": ["start_position", "end_position", "angle"],
            "particle_friction": copy.deepcopy(
                self._config["sandbox"]["material"].get("properties", {}).get("sampled_friction")
            ),
            "particle_density": copy.deepcopy(
                self._config["sandbox"]["material"].get("properties", {}).get("sampled_density")
            ),
            "table_friction": self._config["sandbox"]["box"].get("properties", {}).get("friction"),
            "plate_size": copy.deepcopy(self._plate_size),
            "box_pos": copy.deepcopy(self._box_pos),
            "box_vol": copy.deepcopy(self._box_vol),
            "wall_thickness": self._wall_thickness,
        })
        return metadata

    def _save_env_data_from_cpu(self, path: str | Path, tensors: dict, env_idx: int):
        path = Path(path)
        env_indices = tensors["env_indices"]
        if env_indices is None:
            row_mask = torch.zeros((tensors["states"].shape[0],), dtype=torch.bool)
        else:
            row_mask = env_indices == env_idx

        states = tensors["states"][row_mask]
        states_ = tensors["states_"][row_mask]
        p_starts = tensors["p_starts"][row_mask]
        p_stops = tensors["p_stops"][row_mask]
        angles = tensors["angles"][row_mask]
        filtered_env_indices = env_indices[row_mask] if env_indices is not None else None
        metadata = self._base_sample_metadata()

        raw_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (states, states_, p_starts, p_stops, angles)
        )
        self._log(
            f"Saving env {env_idx} with {states.shape[0]} samples to {path.name} "
            f"({raw_bytes / 1024**2:.1f} MiB raw tensors)"
        )
        data = [
            {
                "state": states[i].clone(),
                "state_": states_[i].clone(),
                "action": (p_starts[i].clone(), p_stops[i].clone(), angles[i].clone()),
                "metadata": self._sample_metadata(
                    metadata,
                    int(filtered_env_indices[i].item()) if filtered_env_indices is not None else env_idx,
                ),
            } for i in range(states.shape[0])
        ]
        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._log(f"Saved {path.name} ({path.stat().st_size / 1024**2:.1f} MiB on disk)")

    def _sample_metadata(self, metadata, env_idx: int | None):
        sample_metadata = copy.deepcopy(metadata)
        if env_idx is not None:
            global_env_idx = self._env_index_offset + env_idx
            sample_metadata.pop("env_settings", None)
            env_metadata = getattr(self, "_env_property_metadata", None)
            if env_metadata is not None:
                sample_metadata.update(copy.deepcopy(env_metadata[env_idx]))
            sample_metadata["local_env_idx"] = env_idx
            sample_metadata["env_idx"] = global_env_idx
        return sample_metadata

    def _save_config(
            self,
            path : str | Path,
            env_idx: int | None = None,
        ):

        with open(path, 'w') as outfile:
            try: 
                config = copy.deepcopy(self._config)
                if env_idx is not None:
                    global_env_idx = self._env_index_offset + env_idx
                    env_metadata = getattr(self, "_env_property_metadata", None)
                    config.setdefault("data_collection_property_sweep", {})
                    config["data_collection_property_sweep"].pop("env_settings", None)
                    config["data_collection_property_sweep"].update(
                        {
                            "mode": "single_environment",
                            "source_n_property_envs": self._n_envs,
                        }
                    )
                    if env_metadata is not None:
                        config["data_collection_property_sweep"].update(
                            copy.deepcopy(env_metadata[env_idx])
                        )
                    config["data_collection_property_sweep"]["local_env_idx"] = env_idx
                    config["data_collection_property_sweep"]["env_idx"] = global_env_idx
                    if hasattr(self, "valid_env_indices"):
                        env_successes = int((self.valid_env_indices == env_idx).sum().item())
                        config["statistics"] = copy.deepcopy(config.get("statistics", {}))
                        config["statistics"].update(
                            {
                                "Environment index": global_env_idx,
                                "Local environment index": env_idx,
                                "Environment samples collected": env_successes,
                                "Environment failed samples": max(
                                    0,
                                    int(config.get("data_collection", {}).get("samples_per_env", 0))
                                    - env_successes,
                                ),
                            }
                        )
                yaml.dump(config, outfile, default_flow_style=False)
            except yaml.YAMLError as exc:
                print(exc)
       
    def build(self):
        """Build the scene with multiple environments"""
        self._scene.build(n_envs=self._n_envs, env_spacing=(self._box_vol[0] *2 , self._box_vol[1] *2 ))  # Adjust env_spacing as needed
        
        dofs_idx = [0, 1, 2, 3, 4, 5]
        self.plate.set_dofs_kp((0.8,) * 6, dofs_idx)
        self.plate.set_dofs_kv((1.0,) * 6, dofs_idx)
        self._cache_particle_sizes()

    def _cache_particle_sizes(self):
        if self._material_type != "rsa":
            self._particle_sizes = None
            return

        sizes = []
        links_idx = []
        dofs_idx = []
        linear_dofs_idx = []
        angular_dofs_idx = []
        for i, particle in enumerate(self.material):
            if hasattr(self, "_rsa_particle_sizes") and i < len(self._rsa_particle_sizes):
                sizes.append(tuple(float(v) for v in self._rsa_particle_sizes[i]))
            else:
                sizes.append(self._get_particle_size(particle))
            links_idx.append(particle.link_start)
            if particle.n_dofs == 6:
                particle_dofs = list(range(particle.dof_start, particle.dof_end))
                dofs_idx.extend(particle_dofs)
                linear_dofs_idx.extend(particle_dofs[:3])
                angular_dofs_idx.extend(particle_dofs[3:])
        self._particle_sizes = torch.tensor(sizes, device=gs.device).view(1, -1, 3)
        self._particle_sizes_expanded = None
        self._particle_links_idx = torch.tensor(links_idx, dtype=gs.tc_int, device=gs.device)
        self._particle_dofs_idx = torch.tensor(dofs_idx, dtype=gs.tc_int, device=gs.device)
        self._particle_linear_dofs_idx = torch.tensor(linear_dofs_idx, dtype=gs.tc_int, device=gs.device)
        self._particle_angular_dofs_idx = torch.tensor(angular_dofs_idx, dtype=gs.tc_int, device=gs.device)

    def _expanded_particle_sizes(self):
        if self._particle_sizes is None:
            self._cache_particle_sizes()
        if (
            self._particle_sizes_expanded is None
            or self._particle_sizes_expanded.shape[0] != self._n_envs
        ):
            self._particle_sizes_expanded = self._particle_sizes.expand(self._n_envs, -1, -1)
        return self._particle_sizes_expanded

    def _get_particle_size(self, particle):
        if hasattr(particle.morph, "size"):
            size = particle.morph.size
            return (float(size[0]), float(size[1]), float(size[2]))
        if hasattr(particle.morph, "height") and hasattr(particle.morph, "radius"):
            diameter = float(particle.morph.radius) * 2
            return (diameter, diameter, float(particle.morph.height))
        diameter = float(particle.morph.radius) * 2
        return (diameter, diameter, diameter)

    def _sample_particle_property(self, value, *, min_value: float | None = None):
        n_particles = len(self.material)
        if isinstance(value, (int, float)):
            values = np.full(n_particles, float(value), dtype=np.float32)
        else:
            if len(value) == 2:
                values = np.random.uniform(float(value[0]), float(value[1]), n_particles).astype(np.float32)
            elif len(value) == n_particles:
                values = np.asarray(value, dtype=np.float32)
            else:
                raise ValueError(
                    "Particle property must be a scalar, [min, max], "
                    "or one value per particle."
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

        This avoids per-environment rigid link info and keeps simulation steps
        on the fast shared-material path.
        """
        particle_frictions = self._sample_particle_property(setting["particle_friction"], min_value=1e-2)
        particle_densities = self._sample_particle_property(setting["particle_density"], min_value=gs.EPS)
        table_friction = max(float(setting["table_friction"]), 1e-2)

        for particle_idx, particle in enumerate(self.material):
            particle.set_friction(float(particle_frictions[particle_idx]))
            self._set_particle_density_value(particle, float(particle_densities[particle_idx]))

        if hasattr(self, "box_parts"):
            for part in self.box_parts.values():
                part.set_friction(table_friction)
        if hasattr(self, "plane") and getattr(self.plane, "material", None) is not None:
            self.plane.set_friction(table_friction)

        material_properties = self._config["sandbox"]["material"].setdefault("properties", {})
        material_properties["sampled_friction"] = particle_frictions.tolist()
        material_properties["sampled_density"] = particle_densities.tolist()
        self._config["sandbox"]["box"].setdefault("properties", {})["friction"] = table_friction

        metadata = copy.deepcopy(setting.get("metadata", {}))
        metadata.update(
            {
                "particle_friction": material_properties["sampled_friction"],
                "particle_density": material_properties["sampled_density"],
                "table_friction": table_friction,
            }
        )
        self._env_property_metadata = [copy.deepcopy(metadata) for _ in range(self._n_envs)]

        self._config["data_collection_property_sweep"] = {
            "mode": "shared_batch",
            "n_property_envs": self._n_envs,
            "env_settings": copy.deepcopy(self._env_property_metadata),
        }
        return {
            "particle_friction": particle_frictions,
            "particle_density": particle_densities,
            "table_friction": table_friction,
        }

    def destroy(self):
        """Destroying environment"""
        gs.destroy()

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        """Simulate all environments (vectorized)"""
        for _ in range(horizon):
            self._step_scene()

    def shuffle_particles(self):
        """
        Randomize particle positions within the box bounds across all environments
        without particle-particle overlap.

        Particles are placed largest-first. Each step samples a batch of candidate
        positions per environment and accepts the first candidate that clears the
        already-placed particles. This keeps the expensive overlap test vectorized
        on the active device while avoiding the long tail of scalar RSA loops.
        """
        if self._material_type != "rsa":
            raise NotImplementedError("Method not implemented for materials other than RSA")

        n_particles = len(self.material)
        if n_particles == 0:
            return

        if self._particle_sizes is None:
            self._cache_particle_sizes()

        half_extents = []
        for particle in self.material:
            if hasattr(particle.morph, "size"):
                size = torch.as_tensor(particle.morph.size, dtype=torch.float32, device=gs.device)
                if size.ndim == 0:
                    size = size.repeat(3)
                half_extent = size[:3] * 0.5
            elif hasattr(particle.morph, "height") and hasattr(particle.morph, "radius"):
                radius = torch.tensor(float(particle.morph.radius), device=gs.device)
                height = torch.tensor(float(particle.morph.height), device=gs.device)
                half_extent = torch.stack((radius, radius, height * 0.5))
            else:
                radius = torch.tensor(float(particle.morph.radius), device=gs.device)
                half_extent = radius.repeat(3)
            half_extents.append(half_extent)
        half_extents = torch.stack(half_extents)
        wall_thickness = float(self._wall_thickness)

        inner_min = torch.tensor(
            [
                self._box_pos[0] - self._box_vol[0] * 0.5,
                self._box_pos[1] - self._box_vol[1] * 0.5,
                self._box_pos[2] + wall_thickness * 0.5,
            ],
            dtype=torch.float32,
            device=gs.device,
        )
        inner_max = torch.tensor(
            [
                self._box_pos[0] + self._box_vol[0] * 0.5,
                self._box_pos[1] + self._box_vol[1] * 0.5,
                self._box_pos[2] + self._box_vol[2] - wall_thickness * 0.5,
            ],
            dtype=torch.float32,
            device=gs.device,
        )

        lower = inner_min.unsqueeze(0) + half_extents
        upper = inner_max.unsqueeze(0) - half_extents
        if (upper[:, :2] < lower[:, :2]).any():
            raise ValueError("At least one particle is too large to fit inside the box.")

        positions = torch.empty((self._n_envs, n_particles, 3), device=gs.device)
        placed_mask = torch.zeros((n_particles,), dtype=torch.bool, device=gs.device)
        order = torch.argsort(torch.prod(half_extents, dim=1), descending=True)
        candidate_batch = max(1024, min(4096, 64 * n_particles))
        max_rounds = 128
        min_gap = 1e-4

        for particle_idx_tensor in order:
            particle_idx = int(particle_idx_tensor.item())
            active_envs = torch.ones((self._n_envs,), dtype=torch.bool, device=gs.device)
            particle_lower = lower[particle_idx, :2]
            particle_span = upper[particle_idx, :2] - particle_lower
            z_pos = inner_min[2] + half_extents[particle_idx, 2] + min_gap

            for _ in range(max_rounds):
                active_idx = torch.nonzero(active_envs, as_tuple=False).squeeze(1)
                if active_idx.numel() == 0:
                    break

                candidate_xy = (
                    torch.rand((active_idx.numel(), candidate_batch, 2), device=gs.device)
                    * particle_span.view(1, 1, 2)
                    + particle_lower.view(1, 1, 2)
                )

                placed_idx = torch.nonzero(placed_mask, as_tuple=False).squeeze(1)
                if placed_idx.numel() == 0:
                    valid = torch.ones(
                        (active_idx.numel(), candidate_batch),
                        dtype=torch.bool,
                        device=gs.device,
                    )
                else:
                    placed_positions = positions[active_idx][:, placed_idx, :2]
                    delta = candidate_xy.unsqueeze(2) - placed_positions.unsqueeze(1)
                    min_sep = half_extents[particle_idx, :2] + half_extents[placed_idx, :2] + min_gap
                    valid = (torch.abs(delta) >= min_sep.view(1, 1, -1, 2)).any(dim=3).all(dim=2)

                has_valid = valid.any(dim=1)
                if has_valid.any():
                    accepted_envs = active_idx[has_valid]
                    first_valid = valid[has_valid].to(torch.int64).argmax(dim=1)
                    positions[accepted_envs, particle_idx, :2] = candidate_xy[has_valid, first_valid, :]
                    positions[accepted_envs, particle_idx, 2] = z_pos
                    active_envs[accepted_envs] = False

            if active_envs.any():
                raise RuntimeError(
                    "Could not randomly shuffle particles without overlap. "
                    "Try a smaller particle size or fewer particles."
                )

            placed_mask[particle_idx] = True

        envs_idx = torch.arange(self._n_envs, device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            particle.set_pos(positions[:, particle_idx, :], envs_idx=envs_idx)
            particle.set_quat(self._random_particle_quats(particle, self._n_envs), envs_idx=envs_idx)

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
        if self._particle_links_idx is not None:
            return self._scene.rigid_solver.get_links_pos(links_idx=self._particle_links_idx)

        positions = torch.empty((self._n_envs, len(self.material), 3), device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            positions[:, particle_idx, :] = particle.get_pos()
        return positions

    def _get_particle_velocities(self):
        if self._particle_links_idx is not None:
            return self._scene.rigid_solver.get_links_vel(links_idx=self._particle_links_idx)

        velocities = torch.empty((self._n_envs, len(self.material), 3), device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            velocities[:, particle_idx, :] = particle.get_vel()
        return velocities

    def _get_particle_quats(self):
        if self._particle_links_idx is not None:
            return self._scene.rigid_solver.get_links_quat(links_idx=self._particle_links_idx)

        quats = torch.empty((self._n_envs, len(self.material), 4), device=gs.device)
        for particle_idx, particle in enumerate(self.material):
            quats[:, particle_idx, :] = particle.get_quat()
        return quats

    def _quat_to_xyz_rotation(self, quats):
        w, x, y, z = quats.unbind(dim=-1)
        rot_x = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sin_pitch = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
        rot_y = torch.asin(sin_pitch)
        rot_z = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.stack((rot_x, rot_y, rot_z), dim=-1)

    def _max_particle_dof_speed_sq(self):
        if self._particle_dofs_idx is None or self._particle_dofs_idx.numel() == 0:
            return None
        vel = self._scene.rigid_solver.get_dofs_velocity(dofs_idx=self._particle_dofs_idx)
        vel = vel.reshape(self._n_envs, len(self.material), 6)
        linear_speed_sq = torch.sum(vel[:, :, :3] * vel[:, :, :3], dim=2).max()
        angular_speed_sq = torch.sum(vel[:, :, 3:] * vel[:, :, 3:], dim=2).max()
        return torch.maximum(linear_speed_sq, angular_speed_sq)

    def _particles_settled(self):
        max_speed_sq = self._max_particle_dof_speed_sq()
        if max_speed_sq is None:
            return False, None
        settled = max_speed_sq < (self._settle_threshold * self._settle_threshold)
        return bool(settled.item()), max_speed_sq

    def _stabilize_settled_particles(self, progress_label: str | None = None):
        if (
            not self._settle_stabilization
            or self._particle_dofs_idx is None
            or self._particle_dofs_idx.numel() == 0
        ):
            return

        vel = self._scene.rigid_solver.get_dofs_velocity(dofs_idx=self._particle_dofs_idx)
        vel = vel.reshape(self._n_envs, len(self.material), 6)

        if self._settle_linear_damping != 1.0:
            vel[:, :, :3] *= self._settle_linear_damping
        if self._settle_angular_damping != 1.0:
            vel[:, :, 3:] *= self._settle_angular_damping

        if self._settle_sleep_threshold > 0:
            linear_speed = torch.linalg.norm(vel[:, :, :3], dim=2, keepdim=True)
            angular_speed = torch.linalg.norm(vel[:, :, 3:], dim=2, keepdim=True)
            vel[:, :, :3] = torch.where(linear_speed < self._settle_sleep_threshold, 0.0, vel[:, :, :3])
            vel[:, :, 3:] = torch.where(angular_speed < self._settle_sleep_threshold, 0.0, vel[:, :, 3:])

        self._scene.rigid_solver.set_dofs_velocity(
            vel.reshape(self._n_envs, -1),
            dofs_idx=self._particle_dofs_idx,
            skip_forward=True,
        )
        if progress_label:
            self._log(f"  {progress_label}: stabilized particle velocities")

    def get_material_state(
            self,
            settle_steps: int | None = None,
            progress_label: str | None = None,
            out: torch.Tensor | None = None,
        ):
        """
        Returns particle state for all environments.
        Optimized for GPU processing.

        Returns:
            Tensor of shape [n_envs, n_particles, 9] with
            (x, y, z, rot_x, rot_y, rot_z, size_x, size_y, size_z).
            Rotations are Euler angles in radians derived from the particle
            quaternion, and sizes are the particle dimensions.
        """
        if self._material_type != "rsa":
            raise NotImplementedError("Method not implemented for materials other than RSA")

        n_p = len(self.material)
        if settle_steps is None:
            settle_steps = self._settle_steps
        if progress_label:
            self._log(f"  {progress_label}: settling {settle_steps} steps")

        # Hold plate still while particles settle.
        plate_pos = self.plate.get_pos()
        self.plate.set_pos(plate_pos)
        self.plate.control_dofs_position_velocity(
            plate_pos,
            torch.zeros_like(plate_pos),
            dofs_idx_local=[0, 1, 2]
        )

        frozen_plate_dofs = self.plate.get_dofs_position()
        for step in range(settle_steps):
            self.plate.set_dofs_position(frozen_plate_dofs)
            self._step_scene(progress_label, step + 1, settle_steps)
            self._log_step_progress(progress_label, step + 1, settle_steps)
            if (
                step + 1 >= self._settle_min_steps
                and self._settle_check_interval > 0
                and (step + 1) % self._settle_check_interval == 0
            ):
                settled, max_speed_sq = self._particles_settled()
                if settled:
                    if progress_label:
                        max_speed = torch.sqrt(max_speed_sq).item()
                        self._log(
                            f"  {progress_label}: settled after {step + 1}/{settle_steps} "
                            f"steps, max speed {max_speed:.5f}"
                        )
                    break

        self._stabilize_settled_particles(progress_label)

        if out is None:
            state = torch.ones((self._n_envs, n_p, 9), device=gs.device)
        else:
            expected_shape = (self._n_envs, n_p, 9)
            if tuple(out.shape) != expected_shape:
                raise ValueError(f"Expected output state shape {expected_shape}, got {tuple(out.shape)}")
            state = out

        if progress_label:
            self._log(f"  {progress_label}: reading particle state")
        state[:, :, 0:3] = self._get_particle_positions()
        # state[:, :, 3:6] = self._quat_to_xyz_rotation(self._get_particle_quats())
        # state[:, :, 6:9] = self._expanded_particle_sizes()

        if progress_label:
            self._log(f"  {progress_label}: done")
        return state

    def get_collected_samples(self):
        """
        Return previously collected samples
        
        Each samples consists of state(i), state(i+1), start_position, end_position, angle, velocity
        """
        if not hasattr(self, "valid_states"):
            return []

        return [
            {
                "state": self.valid_states[i],
                "state_": self.valid_states_[i],
                "action": (self.valid_p_starts[i], self.valid_p_stops[i], self.valid_angles[i]),
            }
            for i in range(self.valid_states.shape[0])
        ]
    
    def plate_velocity_translation(
            self,
            p_start,
            p_end,
            speed,
            angle,
            sweep_steps: int | None = None,
            debug=False,
            progress_label: str | None = None,
        ):
        """
        Move plates with velocity control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            speed: Movement speed (scalar)
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
        
        operation_height = getattr(self, "_action_operation_height", self._operation_height)

        # Horizontal movement
        fix_z_and_rot = torch.stack([
            # x is free dof
            # y is free dof
            torch.full((self._n_envs,), operation_height, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            angle
        ], dim=1)


        # Calculate velocity vector for each environment
        delta = p_end - p_start  # [n_envs, 3]
        dist = torch.linalg.norm(delta, axis=1, keepdim=True)  # [n_envs, 1]  
        direction = delta / (dist + 1e-8)
        v = direction * speed  # [n_envs, 3]

        # Set initial position, velocity and goal for all plates in all environments
        self.plate.set_pos(p_start)
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
        
        if sweep_steps is None:
            max_sweep_distance = float(dist.max().item())
            sweep_steps = max(1, math.ceil(max_sweep_distance / (speed * self._scene.dt) * 1.7))
            if progress_label:
                self._log(
                    f"  {progress_label}: max action distance {max_sweep_distance:.4f}m "
                    f"-> {sweep_steps} sweep steps"
                )

        if progress_label:
            self._log(f"  {progress_label}: sweeping {sweep_steps} steps")

        reached_goal = torch.zeros(self._n_envs, dtype=torch.bool, device=gs.device)
        best_dist = torch.full((self._n_envs,), torch.inf, device=gs.device)
        frozen_pos = self.plate.get_pos()

        for step in range(sweep_steps):
            if self._freeze_reached_envs and reached_goal.any():
                reached_envs_idx = reached_goal.nonzero().squeeze(dim=1)
                n_reached = reached_envs_idx.shape[0]
                self.plate.set_pos(
                    frozen_pos[reached_goal],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )
                self.plate.set_dofs_position(
                    torch.stack([
                        frozen_pos[reached_goal, 0],
                        frozen_pos[reached_goal, 1],
                        torch.full((n_reached,), operation_height, device=gs.device),
                        torch.zeros(n_reached, device=gs.device),
                        torch.zeros(n_reached, device=gs.device),
                        angle[reached_goal],
                    ], dim=1),
                    dofs_idx_local=[0, 1, 2, 3, 4, 5],
                    envs_idx=reached_envs_idx,
                    zero_velocity=True,
                )

            self.plate.set_dofs_position(
                fix_z_and_rot,
                dofs_idx_local=[2, 3, 4, 5],
            )
            self._step_scene(progress_label, step + 1, sweep_steps)
            self._log_step_progress(progress_label, step + 1, sweep_steps)

            cur_pos = self.plate.get_pos()
            cur_dist = torch.linalg.norm(cur_pos[:, :2] - p_end[:, :2], axis=1)
            improved = cur_dist < best_dist
            best_dist = torch.where(improved, cur_dist, best_dist)
            newly_reached = (cur_dist < self._goal_threshold) & ~reached_goal
            frozen_pos = torch.where(newly_reached[:, None], cur_pos, frozen_pos)
            reached_goal |= newly_reached
            if self._freeze_reached_envs and reached_goal.all():
                if progress_label:
                    self._log(f"  {progress_label}: all environments reached target at step {step + 1}")
                break

        if self._freeze_reached_envs:
            final_pos = torch.where(reached_goal[:, None], frozen_pos, self.plate.get_pos())
        else:
            final_pos = torch.where(reached_goal[:, None], p_end, self.plate.get_pos())

        if progress_label:
            self._log(
                f"  {progress_label}: reached {int(reached_goal.sum().item())}/{self._n_envs}; "
                f"best distance range {float(best_dist.min().item()):.4f}-"
                f"{float(best_dist.max().item()):.4f}m"
            )

        return reached_goal, final_pos, sweep_steps
    
    def plate_position_translation(
            self,
            p_start,
            p_end,
            n_steps,
            fix_pose,
            fix_dofs,
            debug=False,
            progress_label: str | None = None,
        ):
        """
        Move plates with position control across all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            n_steps: Number of steps for interpolation
            fix_pose: Pose values for locked DOFs [n_envs, n_dofs] or [n_dofs]
            fix_dofs: Indices of DOFs to lock
        """
        
        if progress_label:
            self._log(f"  {progress_label}: moving {n_steps} steps")

        if n_steps <= 1:
            self.plate.set_pos(pos=p_end)
            self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)
            self._step_scene(progress_label, 1, 1)
            return

        t = torch.linspace(0, 1, n_steps, device=gs.device)
        path = (1 - t[:, None, None]) * p_start[None, :, :] + t[:, None, None] * p_end[None, :, :]

        self.plate.set_pos(path[0])
        for i in range(n_steps):
            self.plate.set_pos(pos=path[i])
            self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)
            self._step_scene(progress_label, i + 1, n_steps)
            self._log_step_progress(progress_label, i + 1, n_steps)

    def generate_action_samples(
            self,
            n_samples: int,
        ):
        """
        Generate random action samples for all environments.
        
        Returns:
            Tuple of (action_starts, action_stops, angles) each of shape [n_envs * n_samples, 3/1]
        """
        box_x, box_y, _ = self._box_pos
        tool_length, tool_width, tool_height = self._plate_size
        self._action_operation_height = self._operation_height + tool_height / 2

        # Generate samples for each environment
        n_total = self._n_envs * n_samples
        angles = (-torch.pi/2) + torch.rand(n_total, device=gs.device) * torch.pi
        
        # Sampling dimensions in x and y from box center
        sample_space_x = self._granular_vol[0]/2 - (torch.cos(angles) * tool_length/2 + abs(torch.sin(angles)) * tool_width/2 + self._safety_margin)
        sample_space_y = self._granular_vol[1]/2 - (abs(torch.sin(angles)) * tool_length/2 + torch.cos(angles) * tool_width/2 + self._safety_margin)

        # Min and max coordinates
        low = torch.stack([box_x - sample_space_x, box_y - sample_space_y], axis=1)
        high = torch.stack([box_x + sample_space_x, box_y + sample_space_y], axis=1)
        
        # Sample start and end positions
        start_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        stop_samples = (high - low) * torch.rand((n_total, 2), device=gs.device) + low
        _z = torch.ones((n_total, 1), device=gs.device) * self._action_operation_height
        
        action_starts = torch.concatenate((start_samples, _z), axis=1)
        action_stops = torch.concatenate((stop_samples, _z), axis=1)
        
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
            speed,
            lift_height,
            sweep_steps: int | None = None,
            progress_label: str | None = None,
        ):
        """
        Execute action (lower, sweep, lift) for all environments.
        
        Args:
            p_start: Starting positions [n_envs, 3]
            p_stop: Stopping positions [n_envs, 3]
            angle: Angles [n_envs]
            speed: Movement speed (scalar)
            lift_height: Lift height [n_envs, 3]
        
        Returns:
            Tensor of shape [n_envs] with success status
        """
        # Lowering
        fix_pose_lower = torch.stack([
            p_start[:, 0],
            p_start[:, 1],
            # z is free dof 
            torch.zeros(self._n_envs, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            angle
        ], dim=1)
        self.plate_position_translation(
            p_start + lift_height,
            p_start,
            self._lower_steps,
            fix_pose_lower,
            [0, 1, 3, 4, 5],
            progress_label=f"{progress_label} lower" if progress_label else None,
        )
        reached_goal, final_pos, _ = self.plate_velocity_translation(
            p_start,
            p_stop,
            speed,
            angle,
            sweep_steps=sweep_steps,
            progress_label=f"{progress_label} sweep" if progress_label else None,
        )
        fix_pose_lift = torch.stack([
            final_pos[:, 0],
            final_pos[:, 1],
            torch.zeros(self._n_envs, device=gs.device),
            torch.zeros(self._n_envs, device=gs.device),
            angle
        ], dim=1)

        self.plate_position_translation(
            final_pos,
            final_pos + lift_height,
            self._lift_steps,
            fix_pose_lift,
            [0, 1, 3, 4, 5],
            progress_label=f"{progress_label} lift" if progress_label else None,
        )
        return reached_goal, final_pos

    def collect_data_samples(
            self,
            n_samples: int = 200,
            speed: float = 0.125,
            path : str | Path = "training",
            settle_steps: int | None = None,
            sweep_steps: int | None = None,
            env_index_offset: int = 0,
        ):
        """
        Collect data samples from all environments efficiently.
        Optimized for GPU processing and memory efficiency.

        Args:
            n_samples: Number of samples to collect per environment
            speed: Plate movement speed
            path: Output path for data
        """
        effective_settle_steps = self._settle_steps if settle_steps is None else settle_steps
        effective_sweep_steps = sweep_steps
        self._env_index_offset = env_index_offset

        self._config.setdefault("data_collection", {})
        self._config["data_collection"].update({
            "n_envs": self._n_envs,
            "env_index_offset": env_index_offset,
            "samples_per_env": n_samples,
            "speed": speed,
            "settle_steps": effective_settle_steps,
            "settle_min_steps": self._settle_min_steps,
            "settle_check_interval": self._settle_check_interval,
            "settle_threshold": self._settle_threshold,
            "sweep_steps": effective_sweep_steps,
            "sweep_steps_mode": "auto" if effective_sweep_steps is None else "fixed",
            "lower_steps": self._lower_steps,
            "lift_steps": self._lift_steps,
            "goal_threshold": self._goal_threshold,
            "progress": self._progress,
            "sample_progress_interval": self._sample_progress_interval,
            "phase_progress_interval": self._phase_progress_interval,
            "trace_scene_steps": self._trace_scene_steps,
            "update_visualizer": self._update_visualizer,
            "settle_stabilization": self._settle_stabilization,
            "settle_angular_damping": self._settle_angular_damping,
            "settle_linear_damping": self._settle_linear_damping,
            "settle_sleep_threshold": self._settle_sleep_threshold,
            "freeze_reached_envs": self._freeze_reached_envs,
        })

        self._log(
            f"Preparing collection: n_envs={self._n_envs}, samples_per_env={n_samples}, "
            f"particles={len(self.material)}, speed={speed}"
        )

        # Setup lift height
        lift_height = self._box_vol[2]
        lift_height_tensor = torch.tensor([0, 0, lift_height], device=gs.device)
        lift_height_tensor = lift_height_tensor.unsqueeze(0).expand(self._n_envs, -1)

        # Generate action samples for all environments
        self._log("Generating action samples...")
        action_starts, action_stops, angles = self.generate_action_samples(n_samples)

        max_samples = n_samples * self._n_envs

        self._log(
            f"Allocating GPU buffers for up to {max_samples} samples "
            f"({n_samples} batches x {self._n_envs} envs)"
        )
        state_dim = 9
        states = torch.empty((n_samples, self._n_envs, len(self.material), state_dim), device=gs.device)
        states_ = torch.empty_like(states)
        p_starts = torch.empty((n_samples, self._n_envs, 3), device=gs.device)
        p_stops = torch.empty((n_samples, self._n_envs, 3), device=gs.device)
        sample_angles = torch.empty((n_samples, self._n_envs), device=gs.device)
        success_mask = torch.empty((n_samples, self._n_envs), dtype=torch.bool, device=gs.device)
        env_indices = torch.arange(self._n_envs, device=gs.device).unsqueeze(0).expand(n_samples, -1)
        particle_sizes = self._expanded_particle_sizes().unsqueeze(0).expand(n_samples, -1, -1, -1)
        states[:, :, :, 6:9] = particle_sizes
        states_[:, :, :, 6:9] = particle_sizes

        current_state = self.get_material_state(
            settle_steps=effective_settle_steps,
            progress_label="initial pre-state",
            out=states[0],
        )
        current_state_slot = 0

        for i, sample_idx in enumerate(range(n_samples)):
            should_log_batch = (
                sample_idx == 0
                or sample_idx == n_samples - 1
                or sample_idx % self._sample_progress_interval == 0
            )
            batch_label = f"batch {sample_idx + 1}/{n_samples}"
            if should_log_batch:
                self._log(f"Collecting action {batch_label}")

            if current_state_slot != sample_idx:
                states[sample_idx].copy_(current_state)

            p_start = action_starts[:, sample_idx, :]  # [n_envs, 3]
            p_stop = action_stops[:, sample_idx, :]    # [n_envs, 3]
            angle = angles[:, sample_idx]              # [n_envs]

            reached_goal, final_pos = self.execute_action(
                p_start,
                p_stop,
                angle,
                speed,
                lift_height_tensor,
                sweep_steps=effective_sweep_steps,
                progress_label=batch_label if should_log_batch else None,
            )

            post_state = self.get_material_state(
                settle_steps=effective_settle_steps,
                progress_label=f"{batch_label} post-state" if should_log_batch else None,
                out=states_[sample_idx],
            )
            p_starts[sample_idx] = p_start
            p_stops[sample_idx] = final_pos
            sample_angles[sample_idx] = angle
            success_mask[sample_idx] = reached_goal

            # Periodically reshuffle particles to ensure diverse interactions and prevent overfitting to specific configurations
            if i % 5 == 0 and sample_idx + 1 < n_samples:
                self.shuffle_particles()
                current_state = self.get_material_state(
                    settle_steps=effective_settle_steps,
                    progress_label=f"{batch_label} reshuffled pre-state" if should_log_batch else None,
                    out=states[sample_idx + 1],
                )
                current_state_slot = sample_idx + 1
            else:
                current_state = post_state
                current_state_slot = None

            if should_log_batch:
                self._log(f"Finished action {batch_label}")

        # self._log("Compacting successful samples...")
        flat_success_mask = success_mask.reshape(max_samples)
        self.valid_states = states.reshape(max_samples, len(self.material), state_dim)[flat_success_mask]
        self.valid_states_ = states_.reshape(max_samples, len(self.material), state_dim)[flat_success_mask]
        self.valid_p_starts = p_starts.reshape(max_samples, 3)[flat_success_mask]
        self.valid_p_stops = p_stops.reshape(max_samples, 3)[flat_success_mask]
        self.valid_angles = sample_angles.reshape(max_samples)[flat_success_mask]
        self.valid_env_indices = env_indices.reshape(max_samples)[flat_success_mask]
        write_ptr = int(flat_success_mask.sum().item())

        # Print statistics
        print("\nStatistics (Multi-Environment Collection)")
        print("=" * 50)
        print(f">> Number of environments   : {self._n_envs}")
        print(f">> Samples per environment  : {n_samples}")
        print(f">> Total samples collected  : {write_ptr}")
        print(f">> Number of failed samples : {max_samples - write_ptr}")

        self._config["statistics"] = {
            "Number of environments"   : self._n_envs,
            "Samples per environment"  : n_samples,
            "Total samples collected"  : write_ptr,
            "Number of failed samples" : max_samples - write_ptr,
        }

        base_dir = Path(__file__).parent
        full_path = base_dir / path
        Path.mkdir(full_path, parents=True, exist_ok=True)

        cpu_samples = self._valid_samples_to_cpu()
        for env_idx in range(self._n_envs):
            filename_prefix = str(env_index_offset + env_idx)
            self._save_config(full_path / f"{filename_prefix}_config.yaml", env_idx=env_idx)
            self._save_env_data_from_cpu(
                full_path / f"{filename_prefix}_data.pkl",
                cpu_samples,
                env_idx,
            )
        self._log(f"Saved {self._n_envs} per-env runs to {full_path}.")
        # Clean up GPU memory
        self._cleanup_gpu_memory()
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after data collection"""
        # Delete large tensors
        if hasattr(self, 'valid_states'):
            del self.valid_states
        if hasattr(self, 'valid_states_'):
            del self.valid_states_
        if hasattr(self, 'valid_p_starts'):
            del self.valid_p_starts
        if hasattr(self, 'valid_p_stops'):
            del self.valid_p_stops
        if hasattr(self, 'valid_angles'):
            del self.valid_angles
        if hasattr(self, 'valid_env_indices'):
            del self.valid_env_indices

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
