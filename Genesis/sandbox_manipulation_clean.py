import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import yaml
from .utilities.materials import *
from .transition_buffer import TransitionBuffer
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

        # Automatic transition recording — every push executed via
        # push_and_record() (real MPC steps and, tagged separately, candidate
        # rollouts) is appended here and flushed to a persistent, ever-growing
        # dataset directory. On by default; see docs/ARCHITECTURE.md /
        # docs/oracle_mpc_design.md for the on-disk schema and rationale.
        dc_cfg = self._config["data_collection"]
        self._record_transitions = bool(dc_cfg.get("record_transitions", True))
        self._transitions_dir = Path(__file__).parent / dc_cfg.get(
            "transitions_dir", "data/mpc_runs")
        self._transition_buffer = TransitionBuffer() if self._record_transitions else None
        # Context (source/episode_idx/seed/...) to tag flushes with — set
        # once per episode via set_transition_context(), used by every
        # push_and_record(flush_after=True) call until the next reset. This
        # is what lets real-step flushes happen incrementally (see
        # push_and_record) while still carrying episode-identifying info,
        # without needing to know the episode's outcome (only known at the
        # end) to flush during it.
        self._transition_context: dict | None = None
        
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
        # settle_steps is a CAP, not a fixed count: update_material_state stops
        # early once the pile is actually at rest. A fixed count is both
        # wasteful for a small pile (which settles in a few tens of steps) and
        # insufficient for a large one — a 200-cube pile is measurably still
        # moving after the full 100 steps, and since each transition's s is the
        # previous transition's s', an unsettled read propagates forward.
        self._settle_steps   = int(self._sim_params.get('settle_steps',   100))
        self._settle_check_every = int(self._sim_params.get('settle_check_every', 10))
        self._settle_vel_threshold = float(
            self._sim_params.get('settle_velocity_threshold', 1e-3))     # m/s
        # Angular rest threshold, derived from the linear one unless set
        # explicitly. A bare rad/s number is not comparable to a m/s number:
        # 0.1 rad/s on a 5 mm cube is a corner speed of 0.35 mm/s, i.e. three
        # times STRICTER than the 1 mm/s linear threshold, so it silently
        # became the binding criterion and kept piles "unsettled" long after
        # their centres had stopped. Converting through the particle's lever
        # arm makes both express the same surface speed.
        _ang_thr = self._sim_params.get('settle_angular_velocity_threshold', None)
        if _ang_thr is None:
            _ps = self._material_params.get("particle_size") or 0.005
            _ps = float(_ps) if isinstance(_ps, (int, float)) else float(max(_ps))
            _lever = max(_ps * math.sqrt(3) / 2, 1e-4)     # half body diagonal
            _ang_thr = self._settle_vel_threshold / _lever
        self._settle_angvel_threshold = float(_ang_thr)  # rad/s
        # Fraction of particles that must be below threshold. A plain max makes
        # the test harder the more envs are batched (32 envs x 200 particles is
        # 6400 chances for one straggler), so the settle never converges and
        # always burns its cap. 0.995 tolerates ~1 straggler per 200-cube env.
        self._settle_rest_quantile = float(
            self._sim_params.get('settle_rest_quantile', 0.995))
        # A max-velocity guard on top of the quantile was tried here and
        # deliberately NOT kept. The worry was that the quantile bounds how
        # MANY particles are still moving but not how fast, so s' could be
        # recorded mid-travel. Measured, the correlation runs the other way:
        # the fastest residual particles (73 mm/s at 50x128, 27 mm/s at
        # 100x64) drifted 0.1-1.2 mm over the next 0.2 s -- they vibrate in
        # place -- while the one genuine late movement (14 mm at 70x64) came
        # from a particle moving only 7 mm/s, a cube at the top of its tipping
        # arc where speed passes through a minimum. A max-velocity test would
        # have paid for extra settling on the harmless cases and still missed
        # the real one. See docs/scaling_to_200_objects.md section 8.
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

        # ---- actuator model (see configs/basic.yaml's `plate:` section) ----
        # The tool is carried by a Cartesian gantry, so it is modelled as a
        # trajectory-tracking servo with the gantry's reflected inertia and a
        # finite force budget, rather than as a free 2.4 g box on a soft
        # spring. _plate_accel shapes the trapezoidal speed reference the servo
        # follows; the gains are derived from moving_mass and bandwidth in
        # build(), where the entity's dofs exist.
        self._plate_moving_mass = float(self._plate_params.get("moving_mass", 0.5))
        self._plate_accel       = float(self._plate_params.get("acceleration", 2.0))
        self._plate_bandwidth   = float(self._plate_params.get("control_bandwidth_hz", 15.0))
        self._plate_max_force   = float(self._plate_params.get("max_force", 30.0))
        # Extra steps after the reference reaches the goal, letting the servo
        # close its remaining tracking error before the sweep is judged.
        self._sweep_settle_steps = int(self._sim_params.get("sweep_settle_steps", 12))


    def _log(self, message: str):
        print(message, flush=True)

    def _step_scene(self):
        _show = self._debug or self._viewer_type is not None
        self._scene.step(
            update_visualizer=_show,
            refresh_visualizer=_show,
        )

    def _default_max_collision_pairs(self) -> int:
        """Contact-pair budget to preallocate when the config doesn't set one.

        Genesis's own default is a flat 150, independent of how many bodies are
        in the scene. Measured occupancy for a settled-then-pushed pile of 50
        cubes of 5 mm is 51 broad-phase pairs and 211 contact points, i.e. a
        required ``max_collision_pairs`` of only **14** — the pile is mostly
        one floor contact per cube (4 points each under ``box_box_detection``)
        plus a few neighbours. So the flat default is *not* the bottleneck it
        looks like, and scaling it aggressively is actively harmful: the
        dominant GPU allocation is the constraint Jacobian, which is
        ``O(max_collision_pairs x contacts_per_pair x n_dofs x n_envs)``, so an
        oversized cap directly costs parallel environments. Raw step time, by
        contrast, is independent of it (measured flat across 150/800 at both
        settings of ``box_box_detection``).

        Measured requirement is close to ``0.26 * n_particles`` (13 at n=50, 52
        at n=200), so Genesis' flat 150 already carries ~2.8x headroom at 200
        particles and only needs to grow past roughly n=570. Hence the gentle
        ``n/2`` scaling below Genesis' floor. The difference is not academic:
        at n_particles=200 a cap of 200 tops out at 16 parallel envs on an 8 GB
        card, while 150 fits 32 — the cap alone doubles throughput.

        Under-estimating is caught loudly by ``_check_contact_budget`` rather
        than silently corrupting contacts, which is the failure mode that
        matters: on overflow the broadphase sets an error bit and stops adding
        pairs, and that bit never surfaces here (see ``_check_contact_budget``).
        """
        n_particles = int(self._material_params.get("n_particles") or 0)
        return max(150, n_particles // 2)

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
        # Exposed so the constraint solver can be swapped without editing this
        # file. Genesis defaults to Newton, which with use_contact_island builds
        # and factorizes a DENSE Hessian per contact island (island_dofs = 6 x
        # entities in the island); measured cost goes as island_size^2.64, and
        # that is the whole explanation for the cost cliff at 200 objects (see
        # docs/scaling_to_200_objects.md section 8.7).
        #
        # "CG" would avoid the dense Hessian entirely and is the obvious escape,
        # but it is BROKEN in Genesis 0.4.5: the kernel references
        # RigidSolver.func_solve_mass_batch, which does not exist, and every
        # scene using it fails at compile time. Left exposed anyway so it can be
        # re-tested against a newer Genesis without touching this file.
        # Default preserves the previous behaviour exactly.
        _cs_name = rigid_cfg.get("constraint_solver", "Newton")
        try:
            _constraint_solver = getattr(gs.constraint_solver, _cs_name)
        except AttributeError as e:
            raise ValueError(
                f"rigid_options.constraint_solver={_cs_name!r} is not a Genesis "
                f"solver; expected one of "
                f"{[n for n in dir(gs.constraint_solver) if not n.startswith('_')]}"
            ) from e
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt       = self._config["simulation"].get('dt', 4e-3),
                substeps = self._config["simulation"].get('substeps', 5),
            ),
            rigid_options=gs.options.RigidOptions(
                iterations=rigid_cfg.get("iterations", 50),
                ls_iterations=rigid_cfg.get("ls_iterations", 50),
                tolerance=rigid_cfg.get("tolerance", 1e-6),
                ls_tolerance=rigid_cfg.get("ls_tolerance", 0.01),
                box_box_detection=rigid_cfg.get("box_box_detection", False),
                use_contact_island=rigid_cfg.get("use_contact_island", False),
                use_hibernation=rigid_cfg.get("use_hibernation", False),
                max_collision_pairs=rigid_cfg.get(
                    "max_collision_pairs", self._default_max_collision_pairs()),
                enable_multi_contact=rigid_cfg.get("enable_multi_contact", True),
                constraint_solver=_constraint_solver,
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
        #
        # friction must be set explicitly: Genesis defaults an unset geom
        # friction to 1.0 and combines a contact as max(mu_a, mu_b), so
        # leaving it None pins *every* plate-particle contact at 1.0 and makes
        # the sampled particle friction have no effect whatsoever at the tool
        # interface — the one interface the action actually acts through.
        self.plate = self._scene.add_entity(
            material=gs.materials.Rigid(
                rho=3000,
                friction=float(self._plate_params.get("friction", 0.3)),
            ),
            morph=gs.morphs.Box(
                pos=(0, 0, height * 2),
                size=self._plate_params["size"]
            ),
            surface=gs.surfaces.Default(color=[0, 1, 0])
        )
        
        # add granular
        # Read from config rather than hardcoded: the key existed in basic.yaml
        # and was silently ignored, so anyone tuning it got no effect at all.
        # The default is the value that was actually in force.
        self._safety_margin = self._config.get("safety_margin", 0.02)

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
        self._configure_plate_actuator()

        self._cache_particle_idx()

    def _configure_plate_actuator(self) -> None:
        """Model the pusher as a gantry axis rather than a free light box.

        Three things, all on the translational dofs (rotation is hard-set every
        step by the sweep/descent loops, so its gains are irrelevant):

        armature
            The plate geometry weighs ~2.4 g, which is far lighter than the
            carriage that actually carries it, so granular reaction would move
            it much more than on the real machine. ``set_dofs_armature`` adds
            the drivetrain's reflected inertia to the mass-matrix diagonal —
            the same matrix the constraint solver uses — so contacts see a
            heavy axis while momentum exchange stays exact. This is the right
            knob rather than a denser plate, which would also change the tool's
            weight and its contact response.

        gains
            Chosen from the modelled mass and a target closed-loop bandwidth:
            kp = m*w^2, kv = 2*z*m*w at z = 1 (critically damped). The default
            15 Hz gives w ~ 94 rad/s against a 0.8 ms substep (w*h ~ 0.075),
            comfortably stable, and a disturbance stiffness of kp ~ 4.4e3 N/m —
            a couple of newtons of granular reaction displaces the tool well
            under half a millimetre.

        force range
            Previously unbounded. With stiff gains a particle wedged against a
            wall would draw an arbitrarily large force; a real stepper loses
            steps instead. A finite budget makes a jam degrade gracefully.
        """
        translational = [0, 1, 2]
        m = self._plate_moving_mass
        omega = 2.0 * math.pi * self._plate_bandwidth
        kp, kv = m * omega ** 2, 2.0 * m * omega

        self.plate.set_dofs_armature((m,) * 3, translational)
        self.plate.set_dofs_kp((kp,) * 3, translational)
        self.plate.set_dofs_kv((kv,) * 3, translational)
        self.plate.set_dofs_force_range(
            (-self._plate_max_force,) * 3, (self._plate_max_force,) * 3,
            translational)

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

    # Genesis' RHO_OBJECT — the density a rigid link resolves to when its
    # material leaves rho unset (genesis/engine/entities/rigid_entity/
    # rigid_link.py). Particles are constructed without an explicit material
    # (see utilities/materials.py), so this, not the configured density, is
    # what their mass is actually built from.
    _GENESIS_DEFAULT_RHO = 600.0

    def _set_particle_density_value(self, particle, density: float):
        # Rescale mass from whatever density the particle's mass currently
        # reflects. On the first call material.rho is still None — meaning the
        # built mass came from Genesis' default rho, not from any configured
        # value — so fall back to that default rather than skipping the update.
        # Skipping it (the previous behaviour) left every particle at the
        # default density while the saved config recorded the sampled one, and
        # every subsequent batch then rescaled from the wrong base, leaving all
        # masses at 600/750 = 0.8x their recorded density.
        old_density = getattr(particle.material, "rho", None) or self._GENESIS_DEFAULT_RHO
        particle.material.rho = float(density)
        if getattr(self._scene, "is_built", False) and old_density > 0:
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
        # Safety net: flush any transitions recorded since the last shuffle
        # (i.e. the episode that's about to be overwritten) before wiping
        # particle state for a new configuration, so buffered data is never
        # silently lost even if a real step's flush_after=True was somehow
        # skipped. No-op if recording is disabled or nothing is buffered
        # (e.g. Genesis/data_collection_clean.py's own flow never calls
        # push_and_record, so this stays a harmless no-op there). Then clear
        # the transition context — a new episode is starting, and the caller
        # (env.reset()) is expected to set a fresh one via
        # set_transition_context() right after this returns.
        self.flush_transitions()
        self.set_transition_context(None)

        n_particles = len(self.material)
        if n_particles == 0:
            return
        n_active = getattr(self, '_n_active', n_particles)

        # Number of stacked layers to spread the particles over on respawn.
        # 1 reproduces the original single-layer behaviour exactly and is
        # always tried first; it is only incremented when a layer genuinely
        # cannot be packed (see the retry handler below), which is what makes
        # particle counts above the box's single-layer RSA capacity
        # (~140 cubes of 5 mm in a 128 mm box) reachable at all. Stacked
        # layers are dropped, not interpenetrating: the caller's subsequent
        # update_material_state() settle collapses them into a natural pile.
        n_layers = 1
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
                order = torch.argsort(torch.prod(half_extents, dim=1), descending=True)
                order = order[order < n_active]  # only place the first n_active particles
                # Candidate draws are split across more, smaller rounds rather
                # than few large ones. The rejection test materializes an
                # (active_envs, candidate_batch, n_placed, 2) intermediate, so
                # a 4096-wide batch costs ~420 MB transiently at n_envs=32 /
                # n_particles=200 — allocated and freed once per particle,
                # which is enough to OOM alongside the solver. Rounds x batch
                # is kept at the previous total, so placement success is
                # unchanged; only the peak allocation drops (4x).
                candidate_batch = max(256, min(1024, 64 * max(n_active, 1)))
                candidate_rounds = 512
                min_gap = 1e-3

                # Vertical pitch between stacked layers — tall enough that the
                # tallest particle in a layer clears the layer below it.
                layer_pitch = float(2.0 * half_extents[:, 2].max().item()) + min_gap
                top_of_stack = float(inner_min[2]) + min_gap + n_layers * layer_pitch
                if top_of_stack > float(inner_max[2]):
                    raise RuntimeError(
                        f"Cannot spawn {n_active} particles: {n_layers} stacked layers "
                        f"need {top_of_stack:.4f} m of box interior but only "
                        f"{float(inner_max[2]):.4f} m is available (box walls are "
                        f"{self._box_params['vol'][2]:.3f} m tall). Use smaller "
                        f"particles, a taller/wider box, or fewer particles."
                    )

                # Split the largest-first placement order across layers (strided,
                # so each layer gets a comparable size mix) and pack each layer
                # independently: overlap is only a constraint within a layer,
                # since layers are vertically separated.
                for layer_idx in range(n_layers):
                    layer_order = order[layer_idx::n_layers]
                    placed = torch.zeros(n_particles, dtype=torch.bool, device=gs.device)
                    layer_z = inner_min[2] + min_gap + layer_idx * layer_pitch
                    for particle_idx_tensor in layer_order:
                        particle_idx = int(particle_idx_tensor.item())
                        active = torch.ones(self._n_envs, dtype=torch.bool, device=gs.device)
                        span_xy = upper[particle_idx, :2] - lower[particle_idx, :2]
                        z_pos = layer_z + half_extents[particle_idx, 2]
                        for _ in range(candidate_rounds):
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
                # Move parked (inactive) particles outside the box, spread over
                # a grid rather than heaped on one point: parking them all at
                # an identical position piles every inactive particle into a
                # single permanent contact cluster, which consumes the contact
                # budget (see _default_max_collision_pairs) and costs solver
                # time on every step of every env, for particles that are not
                # even part of the experiment.
                if n_active < n_particles:
                    n_parked = n_particles - n_active
                    park = torch.tensor(self._park_pos, dtype=torch.float32, device=gs.device)
                    pitch = float(2.0 * half_extents[:, :2].max().item()) + 5e-3
                    cols = int(math.ceil(math.sqrt(n_parked)))
                    idx = torch.arange(n_parked, device=gs.device)
                    offsets = torch.zeros((n_parked, 3), device=gs.device)
                    offsets[:, 0] = (idx % cols).to(torch.float32) * pitch
                    offsets[:, 1] = torch.div(idx, cols, rounding_mode="floor").to(torch.float32) * pitch
                    positions[:, n_active:, :] = (park.view(1, 3) + offsets).unsqueeze(0).expand(
                        self._n_envs, n_parked, 3)
                self._write_particle_poses(
                    positions, self._random_particle_quats_batched(), envs_idx)
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
                    # RSA placement is stochastic, so give the current layer
                    # count one more roll of the dice before concluding the
                    # box is genuinely too full and adding a layer.
                    if attempt % 2 == 1:
                        n_layers += 1
                    print(
                        f"Placement of particles failed due to overlap, retrying "
                        f"{attempt+1}/{max_retries} with {n_layers} layer(s)..."
                    )
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"Could not randomly shuffle particles without overlap after {max_retries} attempts. "
                            "Try a smaller particle size or fewer particles."
                        )
                    # else, try again
                    continue
                else:
                    raise

    def _write_particle_poses(self, pos: torch.Tensor, quat: torch.Tensor,
                              envs_idx: torch.Tensor) -> None:
        """Write every particle's pose in two batched calls instead of 2N.

        ``RigidEntity.set_pos``/``set_quat`` each run a forward-kinematics pass
        over the *whole scene* (``skip_forward=False`` by default), so the
        obvious per-particle loop costs 2N kernel launches and 2N full-scene FK
        passes — 400 of each at n_particles=200, on every reset and every
        snapshot restore (the latter being in the oracle-MPC hot path). The
        solver-level setters take a link-index array, so the same work is two
        launches and a single FK pass at the end.

        pos  : (n_envs, n_particles, 3)
        quat : (n_envs, n_particles, 4)
        """
        solver = self._scene.rigid_solver
        solver.set_base_links_pos(pos, links_idx=self._particle_links_idx,
                                  envs_idx=envs_idx, skip_forward=True)
        solver.set_base_links_quat(quat, links_idx=self._particle_links_idx,
                                   envs_idx=envs_idx, skip_forward=False)

    def _random_particle_quats_batched(self) -> torch.Tensor:
        """Random spawn orientation for every particle, in one shot.

        Returns (n_envs, n_particles, 4). Particles whose morph carries no
        orientable extent (spheres) keep the identity quaternion; the rest get
        a uniform random yaw, which for roll=pitch=0 reduces to
        (cos(yaw/2), 0, 0, sin(yaw/2)).
        """
        n_particles = len(self.material)
        orientable = torch.tensor(
            [hasattr(p.morph, "size") or hasattr(p.morph, "height")
             for p in self.material],
            dtype=torch.bool, device=gs.device,
        )
        yaw = torch.rand((self._n_envs, n_particles), device=gs.device) * math.tau
        yaw = torch.where(orientable.unsqueeze(0), yaw, torch.zeros_like(yaw))
        half = yaw * 0.5
        quat = torch.zeros((self._n_envs, n_particles, 4), device=gs.device)
        quat[:, :, 0] = torch.cos(half)
        quat[:, :, 3] = torch.sin(half)
        return quat

    def _get_particle_positions(self):
        return self._scene.rigid_solver.get_links_pos(links_idx=self._particle_links_idx)

    def _get_particle_quats(self):
        return self._scene.rigid_solver.get_links_quat(links_idx=self._particle_links_idx)

    def update_material_state(self, store_other=False, on_step=None):
        """
        Returns particle state (positions and sizes) for all environments.
        Optimized for GPU processing.

        Args:
            on_step: optional callback(step), invoked after each settle step.
                Same hook convention as ``execute_action``'s ``on_phase``. The
                settle is the one phase with no other window into it — it exits
                on a convergence test, so its duration is not known in advance
                and nothing outside this loop can observe the pile collapsing.
                Used by ``tests/scaling_investigation/record_simulation_video.py``
                to render the layered spawn, which is the least physically
                natural moment in the pipeline and therefore the one most worth
                watching. A no-op when None.

        Returns:
            Tensor of shape [n_envs, n_particles, 4] with (x, y, z, size)
        """

        # Hold the plate still for the duration of the settle.
        #
        # The plate is lifted clear of the pile here, so the only force acting
        # on it is its own 2.4 g weight — 0.0235 N against a translational
        # kp of 4441 N/m, i.e. a steady-state sag of 5.3 um, about 0.1% of a
        # 5 mm particle. The PD alone therefore pins it, and a per-step
        # set_dofs_position is not just redundant but actively harmful:
        # RigidSolver.set_dofs_position calls collider.reset() AND
        # constraint_solver.reset() — discarding the constraint solver's warm
        # start every step, with only 10 iterations to rebuild it — runs a
        # whole-scene forward-kinematics pass, and clears _errno, which is why
        # a contact-budget overflow could never surface. Setting the control
        # target once is enough: ctrl_pos persists (it is only cleared by
        # control_dofs_velocity, a mode switch) and the actuator reads it every
        # substep.
        frozen_plate_dofs = self.plate.get_dofs_position()
        self.plate.zero_all_dofs_velocity()
        self.plate.control_dofs_position_velocity(
            frozen_plate_dofs,
            torch.zeros_like(frozen_plate_dofs),
            dofs_idx_local=[0, 1, 2, 3, 4, 5],
        )

        settled_at = None
        for step in range(self._settle_steps):
            self._step_scene()
            if on_step is not None:
                on_step(step)
            if (step + 1) % self._settle_check_every == 0 and self._pile_is_at_rest():
                settled_at = step + 1
                break

        if settled_at is None and not getattr(self, "_settle_cap_warned", False):
            lin_max, ang_max = self._pile_motion()
            lin_q, ang_q = self._pile_motion(quantile=self._settle_rest_quantile)
            self._settle_cap_warned = True
            self._log(
                f"WARNING: pile still moving after the full {self._settle_steps}-step "
                f"settle. At the q={self._settle_rest_quantile} rest quantile: "
                f"{lin_q*1000:.2f} mm/s linear, {ang_q:.2f} rad/s angular (thresholds "
                f"{self._settle_vel_threshold*1000:.1f} / "
                f"{self._settle_angvel_threshold:.1f}); worst single particle "
                f"{lin_max*1000:.1f} mm/s, {ang_max:.1f} rad/s. The recorded state is "
                f"mid-motion, and because each transition's s comes from the previous "
                f"s', that error propagates. Raise simulation.settle_steps, or relax "
                f"simulation.settle_rest_quantile if the tail is a few stragglers."
            )
        elif self._debug and settled_at is not None:
            self._log(f"settled after {settled_at}/{self._settle_steps} steps")

        self._check_contact_budget()

        self._particle_state[:, :, 0:3] = self._get_particle_positions()
        self._particle_state[:, :, 3:] = self._get_particle_quats()

    def _pile_motion(self, quantile: float | None = None) -> tuple[float, float]:
        """(linear m/s, angular rad/s) particle speed, peak or at a quantile.

        Linear and angular are kept separate rather than reduced to one number:
        a free joint's dofs are [x, y, z, roll, pitch, yaw], so both live in the
        same tensor but carry different units, and a single max over all six
        conflates metres per second with radians per second — which reads as an
        alarming velocity when it is really a mildly spinning cube.

        ``quantile=None`` gives the peak. A quantile is what the rest test
        actually wants: the peak is taken over *every particle in every env*, so
        its strictness scales with n_envs. At 32 envs x 200 particles a single
        straggler anywhere holds up all 6400, and the settle then always runs to
        its cap — which is exactly what happened before this was quantile-based.
        """
        if self._particle_dofs_idx.numel() == 0:
            return 0.0, 0.0
        vel = self._scene.rigid_solver.get_dofs_velocity(
            dofs_idx=self._particle_dofs_idx).reshape(self._n_envs, -1, 6)
        n_active = getattr(self, "_n_active", vel.shape[1])
        vel = vel[:, :n_active]
        lin = vel[..., :3].norm(dim=-1).flatten()
        ang = vel[..., 3:].norm(dim=-1).flatten()
        if quantile is None:
            return float(lin.max()), float(ang.max())
        q = torch.tensor(quantile, device=lin.device, dtype=lin.dtype)
        return float(torch.quantile(lin, q)), float(torch.quantile(ang, q))

    def _pile_is_at_rest(self) -> bool:
        lin, ang = self._pile_motion(quantile=self._settle_rest_quantile)
        return (lin < self._settle_vel_threshold
                and ang < self._settle_angvel_threshold)

    def _check_contact_budget(self) -> None:
        """Warn (once) if the pile is close to exhausting the contact budget.

        Genesis reports contact-pair overflow by setting an error bit that
        ``Simulator.step`` inspects periodically. That mechanism cannot fire
        here: ``RigidSolver.set_dofs_position`` clears the error bit as a side
        effect, and both this settle loop and the sweep loop call it on every
        step, so the bit is always wiped before the next check reads it. The
        failure would therefore be completely silent — contacts dropped, wrong
        physics recorded, no exception — which is exactly the kind of thing
        that must not go unnoticed in collected training data. So check the
        counter directly instead of relying on Genesis to complain.
        """
        if getattr(self, "_contact_budget_warned", False):
            return
        try:
            usage = self.contact_budget_usage()
        except Exception:
            self._contact_budget_warned = True   # counters unavailable; don't retry
            return
        for what, used, cap in (
            ("broad-phase candidate pairs", usage["broad_pairs"], usage["broad_cap"]),
            ("contact points", usage["contact_points"], usage["contact_cap"]),
        ):
            if used >= 0.9 * cap:
                self._contact_budget_warned = True
                self._log(
                    f"WARNING: {used}/{cap} {what} in use. Past the cap Genesis "
                    f"stops adding contacts and only flags it via an error bit "
                    f"that this class's per-step set_dofs_position clears before "
                    f"it can be read — so an overflow here is silent, and the "
                    f"recorded state would come from incomplete contact physics. "
                    f"Raise rigid_options.max_collision_pairs."
                )

    def contact_budget_usage(self) -> dict:
        """Peak collider occupancy across envs, against its two real limits.

        Genesis bounds collision work in two independent places, and
        ``max_collision_pairs`` (``mcp``) sets both:

        * broad-phase candidate pairs, capped at ``mcp * 8``
          (``multiplier_collision_broad_phase``)
        * narrow-phase contact *points*, capped at
          ``mcp * n_contacts_per_pair`` — where ``n_contacts_per_pair`` is 5
          normally but 16 once ``box_box_detection`` is on with more than one
          box, which is this scene's configuration

        The pair count and the point count differ by a large factor, so they
        must be compared against their own caps — a settled pile of cubes
        produces roughly four contact points per floor contact.
        """
        collider = self._scene.rigid_solver.collider
        state, info = collider._collider_state, collider._collider_info
        broad_cap = int(torch.as_tensor(info.max_collision_pairs_broad.to_torch()).max())
        mcp = int(torch.as_tensor(info.max_collision_pairs.to_torch()).max())

        # How the point cap is obtained differs by Genesis version, so read it
        # rather than assume it. From 1.2.x the collider publishes `max_contacts`
        # directly, and it is no longer a simple multiple: the buffer is sized
        # per regime (convex vs nonconvex pairs have different per-pair caps)
        # and then reduced again by link-pair contact pruning. Recomputing
        # `mcp * n_contacts_per_pair` there would overstate the cap and hide a
        # real overflow, which is the one thing this check exists to catch.
        ncp = None
        if hasattr(info, "max_contacts"):
            contact_cap = int(torch.as_tensor(info.max_contacts.to_torch()).max())
        else:                                    # 0.4.5 and earlier
            ncp = int(collider._collider_static_config.n_contacts_per_pair)
            contact_cap = mcp * ncp
        return {
            "broad_pairs": int(torch.as_tensor(state.n_broad_pairs.to_torch()).max()),
            "broad_cap": broad_cap,
            "contact_points": int(torch.as_tensor(state.n_contacts.to_torch()).max()),
            "contact_cap": contact_cap,
            "max_collision_pairs": mcp,
            "n_contacts_per_pair": ncp,
        }

    def escaped_particle_count(self) -> int:
        """Particles outside the tray interior, summed over all envs.

        A particle can only leave by being squeezed through a wall, which means
        the contact solver failed for it. It matters more than it sounds: each
        transition's ``s`` is the previous transition's ``s'``, so one escape
        silently corrupts every later sample in that env, and nothing else in
        the pipeline would notice. Recorded per batch alongside the collected
        data so a finished dataset can be audited without re-running it.

        The tolerance is deliberately loose (5 mm laterally, 20 mm above the
        wall) — this is looking for particles that have plainly left, not for
        ones resting slightly proud of the rim.
        """
        pos = self._get_particle_positions()[:, :getattr(self, "_n_active", None)]
        width, depth, height = self._box_params["vol"]
        half = torch.tensor([width / 2, depth / 2], device=pos.device)
        out_xy = (pos[..., :2].abs() > half + 0.005).any(dim=-1)
        out_z = (pos[..., 2] < -0.005) | (pos[..., 2] > height + 0.02)
        return int((out_xy | out_z).sum())

    def set_particle_state(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """
        Set every environment's particle pose directly from given tensors,
        and zero particle velocities.

        Parameters
        ----------
        pos  : (1, n_particles, 3) broadcast to all ``n_envs``, or
               (n_envs, n_particles, 3) to give each env a different state.
        quat : (1, n_particles, 4) or (n_envs, n_particles, 4), likewise.

        Lower-level primitive underlying ``broadcast_state_from_env``.
        Planners that must keep an *immutable* reference state across
        several rollouts (see
        ``simple_mpc.genesis_oracle.GenesisOracleEnv.snapshot_particles`` /
        ``restore_snapshot``) should call this directly with a frozen
        snapshot, rather than ``broadcast_state_from_env``, whose source
        (``self._particle_state[src_env]``) can itself have been overwritten
        by an intervening rollout. Does not touch the plate:
        ``execute_action`` teleports the plate to its start pose on every
        call, so no explicit plate reset is needed between rollouts.
        """
        envs_idx = torch.arange(self._n_envs, device=gs.device)
        self._write_particle_poses(
            pos.expand(self._n_envs, -1, -1).contiguous(),
            quat.expand(self._n_envs, -1, -1).contiguous(),
            envs_idx,
        )
        if self._particle_dofs_idx.numel() > 0:
            self._scene.rigid_solver.set_dofs_velocity(
                torch.zeros((self._n_envs, self._particle_dofs_idx.numel()), device=gs.device),
                dofs_idx=self._particle_dofs_idx,
                skip_forward=True,
            )
        self._particle_state[:, :, 0:3] = pos.expand(self._n_envs, -1, -1)
        self._particle_state[:, :, 3:7] = quat.expand(self._n_envs, -1, -1)

    def broadcast_state_from_env(self, src_env: int = 0) -> None:
        """
        Copy particle pose (position + quaternion) from environment
        ``src_env``'s CURRENT (live) state to every environment.

        Used for one-off resyncs (e.g. after ``shuffle_particles()``) where
        ``src_env``'s live state is known to be the correct reference. For
        repeated use across several rollouts where ``src_env`` might itself
        be mutated in between (e.g. env 0 also plays the role of a rollout
        worker during planning — see ``GenesisOracleEnv``), capture a
        snapshot once with ``set_particle_state``'s inputs saved externally
        instead of relying on this method's live read.
        """
        self.set_particle_state(
            self._particle_state[src_env:src_env + 1, :, 0:3],
            self._particle_state[src_env:src_env + 1, :, 3:7],
        )

    def plate_velocity_translation(
            self,
            p_start,
            p_end,
            angle,
            debug=False,
            on_step=None,
        ):
        """
        Move plates along a trapezoidal speed profile across all environments.

        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            angle: Rotation angle (scalar)
            on_step: optional callback(step, p_ref, v_ref), invoked after each
                simulation step of the sweep with the reference the servo was
                tracking at that step. A no-op when None. Follows the same
                hook convention as ``execute_action``'s ``on_phase`` (see
                docs/UTILITIES.md) — it exists so diagnostics such as
                ``tests/scaling_investigation/probe_plate_dynamics.py`` can measure the tool's
                realized trajectory against its reference without duplicating
                this control law, which is exactly the kind of drift that
                makes a probe silently stop testing the thing it names.
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

        delta = p_end - p_start                                   # [n_envs, 3]
        dist = torch.linalg.norm(delta, axis=1)                   # [n_envs]
        direction = delta / (dist.unsqueeze(1) + 1e-8)

        prof = self._trapezoid_profile(dist)
        dt = self._scene.dt
        sweep_steps = max(
            1, math.ceil(float(prof["duration"].max().item()) / dt)
        ) + self._sweep_settle_steps

        self.plate.set_pos(p_start)

        for step in range(sweep_steps):
            # Feed the servo a *moving* reference: where the tool should be and
            # how fast it should be going right now. Commanding the endpoint
            # instead (the previous behaviour) turns the same PD into a
            # position servo whose speed is proportional to distance remaining
            # -- it settles at v = v_cruise + kp*remaining/kv, so it overshoots
            # the commanded speed early in a sweep and undershoots near the
            # goal, and never actually travels at plate.speed.
            s, v_mag = self._trapezoid_at(prof, (step + 1) * dt)
            p_ref = p_start + direction * s.unsqueeze(1)
            v_ref = direction * v_mag.unsqueeze(1)
            self.plate.control_dofs_position_velocity(
                p_ref, v_ref, dofs_idx_local=[0, 1, 2])

            # zero_velocity=False: this call constrains z/roll/pitch/yaw only,
            # but RigidEntity.set_dofs_position defaults zero_velocity=True and
            # zeroes *all six* dofs regardless of dofs_idx_local. Leaving the
            # default on reset the plate's x/y velocity every single step, so
            # the sweep restarted from rest at 250 Hz and the tool carried no
            # momentum into the pile.
            self.plate.set_dofs_position(
                self._horizontal_dof_fix,
                dofs_idx_local=self._horizontal_dofs_local,
                zero_velocity=False,
            )
            self._step_scene()
            if on_step is not None:
                on_step(step, p_ref, v_ref)

        # No per-step goal test: the reference itself ends at p_end and holds
        # there, so envs that finish early simply stop, with no freeze
        # bookkeeping. That also removes the two GPU syncs the old loop paid on
        # every step (a .nonzero() and a .item()), which dominated the sweep's
        # per-step cost at small n_envs.
        # Check the contact budget HERE, not only after settling: the pile is
        # most compressed at the end of a sweep, so this is where usage peaks.
        # It also cannot be left to Genesis' own error bit, because the loop
        # above calls set_dofs_position every step and that clears _errno. An
        # unnoticed overflow does not degrade gracefully — with the point cap
        # exceeded it has been observed to corrupt memory outright (CUDA
        # illegal memory access).
        self._check_contact_budget()

        final_pos = self.plate.get_pos()
        final_err = torch.linalg.norm(final_pos[:, :2] - p_end[:, :2], axis=1)
        reached_goal = final_err < self._goal_threshold

        if self._debug:
            print(
                f" > Goal reached : {int(reached_goal.sum().item())}/{self._n_envs}; "
                f" > Final tracking error {float(final_err.min().item()):.4f}-"
                f"{float(final_err.max().item()):.4f}m over {sweep_steps} steps"
            )

        return reached_goal, final_pos

    def _trapezoid_profile(self, dist: torch.Tensor) -> dict:
        """Pre-compute a trapezoidal speed profile per env for a given travel.

        Matches how the real gantry moves: accelerate at ``plate.acceleration``
        to ``plate.speed``, cruise, then decelerate to rest exactly at the
        target. Short moves that never reach cruise speed degenerate to a
        triangular profile with peak sqrt(a*d), handled by the same expression.
        """
        v_max = float(self._plate_params["speed"])
        a = self._plate_accel
        v_peak = torch.clamp(torch.sqrt(a * dist.clamp(min=0.0)), max=v_max)
        t_acc = v_peak / a
        d_acc = 0.5 * a * t_acc ** 2
        d_flat = torch.clamp(dist - 2.0 * d_acc, min=0.0)
        t_flat = d_flat / v_peak.clamp(min=1e-9)
        return {
            "dist": dist, "a": a, "v_peak": v_peak,
            "t_acc": t_acc, "d_acc": d_acc, "t_flat": t_flat, "d_flat": d_flat,
            "duration": 2.0 * t_acc + t_flat,
        }

    def _trapezoid_at(self, prof: dict, t: float):
        """Distance travelled and speed at time ``t``, per env."""
        a, v_peak = prof["a"], prof["v_peak"]
        t_acc, t_flat, d_acc, d_flat = (
            prof["t_acc"], prof["t_flat"], prof["d_acc"], prof["d_flat"])
        t_cruise_end = t_acc + t_flat
        tc = torch.clamp(torch.full_like(v_peak, float(t)),
                         max=prof["duration"])

        t_dec = torch.clamp(tc - t_cruise_end, min=0.0)
        s_acc = 0.5 * a * torch.minimum(tc, t_acc) ** 2
        s_flat = v_peak * torch.minimum(torch.clamp(tc - t_acc, min=0.0), t_flat)
        s_dec = v_peak * t_dec - 0.5 * a * t_dec ** 2

        s = s_acc + s_flat + s_dec
        v = torch.where(
            tc <= t_acc, a * tc,
            torch.where(tc <= t_cruise_end, v_peak, v_peak - a * t_dec),
        )
        return s, torch.clamp(v, min=0.0)
    
    def plate_position_translation(self, p_start, p_end, n_steps: int | None = None,
                                   on_step=None):
        """
        Move plates with position control across all environments.

        Args:
            p_start: Starting positions [n_envs, 3] or [3]
            p_end: Ending positions [n_envs, 3] or [3]
            n_steps: Override step count (defaults to self._pos_ctrl_steps)
            on_step: optional callback(step), invoked after each step. This is
                the descent and the lift — the phases where the tool is moved
                by teleport-then-interpolate rather than by the servo, so if it
                ever passes through a particle it happens here. A no-op when
                None.
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
            if on_step is not None:
                on_step(i)

    def generate_action_samples(
            self,
            n_samples: int,
            placement_aware: bool = False,
            placement_resolution: float = 0.001,
            placement_angles: int = 16,
            placement_clearance: float = 0.0,
            placement_clearance_bias: float = 0.0,
            shared_travel_distance: bool = False,
        ):
        """
        Generate random action samples for all environments.

        Args:
            n_samples: samples per environment.
            placement_aware: choose ``p_start`` (and its yaw) from the tool's
                *free configuration space* instead of blindly from the box
                interior, so the plate does not descend into a particle. See
                Genesis/placement_sampling.py. Falls back to the blind draw,
                per sample, wherever no collision-free placement exists —
                which is the expected outcome once the pile covers enough of
                the tray, and is why this is a refinement rather than a
                replacement.
            placement_resolution: occupancy/C-space grid cell size, metres.
            placement_angles: number of yaw bins the free set is computed for.
            placement_clearance: extra margin added around the tool footprint.
            placement_clearance_bias: >0 biases the draw toward placements with
                more room around them (weight = clearance ** bias).
            shared_travel_distance: give every env the same push length for a
                given sample, keeping its own start, direction and yaw. Envs
                step in lockstep and the sweep is sized from the longest travel
                in the batch, so independent distances make every env run for
                the longest one's duration — measured at 1.54x of a 2.64x
                batching penalty at 8 envs. Off by default so single-env and
                MPC callers are unaffected; collection turns it on.

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

        if placement_aware:
            action_starts, angles = self._apply_placement_aware_starts(
                action_starts, angles, n_samples,
                resolution=placement_resolution,
                n_angles=placement_angles,
                clearance=placement_clearance,
                clearance_bias=placement_clearance_bias,
            )

        if shared_travel_distance and self._n_envs > 1:
            action_stops = self._equalize_batch_travel(
                action_starts, action_stops,
                low.reshape(self._n_envs, n_samples, 2),
                high.reshape(self._n_envs, n_samples, 2))

        return action_starts, action_stops, angles

    def _equalize_batch_travel(self, action_starts, action_stops, low, high):
        """Give every env in a batch the same push length for each sample.

        Envs step in lockstep and ``plate_velocity_translation`` sizes the sweep
        from the LONGEST travel in the batch, so one long push makes every env
        run for its duration. Sharing the distance removes that coupling — worth
        1.54x of a measured 2.64x batching penalty at 8 envs (see
        Genesis/action_sampling.py). Start point, direction and blade yaw stay
        per-env, and the distance still varies from batch to batch, so only the
        within-batch spread is given up.
        """
        from .action_sampling import equalize_travel_distance, shared_batch_distance

        starts_xy, stops_xy = action_starts[..., :2], action_stops[..., :2]
        dist = (stops_xy - starts_xy).norm(dim=-1, keepdim=True)
        target = shared_batch_distance(dist).expand_as(dist)
        new_xy, clipped = equalize_travel_distance(
            starts_xy, stops_xy, low, high, target)
        if self._debug and bool(clipped.any()):
            self._log(f"shared travel distance: {int(clipped.sum())}/"
                      f"{clipped.numel()} pushes truncated at the box boundary")
        return torch.cat((new_xy, action_stops[..., 2:]), dim=-1)

    def _apply_placement_aware_starts(self, action_starts, angles, n_samples, *,
                                      resolution, n_angles, clearance,
                                      clearance_bias):
        """Replace blindly-drawn touchdown poses with collision-free ones.

        Only ``p_start`` and the yaw are overridden — the sweep target is left
        alone, since the tool is *supposed* to run into particles once it is
        down; what must be avoided is materializing inside one on the way down.

        Any (env, sample) for which the free set is empty keeps its blind draw,
        so a fully-covered tray degrades to the previous behaviour instead of
        failing.
        """
        from .placement_sampling import (build_occupancy, clearance_map,
                                         free_placements, sample_free_placements)

        tool_length, tool_width, _ = self._plate_params["size"]
        sizes = self._sampled_params.get("particle_sizes", None)
        if sizes is None:
            sizes = [p.morph.size if hasattr(p.morph, "size")
                     else (p.morph.radius * 2,) * 3 for p in self.material]
        half_xy = torch.as_tensor(sizes, dtype=torch.float32,
                                  device=gs.device)[:, :2] * 0.5
        # a cube free to take any yaw sweeps out sqrt(2) of its side
        is_cube = torch.tensor([hasattr(p.morph, "size") for p in self.material],
                               dtype=torch.float32, device=gs.device)
        half_xy = half_xy * (1.0 + (math.sqrt(2) - 1.0) * is_cube).unsqueeze(1)

        try:
            occ, meta = build_occupancy(
                self._particle_state[:, :, 0:3], half_xy,
                (self._box_params["vol"][0], self._box_params["vol"][1]),
                resolution, active=getattr(self, "_n_active", None))
            yaw_bins = (-torch.pi / 2) + torch.arange(
                n_angles, device=gs.device, dtype=torch.float32) * (torch.pi / n_angles)
            free = free_placements(occ, meta, yaw_bins, tool_length, tool_width,
                                   clearance=clearance,
                                   wall_margin=self._safety_margin)
            dist = clearance_map(occ, meta) if clearance_bias > 0 else None
            xy, yaw, ok = sample_free_placements(
                free, meta, yaw_bins, n_samples, clearance=dist,
                clearance_bias=clearance_bias)
        except Exception as e:                      # never block collection
            self._log(f"placement-aware sampling unavailable ({e}); "
                      f"falling back to blind sampling")
            return action_starts, angles

        n_free = int(ok.sum().item())
        if n_free == 0:
            self._log("placement-aware sampling found no collision-free tool "
                      "placement; falling back to blind sampling")
            return action_starts, angles
        if self._debug and n_free < ok.numel():
            self._log(f"placement-aware: {n_free}/{ok.numel()} samples placed "
                      f"in free space, rest fell back to blind")

        starts = action_starts.clone()
        starts[..., 0] = torch.where(ok, xy[..., 0], starts[..., 0])
        starts[..., 1] = torch.where(ok, xy[..., 1], starts[..., 1])
        return starts, torch.where(ok, yaw, angles)

    def execute_action(
            self,
            p_start,
            p_stop,
            angle,
            on_phase=None,
            on_step=None,
        ):
        """
        Execute action (lower, sweep, lift) for all environments.

        Args:
            p_start: Starting positions [n_envs, 3]
            p_stop: Stopping positions [n_envs, 3]
            angle: Angles [n_envs]
            lift_height: Lift height [n_envs, 3]
            on_phase: optional callback(phase: str), invoked at two points
                inside the push motion:
                    'post_lower' — plate has just reached p_start (about to sweep)
                    'post_sweep' — plate has just reached its stop position
                                   (about to lift)
                Shared across every caller of execute_action (GenesisEnv,
                GenesisOracleEnv, data collection, future MPC/viz code) — e.g.
                to capture an intermediate video frame, log, or debug-plot the
                mid-action state. A no-op when None; callers that don't pass
                it (e.g. batched rollout planning) see no behavior change.
            on_step: optional callback(phase: str, step: int), invoked after
                EVERY simulation step of the push, with phase one of
                'lower' / 'sweep' / 'lift'. Where on_phase gives two snapshots,
                this gives every frame — which is what a video needs, and what
                a per-step diagnostic needs. A no-op when None.

        Returns:
            Tensor of shape [n_envs] with success status
        """
        def _phase_step(phase):
            if on_step is None:
                return None
            return lambda step, *_: on_step(phase, step)

        # Lower: teleport to clearance height, then simulate only the short
        # final descent into operating position.  This skips simulating the
        # approach from the full lift height above.
        self._vertical_dof_fix[:, 0] = p_start[:, 0]
        self._vertical_dof_fix[:, 1] = p_start[:, 1]
        self._vertical_dof_fix[:, 4] = angle
        lower_start = p_start + self._clearance_offset
        self.plate.set_pos(lower_start, zero_velocity=True)
        self.plate_position_translation(lower_start, p_start, self._clearance_ctrl_steps,
                                        on_step=_phase_step('lower'))
        if on_phase is not None:
            on_phase('post_lower')

        # Sweep
        reached_goal, final_pos = self.plate_velocity_translation(
            p_start,
            p_stop,
            angle,
            on_step=_phase_step('sweep'),
        )
        if on_phase is not None:
            on_phase('post_sweep')

        # Lift: simulate only the short ascent to clearance height, then
        # teleport the plate out of the way.  Particles are already below
        # clearance height so there is no contact after this point.
        self._vertical_dof_fix[:, 0] = final_pos[:, 0]
        self._vertical_dof_fix[:, 1] = final_pos[:, 1]
        self.plate_position_translation(
            final_pos, final_pos + self._clearance_offset, self._clearance_ctrl_steps,
            on_step=_phase_step('lift'))
        self.plate.set_pos(final_pos + self._lift_height_tensor, zero_velocity=True)

        return reached_goal, final_pos

    def set_transition_context(self, context: dict | None) -> None:
        """
        Set the episode-level context (e.g. source MPC variant, episode
        index, seed) that subsequent ``push_and_record(flush_after=True)``
        calls will tag their flush with, until this is called again.

        Call once per episode, right after resetting — this is what lets
        real-step flushes carry episode-identifying context *incrementally*,
        during the episode, rather than needing to wait until the episode's
        outcome (reward/success) is known to flush anything. Per-episode
        outcome, once known, is saved separately by the driver script's own
        metrics.json/rewards.npy (joinable by source + episode_idx) — see
        docs/oracle_mpc_design.md.
        """
        self._transition_context = context

    def push_and_record(
            self,
            p_start,
            p_stop,
            angle,
            on_phase=None,
            is_candidate=False,
            mpc_step=None,
            record_all_envs=True,
            flush_after=False,
        ):
        """
        Execute a push, settle (update_material_state), and — unless
        recording is disabled — append the resulting before/after/action
        transition(s) to the internal buffer for later export via
        flush_transitions().

        Replaces the execute_action(...) + update_material_state() pair used
        at every "real" or "candidate rollout" call site
        (env.genesis_env.GenesisEnv.step,
        simple_mpc.genesis_oracle.GenesisOracleEnv.step / rollout_candidates).
        Callers should set self._settle_steps / self._clearance_ctrl_steps to
        the desired values before calling this, exactly as they already do
        before the pair it replaces — neither is read by execute_action
        itself, so setting them any time before update_material_state() runs
        is equivalent.

        Args:
            p_start, p_stop, angle: same as execute_action.
            on_phase: same as execute_action (video-frame hook; unrelated to
                recording).
            is_candidate: False for a real executed step (part of the
                sequential trajectory); True for an optimizer-exploration
                rollout evaluated during planning. See
                docs/oracle_mpc_design.md for why this distinction matters
                for training use.
            mpc_step: which real MPC step's planning phase produced this
                push (the caller's own step counter — SandboxManipulation
                has no notion of "MPC step" itself). Recorded as-is.
            record_all_envs: True (default) records one sample per env — use
                this when every env ran a genuinely distinct action (e.g.
                candidate rollouts). Pass False when all n_envs execute the
                identical broadcast action from an identical state (e.g.
                GenesisOracleEnv.step()'s real step) — recording all n_envs
                there would just duplicate the same transition n_envs times;
                only env 0's sample is appended instead.
            flush_after: if True, immediately flush the buffer (tagged with
                whatever context was last set via set_transition_context())
                after appending. Pass True for real steps so data reaches
                disk incrementally, step by step, rather than accumulating
                in memory for an entire episode — episodes can take a long
                time, and losing hours of buffered candidate-rollout data to
                a crash (or just not seeing any output while a run is in
                progress) is the failure mode this avoids. Leave False for
                candidate rollouts (the common case: many rollouts accumulate
                between one real step's flush and the next).

        Returns:
            (reached_goal, final_pos) — identical to execute_action.
        """
        before = self._particle_state.clone() if self._record_transitions else None
        reached_goal, final_pos = self.execute_action(p_start, p_stop, angle, on_phase=on_phase)
        self.update_material_state()
        if self._record_transitions:
            if record_all_envs:
                self._transition_buffer.append_batch(
                    before, self._particle_state, p_start, p_stop, angle,
                    reached_goal, is_candidate=is_candidate, mpc_step=mpc_step,
                )
            else:
                self._transition_buffer.append(
                    before[0], self._particle_state[0], p_start[0], p_stop[0],
                    float(angle[0]), bool(reached_goal[0]),
                    is_candidate=is_candidate, mpc_step=mpc_step,
                )
            if flush_after:
                self.flush_transitions(context=self._transition_context)
        return reached_goal, final_pos

    def flush_transitions(self, context: dict | None = None) -> str | None:
        """
        Write any buffered transitions (see push_and_record) to
        self._transitions_dir and clear the buffer.

        Returns the saved data file's path, or None if recording is
        disabled or nothing is buffered. ``context=None`` falls back to
        whatever was last set via ``set_transition_context()`` (so the
        context-less safety-net calls from shuffle_particles()/destroy()
        still tag flushes when a context is active). Most callers don't need
        to pass ``context`` explicitly at all: push_and_record's
        ``flush_after=True`` (used for real steps) already does, using the
        stored context automatically.
        """
        if not self._record_transitions or self._transition_buffer.is_empty():
            return None
        return self._transition_buffer.save(
            self._transitions_dir, self._config,
            context=context if context is not None else self._transition_context)

    def collect_data_samples(
            self,
            n_samples: int = 200,
            path : str | Path = "training",
            placement_aware: bool = False,
            shared_travel_distance: bool = True,
        ):
        """
        Collect data samples from all environments efficiently.
        Optimized for GPU processing and memory efficiency.

        Args:
            n_samples: Number of samples to collect per environment
            path: Output path for data
            placement_aware: draw each touchdown pose from the tool's free
                configuration space rather than blindly (see
                generate_action_samples). Falls back per sample where no
                collision-free placement exists.
            shared_travel_distance: share one push length across the batch per
                sample (see generate_action_samples). On by default here: it is
                a large throughput win and costs only within-batch variation in
                one of five action dimensions.
        """
        max_samples = n_samples * self._n_envs

        self._config["data_collection"].update({
            "n_envs": self._n_envs,
            "samples_per_env": n_samples,
            "goal_threshold": self._goal_threshold,
            "placement_aware": bool(placement_aware),
            "shared_travel_distance": bool(shared_travel_distance),
        })

        # Allocate once or reuse if same size
        if (not hasattr(self, '_collection_buffers') or 
            self._collection_buffers['states'].shape[0] != n_samples or
            self._collection_buffers['states'].shape[1] != self._n_envs):
            self._allocate_collection_buffers(n_samples)
        
        # Clear data buffer
        for buf in self._collection_buffers.values():
            buf.zero_()
        
        # Settle first: placement-aware sampling reads self._particle_state, so
        # the occupancy it builds must reflect the pile the tool will actually
        # descend into, not the pre-settle spawn layout.
        self.update_material_state()

        # Generate random action samples per env
        action_starts, action_stops, angles = self.generate_action_samples(
            n_samples, placement_aware=placement_aware,
            shared_travel_distance=shared_travel_distance)
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

        # Two self-audit fields, so a saved dataset carries the evidence that it
        # is trustworthy instead of requiring the run to be repeated to find out.
        # Both failures they cover are silent: a contact-budget overflow drops
        # contacts without raising (the error bit Genesis would set is cleared
        # by the per-step set_dofs_position in the sweep), and an escaped
        # particle poisons every later transition in its env because s comes
        # from the previous s'.
        try:
            usage = self.contact_budget_usage()
            budget = {
                "broad_pairs": usage["broad_pairs"], "broad_cap": usage["broad_cap"],
                "contact_points": usage["contact_points"],
                "contact_cap": usage["contact_cap"],
                "worst_fraction_of_cap": max(
                    usage["broad_pairs"] / max(usage["broad_cap"], 1),
                    usage["contact_points"] / max(usage["contact_cap"], 1)),
            }
        except Exception:
            budget = None

        self._config["statistics"] = {
            "n_envs"   : self._n_envs,
            "samples_per_env"  : n_samples,
            "total_samples_collected"  : num_collected_samples,
            "number_of_failed_samples" : max_samples - num_collected_samples,
            "escaped_particles" : self.escaped_particle_count(),
            "contact_budget" : budget,
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
        self.flush_transitions()   # final safety-net flush — see shuffle_particles()
        gs.destroy()

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        for _ in range(horizon):
            self._step_scene()
