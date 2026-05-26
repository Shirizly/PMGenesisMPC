import genesis as gs
import genesis.utils.geom as gu 
import numpy as np
import yaml
from .utilities.materials import *
import quaternion as qu
from pathlib import Path
import pickle
import os
import torch


class SandboxManipulation:

    def __init__(self, config,):

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
            performance_mode=self._config["simulation"].get('performance_mode', False),
        )

        # PARAMETERS FOR TRAINING
        self._box_pos = self._config["sandbox"]["box"].get('pos', [0.0, 0.0, 0.0])
        self._box_vol = self._config["sandbox"]["box"].get('vol', [0.3, 0.3, 0.1])
        self._wall_thickness = self._config["sandbox"]["box"].get('wall_thickness', 0.02)
        self._particle_size = self._config["sandbox"]["material"]["properties"].get('particle_size', 0.01)
        self._granular_vol = self._config["sandbox"]["material"].get('vol', [0.27, 0.27, 0.1])
        self._material_type = self._config["sandbox"]["material"].get('type', 'rsa')

        self._init_scene()
        self._add_entities()

        self._data_samples = []
        self._n_aborted_down = 0
        self._n_aborted_action = 0

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

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt       = self._config["simulation"].get('dt', 4e3),
                substeps = self._config["simulation"].get('substeps', 1),
            ),
            rigid_options=gs.options.RigidOptions(
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
            ) if self._viewer_options is not None else None,
            show_viewer=viewer_settings.get('show_viewer', False) if self._viewer_options is not None else False,
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
            granular_touch_height = self._particle_size/2 if isinstance(self._particle_size, float) else min(self._particle_size)/4
        
        elif self._material_type == "sand":
            self.material = add_sand(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                sand_color=granular_color
            )
        elif self._material_type == "liquid":
            self.material = add_liquid(
                scene=self._scene,
                box_pos=self._box_pos,
                granular_vol=self._granular_vol,
                material_properties=material_properties,
                wall_thickness=self._wall_thickness,
                color=granular_color,
            )
        else:
            raise ValueError(f"Unsupported material type {self._material_type}. Supported types are 'granular', 'sand', and 'liquid'.")

        self._operation_height = self._box_pos[2] + granular_touch_height + self._wall_thickness/2    

    def _save_sample(self, sample):
        state, state_, action = sample

        self._data_samples.append(
            {
                "state" : state,
                "state_" : state_,
                "action" : action,
            }
        )        

    def _save_data(
            self,
            path : str | Path
    ):
        
        with open(path, 'wb') as handle:
            pickle.dump(self._data_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_config(
            self,
            path : str | Path
        ):

        with open(path, 'w') as outfile:
            try: 
                yaml.dump(self._config, outfile, default_flow_style=False)
            except yaml.YAMLError as exc:
                print(exc)
       
    def build(self):
        self._scene.build(n_envs=1)
        
        dofs_idx = [0, 1, 2, 3, 4, 5]
        self.plate.set_dofs_kp((0.8,) * 6, dofs_idx)
        self.plate.set_dofs_kv((1.0,) * 6, dofs_idx)

    def destroy(self):
        "Destroying environment"
        gs.destroy()

    def view(self, horizon=1000):
        for _ in range(horizon):
            self._scene.visualizer.update()
    
    def simulate(self, horizon=1000):
        for _ in range(horizon):
            self._scene.step()

    def set_material_state(self, positions : list):
        """set position of particles"""
        
        if self._material_type != "rsa":
            raise NotImplementedError("Method not implemented for materials other than RSA")

        if len(positions) != len(self.material):
            raise ValueError(
                f"Number of positions {len(positions)} does not match number of particles {len(self.material)}"
            )

        for pos, particle in zip(positions, self.material):
            particle.set_pos(pos)

    def get_material_state(self,):
        """
        Returns an array of particle positions
        """
        if self._material_type != "rsa":
            raise NotImplementedError("Method not implemented for materials other than RSA")
        
        # check if no particle is moving
        n_p = len(self.material)
        moving = True

        self.plate.set_pos(self.plate.get_pos())
        self.plate.control_dofs_position_velocity(self.plate.get_pos(), torch.zeros((3)), dofs_idx_local=[0, 1, 2])
        while moving:
            
            v = torch.zeros(len(self.material))
            for i, e in enumerate(self.material):
                v[i] = torch.linalg.norm(e.get_vel(), axis=1)
            
            if (v < 0.01).all():
                moving = False
            
            # freeze plate
            self.plate.set_dofs_position(self.plate.get_dofs_position())            
            self._scene.step()
        
        state = torch.zeros((n_p, 4), device=gs.device)
        for i, e in enumerate(self.material):
            pos = e.get_pos()
            size = e.morph.size[0] # only save on dim for cubes
            state[i, 0:3] = pos
            state[i, 3] = size
        return state

    def get_collected_samples(self):
        """
        Return previously collected samples
        
        Each samples consists of state(i), state(i+1), start_position, end_position, angle, velocity
        """
        return self._data_samples.values()
    
    def plate_velocity_translation(self, p_start, p_end, speed, fix_pose, fix_dofs, debug=True):
        if debug:
            self._scene.clear_debug_objects()
            T_start = gu.trans_to_T(p_start)
            T_end = gu.trans_to_T(p_end)
            self._scene.draw_debug_frame(T_start, axis_length=0.05, origin_size=0.001, axis_radius=0.001)
            self._scene.draw_debug_frame(T_end, axis_length=0.05, origin_size=0.001, axis_radius=0.001)
        
        # direction of movement
        delta = p_end - p_start
        dist = torch.linalg.norm(delta)
        
        # speed
        direction = delta / dist
        v = direction * speed

        # move plate
        self.plate.set_pos(p_start)
        self.plate.control_dofs_position_velocity(p_end, v, dofs_idx_local=[0, 1, 2])
                
        # number of steps to reach target position
        n_required = int(torch.ceil(dist/(speed * self._scene.dt)))
        n_current = 0
        reached_goal, abort = False, False
        while not reached_goal and not abort:
            n_current += 1
            self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)
            self._scene.step()
            cur_dist = torch.linalg.norm(self.plate.get_pos()-p_end)

            if cur_dist < 0.002:
                reached_goal = True
            abort = (n_current > n_required*1.7)
        
        if abort:
            print(f"Aborted: Distance at end of translation: {cur_dist}")
        else:
            print("Success")        
    
        return reached_goal
    
    def plate_position_translation(self, p_start, p_end, n_steps, fix_pose, fix_dofs, debug=True):
    
        t = torch.linspace(0, 1, n_steps, device=gs.device)
        path = (1 - t[:, None]) * p_start[None, :] + t[:, None] * p_end[None, :]
                
        self.plate.set_pos(path[0])
        for p in path:
            self.plate.set_pos(pos=p)
            self.plate.set_dofs_position(fix_pose, dofs_idx_local=fix_dofs)
            self._scene.step()

    def generate_action_samples(
            self,
            n_samples: int,
            operation_height: float | None = None,
        ):
        box_x, box_y, _ = self._box_pos
        tool_length, tool_width, tool_height = self._plate_size

        self._operation_height += tool_height/2
        if operation_height is not None:
            self._operation_height = operation_height
        
        angles = (-torch.pi/2) + torch.rand(n_samples, device=gs.device) * torch.pi # np.random.uniform(low=-torch.pi/2, high=torch.pi/2, size=n_samples)
        
        # sampling dimensions in x and y from box center
        sample_space_x = self._granular_vol[0]/2 - (torch.cos(angles) * tool_length/2 + abs(torch.sin(angles)) * tool_width/2 + self._safety_margin)
        sample_space_y = self._granular_vol[1]/2 - (abs(torch.sin(angles)) * tool_length/2 + torch.cos(angles) * tool_width/2 + self._safety_margin)

        # Min and max coordinates of action sample areas
        low = torch.stack([box_x - sample_space_x, box_y - sample_space_y], axis=1)
        high = torch.stack([box_x + sample_space_x, box_y + sample_space_y], axis=1)
        
        
        # Sampling n_samples start and end positions of action  
        start_samples = (high - low) * torch.rand((n_samples, 2), device=gs.device) + low # np.random.uniform(low=low, high=high, size=(n_samples, 2))
        stop_samples = (high - low) * torch.rand((n_samples, 2), device=gs.device) + low # np.random.uniform(low=low, high=high, size=(n_samples, 2))
        _z = torch.ones((n_samples, 1), device=gs.device) * self._operation_height
        action_starts = torch.concatenate((start_samples, _z), axis=1)
        action_stops = torch.concatenate((stop_samples, _z), axis=1)

        return zip(action_starts, action_stops, angles)

    def execute_action(self, p_start, p_stop, angle, speed):

        # Lowering
        self.plate_position_translation(
            p_start + self._lift_height,
            p_start,
            100,
            [p_start[0], p_start[1], 0, 0, angle],
            [0, 1, 3, 4, 5],
        )
        
        # Execute Sweeping
        success = self.plate_velocity_translation(
            p_start,
            p_stop,
            speed,
            [self._operation_height, 0, 0, angle],
            [2, 3, 4, 5],
        )

        p_stop = self.plate.get_pos().squeeze()
        
        if not success:
            self._n_aborted_action +=1
            return success
            
        # Lifting
        self.plate_position_translation(
            p_stop,
            p_stop + self._lift_height,
            100,
            [p_stop[0], p_stop[1], 0, 0, angle],
            [0, 1, 3, 4, 5],
        )

        return success

    def collect_data_samples(
            self,
            n_samples=200,
            operation_height=None,
            speed=0.125,
            lift_height = None,
        ):
        
        samples = self.generate_action_samples(
            n_samples,
            operation_height,
        )

        # STORE ALLOCATION FOR VARIABLES
        if lift_height == None:
            lift_height = self._box_vol[2]

        self._lift_height = torch.tensor([0, 0, lift_height], device=gs.device)
        self._vertical_locked_dof_vals = torch.tensor([-1, -1, 0, 0, -1], device=gs.device) # replace -1 with values
        self._vertical_locked_dof_idxs = torch.tensor([0, 1, 3, 4, 5], device=gs.device)
        self._horizontal_locked_dof_vals = torch.tensor([-1, 0, 0, -1], device=gs.device) # replace -1 with values
        self._horizontal_locked_dof_idxs = torch.tensor([2, 3, 4, 5], device=gs.device)
        
        for (p_start, p_stop, angle) in samples:
            state = self.get_material_state()
                    
            if not self.execute_action(
                p_start,
                p_stop,
                angle,
                speed,
            ):
                continue
            
            state_ = self.get_material_state()
        
            self._save_sample((state, state_, (p_start, p_stop, angle))) 
        
        print("\nStatistics")
        print("==========")
        print(">> Aborted (lowering): ", self._n_aborted_down)
        print(">> Aborted (actions) : ", self._n_aborted_action)
        print(">> Number of samples : ", n_samples)

        self._config["statistics"] = {
            "n_samples": n_samples,
            "n_aborted_down": self._n_aborted_down,
            "n_aborted_action": self._n_aborted_action,
        }

    def export_data_samples(
            self,
            path : str | Path = "training"
        ):

        base_dir = Path(__file__).parent
        full_path = base_dir / path
        Path.mkdir(full_path, parents=True, exist_ok=True)

        n_runs = int(len([name for name in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, name))])/2)
        self._save_config(full_path / (str(n_runs) + "_config.yaml"))
        self._save_data(full_path / (str(n_runs) + "_data.pkl"))
