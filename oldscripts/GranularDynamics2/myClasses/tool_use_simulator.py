import numpy as np
import pygame
from Box2D import b2World, b2PolygonShape, b2CircleShape, b2EdgeShape, b2_dynamicBody, b2_kinematicBody

class PlanarSimulator:
    """
    Planar 2D simulation of interacting disks and a tool using box2d-py and pygame for top-down view.

    Uses individual edge shapes for the tool boundary to match exact line segments,
    avoiding convex hull or triangulation artifacts.

    Friction on a top-down plane is approximated via linear/angular damping on disks.
    """
    def __init__(self,
                 tool_shape_pts,
                 disk_positions,
                 R,
                 tool_init_pose,
                 motions,
                 view = True,
                 v_trans=25.0,
                 v_rot=1.0,
                 friction_params=None,
                 time_step=1/60,
                 vel_iters=6,
                 pos_iters=2,
                 view_scale=2,
                 view_size=(800, 600)):
        # World and view parameters
        self.dt = float(time_step)
        self.vel_iters = vel_iters
        self.pos_iters = pos_iters
        self.scale = view_scale
        self.width, self.height = view_size
        self.view_center = (0, 0)  # center of the view

        # Tool and motion parameters
        self.tool_pts = [tuple(p) for p in tool_shape_pts]
        self.tool_pose = np.array(tool_init_pose, dtype=float)
        self.motions = motions
        self.v_trans = float(v_trans)
        self.v_rot = float(v_rot)

        # Disk parameters
        self.disk_positions = [tuple(p) for p in disk_positions]
        self.R = float(R)

        # Friction parameters\        
        fp = friction_params or {}
        self.mu_disk = fp.get('disk_disk', 0.2)
        self.mu_tool = fp.get('disk_tool', 0.2)
        self.mu_plane = fp.get('disk_plane', 0.2)
        self.dynamic_friction_disks = fp.get('dynamic_friction_disks', 2000.0)

        # Build physics world (no gravity)
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self._create_disks()
        self._create_tool()

        if view:
            # Pygame init
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Planar Simulator')
            self.clock = pygame.time.Clock()
            self.colors = {
                'background': (255, 255, 255),
                'tool': (0, 0, 0),
                'disk': (0, 0, 255),
            }

    def _create_disks(self):
        """Create dynamic disk bodies with damping for top-down friction."""
        self.disks = []
        for pos in self.disk_positions:
            body = self.world.CreateDynamicBody(position=pos)
            body.CreateCircleFixture(radius=self.R, density=1.0, friction=self.mu_disk)
            # body.linearDamping = self.mu_plane
            # body.angularDamping = self.mu_plane
            body.bullet = True  # avoid tunneling
            self.disks.append(body)

    def _create_tool(self):
        """Create a kinematic tool using edge shapes to match exact line segments."""
        x0, y0, th0 = self.tool_pose
        self.tool = self.world.CreateKinematicBody(position=(x0, y0), angle=th0)
        pts = self.tool_pts
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i+1) % n]
            p1 = (float(p1[0]), float(p1[1]))
            p2 = (float(p2[0]), float(p2[1]))
            edge = b2EdgeShape(vertices=(p1, p2))
            self.tool.CreateFixture(shape=edge, density=1.0, friction=self.mu_tool)

    def _world_to_screen(self, v):
        x, y = v
        cx, cy = self.view_center  # <-- you add this as an instance variable
        sx = int(self.width / 2 + (x - cx) * self.scale)
        sy = int(self.height / 2 - (y - cy) * self.scale)
        return sx, sy
    
    def compute_scene_center(self):
        pts = [disk.position for disk in self.disks]
        pts.append(self.tool.position)
        mean = np.mean(pts, axis=0)
        return mean

    def _draw(self):
        self.screen.fill(self.colors['background'])
        # Draw disks
        for disk in self.disks:
            pos = disk.position
            r = int(self.R * self.scale)
            pygame.draw.circle(
                self.screen,
                self.colors['disk'],
                self._world_to_screen((pos[0], pos[1])),
                r,
                1
            )
        # Draw tool boundary
        cos_t = np.cos(self.tool.angle)
        sin_t = np.sin(self.tool.angle)
        tx, ty = self.tool.position
        pts = []
        for vx, vy in self.tool_pts:
            wx = cos_t * vx - sin_t * vy + tx
            wy = sin_t * vx + cos_t * vy + ty
            pts.append(self._world_to_screen((wx, wy)))
        pygame.draw.polygon(self.screen, self.colors['tool'], pts, 2)

    def simulate_and_render(self, fps=60):
        """Execute the scripted motions with real-time pygame rendering."""
        for motion in self.motions:
            # set velocities
            if motion['type'] == 'translate':
                vec = np.array(motion['vector'], dtype=float)
                dist = np.linalg.norm(vec)
                duration = dist / self.v_trans if dist > 0 else 0
                vel = tuple(vec / duration) if duration > 0 else (0, 0)
                self.tool.linearVelocity = vel
                self.tool.angularVelocity = 0
            elif motion['type'] == 'rotate':
                angle = float(motion['angle'])
                omega = np.sign(angle) * self.v_rot
                duration = abs(angle) / self.v_rot if self.v_rot > 0 else 0
                self.tool.angularVelocity = omega
                self.tool.linearVelocity = (0, 0)
            elif motion['type'] == 'stop':
                self.tool.linearVelocity = (0, 0)
                self.tool.angularVelocity = 0
                duration = 10
            else:
                raise ValueError('Unknown motion type')
            self.view_center = self.compute_scene_center()
            steps = int(np.ceil(duration / self.dt))
            for _ in range(steps):
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        return
                for body in self.disks:
                    velocity = body.linearVelocity
                    speed = velocity.length
                    if speed > 0:
                        friction_force_magnitude = self.dynamic_friction_disks  # Set this to your desired constant
                        friction_force = (-velocity / speed) * friction_force_magnitude
                        body.ApplyForce(friction_force, body.worldCenter, wake=True)
                    ang_vel = body.angularVelocity
                    if ang_vel != 0:
                        friction_torque = -self.dynamic_friction_disks * np.sign(ang_vel)*self.R*2/3
                        body.ApplyTorque(friction_torque, wake=True)
                self.world.Step(self.dt, self.vel_iters, self.pos_iters)
                self.world.ClearForces()
                self._draw()
                pygame.display.flip()
                self.clock.tick(fps*5)

            # stop motion
            self.tool.linearVelocity = (0, 0)
            self.tool.angularVelocity = 0

        pygame.quit()
        tool_config = np.array([self.tool.position.x,self.tool.position.y, self.tool.angle])
        disks_config = np.array([disk.position for disk in self.disks]).flatten()
        return np.concatenate([tool_config, disks_config])
    
    def simulate(self):
        """Execute the scripted motions without rendering."""
        for motion in self.motions:
            # set velocities
            if motion['type'] == 'translate':
                vec = np.array(motion['vector'], dtype=float)
                dist = np.linalg.norm(vec)
                duration = dist / self.v_trans if dist > 0 else 0
                vel = tuple(vec / duration) if duration > 0 else (0, 0)
                self.tool.linearVelocity = vel
                self.tool.angularVelocity = 0
            elif motion['type'] == 'rotate':
                angle = float(motion['angle'])
                omega = np.sign(angle) * self.v_rot
                duration = abs(angle) / self.v_rot if self.v_rot > 0 else 0
                self.tool.angularVelocity = omega
                self.tool.linearVelocity = (0, 0)
            elif motion['type'] == 'stop':
                self.tool.linearVelocity = (0, 0)
                self.tool.angularVelocity = 0
                duration = 10
            else:
                raise ValueError('Unknown motion type')
            steps = int(np.ceil(duration / self.dt))
            for step in range(steps):
                for body in self.disks:
                    velocity = body.linearVelocity
                    speed = velocity.length
                    if speed > 0:
                        friction_force_magnitude = self.dynamic_friction_disks  # Set this to your desired constant
                        friction_force = (-velocity / speed) * friction_force_magnitude
                        body.ApplyForce(friction_force, body.worldCenter, wake=True)
                    ang_vel = body.angularVelocity
                    if ang_vel != 0:
                        friction_torque = -self.dynamic_friction_disks * np.sign(ang_vel)*self.R*2/3
                        body.ApplyTorque(friction_torque, wake=True)
                self.world.Step(self.dt, self.vel_iters, self.pos_iters)
                self.world.ClearForces()
            
        tool_config = np.array([self.tool.position.x,self.tool.position.y, self.tool.angle])
        disks_config = np.array([disk.position for disk in self.disks]).flatten()
        return np.concatenate([tool_config, disks_config])

# Example usage
if __name__=='__main__':
    tool_source = 'object_outline.txt'
    # Load tool shape from file
    tool_shape = np.loadtxt(tool_source, dtype=np.float32)
    # roughly approximate tool shape by sampling every Nth point
    tool_shape = tool_shape[::5]
    # tool_shape = [[-1,-0.2],[1,-0.2],[1,0.2],[-1,0.2]]
    disks = [[-12,30], [0,-8], [20,10]]
    motions = [
        {'type':'translate','vector':[5,0]},
        {'type':'rotate','vector':np.pi,'center':[0,0]},
        {'type':'translate','vector':[0,10]},
        {'type':'rotate','vector':np.pi/2,'center':[0,0]},
        {'type':'translate','vector':[-15,0]},
        {'type':'rotate','vector':-np.pi,'center':[0,0]},
        {'type':'translate','vector':[0,-15]},
        {'type':'rotate','vector':np.pi/2,'center':[0,0]},
    ]
    sim = PlanarSimulator(tool_shape, disks, R=5,
                          tool_init_pose=[0,0,0], motions=motions,
                          v_trans=0.7, v_rot=0.2,
                          friction_params={'disk_disk':0.3,'disk_tool':0.2,'disk_plane':1})
    sim.simulate_and_render(fps=60)
