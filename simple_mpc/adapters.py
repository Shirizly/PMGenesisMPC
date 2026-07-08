"""
Model adapters for run_simple_mpc.

Each adapter wraps one model type and exposes five model-agnostic operations:

  obs_to_state       : rendered obs (H,W,5)  →  state tensor on device
                       Also updates any per-step bookkeeping (e.g. particle_dens)
  expand_state       : (1,*S)                →  (n_sample,*S) clone, grad-ready
  predict_step       : (n_sample,*S), (n_sample,4) → (n_sample,*S)  [backprop-able]
  compute_reward     : (n_sample,*S)         →  (n_sample,)          [backprop-able]
  format_states_pred : list of per-step state ndarrays → stored in output dict

Optionally:
  print_step_info    : prints model-specific material / coverage diagnostics
  debug_vis_enabled  : bool — whether the run_simple_mpc debug_vis hooks fire

The factory function ``make_adapter`` auto-selects the right adapter for the
model type.  To support a new model, subclass one of these or write your own
class with the same API and register it in ``make_adapter``.
"""

from __future__ import annotations

import numpy as np
import torch

from utils import depth2fgpcd, fps_np
from transforms.functional import particles_to_occupancy as _particles_to_occupancy
from simple_mpc.occupancy_reward import OccupancyReward

# Depth threshold for foreground detection; must match utils.scale_subgoal_to_material_pixels
_FG_DEPTH_THRESHOLD = 0.599 / 0.8


# ─────────────────────────────────────────────────────── geometry helper ─────

def _gen_s_delta(s_cur: torch.Tensor,
                 action: torch.Tensor,
                 cam_extrinsic: np.ndarray,
                 global_scale: float,
                 pusher_w: float) -> torch.Tensor:
    """
    Compute per-particle displacement field for a world-space push action.

    Extracted from ``PlannerGD.gen_s_delta``; kept standalone to avoid importing
    the planner (which has heavy env dependencies).

    Parameters
    ----------
    s_cur        : (N, particle_num, 3) camera-space particle positions
    action       : (N, 4)  [sx, sy, ex, ey] in world 2-D coords
    cam_extrinsic: (4, 4)  world-to-camera affine transform (from env.get_cam_extrinsics())
    global_scale : float   normalisation factor for camera-space coords
    pusher_w     : float   half-width of the pusher (in normalised camera units)

    Returns
    -------
    (N, particle_num, 3) per-particle displacement to add to s_cur
    """
    N     = action.shape[0]
    dev   = action.device
    dtype = action.dtype

    s_xy = action[:, :2]   # (N, 2)
    e_xy = action[:, 2:]   # (N, 2)
    h    = torch.zeros((N, 1), device=dev, dtype=dtype)  # floor height

    # World 3-D: x_world, y_up=0, z_world = -y_2d  (matches PlannerGD convention)
    s_3d = torch.cat([s_xy[:, 0:1],  h, -s_xy[:, 1:2]], dim=1)   # (N, 3)
    e_3d = torch.cat([e_xy[:, 0:1],  h, -e_xy[:, 1:2]], dim=1)   # (N, 3)

    # World → camera-space transform (equivalent to PlannerGD.world2cam):
    #   cam_coords = (inv(opencv_T_opengl) @ cam_extrinsic @ [world | 1].T).T / global_scale
    # which simplifies to   M = opencv_T_opengl.T @ cam_extrinsic  (since B^{-1}=B for axis-flip)
    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]], dtype=np.float64)
    M = (opencv_T_opengl.T @ cam_extrinsic).astype(np.float32)   # (4, 4)
    M_t = torch.tensor(M, device=dev, dtype=dtype)

    def _w2c(pts3d):   # pts3d: (N, 3)
        ones = torch.ones((pts3d.shape[0], 1), device=dev, dtype=dtype)
        return (M_t @ torch.cat([pts3d, ones], dim=1).T).T[:, :3] / global_scale

    s_cam = _w2c(s_3d)   # (N, 3) — camera-space start
    e_cam = _w2c(e_3d)   # (N, 3) — camera-space end

    push_vec = e_cam - s_cam                                        # (N, 3)
    push_l   = torch.linalg.norm(push_vec, dim=1, keepdim=True)    # (N, 1)
    push_dir = push_vec / push_l                                    # (N, 3)

    # Lateral orthogonal direction (XY plane only)
    zeros  = torch.zeros((N, 1), device=dev, dtype=dtype)
    ortho  = torch.cat([-push_dir[:, 1:2], push_dir[:, 0:1], zeros], dim=1)  # (N, 3)

    # Per-particle projections onto longitudinal and lateral axes
    diff       = s_cur - s_cam[:, None, :]                          # (N, particle_num, 3)
    proj_l     = (diff * push_dir[:, None, :]).sum(-1)              # (N, particle_num)
    proj_ortho = (diff *      ortho[:, None, :]).sum(-1)            # (N, particle_num)

    # Longitudinal gate: particle must be in [0, push_l] along the push direction
    l_mask = ((proj_l > 0.0) & (proj_l < push_l[:, 0:1])).float()  # hard   (N, particle_num)

    # Lateral soft gate: Gaussian falloff outside ±pusher_w
    excess = torch.maximum(torch.clamp(-pusher_w - proj_ortho, min=0.),
                           torch.clamp( proj_ortho - pusher_w, min=0.))
    w_mask = torch.exp(-excess / 0.01)                              # (N, particle_num)

    # Displacement along push direction: how far does the particle have to move to the end?
    to_end      = e_cam[:, None, :] - s_cur                         # (N, particle_num, 3)
    proj_to_end = (to_end * push_dir[:, None, :]).sum(-1)           # (N, particle_num)

    s_delta = (proj_to_end[..., None]                               # (N, particle_num, 3)
               * push_dir[:, None, :]
               * l_mask[..., None]
               * w_mask[..., None])
    return s_delta


# ─────────────────────────────────────── Eulerian adapter ─────────────────────

class EulerianAdapter:
    """Wraps EulerianModelWrapper for run_simple_mpc."""

    debug_vis_enabled = True   # full debug_vis pipeline is available

    # Reward type registry: name → (callable, valid_for_opt, valid_for_report)
    # valid_for_opt/report indicate which contexts the reward is suitable for
    _REWARD_METHODS = {
        'default': ('_reward_default', True, True),
        'iou':     ('_reward_iou', True, True),
    }

    def __init__(self,
                 model_dy,
                 subgoal: np.ndarray,
                 cam_params,
                 global_scale: float,
                 device: str,
                 empty_penalty: float = 0.0,
                 reward_type_opt: str = 'default',
                 reward_type_report: str | None = None):
        self.model_dy     = model_dy
        self.cam_params   = cam_params
        self.global_scale = global_scale
        self.device       = device if torch.cuda.is_available() else 'cpu'

        # Pre-compute goal score map (fixed throughout the run).
        self.score_tensor = model_dy.prepare_goal_reward(
            subgoal, cam_params, device=device, empty_penalty=empty_penalty)
        self.score_np     = self.score_tensor.cpu().numpy()    # (Nx, Ny) for debug vis

        # Set reward type configuration
        self.reward_type_opt = reward_type_opt
        self.reward_type_report = reward_type_report or reward_type_opt

        # Validate reward types
        for rt in [self.reward_type_opt, self.reward_type_report]:
            if rt not in self._REWARD_METHODS:
                raise ValueError(
                    f"Unknown Eulerian reward type '{rt}'. "
                    f"Available: {list(self._REWARD_METHODS.keys())}"
                )
            _, valid_opt, valid_report = self._REWARD_METHODS[rt]
            if rt == self.reward_type_opt and not valid_opt:
                raise ValueError(
                    f"Reward type '{rt}' is not valid for optimization"
                )
            if rt == self.reward_type_report and not valid_report:
                raise ValueError(
                    f"Reward type '{rt}' is not valid for reporting"
                )

    # ── public API ────────────────────────────────────────────────────────────

    def _get_example_state_shape(self) -> tuple:
        """Return state shape (without batch dimension) for benchmarking."""
        return (self.model_dy.grid_res[0], self.model_dy.grid_res[1])

    def obs_to_state(self, obs_np: np.ndarray) -> torch.Tensor:
        """(H,W,5) → (1, Nx, Ny) occupancy tensor (detached, on device)."""
        depth = obs_np[..., -1] / self.global_scale
        pts   = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, self.cam_params)
        pts_t = torch.from_numpy(pts).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            occ = self.model_dy.initial_occ_from_particles(pts_t)
        return occ.detach()                                    # (1, Nx, Ny)

    def expand_state(self, state: torch.Tensor, n_sample: int) -> torch.Tensor:
        """(1, Nx, Ny) → (n_sample, Nx, Ny) — cloned so gradients can flow."""
        return state.expand(n_sample, *state.shape[1:]).clone()

    def predict_step(self, state_batch: torch.Tensor,
                     act_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, Nx, Ny) × (n_sample, 4) → (n_sample, Nx, Ny)."""
        return self.model_dy.predict_one_step_occ(state_batch, act_batch)

    def compute_reward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, Nx, Ny) → (n_sample,) reward (higher = closer to goal)."""
        return (state_batch.clamp(0.0, 1.0) * self.score_tensor
                ).reshape(state_batch.shape[0], -1).sum(dim=-1)

    def compute_reward_iou(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, Nx, Ny) → (n_sample,) IoU reward with goal region.
        
        Computes intersection over union between predicted material
        (occupancy > 0.5) and goal region (score_tensor > 0).
        Works with score tensors marked as 0 or -1 for empty areas.
        """
        # Binarize: predicted material where occupancy > 0.5
        pred_material = (state_batch > 0.5).float()  # (n_sample, Nx, Ny)
        
        # Goal region where score tensor is positive (1=goal, 0 or -1=empty)
        goal_material = (self.score_tensor > 0).float()  # (Nx, Ny)
        
        # Flatten for batch computation
        pred_flat = pred_material.view(pred_material.shape[0], -1)  # (n_sample, Nx*Ny)
        goal_flat = goal_material.view(-1)  # (Nx*Ny,)
        
        # Intersection and union
        intersection = (pred_flat * goal_flat).sum(dim=1)  # (n_sample,)
        union = torch.maximum(pred_flat, goal_flat.unsqueeze(0)).sum(dim=1)  # (n_sample,)
        
        # IoU (add epsilon to avoid division by zero)
        iou = intersection / (union + 1e-6)
        return iou

    # ── private reward implementations ─────────────────────────────────────

    def _reward_default(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Default reward: weighted sum of occupancy × goal score map."""
        return (state_batch.clamp(0.0, 1.0) * self.score_tensor
                ).reshape(state_batch.shape[0], -1).sum(dim=-1)

    def _reward_iou(self, state_batch: torch.Tensor) -> torch.Tensor:
        """IoU reward (delegates to compute_reward_iou)."""
        return self.compute_reward_iou(state_batch)

    # ── reward getter interface ────────────────────────────────────────────

    def obs_to_report_state(self, obs_np: np.ndarray) -> torch.Tensor:
        """Return state for reporting reward evaluation.

        For EulerianAdapter the reporting state is the same as the MPC state:
        both are derived from the full depth image via initial_occ_from_particles,
        so density is already maximal.
        """
        return self.obs_to_state(obs_np)

    def get_reward_fn_opt(self) -> callable:
        """Return the reward function to use for optimization."""
        method_name = self._REWARD_METHODS[self.reward_type_opt][0]
        return getattr(self, method_name)

    def get_reward_fn_report(self) -> callable:
        """Return the reward function to use for reporting."""
        method_name = self._REWARD_METHODS[self.reward_type_report][0]
        return getattr(self, method_name)

    def format_states_pred(self, rollout_seq: list) -> np.ndarray:
        """list[(Nx,Ny) ndarray] × n_ahead → (n_ahead, Nx, Ny) for output dict."""
        return np.stack(rollout_seq)

    def print_step_info(self, state: torch.Tensor, step: int, n_mpc: int):
        """Print material-coverage diagnostics using the occupancy grid."""
        occ_np = state[0].cpu().numpy()
        mask   = occ_np > 0.5
        cov    = mask.mean() * 100
        if mask.any():
            ci, cj = np.where(mask)
            print(f"  step {step}: material {cov:.1f}% of grid, "
                  f"centroid voxel ({ci.mean():.1f}, {cj.mean():.1f}) "
                  f"of {occ_np.shape}  "
                  f"[action range: x_grid 0-{occ_np.shape[0]-1}, "
                  f"y_grid 0-{occ_np.shape[1]-1}]")
        else:
            print(f"  step {step}: WARNING — occupancy grid is entirely empty! "
                  "Check depth threshold and grid bounds.")

    def debug_occ(self, state: torch.Tensor) -> np.ndarray:
        """Return occupancy as numpy (Nx, Ny) for debug_vis helpers."""
        return state[0].cpu().numpy()

    def get_ot_grids(self, state: torch.Tensor) -> tuple:
        """Return (source_grid, goal_grid) as (H, W) float32 numpy arrays.

        ``source_grid`` is the current occupancy; ``goal_grid`` is a binary
        mask of the goal region (1 inside goal, 0 elsewhere).
        """
        source_grid = state.squeeze(0).cpu().numpy().astype(np.float32)
        goal_grid   = (self.score_np > 0).astype(np.float32)
        return source_grid, goal_grid


# ─────────────────────────────────────────── GNN adapter ─────────────────────

class GNNAdapter:
    """
    Wraps PropNetDiffDenModel for run_simple_mpc.

    The GNN model operates on fixed-size particle point clouds (particle_num × 3).
    Action encoding follows PlannerGD.gen_s_delta: each 4-D world-space action
    [sx, sy, ex, ey] is converted into a per-particle displacement field
    (the 'selta' used as GNN input), which is differentiable w.r.t. the action.

    For MPC optimization, a particle-based reward matching the state
    representation is intended, but the FlexEnv implementation
    (config_reward_ptcl) has not been ported to Genesis yet — the 'default'
    optimization reward raises NotImplementedError (see INTERFACES.md §6).

    For reporting/comparison, computes occupancy-based reward from the full raw
    observation (compute_occ_reward_from_obs) so it is on the same scale as the
    Eulerian adapter for model-invariant comparison.

    Required config keys
    --------------------
    cfg['mpc']['particle_num']        (default 50)
    cfg['dataset']['global_scale']
    """

    debug_vis_enabled = False   # debug_vis uses occupancy arrays — skip for GNN

    # Reward type registry: name → (callable, valid_for_opt, valid_for_report)
    # GNN uses particle reward for optimization, can use either for reporting
    _REWARD_METHODS = {
        'default':   ('_reward_default', True,  True),
        'iou':       ('_reward_iou',     False, True),  # report only: needs occ grid
        'eulerian':  ('_reward_eulerian', False, True),  # report only: particles → occ × score
    }

    def __init__(self,
                 model_dy,
                 env,
                 subgoal: np.ndarray,
                 cam_params,
                 cfg: dict,
                 device: str = 'cuda',
                 reward_type_opt: str = 'default',
                 reward_type_report: str | None = None):
        self.model_dy      = model_dy
        self.env           = env
        self.cam_params    = cam_params
        self.global_scale  = cfg['dataset']['global_scale']
        self.device        = device
        self.particle_num  = cfg['mpc'].get('particle_num', 50)
        self.pusher_w      = 0.8 / self.global_scale   # matches PlannerGD convention
        self.cam_extrinsic = env.get_cam_extrinsics()  # (4, 4) ndarray — TBD: GenesisEnv returns np.eye(4); genesis action convention does not use extrinsic matrix
        # The FlexEnv particle reward (config_reward_ptcl) was never ported to
        # Genesis, so the 'default' optimization reward is unavailable; see
        # INTERFACES.md §6.  _reward_default fails loudly instead of silently.
        self._reward_fn    = None

        # particle density — updated each obs_to_state call
        self._particle_dens: float = 1.0

        # Pre-compute goal tensors (fixed throughout the run).
        subgoal_t = torch.from_numpy(subgoal).float().to(device)
        H, W      = subgoal_t.shape

        # goal_coor: (col, row) pixel positions of goal-interior pixels
        goal_coor_np = torch.flip((subgoal_t < 0.5).nonzero(),
                                  dims=(1,)).float().cpu().numpy()
        n_goal = min(self.particle_num * 5, goal_coor_np.shape[0])
        if n_goal > 0 and goal_coor_np.shape[0] > n_goal:
            goal_coor_np, _ = fps_np(goal_coor_np, n_goal)

        self.goal_t      = subgoal_t
        self.goal_coor_t = torch.from_numpy(goal_coor_np).float().to(device)

        # Set reward type configuration
        self.reward_type_opt = reward_type_opt
        self.reward_type_report = reward_type_report or reward_type_opt

        # Validate reward types
        for rt in [self.reward_type_opt, self.reward_type_report]:
            if rt not in self._REWARD_METHODS:
                raise ValueError(
                    f"Unknown GNN reward type '{rt}'. "
                    f"Available: {list(self._REWARD_METHODS.keys())}"
                )
            _, valid_opt, valid_report = self._REWARD_METHODS[rt]
            if rt == self.reward_type_opt and not valid_opt:
                raise ValueError(
                    f"Reward type '{rt}' is not valid for optimization in GNN adapter. "
                    f"Use 'default' for optimization."
                )
            if rt == self.reward_type_report and not valid_report:
                raise ValueError(
                    f"Reward type '{rt}' is not valid for reporting in GNN adapter"
                )

        # Store occupancy grid parameters for compute_occ_reward_from_obs()
        self.grid_bounds = None
        self.grid_res = None
        self.occ_score_tensor = None

    # ── public API ────────────────────────────────────────────────────────────

    def _get_example_state_shape(self) -> tuple:
        """Return state shape (without batch dimension) for benchmarking."""
        return (self.particle_num, 3)

    def obs_to_state(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        (H,W,5) → (1, particle_num, 3) particle tensor (detached, on device).

        Uses a small internal batch to get a robust particle_r estimate, then
        updates self._particle_dens for subsequent predict_step calls.
        """
        # TBD: obs2ptcl_fixed_num_batch is FlexEnv-specific — needs Genesis equivalent
        pts_batch, r_batch = self.env.obs2ptcl_fixed_num_batch(
            obs_np, self.particle_num, batch_size=5)
        particle_r          = float(np.median(r_batch))
        self._particle_dens = 1.0 / (particle_r ** 2)

        pts0 = pts_batch[0]    # (particle_num, 3) — first sampled version
        return (torch.from_numpy(pts0).float()
                .to(self.device).unsqueeze(0).detach())   # (1, particle_num, 3)

    def expand_state(self, state: torch.Tensor, n_sample: int) -> torch.Tensor:
        """(1, particle_num, 3) → (n_sample, particle_num, 3)."""
        return state.expand(n_sample, *state.shape[1:]).clone()

    def predict_step(self, state_batch: torch.Tensor,
                     act_batch: torch.Tensor) -> torch.Tensor:
        """
        (n_sample, particle_num, 3) × (n_sample, 4) → (n_sample, particle_num, 3).

        Gradient flows through both _gen_s_delta (geometry) and
        GNN.predict_one_step (learned dynamics).
        """
        n     = state_batch.shape[0]
        a_cur = torch.zeros(n, self.particle_num,
                            device=self.device, dtype=state_batch.dtype)
        dens  = torch.full((n,), self._particle_dens,
                           device=self.device, dtype=state_batch.dtype)
        s_delta = _gen_s_delta(state_batch, act_batch,
                               self.cam_extrinsic, self.global_scale,
                               self.pusher_w)
        return self.model_dy.predict_one_step(a_cur, state_batch, s_delta, dens)

    def compute_reward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, particle_num, 3) → (n_sample,) reward.

        Uses particle-based reward directly, preserving gradients through the
        optimization. This matches the state representation (particles).
        """
        return self._reward_default(state_batch)

    def compute_reward_iou(self, state_batch: torch.Tensor) -> torch.Tensor:
        """(n_sample, particle_num, 3) → (n_sample,) IoU reward with goal region.
        
        Converts particles to occupancy grid, then computes IoU between
        predicted material (occupancy > 0.5) and goal region (occ_score_tensor > 0).
        Works with score tensors marked as 0 or -1 for empty areas.
        """
        if self.grid_bounds is None or self.occ_score_tensor is None:
            raise RuntimeError(
                "IoU reward requires occupancy parameters to be set via "
                "set_occupancy_params(). This should be called automatically "
                "by make_adapter(); check that occupancy initialization succeeded."
            )
        
        # Convert particles to occupancy
        occ = _particles_to_occupancy(
            state_batch, self.grid_bounds, self.grid_res, sigma=0.0)
        # (n_sample, Nx, Ny)
        
        # Binarize: predicted material where occupancy > 0.5
        pred_material = (occ > 0.5).float()  # (n_sample, Nx, Ny)
        
        # Goal region where score tensor is positive (1=goal, 0 or -1=empty)
        goal_material = (self.occ_score_tensor > 0).float()  # (Nx, Ny)
        
        # Flatten for batch computation
        pred_flat = pred_material.view(pred_material.shape[0], -1)  # (n_sample, Nx*Ny)
        goal_flat = goal_material.view(-1)  # (Nx*Ny,)
        
        # Intersection and union
        intersection = (pred_flat * goal_flat).sum(dim=1)  # (n_sample,)
        union = torch.maximum(pred_flat, goal_flat.unsqueeze(0)).sum(dim=1)  # (n_sample,)
        
        # IoU (add epsilon to avoid division by zero)
        iou = intersection / (union + 1e-6)
        return iou

    # ── private reward implementations ─────────────────────────────────────

    def _reward_default(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Default particle-based reward (delegates to the ported reward fn)."""
        if self._reward_fn is None:
            raise NotImplementedError(
                "GNNAdapter's default particle reward is not available: the "
                "FlexEnv config_reward_ptcl function has not been ported to "
                "Genesis (see INTERFACES.md §6). Use reward_type_report="
                "'iou'/'eulerian' for reporting, or port a particle reward and "
                "assign it to self._reward_fn."
            )
        return self._reward_fn(
            state_batch,
            self.goal_t,
            cam_params=self.cam_params,
            goal_coor=self.goal_coor_t,
            normalize=True,
        )

    def _reward_iou(self, state_batch: torch.Tensor) -> torch.Tensor:
        """IoU reward via occupancy conversion (delegates to compute_reward_iou)."""
        return self.compute_reward_iou(state_batch)

    def _reward_eulerian(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Eulerian-style reward: convert particles → occupancy, then apply occ_score_tensor.

        Produces rewards on the same scale as EulerianAdapter._reward_default,
        enabling direct cross-model comparison.
        Requires occupancy parameters to be set via set_occupancy_params().
        """
        if self.grid_bounds is None or self.occ_score_tensor is None:
            raise RuntimeError(
                "'eulerian' reward type requires occupancy parameters to be set via "
                "set_occupancy_params(). This should be called automatically "
                "by make_adapter(); check that occupancy initialization succeeded."
            )
        occ = _particles_to_occupancy(
            state_batch, self.grid_bounds, self.grid_res, sigma=0.0)  # (n_sample, Nx, Ny)
        return (occ.clamp(0.0, 1.0) * self.occ_score_tensor
                ).reshape(occ.shape[0], -1).sum(dim=-1)  # (n_sample,)

    # ── reward getter interface ────────────────────────────────────────────

    def get_reward_fn_opt(self) -> callable:
        """Return the reward function to use for optimization."""
        method_name = self._REWARD_METHODS[self.reward_type_opt][0]
        return getattr(self, method_name)

    def get_reward_fn_report(self) -> callable:
        """Return the reward function to use for reporting."""
        method_name = self._REWARD_METHODS[self.reward_type_report][0]
        return getattr(self, method_name)

    def compute_occ_reward_from_obs(self, obs_np: np.ndarray) -> float | None:
        """
        Compute occupancy-based reward from a raw rendered observation.

        For reporting/comparison only (not used during MPC optimization).
        Uses the FULL foreground point cloud extracted from the depth channel,
        matching the density of what the Eulerian adapter sees, so rewards are
        on the same scale across model types.

        Returns None if occupancy parameters were not set.
        """
        if self.grid_bounds is None or self.occ_score_tensor is None:
            return None

        depth = obs_np[..., -1] / self.global_scale
        pts_np = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, self.cam_params)
        if pts_np.shape[0] == 0:
            return 0.0

        pts_t = torch.from_numpy(pts_np.astype(np.float32)).to(self.device).unsqueeze(0)
        # Use hard-voxel splatting (sigma=0) — same as EulerianAdapter.obs_to_state
        occ = _particles_to_occupancy(pts_t, self.grid_bounds, self.grid_res, sigma=0.0)
        return float((occ.clamp(0.0, 1.0) * self.occ_score_tensor).reshape(1, -1).sum().item())

    def set_occupancy_params(self, grid_bounds: dict, grid_res: tuple,
                            occ_score_tensor: torch.Tensor):
        """
        Set occupancy grid parameters for compute_occ_reward_from_obs().

        Called by make_adapter to enable observation-based occupancy reporting.
        """
        self.grid_bounds = grid_bounds
        self.grid_res = grid_res
        self.occ_score_tensor = occ_score_tensor

    def format_states_pred(self, rollout_seq: list) -> np.ndarray:
        """list[(particle_num,3) ndarray] × n_ahead → (particle_num,3) last step."""
        return rollout_seq[-1]   # matches step_subgoal_ptcl's states_pred format

    def obs_to_report_state(self, obs_np: np.ndarray) -> torch.Tensor:
        """Return state for reporting reward evaluation.

        For eulerian and iou report types the reward operates on an occupancy
        grid derived from particles, so the quality of the measurement scales
        directly with the number of input points.  Using the coarse 50-particle
        MPC state yields a severely undersampled occupancy grid (<1.2 % voxel
        fill) with high per-step variance that obscures real progress.

        This method returns the *full* foreground point cloud extracted from the
        depth channel (same source used by EulerianAdapter.obs_to_state), so
        that _reward_eulerian / _reward_iou produce dense, low-noise measurements
        on the same scale as EulerianAdapter._reward_default.

        For the default report type the MPC particle state is sufficient, so
        this delegates to obs_to_state.
        """
        if self.reward_type_report in ('eulerian', 'iou'):
            depth  = obs_np[..., -1] / self.global_scale
            pts_np = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD,
                                 self.cam_params)
            if pts_np.shape[0] == 0:
                return torch.zeros((1, 1, 3), device=self.device)
            return (torch.from_numpy(pts_np.astype(np.float32))
                    .to(self.device).unsqueeze(0).detach())  # (1, N_full, 3)
        return self.obs_to_state(obs_np)

    def print_step_info(self, state: torch.Tensor, step: int, n_mpc: int):
        """Print particle-count diagnostics."""
        n_particles = state.shape[1]
        print(f"  step {step}: {n_particles} particles  "
              f"particle_dens={self._particle_dens:.2f}")

    def get_ot_grids(self, state: torch.Tensor) -> tuple:
        """OT grids not available for GNN adapter; returns (None, None)."""
        return None, None


# ─────────────────────────────────────────────────── factory ─────────────────

def make_adapter(model_dy,
                 env,
                 subgoal: np.ndarray,
                 cfg: dict,
                 cam_params,
                 device: str = 'cuda'):
    """
    Return the appropriate ModelAdapter for *model_dy*.

    Each adapter uses the reward function that naturally matches its state
    representation:
    - Eulerian: occupancy-based reward (state is occupancy grid)
    - GNN: particle-based reward (state is particles)

    Occupancy-based metrics can be computed separately for reporting/comparison.

    Parameters
    ----------
    model_dy   : the dynamics model (EulerianModelWrapper or PropNetDiffDenModel)
    env        : FlexEnv (already initialised and reset)
    subgoal    : (H, W) float32 distance-transform;  0 = inside goal
    cfg        : full config dict
    cam_params : camera intrinsics tuple from env.get_cam_params()
    device     : 'cuda' or 'cpu'
    """
    from model.eulerian_wrapper import EulerianModelWrapper
    from model.gnn_dyn import PropNetDiffDenModel

    reward_cfg      = cfg.get('mpc', {}).get('reward', {})
    empty_penalty   = float(reward_cfg.get('empty_penalty', 0.0))
    reward_type_opt = str(reward_cfg.get('opt_type', 'default'))
    reward_type_rep = reward_cfg.get('report_type', None)
    if reward_type_rep is not None:
        reward_type_rep = str(reward_type_rep)
    global_scale    = cfg['dataset']['global_scale']

    if isinstance(model_dy, EulerianModelWrapper):
        return EulerianAdapter(model_dy, subgoal, cam_params,
                               global_scale, device,
                               empty_penalty=empty_penalty,
                               reward_type_opt=reward_type_opt,
                               reward_type_report=reward_type_rep)

    if isinstance(model_dy, PropNetDiffDenModel):
        # GNN: create adapter with particle-based reward for optimization
        adapter = GNNAdapter(model_dy, env, subgoal, cam_params, cfg, device,
                             reward_type_opt=reward_type_opt,
                             reward_type_report=reward_type_rep)

        # Optionally set occupancy parameters for reporting (if needed)
        grid_bounds = EulerianModelWrapper.default_bounds(cfg)
        grid_res = (64, 64)  # standard occupancy grid resolution
        occ_reward = OccupancyReward(grid_bounds, grid_res, global_scale, cam_params)
        occ_score_tensor = occ_reward.compute_score_tensor(
            subgoal, device=device, empty_penalty=empty_penalty)
        adapter.set_occupancy_params(grid_bounds, grid_res, occ_score_tensor)

        return adapter

    raise NotImplementedError(
        f"No simple_mpc adapter for model type '{type(model_dy).__name__}'. "
        "Implement a ModelAdapter and register it in make_adapter()."
    )
