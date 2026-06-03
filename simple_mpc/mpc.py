"""Core gradient-descent MPC loop — model-agnostic via ModelAdapters.

This module contains the main MPC entry point ``run_simple_mpc`` and the
config loader ``load_simple_config``.

Model-specific logic (state extraction, forward prediction, reward) lives in
``simple_mpc.adapters``.  Debug visualisation and benchmarking live in the
sibling modules ``debug_vis`` and ``benchmark``.

Supported models (auto-detected from the model_dy argument):
  * EulerianModelWrapper   → EulerianAdapter  (occupancy-grid MPC)
  * PropNetDiffDenModel    → GNNAdapter        (particle-cloud MPC)
"""

import math
import os
import time
import numpy as np
import torch
import torch.optim as optim

from utils import load_yaml, depth2fgpcd
from model.eulerian_wrapper import _action_to_cam_3d

from simple_mpc.adapters import make_adapter
from simple_mpc.action_sampler import make_action_sampler
from simple_mpc.benchmark import benchmark_push_throughput
from simple_mpc.debug_vis import (
    save_debug_candidates,
    save_predicted_trajectory_video,
    save_debug_winner,
)

# foreground threshold — must match adapters._FG_DEPTH_THRESHOLD
_FG_DEPTH_THRESHOLD = 0.599 / 0.8


def _raw_pts_from_obs(obs_np: np.ndarray,
                      global_scale: float,
                      cam_params) -> np.ndarray:
    """Return raw foreground point cloud (N, 3) from a rendered observation."""
    depth = obs_np[..., -1] / global_scale
    return depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, cam_params)


# ── public helpers ────────────────────────────────────────────────────────────

def load_simple_config(path: str = 'config/mpc/config_simple.yaml') -> dict:
    """Load the simple-MPC config file."""
    return load_yaml(path)


# ── main entry point ──────────────────────────────────────────────────────────


def _compute_occ_reward(adapter, state: torch.Tensor,
                        obs_np = None) -> float:
    """Compute occupancy-based reward for model-invariant reporting.

    For GNN adapters: uses the full raw depth point cloud from *obs_np*
    (via compute_occ_reward_from_obs) so occupancy density matches the
    Eulerian adapter, which is also derived from the full depth image.

    Falls back to compute_reward() when obs_np is not available or the
    adapter does not implement compute_occ_reward_from_obs (e.g. Eulerian,
    whose native reward is already occupancy-based at full density).
    """
    if obs_np is not None and hasattr(adapter, 'compute_occ_reward_from_obs'):
        r = adapter.compute_occ_reward_from_obs(obs_np)
        if r is not None:
            return float(r)
    return adapter.compute_reward(state).item()


def run_simple_mpc(
    env,                        # FlexEnv (already reset; call set_save_render_mode
                                #   before calling if you want rendered frames saved)
    model_dy,                   # EulerianModelWrapper | PropNetDiffDenModel
    subgoal: np.ndarray,        # (H, W)  float;  0 inside goal region
    cfg:     dict,              # full config dict (e.g. from load_simple_config)
    video_recorder=None,        # optional recorder passed to env.step()
    action_sampler_type: str = 'physics_aware',  # type of sampler ('physics_aware' or 'uniform')
    collect_raw_obs: bool = True,
    collect_states: bool = True,
    collect_states_pred: bool = True,
    collect_mpc_transitions: bool = False,
) -> dict:
    """
    Run simple gradient-descent MPC and return a result dict compatible with
    env.step_subgoal_ptcl().

    The loop is model-agnostic: all model-specific operations are dispatched
    through a ModelAdapter (created automatically from *model_dy*).

    Parameters
    ----------
    cfg : dict
        Must contain (at minimum):
          cfg['dataset']['wkspc_w']
          cfg['dataset']['global_scale']
          cfg['mpc']['n_mpc']
          cfg['mpc']['n_look_ahead']
          cfg['mpc']['n_sample']
          cfg['mpc']['n_update_iter']
          cfg['mpc']['gd']['lr']
        GNN model additionally requires:
          cfg['mpc']['particle_num']   (default 50)
    action_sampler_type : str
        Name of the action sampling strategy ('uniform' or custom).
        See simple_mpc.action_sampler for available samplers.
    """

    # ── read config ────────────────────────────────────────────────────────────
    mpc_cfg       = cfg['mpc']
    n_mpc         = mpc_cfg['n_mpc']
    n_look_ahead  = mpc_cfg['n_look_ahead']
    n_sample      = mpc_cfg['n_sample']
    n_update_iter = mpc_cfg['n_update_iter']
    gd_lr         = mpc_cfg['gd']['lr']
    device        = 'cuda' if torch.cuda.is_available() else 'cpu'

    debug_cfg          = mpc_cfg.get('debug', {})
    debug_enabled      = bool(debug_cfg.get('enabled', False))
    debug_dir          = debug_cfg.get('output_dir', 'outputs/simple_mpc_debug')
    debug_top_k        = int(debug_cfg.get('top_k', 5))
    debug_every        = int(debug_cfg.get('save_every_n_iters', 25))
    debug_bench        = bool(debug_cfg.get('benchmark_throughput', False))

    wkspc_w      = cfg['dataset']['wkspc_w']
    global_scale = cfg['dataset']['global_scale']
    cam_params   = env.get_cam_params()

    # ── workspace bounds ──────────────────────────────────────────────────────
    # act_lo / act_hi  = raw workspace bounds passed to the sampler so it can
    # read the workspace half-width and subtract the plate AABB internally.
    # clip_lo / clip_hi = physics-aware bounds used for gradient-step clipping:
    #   conservative AABB = plate_length/2 (worst-case for any travel direction)
    #   → v = wkspc_w − L/2 − safety_margin
    act_lo = np.array([-wkspc_w, -wkspc_w, -wkspc_w, -wkspc_w], dtype=np.float32)
    act_hi = np.array([ wkspc_w,  wkspc_w,  wkspc_w,  wkspc_w], dtype=np.float32)
    plate_length = cfg['dataset'].get('plate_length', 0.04)
    plate_safety = cfg['dataset'].get('plate_safety_margin', 0.01)
    phy_v  = max(wkspc_w - plate_length / 2.0 - plate_safety, 0.0)
    clip_lo = np.array([-phy_v, -phy_v, -phy_v, -phy_v], dtype=np.float32)
    clip_hi = np.array([ phy_v,  phy_v,  phy_v,  phy_v], dtype=np.float32)

    print(f"simple_mpc: model={type(model_dy).__name__}  "
          f"n_mpc={n_mpc}, n_look_ahead={n_look_ahead}, "
          f"n_sample={n_sample}, n_update_iter={n_update_iter}, "
          f"lr={gd_lr}, workspace=[-{wkspc_w},{wkspc_w}]^2  "
          f"sample_range=[±{phy_v:.4f}]")

    # ── create model adapter (dispatches all model-specific ops) ───────────────
    adapter = make_adapter(model_dy, env, subgoal, cfg, cam_params, device)

    # ── create action sampler ────────────────────────────────────────────────
    # Infer grid_size from model's native resolution (Eulerian), or fall back
    # to the workspace pixel width (128 px for 0.064 m half-width).
    _m  = getattr(adapter, 'model_dy', None)
    _gr = getattr(_m, 'grid_res', None)
    grid_size = int(_gr[0]) if _gr is not None else int(round(2 * wkspc_w * 1000))

    # sampler type: function-param default is overridden by cfg['mpc']['action_sampler']
    sampler_type  = mpc_cfg.get('action_sampler', action_sampler_type)
    _coll_kwargs  = mpc_cfg.get('collision_sampler', {})
    _ot_kwargs    = mpc_cfg.get('ot_sampler', {})
    action_sampler = make_action_sampler(
        sampler_type,
        plate_length=plate_length,
        safety_margin=plate_safety,
        grid_size=grid_size,
        wkspc_w=wkspc_w,
        **{**_coll_kwargs, **_ot_kwargs},
    )

    # ── get reward functions (can differ between optimization & reporting) ─────
    reward_fn_opt    = adapter.get_reward_fn_opt()
    reward_fn_report = adapter.get_reward_fn_report()

    if debug_bench and adapter.debug_vis_enabled:
        benchmark_push_throughput(model_dy, wkspc_w=wkspc_w, device=device)

    # ── Eulerian-only: print score map info ──────────────────────────────────
    if hasattr(adapter, 'score_np'):
        _snp = adapter.score_np
        _gv  = int((_snp > 0).sum())
        print(f"  score map: {_gv}/{_snp.size} voxels in goal region "
              f"({100*_gv/_snp.size:.1f}%),  "
              f"range=[{_snp.min():.3f}, {_snp.max():.3f}]  "
              f"shape={_snp.shape}")

    # ── preallocate return arrays ──────────────────────────────────────────────
    H, W        = env.screenHeight, env.screenWidth
    rewards     = np.zeros(n_mpc + 1,           dtype=np.float32)
    occ_rewards = np.zeros(n_mpc + 1,           dtype=np.float32)
    raw_obs     = (np.zeros((n_mpc + 1, H, W, 5), dtype=np.float32)
                   if collect_raw_obs else None)
    states      = [] if collect_states else None   # list[n_mpc+1] of (N, 3) camera-space point clouds
    actions     = np.zeros((n_mpc, 5),           dtype=np.float32)  # [sx,sy,ex,ey,angle_rad]
    states_pred           = [] if collect_states_pred else None
    rew_means             = np.zeros((n_mpc, 1, n_update_iter), dtype=np.float32)
    rew_stds              = np.zeros((n_mpc, 1, n_update_iter), dtype=np.float32)
    best_rewards_per_step = []   # list[n_mpc] of float  — best predicted reward per step
    total_time = rollout_time = optim_time = 0.0
    iter_num   = 0

    # ── mpc transition collection ─────────────────────────────────────────────
    _collect_trans = (collect_mpc_transitions
                      and hasattr(env, '_sim')
                      and hasattr(env._sim, '_particle_state'))
    _trans_states:   list = []
    _trans_states_:  list = []
    _trans_p_starts: list = []
    _trans_p_stops:  list = []
    _trans_angles:   list = []

    # ── t = 0: render and seed initial state ──────────────────────────────────
    obs_cur    = env.render()
    if raw_obs is not None:
        raw_obs[0] = obs_cur
    state_cur      = adapter.obs_to_state(obs_cur)               # (1, *S) — model state
    rewards[0]     = reward_fn_report(
        adapter.obs_to_report_state(obs_cur)).item()
    occ_rewards[0] = _compute_occ_reward(adapter, state_cur, obs_cur)
    if states is not None:
        states.append(_raw_pts_from_obs(obs_cur, global_scale, cam_params))

    # Write initial frame (t=0) to video so the video starts before the first push
    if video_recorder is not None:
        _writer = video_recorder[0] if isinstance(video_recorder, list) else video_recorder
        _bgr = np.ascontiguousarray(
            (np.clip(obs_cur[..., :3], 0.0, 1.0) * 255).astype(np.uint8)[..., ::-1]
        )
        _writer.write(_bgr)

    print(f"  initial reward: {rewards[0]:.4f}")

    # ── MPC loop ──────────────────────────────────────────────────────────────
    for i in range(n_mpc):
        t_step_start = time.time()
        n_ahead      = min(n_look_ahead, n_mpc - i)
        state_init   = state_cur   # (1, *S) — adapter-specific state

        adapter.print_step_info(state_init, i + 1, n_mpc)

        debug_step_dir = os.path.join(debug_dir, f'step_{i+1:03d}')

        # -- optionally seed sampler with current state (OT / collision-aware) --
        if hasattr(action_sampler, 'update_state'):
            _src, _goal = adapter.get_ot_grids(state_init)
            if _src is not None:
                action_sampler.update_state(_src, _goal)

        # -- sample candidate action sequences --------------------------------
        act_seqs = action_sampler.sample(
            n_sample, n_ahead, act_lo, act_hi, device=device
        )
        optimizer = optim.Adam([act_seqs], lr=gd_lr, betas=(0.9, 0.999))

        best_reward = -float('inf')
        best_act    = act_seqs[0].detach().clone()   # (n_ahead, 4)
        best_rewards_per_step.append(None)           # placeholder; updated below

        t_rollout = 0.0
        t_optim   = 0.0

        for it in range(n_update_iter):
            # -- rollout n_sample candidates through dynamics model -----------
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()

            state_batch = adapter.expand_state(state_init, n_sample)
            for s in range(n_ahead):
                state_batch = adapter.predict_step(state_batch, act_seqs[:, s, :])

            ev1.record()
            torch.cuda.synchronize()
            t_rollout += ev0.elapsed_time(ev1)

            # -- reward -------------------------------------------------------
            rew_seqs = reward_fn_opt(state_batch)   # (n_sample,)
            best_i   = int(rew_seqs.argmax())
            if rew_seqs[best_i].item() > best_reward:
                best_reward = rew_seqs[best_i].item()
                best_act    = act_seqs[best_i].detach().clone()

            rew_means[i, 0, it] = float(rew_seqs.mean().item())
            rew_stds[i,  0, it] = float(rew_seqs.std().item())

            # -- debug: snapshot top-K candidates (Eulerian only) ------------
            if debug_enabled and adapter.debug_vis_enabled and (
                    it == 0
                    or (it + 1) % debug_every == 0
                    or it == n_update_iter - 1):
                _k       = min(debug_top_k, n_sample)
                topk_idx = rew_seqs.detach().cpu().numpy().argsort()[::-1][:_k].copy()
                _topk_act_t = act_seqs.detach()[topk_idx, 0, :]   # (K, 4) world 2D
                _s_cam, _e_cam = _action_to_cam_3d(
                    _topk_act_t, adapter.model_dy.cam_extrinsic,
                    adapter.model_dy.global_scale)
                _sg = adapter.model_dy._cam3d_to_grid(_s_cam).cpu().numpy()
                _eg = adapter.model_dy._cam3d_to_grid(_e_cam).cpu().numpy()
                save_debug_candidates(
                    path=os.path.join(debug_step_dir,
                                      f'iter_{it:04d}_top{_k}.png'),
                    occ_init_np=adapter.debug_occ(state_init),
                    topk_occ_nps=[state_batch.detach().cpu().numpy()[j]
                                  for j in topk_idx],
                    topk_acts=act_seqs.detach().cpu().numpy()[topk_idx, 0, :],
                    topk_rews=rew_seqs.detach().cpu().numpy()[topk_idx],
                    score_np=adapter.score_np,
                    step=i + 1, it=it,
                    topk_start_grids=_sg[:, :2],
                    topk_end_grids=_eg[:, :2],
                )

            # -- gradient step ------------------------------------------------
            ev2 = torch.cuda.Event(enable_timing=True)
            ev3 = torch.cuda.Event(enable_timing=True)
            ev2.record()

            loss = -rew_seqs.sum()
            torch.nn.utils.clip_grad_norm_([act_seqs], max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            # -- gradient norm: near-zero means the push misses the material --
            if debug_enabled and (
                    it == 0
                    or (it + 1) % debug_every == 0
                    or it == n_update_iter - 1):
                _gn = (act_seqs.grad.abs().mean().item()
                       if act_seqs.grad is not None else float('nan'))
                print(f"    iter {it+1:4d}: best={best_reward:.4f}  "
                      f"mean={rew_seqs.mean().item():.4f}  "
                      f"std={rew_seqs.std().item():.4f}  "
                      f"grad={_gn:.5f}")
            optimizer.step()

            # clip all candidates into physics-aware bounds (dimension-by-dimension
            # to avoid any tensor-bound broadcasting issues across pyTorch versions)
            with torch.no_grad():
                for d in range(4):
                    act_seqs.data[:, :, d].clamp_(float(clip_lo[d]), float(clip_hi[d]))

            ev3.record()
            torch.cuda.synchronize()
            t_optim += ev2.elapsed_time(ev3)

        rollout_time += t_rollout / 1000.0   # ms → s
        optim_time   += t_optim  / 1000.0
        iter_num     += n_update_iter
        best_rewards_per_step[-1] = best_reward   # record final best for this step

        # -- rollout best action sequence → states_pred -----------------------
        with torch.no_grad():
            pred_seq   = []
            pred_batch = adapter.expand_state(state_init, 1)
            for s in range(n_ahead):
                pred_batch = adapter.predict_step(pred_batch, best_act[s:s+1])
                pred_seq.append(pred_batch[0].cpu().numpy())   # (*S,) ndarray
        if states_pred is not None:
            states_pred.append(adapter.format_states_pred(pred_seq))

        # -- save predicted trajectory video (Eulerian only) ------------------
        if debug_enabled and adapter.debug_vis_enabled:
            save_predicted_trajectory_video(
                path=os.path.join(debug_step_dir, 'predicted_trajectory.avi'),
                occ_init_np=adapter.debug_occ(state_init),
                occ_seq=pred_seq,
                score_np=adapter.score_np,
                step=i + 1,
            )

        best_action_np = best_act[0].cpu().numpy()   # (4,) — first look-ahead

        # Compute the plate orientation angle (same formula as genesis_env.step)
        _sx, _sy, _ex, _ey = best_action_np
        _dxy   = math.hypot(_ex - _sx, _ey - _sy)
        _angle = (math.atan2(_ey - _sy, _ex - _sx) + math.pi / 2
                  if _dxy > 1e-6 else 0.0)
        best_action_5d = np.append(best_action_np, _angle).astype(np.float32)

        delta_r = rew_means[i, 0, -1] - rew_means[i, 0, 0]
        compute_step_time = (t_rollout + t_optim) / 1000.0   # ms → s
        print(f"  step {i+1:3d}/{n_mpc}  best_rew={best_reward:.4f}  "
              f"Δr_optim={delta_r:.4f}  std_last={rew_stds[i,0,-1]:.4f}  "
              f"compute={compute_step_time:.2f}s  "
              f"act=[{_sx:.3f} {_sy:.3f} {_ex:.3f} {_ey:.3f}]  "
              f"angle={math.degrees(_angle):.1f}°")

        # -- execute best action in simulator ---------------------------------
        if _collect_trans:
            _trans_states.append(env._sim._particle_state[0].detach().cpu().clone())
        obs_next = env.step(best_action_np, video_recorder=video_recorder)
        if obs_next is None:
            print("WARNING: simulation exploded — terminating MPC early")
            # Zero-fill remaining entries so output dict shape is consistent
            for j in range(i + 1, n_mpc):
                if states is not None and len(states) > 0:
                    states.append(states[-1])
                if states_pred is not None and len(states_pred) > 0:
                    states_pred.append(states_pred[-1])
            break

        if raw_obs is not None:
            raw_obs[i + 1] = obs_next
        state_cur       = adapter.obs_to_state(obs_next)
        r_next          = reward_fn_report(
            adapter.obs_to_report_state(obs_next)).item()
        if states is not None:
            states.append(_raw_pts_from_obs(obs_next, global_scale, cam_params))
        actions[i]         = best_action_5d
        rewards[i + 1]     = r_next
        occ_rewards[i + 1] = _compute_occ_reward(adapter, state_cur, obs_next)
        if _collect_trans:
            _z = float(env._sim._operation_height)
            _trans_states_.append(env._sim._particle_state[0].detach().cpu().clone())
            _trans_p_starts.append(torch.tensor([_sx, _sy, _z], dtype=torch.float32))
            _trans_p_stops.append(torch.tensor([_ex, _ey, _z], dtype=torch.float32))
            _trans_angles.append(float(_angle))

        # -- debug: winner panel (Eulerian only) ------------------------------
        if debug_enabled and adapter.debug_vis_enabled:
            save_debug_winner(
                path=os.path.join(debug_step_dir, 'winner.png'),
                occ_init_np=adapter.debug_occ(state_init),
                occ_pred_np=pred_seq[0],
                score_np=adapter.score_np,
                best_act=best_action_np,
                predicted_r=best_reward,
                actual_r=float(r_next),
                step=i + 1,
            )

        total_time += time.time() - t_step_start
        print(f"          env reward after step: {r_next:.4f}  "
              f"(predicted: {best_reward:.4f}  gap: {float(r_next)-best_reward:+.4f})")

    compute_time = rollout_time + optim_time
    sim_time     = total_time - compute_time
    print(f"\nsimple_mpc done: r_init={rewards[0]:.4f}  r_final={rewards[n_mpc]:.4f}  "
          f"total_time={total_time:.1f}s  "
          f"(compute={compute_time:.1f}s  sim={sim_time:.1f}s)")

    return {
        'rewards':          rewards,
        'occ_rewards':      occ_rewards,
        'raw_obs':          raw_obs,
        'states':           states,
        'actions':          actions,
        'states_pred':      states_pred,
        'rew_means':        rew_means,
        'rew_stds':         rew_stds,
        'total_time':       total_time,
        'rollout_time':     rollout_time,
        'optim_time':       optim_time,
        'iter_num':         iter_num,
        'best_rewards_per_step': best_rewards_per_step,   # list[n_mpc] of float
        'particle_den_seq':      [],   # unused; present for API compatibility
        'mpc_transitions': {
            'states':   torch.stack(_trans_states),
            'states_':  torch.stack(_trans_states_),
            'p_starts': torch.stack(_trans_p_starts),
            'p_stops':  torch.stack(_trans_p_stops),
            'angles':   torch.tensor(_trans_angles, dtype=torch.float32),
        } if (_collect_trans and _trans_states_) else None,
    }
