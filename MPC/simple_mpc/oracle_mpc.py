"""
Oracle sampling-based MPC — uses the Genesis simulator itself as the
prediction model, via ``GenesisOracleEnv``, and CEM/MPPI (``sampling_optimizers``)
instead of gradient descent (the simulator is not differentiable through
``execute_action``). See docs/oracle_mpc_design.md for the full design.

This module mirrors ``simple_mpc.mpc.run_simple_mpc``'s structure and
**returns the same result-dict schema**, so existing metric extraction /
plotting (``run_experiments.py``) works unchanged against oracle runs.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from utils import load_yaml, depth2fgpcd, write_video_frame
from model.eulerian_wrapper import EulerianModelWrapper
from model_training.losses import build_loss
from model_training.types import ModelOutput
from transforms.functional import particles_to_occupancy

from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.occupancy_reward import OccupancyReward
from simple_mpc.action_sampler import make_action_sampler
from simple_mpc.sampling_optimizers import make_sampling_optimizer

# foreground threshold — must match simple_mpc.adapters / simple_mpc.mpc
_FG_DEPTH_THRESHOLD = 0.599 / 0.8


def load_oracle_config(path: str = 'simple_mpc/config/config_oracle.yaml') -> dict:
    """Load the oracle-MPC config file."""
    return load_yaml(path)


def _raw_pts_from_obs(obs_np: np.ndarray, global_scale: float, cam_params) -> np.ndarray:
    depth = obs_np[..., -1] / global_scale
    return depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, cam_params)


def _report_occupancy_from_obs(
    obs_np: np.ndarray, global_scale: float, cam_params,
    grid_bounds: dict, grid_res: tuple, device: str,
) -> torch.Tensor:
    """Dense depth-derived occupancy for a rendered observation — (1, *grid_res).

    Uses the full foreground point cloud (not sparse particle centers), same
    source as ``EulerianAdapter.obs_to_state``, so oracle-MPC reward reporting
    is on the same scale as the learned-model MPC runs it's meant to bound.
    """
    depth  = obs_np[..., -1] / global_scale
    pts_np = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, cam_params)
    pts_t  = torch.from_numpy(pts_np.astype(np.float32)).to(device).unsqueeze(0)
    with torch.no_grad():
        return particles_to_occupancy(pts_t, grid_bounds, grid_res, sigma=0.0)


def _occupancy_reward(occ: torch.Tensor, score_tensor: torch.Tensor) -> float:
    """(1, *grid_res) occupancy × goal score map → scalar reward (higher = better)."""
    return float((occ.clamp(0.0, 1.0) * score_tensor).reshape(occ.shape[0], -1).sum().item())


def _per_sample_cost(
    occ_pred: torch.Tensor,       # (K, *grid_res)
    occ_cur: torch.Tensor,        # (1 or K, *grid_res)
    occ_goal: torch.Tensor,       # (1, *grid_res)
    score_tensor: torch.Tensor,   # (*grid_res)
    loss_fn,
) -> torch.Tensor:
    """Evaluate the configured loss per-candidate. Returns (K,) cost — lower is better."""
    K = occ_pred.shape[0]
    probs  = occ_pred.clamp(1e-4, 1.0 - 1e-4)
    logits = torch.logit(probs)
    prediction = ModelOutput(logits=logits, probabilities=occ_pred)
    cur  = occ_cur.expand(K, *occ_cur.shape[1:])
    goal = occ_goal.expand(K, *occ_goal.shape[1:])
    batch = {
        "input":             cur.unsqueeze(1),
        "current_occupancy": cur,
        "target":            goal,
        "target_occupancy":  goal,
        "score_map":         score_tensor,
    }
    total, _ = loss_fn(prediction, batch)
    return total   # (K,) since loss_fn was built with per_sample=True


def _discounted_cost(
    occ_steps: list,               # list[n_ahead] of (K, *grid_res)
    occ_cur: torch.Tensor,         # (1, *grid_res)
    occ_goal: torch.Tensor,
    score_tensor: torch.Tensor,
    loss_fn,
    gamma: float,
) -> torch.Tensor:
    """Sum of per-step per-sample costs, discounted so later steps count less."""
    total_cost = None
    occ_prev = occ_cur
    for s, occ_s in enumerate(occ_steps):
        c = _per_sample_cost(occ_s, occ_prev, occ_goal, score_tensor, loss_fn)
        w = gamma ** s
        total_cost = w * c if total_cost is None else total_cost + w * c
        occ_prev = occ_s
    return total_cost


def run_oracle_mpc(
    env: GenesisOracleEnv,       # already reset
    subgoal: np.ndarray,         # (H, W) float; 0 inside goal
    cfg: dict,                   # full config dict (e.g. from load_oracle_config)
    video_recorder=None,
    collect_raw_obs: bool = True,
    collect_states: bool = True,
    collect_states_pred: bool = True,
) -> dict:
    """
    Run CEM/MPPI oracle MPC (Genesis-as-model) and return a result dict
    compatible with ``run_simple_mpc``'s schema.

    Required cfg keys — see simple_mpc/config/config_oracle.yaml for the full
    set and defaults: cfg['mpc']['n_mpc'], ['n_look_ahead'], ['n_sample'],
    ['n_opt_iter'], ['optimizer'] ('cem' | 'mppi'), ['loss'], ['reward'].
    """
    mpc_cfg      = cfg['mpc']
    n_mpc        = int(mpc_cfg['n_mpc'])
    n_ahead      = int(mpc_cfg.get('n_look_ahead', 1))
    n_sample     = int(mpc_cfg.get('n_sample', env.n_envs))
    n_opt_iter   = int(mpc_cfg.get('n_opt_iter', 4))
    cost_mode    = mpc_cfg.get('cost_mode', 'terminal')
    gamma        = float(mpc_cfg.get('gamma', 0.9))
    optimizer_name = mpc_cfg.get('optimizer', 'cem')
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    wkspc_w      = cfg['dataset']['wkspc_w']
    global_scale = cfg['dataset']['global_scale']
    cam_params   = env.get_cam_params()

    # Physics-aware clip bounds — identical formula to simple_mpc.mpc.run_simple_mpc.
    plate_length = cfg['dataset'].get('plate_length', 0.04)
    plate_safety = cfg['dataset'].get('plate_safety_margin', 0.01)
    phy_v   = max(wkspc_w - plate_length / 2.0 - plate_safety, 0.0)
    clip_lo = np.array([-phy_v, -phy_v, -phy_v, -phy_v], dtype=np.float32)
    clip_hi = np.array([ phy_v,  phy_v,  phy_v,  phy_v], dtype=np.float32)

    grid_res    = tuple(mpc_cfg.get('grid_res', (64, 64)))
    grid_bounds = EulerianModelWrapper.default_bounds(cfg, convention='genesis')
    occ_reward  = OccupancyReward(grid_bounds, grid_res, global_scale, cam_params)

    reward_cfg    = mpc_cfg.get('reward', {})
    empty_penalty = float(reward_cfg.get('empty_penalty', 0.0))
    score_tensor  = occ_reward.compute_score_tensor(subgoal, device=device, empty_penalty=empty_penalty)
    goal_mask     = occ_reward.goal_occupancy_mask(subgoal).to(device)   # (1, *grid_res)

    loss_cfg = dict(mpc_cfg.get('loss', {}))
    loss_cfg['per_sample'] = True
    loss_fn = build_loss(loss_cfg)

    action_sampler = make_action_sampler(
        mpc_cfg.get('action_sampler', 'physics_aware'),
        plate_length=plate_length,
        safety_margin=plate_safety,
        grid_size=int(round(2 * wkspc_w * 1000)),
        wkspc_w=wkspc_w,
    )

    footprint_r = env.default_footprint_radius_voxels(grid_bounds, grid_res)

    # NOTE: n_ahead is held constant across the whole run (unlike
    # run_simple_mpc's min(n_look_ahead, n_mpc - i) shrinking horizon near
    # the episode's end) — the optimizer's mean/std tensors are shaped by
    # n_ahead at construction and are not resized mid-run. Rolling a horizon
    # slightly past the nominal n_mpc real steps near the end is harmless for
    # a planning-only rollout; only act_seq[0] is ever executed for real.
    optimizer = make_sampling_optimizer(optimizer_name, n_ahead, clip_lo, clip_hi, mpc_cfg, device=device)

    print(f"oracle_mpc: optimizer={optimizer_name}  n_mpc={n_mpc}  n_look_ahead={n_ahead}  "
          f"n_envs={env.n_envs}  n_sample={n_sample}  n_opt_iter={n_opt_iter}  "
          f"cost_mode={cost_mode}  workspace=[-{wkspc_w},{wkspc_w}]^2  sample_range=[±{phy_v:.4f}]")

    H, W = env.screenHeight, env.screenWidth
    rewards     = np.zeros(n_mpc + 1, dtype=np.float32)
    occ_rewards = np.zeros(n_mpc + 1, dtype=np.float32)
    raw_obs     = (np.zeros((n_mpc + 1, H, W, 5), dtype=np.float32)
                   if collect_raw_obs else None)
    states      = [] if collect_states else None
    actions     = np.zeros((n_mpc, 5), dtype=np.float32)
    states_pred = [] if collect_states_pred else None
    rew_means   = np.zeros((n_mpc, 1, n_opt_iter), dtype=np.float32)
    rew_stds    = np.zeros((n_mpc, 1, n_opt_iter), dtype=np.float32)
    best_rewards_per_step = []
    total_time = rollout_time = optim_time = 0.0
    iter_num = 0

    # ── t = 0: render and seed initial state ──────────────────────────────
    obs_cur  = env.render()
    if raw_obs is not None:
        raw_obs[0] = obs_cur
    occ_cur_report = _report_occupancy_from_obs(
        obs_cur, global_scale, cam_params, grid_bounds, grid_res, device)
    occ_cur_opt = env.particles_to_occ(
        env.current_particles_world(), grid_bounds, grid_res, footprint_r)
    rewards[0]     = _occupancy_reward(occ_cur_report, score_tensor)
    occ_rewards[0] = rewards[0]   # oracle has a single occupancy-reward path
    if states is not None:
        states.append(_raw_pts_from_obs(obs_cur, global_scale, cam_params))
    write_video_frame(obs_cur, video_recorder)

    print(f"  initial reward: {rewards[0]:.4f}")

    n_batches_per_iter = max(1, math.ceil(n_sample / env.n_envs))

    # ── MPC loop ────────────────────────────────────────────────────────────
    for i in range(n_mpc):
        t_step_start = time.time()

        # Freeze env 0's true current state ONCE for this MPC step. Every
        # rollout below restores FROM THIS SNAPSHOT, never from env 0's live
        # state — env 0 is itself one of the n_envs candidate workers (see
        # GenesisOracleEnv docstring) and gets a real candidate action
        # applied to it on every rollout, so its live state drifts after the
        # first one. Restoring from a stale/drifted live state instead of
        # this snapshot was the root cause of a large, compounding
        # predicted-vs-actual reward gap.
        step_snapshot = env.snapshot_particles()

        init_mean = (action_sampler.sample(1, n_ahead, clip_lo, clip_hi, device=device)[0].detach()
                     if i == 0 else optimizer.warm_start_mean())
        optimizer.reset(init_mean)

        t_rollout = t_optim = 0.0

        for it in range(n_opt_iter):
            t0 = time.time()
            cand_batches, cost_batches = [], []
            for _b in range(n_batches_per_iter):
                cand = optimizer.ask(env.n_envs)   # (n_envs, n_ahead, 4)
                if cost_mode == 'discounted':
                    _, step_pos = env.rollout_candidates(
                        cand, step_snapshot, collect_intermediate=True)
                    occ_steps = [env.particles_to_occ(p, grid_bounds, grid_res, footprint_r)
                                 for p in step_pos]
                    cost = _discounted_cost(occ_steps, occ_cur_opt, goal_mask,
                                             score_tensor, loss_fn, gamma)
                else:
                    pos = env.rollout_candidates(cand, step_snapshot)
                    occ = env.particles_to_occ(pos, grid_bounds, grid_res, footprint_r)
                    cost = _per_sample_cost(occ, occ_cur_opt, goal_mask, score_tensor, loss_fn)
                cand_batches.append(cand)
                cost_batches.append(cost)
            t1 = time.time()
            t_rollout += t1 - t0

            candidates = torch.cat(cand_batches, dim=0)
            costs      = torch.cat(cost_batches, dim=0)
            optimizer.tell(candidates, costs)

            rew_means[i, 0, it] = float((-costs).mean().item())
            rew_stds[i, 0, it]  = float(costs.std().item())
            t_optim += time.time() - t1
            iter_num += 1

        best_seq       = optimizer.best()                     # (n_ahead, 4)
        best_action_np = best_seq[0].detach().cpu().numpy()

        rollout_time += t_rollout
        optim_time   += t_optim

        # -- re-roll the winning sequence AT FULL (real-step) FIDELITY, from
        #    the same frozen snapshot, to record states_pred + a
        #    predicted-vs-actual sanity check. With use_rollout_fidelity=False
        #    this uses the exact same settle/clearance budgets as the real
        #    step below, so the two should match almost exactly (same
        #    deterministic sim, same starting state, same action — the model
        #    IS the simulator). Using the reduced rollout-fidelity budgets
        #    here (as an earlier version did) confounded this check with a
        #    genuine physics difference (less settle time = different
        #    resting positions), on top of comparing incompatible occupancy
        #    representations below. record=False: step() below records this
        #    exact transition for real; recording it here too would just
        #    duplicate it n_envs times. ----------------------------------------
        with torch.no_grad():
            winner_batch = best_seq.unsqueeze(0).expand(env.n_envs, -1, -1).contiguous()
            _, pred_step_pos = env.rollout_candidates(
                winner_batch, step_snapshot, collect_intermediate=True,
                use_rollout_fidelity=False, record=False)
        pred_occ_seq = [
            env.particles_to_occ(p[0:1], grid_bounds, grid_res, footprint_r)[0].cpu().numpy()
            for p in pred_step_pos
        ]
        if states_pred is not None:
            states_pred.append(np.stack(pred_occ_seq))         # (n_ahead, Nx, Ny)
        predicted_reward = _occupancy_reward(
            torch.from_numpy(pred_occ_seq[-1]).unsqueeze(0).to(device), score_tensor)
        best_rewards_per_step.append(predicted_reward)

        _sx, _sy, _ex, _ey = best_action_np
        _dxy   = math.hypot(_ex - _sx, _ey - _sy)
        _angle = (math.atan2(_ey - _sy, _ex - _sx) + math.pi / 2
                  if _dxy > 1e-6 else 0.0)
        best_action_5d = np.append(best_action_np, _angle).astype(np.float32)

        delta_r = rew_means[i, 0, -1] - rew_means[i, 0, 0]
        compute_step_time = t_rollout + t_optim
        print(f"  step {i+1:3d}/{n_mpc}  pred_reward={predicted_reward:.4f}  "
              f"Δr_optim={delta_r:.4f}  std_last={rew_stds[i,0,-1]:.4f}  "
              f"compute={compute_step_time:.2f}s  "
              f"act=[{_sx:.3f} {_sy:.3f} {_ex:.3f} {_ey:.3f}]  "
              f"angle={math.degrees(_angle):.1f}°")

        # -- restore the true state (the re-roll above mutated env 0's live
        #    state) THEN execute the best action for real, broadcast to all
        #    K envs. ----------------------------------------------------------
        env.restore_snapshot(step_snapshot)
        obs_next = env.step(best_action_np, video_recorder=video_recorder)

        occ_next_report = _report_occupancy_from_obs(
            obs_next, global_scale, cam_params, grid_bounds, grid_res, device)
        r_next = _occupancy_reward(occ_next_report, score_tensor)

        # Representation-consistent actual reward: same footprint-splat
        # function used for predicted_reward, applied to the REAL post-step
        # particle positions. This isolates the pure physics/determinism
        # check from the (expected, documented) gap against the dense
        # depth-render reward used for rewards[]/occ_rewards[] — those stay
        # on the report-reward convention for comparability with
        # learned-model MPC runs; this is purely a diagnostic.
        occ_cur_opt = env.particles_to_occ(
            env.current_particles_world(), grid_bounds, grid_res, footprint_r)
        r_next_matching = _occupancy_reward(occ_cur_opt, score_tensor)

        if raw_obs is not None:
            raw_obs[i + 1] = obs_next
        if states is not None:
            states.append(_raw_pts_from_obs(obs_next, global_scale, cam_params))
        actions[i]         = best_action_5d
        rewards[i + 1]     = r_next
        occ_rewards[i + 1] = r_next
        occ_cur_report     = occ_next_report

        total_time += time.time() - t_step_start
        # Occupied-voxel counts for both representations, at the SAME real
        # post-step state — a direct way to see whether the dense-vs-sparse
        # gap is a "coverage" effect (dense covers more of the 64x64 grid
        # than the footprint-splat union does) rather than a physics gap.
        # If the footprint deficit (dense_vox - footprint_vox) grows as
        # material clusters into the goal (i.e. as reward improves), that
        # points at footprint disks failing to fully union over tightly
        # packed / rotated particles — see
        # transforms.functional.footprint_radius_voxels' shape_factor.
        dense_vox     = float(occ_next_report.clamp(0.0, 1.0).sum().item())
        footprint_vox = float(occ_cur_opt.clamp(0.0, 1.0).sum().item())
        print(f"          env reward after step (dense/report): {r_next:.4f}  "
              f"gap vs predicted: {r_next - predicted_reward:+.4f}  |  "
              f"same-repr actual: {r_next_matching:.4f}  "
              f"gap vs predicted: {r_next_matching - predicted_reward:+.4f}  "
              f"(should be ≈0 — same-repr isolates the physics/determinism "
              f"check from the dense-vs-sparse occupancy representation gap)")
        print(f"          occupied voxels (of {grid_res[0] * grid_res[1]}): "
              f"dense={dense_vox:.0f}  footprint={footprint_vox:.0f}  "
              f"deficit={dense_vox - footprint_vox:+.0f}")

    compute_time = rollout_time + optim_time
    sim_time     = total_time - compute_time
    print(f"\noracle_mpc done: r_init={rewards[0]:.4f}  r_final={rewards[n_mpc]:.4f}  "
          f"total_time={total_time:.1f}s  (compute={compute_time:.1f}s  sim={sim_time:.1f}s)")

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
        'best_rewards_per_step': best_rewards_per_step,
        'particle_den_seq':      [],   # unused; present for schema compatibility
        'mpc_transitions':       None,  # not collected by oracle_mpc
    }
