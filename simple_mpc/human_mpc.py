"""
Human-demonstration episode session for the Genesis-oracle ceiling baseline.

``simple_mpc.oracle_mpc.run_oracle_mpc`` explores actions automatically
(CEM/MPPI); this module instead lets a human choose the action each real
step (optionally refined by a local grid search, see
``simple_mpc.human_grid_search``), to establish an upper bound on
achievable reward that isn't limited by the sampling optimizer or its
action-sampling prior. See ``docs/human_demo_design.md`` for the full design.

``HumanDemoSession`` owns one episode's worth of state and exposes two
calls, deliberately split apart (unlike ``run_oracle_mpc``'s single
automated loop) so a human — via a GUI or any other front end — can look at
the current state before deciding:

    propose(action5)   -> grid-search-refine a drawn action, WITHOUT
                          committing it (safe to call repeatedly)
    commit(action5=None) -> execute an action for real (default: the last
                          propose()'s winner), advancing the episode

This keeps ``run_oracle_mpc``'s automated loop completely untouched — this
is a parallel, interactive control flow over the same ``GenesisOracleEnv``,
not a variant bolted onto it.
"""

from __future__ import annotations

import os
import json

import numpy as np
import torch

from model.eulerian_wrapper import EulerianModelWrapper
from training.losses import build_loss

from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.occupancy_reward import OccupancyReward
from simple_mpc.human_grid_search import grid_search_refine, ACTION_DIM

# Reused, not reimplemented — same helpers run_oracle_mpc.run_oracle_mpc uses
# for reward reporting, so a human episode's rewards.npy is on the exact same
# scale as an automated oracle-MPC or learned-model MPC run's.
from simple_mpc.oracle_mpc import (
    _occupancy_reward,
    _report_occupancy_from_obs,
    _raw_pts_from_obs,
)


class HumanDemoSession:
    """One episode of human-piloted oracle-MPC. Call order per real step:
    ``propose()`` zero or more times, then ``commit()``; call ``finalize()``
    once at episode end.
    """

    def __init__(self, env: GenesisOracleEnv, subgoal: np.ndarray, cfg: dict):
        mpc_cfg   = cfg['mpc']
        human_cfg = cfg.get('human', {})
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.env          = env
        self.subgoal       = subgoal
        self.cfg            = cfg
        self.wkspc_w        = float(cfg['dataset']['wkspc_w'])
        self.global_scale   = float(cfg['dataset']['global_scale'])
        self.cam_params     = env.get_cam_params()

        # Physics-aware clip bounds for sx/sy/ex/ey — identical formula to
        # simple_mpc.mpc.run_simple_mpc / simple_mpc.oracle_mpc.run_oracle_mpc.
        # The 5th (angle) dimension is normalized [0, 1) by construction (see
        # transforms.functional.action_to_pose) so its bounds are fixed, not
        # derived from workspace geometry.
        plate_length = cfg['dataset'].get('plate_length', 0.04)
        plate_safety = cfg['dataset'].get('plate_safety_margin', 0.01)
        phy_v = max(self.wkspc_w - plate_length / 2.0 - plate_safety, 0.0)
        self.clip_lo = np.array([-phy_v, -phy_v, -phy_v, -phy_v, 0.0], dtype=np.float32)
        self.clip_hi = np.array([ phy_v,  phy_v,  phy_v,  phy_v, 1.0], dtype=np.float32)

        self.grid_res    = tuple(mpc_cfg.get('grid_res', (64, 64)))
        self.grid_bounds = EulerianModelWrapper.default_bounds(cfg, convention='genesis')
        occ_reward       = OccupancyReward(self.grid_bounds, self.grid_res, self.global_scale, self.cam_params)

        reward_cfg    = mpc_cfg.get('reward', {})
        empty_penalty = float(reward_cfg.get('empty_penalty', 0.0))
        self.score_tensor = occ_reward.compute_score_tensor(
            subgoal, device=self.device, empty_penalty=empty_penalty)
        self.score_np     = self.score_tensor.cpu().numpy()
        self.goal_mask    = occ_reward.goal_occupancy_mask(subgoal).to(self.device)

        loss_cfg = dict(mpc_cfg.get('loss', {}))
        loss_cfg['per_sample'] = True
        self.loss_fn = build_loss(loss_cfg)

        self.footprint_r = env.default_footprint_radius_voxels(self.grid_bounds, self.grid_res)

        self.grid_n = int(human_cfg.get('grid_n', 3))
        self.delta  = human_cfg.get('grid_delta', 0.15)
        # Soft cap only — a human episode isn't forced to stop here (see
        # finished()); it just mirrors run_oracle_mpc's n_mpc as a default
        # target so an equal-length comparison is easy to set up.
        self.n_mpc_cap = int(mpc_cfg.get('n_mpc', 20))

        # -- episode bookkeeping (growable — a human may stop early) --------
        self.step_idx            = 0
        self.rewards: list        = []
        self.occ_rewards: list     = []
        self.raw_obs: list         = []
        self.states: list          = []
        self.actions: list          = []
        self.states_pred: list       = []
        self.best_rewards_per_step: list = []
        self._step_snapshot          = None
        self._last_propose           = None

        self._seed_initial_state()

    # ------------------------------------------------------------------ #

    def _seed_initial_state(self) -> None:
        obs_cur = self.env.render()
        self.raw_obs.append(obs_cur)
        self.occ_cur_report = _report_occupancy_from_obs(
            obs_cur, self.global_scale, self.cam_params,
            self.grid_bounds, self.grid_res, self.device)
        self.occ_cur_opt = self.env.particles_to_occ(
            self.env.current_particles_world(), self.grid_bounds,
            self.grid_res, self.footprint_r)
        r0 = _occupancy_reward(self.occ_cur_report, self.score_tensor)
        self.rewards.append(r0)
        self.occ_rewards.append(r0)
        self.states.append(_raw_pts_from_obs(obs_cur, self.global_scale, self.cam_params))
        self.current_reward = r0

    # ------------------------------------------------------------------ #
    #  Action-space helper                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def default_angle_norm(sx: float, sy: float, ex: float, ey: float) -> float:
        """Perpendicular-to-travel yaw, normalized to [0, 1) — the same
        default every automated sampler/optimizer in this codebase always
        uses (see ``transforms.functional.action_to_pose``'s 4-component case).
        A sane starting point for a human to then drag/type away from."""
        import math
        dxy = math.hypot(ex - sx, ey - sy)
        if dxy <= 1e-6:
            return 0.0
        angle = math.atan2(ey - sy, ex - sx) + math.pi / 2
        return (angle % math.pi) / math.pi

    # ------------------------------------------------------------------ #
    #  Interactive step: propose (safe, repeatable) then commit            #
    # ------------------------------------------------------------------ #

    def propose(self, action5: np.ndarray) -> dict:
        """Grid-search-refine ``action5`` against the current committed
        state. Does not mutate episode bookkeeping — safe to call multiple
        times (e.g. the user nudges the drawn action and re-evaluates)
        before ``commit()``. See ``simple_mpc.human_grid_search`` for the
        two-pass (search, then full-fidelity re-roll of the winner) rationale.
        """
        action5 = np.asarray(action5, dtype=np.float32)
        if action5.shape != (ACTION_DIM,):
            raise ValueError(f"action5 must be shape ({ACTION_DIM},), got {action5.shape}")

        self._step_snapshot = self.env.snapshot_particles()
        result = grid_search_refine(
            self.env, self._step_snapshot, action5,
            self.occ_cur_opt, self.goal_mask, self.score_tensor, self.loss_fn,
            self.grid_bounds, self.grid_res, self.footprint_r,
            self.grid_n, self.delta, self.clip_lo, self.clip_hi, self.device)
        # rollout_candidates() (inside grid_search_refine) mutates env 0's
        # live state on every batch — restore before any subsequent
        # propose()/commit() call so "current state" stays correct (same
        # reasoning as GenesisOracleEnv's own docstring on snapshot/restore).
        self.env.restore_snapshot(self._step_snapshot)
        self._last_propose = result
        return result

    def commit(self, action5: np.ndarray | None = None) -> dict:
        """Execute ``action5`` for real and advance the episode by one step.

        ``action5=None`` (default) executes the last ``propose()``'s winning
        action. Pass an explicit action to skip refinement entirely (e.g.
        ``grid_n=1`` / a "just run what I drew" mode).
        """
        if action5 is None:
            if self._last_propose is None:
                raise RuntimeError("commit() with no action requires a prior propose() call")
            action5 = self._last_propose['best_action']
        action5 = np.asarray(action5, dtype=np.float32)
        if action5.shape != (ACTION_DIM,):
            raise ValueError(f"action5 must be shape ({ACTION_DIM},), got {action5.shape}")

        if self._step_snapshot is None:
            self._step_snapshot = self.env.snapshot_particles()
        # Undo whatever propose()'s grid search (or an earlier stale
        # rollout) left env 0 in, then execute the real action from the true
        # current state — mirrors run_oracle_mpc's restore-then-step order.
        self.env.restore_snapshot(self._step_snapshot)
        obs_next = self.env.step(action5)

        occ_next_report = _report_occupancy_from_obs(
            obs_next, self.global_scale, self.cam_params,
            self.grid_bounds, self.grid_res, self.device)
        r_next = _occupancy_reward(occ_next_report, self.score_tensor)
        prev_reward = self.current_reward

        self.raw_obs.append(obs_next)
        self.states.append(_raw_pts_from_obs(obs_next, self.global_scale, self.cam_params))
        self.actions.append(action5.copy())
        self.rewards.append(r_next)
        self.occ_rewards.append(r_next)
        if self._last_propose is not None:
            self.states_pred.append(self._last_propose['occ_pred'][None])   # (1, Nx, Ny)
            self.best_rewards_per_step.append(self._last_propose['predicted_reward'])
        else:
            self.states_pred.append(None)
            self.best_rewards_per_step.append(r_next)

        self.occ_cur_report = occ_next_report
        self.occ_cur_opt = self.env.particles_to_occ(
            self.env.current_particles_world(), self.grid_bounds,
            self.grid_res, self.footprint_r)
        self.current_reward = r_next
        self.step_idx += 1
        self._step_snapshot = None
        self._last_propose = None

        return {'obs': obs_next, 'reward': r_next, 'gain': r_next - prev_reward}

    def finished(self) -> bool:
        """Whether the episode has reached its (soft) step cap. A human
        front end is free to ignore this and keep going, or stop earlier —
        it exists so an episode's default length matches ``mpc.n_mpc``,
        making an equal-length comparison against an automated run easy."""
        return self.step_idx >= self.n_mpc_cap

    # ------------------------------------------------------------------ #

    def finalize(self) -> dict:
        """Package everything committed so far into ``run_oracle_mpc``'s
        result-dict schema (variable length — unlike the automated loop, a
        human episode isn't bound to a fixed ``n_mpc``)."""
        return {
            'rewards':               np.array(self.rewards, dtype=np.float32),
            'occ_rewards':           np.array(self.occ_rewards, dtype=np.float32),
            'raw_obs':               np.stack(self.raw_obs, axis=0) if self.raw_obs else None,
            'states':                self.states,
            'actions':               (np.stack(self.actions, axis=0) if self.actions
                                       else np.zeros((0, ACTION_DIM), dtype=np.float32)),
            'states_pred':           self.states_pred,
            'best_rewards_per_step': self.best_rewards_per_step,
            'n_steps':               self.step_idx,
        }


def _save_object_array(items: list, path: str) -> None:
    """Save a list of variable-shaped arrays as an object-dtype .npy — same
    convention as run_oracle_mpc.py's helper of the same name."""
    arr = np.empty(len(items), dtype=object)
    for k, s in enumerate(items):
        arr[k] = s
    np.save(path, arr, allow_pickle=True)


def _plot_episode_reward(rewards: np.ndarray, out_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    steps = np.arange(len(rewards))
    ax.plot(steps, rewards, 'o-', lw=1.5, color='seagreen')
    ax.fill_between(steps, rewards[0], rewards, alpha=0.12, color='seagreen')
    ax.axhline(rewards[0], color='gray', lw=0.8, linestyle='--', label='initial')
    ax.set_xlabel('step')
    ax.set_ylabel('reward')
    ax.set_title(f'Human demo reward  (gain={rewards[-1] - rewards[0]:+.4f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def save_episode(result: dict, ep_dir: str, ep_idx: int, seed: int, elapsed: float) -> dict:
    """Save a finalized human-demo episode using the same per-file
    conventions as run_oracle_mpc.py's run_one_episode, so both are
    interchangeable when comparing reward curves offline."""
    os.makedirs(ep_dir, exist_ok=True)
    np.save(os.path.join(ep_dir, 'rewards.npy'), result['rewards'])
    np.save(os.path.join(ep_dir, 'occ_rewards.npy'), result['occ_rewards'])
    np.save(os.path.join(ep_dir, 'actions.npy'), result['actions'])
    if result['raw_obs'] is not None:
        np.save(os.path.join(ep_dir, 'raw_obs.npy'), result['raw_obs'])
    _save_object_array(result['states'], os.path.join(ep_dir, 'states.npy'))
    _save_object_array(result['states_pred'], os.path.join(ep_dir, 'states_pred.npy'))
    if len(result['rewards']) > 1:
        _plot_episode_reward(result['rewards'], os.path.join(ep_dir, 'rewards.png'))

    metrics = {
        'episode':      ep_idx,
        'seed':         seed,
        'elapsed_sec':  elapsed,
        'n_steps':      result['n_steps'],
        'reward_init':  float(result['rewards'][0]),
        'reward_final': float(result['rewards'][-1]),
        'reward_gain':  float(result['rewards'][-1] - result['rewards'][0]),
        'source':       'human_demo',
    }
    with open(os.path.join(ep_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics
