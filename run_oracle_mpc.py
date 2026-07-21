#!/usr/bin/env python3
"""
run_oracle_mpc.py — Genesis-as-model ceiling-baseline MPC entry point.

Builds one batched GenesisOracleEnv (shared across episodes — Genesis only
allows gs.init() once per process) and runs CEM/MPPI sampling MPC where the
Genesis simulator itself is the prediction model (see
docs/oracle_mpc_design.md), for ``episodes.n_episodes`` independent episodes.

Each episode gets its own output subdirectory with enough saved data to
recreate its trajectory visually offline (raw RGB+depth frames, an .avi
video if --save-video is passed, and the foreground point cloud per step),
plus the quantitative MPC trace (rewards, actions, per-iteration optimizer
stats, predicted-vs-actual occupancy). A run-level summary aggregates reward
curves across all episodes.

This is a standalone script (not yet wired into run_experiments.py's batch
runner — see docs/oracle_mpc_design.md's Known Limitations).

Usage
-----
    python run_oracle_mpc.py
    python run_oracle_mpc.py --config simple_mpc/config/config_oracle_test.yaml --save-video
    python run_oracle_mpc.py --n-envs 16 --optimizer mppi --n-episodes 3
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GenesisPath = os.path.join(os.path.dirname(__file__), 'Genesis')
sys.path.append(GenesisPath)

from utils import (
    gen_subgoal,
    gen_goal_shape,
    set_seed,
    scale_subgoal_to_material_pixels,
    get_current_YYYY_MM_DD_hh_mm_ss_ms,
)
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.oracle_mpc import run_oracle_mpc, load_oracle_config


def build_goal(cfg: dict, env: GenesisOracleEnv) -> np.ndarray:
    """Return the subgoal distance map, mirroring run_experiments.py::build_goal."""
    task  = cfg['mpc']['task']
    H, W  = env.screenHeight, env.screenWidth
    ttype = task['type']

    if ttype == 'target_control':
        subgoal, _mask = gen_subgoal(
            task['goal_row'], task['goal_col'], task['goal_r'], h=H, w=W)
    elif ttype == 'target_shape':
        subgoal, _goal_img = gen_goal_shape(task['target_char'], h=H, w=W)
    else:
        raise NotImplementedError(f"Unknown task type: {ttype!r}")

    return subgoal


def _save_object_array(items: list, path: str) -> None:
    """Save a list of variable-shaped arrays as an object-dtype .npy (matches
    run_experiments.py's save_states convention)."""
    arr = np.empty(len(items), dtype=object)
    for k, s in enumerate(items):
        arr[k] = s
    np.save(path, arr, allow_pickle=True)


def _plot_episode_reward(rewards: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    steps = np.arange(len(rewards))
    ax.plot(steps, rewards, 'o-', lw=1.5, color='steelblue')
    ax.fill_between(steps, rewards[0], rewards, alpha=0.12, color='steelblue')
    ax.axhline(rewards[0], color='gray', lw=0.8, linestyle='--', label='initial')
    ax.set_xlabel('MPC step')
    ax.set_ylabel('Reward')
    ax.set_title(f'Reward  (gain={rewards[-1] - rewards[0]:+.4f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _plot_summary_reward(rewards_mat: np.ndarray, out_path: str) -> None:
    mean  = rewards_mat.mean(axis=0)
    std   = rewards_mat.std(axis=0)
    steps = np.arange(rewards_mat.shape[1])

    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rewards_mat:
        ax.plot(steps, row, 'k-', alpha=0.15, lw=0.7)
    ax.plot(steps, mean, 'o-', lw=2, color='steelblue',
             label=f'mean (n={len(rewards_mat)})')
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25,
                     color='steelblue', label='± 1 std')
    ax.set_xlabel('MPC step')
    ax.set_ylabel('Reward')
    ax.set_title('oracle_mpc — reward over steps (all episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='simple_mpc/config/config_oracle.yaml')
    p.add_argument('--seed', type=int, default=None,
                    help='override episodes.random_seed_base')
    p.add_argument('--n-episodes', type=int, default=None,
                    help='override episodes.n_episodes')
    p.add_argument('--n-envs', type=int, default=None, help='override mpc.n_envs')
    p.add_argument('--optimizer', choices=['cem', 'mppi'], default=None,
                    help='override mpc.optimizer')
    p.add_argument('--n-mpc', type=int, default=None, help='override mpc.n_mpc')
    p.add_argument('--output-dir', default=None, help='override mpc.output_dir')
    p.add_argument('--save-video', action='store_true',
                    help='also save an .avi per episode (raw_obs.npy is saved '
                         'regardless, and is enough to recreate frames offline)')
    return p.parse_args()


def run_one_episode(env, cfg: dict, ep_idx: int, seed: int, ep_dir: str,
                     save_video: bool) -> tuple:
    """Reset, plan+act for cfg['mpc']['n_mpc'] steps, save everything needed
    to recreate this episode's trajectory later, return (rewards, metrics)."""
    set_seed(seed)
    env.reset()
    # Tag every real step's incremental flush (see genesis_oracle.py's
    # step()/set_recording_context) with this episode's identity, set BEFORE
    # the episode runs — reward/success aren't known yet at this point, so
    # they aren't included here; they're saved separately below
    # (metrics.json) and joinable by source + episode_idx later.
    # save_mpc_transitions (default True) only controls this context
    # tagging — recording itself (dataset.record_transitions) is unaffected.
    if cfg['mpc'].get('save_mpc_transitions', True):
        env.set_recording_context({
            'source':      'simple_mpc.oracle_mpc',
            'episode_idx': ep_idx,
            'seed':        seed,
            'optimizer':   cfg['mpc'].get('optimizer'),
        })

    subgoal  = build_goal(cfg, env)
    obs_init = env.render()
    subgoal  = scale_subgoal_to_material_pixels(
        subgoal, obs_init[..., -1], cfg['dataset']['global_scale'])

    os.makedirs(ep_dir, exist_ok=True)

    video_recorder = None
    video_path = None
    if save_video:
        import cv2
        fps        = float(cfg['mpc'].get('video_fps', 2))
        video_path = os.path.join(ep_dir, 'rollout.avi')
        fourcc     = cv2.VideoWriter_fourcc(*'XVID')
        writer     = cv2.VideoWriter(video_path, fourcc, fps,
                                      (env.screenWidth, env.screenHeight))
        video_recorder = [writer]

    t0     = time.time()
    result = run_oracle_mpc(env, subgoal, cfg, video_recorder=video_recorder,
                             collect_raw_obs=True, collect_states=True,
                             collect_states_pred=True)
    elapsed = time.time() - t0

    if video_recorder is not None:
        video_recorder[0].release()

    # -- quantitative trace --------------------------------------------------
    np.save(os.path.join(ep_dir, 'rewards.npy'),     result['rewards'])
    np.save(os.path.join(ep_dir, 'occ_rewards.npy'), result['occ_rewards'])
    np.save(os.path.join(ep_dir, 'actions.npy'),     result['actions'])
    np.savez(
        os.path.join(ep_dir, 'episode_data.npz'),
        rewards=result['rewards'],
        actions=result['actions'],
        rew_means=result['rew_means'],
        rew_stds=result['rew_stds'],
        best_rewards_per_step=np.array(result['best_rewards_per_step'], dtype=np.float32),
    )
    _plot_episode_reward(result['rewards'], os.path.join(ep_dir, 'rewards.png'))

    # -- data sufficient to visually recreate this trajectory offline -------
    if result['raw_obs'] is not None:
        # (n_mpc+1, H, W, 5) float32 — RGB + material mask + depth per real
        # step; the most direct route to recreating frames without re-running
        # any simulation.
        np.save(os.path.join(ep_dir, 'raw_obs.npy'), result['raw_obs'])
    if result['states'] is not None:
        # list[n_mpc+1] of (N_i, 3) foreground point clouds (depth-derived),
        # for 3-D / top-down reconstruction independent of the RGB frames.
        _save_object_array(result['states'], os.path.join(ep_dir, 'states.npy'))
    if result['states_pred'] is not None:
        # list[n_mpc] of (n_look_ahead, Nx, Ny) predicted occupancy from each
        # chosen action's own rollout — compare against raw_obs/states to see
        # the oracle's predicted-vs-actual gap (should be small; see
        # genesis_oracle.py docstring on why it's a built-in sanity check).
        _save_object_array(result['states_pred'], os.path.join(ep_dir, 'states_pred.npy'))

    metrics = {
        'episode':       ep_idx,
        'seed':          seed,
        'elapsed_sec':   elapsed,
        'reward_init':   float(result['rewards'][0]),
        'reward_final':  float(result['rewards'][-1]),
        'reward_gain':   float(result['rewards'][-1] - result['rewards'][0]),
        'total_time':    result['total_time'],
        'rollout_time':  result['rollout_time'],
        'optim_time':    result['optim_time'],
        'iter_num':      result['iter_num'],
        'video_path':    video_path,
    }
    with open(os.path.join(ep_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Real-step (and, tagged separately, candidate-rollout) transitions were
    # already flushed to disk incrementally during the episode — see
    # set_recording_context() above and
    # simple_mpc.genesis_oracle.GenesisOracleEnv.step()/push_and_record
    # (docs/oracle_mpc_design.md). Nothing left to do here; this episode's
    # outcome (below) is joinable with its transitions by source + episode_idx.

    print(f"  [episode {ep_idx}] reward {result['rewards'][0]:.4f} -> "
          f"{result['rewards'][-1]:.4f}  ({elapsed:.1f}s)  saved to {ep_dir}")

    return result['rewards'], metrics


def main():
    args = parse_args()
    cfg  = load_oracle_config(args.config)

    if args.n_envs is not None:
        cfg['mpc']['n_envs'] = args.n_envs
    if args.optimizer is not None:
        cfg['mpc']['optimizer'] = args.optimizer
    if args.n_mpc is not None:
        cfg['mpc']['n_mpc'] = args.n_mpc
    if args.output_dir is not None:
        cfg['mpc']['output_dir'] = args.output_dir

    episodes_cfg = cfg.get('episodes', {})
    n_episodes = (args.n_episodes if args.n_episodes is not None
                  else int(episodes_cfg.get('n_episodes', 1)))
    seed_base = (args.seed if args.seed is not None
                 else int(episodes_cfg.get('random_seed_base', 0)))

    n_envs = int(cfg['mpc']['n_envs'])
    env = GenesisOracleEnv(cfg, n_envs=n_envs)

    run_dir = os.path.join(
        cfg['mpc'].get('output_dir', 'outputs/oracle_mpc'),
        get_current_YYYY_MM_DD_hh_mm_ss_ms(),
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'run_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"oracle_mpc run: n_episodes={n_episodes}  seed_base={seed_base}  "
          f"n_envs={n_envs}  optimizer={cfg['mpc'].get('optimizer')}  -> {run_dir}")

    all_rewards = []
    all_metrics = []
    for ep in range(n_episodes):
        ep_dir = os.path.join(run_dir, f'episode_{ep:03d}')
        rewards, metrics = run_one_episode(
            env, cfg, ep, seed_base + ep, ep_dir, args.save_video)
        all_rewards.append(rewards)
        all_metrics.append(metrics)

    env.destroy()

    rewards_mat = np.stack(all_rewards)   # (n_episodes, n_mpc+1)
    np.save(os.path.join(run_dir, 'rewards_all.npy'), rewards_mat)
    _plot_summary_reward(rewards_mat, os.path.join(run_dir, 'rewards_summary.png'))

    gains = rewards_mat[:, -1] - rewards_mat[:, 0]
    summary = {
        'n_episodes':        n_episodes,
        'reward_gain_mean':  float(gains.mean()),
        'reward_gain_std':   float(gains.std()),
        'reward_final_mean': float(rewards_mat[:, -1].mean()),
        'episodes':          all_metrics,
    }
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {n_episodes} episode(s) saved to {run_dir}")
    print(f"reward gain: {summary['reward_gain_mean']:+.4f} ± {summary['reward_gain_std']:.4f}")


if __name__ == '__main__':
    main()
