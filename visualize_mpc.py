# Using new version

import os
import cv2
import pickle
import numpy as np
from env.genesis_env import GenesisEnv
import multiprocessing as mp
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
from scipy.special import softmax

# utils
from utils import load_yaml, save_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, set_seed, pcd2pix, gen_goal_shape, gen_subgoal, gt_rewards, gt_rewards_norm_by_sum, lighten_img, rmbg
from model.gnn_dyn import PropNetDiffDenModel
from model.eulerian_wrapper import EulerianModelWrapper, SplatPushModel, FluidPushModel

# ---------------------------------------------------------------------------
# Simple MPC switch — set to True to use simple_mpc.run_simple_mpc() instead
# of the full env.step_subgoal_ptcl() pipeline.  When True the config is
# loaded from SIMPLE_MPC_CONFIG and replaces config/mpc/config.yaml.
# ---------------------------------------------------------------------------
USE_SIMPLE_MPC    = True
SIMPLE_MPC_CONFIG = 'simple_mpc/config/config_simple.yaml'

# ---------------------------------------------------------------------------
# Eulerian model configuration — edit these values to switch models.
# Set USE_EULERIAN_MODEL = False to use the original GNN model instead.
# ---------------------------------------------------------------------------
USE_EULERIAN_MODEL  = True       # True → use Eulerian wrapper; False → use GNN
EULERIAN_MODEL_TYPE = 'splat'     # 'splat' or 'fluid'

GRID_RES = (64, 64)               # (Nx, Ny) voxels for the occupancy grid

# SplatPushModel parameters (active when EULERIAN_MODEL_TYPE == 'splat')
SPLAT_WIDTH              = 8.0
SPLAT_SIGMA              = 1.5
SPLAT_REDISTRIBUTE       = False
SPLAT_REDISTRIBUTE_ITERS = 10

# FluidPushModel parameters (active when EULERIAN_MODEL_TYPE == 'fluid')
FLUID_WIDTH              = 5.0
FLUID_SIGMA              = 1.5
FLUID_N_STEPS            = 20
FLUID_DECAY              = 0.95
FLUID_MEDIA_SHARPNESS    = 5.0
FLUID_BLUR_SIGMA         = 1.0
FLUID_CORRECT_DIVERGENCE = False
FLUID_REDISTRIBUTE       = False
FLUID_REDISTRIBUTE_ITERS = 10
# ---------------------------------------------------------------------------

def save_observations_as_images(raw_obs, output_dir, prefix='step'):
    """
    Save each observation as a PNG image.
    
    Args:
        raw_obs: Array of shape (n_steps, height, width, 5) containing RGBA and depth
        output_dir: Directory to save images
        prefix: Prefix for image filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, obs in enumerate(raw_obs):
        # Extract RGB and convert to BGR for cv2
        # np.ascontiguousarray ensures proper memory layout for cv2
        img_bgr = np.ascontiguousarray(
            np.clip(obs[:, :, :3][:, :, ::-1], 0, 255).astype(np.uint8)
        )
        
        filename = os.path.join(output_dir, f'{prefix}_{i:04d}.png')
        cv2.imwrite(filename, img_bgr)
    
    print(f"Saved {len(raw_obs)} images to {output_dir}")

def save_observations_as_video(raw_obs, output_path, fps=5, speedup=1.0):
    """
    Save observations as a video file.
    
    Args:
        raw_obs: Array of shape (n_steps, height, width, 5) containing RGBA and depth
        output_path: Path for output video file
        fps: Frames per second for the video
        speedup: Speedup factor (e.g., 2.0 = 2x speed)
    """
    if len(raw_obs) < 2:
        print("Not enough frames to create video")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Get dimensions from first frame
    height, width = raw_obs[0].shape[:2]
    
    # Apply speedup to fps (ensure at least 1)
    effective_fps = max(1, int(fps * speedup))
    
    # Use MJPG/AVI for broad compatibility
    avi_path = output_path.rsplit('.', 1)[0] + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(avi_path, fourcc, effective_fps, (width, height))
    
    if not out.isOpened():
        print(f"Failed to open video writer for {avi_path}")
        return
    
    frames_written = 0
    for obs in raw_obs:
        # Extract RGB, convert to BGR, ensure contiguous uint8
        img_bgr = np.ascontiguousarray(
            np.clip(obs[:, :, :3][:, :, ::-1], 0, 255).astype(np.uint8)
        )
        out.write(img_bgr)
        frames_written += 1
    
    out.release()
    print(f"Saved video to {avi_path} with {frames_written} frames at {effective_fps} fps")

def plot_reward_evolution(rewards, output_path, title='Reward Evolution Over MPC Steps'):
    """
    Plot and save reward evolution graph.
    
    Args:
        rewards: Array of rewards for each step
        output_path: Path for output plot file
        title: Title for the plot
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker='o', linewidth=2, markersize=6)
    plt.xlabel('MPC Step', fontsize=12)
    plt.ylabel('Reward (Distance to Goal)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reward plot to {output_path}")
    print(f"Reward stats - Initial: {rewards[0]:.4f}, Final: {rewards[-1]:.4f}, "
          f"Mean: {rewards.mean():.4f}, Improvement: {rewards[-1] - rewards[0]:.4f}")

def main():
    config = load_yaml(SIMPLE_MPC_CONFIG)

    model_folder = config['mpc']['model_folder']
    model_iter = config['mpc']['iter_num']
    n_mpc = config['mpc']['n_mpc']
    n_look_ahead = config['mpc']['n_look_ahead']
    n_sample = config['mpc']['n_sample']
    n_update_iter = config['mpc']['n_update_iter']
    gd_loop = config['mpc']['gd_loop']
    mpc_type = config['mpc']['mpc_type']

    task_type = config['mpc']['task']['type']
    
    # Get visualization options (with defaults if not present)
    vis_config = config['mpc'].get('visualization', {})
    save_images = vis_config.get('save_images', False)
    save_video = vis_config.get('save_video', False)
    video_fps = vis_config.get('video_fps', 5)
    video_speedup = vis_config.get('video_speedup', 1.0)
    plot_rewards = vis_config.get('plot_rewards', False)
    output_base_dir = vis_config.get('output_dir', 'outputs')

    if not USE_EULERIAN_MODEL:
        model_root = 'data/gnn_dyn_model/'
        model_folder = os.path.join(model_root, model_folder)
        GNN_single_model = PropNetDiffDenModel(config, True)
        if model_iter == -1:
            GNN_single_model.load_state_dict(torch.load(f'{model_folder}/net_best.pth'), strict=False)
        else:
            GNN_single_model.load_state_dict(torch.load(f'{model_folder}/net_epoch_0_iter_{model_iter}.pth'), strict=False)
        GNN_single_model = GNN_single_model.cuda()
    else:
        GNN_single_model = None   # built after env.reset() once cam params are available

    env = GenesisEnv(config)
    screenWidth  = env.screenWidth
    screenHeight = env.screenHeight

    if task_type == 'target_control':
        goal_row = config['mpc']['task']['goal_row']
        goal_col = config['mpc']['task']['goal_col']
        goal_r = config['mpc']['task']['goal_r']
        subgoal, mask = gen_subgoal(goal_row,
                                    goal_col,
                                    goal_r,
                                    h=screenHeight,
                                    w=screenWidth)
        goal_img = (mask[..., None]*255).repeat(3, axis=-1).astype(np.uint8)
    elif task_type == 'target_shape':
        goal_char = config['mpc']['task']['target_char']
        subgoal, goal_img = gen_goal_shape(goal_char,
                                            h=screenHeight,
                                            w=screenWidth)
    else:
        raise NotImplementedError
    
    env.reset()

    if USE_EULERIAN_MODEL:
        cam_extrinsic = env.get_cam_extrinsics()
        global_scale  = config['dataset']['global_scale']
        bounds        = EulerianModelWrapper.default_bounds(config, convention='genesis')
        if EULERIAN_MODEL_TYPE == 'splat':
            push_model = SplatPushModel(
                width=SPLAT_WIDTH, sigma=SPLAT_SIGMA,
                redistribute=SPLAT_REDISTRIBUTE,
                redistribute_iters=SPLAT_REDISTRIBUTE_ITERS,
            )
        elif EULERIAN_MODEL_TYPE == 'fluid':
            push_model = FluidPushModel(
                width=FLUID_WIDTH, sigma=FLUID_SIGMA,
                n_steps=FLUID_N_STEPS, decay=FLUID_DECAY,
                media_sharpness=FLUID_MEDIA_SHARPNESS,
                blur_sigma=FLUID_BLUR_SIGMA,
                correct_divergence=FLUID_CORRECT_DIVERGENCE,
                redistribute=FLUID_REDISTRIBUTE,
                redistribute_iters=FLUID_REDISTRIBUTE_ITERS,
            )
        else:
            raise ValueError(f"Unknown EULERIAN_MODEL_TYPE: {EULERIAN_MODEL_TYPE!r}")
        GNN_single_model = EulerianModelWrapper(
            push_model, bounds, GRID_RES, cam_extrinsic, global_scale,
            action_convention='genesis',
        ).cuda()
        print(f"Using EulerianModelWrapper ({EULERIAN_MODEL_TYPE}) with grid {GRID_RES}")

    if not USE_SIMPLE_MPC:
        funnel_dist = np.zeros_like(subgoal)
        action_seq_mpc_init = np.load('init_action/init_action_'+ str(n_sample) +'.npy')[np.newaxis, ...]
        action_label_seq_mpc_init = np.zeros(1)

    # Create output directory with timestamp
    if save_images or save_video or plot_rewards:
        timestamp = get_current_YYYY_MM_DD_hh_mm_ss_ms()
        output_dir = os.path.join(output_base_dir, f'mpc_run_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving outputs to: {output_dir}")
        
        # Save goal image for reference
        goal_path = os.path.join(output_dir, 'goal.png')
        cv2.imwrite(goal_path, goal_img)
        print(f"Saved goal image to {goal_path}")


    # Set up video recorder if requested — records every intermediate frame
    video_recorder = None
    if save_video:
        video_path = os.path.join(output_dir, 'mpc_trajectory.avi')
        effective_fps = max(1, int(video_fps * video_speedup))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vw = cv2.VideoWriter(video_path, fourcc, effective_fps,
                             (screenWidth, screenHeight))
        if vw.isOpened():
            video_recorder = [vw]
            print(f"Recording full manipulation video at {effective_fps} fps")
        else:
            print("Warning: Failed to open video writer")

    if USE_SIMPLE_MPC:
        from simple_mpc import run_simple_mpc
        subg_output = run_simple_mpc(env, GNN_single_model, subgoal, config,
                                     video_recorder=video_recorder)
    else:
        subg_output = env.step_subgoal_ptcl(subgoal,
                                            GNN_single_model,
                                            None,
                                            n_mpc=n_mpc,
                                            n_look_ahead=n_look_ahead,
                                            n_sample=n_sample,
                                            n_update_iter=n_update_iter,
                                            mpc_type=mpc_type,
                                            gd_loop=gd_loop,
                                            particle_num=-1,
                                            funnel_dist=funnel_dist,
                                            action_seq_mpc_init=action_seq_mpc_init,
                                            action_label_seq_mpc_init=action_label_seq_mpc_init,
                                            time_lim=config['mpc']['time_lim'],
                                            auto_particle_r=True,
                                            video_recorder=video_recorder,)

    # Finalize video
    if video_recorder is not None:
        video_recorder[0].release()
        video_size = os.path.getsize(video_path)
        print(f"Saved full manipulation video to {video_path} ({video_size:,} bytes)")
    
    # Extract results from MPC output
    rewards = subg_output['rewards']
    raw_obs = subg_output['raw_obs']
    actions = subg_output['actions']
    
    print(f"\nMPC completed: {n_mpc} steps")
    print(f"Initial reward: {rewards[0]:.4f}")
    print(f"Final reward: {rewards[-1]:.4f}")
    print(f"Reward improvement: {rewards[-1] - rewards[0]:.4f}")
    
    # Save images of each MPC step (one per action)
    if save_images:
        images_dir = os.path.join(output_dir, 'images')
        save_observations_as_images(raw_obs, images_dir)
    
    # Plot and save reward evolution
    if plot_rewards:
        reward_plot_path = os.path.join(output_dir, 'reward_evolution.png')
        plot_reward_evolution(rewards, reward_plot_path)
        
        # Also save rewards as numpy array for later analysis
        rewards_path = os.path.join(output_dir, 'rewards.npy')
        np.save(rewards_path, rewards)
        
        # Save actions for reference
        actions_path = os.path.join(output_dir, 'actions.npy')
        np.save(actions_path, actions)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
