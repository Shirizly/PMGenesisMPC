#!/usr/bin/env python3
"""
Diagnostic: compare direct model path (compare_model_emd) vs wrapper path (debug_mpc_gui).

Run with:
    conda activate pme
    python diag_wrapper.py

Produces a side-by-side PNG: diag_wrapper_output.png
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = 'runs_cubes/nfu_mse_mass2/unet_best.pth'
DATA_FOLDER  = 'corl/cube'           # relative to Genesis/data/
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── imports ────────────────────────────────────────────────────────────────────
from Genesis.training.dataset import PileSweepData
from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLM
from model.eulerian_wrapper import (
    EulerianModelWrapper, UNetFiLMPushModel,
    _action_to_cam_3d_genesis,
)
from utils import load_yaml


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load dataset sample (ground truth for the "correct" path)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Loading dataset sample…")
ds = PileSweepData([DATA_FOLDER], split="train")
print(f"  dataset size: {len(ds)} samples")

(input_grid, physics), output_grid = ds[0]

occ_ds   = input_grid[0]          # (H, W) world_y × world_x, values {0,1}
act_ds   = input_grid[1]          # (H, W) action channel
out_ds   = output_grid             # (H, W) label after push

print(f"\n[Dataset sample 0]")
print(f"  occ_ds   shape={tuple(occ_ds.shape)}  "
      f"min={occ_ds.min():.3f}  max={occ_ds.max():.3f}  "
      f"mean={occ_ds.mean():.4f}  nonzero={occ_ds.gt(0.5).sum().item()} px")
print(f"  act_ds   shape={tuple(act_ds.shape)}  "
      f"min={act_ds.min():.3f}  max={act_ds.max():.3f}  "
      f"mean={act_ds.mean():.4f}  nonzero={act_ds.gt(0.1).sum().item()} px")
print(f"  physics  = {physics.tolist()}")
print(f"  out_ds   nonzero={out_ds.gt(0.5).sum().item()} px")

# physics is already normalised [0,1] by PileSweepData._det_physics
physics_norm = physics

H, W = occ_ds.shape  # should be 128, 128


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DIRECT path (compare_model_emd style)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DIRECT path (compare_model_emd)…")

unet_direct = NFDUNetFiLM().to(DEVICE)
sd = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
unet_direct.load_state_dict(sd)
unet_direct.eval()

x_direct    = input_grid.unsqueeze(0).to(DEVICE)                           # (1, 2, H, W)
phys_direct = physics_norm.unsqueeze(0).to(DEVICE)   # normalised [0,1]     # (1, 3)

with torch.no_grad():
    logits_direct = unet_direct(x_direct, phys_direct).squeeze(1)   # (1, H, W)
    pred_direct   = torch.sigmoid(logits_direct)[0].cpu().numpy()    # (H, W)

print(f"  logits  min={logits_direct.min():.3f}  max={logits_direct.max():.3f}  "
      f"mean={logits_direct.mean():.4f}")
print(f"  pred    min={pred_direct.min():.4f}  max={pred_direct.max():.4f}  "
      f"mean={pred_direct.mean():.4f}  >0.5: {(pred_direct>0.5).sum()} px")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  WRAPPER path (debug_mpc_gui style)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("WRAPPER path (UNetFiLMPushModel)…")

cfg = load_yaml('simple_mpc/config/config_simple.yaml')

# Physics from MPC config (what debug_mpc_gui actually uses), normalised to [0,1]
_f  = cfg['dataset'].get('particle_friction', 0.05)
_d  = cfg['dataset'].get('particle_density',  750.0)
_bf = cfg['dataset'].get('box_friction',       0.05)
physics_mpc = torch.tensor([
    (_f  - 0.05) / (0.50 - 0.05),
    (_d  -  750) / (5000 -  750),
    (_bf - 0.05) / (0.50 - 0.05),
], dtype=torch.float32)
print(f"  MPC physics (raw)        = [{_f}, {_d}, {_bf}]")
print(f"  MPC physics (normalised) = {physics_mpc.tolist()}")
print(f"  DS  physics  = {physics.tolist()}  (from dataset sample)")
print(f"  DS  physics (normalised) = {physics_norm.tolist()}")

unet_wrap = NFDUNetFiLM(in_channels=2, out_channels=1, cond_dim=3).to(DEVICE)
unet_wrap.load_state_dict(sd)
unet_wrap.eval()

global_scale = float(cfg['dataset']['global_scale'])   # 0.6
wkspc_w      = float(cfg['dataset']['wkspc_w'])        # 0.064
box_full_m   = wkspc_w * 2                             # 0.128
grid_n       = round(box_full_m * 1000)                # 128
grid_res     = (grid_n, grid_n)

plate_L_m  = cfg['dataset'].get('plate_length', 0.04)
plate_W_m  = cfg['dataset'].get('plate_width',  0.002)
px_per_m   = grid_n / box_full_m                       # 1000 px/m
plate_L_px = plate_L_m * px_per_m
plate_W_px = plate_W_m * px_per_m

push_model = UNetFiLMPushModel(
    unet_film=unet_wrap,
    physics=physics_mpc,
    grid_size=grid_res,
    plate_length_px=plate_L_px,
    plate_width_px=plate_W_px,
).to(DEVICE)

bounds = UNetFiLMPushModel.default_bounds(cfg)
cam_extrinsic = np.eye(4)   # unused for 'genesis' convention

wrapper = EulerianModelWrapper(
    push_model, bounds, grid_res, cam_extrinsic, global_scale,
    action_convention='genesis',
).to(DEVICE)

# --- Convert dataset occupancy to EulerianWrapper convention ---
# Dataset: (H=world_y, W=world_x)
# EulerianWrapper: (Nx=world_x, Ny=-world_y)
# Inverse of flip(dim-1) + transpose(-2,-1) is transpose(-2,-1) + flip(dim-1)
occ_ew = occ_ds.unsqueeze(0).to(DEVICE)                     # (1, H=128, W=128)
occ_ew = occ_ew.transpose(-2, -1).flip(dims=[-1])           # (1, Nx=world_x, Ny=-world_y)
print(f"\n[occ in EulerianWrapper convention]")
print(f"  shape={tuple(occ_ew.shape)}  "
      f"min={occ_ew.min():.3f}  max={occ_ew.max():.3f}  "
      f"nonzero={occ_ew.gt(0.5).sum().item()} px  "
      f"(should match dataset: {occ_ds.gt(0.5).sum().item()} px)")

# --- Recover the action from the dataset sample ---
# PileSweepDataWithActions would give plate_pos (world_x_px, world_y_px) directly.
# Here we extract from the run's raw data.
run_idx = ds._run_lookup[0]
run     = ds.runs[run_idx]
_, _, plate_pos, plate_pos_, angle = ds._extract_sample_in_pxl(run, 0)
# plate_pos[:2] = (world_x_px, world_y_px) = (col, row) in 128px grid
# Convert px → world metres:  world_xy = (px - 64) / 1000
sx_m = (plate_pos[0].item()  - 64) / 1000.0
sy_m = (plate_pos[1].item()  - 64) / 1000.0
ex_m = (plate_pos_[0].item() - 64) / 1000.0
ey_m = (plate_pos_[1].item() - 64) / 1000.0
action_np = np.array([[sx_m, sy_m, ex_m, ey_m]], dtype=np.float32)
print(f"\n[Action derived from dataset sample]")
print(f"  start_px=({plate_pos[0].item():.1f}, {plate_pos[1].item():.1f})  "
      f"end_px=({plate_pos_[0].item():.1f}, {plate_pos_[1].item():.1f})  "
      f"angle={angle:.3f} rad")
print(f"  action_world=[{sx_m:.4f}, {sy_m:.4f}, {ex_m:.4f}, {ey_m:.4f}] m")

act_t = torch.tensor(action_np, device=DEVICE)   # (1, 4)

with torch.no_grad():
    occ_pred_ew = wrapper.predict_one_step_occ(occ_ew.clone(), act_t)  # (1, Nx, Ny)

print(f"\n[Wrapper prediction (EulerianWrapper convention)]")
print(f"  occ_pred_ew  min={occ_pred_ew.min():.4f}  max={occ_pred_ew.max():.4f}  "
      f"mean={occ_pred_ew.mean():.4f}  >0.5: {occ_pred_ew.gt(0.5).sum().item()} px")

# Convert back to dataset convention for comparison
pred_ew_ds = occ_pred_ew[0].flip(dims=[-1]).transpose(-2, -1).cpu().numpy()
print(f"  pred (dataset conv) min={pred_ew_ds.min():.4f}  max={pred_ew_ds.max():.4f}  "
      f"mean={pred_ew_ds.mean():.4f}  >0.5: {(pred_ew_ds>0.5).sum()} px")


# ══════════════════════════════════════════════════════════════════════════════
# 3b. WRAPPER path with DATASET physics (to isolate physics mismatch)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("WRAPPER path with DATASET physics (isolating physics mismatch)…")

push_model_ds_phys = UNetFiLMPushModel(
    unet_film=unet_wrap,
    physics=physics_norm,      # <-- from the dataset sample (density normalised /1000)
    grid_size=grid_res,
    plate_length_px=plate_L_px,
    plate_width_px=plate_W_px,
).to(DEVICE)

wrapper_ds_phys = EulerianModelWrapper(
    push_model_ds_phys, bounds, grid_res, cam_extrinsic, global_scale,
    action_convention='genesis',
).to(DEVICE)

with torch.no_grad():
    occ_pred_ds_phys = wrapper_ds_phys.predict_one_step_occ(occ_ew.clone(), act_t)

pred_ds_phys_np = occ_pred_ds_phys[0].flip(dims=[-1]).transpose(-2,-1).cpu().numpy()
print(f"  pred (dataset phys) min={pred_ds_phys_np.min():.4f}  max={pred_ds_phys_np.max():.4f}  "
      f"mean={pred_ds_phys_np.mean():.4f}  >0.5: {(pred_ds_phys_np>0.5).sum()} px")


# ══════════════════════════════════════════════════════════════════════════════
# 3c. Step inside UNetFiLMPushModel.forward — print internal stats
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Internal stats inside UNetFiLMPushModel.forward (MPC physics)…")

# Replicate forward manually to inspect internals
Ny = push_model.Ny
s_3d_cam, e_3d_cam = _action_to_cam_3d_genesis(act_t, global_scale)
start_grid = wrapper._cam3d_to_grid(s_3d_cam)
end_grid   = wrapper._cam3d_to_grid(e_3d_cam)

print(f"  start_grid={start_grid[0].tolist()}  end_grid={end_grid[0].tolist()}")

occ_in = occ_ew.clone()   # (1, Nx, Ny)

# Step 1: flip + transpose to dataset convention
occ_ds_hat = occ_in.flip(dims=[-1]).transpose(-2, -1)    # (1, H, W)
print(f"  occ_ds_hat  min={occ_ds_hat.min():.3f}  max={occ_ds_hat.max():.3f}  "
      f"mean={occ_ds_hat.mean():.4f}  nonzero={occ_ds_hat.gt(0.5).sum().item()} px")

# Step 2: y-indices
iy_s_cam = start_grid[:, 1]
iy_e_cam = end_grid[:, 1]
iy_s_ds  = (Ny - 1) - iy_s_cam
iy_e_ds  = (Ny - 1) - iy_e_cam
print(f"  iy_s_cam={iy_s_cam[0].item():.1f}  iy_e_cam={iy_e_cam[0].item():.1f}")
print(f"  iy_s_ds ={iy_s_ds[0].item():.1f}  iy_e_ds ={iy_e_ds[0].item():.1f}")

# Step 3: angle
dx    = end_grid[:, 0] - start_grid[:, 0]
dy_ds = iy_e_ds - iy_s_ds
dxy   = torch.hypot(dx, dy_ds)
angle_draw = torch.where(dxy > 1e-4, torch.atan2(dy_ds, dx), torch.zeros_like(dxy))
print(f"  dx={dx[0].item():.2f}  dy_ds={dy_ds[0].item():.2f}  "
      f"angle_draw={angle_draw[0].item():.3f} rad "
      f"({np.degrees(angle_draw[0].item()):.1f} deg)")

# Step 4: draw action channel using push_model's method
start_center = torch.stack([iy_s_ds, start_grid[:, 0]], dim=1)
end_center   = torch.stack([iy_e_ds, end_grid[:, 0]],   dim=1)
act_start = push_model._draw_plate_soft(start_center, angle_draw.detach(), 0.5)
act_end   = push_model._draw_plate_soft(end_center,   angle_draw.detach(), 1.0)
act_ch    = torch.maximum(act_start, act_end)

print(f"  act_ch  min={act_ch.min():.4f}  max={act_ch.max():.4f}  "
      f"mean={act_ch.mean():.4f}  nonzero={act_ch.gt(0.1).sum().item()} px")
print(f"  training act_ds nonzero: {act_ds.gt(0.1).sum().item()} px  "
      f"max={act_ds.max():.3f}")

# Step 5-6: build input and run model
x_wrap = torch.stack([occ_ds_hat, act_ch], dim=1)    # (1, 2, H, W)
phys_w = push_model._physics.expand(1, -1)            # (1, 3)
print(f"  x_wrap  channel0 mean={x_wrap[:,0].mean():.4f}  "
      f"channel1 mean={x_wrap[:,1].mean():.4f}")
print(f"  phys_w  = {phys_w[0].tolist()}")

with torch.no_grad():
    logits_wrap = unet_wrap(x_wrap, phys_w).squeeze(1)   # (1, H, W)
    pred_wrap   = torch.sigmoid(logits_wrap)[0].cpu().numpy()

print(f"  logits_wrap  min={logits_wrap.min():.3f}  max={logits_wrap.max():.3f}  "
      f"mean={logits_wrap.mean():.4f}")
print(f"  pred_wrap    min={pred_wrap.min():.4f}  max={pred_wrap.max():.4f}  "
      f"mean={pred_wrap.mean():.4f}  >0.5: {(pred_wrap>0.5).sum()} px")

# Also run with dataset physics for comparison
phys_ds_t = physics.unsqueeze(0).to(DEVICE)
with torch.no_grad():
    logits_dsph = unet_wrap(x_wrap, phys_ds_t).squeeze(1)
    pred_dsph   = torch.sigmoid(logits_dsph)[0].cpu().numpy()
print(f"\n  [same x_wrap, dataset physics={physics.tolist()}]")
print(f"  logits_dsph  min={logits_dsph.min():.3f}  max={logits_dsph.max():.3f}  "
      f"mean={logits_dsph.mean():.4f}")
print(f"  pred_dsph    min={pred_dsph.min():.4f}  max={pred_dsph.max():.4f}  "
      f"mean={pred_dsph.mean():.4f}  >0.5: {(pred_dsph>0.5).sum()} px")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Summary figure
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Saving figure…")

occ_ds_np  = occ_ds.numpy()
act_ds_np  = act_ds.numpy()
out_ds_np  = out_ds.numpy()
act_ch_np  = act_ch[0].cpu().numpy()           # (H, W) act channel from wrapper
occ_ew_ds_np = occ_ew[0].flip(dims=[-1]).transpose(-2,-1).cpu().numpy()  # back to ds convention

fig = plt.figure(figsize=(20, 10))
gs  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.3)

def show(ax, data, title, vmin=0, vmax=1, cmap='viridis'):
    im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

# Row 0: training-convention inputs and direct prediction
show(fig.add_subplot(gs[0,0]), occ_ds_np,  'occ_ds (training)\n[input ch0]')
show(fig.add_subplot(gs[0,1]), act_ds_np,  'act_ds (training)\n[input ch1]')
show(fig.add_subplot(gs[0,2]), out_ds_np,  'output (label)\n[after push]')
show(fig.add_subplot(gs[0,3]), pred_direct,'direct pred\n[compare_model_emd]')
show(fig.add_subplot(gs[0,4]), act_ch_np,  'act_ch (wrapper)\n[MPC action channel]')
show(fig.add_subplot(gs[0,5]), occ_ew_ds_np,'occ from EW→DS\n[should=occ_ds]')

# Row 1: wrapper predictions
show(fig.add_subplot(gs[1,0]), pred_ew_ds,     'wrapper pred\n[MPC physics]')
show(fig.add_subplot(gs[1,1]), pred_ds_phys_np,'wrapper pred\n[dataset physics]')
show(fig.add_subplot(gs[1,2]), pred_dsph,      'same x_wrap,\ndataset physics')
show(fig.add_subplot(gs[1,3]), pred_wrap,      'same x_wrap,\nMPC physics')

# Difference maps
diff_mpc = pred_ew_ds - pred_direct
diff_dsp = pred_ds_phys_np - pred_direct
ax_d1 = fig.add_subplot(gs[1,4])
ax_d2 = fig.add_subplot(gs[1,5])
show(ax_d1, np.clip(diff_mpc,-1,1), 'wrapper(MPC ph) − direct', vmin=-1, vmax=1, cmap='RdBu')
show(ax_d2, np.clip(diff_dsp,-1,1), 'wrapper(DS ph) − direct',  vmin=-1, vmax=1, cmap='RdBu')

fig.suptitle(
    f'Dataset physics={physics.tolist()}\n'
    f'MPC physics={physics_mpc.tolist()}',
    fontsize=10)

out_path = 'diag_wrapper_output.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved to {out_path}")
print("\nDone.")
