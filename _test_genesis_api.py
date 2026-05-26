"""
One-shot test to answer all outstanding questions about the Genesis API.
Run from the repo root after: conda activate pme
"""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'Genesis'))

import genesis as gs
# SandboxManipulation calls gs.init() itself — don't pre-init here

from Genesis.sandbox_manipulation_clean import SandboxManipulation

GENESIS_CFG = {
    'simulation': {'dt': 4e-3, 'substeps': 5, 'backend': 'gpu', 'precision': '32', 'performance_mode': True},
    'rigid_options': {'iterations': 10, 'ls_iterations': 10, 'tolerance': 1e-4, 'ls_tolerance': 0.05,
                      'box_box_detection': True, 'use_contact_island': True, 'use_hibernation': False},
    'box': {'vol': [0.128, 0.128, 0.04], 'wall_thickness': 0.02},
    'material': {'vol': [0.126, 0.126, 0.05], 'shape': 'sphere', 'particle_size': 0.015,
                 'n_particles': 5, 'density': 1000.0, 'friction': 0.5},
    'plate': {'speed': 0.125, 'size': [0.04, 0.002, 0.01]},
    'data_collection': {'sampled': {}},
}

sim = SandboxManipulation(GENESIS_CFG, n_envs=1)

W, H = 128, 128
cam = sim._scene.add_camera(res=(W, H), pos=(0.0, 0.0, 0.3),
                             lookat=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov=45.0, GUI=False)

sim.build()
sim.shuffle_particles()
sim.update_material_state()

print("\n=== 1. Camera intrinsics ===")
print(f"  cam.f  = {cam.f}")
print(f"  cam.cx = {cam.cx}")
print(f"  cam.cy = {cam.cy}")
print(f"  expected f = {0.5 * H / __import__('math').tan(__import__('math').radians(22.5)):.2f}")

print("\n=== 2. particle.idx exists? ===")
for i, p in enumerate(sim.material):
    has_idx = hasattr(p, 'idx')
    print(f"  particle[{i}]: has .idx={has_idx}, link_start={p.link_start}", end='')
    if has_idx:
        print(f", .idx={p.idx}", end='')
    print()

print("\n=== 3. segmentation_idx_dict vs seg_idxc_map ===")
vis = sim._scene.visualizer
has_dict = hasattr(vis, 'segmentation_idx_dict')
has_map  = hasattr(vis, 'seg_idxc_map')
print(f"  visualizer has segmentation_idx_dict: {has_dict}")
print(f"  visualizer has seg_idxc_map:          {has_map}")

print("\n=== 4. camera.render() — shapes and types (before scene step) ===")
rgb_arr, depth_arr, seg_arr, normal_arr = cam.render(rgb=True, depth=True, segmentation=True)
import numpy as np

def desc(name, x):
    if x is None:
        print(f"  {name}: None")
    else:
        import torch
        t = "torch" if isinstance(x, torch.Tensor) else "numpy"
        arr = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        print(f"  {name}: type={t}, shape={arr.shape}, dtype={arr.dtype}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}")

desc("rgb_arr",    rgb_arr)
desc("depth_arr",  depth_arr)
desc("seg_arr",    seg_arr)
desc("normal_arr", normal_arr)

print("\n=== 5. segmentation_idx_dict contents (first 10 entries) ===")
if has_dict:
    idxc_map = vis.segmentation_idx_dict
    print(f"  type={type(idxc_map).__name__}, len={len(idxc_map)}")
    for k, v in list(idxc_map.items())[:10]:
        print(f"    idxc={k!r:6s} -> seg_key={v!r}")

print("\n=== 6. _operation_height type ===")
oh = sim._operation_height
print(f"  _operation_height = {oh!r}, type={type(oh).__name__}")

print("\n=== 7. VisOptions segmentation_level ===")
try:
    sl = sim._scene._visualizer._context.segmentation_level
    print(f"  segmentation_level = {sl!r}")
except Exception as e:
    print(f"  could not read: {e}")

print("\nDone.")
