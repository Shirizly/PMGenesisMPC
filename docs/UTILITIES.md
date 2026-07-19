# Utilities

This document lists reusable utility primitives and their intended ownership boundaries.

## 1. Physics Normalization

Single source of truth:
- `physics/normalization.py`
- `PhysicsBounds`

Use `PhysicsBounds` everywhere instead of inline formulas.

```python
from physics.normalization import PhysicsBounds

bounds = PhysicsBounds.default()
norm = bounds.normalize(raw_tensor)
raw = bounds.denormalize(norm_tensor)
```

Policy:
- Training, evaluation, MPC, and system ID must all use the same bounds definition.
- Bounds are config/model-card data, not hard-coded callsite logic.

## 2. Representation Transforms

Transform logic is stateless and lives in `transforms/functional.py`, which
imports nothing from the heavy model/wrapper modules (the dependency runs the
other way: `model/eulerian_wrapper.py` imports from here).

Pure utilities in `transforms/functional.py`:
- `particles_to_occupancy(particles, bounds, resolution, sigma=0.0)`
- `draw_plate_soft(center, angle, grid_size, plate_length_px, plate_width_px, intensity, sigma=1.5)`
- `genesis_action_to_cam3d(action, scale)`
- `build_action_delta(s_cur_xyz, p_start_xyz, p_stop_xyz, sigma_m)`
- `get_grid_axes(ndim)` / `grid_axis_indices(axes)` (coordinate bookkeeping)

These are reusable in both dataset preparation and MPC adapters.

## 3. Transform Pipeline Pattern

Use composable callable transforms for representation conversion.

Example structure:

```python
batch = {"particles": particles, ...}
for t in transforms:
    batch = t(batch)
```

Benefits:
- train/test/MPC preprocessing parity
- explicit, debuggable stage boundaries
- lower coupling to heavy wrapper classes

## 4. Current Limitations to Document Explicitly

- Some conversion utilities are still embedded as methods in wrapper/dataset classes
  (e.g. `_occupancy_to_particles` and `_action_to_cam_3d` in `model/eulerian_wrapper.py`).
- Inverse conversions are not always available or information-preserving.
- Legacy scripts may assume implicit channel ordering; new utilities should expose explicit keys where possible.
- Root-level `utils.py` is an unsplit grab-bag (~50 helpers: YAML I/O, action
  geometry, point-cloud ops, goal-shape generation, reward helpers, image ops)
  that both `model/eulerian_wrapper.py` and `simple_mpc/` depend on. New code
  should not add to it; candidates should go to `transforms/` or a scoped
  module, and a future pass should split it along those lines.

## 5. Guidance for New Utilities

- Keep functions side-effect free.
- Accept and return tensors/dicts with documented shapes.
- Co-locate shape and coordinate assumptions in docstrings.
- Add fixture tests for orientation-sensitive transforms.
