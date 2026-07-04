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

Transform logic should be stateless and importable from lightweight modules.

Target pure utilities:
- `particles_to_occupancy(particles, bounds, resolution, sigma=0.0)`
- `draw_plate_soft(center, angle, intensity, ...)`
- `genesis_action_to_cam3d(action, scale)`

These should be reusable in both dataset preparation and MPC adapters.

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

- Some conversion utilities are still embedded as methods in wrapper/dataset classes.
- Inverse conversions are not always available or information-preserving.
- Legacy scripts may assume implicit channel ordering; new utilities should expose explicit keys where possible.

## 5. Guidance for New Utilities

- Keep functions side-effect free.
- Accept and return tensors/dicts with documented shapes.
- Co-locate shape and coordinate assumptions in docstrings.
- Add fixture tests for orientation-sensitive transforms.
