# Codebase Overview (Archived)

Archived from repository root on 2026-07-03. This deep-dive remains useful historical context but is no longer the primary source of truth.

Primary living docs:
- ARCHITECTURE.md
- INTERFACES.md
- UTILITIES.md

Original content was moved here verbatim to preserve historical details while reducing active-document redundancy.

---

# Codebase Overview — Pile Manipulation Training Pipeline

Focuses on data formats at each interface. All grids are **float32, values in [0, 1]**.

---

## 1. Raw Simulation Data on Disk

Produced by `Genesis/sandbox_manipulation_clean.py → _save_data()`.

**File pairs per run** (under `Genesis/data/{dataset}/{shape}/n{N}/size{S}/`):
```
_{id}_data.pt      ← torch.save'd dict of tensors (successful samples)
_{id}_config.yaml  ← YAML with physics/geometry metadata
```

**`_data.pt` dict keys:**

| Key | Shape | dtype | Description |
|---|---|---|---|
| `states` | `(N_samples, N_particles, 7)` | float32 | Particle states **before** action. `[:3]` = x,y,z in metres (centred at 0), `[3:]` = quaternion w,x,y,z |
| `states_` | `(N_samples, N_particles, 7)` | float32 | Particle states **after** action (same layout) |
| `p_starts` | `(N_samples, 3)` | float32 | Plate tool start position in metres |
| `p_stops` | `(N_samples, 3)` | float32 | Plate tool end position in metres |
| `angles` | `(N_samples,)` | float32 | Plate yaw angle in radians |

**Config YAML structure (relevant fields for training):**
```yaml
box:
  vol: [0.128, 0.128, 0.04]   # workspace size in metres → determines grid pixel size
  friction: <float>
material:
  shape: cube | sphere
  n_particles: <int>
  particle_size: <float>       # base size in metres
  friction: <float>
  density: <float>
plate:
  size: [0.04, 0.002, 0.01]   # tool dimensions in metres
data_collection:
  sampled:
    particle_sizes: list of (x,y,z) tuples, one per particle  # actual per-particle sizes
```

---

## 2. `PileSweepData` Dataset (`Genesis/training/dataset.py`)

Converts raw `.pt` + `.yaml` pairs into PyTorch training samples.

**Construction:**
```python
PileSweepData(paths: list[str], split: "train"|"val"|"test"|None)
```
- `paths` are relative to `Genesis/data/`
- Split is **deterministic by hashing physics params** (friction, density, box_friction, shape, n_particles, particle_size), so all runs of the same condition land in the same split.

**Grid dimensions** are derived from config at load time:
```
H = W = int(box.vol[0] * 1000)   # e.g. 0.128 m → 128 px
```
`TO_PXL = 1000` px/m. Origin is the box centre projected to pixel space.

**`__getitem__` output:**
```
((input_grid, physics), output_grid)
```

| Tensor | Shape | Values | Description |
|---|---|---|---|
| `input_grid` | `(2, H, W)` | [0, 1] | Channel 0: particle occupancy **before** action; Channel 1: tool trajectory map |
| `physics` | `(3,)` | raw floats | `[material.friction, material.density, box.friction]` |
| `output_grid` | `(H, W)` | [0, 1] | Particle occupancy **after** action (training label) |

**Channel 0 / output — particle rendering:**
- **sphere**: filled circle, radius = `particle_size * 500` px
- **cube**: filled rotated rectangle, size = `particle_size * 1000` px, rotated by quaternion yaw

**Channel 1 — tool action rendering:**
- Filled rotated rectangle at **start position** drawn with value **0.5**
- Filled rotated rectangle at **end position** drawn with value **1.0**
- Both at the tool's yaw angle

---

## 3. Training Loop (`train_unet_genesis.py`)

**DataLoader batch shapes** (after optional ×8 augmentation — 4 rotations × 2 flips):

| Variable | Shape | Values | Notes |
|---|---|---|---|
| `inputs` | `(B, 2, H, W)` | {0, 1} | Ch 0: **binary** particle occupancy before action (hard cv2 fill); Ch 1: tool trajectory map |
| `physics` | `(B, 3)` | raw floats | `[material.friction, material.density, box.friction]` |
| `outputs` | `(B, H, W)` | {0, 1} | Binary occupancy label after action (rendered by cv2, hard-filled) |
| `logits` | `(B, H, W)` | ℝ | `model(inputs, physics).squeeze(1)` — **raw output, NOT a probability** |
| `probs` | `(B, H, W)` | (0, 1) | `torch.sigmoid(logits)` — used for all metrics |

### Loss function

The primary loss is **MSE on sigmoid output** (not on raw logits):

```python
probs = torch.sigmoid(logits)
mse   = F.mse_loss(probs, outputs)          # MSE_WEIGHT = 1.0 default
```

Full `combined_loss` (default active terms only — other weights are 0.0):

```python
loss = MSE_WEIGHT   * F.mse_loss(sigmoid(logits), outputs)   # = 1.0
     + MASS_WEIGHT  * |probs.sum() - outputs.sum()| / N       # = 0.2
     # BCEWithLogitsLoss, soft Dice, TV, sharpness, add/remove losses are
     # available but their weights are 0.0 in the default configuration.
```

**Why MSE on sigmoid, not on logits?**  The label `outputs` is binary {0, 1}.
Penalising `(sigmoid(logit) − target)²` rather than `(logit − target)²`
keeps the gradient well-scaled regardless of how saturated the logit is, and
directly trains the network to produce a calibrated occupancy probability.

### Skip connection and output semantics

`NFDUNetFiLM` returns `head(d1) + x[:, 0:1]` where:
- `x[:, 0:1]` = **channel 0 of the input** = soft occupancy ∈ [0, 0.5 typical]
- `head(d1)` = learned correction ∈ ℝ (unconstrained)
- Combined: a **logit** whose sigmoid is the predicted occupancy

At convergence a well-trained model produces:

| Cell type | Typical logit | `sigmoid(logit)` | Comment |
|-----------|---------------|-----------------|---------|
| Material, stays | ≈ +3 to +5 | ≈ 0.95–1.0 | kept by head+skip |
| Material, swept away | ≈ −3 to −5 | ≈ 0.005–0.05 | head strongly negative |
| Empty, stays empty | ≈ −3 to −5 | ≈ 0.005–0.05 | head small-negative, skip≈0 |
| Empty, receives material | ≈ +3 to +5 | ≈ 0.95–1.0 | head strongly positive, skip≈0 |

Crucially: **logits outside (0, 1) produce zero gradient through `clamp(0, 1)`**,
which is why the MPC reward function must operate on `sigmoid(logits)` rather
than on raw logits (see Section 11).

### Tool-channel drawing convention (training)

`_draw_plate_cv2` renders the action channel using:
- **Center** = `(world_x_px, world_y_px)` passed to `cv2.boxPoints` as `(cx, cy)` —
  i.e. cv2 x-axis = world_x (columns), cv2 y-axis = world_y (rows).
- **Angle** = plate yaw from the simulator `run["angles"]`, in radians → converted
  to degrees.  At `angle_sim = 0`: long axis (plate_dim_x = 40 px) along world_x.
  At `angle_sim = π/2`: long axis along world_y (perpendicular to +x travel).
- **Intensities**: start position → 0.5, end position → 1.0.

`UNetFiLMPushModel._draw_plate_soft` draws the equivalent using a differentiable
soft rectangle.  It uses `angle_draw = atan2(Δworld_y, Δworld_x)` (travel direction),
which equals `angle_sim − π/2` — the same physical orientation, just measured
from a different zero.

### Data augmentation

In-loop ×8 per batch: 4 rotations × 2 horizontal flips on spatial dims `(-2,-1)`.
Physics tensor is tiled ×8.  The batch-size argument is divided by 8 to keep
effective memory usage constant.

---

## 4. Models (`GranularDynamics2/myClasses/`)

### `NFDUNetFiLM` *(used by `train_unet_genesis.py` and the MPC)*
```
forward(x: Tensor[B, 2, H, W], props: Tensor[B, 3]) → Tensor[B, 1, H, W]
```
- FiLM conditioning: each conv block receives `γ(props) ⊙ features + β(props)`.
  FiLM MLPs initialised to identity (`γ=1, β=0`) so the model starts as an
  unconditioned U-Net.
- Architecture: 3-level encoder (b, b×2, b×4 channels, b=8 default), MaxPool2d
  downsampling, bilinear upsampling; bottleneck at H/8 × W/8.
- **Output = `head(d1) + x[:, residual_channel:residual_channel+1]`** — a logit.
  With `residual_channel=0` (standard mode) the skip is the input occupancy
  channel (Ch 0, ∈ [0, 1]).  This means the output is ℝ-valued; apply sigmoid
  to get an occupancy probability.
- **No activation on the output** — the skip connection anchors the scale near the
  input occupancy, but the combined value is unbounded.

### `NFDUNetFiLMShallow` *(lightweight variant)*
Same API as `NFDUNetFiLM` with fewer encoder/decoder levels.

### `UNetConditioned` *(legacy)*
```
forward(x: Tensor[B, 2, H, W], physics: Tensor[B, P]) → Tensor[B, 1, H, W]
```
- Concatenates spatially-broadcast physics channels directly onto the input image
  (`physics_dim=6` default — note mismatch with dataset's 3-element vector,
  requires matching at construction time)

### `UNet` *(modular, no physics)*
```
forward(x: Tensor[B, C_in, H, W]) → Tensor[B, C_out, H, W]
```
- Controlled by `structure_parameters` dict (features, activation, bottleneck type, etc.)
- Optional `residual` mode: output = `x[:,0:1] + model_delta`

---

## 5. Data Flow Summary

```
Simulator (Genesis)
      │
      ▼
_{id}_data.pt  +  _{id}_config.yaml
      │
      ▼  PileSweepData.__getitem__
      │  • convert positions m→px
      │  • render particles with cv2 → Channel 0 (H×W) — binary occ {0,1} (hard cv2 fill)
      │  • render tool path with cv2 → Channel 1 (H×W) — intensities 0.5/1.0
      │  • extract physics vector (len 3)
      ▼
input_grid (2,H,W)  +  physics (3,)
      │
      ▼  NFDUNetFiLM.forward()
      │
logits (B,H,W)  =  head(d1)  +  input_grid[:,0:1]        ← raw ℝ values
      │
      ▼  torch.sigmoid()
      │
probs (B,H,W)  ∈ (0,1)                                   ← occupancy probability
      │
      ├── Training:  F.mse_loss(probs, output_grid)  +  mass_loss + …
      │              (output_grid = binary label, rendered by cv2)
      │
      └── MPC:       (probs.clamp(0,1) * score_tensor).sum()   → reward
                     ── sigmoid ensures clamp is identity → gradients flow ──
```

---

## Key Numbers for Real-Data Adaptation

| Quantity | Sim value | Needed from real data |
|---|---|---|
| Grid size H×W | 128×128 (0.128 m @ 1000 px/m) | Match to camera crop/scale |
| Input channels | 2 (particles + tool) | Same — provide binary mask + tool channel |
| Physics vector | 3 floats (friction, density, box_friction) | Estimated or measured per sequence |
| Label | binary occupancy mask (H,W) | Direct binary mask from real image |
| Tool channel | rendered from (start, end, angle) in px | Derive from calibrated robot poses |

---

## 6. Real Experiment Data Format (`RealData/`)

Designed to be as close as possible to the simulation format so that both `PileSweepData`
and `RealPileSweepData` return identical `((input_grid, physics), output_grid)` batches.

### Pre-processing contract (before saving)
The caller is responsible for:
1. **Segmenting** each camera frame into a binary occupancy mask and **resizing** it to the
   target grid resolution (e.g. 128 × 128).
2. **Calibrating** robot/tool positions to image-pixel coordinates in that cropped grid,
   using a homography or known camera intrinsics + extrinsics.
3. Storing tool positions as pixel coordinates with the **top-left corner as origin**,
   consistent with the convention used by the sim dataset's cv2 drawing calls.

### File pairs per run (under `real_data/{experiment_id}/`)
```
_{id}_data.pt      ← torch.save'd dict of tensors
_{id}_config.yaml  ← grid, tool, and physics metadata
```

### `_data.pt` dict keys

| Key | Shape | dtype | Description |
|---|---|---|---|
| `masks_before` | `(N, H, W)` | float32 | Binary occupancy mask **before** the action, values in [0,1] |
| `masks_after` | `(N, H, W)` | float32 | Binary occupancy mask **after** the action |
| `p_starts_px` | `(N, 2)` | float32 | Tool centre at action start `(x_col, y_row)` in pixels |
| `p_stops_px` | `(N, 2)` | float32 | Tool centre at action end `(x_col, y_row)` in pixels |
| `angles` | `(N,)` | float32 | Tool yaw angle in radians |

> **Coordinate note:** `x_col` = horizontal (column) axis, `y_row` = vertical (row) axis,
> matching cv2's `(cx, cy)` convention so that drawing functions are identical to the sim path.

### Config YAML structure
```yaml
grid:
  height: 128        # pixels — must match mask tensor H
  width:  128        # pixels — must match mask tensor W
tool:
  size_px: [40, 2]   # [width_px, height_px] of the tool rectangle
physics:             # known/estimated values; null = unknown (dataset returns 0.0)
  friction:     null
  density:      null
  box_friction: null
experiment:          # free-form metadata (not used by dataset)
  material: chickpeas
  surface:  wood
  date:     "2025-01-01"
```

### Split strategy
Deterministic by hashing the **run file path** (not physics, which may be unknown).
`val_pct=10`, `test_pct=10` by default (same API as `PileSweepData`).

---

## 7. `RealPileSweepData` Dataset (`RealData/dataset.py`)

Drop-in replacement for `PileSweepData` when working with real data.

**Construction:**
```python
RealPileSweepData(
    data_root: str | Path,   # root directory; paths are resolved relative to this
    paths: list[str] | str,  # subdirectories under data_root
    split: "train"|"val"|"test"|None,
    default_physics: [f, d, bf] | None,  # fallback when config has null values
)
```

**`__getitem__` output** — identical to `PileSweepData`:
```
((input_grid: Tensor[2, H, W], physics: Tensor[3]), output_grid: Tensor[H, W])
```

| Channel | Source | Rendering |
|---|---|---|
| `input_grid[0]` | `masks_before[i]` | copied directly — no rendering needed |
| `input_grid[1]` | `p_starts_px`, `p_stops_px`, `angles` | same cv2 rotated-rect draw as sim (start→0.5, end→1.0) |
| `output_grid` | `masks_after[i]` | copied directly |
| `physics` | config `physics.*` or `default_physics` | `[friction, density, box_friction]` |

---

## 8. System Identification (`sysid_unet_cg.py`)

Given a trained `UNetFiLM` and one or more real experiment runs, finds the physics
vector `p*` that minimises MSE between model prediction and observed after-mask.

### Problem formulation
$$p^* = \arg\min_p \sum_i \| \text{UNetFiLM}(x_i,\, p) - y_i \|^2$$

where $x_i$ = `input_grid` (before mask + tool channel), $y_i$ = `masks_after`.

### Physics parameterisation
Each scalar physics value is sigmoid-warped to enforce bounds:

$$p_j = p_j^{\min} + \sigma(\theta_j)\,(p_j^{\max} - p_j^{\min})$$

| Physics param | Min | Max | Unit |
|---|---|---|---|
| `friction` | 0.05 | 0.50 | — |
| `density` | 750 | 5000 | kg/m³ |
| `box_friction` | 0.05 | 0.50 | — |

Raw parameters $\theta \in \mathbb{R}^3$ are initialised to **0** (→ midpoint of each range).

### Modes
- **`global`** (default): single shared $p$ optimised over all provided runs.
- **`per_run`**: separate $p$ per `.pt` file; reports mean ± std across runs at the end.

### Optimiser
Adam on $\theta$ only; model weights are **frozen** (`requires_grad=False`).
Gradients flow through the frozen model's computation graph to reach $\theta$.

### Outputs
- Identified physics printed and saved to `{log_dir}/identified_physics.yaml`
- Loss curve saved as `{log_dir}/loss_curve.png`
- Per-run YAML when `per_run=True`

---

## 10. Coordinate Conventions

Three distinct coordinate systems are used across the pipeline. Getting them
wrong is the single most common source of silent reward ≈ 0 bugs.

### 10.1 Genesis world frame

| Axis | Direction | Notes |
|------|-----------|-------|
| world_x | right (+x) | table width |
| world_y | forward (+y, "north") | table depth |
| world_z | up (+z) | Genesis is z-up |

Workspace is centred at origin; `wkspc_w = 0.064 m` half-width.
Overhead camera at `(0, 0, cam_h=0.3 m)`, looking at origin, with `up=(0,1,0)`.

### 10.2 `depth2fgpcd` / `EulerianWrapper` convention

`depth2fgpcd` converts a depth image → 3-D point cloud using camera intrinsics:

```
fgpcd[:, 0] = (col - cx) * depth / fx  =  cam_x  =  world_x / global_scale
fgpcd[:, 1] = (row - cy) * depth / fy  =  cam_y  =  −world_y / global_scale
```

Image rows increase downward; the camera "up" direction (+world_y) corresponds
to **decreasing** row, so `cam_y = −world_y / gs`.

`_particles_to_occupancy` bins the (cam_x, cam_y) point cloud on a 2-D grid:

```
occ[b,  ix,  iy]   —   ix = camera-x index = world-x index
                    —   iy = camera-y index = −world-y index
```

**EulerianWrapper grid:** `dim 0 = world-x  (ix)`,  `dim 1 = −world-y  (iy)`.
Larger `iy` → more negative world-y (further "south").

### 10.3 Training / PileSweepData convention

`PileSweepData` (and the cv2 drawing code in `sandbox_manipulation_clean.py`)
builds `input_grid` with:

```python
col = world_x * TO_PXL + x_pxl/2   # 1000 px/m, centre at 64 px
row = world_y * TO_PXL + x_pxl/2
cv2.circle(grid_np, (col, row), ...)   # fills grid_np[row, col]
```

So `input_grid[ch, row, col]` where **larger row = larger world_y** (north).

**Dataset grid:** `dim 0 = world-y  (row)`,  `dim 1 = world-x  (col)`.

This is the standard NumPy/cv2 (H × W) layout **but with world-y pointing
upward** — the opposite of a camera image.

### 10.4 Plate orientation convention

The plate occupies a rotated rectangle.  The `(width, height)` passed to
`cv2.boxPoints` is `(plate_dim_x, plate_dim_y)` in pixels:

| angle\_sim | plate\_dim\_x (long, 40 px) | plate\_dim\_y (short, 2 px) |
|-----------|------------------------------|------------------------------|
| 0         | along world-x (cols)         | along world-y (rows)         |
| π/2       | along world-y (rows)         | along world-x (cols)         |

For a plate **perpendicular to its direction of travel**:
`angle_sim = atan2(Δworld_y, Δworld_x) + π/2`.

### 10.5 `UNetFiLMPushModel` convention bridge

`UNetFiLMPushModel.forward` bridges EulerianWrapper ↔ dataset convention:

| Step | Operation | Result shape |
|------|-----------|--------------|
| Input | `occ` (EulerianWrapper) | `(B, world_x, −world_y)` |
| flip dim 1 | `occ.flip([-1])` | `(B, world_x, world_y)` |
| transpose | `.transpose(-2,-1)` | `(B, world_y, world_x)` = dataset |
| ... model forward ... | | `(B, world_y, world_x)` |
| transpose back | `.transpose(-2,-1)` | `(B, world_x, world_y)` |
| flip dim 1 | `.flip([-1])` | `(B, world_x, −world_y)` = EulerianWrapper |

Action center passed to `_draw_plate_soft`: `(world_y_idx, world_x_idx)` —
**world-y first** (matches dataset dim 0).

Draw-angle for `_draw_plate_soft`:
`angle_draw = atan2(Δworld_y, Δworld_x)` (direction of travel, **no +π/2**).
At `angle_draw=0` the plate's long axis is along `dim 0 = world_y` (⊥ to
rightward travel), which matches cv2's `angle_sim=π/2` convention.

---

## 9. Real-Data Training (`train_unet_cg.py`)

Structurally identical to `train_unet_genesis.py`. Key differences:

| | `train_unet_genesis.py` | `train_unet_cg.py` |
|---|---|---|
| Dataset class | `PileSweepData` | `RealPileSweepData` |
| Data root | hardcoded `Genesis/data/` | configurable `data_root` |
| Physics source | sim config (known) | config or `default_physics` (may be 0s) |
| Fine-tuning | — | `pretrained_path` loads weights before training |

---

## 11. Training ↔ MPC Interface: the Sigmoid Requirement

This section documents the critical interface contract between the trained model
and the MPC optimizer.  Violating it causes silent reward ≈ 0 and zero gradients.

### 11.1 What the model outputs

`NFDUNetFiLM.forward()` returns:
```python
logit = head(d1) + x[:, 0:1]   # (B, 1, H, W)  ∈ ℝ  (unbounded)
```
The model is **trained to produce a logit** — not a probability.  The loss
`F.mse_loss(sigmoid(logit), target)` pushes the logit toward values where
`sigmoid(logit) ≈ target ∈ {0, 1}`.  At convergence, typical logit magnitudes
are ±3 to ±5 (sigmoid saturated at ~0.05 or ~0.95).

### 11.2 What the MPC reward function expects

`EulerianAdapter._reward_default` (the optimization reward) computes:
```python
reward = (state_batch.clamp(0.0, 1.0) * score_tensor).reshape(B, -1).sum(-1)
```
`score_tensor ∈ [−1, +1]`: goal cells ≈ +1, non-goal ≈ −`empty_penalty`.

For this to work correctly, `state_batch` must contain occupancy probabilities
in `(0, 1)`, **not raw logits**.

### 11.3 The clamp(0,1) gradient trap

| `state_batch` value | After `clamp(0,1)` | `d(clamp)/d(state)` | Optimizer gradient |
|---|---|---|---|
| logit = +5 (material) | 1.0 | **0** (saturated) | **dead** |
| logit = −5 (empty) | 0.0 | **0** (saturated) | **dead** |
| logit = 0.3 (ambiguous) | 0.3 | 1 | non-zero |
| sigmoid(+5) = 0.993 | 0.993 | 1 (inside range) | non-zero |
| sigmoid(−5) = 0.007 | 0.007 | 1 (inside range) | non-zero |

With raw logits: almost all values are outside `(0, 1)` → `clamp` saturates →
`grad = 0.00000` printed by `mpc.py`.  The optimizer cannot improve the actions.

The observable symptom:
```
iter    1: best=0.0000  mean=-8.7859  std=7.7005  grad=0.00000
iter   50: best=0.0000  mean=-8.7859  std=7.7005  grad=0.00000   # no progress
```
- `best = 0.0`: the best sample predicts empty (all logits ≤ 0 → clamp → 0 → reward = 0).
  This beats actions that predict material in penalty regions (reward < 0).
- `mean = −8.7`: other samples predict material at non-goal cells (logit ≥ 1 → clamp = 1
  → penalty score).
- `grad = 0.00000`: ALL logits outside `(0, 1)` → zero gradient everywhere.

### 11.4 The fix

`UNetFiLMPushModel.forward` applies sigmoid **before** returning:
```python
return torch.sigmoid(occ_pred_ds).transpose(-2, -1).flip(dims=[-1])
```

Effects:
1. **Gradient flows**: sigmoid output ∈ (0, 1) → `clamp` is identity → gradient = 1
   → chain rule reaches `act_seqs` → optimizer converges.
2. **Correct reward values**: `reward = (sigmoid(logit) * score).sum()` uses
   the same quantity the model was trained to optimise.
3. **Multi-step consistency**: each step receives `sigmoid(logit) ∈ (0, 1)` as
   input, matching the training distribution (occupancy maps, not logits).
4. **Gradient magnitude**: even at saturated values (`sigmoid(±5) ≈ 0.007/0.993`),
   `d(sigmoid)/d(logit) = sigmoid(1−sigmoid) ≈ 0.007 > 0` — tiny but non-zero,
   so the optimizer escapes the region over multiple iterations.

### 11.5 Action channel conventions (training vs. MPC)

Both use a plate rectangle drawn at (start, end) with intensities (0.5, 1.0).

| | Training (`_draw_plate_cv2`) | MPC (`_draw_plate_soft`) |
|---|---|---|
| Center format | `(world_x_px, world_y_px)` as cv2 `(cx, cy)` | `(world_y_idx, world_x_idx)` as `(dim0, dim1)` |
| Angle convention | `angle_sim` = plate yaw (from simulator) | `angle_draw = angle_sim − π/2` = travel direction |
| Plate long axis at zero angle | along world_x | along world_y (dim 0) |
| Implementation | cv2 hard-filled, non-differentiable | soft sigmoid mask, differentiable |

Both produce the same physical rectangle (plate perpendicular to travel).
The `+π/2` offset and the dim-swap from `transpose(-2,-1)` cancel exactly.

---

## 12. `EulerianModelWrapper` (`model/eulerian_wrapper.py`)

### Constructor

```python
EulerianModelWrapper(
    user_model,          # nn.Module — any model with forward(occ, start_grid, end_grid)
    grid_bounds,         # dict with x_min/x_max/y_min/y_max/z_min/z_max (normalised)
    grid_res,            # (Nx, Ny) — e.g. (128, 128)
    cam_extrinsic,       # (4,4) view matrix from env.get_cam_extrinsics() — ignored for 'genesis'
    global_scale,        # float — config['dataset']['global_scale']
    splat_sigma=0.0,     # 0 = hard voxel, >0 = Gaussian splat (differentiable but slower)
    occ_threshold=0.5,   # binarisation threshold for occ→particles back-conversion
    action_convention,   # 'flex' (PyFleX) or 'genesis' (overhead z-up camera)
)
```

### `default_bounds()` — heuristic models (1.2× expansion)

```python
EulerianModelWrapper.default_bounds(config, convention='genesis')
# → {x_min: -w_n*1.2, x_max: w_n*1.2, y_min: -w_n*1.2, y_max: w_n*1.2,
#    z_min: 0.45, z_max: 0.55}   (for genesis, z_table=0.5, z_margin=0.05)
```

`UNetFiLMPushModel.default_bounds(config)` uses **exact** ±w_n (no 1.2×) to align
the 1 px/mm grid with the training dataset.

### `predict_one_step_occ(occ_cur, action)`

```
action (B,4) [sx,sy,ex,ey] world metres
  │  _action_to_cam_3d_genesis(action, global_scale)
  ▼
s_3d_cam, e_3d_cam  (B,3)  [cam_x, cam_y, z_norm]
  │  _cam3d_to_grid(pts)
  ▼
start_grid, end_grid  (B,3)  [ix_cam_x, iy_cam_y, iz]  grid indices in [0,N-1]
  │  user_model.forward(occ_cur, start_grid, end_grid)
  ▼
occ_pred  (B, Nx, Ny)  same convention as occ_cur
```

### `_cam3d_to_grid(pts_3d)` (line ≈ 658)

Maps `(B,3)` normalised cam coords to `(B,3)` fractional grid indices:
```
ix = (cam_x − x_min) / (x_max − x_min) * (Nx − 1)
iy = (cam_y − y_min) / (y_max − y_min) * (Ny − 1)
```
For 2-D grids (`_get_axes(2)` returns `('x','y')`): uses cam_x and cam_y only.

### `initial_occ_from_particles(s_cur)` (line ≈ 490)

```python
s_cur: (B, N, 3)  particles in normalised cam coords
→ occ:  (B, Nx, Ny)  float32 occupancy in [0,1], detached (no grad)
```
Calls `_particles_to_occupancy` with `sigma=splat_sigma` (default 0 → hard voxel).

### `_get_axes(ndim)` / axis convention

| `ndim` | Axes | Grid dim 0 | Grid dim 1 |
|--------|------|-----------|-----------|
| 2 | `('x','y')` | cam_x = world_x | cam_y = −world_y |
| 3 | `('x','y','z')` | cam_x | cam_y | cam_z |

---

## 13. `UNetFiLMPushModel` (`model/eulerian_wrapper.py`, line ≈ 1098)

### Role

Wraps a trained `NFDUNetFiLM` as a `user_model` plug-in for `EulerianModelWrapper`.
Bridges from EulerianWrapper grid convention (dim0=world_x, dim1=−world_y) to the
dataset convention (dim0=world_y, dim1=world_x) expected by the model.

### Constructor

```python
UNetFiLMPushModel(
    unet_film,           # trained NFDUNetFiLM instance
    physics,             # Tensor(3,): [particle_friction, density, box_friction]
    grid_size,           # (Nx, Ny) — must match training resolution
    plate_length_px=40.0,
    plate_width_px=2.0,
    sigma=1.5,           # soft-mask pixel smoothness
)
```

### `forward(occ, action_start, action_end)` — full pipeline

```
occ: (B, Nx, Ny) EulerianWrapper — dim0=world_x, dim1=−world_y
  │ flip(dim1) + transpose(-2,-1)
  ▼
occ_ds: (B, Ny, Nx) dataset — dim0=world_y, dim1=world_x
  │
  │  action_start/end: (B,3) [ix_cam_x, iy_cam_y, iz]
  │  iy_ds = (Ny−1) − iy_cam   (flips cam_y → world_y index)
  │  angle  = atan2(iy_e_ds − iy_s_ds,  ix_e − ix_s)  [travel direction, no +π/2]
  │  center = (iy_ds, ix_cam)  [dim0=world_y, dim1=world_x]
  │  _draw_plate_soft(center, angle, intensity)
  ▼
act_ch: (B, Nx, Ny) soft plate in dataset convention
  │
  │  x = stack([occ_ds, act_ch], dim=1)  — (B, 2, Nx, Ny)
  │  phys = self._physics.expand(B, -1)  — (B, 3)
  │  unet_film(x, phys) → raw logit (B, 1, Nx, Ny)
  │  .squeeze(1) → (B, Nx, Ny)
  │  + residual (input channel 0 = occ_ds) already applied inside NFDUNetFiLM
  ▼
occ_pred_ds: (B, Nx, Ny) raw logit
  │ sigmoid  →  (B, Nx, Ny) occupancy ∈ (0,1)
  │ transpose(-2,-1) + flip(dim1)           [inverse of opening transform]
  ▼
return: (B, Nx, Ny) EulerianWrapper convention, values in (0,1)
```

**Why sigmoid here?**  `NFDUNetFiLM` is trained with `MSE(sigmoid(logit), target)`.
Raw logits ±3–5 are outside `clamp(0,1)` → zero MPC gradient.  Sigmoid maps to
(0.007, 0.993) → `clamp` is identity → gradient flows. See §11.

### `default_bounds(config)` (no 1.2× expansion)

```python
# w_n = wkspc_w / global_scale = 0.064 / 0.6 ≈ 0.1067
{x_min: -w_n, x_max: w_n, y_min: -w_n, y_max: w_n,
 z_min: 0.45, z_max: 0.55}
```

Grid exactly covers the physical workspace box: 1 grid cell = 1 mm.

### `_draw_plate_soft` angle convention

`angle = atan2(Δworld_y, Δworld_x)` = direction of travel.
At `angle=0` (travel along world_x): plate long-axis is along `dim 0` = world_y
(i.e. plate is perpendicular to travel) — matches training data.
The physical simulator angle `angle_sim = angle + π/2`.

---

## 14. `GenesisEnv` (`env/genesis_env.py`)

### Camera

```python
# camera at (0, 0, cam_h=0.3 m), lookat=(0,0,0), up=(0,1,0)
# render() → obs (H, W, 5): [R, G, B, is_material, depth_raw]
#   material pixels:   obs[..., 4] = cam_h
#   background pixels: obs[..., 4] = 2 * cam_h
# global_scale = 2 * cam_h  → depth_norm = obs[...,4] / global_scale
#   material: depth_norm = 0.5       ← foreground threshold < 0.599/0.8 ≈ 0.749
#   background: depth_norm = 1.0
get_cam_params() → [fx, fy, cx, cy]   # intrinsics only; no extrinsic needed
```

### `step(action)` — action convention

```python
# action = [sx, sy, ex, ey]  world metres (x,y only; z is plate height)
angle = atan2(ey - sy, ex - sx) + π/2   # plate yaw = perpendicular to travel
```

`_action_to_cam_3d_genesis(action, global_scale)`:
```python
s = [sx/gs, -sy/gs, 0.5]   # cam_x = world_x/gs, cam_y = -world_y/gs
e = [ex/gs, -ey/gs, 0.5]
```

### Foreground detection

`_FG_DEPTH_THRESHOLD = 0.599 / 0.8 ≈ 0.749` (defined in both `mpc.py` and `adapters.py`).
Material pixels normalised to 0.5 → well below threshold → correctly flagged as foreground.

---

## 15. MPC Inference Data Flow

```
env.render()  →  obs (H,W,5)
    │
    │  depth = obs[...,4] / global_scale
    │  pts   = depth2fgpcd(depth, depth < _FG_DEPTH_THRESHOLD, cam_params)
    │          → (N, 3)  [cam_x, cam_y, 0.5] = [world_x/gs, -world_y/gs, 0.5]
    │  pts_t = pts.unsqueeze(0)  →  (1, N, 3)
    │  EulerianModelWrapper.initial_occ_from_particles(pts_t)
    │          → _particles_to_occupancy(pts_t, bounds, grid_res, sigma=0)
    ▼
occ_cur  (1, Nx, Ny)  [dim0=world_x, dim1=−world_y]  binary {0,1}
    │
    │  EulerianAdapter.expand_state(occ_cur, n_sample)
    ▼
occ_batch  (n_sample, Nx, Ny)                          ← cloned, grad-ready
    │
    │  Adam optimizer on action_seqs (n_sample, n_look_ahead, 4)
    │  for each iter:
    │    predict_one_step_occ(occ_batch, act_batch)
    │      → UNetFiLMPushModel.forward()              (see §13)
    ▼
occ_pred  (n_sample, Nx, Ny)  values in (0,1)           ← sigmoid applied
    │
    │  reward = (occ_pred.clamp(0,1) * score_tensor).sum(dim=[1,2])
    │  loss   = -reward.mean()
    │  loss.backward(); optimizer.step()
    ▼
best action  →  env.step(action)
```

**Key numbers (128×128 grid, default config):**
- `global_scale = 0.6 m`, `wkspc_w = 0.064 m`, `w_n = 0.1067`
- Grid: 128×128 px, 1 px = 1 mm (workspace 128 mm × 128 mm)
- `cam_h = 0.3 m`, `fx = fy ≈ 360 / tan(22.5°) ≈ 869`
- Particle footprint in grid: ≈ 9 px diameter for a 9 mm cube

---

## 16. Physics Parameter Reference

### Training data values (from `Genesis/data/corl/cube/`)

All training configs in `corl/cube/` use **fixed** physics (no variation across runs):

| Physics param | Dataset key | Raw training value | Unit |
|---|---|---|---|
| particle friction | `material.friction` | 0.05 | — |
| particle density | `material.density` | 750.0 | kg/m³ |
| box friction | `box.friction` | 0.05 | — |

### Physics normalisation (`PileSweepData._det_physics`)

The dataset normalises raw physics to `[0, 1]` before returning them to the training loop:

```python
physics[0] = (friction     - 0.05) / (0.50 - 0.05)   # range: friction ∈ [0.05, 0.50]
physics[1] = (density      -  750) / (5000 -  750)   # range: density  ∈ [750,  5000] kg/m³
physics[2] = (box_friction - 0.05) / (0.50 - 0.05)   # range: friction ∈ [0.05, 0.50]
```

For the standard training values `[0.05, 750, 0.05]` the normalised vector is `[0, 0, 0]`.
This same normalisation must be applied at inference so the FiLM generator receives
inputs consistent with training.

### MPC config physics (`simple_mpc/config/`)

```yaml
particle_friction: 0.05    # raw value — normalised inside load_model
particle_density:  750.0   # raw value — normalised inside load_model
box_friction:      0.05    # raw value — normalised inside load_model
```

### `load_model` physics vector (`run_experiments.py`)

```python
_f  = cfg['dataset'].get('particle_friction', 0.05)
_d  = cfg['dataset'].get('particle_density',  750.0)
_bf = cfg['dataset'].get('box_friction',       0.05)
physics_vec = torch.tensor([
    (_f  - 0.05) / (0.50 - 0.05),
    (_d  -  750) / (5000 -  750),
    (_bf - 0.05) / (0.50 - 0.05),
], dtype=torch.float32)
```
Set `particle_friction`, `particle_density`, and `box_friction` in your experiment config
under `dataset:` to match the actual simulation parameters; `load_model` normalises them.
