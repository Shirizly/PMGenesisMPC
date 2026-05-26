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

| Variable | Shape | Notes |
|---|---|---|
| `inputs` | `(B, 2, H, W)` | float32, [0,1] |
| `physics` | `(B, 3)` | float32, raw physics values |
| `outputs` | `(B, H, W)` | float32, [0,1] — the label |
| `pred_next` | `(B, 1, H, W)` | model output, squeezed to `(B, H, W)` for loss |

**Loss:** `MSELoss(pred_next.squeeze(1), outputs)`

**Data augmentation** (in-loop, multiplies batch by 8):
- Rotations: 0°, 90°, 180°, 270° on spatial dims `(-2,-1)`
- Mirrors: horizontal flip of each rotation
- `physics` is repeated ×8 (tile along batch dim)

---

## 4. Models (`GranularDynamics2/myClasses/`)

### `UNetFiLM` *(used in `train_unet_genesis.py`)*
```
forward(x: Tensor[B, 2, H, W], physics: Tensor[B, 3]) → Tensor[B, 1, H, W]
```
- `physics_dim=3` by default
- Physics vector → MLP → per-stage FiLM (γ, β) applied as `feature * γ + β`
- Architecture: encoder [64, 128], bottleneck 256, decoder [128, 64], 1×1 conv head
- **Spatial resolution is preserved** (no stride, only MaxPool+ConvTranspose)

### `UNetConditioned`
```
forward(x: Tensor[B, 2, H, W], physics: Tensor[B, P]) → Tensor[B, 1, H, W]
```
- Concatenates spatially-broadcast physics channels directly onto the input image (`physics_dim=6` default — note mismatch with dataset's 3-element vector, requires matching at construction time)

### `UNet` (modular, no physics)
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
      │  • render particles → Channel 0 (H×W)
      │  • render tool path → Channel 1 (H×W)
      │  • extract physics vector (len 3)
      ▼
input_grid (2,H,W)  +  physics (3,)  →  [DataLoader]  →  UNetFiLM  →  pred (B,1,H,W)
                                                                               │
output_grid (H,W)  ──────────────────────────────────────────────────  MSELoss
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

## 9. Real-Data Training (`train_unet_cg.py`)

Structurally identical to `train_unet_genesis.py`. Key differences:

| | `train_unet_genesis.py` | `train_unet_cg.py` |
|---|---|---|
| Dataset class | `PileSweepData` | `RealPileSweepData` |
| Data root | hardcoded `Genesis/data/` | configurable `data_root` |
| Physics source | sim config (known) | config or `default_physics` (may be 0s) |
| Fine-tuning | — | `pretrained_path` loads weights before training |
