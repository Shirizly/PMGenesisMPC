# OT-Guided Action Sampling: Design & Implementation Plan

## 1. Motivation

`PhysicsAwareActionSampler` samples push start/end positions uniformly within physics-aware workspace bounds. It is fast and avoids plate collisions with walls, but has no knowledge of *where* mass is or *where* it needs to go. Most sampled actions miss the pile entirely or push in unhelpful directions, so the gradient-descent optimisation carries a heavy burden.

`OTPlannerSparse` already computes, for every occupied cell, a displacement vector pointing toward where mass "should" go according to the optimal transport plan. Regions where those vectors are locally aligned (low `div_mag`) are **coherent-flow** regions: the OT plan agrees that all mass there should move in roughly the same direction. A push that sweeps through such a region, travelling in that direction, is highly likely to achieve useful transport without wasted effort.

**Core idea**: seed the MPC candidate pool with pushes derived directly from OT coherent-flow regions, then let gradient descent refine them. The remaining candidates are random (physics-aware) for exploration.

---

## 2. Current State of OTPlannerSparse (what already works)

`OTPlannerSparse.solve(source, goal)` is complete and returns:

| Field | Shape | Content |
|---|---|---|
| `src_coords` | `(n_src, 2)` | `(col, row)` of each occupied source cell |
| `v_sparse` | `(n_src, 2)` | Displacement vector per occupied cell |
| `vectors_2d` | `(n, n, 2)` | Dense displacement field, zero outside support |
| `div` | `(n, n)` | Signed divergence (positive = local source, negative = sink) |
| `div_mag` | `(n, n)` | `|div|`; **low values = coherent/laminar flow = good push region** |
| `source_mask` | `(n, n)` | `True` where source has mass |

The coordinate convention throughout is `(col, row) = (x, y)`, origin at lower-left, matching `matplotlib origin='lower'` and the training-data convention where `col = world_x * 1000 + n/2`, `row = world_y * 1000 + n/2`.

**What is missing**: a method to extract concrete push action proposals `(sx, sy, ex, ey)` in world coordinates from the OT result.

---

## 3. What to Add to OTPlannerSparse

### 3.1 `extract_push_candidates` (public method)

This is the single addition needed. It converts `OTResult` into a ranked list of push proposals.

**Signature**:
```python
def extract_push_candidates(
    self,
    result          : OTResult,
    wkspc_w         : float,
    n_candidates    : int  = 8,
    div_percentile  : float = 30.0,   # coherent if div_mag < this percentile of occupied cells
    min_region_cells: int  = 4,       # ignore very small fragments
    approach_px     : float = 3.0,    # how far upstream of the region to start the push
    exit_px         : float = 3.0,    # how far downstream to end the push
) -> list[dict]:
```

**Returns** a list (up to `n_candidates`) of dicts, sorted by descending score, each containing:
```python
{
    'start_world': np.ndarray,  # (2,)  [world_x, world_y] of push start
    'end_world':   np.ndarray,  # (2,)  [world_x, world_y] of push end
    'direction':   np.ndarray,  # (2,)  unit vector of push travel direction
    'score':       float,       # region_area / (mean_div_mag + ε)  — higher is better
    'centroid_grid': np.ndarray, # (2,)  [col, row] for debugging
}
```

**Algorithm**:

1. **Coherence mask**:
   ```
   occupied       = result.source_mask                      # (n, n) bool
   threshold      = np.percentile(result.div_mag[occupied], div_percentile)
   coherent_mask  = occupied & (result.div_mag <= threshold) # (n, n) bool
   ```

2. **Connected components** (each component = one candidate push region):
   ```python
   from scipy.ndimage import label
   labeled, n_labels = label(coherent_mask)
   ```

3. **For each region label `k` in 1..n_labels**:
   - `region_cells_rc` = `(row, col)` pairs where `labeled == k`
   - `region_coords_colrow` = same as `(col, row)` pairs (flip indices)
   - Skip if `len(region_cells_rc) < min_region_cells`
   - `centroid_col = region_coords_colrow[:, 0].mean()`
   - `centroid_row = region_coords_colrow[:, 1].mean()`
   - Mean displacement: look up `result.v_sparse` for the subset of `result.src_coords` that fall in this region.
     - More efficiently: use `result.vectors_2d[row, col]` for each `(row, col)` in the region.
     - `dx = result.vectors_2d[rows, cols, 0].mean()`
     - `dy = result.vectors_2d[rows, cols, 1].mean()`
   - `d_hat = (dx, dy) / ||(dx, dy)||` — skip region if `||(dx, dy)|| < 1e-3` (no net movement)
   - **Push extent**: project region cells onto `d_hat`:
     - `cells_colrow` = `np.stack([cols, rows], axis=1)` — shape `(k, 2)`
     - `centroid` = `np.array([centroid_col, centroid_row])`
     - `proj = (cells_colrow - centroid) @ d_hat` — shape `(k,)`, scalar projections along push direction
     - `p_min, p_max = proj.min(), proj.max()` — upstream/downstream extent of region
   - **Start and end in grid coords** (plate center path):
     - `start_grid = centroid + (p_min - approach_px) * d_hat`
     - `end_grid   = centroid + (p_max + exit_px)    * d_hat`
   - **Convert to world coordinates**:
     - `world_x = (col_coord - n/2) * (2 * wkspc_w / n)`
     - `world_y = (row_coord - n/2) * (2 * wkspc_w / n)`
     - (Equivalently, for n=128 and wkspc_w=0.064: `world_x = (col - 64) / 1000`)
   - **Score** = `len(region_cells_rc) / (result.div_mag[rows, cols].mean() + 1e-6)` — rewards large, coherent regions
   - Clip start and end world coords to `[-wkspc_w + margin, wkspc_w - margin]²`

4. Sort by score descending; return top `n_candidates`.

> **Edge case — no coherent regions**: If the threshold produces no connected components with `>= min_region_cells` cells, widen the threshold to the 50th percentile and retry. If still empty, return an empty list (the sampler falls back to purely random sampling).

> **Note on `div_mag` vs. vector-alignment**:  `div_mag` is already computed and correctly captures what we need: zero divergence means the flow is perfectly laminar/solenoidal. An alternative is to compute the circular variance of displacement directions in each region, but `div_mag` is cheaper and equally indicative.

---

## 4. Pre-Step: `CollisionAwareActionSampler`

### 4.1 Motivation

`PhysicsAwareActionSampler` prevents the plate from hitting the workspace walls but allows its footprint to overlap existing granular material at the start of a push. In simulation this causes an instantaneous contact impulse before the sweep even begins, perturbing the pile in ways the dynamics model cannot predict. This pre-step introduces `CollisionAwareActionSampler`: a drop-in replacement that guarantees every sampled **start** position is free of material.

Only the start position is collision-checked. The end position is intentionally allowed inside the material — the plate is supposed to sweep *through* it.

### 4.2 Core Primitive: Morphological Dilation

For plate yaw θ the set of forbidden start positions is the **morphological dilation** of the binary occupancy grid O with a structuring element K_θ shaped as the plate footprint rotated by θ:

```
forbidden_θ  =  dilate(O, K_θ)
valid_θ      =  ~forbidden_θ  ∩  workspace_bounds_θ
```

**Why dilation?** If the plate centre is at p, the plate occupies `{p} ⊕ K_θ` (Minkowski sum). The plate overlaps occupied cell c iff `c ∈ {p} ⊕ K_θ`, i.e., `p ∈ {c} ⊕ K_θ`. The union over all occupied c is `dilate(O, K_θ)` by definition. Therefore `valid_θ` is exactly the set of plate-centre positions where the plate is entirely clear of material.

### 4.3 Orientation Discretisation

The plate has 180° symmetry, so yaw needs only to cover `[0, π)`. Discretise into `n_angles` uniform steps:

```
θ_k = k · π / n_angles,   k = 0, 1, …, n_angles − 1
```

For the standard plate (L = 40 px, W = 2 px) the footprint changes substantially between 0° and 90° (clearance mainly in x vs y). A step of 22.5° (8 orientations) is accurate enough for seeding; 16 orientations is a safe default for tighter control.

**Recommended default**: `n_angles = 8`.

### 4.4 Kernel Construction

Precomputed once at `__init__`. Using cv2 (already a project dependency):

```python
def _make_plate_kernel(L_px: int, W_px: int, theta_rad: float) -> np.ndarray:
    """Binary kernel = filled plate footprint at yaw theta_rad."""
    theta_deg = np.degrees(theta_rad)
    c, s = abs(np.cos(theta_rad)), abs(np.sin(theta_rad))
    # Exact AABB of the rotated rectangle (+3 px for sub-pixel safety, must be odd)
    bbox_h = int(np.ceil(L_px * s + W_px * c)) + 3
    bbox_w = int(np.ceil(L_px * c + W_px * s)) + 3
    bbox_h += (bbox_h % 2 == 0)
    bbox_w += (bbox_w % 2 == 0)
    center = (bbox_w // 2, bbox_h // 2)
    kernel = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
    pts = np.int32(cv2.boxPoints((center, (L_px, W_px), theta_deg)))
    cv2.fillPoly(kernel, [pts], 1)
    return kernel
```

`cv2.boxPoints` uses the same angle convention (degrees CCW from +x axis) as the training pipeline, so kernels are directly consistent with how the plate is drawn in `sandbox_manipulation_clean.py`.

### 4.5 Per-Orientation Workspace Bounds

The exact per-angle AABB formula (taken from `sandbox_manipulation_clean.py`, which already uses it for data collection) is:

```
c_k = |cos θ_k|,  s_k = |sin θ_k|
valid_half_x_k = box_half_x − (c_k · L/2 + s_k · W/2 + safety_margin)
valid_half_y_k = box_half_y − (s_k · L/2 + c_k · W/2 + safety_margin)
```

This is stricter than the conservative `L/2` used by `PhysicsAwareActionSampler` at 0° and 90° (where it equals exactly `L/2`), and is *less* strict at 45° (where it equals `(L + W)/2·√2/2 ≈ 0.85·L/2`). It correctly expands the valid start region at diagonal orientations.

One boolean mask `workspace_masks[k]` per orientation is precomputed at `__init__` (does not change with occupancy).

### 4.6 `PlateCollisionChecker` Helper Class

Extract all collision-checking logic into a standalone helper so that both `CollisionAwareActionSampler` **and** `OTGuidedActionSampler` can reuse it via composition:

```python
class PlateCollisionChecker:
    def __init__(
        self,
        grid_size      : int,
        wkspc_w        : float,   # workspace half-width in metres
        plate_length_m : float,   # L
        plate_width_m  : float,   # W
        n_angles       : int   = 8,
        safety_margin_m: float = 0.01,
    ):
        # self._kernels[k]         : (h_k, w_k) uint8  — plate footprint at θ_k
        # self._workspace_masks[k] : (H, W) bool       — valid centre region at θ_k
        # self._valid_pts[k]       : (M_k, 2) int32    — populated by update()
        ...

    def update(self, occupancy_grid: np.ndarray) -> None:
        """Recompute valid_pts[k] from the current (H, W) binary occupancy."""
        occ = (occupancy_grid > 0).astype(np.uint8)
        for k in range(self._n_angles):
            dilated   = cv2.dilate(occ, self._kernels[k])   # (H, W) uint8
            valid     = (dilated == 0) & self._workspace_masks[k]
            rows, cols = np.where(valid)
            self._valid_pts[k] = np.stack([cols, rows], axis=1).astype(np.int32)

    def sample_starts(
        self,
        k_assignments : np.ndarray,  # (n_sample, n_ahead) int  — orientation index per slot
        fallback_fn,                 # callable(n, n_ahead) → (n, n_ahead, 2) col/row, used when valid_pts[k] is empty
    ) -> np.ndarray:                 # (n_sample, n_ahead, 2)  col/row
        ...
```

### 4.7 `CollisionAwareActionSampler` Class

Sampling proceeds orientation-by-orientation (vectorised within each bucket):

```python
class CollisionAwareActionSampler(ActionSampler):
    def __init__(self, grid_size, wkspc_w, plate_length=0.04, plate_width=0.002,
                 safety_margin=0.01, n_angles=8, d_min=None, d_max=None):
        self._checker  = PlateCollisionChecker(...)
        self._d_min    = d_min or plate_length / 2          # default: half plate-length
        self._d_max    = d_max or 2.0 * np.sqrt(2) * wkspc_w  # max diagonal push
        ...

    def update_state(self, source_grid: np.ndarray, goal_grid=None) -> None:
        self._checker.update(source_grid)

    def sample(self, n_sample, n_ahead, act_lo, act_hi, device='cuda') -> torch.Tensor:
        n         = self._grid_size
        pxl_scale = (2 * self._wkspc_w) / n              # metres per pixel
        k_assign  = np.arange(n_sample) % self._n_angles  # round-robin orientation assignment
        acts      = np.zeros((n_sample, n_ahead, 4), dtype=np.float32)

        for k in range(self._n_angles):
            bucket    = np.where(k_assign == k)[0]         # (b,) sample indices
            if len(bucket) == 0:
                continue
            valid_pts = self._checker.get_valid_pts(k)     # (M_k, 2) col/row

            if len(valid_pts) == 0:                        # fallback: unconstrained physics-aware
                acts[bucket] = self._fallback.sample(len(bucket), n_ahead, act_lo, act_hi).numpy()
                continue

            θ_k          = k * np.pi / self._n_angles
            travel_angle = θ_k - np.pi / 2               # travel ⊥ plate long axis
            tvec         = np.array([np.cos(travel_angle), np.sin(travel_angle)])

            # Sample starts from valid grid positions (same mask for all n_ahead steps)
            idxs = np.random.randint(0, len(valid_pts), size=(len(bucket), n_ahead))
            sel  = valid_pts[idxs]                         # (b, n_ahead, 2) col/row
            sx   = (sel[..., 0] - n / 2) * pxl_scale      # world_x
            sy   = (sel[..., 1] - n / 2) * pxl_scale      # world_y

            # Sample push distance and compute ends
            d    = np.random.uniform(self._d_min, self._d_max, size=(len(bucket), n_ahead))
            phy_vx, phy_vy = self._phy_v_k[k]             # per-orientation workspace half-widths
            ex   = np.clip(sx + d * tvec[0], -phy_vx, phy_vx)
            ey   = np.clip(sy + d * tvec[1], -phy_vy, phy_vy)

            acts[bucket] = np.stack([sx, sy, ex, ey], axis=-1)

        return torch.tensor(acts, device=device, requires_grad=True)
```

> **`n_ahead > 1`**: The same current-state valid masks are used for all look-ahead steps. This is conservative (step-2 material positions are unknown) but safe.

### 4.8 Integration in `OTGuidedActionSampler`

`OTGuidedActionSampler` composes a `PlateCollisionChecker` for its random-fill portion:

- `update_state(source, goal)` calls both `self._checker.update(source)` and `self._planner.solve(source, goal)`.
- The `n_rand` fill samples are drawn via the collision-aware orientation-stratified logic (delegating to `PlateCollisionChecker`), not the original unconstrained `PhysicsAwareActionSampler`.
- The `n_ot` OT-derived starts are upstream of the material by construction (`start_grid = centroid + (p_min − approach_px) · d_hat`), so they are inherently collision-free. Optionally validate against `PlateCollisionChecker` as a sanity check.

### 4.9 Performance

| Backend | Per dilation | 8 orientations total | Notes |
|---|---|---|---|
| `scipy.ndimage.binary_dilation` | 3–8 ms | 24–64 ms | Simple, no extra dep |
| **`cv2.dilate`** | 0.3–1 ms | **~3–5 ms** | **Recommended** — already a dependency |
| PyTorch `F.conv2d` (GPU, batched) | — | ~1–2 ms | Good if GPU is available and kernel padding is handled |

For a 128 × 128 grid with 8 orientations, the full `update` call via `cv2.dilate` takes **≈ 3–5 ms** — negligible compared to Sinkhorn (∼100 ms) or the MPC rollout (∼500 ms per step).

Memory: 8 boolean masks of 128 × 128 ≈ 16 KB; `valid_pts` lists ≈ up to ∼640 KB when the pile is small. Both are negligible.

---

## 5. New Class: `OTGuidedActionSampler`

Add this to `simple_mpc/action_sampler.py`, importing `OTPlannerSparse` and `OTResult` from `simple_mpc.ot_planner`.

### 5.1 Constructor
```python
class OTGuidedActionSampler(ActionSampler):
    def __init__(
        self,
        grid_size      : int,
        wkspc_w        : float,
        reg            : float = 0.002,   # Sinkhorn regularisation
        ot_fraction    : float = 0.7,     # fraction of n_sample drawn from OT guidance
        noise_std_m    : float = 0.005,   # Gaussian noise (metres) added to OT proposals
        plate_length   : float = 0.04,
        plate_width    : float = 0.002,
        safety_margin  : float = 0.01,
        div_percentile : float = 30.0,
        n_ot_seeds     : int   = 8,       # max OT push proposals to generate per step
    ):
```

Fields:
- `self._planner = OTPlannerSparse(grid_size, reg=reg)` — cached across steps
- `self._candidates: list[dict] = []` — updated by `update_state`
- `self._checker = PlateCollisionChecker(grid_size, wkspc_w, plate_length, plate_width, n_angles, safety_margin)` — shared with `CollisionAwareActionSampler`; provides collision-free random-fill samples (§4.6)

### 5.2 `update_state` (call once per MPC step, before `sample`)
```python
def update_state(self, source_grid: np.ndarray, goal_grid: np.ndarray) -> None:
    """Re-solve the OT plan and update collision masks for the current state."""
    self._checker.update(source_grid)   # refresh valid-start masks for random-fill portion
    result = self._planner.solve(source_grid, goal_grid)
    self._candidates = self._planner.extract_push_candidates(
        result, self._wkspc_w,
        n_candidates=self._n_ot_seeds,
        div_percentile=self._div_percentile,
    )
```

### 5.3 `sample`
```python
def sample(
    self,
    n_sample: int,
    n_ahead : int,
    act_lo  : np.ndarray,
    act_hi  : np.ndarray,
    device  : str = 'cuda',
) -> torch.Tensor:
```

**Logic**:
1. `n_ot = int(n_sample * self._ot_fraction)` if `self._candidates` else `0`
2. `n_rand = n_sample - n_ot`

3. **Build OT-guided candidates** (`n_ot` samples):
   - Cycle through `self._candidates` (with repetition if `n_ot > len(self._candidates)`)
   - For each candidate, take its `start_world`, `end_world` and perturb with Gaussian noise: `+ N(0, noise_std_m)`
   - This gives `[sx, sy, ex, ey]` in world coords for step 0 of the action sequence
   - For steps `1..n_ahead-1`: use random physics-aware sampling (the OT plan is only valid for the current state)

4. **Build random candidates** (`n_rand` samples, collision-free):
   - Sample via `CollisionAwareActionSampler` orientation-stratified logic, delegating to `self._checker` — guarantees random fills also have valid start positions (no extra compute beyond what `update_state` already did)

5. Stack OT-guided and random, convert to `torch.Tensor` on `device`, `requires_grad=True`

**Fallback**: if `n_ot == 0` (no candidates), return `self._fallback.sample(n_sample, n_ahead, act_lo, act_hi, device)` unchanged.

### 5.4 `make_action_sampler` factory extension
```python
elif sampler_type == 'collision_aware':
    valid = {'grid_size', 'wkspc_w', 'plate_length', 'plate_width',
             'safety_margin', 'n_angles', 'd_min', 'd_max'}
    return CollisionAwareActionSampler(**{k: v for k, v in kwargs.items() if k in valid})

elif sampler_type == 'ot_guided':
    valid = {
        'grid_size', 'wkspc_w', 'reg', 'ot_fraction', 'noise_std_m',
        'plate_length', 'plate_width', 'safety_margin', 'div_percentile', 'n_ot_seeds',
    }
    return OTGuidedActionSampler(**{k: v for k, v in kwargs.items() if k in valid})
```

`grid_size` and `wkspc_w` are **required** for OT-guided sampling (must be passed by the caller in `run_simple_mpc`).

---

## 6. Integration with `mpc.py`

### 6.1 Create the sampler (in `run_simple_mpc`)

The `make_action_sampler` call already exists. The only change: pass `grid_size` and `wkspc_w` when `action_sampler_type == 'ot_guided'`:

```python
action_sampler = make_action_sampler(
    action_sampler_type,
    plate_length = plate_length,
    safety_margin = plate_safety,
    grid_size = int(2 * wkspc_w * 1000),   # e.g. 128 for wkspc_w=0.064
    wkspc_w   = wkspc_w,
)
```

This is backward-compatible: `PhysicsAwareActionSampler` and `RandomUniformSampler` ignore unknown kwargs via the existing `valid`-key filtering.

### 6.2 `update_state` hook in the MPC loop

Add **two lines** before the `action_sampler.sample(...)` call inside the `for i in range(n_mpc)` loop:

```python
# -- optionally seed OT-guided sampler with current occupancy --------
if hasattr(action_sampler, 'update_state'):
    _src_grid, _goal_grid = adapter.get_ot_grids(state_init)
    action_sampler.update_state(_src_grid, _goal_grid)
```

### 6.3 Adapter method: `get_ot_grids`

Add to the base `ModelAdapter` class in `adapters.py`:

```python
def get_ot_grids(self, state: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (source_grid, goal_grid) as (H, W) float32 numpy arrays.
    
    Returns (None, None) by default; subclasses that support OT override this.
    """
    return None, None
```

**`EulerianAdapter` override**:
- `state` is already a `(1, H, W)` occupancy tensor (or `(H, W)`)
- `source_grid = state.squeeze().cpu().numpy()`
- `goal_grid` = `self._subgoal` (already stored as numpy in the adapter)

**`GNNAdapter`**: leave as the default `(None, None)`. The OT sampler will see an empty candidates list and fall back to purely random. Log a warning if `action_sampler_type == 'ot_guided'` is requested with a GNN model.

The `update_state` guard in `mpc.py` then becomes:
```python
if hasattr(action_sampler, 'update_state'):
    _src, _goal = adapter.get_ot_grids(state_init)
    if _src is not None:
        action_sampler.update_state(_src, _goal)
```

---

## 7. Coordinate Conventions (Critical)

From `CODEBASE_OVERVIEW.md §10.3` and the `OTPlannerSparse` docstring:

| System | col/x origin | row/y origin | y direction |
|---|---|---|---|
| OT grid (origin lower-left) | col=0 → left edge | row=0 → bottom edge | row increases upward |
| Training / PileSweepData | col = world_x * 1000 + n/2 | row = world_y * 1000 + n/2 | row increases with world_y (upward) |
| MPC world coords | — | — | standard +y = forward |

**Conversion** (OT grid → world, for n=128, wkspc_w=0.064 m):
```
world_x = (col - n/2) / 1000   =  (col - 64) / 1000
world_y = (row - n/2) / 1000   =  (row - 64) / 1000
```

General form (works for any square grid):
```
world_coord = (grid_coord - n/2) * (2 * wkspc_w / n)
```

The two systems are consistent: row=0 → `world_y = -0.064` (south edge), row=128 → `world_y = +0.064` (north edge). **No axis flip needed.**

---

## 8. Design Decisions and Tradeoffs

### 8.1 `div_mag` vs. directional coherence
`div_mag` captures exactly what we need: zero divergence = perfectly laminar flow. An alternative is the **circular variance** of displacement angle in each region (explicitly measures directional agreement). `div_mag` is already computed and strongly correlated with directional coherence, so it is preferred for simplicity.

### 8.2 `n_ahead > 1`
The OT plan is only valid for the current occupancy. For `n_ahead > 1`:
- **Step 0**: OT-guided start/end
- **Steps 1..n_ahead-1**: random physics-aware sampling per step

This is a conservative choice. A more aggressive option is to repeat the step-0 action for all look-ahead steps (one long push), which may be reasonable when the displacement vectors are large. This could be a configurable flag (`repeat_first_step: bool`).

### 8.3 Sinkhorn regularisation (`reg`)
Low `reg` (≈ 0.001) gives a sparse, sharp plan — good for well-separated source and goal but may fail to converge for near-complete overlap. The default `0.002` balances convergence speed and plan sharpness. Consider adding a fallback that increases `reg` if Sinkhorn fails to converge (detectable via `log['err'][-1] > 1e-3`).

### 8.4 Computational cost of OT per MPC step
On a 128×128 grid with ~200 occupied cells, `OTPlannerSparse.solve` runs in roughly 50–200 ms (depending on hardware and `reg`). This adds to per-step compute time. If this is a bottleneck:
- Downsample the grid to 64×64 for OT, then map back to 128×128 world coordinates
- Only recompute OT every K steps (acceptable if the pile doesn't move much between steps)

### 8.5 Conservative union mask vs. orientation-stratified sampling
A simpler alternative to orientation-stratified sampling is to compute one **union-invalid mask** over all orientations and sample starts only from positions clear in every orientation. This is correct but very conservative: at 45° the forbidden zone is approximately a disk of radius `(L+W)/2·√2/2` around every occupied cell, enlarging the forbidden region compared to any single orientation. Orientation-stratified sampling (see §4.7) is both more accurate and expands available start positions at diagonal orientations, at the cost of round-robin assignment of push direction to each candidate.

### 8.6 `d_min` / `d_max` for collision-aware push sampling
`d_min = plate_length / 2` ensures every push has a minimum stroke long enough for the plate to physically engage with material that is a few pixels downstream of the start. Setting `d_max = 2·√2·wkspc_w` covers the full workspace diagonal. GD is free to shorten or extend the push from this initial seed. If end positions are frequently clipped to workspace bounds (visible as flat `ex`/`ey` distributions), reduce `d_max`.

### 8.7 OT-derived start positions and collision checking
OT candidate starts are placed `approach_px` pixels upstream of the coherent region, which is by construction clear of material. Formally validating them against `PlateCollisionChecker` at OT-extraction time is a low-cost sanity check (§4.8) that catches edge cases (e.g., a region that borders the pile at high Sinkhorn regularisation).

---

## 9. Config File Extension

Add to the MPC config YAML (e.g. `simple_mpc/config/config_simple.yaml`):

```yaml
mpc:
  action_sampler: physics_aware   # options: physics_aware | uniform | collision_aware | ot_guided

  collision_sampler:              # used when action_sampler: collision_aware  (also applies to
    n_angles: 8                   # the random-fill portion of ot_guided)
    d_min: 0.02                   # metres — minimum push stroke (≈ plate_length / 2)
    d_max: 0.18                   # metres — maximum push stroke (≈ diagonal of workspace)

  ot_sampler:                     # only used when action_sampler: ot_guided
    reg: 0.002
    ot_fraction: 0.7
    noise_std_m: 0.005
    div_percentile: 30.0
    n_ot_seeds: 8
```

The `collision_sampler` block is read by both `CollisionAwareActionSampler` and `OTGuidedActionSampler` (since the latter composes a `PlateCollisionChecker` with the same parameters).

In `run_simple_mpc`, read these and forward to `make_action_sampler`:
```python
action_sampler_type = mpc_cfg.get('action_sampler', 'physics_aware')
ot_cfg = mpc_cfg.get('ot_sampler', {})
action_sampler = make_action_sampler(
    action_sampler_type,
    plate_length  = plate_length,
    safety_margin = plate_safety,
    grid_size     = int(2 * wkspc_w * 1000),
    wkspc_w       = wkspc_w,
    **ot_cfg,
)
```

---

## 10. Implementation Order

1. **`PlateCollisionChecker` + `CollisionAwareActionSampler`** (in `action_sampler.py`)
   - Precompute `n_angles` plate kernels using `cv2.boxPoints`/`cv2.fillPoly` and per-orientation workspace masks (§4.4–4.5)
   - Implement `PlateCollisionChecker.update` using `cv2.dilate` (§4.6)
   - Implement `CollisionAwareActionSampler.sample` with orientation-stratified vectorised loop (§4.7)
   - Add `'collision_aware'` entry to `make_action_sampler` factory (§5.4)
   - Unit test: create a 128×128 grid with a known pile, call `update`, verify that all sampled starts have zero overlap with the dilated pile for all orientations; visually inspect valid_masks for a few angles
   - Benchmark: confirm `cv2.dilate` ×8 orientations runs in ≤ 10 ms on target hardware

2. **`OTPlannerSparse.extract_push_candidates`** (in `ot_planner.py`)
   - Implement the algorithm from §3.1
   - Unit test: run on the demo distributions in `__main__`, assert `len(candidates) > 0`, plot arrows on the source grid to visually verify

3. **`OTGuidedActionSampler`** (in `action_sampler.py`)
   - Compose `PlateCollisionChecker` for random-fill portion (§5.1–5.3)
   - `update_state` calls both `checker.update` and `planner.solve`
   - Random fill delegated to collision-aware orientation-stratified logic
   - Add `'ot_guided'` entry to `make_action_sampler` factory (§5.4)
   - Unit test: call `update_state` with demo distributions, sample 512 candidates, verify shape `(512, 1, 4)` and that OT-seeded rows have `(sx, sy)` near known coherent regions; verify random-fill rows are collision-free

4. **`ModelAdapter.get_ot_grids`** (in `adapters.py`)
   - Add default no-op to base class
   - Implement for `EulerianAdapter`

5. **`mpc.py` hook** (§6.2)
   - 3-line addition inside the MPC loop before `action_sampler.sample`
   - Also pass `grid_size` and `wkspc_w` to `make_action_sampler`

6. **Config extension** (§9) — optional, for clean integration

7. **End-to-end smoke test**
   - Run one MPC experiment with `action_sampler: collision_aware`, then `ot_guided`
   - Compare reward curve vs. `physics_aware` baseline on the same seed
   - Check per-step timing increase is acceptable

---

## 11. Open Questions

- **Does the OT plan need to be recomputed after each env step, or can it be cached?** In principle yes, but if the pile changes substantially each step (it should), recomputing is necessary.
- **What grid size does the Eulerian adapter use?** If it differs from `int(2 * wkspc_w * 1000)`, the OT grid resolution and the adapter occupancy grid will be inconsistent. Verify at construction time.
- **Can the OT candidates be used to initialise GD *at* the candidate rather than from scratch?** The current design just seeds the initial sample; GD then refines. This is correct and simpler than warm-starting from OT proposals.
- **Score function**: `area / mean_div_mag` rewards large coherent regions. A strong alternative is `area * mean_displacement_magnitude / mean_div_mag` — also rewarding regions where the OT says mass has far to travel. Worth experimenting with once the basic integration works.
