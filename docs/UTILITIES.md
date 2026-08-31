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
- `particles_to_occupancy(particles, bounds, resolution, sigma=0.0, footprint_radius=0.0)` —
  `footprint_radius` (voxels) hard-disk-splats each particle instead of just its
  nearest voxel; used when particle *centers* are too sparse to compare
  against a dense depth-derived occupancy grid (see `footprint_radius_voxels`).
- `footprint_radius_voxels(particle_size_m, global_scale, bounds, resolution, shape_factor=1.0)` —
  converts a particle's physical size into a voxel radius for the above.
  `shape_factor=1.0` (default) is correct for spheres; use `sqrt(2)` for cubes
  (the circumscribed/half-diagonal radius) so touching/rotated cube neighbours'
  disks don't leave gaps a solid silhouette wouldn't have.
- `draw_plate_soft(center, angle, grid_size, plate_length_px, plate_width_px, intensity, sigma=1.5)`
- `genesis_action_to_cam3d(action, scale)` — action endpoints → normalized camera coords
- `genesis_particles_to_cam3d(pos_world, scale)` — same convention, for particle positions
- `build_action_delta(s_cur_xyz, p_start_xyz, p_stop_xyz, sigma_m)`
- `get_grid_axes(ndim)` / `grid_axis_indices(axes)` (coordinate bookkeeping)

These are reusable in both dataset preparation and MPC adapters (including
`simple_mpc.oracle_mpc`, which has no adapter of its own — see INTERFACES.md §3.4).

## 1.1 Video / Recording

`write_video_frame(obs, video_recorder)` (`utils.py`) is the single place that
converts an env obs array's RGB channels to a BGR frame and writes it to a
`cv2.VideoWriter` (or a length-1 list wrapping one — the convention used
throughout so a writer can be swapped without re-threading it through a call
chain). Used by `env/genesis_env.py`, `simple_mpc/genesis_oracle.py`, and
`simple_mpc/oracle_mpc.py`; new video-recording code should call this rather
than re-deriving the obs→BGR conversion.

It pairs with `Genesis/sandbox_manipulation_clean.py::SandboxManipulation.execute_action`'s
`on_phase(phase: str)` callback (fired at `'post_lower'` and `'post_sweep'`,
a no-op if omitted) — the simulator core only exposes *that a phase
boundary happened*; it has no idea a camera or video writer exists. Env
wrappers close the loop by passing a callback that calls `write_video_frame`.
This on_phase/write_video_frame pairing is the reference example of the
"simulator core stays purpose-agnostic" / "shared utilities over duplicated
logic" principles in `ARCHITECTURE.md`'s Design Philosophy — follow the same
shape (generic hook in the low-level primitive, purpose-specific logic one
layer up, shared conversion utility in between) for any similar cross-cutting
need.

## 1.2 Automatic Transition Recording

`SandboxManipulation.push_and_record(p_start, p_stop, angle, is_candidate,
mpc_step, flush_after)` (`Genesis/sandbox_manipulation_clean.py`) replaces
the `execute_action()` + `update_material_state()` pair at every real-step or
candidate-rollout call site and, unless disabled, appends the resulting
before/after particle state + action to an internal `TransitionBuffer`
(`Genesis/transition_buffer.py`) — on by default, opt out via
`dataset.record_transitions: false`.

Data is written incrementally, not just at the end of a run: every **real**
step calls `push_and_record(..., flush_after=True)`, which immediately
writes that step's own transition plus every candidate rollout accumulated
since the previous flush (`flush_transitions()` → `TransitionBuffer.save()`)
to a shared, ever-growing directory (`Genesis/data/mpc_runs/` by default), in
the same on-disk schema `Genesis/data_collection_clean.py`'s dataset files
use (a strict superset — extra keys `success`/`is_candidate`/`mpc_step` are
ignored by existing loaders). Candidate rollouts themselves never trigger a
flush (`flush_after=False`) — they just accumulate until the next real step's
flush. This matters in practice: an oracle MPC episode's real steps can each
take minutes, so flushing per real step (rather than once at episode end)
means data appears on disk throughout a long run instead of only after it
finishes, and a crash loses at most one step's worth of buffered candidates
rather than the whole episode.

Unlike the on_phase/write_video_frame pair, this is *not* a pure external
hook — the simulator core performs the recording itself, since it only needs
state the core already tracks (`_particle_state`) and no downstream-specific
knowledge (see `ARCHITECTURE.md`'s Design Philosophy for why this is an
intentional, narrow exception). The one piece of context recording genuinely
can't supply on its own — which episode a flush belongs to (source, index,
seed) — is set **once, before the episode runs**, via
`env.set_recording_context({...})` (exposed by both
`env/genesis_env.py::GenesisEnv` and
`simple_mpc/genesis_oracle.py::GenesisOracleEnv`; cleared automatically on
the next `reset()`), and every subsequent per-step flush during that episode
uses it automatically. Reward/success aren't included — they're not known
until the episode ends — so they stay in the driver's own
`metrics.json`/`rewards.npy`, joinable by `source` + `episode_idx`.
`save_recorded_transitions(context=...)` (an alias for `flush_transitions`)
remains available for forcing an out-of-band flush, but neither driver
script needs it in normal use. Everything else (which env, which real step,
real-vs-candidate tagging) is tracked by the env wrapper's own step
counter — no other function signature in the MPC loops changed to support
this.

## 2.0 SE(2) Push-Frame Warp

`transforms/functional.py` also owns the canonical push-frame warp of
Suh & Tedrake 2020 Fig. 4 — `push_frame_transform`, `to_push_frame`,
`from_push_frame`, `push_frame_validity_mask`, `blend_push_prediction`,
`invert_affine`, `warp_affine_occ`. Stateless torch, differentiable
(`grid_sample`), tested in `tests/test_push_frame_warp.py`.

Conventions, fixed in one place because three callers need to agree:

- occupancy is `(B, H, W)`, rows then cols;
- push endpoints are pixels in **(col, row)** order — matching `grid_sample`'s
  grid order and what the heuristic push models already receive;
- **grids must be square**, and the helper raises otherwise: normalized
  `[-1, 1]` coordinates are only isotropic when `H == W`, and a rotation in
  anisotropic normalized space silently shears;
- `align_corners=False` throughout;
- the canonical frame puts the push midpoint at the image center and the push
  direction along `+x`.

Two measured properties worth knowing before building on this:

- The warp is **mass-preserving** (±0.00%), so any mass drift observed
  downstream belongs to the physics or the model, never to the resampling.
- The round trip is **not** free on high-frequency fields. Bilinear rotation of
  an occupancy field with features at the pixel scale costs more accuracy than
  one push changes; a σ≈1 px pre-smoothing removes the cost entirely. See
  `reports/linear_foresight_report.md` §1. Score any warp-based image model
  against an identity-operator baseline (`A = I`) as well as persistence, or
  resampling loss gets silently attributed to the model.

## 2.1 Action Sampling and Action-Space Restriction

`Genesis/action_sampling.py` owns push-action geometry that is *pure torch* —
no `import genesis`, so it is unit-testable without a GPU
(`tests/test_action_sampling.py`). Two groups, with different purposes:

- **Batch shaping** — `equalize_travel_distance`, `shared_batch_distance`.
  Reduce what a lockstep batch costs without changing which actions are
  reachable.
- **Action-space restriction** — `blade_normal`, `sampling_box`,
  `constrain_push`, `relative_blade_angle`. Change which actions are drawn at
  all: perpendicular-only pushes and/or a fixed push length. Added for the
  switched-linear visual-foresight baseline
  (`docs/linear_visual_foresight_baseline.md` §7).
- `ray_box_max_travel` is the shared primitive both groups use.

Ownership boundary: this module holds the *geometry*;
`SandboxManipulation.generate_action_samples` holds the *policy* (what order to
apply things in, which flags default to what) and `_constrain_push_geometry` is
the thin bridge. Anything that needs `gs.device`, the plate config, or the
tray dimensions belongs on the class, not here.

Two invariants any change must preserve:

- **Restriction runs after placement-aware sampling**, which replaces both the
  start point and the yaw. Running it before lets the yaw change underneath and
  silently breaks perpendicularity. `_constrain_push_geometry` also recomputes
  the sampling box from the *current* angles, since the box depends on yaw.
- **A constrained push is never shortened to fit the tray — the start moves
  instead.** Shortening would drop a transition out of its requested length
  bin (silently poisoning a single-operator fit) and would distort the
  push-length distribution the restriction is meant to leave alone. Genuinely
  impossible pushes are reported via the `truncated` mask and logged loudly.

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
  `write_video_frame` was added here anyway (§1.1) since it's a tiny,
  single-purpose function with no natural home in `transforms/` (it's I/O, not
  a representation conversion) and three call sites already needed it — a
  scoped `viz.py`/`video.py` module would be the cleaner destination whenever
  that future split happens.

## 5. Guidance for New Utilities

- Keep functions side-effect free.
- Accept and return tensors/dicts with documented shapes.
- Co-locate shape and coordinate assumptions in docstrings.
- Add fixture tests for orientation-sensitive transforms.
