# Piled Configurations and Pile-Aware Action Sampling

**Status:** implemented on branch `VisualForesight`.
**Why it exists:** the datasets collected before this change were, physically, *sparse
single-layer scatterings*, not piles — and both the linear-foresight baseline and
every learned model were being trained and judged on them.

Two measurements forced the change
([`reports/linear_foresight_report.md`](../reports/linear_foresight_report.md) §2.3):

- **The "pile" was one particle deep.** 100% of particles sat in layer 0 — zero
  stacking. 30–50 cubes of 5 mm spread by rejection sampling over a 128 mm tray
  cover ~12% of it, so they almost never touch each other. Granular behaviour
  (load transfer through contact chains, a pushed front bulldozing material
  ahead of it, material flowing around the blade) barely arises.
- **Most pushes barely touched the material.** A typical push had ~14% of the
  pile in its path, and on the lower half of pushes by contact *no model beat
  "predict nothing moved"*. Those transitions cost full simulation time and
  carried almost no learnable signal.

So the previous experiments tested a regime the paper's method was never about.
This document covers the two changes that fix that, and the flags that turn them
on. Both are **off by default** — every existing config, dataset and script
behaves exactly as before.

---

## 1. Piled spawns

### What changed

`SandboxManipulation.shuffle_particles()` gained two optional parameters:

| parameter | meaning |
|---|---|
| `pile_extent` | half-width, in metres, of a **square at the tray centre** that the xy draw is confined to. `None` = the whole tray (previous behaviour). |
| `pile_layers` | number of stacked spawn layers. `None` = derived from the footprint and particle count. |

Rejection sampling then packs that small square instead of the tray, which
forces the particles into several stacked layers. Layers are **dropped, not
interpenetrating** — the existing `update_material_state()` settle collapses them
into a natural heap, exactly as it already did for the >140-particle case that
first introduced layered spawning.

### Why the pile is centred

Deliberately, not incidentally. A pile against a wall behaves differently under
pushing — the wall carries load, and material piles up against it instead of
flowing — so a wall-adjacent pile mixes two dynamics into one dataset. Keeping
the heap central (a 30 mm heap in a 128 mm tray leaves ~45 mm of clear tray on
every side) means wall interaction is *absent* rather than *confounded*, and can
be added later as a deliberate variable.

### Layer count, and why it adapts

The layer count is estimated up front rather than discovered through placement
failures — the retry handler does converge, but each failed attempt burns a full
rejection-sampling sweep and fills the log with alarming "placement failed"
lines. The estimate assumes rejection sampling saturates near 45% area coverage
(planar RSA saturates near 55%; the margin covers the √2 yaw inflation already
built into the collision extents).

The **box height caps** how many layers can exist at all. At the stock 40 mm box
and 5 mm cubes that cap is 3 layers. If the requested footprint would need more,
the footprint is **widened automatically** to the smallest one that fits in the
feasible number of layers, and the change is logged:

```
pile spawn: 30 particles need more than the 3 layer(s) the 0.040 m box allows at
pile_extent=0.0150 m; widening the footprint to 0.0167 m so they fit in 3 layer(s)
```

This is preferred over raising an error: the caller asked for a compact heap, and
the nearest achievable compact heap is more useful than a failure. It does mean
**the effective extent can differ from the requested one** — the log and the
saved config both record what was actually used.

### Interaction with the state library

None needed. `build_state_library()` calls `shuffle_particles()` with no
arguments, which picks up the instance defaults, so a piled library is built by
configuring the spawn and then building the library as usual. Symmetry
augmentation (×8 for a square tray) still applies and is still useful: rotating a
centred heap produces a genuinely different arrangement of particles, not a
relabelling.

**A piled library is not interchangeable with a scattered one.** The
compatibility check in `load_or_build_state_library` covers particle count, size
and shape but **not** the spawn geometry, so pointing a piled run at a directory
holding a scattered library will silently reuse the scattered states. Keep them
in separate `--output-root` trees (as
`configs/collection_pile30.yaml` does) or pass `--rebuild-state-library`.

---

## 2. Pile-aware action sampling

### What changed

`generate_action_samples(pile_aware=True, ...)` replaces the blind start draw
entirely (rather than refining it, which is what `placement_aware` does). For
each sample it works in the frame of its own push direction:

1. Draw a push heading uniformly over the full circle.
2. Project every *active* particle onto the push axis and the lateral axis.
3. Choose the lateral offset centred on a randomly chosen particle, jittered
   across the blade face, so the blade's swath is guaranteed to contain that
   particle. Re-draw (up to 8 times) until the swath holds at least
   `min_swath_particles`.
4. Place the blade start along the push axis **one particle-width behind the
   nearest particle in the swath** — so the sweep begins in contact and no
   travel is spent crossing empty tray.
5. Set the blade yaw normal to the travel direction (the planar-pushing
   convention), and the end point at `push_length` along the heading, capped by
   a ray-box test so the blade stays inside the tray.

Geometry lives in [`Genesis/action_sampling.py`](../Genesis/action_sampling.py)
(`pile_contact_starts`) — pure torch, no `genesis` import, unit-tested in
`tests/test_action_sampling.py` without a GPU, like the rest of that module.

### It owns the push direction, and must

`pile_aware` deliberately **bypasses** the perpendicular / fixed-length
constraint (`constrain_push`) rather than composing with it. That is not an
optimisation: `constrain_push(perpendicular=True)` re-draws the ± sign of the
blade normal at random, which would send half of these carefully aimed pushes
*away* from the pile. Pile-aware sampling therefore applies the perpendicular
convention itself and computes its own end points. The saved config still records
`perpendicular_pushes: true`, because the geometry is perpendicular either way.

It also supersedes `placement_aware`, which chooses a collision-free touchdown
pose — a contradictory goal here, since the whole point is to touch down right
against the pile. If both are passed, `pile_aware` wins.

### What it does not guarantee

- **The pile outgrows the blade's workspace box, and that (not the tray) is why
  a 40 mm push does not fit.** Measured on the first piled run: a settled,
  pushed-around 30-cube pile reaches a particle radius of **p95 34.6 mm, max
  54 mm**, while the blade's allowed box is only **23.5-42.5 mm** half-extent
  (it is the tray shrunk by the blade footprint and safety margin, and it is
  yaw-dependent). Placing the blade one particle-width behind such a pile's near
  face therefore lands **outside** the box **35.8%** of the time — and a start
  outside the box has nowhere to travel, so **3.3% of pushes came out at ~0 mm**:
  no-op transitions costing a full simulation each, which is precisely what
  pile-aware sampling exists to prevent.

  An earlier version of this document blamed the tray size and the sum
  `pile diameter + clearance + push length`. That framing was wrong: the binding
  constraint is that the pile's *outer radius* is comparable to the blade box's
  half-extent, so the problem appears even for short pushes and shortening the
  push alone does not fix it.

  Two changes address it:

  1. `_pile_aware_stops` now **clamps the start into the workspace box**. For a
     tightly packed pile this changes nothing; for a spread one it means the
     blade starts just *inside* the pile rather than just behind it, which still
     sweeps material and — unlike the previous behaviour — can actually move.
  2. `push_length` in the shipped plan is **20 mm, not 40**, so the remaining
     travel comfortably fits. 20 mm is also 4 cube widths, keeping the
     perturbation modest relative to the pile — the regime in which a
     first-order (linear) model is most defensible.

  A fixed `push_length` can still be shortened if even a clamped start has too
  little room; that is reported as a `WARNING`, because a shortened push is no
  longer in its length bin. It only matters for a *single-operator* fit; anything
  treating push length as an input is unaffected.

- **`min_swath_particles` can be unreachable** (e.g. a sparse configuration where
  no lateral alignment catches enough particles). Those draws are counted and
  logged, and `pile_contact_starts` returns an `ok` mask so a caller can drop
  them rather than simulate a push through nothing.

---

## 3. Flags and configs

### Command line — `Genesis/data_collection_clean.py`

| flag | default | effect |
|---|---|---|
| `--pile-extent METRES` | off | confine the spawn to a square of this half-width at the tray centre |
| `--pile-layers N` | derived | force the number of stacked spawn layers |
| `--pile-aware-actions` | off | start every push in contact with the pile and sweep through it |
| `--pile-clearance METRES` | one particle size | blade-to-pile gap at the start |
| `--min-swath-particles N` | 3 | minimum particles the blade swath must contain |

A 30-cube piled run with contact-aware 40 mm pushes:

```bash
python -m Genesis.data_collection_clean \
    --num-particles 30 --particle-sizes 0.005 --particle-shape cube \
    --n-envs 32 --samples-per-env 5 --n-batches 16 --state-library 8 \
    --constant-params \
    --pile-extent 0.015 --pile-aware-actions --push-length 0.04 \
    --min-swath-particles 3 \
    --output-root data/foresight/pile30
```

### Collection plan — `Genesis/run_collection.py`

The same keys are read from a plan's `plan:` block, so a multi-size sweep needs
no command line:

```yaml
plan:
  pile_extent: 0.015
  pile_layers: null              # null = derive from footprint and count
  pile_aware_actions: true
  pile_clearance: null           # null = one particle size
  min_swath_particles: 3
```

Ready-made plan:
[`Genesis/configs/collection_pile30.yaml`](../Genesis/configs/collection_pile30.yaml).

### Simulator config — the `spawn:` block

For callers that construct `SandboxManipulation` directly (including
`env/genesis_env.py`, so MPC resets get piled configurations too):

```yaml
spawn:
  pile_extent: 0.015    # null or absent -> spread over the whole tray
  pile_layers: null     # null -> derived
```

`data_collection_clean.py` writes this block from its flags before constructing
the simulator, so the two routes are equivalent.

### What ends up in the saved dataset

`collect_data_samples` records all of it in each batch's config, so a piled
contact-sampled dataset is distinguishable on disk from a scattered blind one:

```yaml
data_collection:
  perpendicular_pushes: true
  push_length: 0.04
  pile_aware: true
  pile_clearance: null
  min_swath_particles: 3
  spawn_pile_extent: 0.015
  spawn_pile_layers: null
```

Note `spawn_pile_extent` is the **requested** value; if the box-height cap
widened it, the effective value appears only in the run log. Worth tightening if
that path is ever exercised routinely.

---

## 4. Expected cost

Slower than scattered collection, as intended. A compact multi-layer heap is one
large contact island, so most particles are in persistent contact and the solver
does more work per step:

- the per-step cost scales with the densest contact graph in the batch
  (`Genesis/action_sampling.py`'s module docstring covers why batching amplifies
  this), and a heap maximises it;
- `n_envs` optima measured in
  `Genesis/configs/measured/throughput_optimal.yaml` were taken on **scattered**
  piles and will be optimistic here — expect to reduce `n_envs`;
- settling a dropped multi-layer heap takes longer than settling a scattered
  layer, which is an argument for a larger `--state-library` (the settle is paid
  once per library, not once per batch).

Against that, pile-aware sampling removes the sweep distance previously spent
crossing empty tray, and every transition it produces carries signal — so the
cost per *useful* transition may well be lower even where the cost per step is
higher.

---

## 5. Open question this is meant to answer

Whether the linear-foresight baseline's failure
([`reports/linear_foresight_report.md`](../reports/linear_foresight_report.md))
is specific to the sparse single-layer regime. The measured cost of linearity
there was ~31% of the achievable signal (R² 0.576 linear vs 0.836 nonlinear on
per-push displacement). Re-running that decomposition on piled data is the test:
if the linear share rises, the paper's claim holds for actual piles and the
earlier result was a statement about scattered objects; if it falls, linearity is
genuinely insufficient for granular pushing and the conclusion strengthens.

The comparison must use the identical pipeline — `fit_linear_foresight.py`,
`loro_foresight.py`, `variance_decomposition.py` all take a dataset config, so
only the config changes.
