# Scaling pile simulation and data collection to 200 objects

What changed in the simulator to make 200-object piles collectable, what it
costs, and how to run a collection. This is a **final-state** document: it
describes the code as it now stands, not how it got there.

The investigation that produced these changes — probe scripts, raw measurements,
and a full record of the settling diagnosis — lives in
`tests/scaling_investigation/`. Every number quoted here is reproducible from
the scripts in that directory.

Measured on an **RTX 4070 Laptop GPU (8188 MiB)**, Genesis 0.4.5, 5 mm cubes
unless stated otherwise.

> Intended to be folded into a general simulation-setup document once one
> exists; until then this is the reference for the simulator's non-obvious
> settings and why they hold their current values.

---

## 1. Changes to the simulator

### 1.0 Before / after, at a glance

| | before this work | now |
|---|---|---|
| max objects placeable | **~139** (single-layer spawn; 150 and 200 failed at every size and were silently skipped) | **441** at 5 mm (layered spawn) |
| particle mass | 0.8× the recorded density | matches the recorded density |
| plate–particle friction | pinned at 1.0; sampled friction had no effect at the tool | explicit `plate.friction`, sampled friction effective |
| plate x/y velocity | zeroed every step (no momentum into the pile) | retained |
| plate trajectory error | up to **23.9 mm** off the commanded line | **0.3 mm**, cruising at 124 vs 125 mm/s commanded |
| settling | fixed 100 steps, no rest check; a 200-cube pile was recorded mid-motion | convergence-based, cap 2500; ~250 steps after a push |
| settled-pile noise floor | 0.90 mm/s, worst particle 25.1 mm/s / 7.22 rad/s | **0.21 mm/s**, worst 1.0 mm/s / 0.28 rad/s |
| `max_collision_pairs` | flat 150 regardless of pile size | `max(150, n/2)`, sized from measurement, checked at peak |
| envs at n=200 | 4 (with an oversized contact cap) | **32** |
| reset | `shuffle` + full settle every batch | restore from a reusable settled-state library (~6 ms) |
| tool touchdown | blind; ~100 % of touchdowns landed on a particle | C-space sampling; ~19 % |
| solver budget | 10 / 1e-4 | unchanged — measured, and confirmed correct (§1.4) |

Nothing below is a preference; each entry is backed by a measurement in
`tests/scaling_investigation/`.

### 1.1 Correctness

| # | Was | Now | Why it mattered |
|---|---|---|---|
| 1 | Sweep loop called `set_dofs_position` every step to constrain z/roll/pitch/yaw. `RigidEntity.set_dofs_position` overrides the base signature with `zero_velocity=True`, and `zero_all_dofs_velocity` ignores `dofs_idx_local`. | `zero_velocity=False` | The plate's x/y velocity was reset at 250 Hz, so the tool carried no momentum into the pile. |
| 2 | Particles built with no explicit material → `rho=None` → Genesis' `RHO_OBJECT=600`. `_set_particle_density_value` skipped `set_mass` on the first call, then rescaled from 600 as if it were 750. | seed `old_density` from Genesis' default | **Every particle mass was 0.8× its recorded density**, in every dataset collected before this. |
| 3 | Plate friction never set → Genesis default **1.0**, and contacts combine as `max(µa, µb)`. | explicit `plate.friction` (config + material) | Sampled particle friction had *zero effect at the tool interface* — the one interface the action acts through. |
| 4 | `dt` fallback `4e3` (4000 s), `substeps` fallback 1 | `4e-3` / `5` | Latent, but a landmine for any config not setting them. |
| 5 | `set_pos`/`set_quat` per particle → 2N kernel launches and 2N **whole-scene** forward-kinematics passes per reset and per snapshot restore (400 each at n=200). | two batched solver calls + one FK | The restore path is oracle-MPC hot. |
| 6 | `set_n_active` parked every inactive particle at the **same point**. | spread over a grid | Heaped 150 cubes into one permanent contact cluster. |
| 7 | Single-layer particle spawn only. | layered fallback | 150 and 200 particles could not be placed **at any size**; `data_collection_clean.py` silently skipped those batches. |
| 8 | `_particle_state_` allocated, never used. | removed | — |

Fixes 1–3 change the physics of every transition, so **data collected before and
after them is not comparable**.

### 1.2 The pusher plate is now modelled as a gantry axis

> The tool now has its own reference, **`plate_model.md`** — actuator model, the three phases of a push, and the per-action reaction-load report. This section covers only what changed and why.

The plate is a 2.4 g box — the *lightest* dynamic object in the scene, not a
heavy one. What previously kept it on course was that z/roll/pitch/yaw were
hard-set every step and its x/y velocity was zeroed every step.

Measured on one 90 mm crossing sweep, 50 cubes:

| | tracking error vs commanded path | reaction displacement (loaded − free) | cruise speed |
|---|---|---|---|
| before | mean 9.24 mm, **max 23.90 mm** | max 0.12 mm (0.46 mm at ρ=5000) | 102 mm/s |
| **now** | mean 0.32 mm, max 0.92 mm | **max 0.04 mm** | **124.0 mm/s vs 125 commanded** |

Granular reaction was never displacing the tool much. The dominant error was the
control law, which put the tool up to **23.9 mm** — five particle diameters —
off its commanded straight line, because it was handed the *endpoint* as a PD
position target and therefore ran at a speed proportional to distance remaining.

The model now, all config-driven under `plate:` in `configs/basic.yaml`:

- **`set_dofs_armature(moving_mass)`** on x/y/z — the drivetrain's reflected
  inertia, added to the mass-matrix diagonal the constraint solver already uses.
  This is the correct knob rather than a denser plate, which would also change
  the tool's weight and its contact response.
- **gains from mass and bandwidth**: `kp = m·ω²`, `kv = 2ζmω` at ζ=1.
- **`set_dofs_force_range(±max_force)`** — previously unbounded; a real stepper
  loses steps rather than applying unlimited force to a jam.
- **a trapezoidal position + velocity reference** replacing the endpoint target.

Side effect: a sweep now takes **208 steps instead of 307**, because the step
count comes from the trapezoid's real duration rather than a `1.7×` fudge that
was compensating for the speed error, and the per-step `.nonzero()`/`.item()`
GPU syncs are gone.

### 1.3 Settling is convergence-based, and the solver is no longer starved

`update_material_state` previously ran a fixed 100 steps and read the state with
no check that anything had stopped. `settle_steps` is now a **cap** with a
velocity-convergence exit and a loud warning if the cap is hit.

Three things were needed to make that criterion work:

1. **A quantile, not a max.** Testing `max` over every particle in every env
   makes the criterion harder the more envs are batched — 32 envs × 200
   particles is 6400 chances for one straggler to block the batch, so the settle
   always ran to its cap. `settle_rest_quantile` (0.995) tolerates ~1 straggler
   per 200-cube env.
2. **An angular threshold derived from the linear one.** A bare rad/s number is
   not comparable to a m/s number: 0.1 rad/s on a 5 mm cube is a corner speed of
   0.35 mm/s, three times *stricter* than the 1 mm/s linear threshold, so it
   silently became the binding criterion. It is now derived through the
   particle's half body diagonal (0.231 rad/s at 5 mm).
3. **Not resetting the solver every step.** Holding the plate with a per-step
   `set_dofs_position` called `collider.reset()` and `constraint_solver.reset()`
   on every settle step, discarding the constraint solver's warm start. The
   plate is lifted clear during a settle and its PD holds it to a **5.3 µm sag**
   (0.0235 N against `kp` = 4441 N/m), so the teleport bought nothing. The
   control target is now set once; `ctrl_pos` persists and the actuator reads it
   every substep.

A/B on an identical spawn (`tests/scaling_investigation/probe_settle_ab.py`):

| variant | worst particle | q99.5 floor | cost/step |
|---|---|---|---|
| per-step teleport, 10 solver iters | 25.1 mm/s, 7.22 rad/s | 0.90 mm/s | 1.00× |
| **PD hold, 10 iters (shipped)** | **1.0 mm/s, 0.28 rad/s** | **0.21 mm/s** | **0.75×** |
| PD hold + 50 iters | 0.2 mm/s, 0.07 rad/s | 0.11 mm/s | 0.81× |

Essentially all of the improvement is the plate hold, not the solver budget:
at 10 iterations the worst particle already drops 25× and the floor sits
comfortably inside the 1 mm/s rest threshold. The solver budget is left at the
project's original 10/1e-4 — see §1.4 for why raising it is a bad trade.

**Post-push settling is ~250 steps (1 s of simulated time).** The expensive
1460-step settle applies only to a fresh two-layer respawn, which happens at
reset, not per transition. See `tests/scaling_investigation/settling_investigation.md`.

### 1.4 Push cost, and three solver knobs that must not be touched

A **push** step is the unit that dominates collection cost, and it behaves
nothing like a settle step. Measured at 200 cubes with the blade broadside
(`tests/scaling_investigation/probe_push_cost.py`):

| `rigid_options` variant | push ms/step, 1 env | 4 envs | verdict |
|---|---|---|---|
| **as shipped** (10 iters, islands on, box-box on) | **770** | **1358** | — |
| 50 solver iterations | 1234 (1.6×) | 2447 (1.8×) | **slower; reverted** |
| `use_contact_island: False` | **844 734** | — | **~1100× slower** |
| `box_box_detection: False` | crash | crash | **memory corruption** |

Three things follow, and each is a trap worth stating explicitly.

**Blade orientation dominates push cost.** The plate is 40 mm along its local x
and 2 mm along local y, and its yaw is sampled *independently* of the push
direction. A blade travelling along its own long axis slices edge-on with a 2 mm
swath and barely loads; broadside it sweeps the full 40 mm. The two differ by
2–9×. Any push measurement must state its yaw, or it is not comparable.

**Contact islands are load-bearing.** Turning them off makes a push step take 845
*seconds* rather than 0.77 s. They are doing enormous work; this is not a knob to
experiment with.

**`box_box_detection` silently owns the contact budget.** It sets
`n_contacts_per_pair` to 16 rather than 5, so turning it off cuts the
contact-*point* cap from `max_collision_pairs × 16` to `× 5` without touching
`max_collision_pairs` itself. A 200-cube pile uses ~826 points during a push:
fits 2400, overflows 750. That overflow does **not** raise — it was observed as a
CUDA illegal memory access, because the sweep loop's per-step
`set_dofs_position` clears Genesis' error bit before it can be checked. The
budget is therefore checked directly at the end of every sweep, where usage
peaks (`_check_contact_budget`).

### 1.5 `max_collision_pairs`

Measured requirement is `≈0.26 × n_particles` (13 at n=50, 52 at n=200), so
Genesis' flat 150 was always sufficient for these pile sizes. The default is now
`max(150, n_particles // 2)`.

This is not cosmetic: the constraint Jacobian is
`O(max_collision_pairs × contacts_per_pair × n_dofs × n_envs)` while raw step
time is *independent* of the cap, so an oversized value converts directly into
lost parallelism — at n=200 a cap of 200 tops out at 16 envs where 150 fits 32.

Overflow is silent and cannot be detected by Genesis' own mechanism here: the
broadphase sets an error bit that `Simulator.step` checks periodically, but
`set_dofs_position` clears `_errno` as a side effect. `contact_budget_usage()` /
`_check_contact_budget()` therefore read the counters directly, comparing
broad-phase pairs against `mcp × 8` and contact *points* against
`mcp × n_contacts_per_pair` — two limits that differ by more than an order of
magnitude and must each be checked against their own cap.

---

## 2. Capacity: what fits in the tray

`shuffle_particles` places particles in layers, treating a free-yaw cube as an
axis-aligned square of side `size·√2`. The tray's usable interior height is
**20 mm**, not the configured 40 mm — the walls span z ∈ [−0.01, +0.03] while the
floor surface is at z = +0.01.

| cube size | per layer | max layers | max N | layers for 50/70/100/150/200 |
|---|---|---|---|---|
| 5.00 mm | 147 | 3 | **441** | 1/1/1/2/2 |
| 6.75 mm | 80 | 2 | 160 | 1/1/2/2/✗ |
| 8.50 mm | 51 | 1 | 51 | 1/✗/✗/✗/✗ |
| 10.25 mm | 35 | 1 | 35 | ✗ |
| 12.00 mm | 25 | 1 | 25 | ✗ |

**5 mm is the only size in the historical sweep that reaches 200 objects in the
current 128 mm tray.** Chickpea scale (~8.5 mm) needs a **179 mm** tray at two
layers or **253 mm** at one, plus walls taller than 40 mm.

Combinations that cannot be placed raise with a descriptive error and are
skipped by the collection driver rather than being special-cased.

---

## 3. Cost

### 3.0 How many environments to actually use

Environments in a batch step in lockstep, so a batch costs what its *worst*
member costs. Two couplings follow, and both were measured at 100 objects
(`tests/scaling_investigation/probe_action_coupling.py`, 1 → 8 envs):

| | 1 env | 8 envs | |
|---|---|---|---|
| identical action in every env | 13.47 s | **15.21 s** | 1.13× for 8× the work |
| independently sampled actions | 13.34 s | **40.17 s** | 3.01× |

Batching is close to free when the actions are homogeneous. The gap decomposes
into **1.54× from step count** — `sweep_steps` follows the *longest* travel in
the batch, so every env runs for the longest one's duration — and **1.72× from
contact complexity**, since each step costs what the densest contact graph in
the batch costs.

The step-count half is removable, and is removed: collection shares one travel
distance across the batch (§4.3). Measured effect on the throughput grid,
seconds per batch:

| n_obj | envs | independent distance | shared distance | speedup |
|---|---|---|---|---|
| 50 | 4 | 18.8 | 6.5 | 2.9× |
| 100 | 4 | 92.1 | 18.3 | **5.0×** |
| 150 | 4 | 803.2 | 66.5 | **12.1×** |
| 200 | 4 | 2706.6 | 242.7 | **11.2×** |

**Throughput-optimal env counts** (`benchmark_throughput.py`, shared distance,
2 transitions per cell):

| n_objects | best n_envs | transitions/s | s per transition | vs 1 env |
|---|---|---|---|---|
| 20 | 128 | 14.495 | 0.07 | 33.8× |
| 50 | 128 | 3.920 | 0.26 | 21.2× |
| 70 | 64 | 1.595 | 0.63 | 11.9× |
| 100 | 64 | 0.787 | 1.27 | 8.7× |
| 150 | 2 | 0.063 | 15.83 | 1.3× |
| 200 | 1 | 0.026 | 37.86 | 1.0× |

Before the shared-distance fix the same benchmark recommended **2 envs at n=70
and n=100**; it now recommends 64, worth 11.9× and 8.7× over a single
environment. That earlier table was measuring the sampler, not the simulator.

**Batching stops paying somewhere between 100 and 150 objects.** Beyond that the
residual contact-complexity coupling means per-batch time grows at least as fast
as the env count, so a single environment is as good as any. That is a real
property, not an artifact: it is what remains after the removable half is gone.

Two caveats on this table. It is 2 transitions per cell, so it is indicative
rather than tight — raise `transitions_per_cell` before treating small
differences as real. And the sweep stops climbing after two non-improving steps,
so at n=150/200 the larger env counts were never tried; given throughput is flat
to falling there, that is unlikely to hide a better operating point.

The full grid is at `outputs/scaling_benchmark/throughput_full.json`, and the
recommendation is written as `throughput_optimal.yaml` in the exact shape a
collection plan's `plan.n_envs` expects.

### 3.1 Parallel environments (memory ceiling)

Measured OOM boundary with the shipped `max_collision_pairs` default:

| n_particles | max n_envs | VRAM there |
|---|---|---|
| 50 | **128** | 7.22 GiB |
| 70 | **64** | 4.83 GiB |
| 100 | **64** | 7.28 GiB |
| 150 | **32** | 5.26 GiB |
| 200 | **32** | 6.98 GiB |

These are ceilings, not recommendations — n=50 at 128 envs used 7.22 of 7.62
GiB, so any other GPU consumer will push it over.

### 3.2 Reset

`shuffle_particles()` runs **zero simulation steps**; all of a reset's cost is
the settle that follows. Restoring a pre-settled state needs no settle at all:

| n_particles | shuffle + settle | `set_particle_state` restore | speedup |
|---|---|---|---|
| 50 | 5.08 s | 0.094 s | 54× |
| 200 | 37 s | 0.006 s | **6184×** |

### 3.3 Storage

`states`/`states_` are `float32 [N, n_particles, 7]` (position + quaternion); no
images. Per transition = `2·n·7·4 + 28` bytes.

| n_particles | per transition | 100 k | 1 M |
|---|---|---|---|
| 50 | 2.76 KiB | 0.26 GiB | 2.7 GiB |
| 200 | **10.96 KiB** | 1.05 GiB | 10.5 GiB |

Storage is not a constraint at any scale considered here.

### 3.4 Rebuilds

`n_particles`, `particle_size`, `shape`, `box.vol`, `n_envs`, `dt`, `substeps`
and every `rigid_options` field require a full rebuild — particles are created in
`__init__`, before `scene.build()`. Only particle friction, particle density and
box friction change in place via `set_material_properties()`.

Build takes 32–117 s and does **not** amortize across configurations: every
distinct `(n_particles, n_envs, max_collision_pairs)` triple pays a full
recompile, because `performance_mode: True` disables Genesis' ndarray/fastcache
path and specializes kernels on the scene's array shapes. `set_n_active(n)`
avoids a rebuild when only the particle count changes.

---

## 4. Collection features

Both are opt-in; omitting the flags reproduces the previous behaviour exactly.

### 4.1 Settled-state library — `Genesis/state_library.py`

`--state-library N` settles N piles once per build, expands each by the
container's symmetry group, saves `settled_states.pt` beside the data, and resets
by restoring instead of re-settling.

The symmetry expansion is what makes a dozen settles worth it: a settled
arrangement rotated or mirrored into another orientation of the tray is still a
valid settled arrangement, and is a different configuration to sample from. A
square tray admits the full dihedral group D4 — **8 variants per settle**; a
rectangular one admits 4. Each `shuffle_particles()` also randomizes every
parallel env independently, so the bank is `N × n_envs × 8` states for the cost
of `N` settles.

Mirroring is applied to orientations, not just positions. A reflection `M` is
improper, but `M R M` is a proper rotation — in quaternion terms
`(w,x,y,z) → (w,−x,y,−z)`. This is legitimate because cubes and spheres are
achiral. Verified in `tests/test_state_library.py` against explicit rotation
matrices, including `det = +1`.

**All environments in a batch share one initial state**, drawn from the library
*without replacement* — a shuffled permutation, reshuffled only once every state
has been used, with the first wrap reported so a run long enough to exhaust the
library says so. A batch of identical piles is cheaper to simulate than a batch
of distinct ones, and within a batch the sampled action parameters already vary
the dynamics substantially, so the variance given up is small next to the
throughput gained. Diversity is preserved *across* batches instead of within
them. `StateLibrary.apply_per_env` gives per-env states where that is genuinely
needed, but it is not what collection uses.

`--state-library-damping` adds temporary viscous damping *during those settles
only*. It is a numerical convergence aid, not a physical model — real air drag on
a 5 mm cube at 50 mm/s is ~3×10⁻⁵ of its weight. It is deliberately not applied
to the post-push settle, where cutting the relaxation short would bias recorded
`s'` toward smaller displacements.

### 4.2 Placement-aware action sampling — `Genesis/placement_sampling.py`

`--placement-aware` draws the tool's touchdown pose from its free configuration
space, so the plate does not descend *into* a particle (which the solver resolves
by ejecting it — an artifact recorded as though it were a push).

Occupancy grid → rotated-rectangle dilation per yaw bin (a Minkowski sum, turning
"does the tool overlap an obstacle" into "is the tool's centre in a forbidden
region") → free set, sampled directly. A Euclidean distance transform supplies a
clearance value for biasing draws toward roomier placements.

Measured at n=200: touchdowns overlapping a particle drop from **100 % to 19 %**.
It is a refinement, not a guarantee — as the tray fills the free set shrinks and
eventually empties, so it falls back to the blind draw per sample.

---

### 4.3 Shared travel distance within a batch

Collection gives every env in a batch the same push *length* for a given sample,
while each keeps its own start point, direction and blade yaw
(`Genesis/action_sampling.py`). `--independent-travel-distance` restores
per-env lengths.

This is a batching artefact fix, not a modelling choice. `sweep_steps` is derived
from the longest travel in the batch, so one long push makes every env run for
its duration — worth 1.54× of a 2.64× batching penalty at 8 envs, and up to 12×
of end-to-end batch time at 150 objects (§3.0). What is given up is the
*within-batch* spread of one of five action dimensions; the distance is drawn
from one env's own sample, so its distribution across batches is unchanged. A
push that cannot reach the shared distance without leaving its sampling box is
truncated at the boundary and reported.

## 5. Running a collection

```
python -m Genesis.run_collection --plan configs/collection_dry_run.yaml
```

`Genesis/run_collection.py` runs one subprocess per pile size (a rebuild is
required anyway), so an OOM or an infeasible placement at one size cannot abort
the rest. It performs preflight checks (placement feasibility, free VRAM against
an estimate fitted to the measured ceilings) and postflight validation on the
files actually written: shapes and dtype, no NaNs, **`s'` actually differs from
`s`** — a run that "succeeds" while nothing moves is the silent failure worth
guarding against — implausible-displacement detection, goal-reach rate, and
state-library size.

The plan lives in `Genesis/configs/collection_dry_run.yaml`: pile sizes, per-size
env counts, constant physical parameters, samples per env, batch count, library
size, and the placement-aware flag. Scaling up is a two-line edit.

Everything must be run as a module from the **repo root**
(`python -m Genesis.…`): `sandbox_manipulation_clean` uses relative imports that
only resolve inside the package.

### Viewing a run

`--viewer {bird,observer,leveled}` opens a live window and forces `n_envs=1`
(the viewer renders every env). It needs a display on the machine itself.
`leveled` is the most informative for a 200-object pile, since it shows whether
the plate shears the pile or ploughs through its middle.

---

## 6. Open decisions

| # | Decision | Why it matters |
|---|---|---|
| 1 | **Particle size vs. tray size** | 5 mm is the only size reaching 200 objects in the current tray (§2). Chickpea scale needs a substantially larger tray. |
| 2 | **Re-collection** | Fixes 1–3 in §1.1 change the physics of every existing transition. |
| 3 | **Plate operating height** | The plate rides at half-particle height, contacting cubes above their centre of mass; at 200 particles the pile is two layers, so the tool ploughs through it at roughly half depth. `_operation_height` is also derived from the *nominal* particle size, not the sampled one. |
| 4 | **`constraint_timeconst`** | 0.01 s against a 4 ms step leaves compliant contacts, the likely source of the residual numerical creep described in the settling investigation. Untested. |
| 5 | **Is n=200 worth collecting, and in what tray?** | A transition costs 896 s at 200 objects against 10.5 s at 100. Diagnosed (§8.7): cost goes as (largest contact island)^2.64, and island size is set by **packing fraction**, not particle count. A 1.5x tray makes the same 200 cubes 17.5x cheaper — ~14 GPU-hours instead of ~10 GPU-days — but it is a less dense pile, i.e. a different task. This is the same decision as row 1, now with a price attached. |
| 6 | **Placement-aware sampling is inactive at n=200** | The free set is empty there (§8.4), so every touchdown falls back to blind. Either accept blind touchdowns at the top of the range, or trade the wall margin / clearance for availability. |

Resolved: the `max(µ_particle, µ_box)` friction-combining rule is acceptable
provided the tool's own friction is recorded, which it now is (§1.1 fix 3).

---

## 7. Where the investigation lives

`tests/scaling_investigation/` — probe and benchmark scripts, raw measurement
outputs, and `settling_investigation.md`, the full diagnosis of why piles
appeared not to settle. See that directory's `README.md` for what each script
measures and how to re-run it.


---

## 8. End-to-end verification of a realistic run

Everything above verifies a mechanism. This section verifies the **output**: a
short but otherwise real collection was run at each planned operating point
(`tests/scaling_investigation/probe_collection_health.py`, 5 pushes per env,
seeded from the state library), and several runs were rendered to video
(`record_simulation_video.py`). Raw numbers in
`tests/scaling_investigation/results/collection_health.json`.

### 8.1 What came out clean

| check | result across 20–200 objects |
|---|---|
| goal reached | **100 %** of pushes, worst final tracking error 0.01 mm — `reached_goal` is not silently discarding samples |
| escaped particles | **0** everywhere — nothing is squeezed through a wall |
| contact budget | 4 % of cap at n=20 rising to **45 % at n=200** — `max_collision_pairs` is correctly sized, with headroom |
| per-env settling | **0 envs** above the rest thresholds at the moment the pooled criterion trips, at every object count |
| `s' == s` | 8 % at n=20 (a sparse tray, so some pushes genuinely miss every cube), 0–2 % from n=50 up |

The per-env result matters because the rest criterion is a quantile pooled over
*all* envs, which in principle permits one env to be entirely unsettled while
the batch passes. Measured, it never happened.

### 8.2 Residual motion at `s'` is jitter, not travel — with one exception

The rest test bounds velocity, so the honest question is whether a particle that
is still moving when `s'` is recorded actually *goes* anywhere. Measured by
holding the pile for a further 0.2 s after the criterion trips and taking net
displacement:

| | worst residual speed | net drift over the next 0.2 s |
|---|---|---|
| n=20 x128 | 2.6 mm/s | 0.12 mm |
| n=50 x128 | **73 mm/s** | 0.14 mm |
| n=70 x64 | 7.2 mm/s | **14.4 mm** |
| n=100 x64 | 27 mm/s | 1.2 mm |
| n=200 x1 | 6.0 mm/s | 0.51 mm |

Typical drift is 0.04–0.35 mm against pushes that displace particles 20–75 mm,
i.e. **well under 1 % of the signal**. Note the anti-correlation: the *fastest*
residual particles barely move (they vibrate in place), while the one genuine
late movement — a 14 mm local avalanche in one env of 64, once in 15 batches —
came from a particle travelling only 7 mm/s, a cube at the top of its tipping
arc where speed passes through a minimum.

A max-velocity guard on top of the quantile was implemented and then **removed**
on this evidence: it would have paid for extra settling on every harmless
vibrating particle and still missed the only real event. Catching metastable
collapse needs a persistence or displacement test, not a faster-threshold one.
Left as a known, quantified tail rather than fixed speculatively.

### 8.3 Two real defects found and fixed

1. **`safety_margin` was dead config.** `Genesis/configs/basic.yaml` declared
   `safety_margin: 0.005` while the code hardcoded `0.02` and never read the
   key, so tuning it did nothing. The code now reads it, and the config states
   the value that was actually in force (`0.02`) so behaviour is unchanged.

2. **Placement-aware sampling did not keep the tool clear of the walls.** It
   excluded centres whose footprint would poke through a wall, but not the
   additional `safety_margin` the blind sampler applies — so it drew touchdowns
   the blind sampler never would. Measured at n=20: **26 % of placement-aware
   touchdowns in the outer third of the tray against 0 % blind**, mean radius
   0.55 vs 0.33. `free_placements` now takes a `wall_margin`, and the two
   samplers agree: 0 % vs 0 %, so the only remaining difference between them is
   particle avoidance, which is the intended one.

### 8.4 Known costs, unfixed

**Travel truncation.** Sharing one push length across a batch truncates pushes
that would leave the tray: 37 % at n=20 (where the shared distance can be as
long as 76 mm), 10–26 % elsewhere. Truncated pushes are systematically shorter
near walls. The alternative is giving up the throughput win in §4.3.

**Placement-aware sampling buys nothing at n=200.** The free set was empty for
**0/5** samples at 200 objects, so every touchdown fell back to the blind draw
— the documented degradation path (§4.2), but it means the feature is inactive
at exactly the object count this work targets. It was fully available (100 %) at
20–150.

**Per-transition cost, which is the real limit on n=200:**

| n_objects | n_envs | seconds per transition |
|---|---|---|
| 100 | 64 | 10.5 |
| 150 | 2 | 106 |
| 200 | 1 | **896** |

That is an 85x jump from 100 to 200 objects for 2x the particles and only 2.4x
the contact points — far steeper than contact count explains. **Diagnosed in
§8.7:** the cost is set by contact-island size, which is set by packing
fraction. At this cost a 1000-transition dataset at 200 objects is roughly ten
GPU-days, so the binding constraint at n=200 is time, not memory.

### 8.5 Visual check

`record_simulation_video.py` renders the spawn, settle, descent and pushes from
three angles. Reviewed at 200 objects: the layered spawn collapses plausibly
with no interpenetration at t=0, the pile rests in one to two layers, the blade
bulldozes material into a berm ahead of itself and leaves a clean swept channel
behind with spill to the sides, and nothing leaves the tray.

Note that a truly level camera sees nothing — the tray walls are 40 mm and the
pile is one or two 5 mm layers, so the near rim occludes the interior. The
`leveled` preset uses the lowest angle that clears it.

### 8.6 Reproducibility

`data_collection_clean.py` had no seed control at all: `np.random.default_rng()`
with no argument, and the torch draws (spawn poses, orientations, every action)
unseeded. A run could not be repeated, which also meant a run that produced
something odd could not be replayed to look at it. Both generators now take
`--seed`, it is recorded in each batch's saved config, and `run_collection.py`
forwards it offset per object count so the runs do not all draw identical
actions.

Each saved batch now also records `escaped_particles` and peak `contact_budget`
usage, so a finished dataset carries the evidence that it is trustworthy instead
of requiring the run to be repeated to find out.
### 8.7 Why 200 objects costs 85x more than 100: contact-island size

Measured by `tests/scaling_investigation/probe_contact_islands.py`, which reads
the solver's own island decomposition
(`rigid_solver.constraint_solver.contact_island`) while a blade is driven
broadside through the pile.

**Mechanism.** Genesis defaults to the **Newton** constraint solver. With
`use_contact_island` it partitions constraints into independent islands and, per
island, builds *and factorizes* a **dense** Hessian over that island's dofs
(`constraint/solver_island.py::_func_nt_hessian_direct`), where
`island_dofs = 6 x entities in the island`. Assembly is quadratic in island
size, factorization cubic.

While settling, every cube rests on the floor touching nobody, so the graph is
~n singleton islands and the dense term is nothing — **settle cost stays linear
in n** (33 → 63 → 110 → 161 ms/step for n = 50 → 200). During a push the blade
couples particles that touch other particles and the contact graph percolates.

**Measured, one env, identical broadside push:**

| n | tray | packing | settle ms/step | push ms/step | largest island | contact points |
|---|---|---|---|---|---|---|
| 50 | 1.0x | — | 33.2 | 109 | 11 | 229 |
| 100 | 1.0x | — | 63.2 | 437 | 24 | 452 |
| 150 | 1.0x | — | 110.1 | 2356 | 41 | 718 |
| 200 | 1.0x | 0.305 | 160.9 | 7835 | 56 | 1067 |
| 200 | 1.25x | 0.195 | 125.6 | 938 | 29 | 879 |
| 200 | 1.5x | 0.136 | 125.5 | 483 | 21 | 854 |

Across all of it, **cost ~ (largest island)^2.64** — between the quadratic
assembly and cubic factorization terms, which is what a mix of the two looks
like. Percolation turns out to be *partial*: the largest island reaches ~0.28n,
never the whole pile, so the growth is smooth rather than a cliff.

**Decisive test 1: equal island size, different particle count.**

| | largest island | push ms/step |
|---|---|---|
| 100 objects, standard tray | 24 | 437 |
| **200 objects, 1.5x tray** | 21 | **483** |

Twice the objects, the same cost. Contact points differ by 1.9x between those
two rows and cost does not follow them.

**Decisive test 2: is it the objects touching, or confinement against a wall?**
The tray-widening above changes two things at once — it shrinks the islands
*and* it takes the walls away — so on its own it cannot tell those apart. Two
controls separate them, both restoring the *same recorded pile* so the contact
graph is identical and only the walls move:

| n=200, identical library pile | particles touching a wall | largest island | push ms/step |
|---|---|---|---|
| standard tray | many | 57 | 8440 |
| fence moved out 1.5x | **0** | 52 | 7361 |
| fence moved out 2.0x | **0** | 58 | **8342** |
| pile + particles + blade scaled 0.6x in the standard tray | **0** | 60 | 10461 |

Removing wall contact entirely changes nothing: 8342 vs 8440 ms is 1.2 %, inside
run-to-run noise. The same control at n=50 (wall contacts 2 → 0) gives 108.2 vs
109.7 ms. So **it is the number of objects mutually in contact, and confinement
plays no part.** This also fixes the reading of the tray-widening result: its
17.5x came from the pile *spreading out*, not from the walls receding.

That is what should be expected from the implementation — static bodies do not
propagate contact islands, so walls and floor can never merge islands. They can
only matter by holding particles against each other, and at this packing the
pile is self-supporting.

**The collapse.** Across every configuration — particle count varying 4x, tray
size 2x, absolute scale 1.7x, wall contact from many to zero — `ms / island^2.64`
stays within 0.099–0.217 over a **96x range in cost**:

| n | box | fence | pile | largest island | push ms | ms/isl^2.64 |
|---|---|---|---|---|---|---|
| 50 | 1.0 | 1.0 | 1.0 | 11 | 109 | 0.195 |
| 200 | 1.5 | 1.0 | 1.0 | 21 | 483 | 0.156 |
| 100 | 1.0 | 1.0 | 1.0 | 24 | 437 | 0.099 |
| 200 | 1.25 | 1.0 | 1.0 | 29 | 938 | 0.129 |
| 150 | 1.0 | 1.0 | 1.0 | 41 | 2356 | 0.130 |
| 200 | 1.0 | 1.5 | 1.0 | 52 | 7361 | 0.217 |
| 200 | 1.0 | 1.0 | 1.0 | 57 | 8440 | 0.195 |
| 200 | 1.0 | 2.0 | 1.0 | 58 | 8342 | 0.184 |
| 200 | 1.0 | 1.0 | 0.6 | 60 | 10461 | 0.211 |

Cost depends on one variable: how many objects are mutually in contact.

**What this means practically.** The lever is **packing fraction, not particle
count**. The same 200 cubes in a 1.5x tray run **17.5x faster**, which turns
~896 s per transition into roughly 50 s — a 1000-transition dataset goes from
~10 GPU-days to ~14 hours. That is not free: it is a different task, with the
pile less dense and a push engaging fewer neighbours. It does give open decision
1 (particle size vs tray size, §6) a quantitative cost argument it did not have.

**Neither solver escape is available.**

- **CG** would never form a dense Hessian and is the obvious fix, but it is
  **broken in Genesis 0.4.5**: the kernel references
  `RigidSolver.func_solve_mass_batch`, which does not exist, so any scene using
  it fails at compile time. `rigid_options.constraint_solver` is now exposed
  anyway so it can be re-tested against a newer Genesis without a code change.
- **`use_contact_island: False`** is worse, not better. It replaces the
  per-island blocks with one global `n_dofs^2` Hessian: at n=200 that is
  1206^2 ≈ 1.45M entries against ~120k summed over the islands, i.e. 12x more
  dense work — consistent with the ~1100x slowdown `probe_push_cost.py`
  measured for that variant. §1.4's "do not turn this off" stands.

Reducing solver `iterations` would scale the dense term down proportionally, but
§1.4 already established 10 as the working value and this was not re-tested.

### 8.8 Solver choice: Newton is kept, and why the alternatives lose

§8.7 traced the cost cliff to Newton's dense per-island Hessian. The obvious
response is to change solver, so all three available configurations were
measured against the baseline for *physics*, not just speed
(`tests/scaling_investigation/probe_solver_equivalence.py`). Speed without
equivalence is worthless here: a dataset collected under one solver is not
comparable to one collected under another, switching by object count entangles
particle count with solver, and switching mid-transition would have `s'`
produced by different physics than the dynamics that created it.

**Method, because a naive comparison cannot work.** A granular pile is chaotic:
two solvers on an identical state under an identical action diverge regardless
of whether both are correct, and so does one solver run twice against itself
(measured — Genesis is not bit-deterministic). So the probe measures a **noise
floor** from configurations known to be physically identical (a bit-identical
rerun, and a 1 um action perturbation) and counts only differences exceeding it;
it leads with **penetration**, a property of the state rather than the
trajectory, which stays meaningful after two runs have diverged; it uses fixed
step counts so configurations are compared at matched simulation time rather
than at whichever declares convergence first; and it compares distributions over
replicates, because a single run per cell is one draw — a 3.4 mm difference was
observed once and failed to reproduce.

**Results** (n=50, 12 replicates unless noted):

| | particle-particle penetration | stability | bias vs Newton |
|---|---|---|---|
| **Newton + islands** | 0.395 mm, deterministic to 3 dp | 0/12 gross events | — |
| CG (islands off) | 0.175–0.443 mm (**better**) | 1/12 | ~3 % under-transport at `offset` |
| CG, 30 iterations | 0.19–0.43 mm | 2/12 | bulk matches Newton |
| hibernation | 0.576 vs 1.086 mm (−47 %) | deterministic | **−4 % transport, exactly repeatable** |

Two traps this ran into, both worth remembering:

* The gross events are **particle-vs-static** contacts (floor/wall), not cubes
  passing through each other — a 97 mm overlap between 5 mm cubes is
  geometrically impossible, which is the tell. An earlier version of the probe
  classified everything that was not the plate as particle-particle and
  reported exactly that impossibility. Classify contacts by both endpoints.
* `cg` differs from `newton` in **two** ways (solver *and* islands off), so its
  bias could have come from either. Controlled with a `newton_islands_off` run:
  its COM (4.51, 4.55 mm) and displaced mass (226.1, 228.2 mm) fall inside plain
  Newton's own spread (4.45–4.55, 223.3–228.1), so the island setting does not
  bias the physics and the CG bias belongs to CG.

**Decision: keep Newton with contact islands.** CG's push advantage cannot be
realised — upstream its per-island linear solve is Newton-only, so CG is always
islands-off in practice, which is 5.9x *worse* on the settle and is exactly the
contrast measured above. Hibernation is disqualified on its own numbers here,
but see the caveat below. `rigid_options.constraint_solver` is now set
explicitly in `basic.yaml` rather than left implicit, because the neighbouring
`use_contact_island` default flipped upstream between releases.

**Caveat on the hibernation result.** Genesis PR #2930 (merged 2026-06-12,
released in 1.1.2 — after the 0.4.5 pinned here) states that "the contact island
constraint solver was unusable: its main kernel did not even compile", and fixes
two **hibernation-gated** bugs: `set_pos`/`set_quat` bypassing wake-up when
hibernation is enabled, and stale hibernated-island chains leaving separated
bodies merged. A systematic under-transport with exactly zero variance is what
failure-to-wake looks like, so the hibernation numbers above are probably
measuring that bug rather than hibernation. **Re-test after upgrading; do not
write hibernation off on this evidence.**

The same PR does *not* invalidate §8.7: those runs had hibernation off, and the
third bug is a compile error this scene never hit — the island decomposition was
read directly and behaved coherently across ten configurations. The real
performance path is the upgrade, not a solver swap.
