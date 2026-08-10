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
| PD hold, 10 iters | 1.0 mm/s, 0.28 rad/s | 0.21 mm/s | 0.75× |
| **PD hold + Genesis' default solver budget** | **0.2 mm/s, 0.07 rad/s** | **0.11 mm/s** | **0.81×** |

`rigid_options` are therefore back at Genesis' defaults (`iterations: 50`,
`tolerance: 1e-6`) — an 8× lower noise floor at **0.81×** the previous cost per
step, because the removed collider/constraint resets cost more than five times
the solver iterations do.

**Post-push settling is ~250 steps (1 s of simulated time).** The expensive
1460-step settle applies only to a fresh two-layer respawn, which happens at
reset, not per transition. See `tests/scaling_investigation/settling_investigation.md`.

### 1.4 `max_collision_pairs`

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

### 3.1 Parallel environments

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

Resolved: the `max(µ_particle, µ_box)` friction-combining rule is acceptable
provided the tool's own friction is recorded, which it now is (§1.1 fix 3).

---

## 7. Where the investigation lives

`tests/scaling_investigation/` — probe and benchmark scripts, raw measurement
outputs, and `settling_investigation.md`, the full diagnosis of why piles
appeared not to settle. See that directory's `README.md` for what each script
measures and how to re-run it.
