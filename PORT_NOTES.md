# Simulator fidelity port — `port-to-dino`

Semantic port of the useful parts of two older research branches (`refactor`,
`GenesisWorld`) onto current `upstream/dino_integration`. Not a merge: the
upstream code has moved on, so each change was re-implemented against the
current architecture and **re-measured on Genesis 1.3.3** rather than trusted
from the old branch.

Detailed measurements, methodology and the reasoning behind each value live in
[`scaling_to_200_objects.md`](scaling_to_200_objects.md). Read the
[corrections](#corrections-to-the-scaling-guide) section below first — two of
that document's claims do not survive the move to the current tray model.

Every number in this file was measured on this branch, not carried over from
the old one.

**No performance figure is quoted anywhere in this document.** Timings and
speedups depend on the machine, the backend and the particle count, and Genesis
falls back from GPU to CPU *silently* — so a number written down here would be
a promise about someone else's hardware. What is quoted instead is step counts,
distances, angles and particle counts, which are properties of the physics and
should reproduce anywhere; a disagreement in those is a real regression.

For performance on your own machine, run the scripts in
[`tests/benchmarks/`](tests/benchmarks/) — they print the backend they actually
used, which is the first thing to check.

---

## 1. Bug fixes that change recorded physics

These four alter the dynamics of **every transition**. Data collected before
and after them is not comparable.

| # | Was | Now | Measured effect |
|---|---|---|---|
| 1 | Sweep loop's per-step `set_dofs_position` zeroed the plate's velocity. `RigidEntity.set_dofs_position` overrides its base signature to default `zero_velocity=True`, and `zero_all_dofs_velocity` zeroes `slice(0, n_dofs)` — it ignores `dofs_idx_local`. So a call meant to pin z/roll/pitch/yaw zeroed **x and y velocity at 250 Hz**, inside the loop whose job is to move the plate along x/y. | `zero_velocity=False` | Sweep tracking error **5.00 → 0.33 mm** mean; cruise **121.6 → 125.0 mm/s** against 125 commanded. |
| 2 | Particles are built with no explicit material, so `rho` is `None` and their mass comes from Genesis' `RHO_OBJECT = 600`. `_set_particle_density_value` skipped `set_mass` on the first call (guarded on `old_density is not None`), then rescaled from 600 as if it were the configured value. | Seed `old_density` from `RHO_OBJECT`, imported from Genesis rather than hardcoded | Implied particle density **600 → 750**, i.e. every particle mass was **0.8× its recorded density** in every dataset collected before this. |
| 3 | Plate friction never set → Genesis default **1.0**, and contacts combine as `max(µa, µb)`. | Explicit `plate.friction: 0.3` | Sampled particle friction had **zero effect at the tool interface** — the one interface the action acts through. |
| 4 | `enable_torsional_friction` off (Genesis default) | `True` | A cube resting on the tray had no resistance to twisting in place. The tool is a thin blade that strikes most particles off their centre of mass, so induced spin is a large part of what a push does. |

Two latent fixes with no behavioural change today: `dt`/`substeps` fallbacks
were `4e3` (4000 seconds) and `1`, now `4e-3` and `5`; and `safety_margin` was
declared in `basic.yaml` but hardcoded at `0.02` in the code and never read —
the code now reads it and the config states the value that was in force.

## 2. The plate is modelled as a gantry axis

The tool is a 2.4 g box — the *lightest* dynamic object in the scene. What
previously kept it on course was that four of its DOFs were hard-set every step
and its x/y velocity was zeroed every step. The dominant tracking error was
never granular reaction; it was the control law, which handed the tool its
**endpoint** as a PD position target and therefore ran it at a speed
proportional to distance remaining.

Now, all config-driven under `plate:`:

- `set_dofs_armature(moving_mass)` on x/y/z — the drivetrain's reflected
  inertia, added to the mass-matrix diagonal the constraint solver already
  uses. The correct knob rather than a denser plate, which would also change
  the tool's weight and contact response.
- gains from mass and bandwidth: `kp = mω²`, `kv = 2ζmω` at ζ=1 →
  4441.3 N/m and 94.25 N·s/m at the defaults, verified against the built scene.
- `set_dofs_force_range(±max_force)` — previously unbounded; a real stepper
  loses steps rather than applying unlimited force to a jam.
- a **trapezoidal** position + velocity reference replacing the endpoint target.

| | measured |
|---|---|
| tracking error vs commanded path | mean **0.33 mm**, max 0.97 mm |
| cruise speed | **125.0 mm/s** vs 125 commanded |
| final tracking error | **0.010 mm** |
| goal reached, sampled actions | **12/12** |
| sweep step count, 90 mm | **208** vs 306 under the old law |

The step count falls because it now comes from the trapezoid's real duration
rather than a `1.7×` fudge that was compensating for the speed error, and the
per-step `.nonzero()`/`.item()` GPU syncs are gone.

`plate.approach_mode: servo` (new default) drives the descent with the actuator
instead of teleporting the pose each step, so particles can resist it.
`hold_mode: servo` was built, measured and **rejected** — 3–6× higher particle
penetration — and is kept only as a documented option.

## 3. Settling is convergence-based

`update_material_state` ran a fixed 200 steps with no check that anything had
stopped. `settle_steps` is now a **cap** with a velocity-convergence exit and a
loud warning if the cap is hit.

Three things were needed to make the criterion work:

1. **A quantile, not a max.** Testing `max` over every particle in every env
   makes the criterion harder the more envs are batched, so the settle always
   ran to its cap. `settle_rest_quantile: 0.995` tolerates ~1 straggler per
   200-cube env.
2. **An angular threshold derived from the linear one.** A bare rad/s number is
   not comparable to a m/s number: 0.1 rad/s on a 5 mm cube is a corner speed
   of 0.35 mm/s, three times *stricter* than the 1 mm/s linear threshold, so it
   silently became the binding criterion. It is now derived through the
   particle's half body diagonal — 0.2309 rad/s at 5 mm.
3. **Not resetting the solver every step.** Holding the plate with a per-step
   `set_dofs_position` calls `collider.reset()` and
   `constraint_solver.reset()`, discarding the constraint solver's warm start
   with only 10 iterations to rebuild it. The plate is lifted clear during a
   settle and its PD holds it, so the teleport bought nothing. The control
   target is now set once.

True steps-to-rest, measured with the check interval set to 1:

| | n=50 | n=200 |
|---|---|---|
| fresh spawn | 34 | 34 |
| after a push | 1 | 1 |

Flat in `n`, and **~6× cheaper than the fixed 200 on a reset, ~20× per
transition**. Post-push is nearly free because the pile has already relaxed
during `execute_action`'s 40-step lift — it is at rest before the loop starts.

The criterion is genuine rather than lenient: at exit the *peak* particle speed
(not just the quantile) is 0.017 mm/s against a 1 mm/s threshold, and holding
the pile 200 further steps moves it **0.001 mm**, with no particle drifting
more than 1 mm.

`settle_check_every: 10` also acts as a minimum settle. That floor is
deliberate — it guards the one failure this criterion cannot see, a cube at the
top of its tipping arc, whose speed passes through a *minimum*. It costs ~2% of
a batch.

## 4. Contact budget is checked, not assumed

Genesis reports contact-pair overflow through an error bit that
`Simulator.step` inspects periodically. That mechanism **cannot fire here**:
`set_dofs_position` clears the bit as a side effect and the sweep loop calls it
every step, so the bit is always wiped before the next check reads it. Overflow
would be completely silent — contacts dropped, wrong physics recorded, no
exception. It has been observed as a CUDA illegal memory access rather than a
clean failure.

`contact_budget_usage()` therefore reads the collider's counters directly at
the end of every sweep, where usage peaks, comparing broad-phase pairs and
contact *points* against **their own separate caps** — two limits that differ
by more than an order of magnitude. The point cap is read from
`ColliderInfo.max_contacts` rather than recomputed as
`max_collision_pairs × n_contacts_per_pair`: since Genesis 1.2.x the buffer is
sized per contact regime and then reduced by link-pair pruning, so the old
product would *overstate* the cap and hide exactly the overflow this check
exists to catch.

`max_collision_pairs` now defaults to `max(150, n_particles // 2)`. This is not
cosmetic — the constraint Jacobian is
`O(max_collision_pairs × contacts_per_pair × n_dofs × n_envs)` while raw step
time is independent of the cap, so an oversized value converts directly into
lost parallelism.

`escaped_particle_count()` reports particles that have left the tray, which can
only happen if the contact solver failed for them.

## 5. Performance, with no behavioural change

**Batched particle pose writes.** `RigidEntity.set_pos`/`set_quat` each run a
forward-kinematics pass over the *whole scene*, so the per-particle loop in
`_set_particle_positions` cost 2N kernel launches and 2N full-scene FK passes —
400 of each at n=200, on every reset and every state restore. Replaced by two
solver-level calls taking a link-index array plus a single FK pass.

**Bit-identical to the old loop** — max difference 0.0 on both position and
orientation, which is the part that must hold regardless of hardware. The
speedup is entirely backend- and count-dependent, so it is not quoted here:
run `tests/benchmarks/bench_performance.py` for your machine's figure.

**`set_n_active(n)`** places only the active prefix and parks the rest outside
the tray *on a grid*. Particle count is otherwise a rebuild-only parameter
(particles are created before `scene.build()`, and `performance_mode` makes
every distinct scene shape pay a full kernel recompile). Parking inactive
particles at one shared point — the obvious implementation — piles them into a
single permanent contact cluster that costs solver time on every step of every
env, which matters more than it sounds given Newton's dense per-island Hessian.

## 6. New collection features

All opt-in. Omitting the flags reproduces the previous behaviour exactly.

### Settled-state library — `Genesis/state_library.py`

`--state-library N` settles N piles once per build, expands each by the
container's symmetry group, saves `settled_states.pt` beside the data, and
resets by restoring instead of re-settling. `shuffle_particles()` runs zero
simulation steps, so all of a reset's cost is the settle that follows it.

The symmetry expansion is what makes a handful of settles worth it: a settled
arrangement rotated or mirrored into another orientation of the tray is still a
valid settled arrangement, and a different configuration to sample from. A
square tray admits the full dihedral group **D4 — 8 variants per settle**.
Mirroring is applied to orientations too: a reflection `M` is improper, but
`M R M` is a proper rotation, which is legitimate because cubes and spheres are
achiral. Verified against explicit rotation matrices including `det = +1`.

The library size is exact arithmetic: `n_settles × n_envs × |symmetry group|`.
The property that matters is that **a restored pile is already at rest, so no
settle follows it** — verified directly (peak particle speed 0.0000 mm/s
immediately after a restore, with `_pile_is_at_rest()` true). How much time that
saves is a timing, so measure it:
`tests/benchmarks/bench_performance.py`.

All envs in a batch share one initial state, drawn without replacement. A batch
of identical piles is cheaper to simulate, and within a batch the sampled
action parameters already vary the dynamics substantially, so the variance
given up is small. Diversity is preserved *across* batches.
`StateLibrary.apply_per_env` exists where per-env states are genuinely needed.

### Touchdown pose sampling — `Genesis/placement_sampling.py`

`--start-sampling` selects between four samplers. The two mechanisms are
complementary rather than rival: density-weighted answers *where is there
material worth pushing* (a property of the pile), free-space answers *where can
the tool actually come down* (a property of the tool).

| mode | touchdown overlap | mean start radius |
|---|---|---|
| `uniform` | 89 % | 23.8 mm |
| `density` (upstream's, current default via `auto`) | 95 % | 34.1 mm |
| `free` | **16 %** | 25.1 mm |
| `composed` | **28 %** | **34.2 mm** |

*Overlap* = the blade's footprint at touchdown contains a particle centre. The
plate descending **into** a particle is resolved by the solver ejecting it — an
artifact recorded as though it were a push.

`density` deliberately *raises* overlap, because it aims at material. `free`
avoids particles but drifts toward empty tray, where a push moves nothing.
`composed` lets density choose the neighbourhood and then moves the pose the
shortest distance that makes it legal: it keeps density's spatial distribution
(34.2 vs 34.1 mm) while cutting overlap to 28 %.

`free` and `composed` fall back to the underlying draw per sample wherever the
free set is empty. They are refinements, not guarantees — as the tray fills the
free set shrinks and eventually empties.

### Shared travel distance — `Genesis/action_sampling.py`

`--shared-travel-distance` gives every env in a batch the same push *length*
for a given sample, while each keeps its own start point, direction and blade
yaw. A batching artefact fix, not a modelling choice: `sweep_steps` is derived
from the *longest* travel in the batch, so one long push makes every env run
for its duration. What is given up is the within-batch spread of one of five
action dimensions; a push that cannot reach the shared distance without leaving
its sampling box is truncated at the boundary.

### Reproducibility and audit trail

`data_collection.py` had no seed control: `np.random.default_rng()` with no
argument, and every torch draw (spawn poses, orientations, actions) unseeded. A
run could not be repeated, which also meant a run that produced something odd
could not be replayed to look at it.

`--seed` now seeds **both** generators and is recorded in each batch's config.
Verified: two independent runs with the same seed are **bit-identical**,
including `states_`.

Each saved batch also records `unchanged_transitions` (a run that "succeeds"
while nothing moves is the silent failure worth guarding against),
`escaped_particles`, and peak `contact_budget` usage — so a finished dataset
carries the evidence that it is trustworthy instead of requiring the run to be
repeated to find out.

## 7. Placement capacity, and the warning that matters more

Two different questions, and conflating them is easy:

- **Will it fit?** `single_layer_capacity()` — the exact ceiling the simulator
  enforces.
- **Is there anywhere left to act?** A much lower bar, and the one that decides
  whether a dataset is worth collecting.

The ceiling is set by two things that are both easy to get wrong, so it is
measured against the simulator rather than estimated. Particles are placed
**twice with different clearances**: `random_sequential_addition` (creation,
pre-build) clears `size/2`, while `_sample_nonoverlapping_particle_positions`
(every reset) clears `size/2 · √2` — the footprint a cube sweeps at free yaw,
1.41× per axis and so roughly twice the area. The reshuffle binds, because it
runs every batch. And within the reshuffle, rejection sampling gives up well
before the tray is full (132 of a possible 225 at 5 mm) and **falls through to
`_grid_particle_positions` silently**, so the effective ceiling is the grid's.

Verified at the boundary: 8.5 mm cubes shuffle at n=81 and **fail at n=82**;
5 mm shuffles at n=225.

| particle size | one-layer ceiling | 70 % of it |
|---|---|---|
| 5.00 mm | 225 | 157 |
| 6.75 mm | 144 | 100 |
| 8.50 mm | 81 | 56 |
| 10.25 mm | 64 | 44 |
| 12.00 mm | 49 | 34 |

`capacity_table()` returns this so the code and the docs cannot drift apart.

Three things follow from it, all now enforced in code:

**`plan_layers()` plans for room to act, not merely room to fit.** It targets
70 % per-layer occupancy, so 150 cubes of 8.5 mm gets 3 layers at 62 % rather
than the 2 layers at 93 % that would fit but immediately warn. `target_fraction
=1.0` recovers the old behaviour. A test asserts that whatever it returns never
trips the packing check.

**The blade is scaled to the stack.** The blade's bottom edge is pinned to the
resting particles' centre height, so its reach above the floor is `size/2 +
plate.size[2]` — and the stock 10 mm blade covers only 2 layers of 5 mm cubes.
The layered path grows `plate.size[2]` to `size · (L − 0.5)`, which is what it
takes to reach the top of a settled L-layer pile (10 → 21.25 mm for 3 layers of
8.5 mm). Sized for the *settled* pile rather than the spawn stack, because
nothing is pushed before the first settle. `plate.scale_height_with_layers:
false` opts out. Note a taller blade is not free: it is heavier and sweeps a
taller column of material.

**Stacking does not escape the problem, and the code says so.** A pile spreads
as it is pushed, so over a trajectory it tends toward a monolayer — at which
point the *total* count is what matters, not the per-layer figure. So a
multi-layer configuration gets a second, separate warning whenever the total
would be too full flat. At 150 objects of 8.5 mm that is 185 % of one layer,
and no layer count fixes it.

**`check_packing_fraction()` warns from 70 % of the ceiling**, at construction,
before paying for a scene build (a rebuild recompiles kernels, so it is not
cheap on any machine) and a much longer collection. A tray filled near its
placement ceiling is not a denser pile, it is a *stuck* one: the tool needs a
40 × 2 mm footprint clear at some yaw to touch down, and that vanishes long
before the last particle stops fitting.

The 70 % figure is not arbitrary. The placement-aware sampler's free set was
measured **fully available at 150 of 5 mm cubes and completely empty at 200** —
67 % and 89 % of the 225 ceiling — so the collapse sits between them. A test
asserts the threshold stays bracketed by those two measurements.

The consequence for the historical sweep is worth stating plainly: **no
particle size reaches 200 objects in one layer with room left to act.** Not even
5 mm, where 200 is 89 % of capacity. The guide's "5 mm is the only size that
reaches 200 objects" is true of placement and false of usability.

Reaching 150–200 objects usably therefore needs less *area per particle*, not
more layers. The levers are a smaller particle, a larger tray, or a shape that
packs tighter than a free-yaw cube — a sphere of the same size needs no √2
inflation and so gives roughly twice the capacity (441 vs 225 at 5 mm, 169 vs
81 at 8.5 mm). Whether a sphere pile is *usable* is a separate question from
whether it packs, and is measured in §8.

## 8. Do spheres help? Measured

Asked because stacking cubes does not escape the collapse problem (§7), so the
remaining lever is a shape that occupies less area. Spheres get no free-yaw
inflation, so statically they win outright:

| size | cube capacity | sphere capacity | gain |
|---|---|---|---|
| 5.00 mm | 225 | **441** | 1.96x |
| 6.75 mm | 144 | **256** | 1.78x |
| 8.50 mm | 81 | **169** | 2.09x |

But "packs tighter" is not "piles better", and measured at high friction
(0.9 on particles and tray, 8.5 mm, n=100, forced to 2 layers) they are
*strictly worse* at holding a stack:

| | cube | sphere |
|---|---|---|
| stacked particles at spawn | 50/100 | 50/100 |
| **stacked after settling** | **42/100** | **0/100** |
| stacked after 5 pushes | 29/100 | 0/100 |
| top surface after settling | 26.93 mm | **18.45 mm** (= one layer) |
| settle steps | **70** | **3000 — hit the cap** |
| mean radius, spawn -> settled | 49.6 -> 49.4 mm | 48.0 -> **54.6** mm |

Spheres collapse to a perfect monolayer immediately and spread outward doing
it, and the settle failed to converge at all — it burned its full step cap
where cubes converged in tens of steps. (Step counts, unlike timings,
reproduce anywhere: `tests/benchmarks/probe_piles.py --pour`.)

**Rolling resistance exists and fixes the settle.** Genesis 1.3.3 has
`rigid_options.enable_rolling_friction` (default `False`) and a per-geom
`friction_rolling` (`gs.materials.Rigid`, or `link.set_friction_rolling()`;
default 1e-4, i.e. negligible). Turning it on takes the sphere settle from
3000 steps *hitting the cap* to 230–560 steps and converging:

| `friction_rolling` | settle steps | stacked | converged |
|---|---|---|---|
| 1e-4 (default) | 1500 | 0/100 | yes |
| 0.01 | 300 | 0/100 | yes |
| 0.1 | 560 | 0/100 | yes |
| 0.3 | 260 | 0/100 | yes |
| 1.0 | 230 | 0/100 | yes |

It does **not** make spheres hold a stack, at any coefficient — which turned
out to be the wrong thing to expect of it. See §8.1.

**Now enabled automatically for rolling shapes.** `_init_scene` turns
`enable_rolling_friction` on when `material.shape` is `sphere` or `cylinder`,
and `set_material_properties` applies `material.rolling_friction` (default 0.3
when unset, to particles *and* the tray — a sphere rolls on the floor too).
Deliberately **off for cubes**: a cube resists rotation with its faces, so the
extra constraint rows would cost solver time for no effect. An explicit
`rigid_options.enable_rolling_friction` always wins. The resolved value is
recorded in each batch's config as `sampled.rolling_friction`.

One wart worth knowing: `basic.yaml` writes unset values as the literal `None`,
which YAML loads as the **string** `"None"`. Every other placeholder in that
file is overwritten by `data_collection.py` before use, so `rolling_friction`
is the first that has to cope with it, and does.

So the answer is a genuine yes, but not for the reason it was asked. Spheres do
not help you pile; they remove the need to. Since a pushed pile tends flat
anyway (§7), the number that matters is how many fit in *one* layer:

| shape / size | 150 objects | 200 objects |
|---|---|---|
| cube 5.00 mm | 67 % — ok | 89 % — too full |
| cube 6.75 mm | 104 % | 139 % |
| cube 8.50 mm | 185 % | 247 % |
| **sphere 5.00 mm** | **34 % — ok** | **45 % — ok** |
| sphere 6.75 mm | 59 % — ok | 78 % — too full |
| sphere 8.50 mm | 89 % | 118 % |

**5 mm spheres are the only configuration in the sweep that reaches 200 objects
flat with room left to act**, at 45 % occupancy. 6.75 mm spheres reach 150.
Against that, the settle-convergence problem above is the open cost.

### 8.1 But if the pile itself is the object of study, spheres are wrong

Everything above answers "how do I fit more objects". If the *pile* is the
phenomenon — how a 3D heap responds to a sweep — the question is which shape
holds a stack, and the answer is the opposite way round.

Measured, spheres at 8.5 mm with rolling friction 0.3, `set_n_active` sweeping
occupancy on one build:

| n | % of placement capacity | settle steps | stacked | result |
|---|---|---|---|---|
| 60 | 36 % | 190 | 0/60 | flattened |
| 120 | 71 % | 260 | 0/120 | flattened |
| 200 | 118 % | 1510 | 0/200 | flattened |

Spheres flatten at **every** count, including one the placer calls
over-capacity. That is not a contradiction: `single_layer_capacity` bounds what
the *placer* can generate on a gapped axis-aligned grid (169 here), while
hexagonal close packing puts **~262** 8.5 mm spheres in this tray. A flat
arrangement exists at n=200, and spheres find it.

**Cubes do the opposite.** At n=100 — only a **44 % area fraction** if laid
flat, so a flat arrangement is trivially available — 42/100 cubes still sat in
a second layer after settling, decaying to 29/100 over five pushes. Cubes hold
a stack they have no need to hold.

So piling here is **jamming**, a property of shape and friction, not an
occupancy threshold. Flat faces plus a high coefficient let cubes interlock
into a metastable heap; a sphere can always roll into a lower configuration,
and rolling friction slows that without preventing it.

Practical consequences:

- **Cubes are already the right primitive for studying piles.** The current
  configuration produces genuine 3D structure that degrades under pushing —
  which is presumably the phenomenon of interest.
- **Spheres are actively wrong for it**, whatever their packing advantage.
- The lever for *more objects in a pile* is therefore a **larger tray**, not a
  different shape: it raises the object count at a fixed packing fraction.
  The scaling guide's §8.7 already priced this (a 1.5x tray makes 200 cubes
  substantially cheaper to simulate, at a lower packing fraction — "a
  different task"; the guide's speedup figure is specific to the machine it
  was measured on). There is now a second reason to widen it: it is the only way to get
  a pile *and* room for the tool to act, which are otherwise in direct
  tension.
- **Cylinders are the untested rigid option.** `_particle_dimensions` already
  handles them and `_random_particle_quats` already spawns them upright or
  lying, so they can interlock like a cube while packing closer to a sphere.
  Worth one measurement before concluding.
- `gs.materials.MPM.Sand` is **not** an option here: it is Drucker-Prager
  plasticity over material points with no per-object shape, no orientation and
  no grain identity (`MPMEntity` exposes only `get_particles_pos` /
  `get_particles_vel`). It cannot express the `states [N, n, 7]` representation
  the datasets and models are built on. Same for `MPM.Snow`, `SPH.Liquid` and
  `PBD.Particle`. The rigid primitives are Box, Sphere and Cylinder only —
  there is no capsule — plus arbitrary `Mesh`.

## 9. Layered spawn — `Genesis/layered/`

A separate, self-contained path for particle counts that will not fit in one
layer (§7). The monolayer code in `Genesis/` is untouched by it. See
[`Genesis/layered/README.md`](Genesis/layered/README.md) for the full account.

**At 5 mm you don't need it.** The single-layer ceiling is set by the *grid
fallback* that `_sample_nonoverlapping_particle_positions` drops into when
rejection sampling fails, computed from the reshuffle's `size/2 · √2` free-yaw
clearance — not by the creation-time placer, which clears only `size/2` and is
therefore ~2.4x more permissive. Verified against the simulator: 8.5 mm cubes
shuffle at n=81 and fail at n=82. In the stock 128 mm tray, cubes:

| particle size | max in one layer | guide's assumed per-layer |
|---|---|---|
| 5.00 mm | **225** | 147 |
| 6.75 mm | **144** | 80 |
| 8.50 mm | **81** | 51 |
| 10.25 mm | **64** | 35 |
| 12.00 mm | **49** | 25 |

So at 5 mm the whole historical 50–200 sweep fits in one layer, but **14 of its
25 (size, count) cells need layering** — every count at 12 mm included.
`plan_layers()` answers the question analytically for any configuration, and
`--n-layers auto` uses it.

Implemented as **copies** rather than subclasses, because the two things that
must change — the box height and the creation-time placement — both happen
inside `__init__` before `scene.build()`, and one lives in a module-level
function with nothing to override. Each copy is ≤238 changed lines, opens with
a docstring listing exactly what differs, and was kept textually minimal so
`diff` stays the re-sync tool.

Two costs you accept by using it, both documented rather than hidden: a second
layer **occludes the first from the top-down camera**, so image-based models see
partial observations; and a pile deeper than the blade is tall has material the
tool cannot reach — the code warns at construction and reports by how much.
Settling shrinks that but need not remove it: measured at 150 cubes of 8.5 mm,
the 2-layer spawn stack reaches 30.0 mm and the settled pile's top surface is
27.0 mm against a blade top edge of 24.25 mm, so 2.74 mm remains out of reach.
The blade cuts a 10 mm band out of a 17 mm pile — 58 % of the pile height
engaged, 2.74 mm above it and 4.27 mm below. The lower gap is not a layering
artefact: the blade rides half a particle above the floor by construction, so
the bottom half of the resting layer passes under it in the monolayer case too
(the guide's open decision 3).

Output goes under `.../size<S>/layers<L>/`. The `layers<L>` component is
deliberate: layered and monolayer data at the same (shape, count, size) have
different dynamics and must not be loaded as one distribution.

---

## 10. Collection driver — `Genesis/run_collection.py`

One subprocess per particle count. Not a convenience: changing `n_particles`
needs a full scene rebuild — particles are created in `__init__`, before
`scene.build()`, and `performance_mode` recompiles kernels per scene shape — so
each count is a separate process whether or not you want it to be. Making it
explicit means the env count can differ per pile size, a crash or OOM at one
size cannot take the rest of the run with it, and each size is checked going in
and validated coming out.

**Preflight** uses the same `single_layer_capacity` the simulator enforces, so
"fits" here means the reshuffle will not fail on batch two:

```
[ok  ] n=100   placement: 100/225 = 44% of a layer
[warn] n=200   placement: 200 is 89% of the 225-particle layer capacity. It
               will place, but the tool has little room to touch down and
               placement-aware sampling will mostly fall back to blind.
[SKIP] n=200   placement: 200 exceeds the single-layer capacity of 49 (408%).
               The reshuffle placement would fail. Use Genesis/layered/ ...
```

**Postflight** validates the files each size actually wrote — shapes, dtype,
non-finite values, plausible displacement — reads back the audit fields the run
recorded about itself (escaped particles, unchanged transitions, peak contact
budget), and checks the state library is the size the symmetry expansion
implies. Above all it checks **s′ actually differs from s**; verified by
corrupting a batch so `s' == s` and confirming it reports
`NO sample changed state - the actions had no effect`.

Exits nonzero if any size failed to run or failed validation, and writes
`run_collection_report.json` beside the data.

```bash
cd Genesis
python run_collection.py --plan configs/collection_dry_run.yaml --preflight-only
python run_collection.py --plan configs/collection_dry_run.yaml
```

`plan.n_envs` takes a single integer, a `{count: envs}` mapping, or a path to a
yaml written by a throughput benchmark — so a plan can reference a measurement
instead of copying numbers out of it and going stale. A referenced file's
material is checked against the plan's, because a throughput optimum does not
transfer across materials.

The VRAM preflight only ever *warns*. Its per-env estimate was fitted on one
card, and this document's policy is that no performance figure is a promise
about someone else's hardware — so it is overridable (`--vram-per-env`) and an
over-estimate costs one size, not the run.

## 11. MPC stack — `MPC/`

The MPC / world-model research stack, ported to run against the current
simulator. One directory, kept separate from the simulator work so the
`Genesis/` PR stays reviewable on its own. Full account in
[`MPC/README.md`](MPC/README.md); the five things the port changed:

1. **`training/` → `model_training/`.** `Genesis/training/` also exists, and
   with both directories on `sys.path` a plain `import training` resolved to
   whichever came first — this package's had an `__init__.py`, so it won and
   `training.dataset` became unreachable. Import order should not decide which
   package you get.
2. **`from Genesis.x import y` → flat imports**, via one `genesis_path.py`.
   Four stale hacks computing the path relative to the old file locations —
   pointing at a non-existent `MPC/Genesis` — were removed.
3. **Transition recording moved into a subclass.** `push_and_record`,
   `flush_transitions`, `set_transition_context` and `broadcast_state_from_env`
   were methods on the historical simulator and are not in the current one.
   They are now `MPC/env/recording_sandbox.py`'s `RecordingSandbox`, because
   the simulator already has a recording path and a second overlapping one is
   what the simulator PR should not carry. Subclassing works here — unlike the
   layered spawn, which had to be a copy — because none of it runs before
   `scene.build()`.
4. **`reset_warmup_steps` 10 → 500.** A semantic change, not tuning:
   `settle_steps` used to be a fixed count and is now a cap with a convergence
   exit. Under the old semantics a bigger number was pure cost, which is why
   the config read `reset_warmup_steps: 10  # was 500 but was a no-op`. Under
   cap semantics 10 is far too small, and the simulator says so — the pile is
   recorded mid-motion at 9.95 mm/s against a 1.0 mm/s threshold.
5. **`model/futureintegration/` lost its four `.py` files**, all byte-identical
   to files upstream maintains in `GranularDynamics2/myClasses/`. The notes
   stay; the code cannot drift. One of them, `Diff_Renderer.py`, does not even
   parse.

Verified by importing every module (49/50 — the one failure is a design sketch
that never imported) and by constructing a `GenesisEnv`, calling
`push_and_record`, and confirming the flushed transition file has the right
shapes and the episode context attached.

## 12. Dataset: what was taken, and one defect fixed

`Genesis/training/dataset.py` is **upstream's**, kept as-is apart from the
changes below. The old branch's version of this file was not a superset — each
side had gained things the other lacked (upstream: `include_sweep_removed`,
reading `_rollout.pt` via `source=`; the old branch: MPC accessors and
configurable physics bounds), so replacing it would have lost real capability.

### Physics normalisation stays hardcoded

`_det_physics` maps `(friction, density, box_friction)` onto `[0, 1]` over
friction 0.05–0.50, density 750–5000, box friction 0.05–0.50. The old branch
made those bounds a config-driven `PhysicsBounds` object; that was **not**
ported, and the call site in `MPC/registry/dataset_registry.py` now uses
upstream's signature unchanged.

Those constants are not arbitrary — they are the endpoints of the original
collection sweep (`np.linspace(0.05, 0.5, 5)`, `np.linspace(750, 5000, 5)` in
`data_collection.py`), so the normalisation spans exactly the range the data
was drawn from. Verified: upstream's `PileSweepData` loads what the current
collector writes, with no error.

**Physics is held fixed per collection run by default**, so the normalised
vector is a constant — `[0.1556, 0.0, 0.1556]` at the shipped
friction 0.12 / density 750 / box friction 0.12. That is deliberate, and it has
one consequence worth stating: **the FiLM-conditioned models are only
meaningful when physics is varied.** With a constant conditioning vector there
is nothing for FiLM to modulate on, so use the plain U-Nets instead; the FiLM
variants are kept for the case where a sweep is turned back on. (At the default
config, FiLM generators are 38 % of `NFDUNetFiLM`'s parameters.)

If a sweep ever uses a different range, those constants become *wrong* rather
than merely fixed. The place to put an override back is the **wrapper, not the
dataset** — which is what the real-data path already does:
`RealPileSweepData` returns raw physical units and
`_RealEulerianDatasetWrapper` applies `bounds.normalize()`. Copying that
pattern keeps one source of truth for the bounds and leaves upstream's dataset
alone.

### Fixed: no val or test split at a single physics setting

Reproduced before the fix:

```
PileSweepData("<one n, one size, one physics>", split="val")
→ ValueError: No configs found for dataset.
```

`train` worked; `val` and `test` both raised.

Splits are stratified by `_physics_key` = `(shape, n_particles, particle_size,
friction, density, box_friction)`, so that no physics setting appears in both
train and val. That is the right guarantee when physics varies. But with
physics fixed per run — the default — every run in a folder shares one key, so
there was **one group**, `_assign_group_splits(1, …)` returned `['train']`, and
val and test came back empty.

`_filter_split` now falls back to splitting **by run** when a folder yields
fewer than `MIN_GROUPS_TO_STRATIFY_BY_PHYSICS` (3) distinct physics groups. The
run is the right unit: runs are independent (separate shuffles, separate action
draws) while samples *within* a run are a trajectory — sample i+1's state is
sample i's next-state — so a finer split would put the two ends of one
transition on opposite sides. Runs are ordered by a hash of the file *name*, so
the partition is identical on any machine and across the separate
train/val/test constructions.

Measured on 5 runs at one physics setting, 30 samples:

| split | samples | runs |
|---|---|---|
| train | 18 | `_2`, `_3`, `_4` |
| val | 6 | `_0` |
| test | 6 | `_1` |

Disjoint by file, covering all five runs. Datasets with ≥3 physics groups are
untouched and still stratify by physics — `configs/dataset/genesis_dmdc_cube_n50.yaml`
("7 physics groups") behaves exactly as before.

One caveat: if a state library is in use, two runs can start from the same
settled pile, so a val run may share an initial state with a train run. The
actions and every subsequent state still differ.

### Added: three accessors

Purely additive, used only by `MPC/dmdc_baseline.py`, which needs the push in
world metres rather than painted into an input channel:

- `get_raw_action(idx)` → `[sx, sy, ex, ey]` in metres
- `get_run_index(idx)` → which run a sample came from, usable as an episode id.
  **Instance-local** — it indexes the files *this split* kept, so run 0 of val
  is a different file from run 0 of train. Group by it; do not identify by it.
- `workspace_bounds` → `((x_min, y_min), (x_max, y_max))` in the same frame

`__getitem__` is unchanged, so nothing a training run sees is affected. The
DINO exporter is unaffected too — it builds the class with
`object.__new__(PileSweepData)` for its rasterisation methods only and never
reaches `__init__` or the split logic.

## Corrections to the scaling guide

[`scaling_to_200_objects.md`](scaling_to_200_objects.md) was written against the
older simulator. Two of its claims do not transfer, and the guide should be read
with these in mind until it is revised:

**§1.1 fix 7 and all of §2 (capacity) describe a tray that no longer exists.**
Upstream now *derives* the box height rather than reading it from config:

```python
self._box_params["vol"][2] = self._wall_thickness + max_particle_height(...)
```

`max_particle_height`'s docstring states its purpose — "so a resting monolayer
never sticks out above the walls" — and `random_sequential_addition` raises if
the height is any less. The tray is therefore **a monolayer by construction**,
which a top-down camera feeding the DINO pipeline also depends on. Measured on
this branch: n=200 at 5 mm places and settles fine in a single flat layer.
So the guide's "150 and 200 could not be placed at any size" was a limitation
of the *old* placer, which upstream's rejection sampler with grid fallback has
already solved. The layered spawn is not part of the monolayer path; it lives in
[`Genesis/layered/`](Genesis/layered/) (§9), where the guide's ~1460-step
respawn settle is the reason for the higher `settle_steps` cap. Its §2 capacity
table is superseded outright — the measured single-layer ceilings are ~2.2x
what it assumed.

**§1.3's settle step counts are two-layer figures.** The ~250 post-push and
~1460 fresh-respawn numbers were measured on a spawn where particles fell from
a second layer and had to collapse. On the monolayer, true steps-to-rest is
34 fresh and 1 post-push, flat from n=50 to n=200 (§3 above). The cap here is
500, not the guide's 2500.

The rest of the guide — the plate model, the solver-knob measurements (§1.4),
the contact-island cost law (§8.7), the solver-equivalence study (§8.8) and the
end-to-end verification (§8) — is unaffected.

---

## Not ported

| | Why |
|---|---|
| Deletion of `GranularDynamics2/*`, `train_unet_genesis.py`, `Genesis/training/dataset_cop.py` | Those deletions were a reorganisation on the old branch. Upstream uses these files actively alongside `dino_wm`. |
| `Genesis/training/dataset.py` refactor, as a whole | Upstream's version is kept. It gained things the old one lacks (`include_sweep_removed`, reading `_rollout.pt` via `source=`), and the old one removed upstream capability (`run`, `split=None`). Only three additive accessors were taken across, plus one defect fix — see section 12. |
| Layered spawn | Superseded here; separate `_layered` script. See corrections above. |
| `Genesis/transition_buffer.py`, `push_and_record` | An incremental recording path built for oracle MPC. Upstream has its own (`_save_rollout`, `_render_all_envs`, `export_dino_wm_dataset`); two overlapping mechanisms would be worse than either. |
| The MPC / model research stack | Large, separable, and belongs in its own branch under `MPC/`. Verified separable: the new `Genesis/*` modules import nothing outside the `Genesis` package. |
| Data files, run outputs, rendered videos | ~2500 generated files on the old branches. |

## Pre-existing issues found but deliberately left alone

- **Eight of the nine files in `Genesis/configs/` no longer load.**
  `basic_example.yaml`, the six `chick*.yaml` and `param_optim.yaml` use an
  older nested schema (`sandbox: {box:, material:, safety_margin:}` with
  `properties:` sub-dicts) while `__init__` reads `box`, `material` and `plate`
  at the top level, so they raise `KeyError`. All predate this work. Migrating
  configs that cannot be tested here is out of scope for a physics change; they
  may also be kept deliberately as references for the older pipeline.
- Unknown `rigid_options` keys now **warn** rather than raise. Previously they
  were silently dropped, which is how `enable_torsional_friction: True` could
  have been written in `basic.yaml` and done nothing — the same failure mode
  `safety_margin` had.
- **Five `.pyc` files are tracked in git** (`Genesis/training/__pycache__/`,
  `GranularDynamics2/__pycache__/`, `GranularDynamics2/myClasses/__pycache__/`).
  `.gitignore` lists those directories, but it cannot untrack files already
  committed, so the working tree goes dirty after anything imports them. All
  predate this work. `git rm --cached` on those five paths would settle it;
  left alone as it is not this change's business.
- Saved configs contain `!!python/tuple` tags (from `particle_sizes`), so they
  need `yaml.full_load`, not `safe_load`. Upstream's readers already use
  `full_load`; noted only because a new consumer would trip on it.

## Still to do

- A `--rolling-friction` CLI flag, if the value ever needs sweeping. It is
  currently config-only (`material.rolling_friction`).
- `koopman_skeleton.py` is a design sketch with undefined module-level names
  and does not import. Left as it was.

## Testing

Two gates. The first is fast and needs no simulator; the second builds a scene
and asserts every fix in this document.

```bash
python -m pytest tests/ -q          # 108 passing, pure torch, no Genesis needed
python tests/verify_fixes.py        # 54 checks, exits nonzero on any failure
```

`tests/verify_fixes.py` is the answer to "does all of this still work" as a
single command. It asserts rather than prints, and every check is a step count,
a distance, an angle or an exact equality — never a timing — so it should
reproduce anywhere and a failure is a real regression.

Two of its checks are **differential**: they force the old buggy behaviour back
and require the metric to get materially worse. `SWEEP-REGRESSES` measures 15×
worse tracking with `zero_velocity=True` restored; the density check requires
re-application to be idempotent. Those matter because a passing absolute
threshold can hide a fix that has quietly stopped doing anything — an
absolute-only gate would still pass if the fix were reverted *and* something
else compensated.

It also found a bug in the probe it was derived from: `probe_physics.py` swept
90 mm **edge-on**, which puts the blade's leading edge at 65 mm against a 64 mm
tray half-width. That drove it into the wall and reported 7 N of "contact" and
a 1 mm terminal deflection **on an empty tray** — indistinguishable from a
control failure. Both now sweep 80 mm broadside, inside
`generate_action_samples`' own bound for that yaw, and `verify_fixes.py`
asserts that bound is respected so the mistake cannot come back.

Benchmarks and probes — these need Genesis and a built scene, so they are not
part of the test suite. Each prints the backend it used:

```bash
python tests/benchmarks/bench_performance.py   # the only timings; machine-specific
python tests/benchmarks/probe_physics.py       # plate, settling, actions
python tests/benchmarks/probe_piles.py --pour --shape sphere
```

See [`tests/benchmarks/README.md`](tests/benchmarks/README.md).

Collection, run from inside `Genesis/` (upstream's convention — this package
uses flat sibling imports, not relative ones):

```bash
cd Genesis
python data_collection.py \
    --num-particles 50 --particle-sizes 0.005 \
    --n-envs 4 --samples-per-env 5 \
    --seed 0 --state-library 8 \
    --start-sampling composed --shared-travel-distance
```

## New config keys

All have defaults matching the values documented in `Genesis/configs/basic.yaml`,
where each carries its measured justification.

| section | keys |
|---|---|
| `simulation` | `settle_steps`, `settle_check_every`, `settle_velocity_threshold`, `settle_rest_quantile`, `settle_angular_velocity_threshold` (derived if absent), `pos_ctrl_steps`, `sweep_settle_steps` |
| `rigid_options` | `constraint_solver`, `enable_torsional_friction`, `max_collision_pairs` — plus any other `RigidOptions` field, now forwarded rather than silently dropped |
| `plate` | `friction`, `moving_mass`, `acceleration`, `control_bandwidth_hz`, `max_force`, `hold_mode`, `approach_mode`, `arrival_steps`, `orientation_inertia`, `orientation_bandwidth_hz`, `max_torque` |
| `material` | `rolling_friction` — unset follows the shape (0.3 for spheres/cylinders, unused for cubes) |
| top level | `safety_margin` (now actually read) |

## New files

| file | lines | what it is |
|---|---|---|
| `Genesis/state_library.py` | 410 | settled-state bank, symmetry augmentation |
| `Genesis/placement_sampling.py` | 296 | occupancy grid, C-space free set, `nearest_free_placement` |
| `Genesis/action_sampling.py` | 93 | shared batch travel distance |
| `tests/test_state_library.py` | 235 | 27 tests |
| `tests/test_placement_sampling.py` | 289 | 19 tests |
| `tests/test_action_sampling.py` | 103 | 6 tests |
| `tests/test_layered_spawn.py` | 190 | 35 tests |
| `tests/test_packing_capacity.py` | 201 | 21 tests |
| `Genesis/layered/` | — | the layered spawn path (§7); see its README |

`requirements.txt` pins `genesis-world==1.3.3` (was unpinned) and adds `scipy`,
previously an undeclared transitive dependency.
