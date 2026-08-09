# Scaling pile data collection to 200 objects

Measured on an **RTX 4070 Laptop GPU (8188 MiB, ~4.7 GiB usable with the
desktop running)**, Genesis 0.4.5, torch 2.11.0+cu130, `pme` conda env.

Everything below was produced by three new scripts, all runnable from the repo
root:

| script | question it answers |
|---|---|
| `Genesis/benchmark_scaling.py` | cost & VRAM per `(n_particles, n_envs)` cell |
| `Genesis/probe_step_cost.py` | what one sim step costs, and which knob drives it |
| `Genesis/probe_contact_counts.py` | how many collision pairs a pile really needs |
| `Genesis/probe_plate_dynamics.py` | does the pusher plate track its commanded path |

Raw results: `outputs/scaling_benchmark/`.

---

## 1. Headline answers

### 1.1 How many envs fit at once

Measured OOM boundary (VRAM figure is the last cell that fit):

| n_particles | max n_envs | VRAM at that cell | first OOM | envs × particles |
|---|---|---|---|---|
| 50 | **64** | 4.49 GiB | — (64 was the sweep max) | 3200 |
| 70 | **32** | 4.31 GiB | 64 | 2240 |
| 100 | **16** | 5.15 GiB | 32 | 1600 |
| 150 | **8** | 4.95 GiB | 16 | 1200 |
| 200 | **4** | 4.10 GiB | 8 | 800 |

The `envs × particles` budget *shrinks* as the pile grows — it is not a
constant. That is not intrinsic: it is because this sweep scaled
`max_collision_pairs` with `n_particles` (`4 × n`), and the dominant GPU
allocation is the constraint Jacobian, which is
`O(max_collision_pairs × contacts_per_pair × n_dofs × n_envs)`. Since
`n_dofs = 6·(n+1)`, tying the cap to `n` makes VRAM grow as `n²`.

**Setting the cap from measurement instead (§3) is the single biggest lever on
how many envs fit at 200 particles.**

### 1.2 Reset cost vs. action cost

`shuffle_particles()` runs **zero simulation steps** — it is pure GPU
rejection sampling plus per-particle pose writes. All of a reset's real cost is
the *settle* that has to follow it.

Measured (n_envs = 1):

| n_particles | `shuffle_particles` | settle (100 steps) | ms per settle step |
|---|---|---|---|
| 50 | 0.37 s | 4.71 s | 47 |
| 70 | 0.31 s | 7.28 s | 73 |
| 100 | 0.41 s | 12.36 s | 124 |
| 150 | 0.81 s | 23.97 s | 240 |
| 200 | 2.21 s | 40.05 s | 400 |

In *simulation steps*, which is the confound-free currency:

- **reset** = 0 (shuffle) + `settle_steps` = **100 steps**
- **one action** = lower (25–60) + sweep (~300–580) + lift (25–60) + settle
  (100) ≈ **450–800 steps**

So **a full re-randomization costs 12–22 % of one transition.** Resetting after
every single sample instead of every fifth would raise total wall-clock by only
~20 %, and would remove the strong correlation between the 5 sequential pushes
that `--samples-per-env 5` currently takes on one pile, assuming that is desired.

### 1.3 The cheapest reset is loading from settled snapshot

`set_particle_state()` — the snapshot-restore path the oracle MPC already uses
— was measured at **0.09 s (n=50) to 0.84 s (n=200)**, and it needs **no
settle at all** if the snapshot came from an already-settled pile.

| n_particles | shuffle + settle | restore | speedup |
|---|---|---|---|
| 50 | 5.08 s | 0.094 s | 54× |
| 100 | 12.77 s | 0.159 s | 80× |
| 200 | 42.26 s | 0.763 s | **55×** |

**Recommendation:** generate a library of settled pile states once per
`(shape, size, n)` build, then draw from it with `set_particle_state()` instead
of paying `shuffle + settle` every episode. State diversity then costs
essentially nothing, and is bounded only by the library size rather than by
sim time.

---

## 2. Per-step cost — where the time actually goes

*(filled in from `probe_step_cost.py`)*

---

## 3. `max_collision_pairs` — real mechanism, much smaller than first feared

Two things are true and independent.

**The overflow really is silent.** On overflow Genesis's broadphase sets an
error bit and stops adding pairs, dropping the rest. That bit is surfaced by a
periodic `check_errno()` at the *start* of `Simulator.step` — but
`RigidSolver.set_dofs_position` clears `_errno` as a side effect, and both the
settle loop and the sweep loop call it every step. So the flag is always wiped
before it can be read, and an overflow here would never raise. That is why
`SandboxManipulation.contact_budget_usage()` / `_check_contact_budget()` now
read the counters directly.

**But the pile does not come close to the cap.** Measured for 50 cubes of 5 mm,
settled and then pushed:

| quantity | used | cap at `mcp`=264 |
|---|---|---|
| broad-phase candidate pairs | 51 | 2112 (`mcp × 8`) |
| narrow-phase contact points | 211 | 4224 (`mcp × n_contacts_per_pair`, ncp=16) |

which implies a required `max_collision_pairs` of only **14**. A settled pile is
mostly one floor contact per cube — four contact *points* each under
`box_box_detection` — plus a few neighbours.

Two corrections to what I said earlier in this investigation:

- I first compared `n_contacts` against `max_collision_pairs`. That is wrong:
  `n_contacts` counts contact **points** (cap `mcp × ncp`), while the pair
  limit is `n_broad_pairs` (cap `mcp × 8`). The two differ by more than an
  order of magnitude.
- Consequently **the flat default of 150 is not the bottleneck I suspected**,
  and my first scaled default (`4n + 64` = 864 at n=200) was 6–16× larger than
  needed. Since the constraint Jacobian is
  `O(mcp × ncp × n_dofs × n_envs)` and raw step time is *independent* of `mcp`,
  an oversized cap buys nothing and directly costs parallel environments. The
  default is now `max(150, n_particles)`, with the runtime check catching an
  under-estimate loudly.

**The env-count table in §1.1 was measured with `mcp = 4n`, so it understates
what fits.** At n=200 the cap drops 864 → 200, a ~4× reduction in the dominant
allocation; the ceiling should rise well above the 4 envs measured. That needs
re-measuring on a free GPU.

---

## 4. Placement: 200 objects did not fit before this work

`shuffle_particles` placed every particle in a **single layer**, and treats a
free-yaw cube as an axis-aligned square of side `size·√2`. Replicating its
exact acceptance test on CPU gives the RSA saturation count per size:

| box 128×128 mm | 5.00 mm | 6.75 mm | 8.50 mm | 10.25 mm | 12.00 mm |
|---|---|---|---|---|---|
| max placeable, one layer | 139 | 79 | 50 | 34 | 25 |

So `n=150` and `n=200` raised `RuntimeError` at **every** size in
`data_collection_clean.py`'s `PARTICLE_SIZES`, and
`data_collection_clean.py:145` catches that and silently skips the batch. This
was already biting at the current scale — it is why line 111 special-cases
`if n_p == 50: sizes = sizes[:-1]`.

**Change made:** `shuffle_particles` now tries a single layer first (bit-for-bit
unchanged behaviour whenever one fits) and only adds a stacked layer when a
layer genuinely cannot be packed, raising a descriptive error if the stack
would exceed the box walls. Capacity in the *current* box becomes:

| cube size | per layer | max layers (20 mm interior) | max N | layers for 50/70/100/150/200 |
|---|---|---|---|---|
| 5.00 mm | 147 | 3 | **441** | 1/1/1/2/2 |
| 6.75 mm | 80 | 2 | **160** | 1/1/2/2/✗ |
| 8.50 mm | 51 | 1 | **51** | 1/✗/✗/✗/✗ |
| 10.25 mm | 35 | 1 | 35 | ✗ |
| 12.00 mm | 25 | 1 | 25 | ✗ |

**5 mm is the only size in the current sweep that reaches 200 objects in the
current box.** If chickpea scale (~8.5 mm) must be preserved, the tray has to
grow: 200 × 8.5 mm needs a **179 mm** box at 2 layers, or **253 mm** at one
layer, and the 2-layer option also needs walls taller than 40 mm.

Note the effective interior height is **20 mm**, not the configured 40 mm: the
walls span z ∈ [−0.01, +0.03] while the floor surface sits at z = +0.01.

---

## 5. The pusher plate — modelled as a gantry axis

### 5.1 The premise was inverted

The plate is **2.4 g** — `rho=3000` on an `[0.04, 0.002, 0.01]` box — making it
the lightest dynamic object in the scene, not a heavy one. What actually kept
it on course was that z/roll/pitch/yaw were hard-set every step and its x/y
velocity was zeroed every step.

### 5.2 What was actually wrong

Measured on one 90 mm crossing sweep, 50 cubes of 5 mm, comparing the original
configuration against the new one:

| | tracking error vs commanded path | reaction displacement (loaded − free) | cruise speed |
|---|---|---|---|
| **legacy** (endpoint target, kp=0.8, no armature, velocity zeroed) | mean 9.24 mm, **max 23.90 mm** | mean 0.07 mm, max 0.12 mm | 102 mm/s |
| **legacy, dense pile** (ρ=5000) | same | mean 0.33 mm, max 0.46 mm | 102 mm/s |
| **current** (trapezoid + gantry actuator) | mean 0.32 mm, max 0.92 mm | mean 0.00 mm, **max 0.04 mm** | **124.0 mm/s vs 125 commanded** |

The headline: **granular reaction was never displacing the tool much** — 0.12 mm
at ρ=1000, 0.46 mm at ρ=5000, the densest material in the sweep. The dominant
trajectory error was the *control law*, which put the tool up to **23.9 mm**
(nearly five particle diameters) off its commanded straight line. So the heavy-
plate workaround was aimed at the wrong problem; the tool wasn't being pushed
around, it was being driven wrong.

### 5.3 The model now used

Matching a 3D-printer-style Cartesian gantry: heavy carriage, steppers with
large force margin, trapezoidal velocity profile.

- **`set_dofs_armature(moving_mass)` on x/y/z** — reflected drivetrain inertia,
  added to the mass-matrix diagonal the constraint solver already uses. This is
  the right knob rather than a denser plate, which would also change the tool's
  weight and its contact response.
- **Gains derived from mass and bandwidth**: `kp = m·ω²`, `kv = 2ζmω` at ζ=1.
  The 15 Hz default gives ω ≈ 94 rad/s against a 0.8 ms substep (ω·h ≈ 0.075).
- **`set_dofs_force_range(±max_force)`** — previously unbounded; a real stepper
  loses steps rather than applying unlimited force to a jam.
- **A trapezoidal position + velocity reference** replaces the endpoint target.
  This is the change that fixes the speed: commanding the endpoint makes the PD
  settle at `v = v_cruise + kp·Δ/kv`, so speed depended on distance remaining.

All four are config-driven under `plate:` in `configs/basic.yaml`.

Side benefits: the sweep now takes **208 steps instead of 307** (−32 %), because
the step count comes from the trapezoid's actual duration rather than a `1.7×`
fudge factor that was compensating for the old speed error; and the per-step
`.nonzero()` and `.item()` syncs are gone, since the reference holds at the goal
and no freeze bookkeeping is needed.

### 5.4 Known residual

The sweep lands **0.92 mm** short of target. It is *not* a settling transient
(60 settle steps gives 0.95 mm) and *not* disturbance stiffness (2.8× the gain
gives 0.86 mm) — it is gain-insensitive and unexplained. It currently sits just
inside the 1.0 mm `goal_threshold`, so `reached_goal` passes, and the recorded
action uses the *actual* final position (`collect_data_samples` overwrites
`p_stop` with `final_pos`), so it is not a labelling error. But the margin is
thin: if it ever crosses 1 mm, samples get discarded as failures. Worth either
understanding or giving `goal_threshold` explicit margin before a long run.

---

## 6. Storage

`states`/`states_` are `float32 [N, n_particles, 7]` (position + quaternion);
no images are stored. Per transition = `2·n·7·4 + 28` bytes.

| n_particles | per transition | 10 k | 100 k | 1 M |
|---|---|---|---|---|
| 50 | 2.76 KiB | 27 MiB | 0.26 GiB | 2.7 GiB |
| 100 | 5.50 KiB | 54 MiB | 0.52 GiB | 5.2 GiB |
| 200 | **10.96 KiB** | 107 MiB | **1.05 GiB** | 10.5 GiB |

Storage is not a constraint at any scale considered here.

---

## 7. Build / rebuild cost

`n_particles`, `particle_size`, `shape`, `box.vol`, `n_envs`, `dt`, `substeps`
and every `rigid_options` field **require a full rebuild** (particles are
created in `__init__`, before `scene.build()`). Only particle friction,
particle density and box friction can change in place via
`set_material_properties()` — which is why the current sweep's inner
100-iteration material loop is cheap.

Measured build time was **32–117 s per cell**, and it did *not* amortize across
cells: every distinct `(n_particles, n_envs, max_collision_pairs)` triple paid a
full recompile. That is a direct consequence of `performance_mode: True`
(`configs/basic.yaml:7`), which disables Genesis's ndarray/fastcache path and
specializes kernels on the scene's array shapes.

Two ways to avoid paying it 25 times over a `shape × n × size` sweep:

1. **`set_n_active(n)`** already reduces the particle count without a rebuild,
   so one build at n=200 can serve 50/70/100/150/200. Caveat: it parks every
   inactive particle at the *same* point
   (`sandbox_manipulation_clean.py:597-599`), which would stack 150 cubes in
   one spot and consume the contact budget — needs a per-index offset first.
   Also note per-step solver cost stays at the n=200 level for every point.
2. **Benchmark `performance_mode: False`** for collection runs. Genesis's own
   log describes ndarray mode as existing "to avoid scene-specific
   compilation"; field mode buys a few percent of runtime in exchange.

---

## 8. Recommended collection plan

1. Set `max_collision_pairs` from the measured peak (§3), not from a guess —
   this both fixes the silent contact dropping and buys back envs.
2. Collect at **5 mm cubes** if the object count is what matters; widen the box
   to ≥180 mm if chickpea scale is what matters.
3. Pre-generate a settled-state library per build and reset via
   `set_particle_state()` (§1.3), not `shuffle + settle`.
4. Land the correctness fixes in §9 *before* collecting, since three of them
   change the physics of every transition.
5. Budget from the measured boundary: **4 envs at n=200** with today's
   settings, more once the contact cap is tightened.

---

## 9. Correctness fixes applied

All verified against the Genesis 0.4.5 source, and asserted end-to-end by
`Genesis/verify_fixes.py` (11/11 checks pass at n=200).

| # | Defect | Fix | Verified by |
|---|---|---|---|
| 1 | `RigidEntity.set_dofs_position` overrides the base signature with `zero_velocity=True`, and `zero_all_dofs_velocity` ignores `dofs_idx_local` — so the sweep loop, which meant to constrain z/roll/pitch/yaw, reset the plate's x/y velocity **every step**. | pass `zero_velocity=False` on that call | plate now retains x/y velocity through a sweep |
| 2 | Particles are built with no explicit material, so `rho=None` → `RHO_OBJECT=600`. `_set_particle_density_value` skipped `set_mass` entirely on the first call, then rescaled from 600 as if it were 750 — leaving **every mass at 0.8× its recorded density**. | seed `old_density` from Genesis' default instead of skipping | mass ratio 1.000 (was 0.800) |
| 3 | Plate friction never set → Genesis default 1.0, and contacts combine as `max(µa, µb)`, so sampled particle friction had **no effect at the tool interface**. | explicit `plate.friction` (config + material) | µ_plate = 0.3 |
| 4 | `dt` fallback was `4e3` — 4000 s. `substeps` fallback was 1. | `4e-3` / `5` | resolved values asserted |
| 5 | `max_collision_pairs` defaulted to a flat 150 regardless of pile size. | default now scales with `n_particles` | see §3 |
| 6 | `set_pos`/`set_quat` per particle → 2N kernel launches and 2N **whole-scene** forward-kinematics passes per reset and per snapshot restore (400 each at n=200; the restore path is oracle-MPC hot). | two batched solver calls + one FK | poses round-trip exactly (0.00e+00 error) |
| 7 | `set_n_active` parked every inactive particle at the **same point**, heaping them into one permanent contact cluster. | spread over a grid | 100 parked, 127 mm max separation (was 0) |
| 8 | `_particle_state_` allocated, never used. | removed | — |

### Fix 9 — the actuator model (resolves the consequence of fix 1)

Fix 1 exposed that neither the old nor the naively-fixed control law tracked the
commanded speed. Replaced with a gantry model: reflected inertia via armature,
gains from mass × bandwidth, a force limit, and a trapezoidal reference. See §5
for the measurements. Verified by `verify_fixes.py`: cruise speed **124.0 mm/s
against 125 commanded (0.99×)**, where the endpoint-target law could not track a
commanded speed at all by construction.

### Superseded note (kept for the record)

The plate was previously restarted from rest every step, and the sweep's step
count carries a `1.7×` fudge factor that was silently absorbing the resulting
speed deficit. With momentum retained, the plate now **overshoots**: measured
mean |v_xy| = **180 mm/s against a commanded 125 mm/s**.

That is inherent to the control law, not to the fix. `plate_velocity_translation`
sets the *endpoint* as the PD position target, so
`F = kp·(p_end − q) + kv·(v_target − v)` settles at
`v = v_target + kp·Δ/kv` — the plate runs fast when far from the goal and slows
as it approaches, rather than tracking a constant speed. With `kp=0.8, kv=1.0`
and Δ up to 0.09 m that predicts 0.197 m/s, which matches the measurement.

So the old behaviour was "sluggish and momentum-free" and the new one is
"overshoots by 44 %"; neither tracks the commanded 125 mm/s. Options:

1. **Ramp the position target along the path** (as `plate_position_translation`
   already does for the descent) instead of setting the endpoint, plus stiffer
   gains. This makes the tool follow a genuine Cartesian trajectory — closest
   to a robot arm executing a straight-line move, and the best match to the
   intent behind the current heavy-tool workaround.
2. **Retune `kv` upward** so `kp·Δ/kv` is small. Cheapest, keeps the servo
   structure.
3. Revert fix 1 and keep the compensating pair of errors.

This is a modelling decision about what the actuator represents, so I have not
made it unilaterally — see §5.

### Not applied (deliberately)

- **Batching `set_material_properties`' 2N per-particle calls.** Real, but it
  is per *material batch* (~2 s per build) against 400 ms per *step* — not
  worth the diff risk.
- **Removing the per-step `set_dofs_position` from the settle loop.** It is
  what holds the 2.4 g plate against gravity; dropping it would let the tool
  sag. Worth revisiting together with the actuator decision above, since it is
  both a large per-step cost (§2) and the reason contact-overflow errors are
  silent (§3).
- **`plate_position_translation`'s per-step `zero_velocity`.** That path
  teleports the plate along a scripted path each step, so velocity is
  meaningless there by construction; changing only this call would not make it
  physical.
