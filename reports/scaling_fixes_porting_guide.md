# Porting guide — the "scale to 200 objects" change set

Companion to `docs/scaling_to_200_objects.md`, which explains *why* each
change was made and shows the measurements. This document is the **what and
where**, organised so you can take the changes one at a time into another fork.

Everything was developed against **Genesis 0.4.5**, torch 2.11.0+cu130.

---

## 0. Scope at a glance

### Files modified (all inside `Genesis/`)

| file | +/− | contains |
|---|---|---|
| `Genesis/sandbox_manipulation_clean.py` | +268 / −71 | every code fix |
| `Genesis/configs/basic.yaml` | +3 | one new key (`plate.friction`) |

### Files added inside `Genesis/` — tooling only, imported by nothing

`benchmark_scaling.py`, `probe_step_cost.py`, `probe_contact_counts.py`,
`probe_plate_dynamics.py`, `verify_fixes.py`

### ⚠️ Files added OUTSIDE `Genesis/`

| path | what it is | must you port it? |
|---|---|---|
| **`docs/scaling_to_200_objects.md`** | the measurement report | no — documentation only |
| **`reports/scaling_fixes_porting_guide.md`** | this file | no |
| `outputs/scaling_benchmark/` | raw benchmark JSON/logs | no — and it is `.gitignore`d (`outputs/`) |

**No source file outside `Genesis/` was touched.** In particular
`env/genesis_env.py`, `simple_mpc/genesis_oracle.py`, `training/`, `model/`,
`transforms/` and `tests/` are untouched, and all 80 existing tests still pass.

`Genesis/parallelogram differentiable.py` shows as untracked but predates this
work — **not part of this change set.**

### Not done, and owed

Per the repo's documentation policy, `C11` below adds a new public method
(`contact_budget_usage()`) and `C5` changes a default, which should be
reflected in `docs/ARCHITECTURE.md` / `docs/INTERFACES.md`. **That has not been
written yet** — if you port `C11`, that doc update comes with it, and it is
outside `Genesis/`.

Also: sections 2, 3 and 5 of `scaling_to_200_objects.md` are still
placeholders. The probes that fill them (`probe_step_cost`,
`probe_contact_counts`, `probe_plate_dynamics`) were stopped before completing.

---

## 1. Suggested merge mechanics — probably simpler than hand-applying

You do not need to transcribe hunks. Two better routes, in order of preference:

### Route A — atomic commits → `git format-patch` → `git am -3` (recommended)

The whole change set lives in one working tree right now. Split it into one
commit per change ID below, then export:

```bash
# in THIS repo
git add -p Genesis/sandbox_manipulation_clean.py     # stage one change's hunks
git commit -m "fix(genesis): <change ID + summary>"  # repeat per change
git format-patch -o /tmp/scaling-patches <base-sha>
```

then in the fork, take only what you want, with real 3-way merge:

```bash
git am -3 /tmp/scaling-patches/0004-fix-genesis-....patch
```

`git am -3` will fall back to a proper merge (and leave conflict markers)
rather than refusing, which is what makes this work across diverged forks.
**I can do the commit-splitting for you** — say the word and I'll produce the
patch series without touching your current branch state beyond the commits.

### Route B — if the forks share history

`git cherry-pick <sha>` per commit. Same granularity, no patch files. Check
first whether they do:

```bash
git -C <fork> merge-base HEAD <this-repo-HEAD>
```

### Route C — wholesale file copy (only if safe)

Because only one real source file changed, copying
`Genesis/sandbox_manipulation_clean.py` wholesale is viable **iff the fork's
copy has not diverged**. Verify before trusting it:

```bash
diff <(git show HEAD:Genesis/sandbox_manipulation_clean.py) \
     <fork>/Genesis/sandbox_manipulation_clean.py
```

Empty diff → copy is safe and takes seconds. Non-empty → use Route A.

---

## 2. Prerequisites

These changes depend on Genesis API details. Check them in the fork's Genesis
version before porting:

| used by | API | notes |
|---|---|---|
| C4 | `set_dofs_position(..., zero_velocity=)` | `RigidEntity` overrides the base default to `True` |
| C8 | `rigid_solver.set_base_links_pos/quat(..., links_idx=, envs_idx=, skip_forward=)` | public solver API |
| C2 | `gs.materials.Rigid(friction=)` | field exists, defaults `None` |
| C3 | `RHO_OBJECT = 600.0` | `genesis/engine/entities/rigid_entity/rigid_link.py` |
| **C11** | `collider._collider_state.n_broad_pairs` / `.n_contacts`, `_collider_info.max_collision_pairs*`, `_collider_static_config.n_contacts_per_pair` | **private internals — the most version-fragile part of the set.** It is wrapped in `try/except` and degrades to a no-op, so a version mismatch is not fatal |

---

## 3. Change catalogue

Class key:
**S** straight fix (no intended behavioural difference) ·
**B** behaviour-changing fix (correct, but sim output changes) ·
**A** additive (only affects previously-failing or previously-unused cases) ·
**P** performance only · 
**T** tooling

All anchors are in `Genesis/sandbox_manipulation_clean.py` unless stated.

---

### C1 — `dt` / `substeps` fallbacks were wrong by 10⁶ · **S**

**Issue.** `_init_scene` read `.get('dt', 4e3)` — a fallback of **4000
seconds** — and `.get('substeps', 1)`. Latent, because every shipped config
sets both, but it silently produces nonsense for any config that doesn't.

**Fix.** `4e3` → `4e-3`, `1` → `5`.

**Anchor.** `_init_scene`, `gs.options.SimOptions(...)`.

**Risk.** None if your configs set `dt`. Port unconditionally.

---

### C2 — plate friction was pinned at 1.0, masking all sampled friction · **B**

**Issue.** The plate entity was created as `gs.materials.Rigid(rho=3000)` with
no `friction`. Genesis defaults an unset geom friction to **1.0**, and combines
a contact as **`max(µ_a, µ_b)`**. Since particle friction is sampled from
[0.05, 0.5], *every* plate–particle contact ran at µ=1.0 — so the sampled
friction had **no effect at the tool interface**, which is the one interface
the action acts through.

**Fix.** Pass `friction=float(self._plate_params.get("friction", 0.3))`, and
add the key to config.

**Anchors.** `_add_entities` (the `self.plate = ...` block);
`Genesis/configs/basic.yaml` under `plate:`.

**⚠ Behavioural impact.** Changes plate–particle friction from 1.0 → 0.3
everywhere. **Invalidates comparability with previously collected data.**

**⚠ Config note.** Only `basic.yaml` gets the explicit key. Every *other*
config (`chickpeas_on_glass.yaml`, etc.) picks up the **code default of 0.3**
silently — same behaviour change, no config edit. If you want those to keep
µ=1.0, set `plate.friction: 1.0` in them explicitly.

**Related, not fixed.** The same `max()` rule means `(µ_particle, µ_box)` — both
drawn from [0.05, 0.5] — collapses to one identifiable dimension, so roughly
half your friction conditioning grid is duplicate physics under different
labels. Worth re-parameterising the sweep; out of scope here.

---

### C3 — every particle mass was 0.8× its recorded density · **B**

**Issue.** Particles are constructed in `utilities/materials.py` with **no
`material=` argument**, so `rho=None`, which Genesis resolves at build time to
`RHO_OBJECT = 600.0`. `_set_particle_density_value` did:

```python
old_density = getattr(particle.material, "rho", None)
...
if is_built and old_density is not None and old_density > 0:
    particle.set_mass(get_mass() * density / old_density)
```

On the **first** call `old_density is None`, so `set_mass` was skipped
entirely — particles stayed at 600·V while the saved config recorded 750. Every
**subsequent** call then rescaled from 750 as the base, giving
`mass = 600·V·(ρ/750) = 0.8·ρ·V`.

**Fix.** Seed `old_density` from Genesis' default instead of skipping:
`getattr(...) or self._GENESIS_DEFAULT_RHO` (new class constant `600.0`), and
drop the now-redundant `is not None` guard.

**Anchor.** `_set_particle_density_value`.

**⚠ Behavioural impact.** All particle masses change by ×1.25. **Every
transition ever collected was recorded with masses 20 % below their label** —
this is the strongest argument for re-collecting rather than mixing datasets.

**Verified.** `verify_fixes.py` asserts mass ratio 1.000 (was 0.800).

---

### C4 — the plate's velocity was zeroed every simulation step · **B**

**Issue.** `RigidEntity.set_dofs_position` **overrides the base signature** with
`zero_velocity=True`, and the method it calls, `zero_all_dofs_velocity`,
ignores `dofs_idx_local` and zeroes **all six** DOFs. The sweep loop's call
intends only to constrain z/roll/pitch/yaw, but was resetting the plate's x/y
velocity at 250 Hz — the tool carried no momentum into the pile.

**Fix.** Pass `zero_velocity=False` on that one call.

**Anchor.** `plate_velocity_translation`, the per-step `set_dofs_position` with
`dofs_idx_local=self._horizontal_dofs_local`.

**⚠ Behavioural impact — read before porting.** This exposes a second problem
it was compensating for. `plate_velocity_translation` sets the sweep
*endpoint* as the PD position target, so the plate settles at
`v = v_target + kp·Δ/kv`: fast when far from the goal, slowing as it
approaches. With `kp=0.8, kv=1.0` and Δ up to 0.09 m that predicts 0.197 m/s,
and measurement gives **180 mm/s against a commanded 125 mm/s**. Meanwhile
`sweep_steps` carries a hard-coded `1.7×` fudge factor that was absorbing the
*old* deficit.

So: before = sluggish and momentum-free; after = overshoots by 44 %. Neither
tracks the commanded speed. **Porting C4 alone makes pushes ~44 % faster than
commanded.** Options, none of which are applied here:

1. Ramp the position target along the path (as `plate_position_translation`
   already does for the descent) with stiffer gains — a genuine Cartesian
   trajectory, closest to a robot arm holding the plate.
2. Raise `kv` so `kp·Δ/kv` becomes small — cheapest, keeps the servo structure.
3. Don't port C4; keep the two compensating errors.

`probe_plate_dynamics.py` (C13) runs both actuator models in one build and
reports realized speed, tracking error and granular reaction displacement, so
you can decide on measurements rather than argument.

**Not changed:** `plate_position_translation`'s per-step `zero_velocity`. That
path teleports the plate along a scripted path each step, so velocity is
meaningless there by construction.

---

### C5 — `max_collision_pairs` default did not scale with pile size · **B**

**Issue.** `_init_scene` used `rigid_cfg.get("max_collision_pairs", 150)`, and
no config overrides it. 150 is Genesis' own default and is independent of how
many bodies are in the scene.

Overflow is **not** a clean failure: the broadphase sets an error bit and stops
adding pairs, so remaining contacts are silently dropped and the step runs with
incomplete contact physics.

**Fix.** New method `_default_max_collision_pairs()` returning
`max(150, 4 * n_particles + 64)`, used as the fallback.

**Anchors.** new method before `_init_scene`; the `RigidOptions(...)` call.

**⚠ Behavioural impact.** Where contacts were previously being dropped, they
now aren't — that is the point, but it *is* a physics change. It also raises
VRAM: the constraint Jacobian is
`O(max_collision_pairs × contacts_per_pair × n_dofs × n_envs)`, so tying the cap
to `n` makes VRAM grow ~`n²` and reduces how many envs fit. Measured raw step
time is **independent** of the cap, so overshooting costs memory, not speed.

**⚠ The `4n + 64` coefficient is provisional.** It was chosen before the
measurement probe finished; at n=200 it produced 864 and the pile used a large
fraction of it. Treat it as a starting point and calibrate with
`probe_contact_counts.py` (C13) on your own particle size and shape.

---

### C6 — 150 and 200 particles could not be placed at all · **A**

**Issue.** `shuffle_particles` placed every particle in a **single layer** and
treats a free-yaw cube as an axis-aligned square of side `size·√2`. RSA
saturation in the 128 mm box is ~139 cubes at 5 mm and ~50 at 8.5 mm, so
`n=150`/`n=200` raised `RuntimeError` at *every* size in
`data_collection_clean.py`'s `PARTICLE_SIZES` — and `data_collection_clean.py`
catches that and **silently skips the batch**.

**Fix.** Introduce `n_layers` (starts at 1). Split the largest-first placement
order across layers with a stride, pack each layer independently against its
own `placed` mask, and give each layer a `z` offset of `layer_pitch =
2·max_half_height + min_gap`. Raise a descriptive error if the stack would
exceed the wall top. The retry handler increments `n_layers` every second
attempt rather than retrying the same geometry.

**Anchors.** `shuffle_particles`: the `n_layers = 1` declaration before the
retry loop; the placement block; the `except RuntimeError` handler.

**Behavioural impact.** **None when a monolayer fits** — with `n_layers == 1`
the strided split is the identity and the loop is equivalent to the original.
Only previously-failing configurations change.

**⚠ Coupling.** This hunk and `C7`/`C8` are adjacent inside
`shuffle_particles`. If you port them separately, expect to resolve overlap
manually.

**⚠ Note.** Stacked layers are *dropped*, not interpenetrating — the caller's
subsequent `update_material_state()` settle collapses them into a pile. So a
2-layer spawn needs enough `settle_steps` to come to rest; the fixed 100-step
settle was never validated for that.

---

### C7 — parked particles were all heaped on one point · **A**

**Issue.** `set_n_active(n)` parks particles `[n, N)` outside the box, but at
an **identical** position — stacking e.g. 150 cubes at one spot, creating a
permanent contact cluster that consumes the contact budget and costs solver
time on every step of every env.

**Fix.** Lay them out on a grid with pitch `2·max_half_extent + 5 mm`.

**Anchor.** `shuffle_particles`, the `if n_active < n_particles:` block.

**Behavioural impact.** Only affects callers of `set_n_active` — currently just
`env/genesis_env.py:348`. **Port this if you intend to use `set_n_active` to
sweep `n` without rebuilding**, which is the cheap way to do an `n` sweep (one
build at n=200 serves 50/70/100/150/200; `particle_size` still needs a rebuild).

**Verified.** 100 parked particles, 127 mm max separation (was 0).

---

### C8 — per-particle pose writes cost 2N whole-scene FK passes · **P**

**Issue.** `RigidEntity.set_pos`/`set_quat` default to `skip_forward=False`,
which runs forward kinematics over the **entire scene**. The per-particle loops
in `shuffle_particles` and `set_particle_state` therefore cost 2N kernel
launches *and* 2N full-scene FK passes — 400 of each at n=200, on every reset
and every snapshot restore. `set_particle_state` is in the **oracle-MPC hot
path** (`simple_mpc/genesis_oracle.py:316`).

**Fix.** New `_write_particle_poses(pos, quat, envs_idx)` using
`rigid_solver.set_base_links_pos(..., skip_forward=True)` then
`set_base_links_quat(..., skip_forward=False)` — two launches, one FK. New
`_random_particle_quats_batched()` builds all quats as one `(n_envs, N, 4)`
tensor. The old per-particle `_random_particle_quats` is deleted.

**Anchors.** two new methods where `_random_particle_quats` used to be; call
sites in `shuffle_particles` and `set_particle_state`.

**⚠ Subtle behavioural impact.** The batched quaternion draw consumes the RNG
stream differently from N per-particle draws, so **spawn orientations differ
for the same seed**. Statistically identical, not bit-identical — don't expect
seed-for-seed reproduction against old runs.

**Verified.** Poses round-trip at 0.00e+00 max position error.

---

### C9 — redundant `get_pos()` calls in the settle path · **P**

**Issue.** `update_material_state` called `self.plate.get_pos()` three times in
four lines.

**Fix.** Hoist to one `plate_pos`.

**Anchor.** `update_material_state`, "Hold plate still" block.

**Risk.** None.

---

### C10 — dead allocation · **S**

**Issue.** `self._particle_state_` was allocated in `__init__` and never read
or written anywhere in the repo.

**Fix.** Removed.

**Anchor.** `__init__`, next to `self._particle_state`.

**Risk.** None — grep-confirmed no readers.

---

### C11 — contact-budget overflow is undetectable here · **A (new feature)**

**Issue.** Genesis surfaces contact overflow via an error bit checked at the
start of `Simulator.step`, only when `substep_global % 10 == 0`. But
`RigidSolver.set_dofs_position` **clears that error bit as a side effect**, and
both the settle loop and the sweep loop call it on *every* step. The bit is
therefore always wiped before any check reads it — **an overflow in this
codebase is completely silent.** Contacts dropped, wrong physics recorded, no
exception.

**Fix.** Two new methods:

- `contact_budget_usage()` — **new public method** returning peak
  `broad_pairs` / `contact_points` and their real caps. Note these have
  *different* caps: broad-phase pairs are capped at `mcp × 8`, contact points
  at `mcp × n_contacts_per_pair` (5 normally, **16** with `box_box_detection`
  on and >1 box, which is this scene). Comparing the wrong pair is easy to get
  wrong — I did, initially.
- `_check_contact_budget()` — warns once per instance at ≥90 % of either cap;
  called at the end of `update_material_state`.

**Anchors.** both methods after `update_material_state`; one call inside it.

**Cost.** Three device→host reads per `update_material_state` until the warning
fires (then it self-disables). Negligible against 100 settle steps, but it *is*
a sync — if you have a latency-critical path, gate it.

**⚠ Version fragility.** Reads Genesis private collider internals. Wrapped in
`try/except` that self-disables on failure, so a version mismatch degrades to a
no-op rather than crashing.

**⚠ Doc debt.** New public method → owes an `docs/INTERFACES.md` /
`docs/ARCHITECTURE.md` entry (outside `Genesis/`).

---

### C12 — deliberately NOT changed

| candidate | why not |
|---|---|
| Batch `set_material_properties`' 2N per-particle `set_friction`/`set_mass` calls | real, but per *material batch* (~2 s/build) vs 400 ms per *step*. Not worth the diff risk. Batched APIs exist: `set_geoms_friction`, `set_links_inertia` |
| Remove the per-step `set_dofs_position` from the settle loop | it is what holds the 2.4 g plate against gravity; removing it lets the tool sag. This is both a large per-step cost and the reason C11 exists — revisit together with the C4 actuator decision |
| `performance_mode: False` | disables kernel shape-specialisation; would turn ~25 recompiles into ~1 across a sweep, at a few % runtime. Untested here |
| Yaw-exact AABB instead of the √2 worst case in placement | ~23 % more packing capacity *and* more correct, but changes placement statistics |
| Velocity-threshold settling instead of fixed 100 steps | likely both faster and more correct at n≥150, but changes every recorded `s'` |

---

### C13 — new tooling · **T** · zero effect on existing code

All in `Genesis/`, all standalone (`python -m Genesis.<name>` from repo root),
none imported by library code. Port only if useful.

| file | purpose |
|---|---|
| `verify_fixes.py` | asserts C1–C8, C11 against a live scene. **Port this with any subset of the fixes** — it is how you confirm the port landed |
| `benchmark_scaling.py` | `(n_particles, n_envs)` cost + VRAM sweep, subprocess per cell so an OOM doesn't abort the run |
| `probe_step_cost.py` | isolates raw `scene.step()` from the sweep loop's control/sync overhead; sweeps `max_collision_pairs` and `box_box_detection` |
| `probe_contact_counts.py` | measures real collider occupancy → calibrates C5's coefficient |
| `probe_plate_dynamics.py` | runs both actuator models (C4 on/off) and reports tracking error vs granular reaction displacement |

---

## 4. Suggested ordering

If you want the safest possible subset first:

1. **Port with no hesitation:** C1, C9, C10 (no behavioural effect), plus
   `verify_fixes.py`.
2. **Port if you are re-collecting data anyway:** C3, C2 — both are outright
   correctness bugs, both invalidate comparability with existing datasets.
3. **Port if you want >100 particles:** C6, then C5 (calibrate with
   `probe_contact_counts.py`), then C7 if you'll use `set_n_active`.
4. **Port for speed:** C8 (accepting the RNG-stream note).
5. **Port for safety:** C11.
6. **Decide first:** C4 — it is correct in isolation but needs a companion
   actuator decision, or pushes run 44 % fast.

C3 and C2 together mean **any dataset collected before them is not comparable
with one collected after**. If you are going to take them, take them before
starting a large collection run, not during.
