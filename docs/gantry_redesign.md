# Pusher plate redesign: from a pinned free body to a real gantry

**Status: DESIGN — nothing implemented yet.** This is a living document. Every
open question below is resolved by a test, and the answer is written back here,
including answers that kill part of the design. When the design passes the
acceptance checks in §7 this document describes what was built, not what was
intended.

---

## 1. Why

The plate is currently a **free 6-DOF rigid body** whose unwanted degrees of
freedom are overwritten *every simulation step* with `set_dofs_position`. That
one decision causes three separate, measured harms.

### 1.1 It silently discards solver errors

`RigidSolver.set_dofs_position` zeroes `_errno`. `Simulator.step` only inspects
`_errno` when `_cur_substep_global % RATE_CHECK_ERRNO == 0` with
`RATE_CHECK_ERRNO = 10`, and at `substeps: 5` that is the start of every
**second** step. Our loops clear it immediately before each step, so the error
bit raised by step *N*'s physics is wiped before step *N+1*'s check can read it —
every time, for the whole duration of a descent, sweep or lift.

The bits being discarded are not cosmetic: `OVERFLOW_COLLISION_PAIRS`,
`OVERFLOW_CANDIDATE_CONTACTS`, `OVERFLOW_CONTACTS`,
`OVERFLOW_HIBERNATION_ISLANDS`, `INVALID_CONTACT_NAN`, `INVALID_FORCE_NAN`,
`INVALID_ACC_NAN`. This is why `contact_budget_usage()` exists at all — it reads
the collider counters directly, because Genesis' own overflow reporting could
never reach us.

### 1.2 It blocks hibernation, which is worth 57x

With `use_hibernation=True` the largest contact island during a push falls from
61 entities to 7 and step cost falls from 3522 ms to 62 ms. But driving a body
with `set_dofs_position` under hibernation produces `nan` constraint forces the
moment it touches a hibernated body — reduced to a standalone two-cube
reproduction in `tests/scaling_investigation/repro_hibernation_nan.py`, where the
identical motion driven by `set_pos` is clean. Partial vs full DOF write makes no
difference; so do waking every step, dropping `zero_velocity`, quartering the
step size, removing the actuator model, and raising `max_collision_pairs`.

### 1.3 It resets the constraint solver's warm start every step

`set_dofs_position` calls `collider.reset()` and `constraint_solver.reset()` and
runs a whole-scene forward-kinematics pass. Removing exactly this from the
*settle* loop (§1.3 of `scaling_to_200_objects.md`) dropped the worst residual
particle from 25.1 to 1.0 mm/s **and** cost 25 % less per step. The sweep still
pays that price on every one of its ~132 steps.

### 1.4 It is not what the machine is

The real device is a Cartesian gantry: three driven linear axes plus a rotary
axis for blade yaw. It has no roll or pitch to pin, and its Z is a leadscrew, not
a constraint re-imposed 250 times a second. The current model invents two DOFs
and then fights them.

---

## 2. Current model, precisely

| aspect | how it works now |
|---|---|
| entity | `scene.add_entity(gs.morphs.Box(size=plate.size))` → free body, 6 DOFs |
| x, y | PD via `control_dofs_position_velocity` on dofs `[0,1,2]`, with `set_dofs_armature(moving_mass)`, `kp = m·ω²`, `kv = 2mω`, `set_dofs_force_range(±max_force)` |
| z | pinned every step to `_operation_height` |
| roll, pitch | pinned every step to 0 |
| yaw | pinned every step to the action's blade angle (constant within a push, varies between pushes) |
| descent / lift | `plate_position_translation`: per-step `set_pos` along an interpolated path **plus** per-step partial `set_dofs_position` |
| sweep | `plate_velocity_translation`: trapezoidal position+velocity reference to the PD servo, **plus** per-step partial `set_dofs_position(zero_velocity=False)` |
| teleports | `set_pos(..., zero_velocity=True)` to clearance height before descent and away after lift |

Note yaw is **not** constant across the dataset, so the redesign needs a real
rotary DOF, not a rigid orientation.

---

## 3. Proposed model

Q4's answer changes this section. The problem is **per-step DOF writes**, not the
free-body model as such: a one-off `set_dofs_position` and a PD-driven move are
both clean under hibernation. Two designs therefore reach the same goal, and the
cheaper one is now preferred.

### Option B — hold orientation with a stiff servo (PREFERRED)

Keep the plate as a free 6-DOF box. Replace the per-step
`set_dofs_position` on z/roll/pitch/yaw with **PD gains on those DOFs**, so they
are *held by servos* rather than *overwritten by fiat*:

* translation x/y/z: PD as today, armature `moving_mass`, `force_range ±max_force`
* roll/pitch: stiff PD to 0
* yaw: stiff PD to the action's blade angle
* descent / lift: PD move on z (no per-step write)
* approach to clearance and park: a single `set_dofs_position` or `set_pos`

Why prefer it over option A: no MJCF asset, no dependence on Genesis' joint
parsing (Q1, Q2), no exposure to depth-first DOF reordering (Q3), and no change
to how the entity is created — so the 14 call sites and every downstream caller
keep working. It achieves all four benefits below for a fraction of the risk.

What it gives up: roll and pitch remain *finite-stiffness* rather than
nonexistent, so a cube can in principle tilt the blade. That is the same
trade already accepted for z and yaw, and Q5 measures it. If Q5 shows
unacceptable tilt that stiff gains cannot fix, fall back to option A.

### Option A — a real kinematic chain (fallback)

A **4-DOF kinematic chain**, defined as MJCF and loaded with `gs.morphs.MJCF`:

```
world
 └─ slide x   (prismatic, along X)
     └─ slide y   (prismatic, along Y)
         └─ slide z   (prismatic, along Z)
             └─ hinge yaw   (revolute, about Z)
                 └─ blade body  (box geom, plate.size, plate.friction)
```

Roll and pitch do not exist, so nothing pins them. Every commanded motion becomes
a normal actuated-joint move:

| phase | now | proposed |
|---|---|---|
| approach to clearance | `set_pos` teleport | one-off `set_dofs_position` on 4 DOFs (**see Q4**) |
| descent | per-step `set_pos` + partial write | PD move on the z DOF |
| sweep | PD on x/y + per-step partial write | PD on x/y, z and yaw simply *held by their own servos* |
| lift | per-step `set_pos` + partial write | PD move on the z DOF |
| settle | plate PD target set once (already correct) | unchanged |

Actuator parameters carry over unchanged in spirit: `armature = moving_mass` on
the three linear DOFs, gains from mass and bandwidth, `force_range = ±max_force`.
The blade *body* keeps its true ~2.4 g mass; the carriage inertia stays in the
armature, exactly as today.

Z and yaw need their own gains. The physical justification is that a leadscrew
and a held stepper are stiff and effectively non-backdrivable, so both should be
stiff enough that granular reaction moves them far less than a particle
diameter — but they are now *finite*, which is a behaviour change: a cube can in
principle lift the blade or twist it. **That is more faithful, and it is a
change; see Q5.**

### Expected benefits

1. Hibernation becomes usable → up to 57x on pushes (to be re-measured).
2. Solver errors become visible for the first time.
3. The per-step collider/constraint reset disappears from the sweep. §1.3
   measured 25 % saved per step when this was removed from the settle, so the
   push may get materially cheaper **even with hibernation off**.
4. One control path instead of two (kinematic teleport vs PD servo).

---

## 4. Open questions — each resolved by a test

| # | Question | Why it could kill the design | Status |
|---|---|---|---|
| Q1 | Does `gs.morphs.MJCF` in 1.3.3 load `slide` + `hinge` joints and expose them as 4 controllable DOFs? | Whole design depends on it | **OPEN** |
| Q2 | Do `set_dofs_armature`, `set_dofs_kp/kv`, `set_dofs_force_range`, `control_dofs_position_velocity` work on prismatic DOFs of an MJCF chain? | The gantry actuator model must survive | **OPEN** |
| Q3 | With DOFs parsed depth-first (1.2.0), is the order really `[x, y, z, yaw]`? | Wrong order silently drives the wrong axis | **OPEN** |
| Q4 | Is a **one-off** `set_dofs_position` safe under hibernation, or does it NaN like the per-step case? | If even one-off writes fail, the approach-to-clearance teleport must become a PD move too, and hibernation may still be unreachable | **ANSWERED: one-off is safe, PD-driven is safe, only PER-STEP writes NaN. See §8. Changes the design — see option B.** |
| Q5 | With finite z / roll / pitch / yaw stiffness, how far does the blade actually deviate under a 200-cube push? | If it rides up over cubes or tilts, the tool no longer shears the pile at a controlled depth | **OPEN — now the gating question, and the reaction report (§9) is the instrument for it** |
| Q6 | Does a 4-DOF chain change contact behaviour vs a free body with 2 DOFs pinned? | Datasets would differ for a second reason beyond torsional friction | **OPEN** |
| Q7 | Is the push actually cheaper once the per-step reset is gone (hibernation off)? | The benefit in §3.3 is inferred from the settle, not measured for the sweep | **OPEN** |
| Q8 | Does hibernation then pass `probe_solver_equivalence.py`? | A 57x speedup is worthless if the physics differs — the standing rule from §8.8 | **OPEN** |

Q4 is the one to test first, because it is cheap and it gates the value of
everything else.

---

## 5. Compatibility

The plate is referenced **only** inside `Genesis/sandbox_manipulation_clean.py`
(14 call sites). `Genesis/tmp.py` and `Genesis/sandbox_manipulation_single_env.py`
also touch a plate but are the legacy single-env path and are out of scope.
Nothing in `env/`, `simple_mpc/` or `training/` reaches the plate directly — they
go through `execute_action` / `push_and_record`, whose signatures do not change.

Things that must keep working unchanged:

- `execute_action(p_start, p_stop, angle, on_phase=, on_step=)` — same signature
  and same meaning of `reached_goal` (final tracking error < `goal_threshold`).
- `_operation_height` and `_clearance_offset` semantics, recomputed against the
  gantry's zero rather than a world-frame box pose.
- The `on_phase` / `on_step` hooks, used by the video recorder and probes.
- The state library, which touches particles only.

---

## 6. Data comparability

This changes tool dynamics, so transitions will not be comparable to those
collected before it — on top of the torsional-friction change already made on
this branch. Both should land before any long collection, not after.

---

## 7. Acceptance checks

The design is done when all of these pass on 1.3.3:

- [ ] `pytest tests/ -q` — 133 passing
- [ ] `verify_fixes.py` — 12/12, in particular plate cruise speed ≈124 mm/s and sweep tracking ≤0.01 mm
- [ ] `verify_new_features.py` — 10/10
- [ ] `probe_collection_health.py` at 50:16 and 200:1 — no flags
- [ ] `probe_solver_equivalence.py` — the new plate vs the old one, judged by the §8.8 criterion (noise floor, replicates, penetration first)
- [ ] `record_simulation_video.py` — visual check that the blade descends and sweeps plausibly
- [ ] **`_errno` reads 0 at every step of every phase** — the check that was impossible before
- [ ] a dry run resolves and completes

---

## 8. Status log

Newest first. Every entry is a test result, including the ones that force a
design change.

- **Q4 ANSWERED — design changed.** Standalone 2-cube scene, hibernation on,
  identical descent: per-step `set_dofs_position` → NaN; **one-off**
  `set_dofs_position` → clean; **PD-driven** descent with no per-step write →
  clean. So the defect is specifically per-step DOF writes. Added option B
  (stiff PD on the unwanted DOFs, free body retained) and made it preferred:
  it drops Q1, Q2 and Q3 entirely, since there is no MJCF asset and no joint
  parsing involved. Q5 is now the gating question — with roll/pitch/yaw/z held
  by servos instead of pinned, how far does the blade actually deviate? The
  reaction report (§9) is being built as the instrument to answer it.
- *(start)* — document created; Q1–Q8 all open; Q4 to be tested first.
