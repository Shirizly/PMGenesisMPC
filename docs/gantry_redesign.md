# Pusher plate redesign: from a pinned free body to a real gantry

**Status: IMPLEMENTED AND REJECTED.** Option B is built and available behind
`plate.hold_mode: servo`, but it fails the equivalence gate and is **not**
adopted; `pinned` remains the default. Both original motivations died under
measurement — hibernation on physics, the speedup on timing. Read §7b for the
bottom line before anything else. This is a living document. Every
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

1. Hibernation becomes usable. **The 57x figure is retracted — see §8; the real
   figure is 1.56x on the push.**
2. Solver errors become visible for the first time.
3. ~~The per-step collider/constraint reset disappears from the sweep, so the
   push may get cheaper even with hibernation off.~~ **Refuted by Q7: servo is
   15 % slower on the push and 48 % slower on the settle.**
4. One control path instead of two (kinematic teleport vs PD servo).
5. **The reaction report becomes trustworthy.** In pinned mode the descent
   reports 30.00 N at 100 % saturation, which is the teleport fighting a stale
   servo target, not load. Since the whole point of that report is to set
   real-robot limits, a phase that reports a permanent structural-limit hit is
   worse than no number at all.

Note one asymmetry worth remembering: servo mode *reduces* the largest island
(61 -> 48 entities) yet costs more per step. Island size counts links, and the
plate is one link either way — but in pinned mode its dofs are hard-set and so
effectively absent from the solve, whereas in servo mode all six participate. The
dense block is priced in DOFs, not entities.

---

## 4. Open questions — each resolved by a test

| # | Question | Why it could kill the design | Status |
|---|---|---|---|
| Q1 | Does `gs.morphs.MJCF` in 1.3.3 load `slide` + `hinge` joints and expose them as 4 controllable DOFs? | Whole design depends on it | **OPEN** |
| Q2 | Do `set_dofs_armature`, `set_dofs_kp/kv`, `set_dofs_force_range`, `control_dofs_position_velocity` work on prismatic DOFs of an MJCF chain? | The gantry actuator model must survive | **OPEN** |
| Q3 | With DOFs parsed depth-first (1.2.0), is the order really `[x, y, z, yaw]`? | Wrong order silently drives the wrong axis | **OPEN** |
| Q4 | Is a **one-off** `set_dofs_position` safe under hibernation, or does it NaN like the per-step case? | If even one-off writes fail, the approach-to-clearance teleport must become a PD move too, and hibernation may still be unreachable | **ANSWERED: one-off is safe, PD-driven is safe, only PER-STEP writes NaN. See §8. Changes the design — see option B.** |
| Q5 | With finite z / roll / pitch / yaw stiffness, how far does the blade actually deviate under a 200-cube push? | If it rides up over cubes or tilts, the tool no longer shears the pile at a controlled depth | **PASSED. n=200: tilt 0.0000°, tracking error 0.343 mm vs 0.344 pinned, final error 0.008 mm, reached_goal True.** |
| Q6 | Does a 4-DOF chain change contact behaviour vs a free body with 2 DOFs pinned? | Datasets would differ for a second reason beyond torsional friction | **OPEN** |
| Q7 | Is the push actually cheaper once the per-step reset is gone (hibernation off)? | The benefit in §3.3 is inferred from the settle, not measured for the sweep | **REFUTED. Servo is SLOWER: push 820.9 s vs 716.1 s (+15 %), settle 172.4 s vs 116.1 s (+48 %) at n=200.** |
| Q8 | Does hibernation then pass `probe_solver_equivalence.py`? | A speedup is worthless if the physics differs — the standing rule from §8.8 | **FAILED. n=100 broadside: penetration 1.95 vs 1.61 mm (+21 % worse), COM −3.1 %, zero variance over 4 replicates. Hibernation rejected.** |

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

**Not met — the design is rejected, see §7b.** The equivalence check is the one
that failed; the rest were not run once that was decided.

- [ ] `pytest tests/ -q` — 133 passing
- [ ] `verify_fixes.py` — 12/12, in particular plate cruise speed ≈124 mm/s and sweep tracking ≤0.01 mm
- [ ] `verify_new_features.py` — 10/10
- [ ] `probe_collection_health.py` at 50:16 and 200:1 — no flags
- [ ] `probe_solver_equivalence.py` — the new plate vs the old one, judged by the §8.8 criterion (noise floor, replicates, penetration first)
- [ ] `record_simulation_video.py` — visual check that the blade descends and sweeps plausibly
- [ ] **`_errno` reads 0 at every step of every phase** — the check that was impossible before
- [ ] a dry run resolves and completes

---

## 7b. Recommendation

Two defensible configurations:

| config | per transition at n=200 | correctness |
|---|---|---|
| `pinned`, no hibernation (today) | 832 s | errors unreportable; descent reports a false 30 N limit hit |
| `servo`, no hibernation | ~1060 s (1.28x slower) | correct |
| ~~`servo` + hibernation~~ | ~~708 s~~ | **REJECTED — Q8 failed** |

**Both rows lost.** Q8 killed hibernation on physics, and `servo` itself then
failed the same equivalence gate: particle-particle penetration 3–6x higher than
`pinned` at n=100, in both actions, with COM up to 20 % different. So the
recommendation is to **keep `pinned` and not adopt the redesign**. `hold_mode`
stays as a flag for future work.

One caveat worth keeping, because it cuts the other way: `pinned`'s low
penetration may partly be an artifact of `set_dofs_position` calling
`collider.reset()` every step, so contacts never persist long enough to
accumulate. That would mean `pinned` flatters itself on exactly the metric used
to judge it. Deciding which mode is *right* — rather than which differs — needs
a reference the simulator cannot provide, i.e. physical measurement of a real
push. Until then the conservative choice is the one every existing dataset was
collected under.

Superseded: ~~**`servo` buys correctness for ~28 % more time.**~~ Whether that is worth it
depends on whether a trustworthy reaction report and reportable solver errors
matter more than throughput at n=200, which is a judgement about the project,
not about the simulator.

What `servo` buys, concretely:

1. Solver errors become reportable at all. Under `pinned`, `set_dofs_position`
   clears `_errno` before every step and `Simulator.step` only reads it every
   second step, so NaN and buffer overflow during a push are structurally
   invisible. That is not hypothetical: it is what hid the hibernation NaN, and
   what made the 57x measurement look valid.
2. The reaction report stops lying about the descent. `pinned` reports 30.00 N at
   100 % saturation there — the teleport fighting a stale servo target, not load.
   `servo` reports 1.04 N at 0 %. Since the report exists to set real-robot
   limits, a phase that always claims a structural-limit hit is worse than no
   number.
3. It is closer to the machine: a leadscrew and a held rotary axis are stiff,
   not infinitely rigid.

What it costs: 15 % on the push, 48 % on the settle, ~28 % per transition.

---

## 8. Status log

Newest first. Every entry is a test result, including the ones that force a
design change.

- **FINAL: redesign rejected, `pinned` kept.** With hibernation rejected (Q8)
  and the speedup refuted (Q7), servo's only remaining argument was correctness
  — and it fails equivalence itself, by a wider margin than hibernation did.
  What the work leaves behind that is worth keeping: the standalone Genesis bug
  reproduction, the finding that `set_dofs_position` makes solver errors
  unreportable (and the audit showing no error was actually being hidden in the
  shipped config), the reaction report, and the retraction of the 57x.

- **Servo-vs-pinned equivalence FAILED, diagnosed to an implementation bug, fixed.**
  First run at n=100 showed servo differing on everything, worst of all
  particle-particle penetration: 0.47 -> 1.61 mm broadside and 0.45 -> 2.93 mm
  offset, up to 6.5x, plus COM -19.9 % on offset. That is far worse than
  hibernation managed and did not look like compliance, since blade tilt was
  still 0.0000 deg.

  Diagnosis: blade z **during the sweep** was identical in both modes
  (0.0175 m, min = mean = max), so it was not riding at the wrong depth. But the
  descent ended at 0.0196 m in servo mode against 0.0180 m pinned — 2.1 mm
  short — because a PD servo trailing a moving ramp lags by about `v*tau`
  (0.5 m/s x 10.6 ms ~ 5 mm, the right order). `plate_velocity_translation` then
  called `set_pos(p_start)`, teleporting the blade those 2.1 mm straight down
  into the pile in one step. The pile was being *punched*, not swept.

  Two fixes: hold the final descent target for `plate.arrival_steps` (12,
  ~4 time constants) so the servo actually arrives, and skip the start-of-sweep
  teleport in servo mode, where the descent has already placed the blade under
  its own actuator. The descent now converges to **0.069 mm** of target, better
  than pinned's 0.498 mm.

  **The fix did not rescue equivalence.** Re-run at n=100: penetration still
  0.47 -> 1.74 mm broadside and 0.47 -> 2.94 mm offset, essentially unchanged.
  Nor is it a gain-tuning problem — at n=50, pinned gives 0.366 mm against servo
  1.286 mm at 15/30 Hz, **2.341 mm at 60/120 Hz and 1.779 mm at 200/400 Hz**:
  stiffer is not better and the response is not even monotonic. The two modes
  simply produce different contact states.

  Worth noting the shape of this: the failure was not in the physics of holding
  the tool with a servo, which Q5 showed is fine. It was that replacing a
  teleport with an actuator makes *arrival* something that has to be waited for
  rather than assumed, and one leftover teleport then undid it.

- **Q8 FAILED — hibernation rejected.** With the plate in `servo` mode (the only
  mode where hibernation runs at all) and `errno` verified 0, judged by §8.8's
  criterion at n=100 over 4 replicates: `broadside` shows penetration **1.95 vs
  1.61 mm, 21 % worse**, and COM 3.76 vs 3.88 mm, −3.1 %. `offset` passes. Every
  hibernation metric has **exactly zero variance** across replicates — freezing
  bodies suppresses the chaotic divergence — so these are systematic, not noise.
  Penetration getting worse is the decisive one: it is the primary metric
  precisely because it describes the state rather than the trajectory, and
  bodies sinking further into each other is what excluding part of the contact
  graph from the solve does.

  Note this is the **same signature** as the 0.4.5 hibernation result (−4 %
  transport there, −3.1 % here). That was attributed to the PR #2930 wake-up
  bug; on 1.3.3 with the bug fixed the bias persists, so it was never only the
  bug. Against a 21 % penetration degradation, 1.56x on the push was never a
  strong enough reason.

- **RETRACTION: the 57x hibernation speedup was measured on a broken
  simulation.** `probe_contact_islands.py` measured hibernation in *pinned*
  mode, which was later proved to produce NaN constraint forces — and because
  `set_dofs_position` clears `_errno` every step, it never raised. Its 62 ms/step
  and largest-island-7 figures therefore came from a run silently producing NaN.
  Re-measured in servo mode with `errno` verified 0 at n=200: push 820.9 s
  without hibernation, 527.0 s with, i.e. **1.56x on the push**, and 1.18x per
  whole transition against the pinned baseline (832 s -> 708 s) because the
  settle gets slower. The island sizes explain it: in a *valid* simulation
  hibernation takes the largest island from **48 to 43** (10 %), not from 61 to 7
  (87 %). And (48/43)^2.64 = 1.34 against 1.56x measured, so the island^2.64 law
  from `scaling_to_200_objects.md` section 8.7 holds — hibernation simply has
  little to sleep, because the contact-connected neighbourhood of a ploughing
  blade is by definition awake. This is the general hazard of §1.1: any measurement taken
  while per-step DOF writes are suppressing the error flag may be measuring a
  simulation that has already failed.

- **Q5 PASSED, Q7 REFUTED — option B implemented behind `plate.hold_mode`.**
  `servo` mode gives, at n=200, `errno` 0 throughout, `reached_goal` True, final
  error 0.008 mm, tilt **0.0000°** and tracking error 0.343 mm against 0.344 mm
  pinned — so stiff servos hold the blade exactly as well as hard pinning, and
  Q5 is settled. But the hoped-for speedup is not there: servo is 15 % slower on
  the push and 48 % slower on the settle. Most likely because roll/pitch/yaw are
  now genuine dynamic DOFs inside the constraint solve rather than hard-set, so
  the plate contributes more to its island's dense Hessian — which is exactly
  what the island^2.64 law predicts.

  So the redesign's value is **correctness, not throughput**:
    1. solver errors become reportable at all (`errno` readable, verified 0),
    2. the descent stops saturating the actuator — 30.00 N at 100 % of steps
       becomes 1.04 N at 0 %, which was a pure control artifact,
    3. hibernation becomes usable (worth 1.56x on the push, not 57x).
  `hold_mode` defaults to `pinned` so nothing changes until this is decided.

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
