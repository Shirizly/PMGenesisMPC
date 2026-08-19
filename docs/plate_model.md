# The pusher plate: tool model, control, and reaction loads

How the tool is modelled and driven, and how to read the reaction loads it
reports. Describes the code as it stands.

Related: `scaling_to_200_objects.md` §1.2 covers why the actuator model looks the
way it does and what it replaced. A servo-held variant of the DOF handling in §2
was built, measured and rejected — see `rejected_plate_servo_redesign.md` for
why, so the same ground is not re-explored.

---

## 1. The tool

A single box entity, `plate.size` = 40 × 2 × 10 mm, `plate.friction` = 0.3. It is
a **free 6-DOF rigid body**, not a kinematic chain, and it weighs what it looks
like — about 2.4 g, the *lightest* dynamic object in the scene.

Friction must be set explicitly: unset, Genesis defaults it to 1.0, and contacts
combine as `max(µ_plate, µ_particle)`, which would mask the sampled particle
friction at the one interface the action acts through.

## 2. Actuator model

The real device is a 3D-printer-style Cartesian gantry: a heavy carriage driven
by steppers with far more force than the push needs, tracking a trapezoidal
velocity profile. So the quantity resisting granular reaction is the drivetrain's
**reflected inertia**, not the plate's own weight. All of it is config-driven
under `plate:` in `configs/basic.yaml`.

| knob | value | meaning |
|---|---|---|
| `moving_mass` | 0.5 kg | applied via `set_dofs_armature` on x/y/z — adds to the mass-matrix diagonal the constraint solver already uses, so contacts see a heavy axis while momentum exchange stays exact |
| `control_bandwidth_hz` | 15 | gains follow from mass and bandwidth: `kp = m·ω²`, `kv = 2ζmω` at ζ=1 → `kp` ≈ 4441 N/m |
| `max_force` | 30 N/axis | via `set_dofs_force_range`; a real stepper skips steps rather than applying unbounded force to a jam |
| `speed` | 0.125 m/s | trapezoid cruise |
| `acceleration` | 2.0 m/s² | trapezoid ramp |

**Which DOFs the action commands.** x and y are driven by the PD servo against a
trapezoidal position+velocity reference. z, roll and pitch are held fixed, and
yaw is held at the action's blade angle — constant within a push, varying between
pushes. Those four are held by `plate.hold_mode`:

* **`pinned`** (default) — overwritten every step with a partial
  `set_dofs_position`. Note `zero_velocity=False` is essential there:
  `RigidEntity.set_dofs_position` defaults it to `True` and zeroes *all six* DOFs
  regardless of `dofs_idx_local`, which would reset x/y velocity at 250 Hz and
  leave the tool carrying no momentum into the pile.
* `servo` — held by stiff PD instead. Implemented but **not recommended**; see
  `rejected_plate_servo_redesign.md`.

## 3. The three phases of a push

`execute_action` runs lower → sweep → lift, and exposes two hooks used by the
video recorder and the probes: `on_phase(phase)` at two boundaries, and
`on_step(phase, step)` on every step.

| phase | how it moves | steps |
|---|---|---|
| **lower** | teleport to clearance height, then `plate_position_translation` interpolates down with a per-step `set_pos` | `clearance_ctrl_steps` (25) |
| **sweep** | `plate_velocity_translation` feeds the servo a *moving* trapezoidal reference — position and velocity at the current time, not the endpoint | from the trapezoid's real duration, ~148 |
| **lift** | interpolate up, then teleport clear of the pile | 25 |

The sweep's reference must be a moving one. Commanding the endpoint turns the
same PD into a position servo whose speed is proportional to distance remaining,
which overshoots early and undershoots near the goal and never actually travels
at `plate.speed`.

`reached_goal` means **final tracking error < `goal_threshold`** (1 mm) — not
"reached at some point during the sweep". Measured across the real random action
distribution it is 100 % with a worst error of 0.01 mm.

## 4. Reaction loads: `reaction_report()`

Call it after any `execute_action`. It returns the peak loads on the tool during
that action, for setting real-robot limits from measurement rather than guesswork.

```python
sim.execute_action(p_start, p_stop, angle)
rep = sim.reaction_report()
rep["sweep"]["force_N"]         # peak |actuator force| on x/y/z
rep["sweep"]["torque_Nm"]       # peak |actuator torque| on roll/pitch/yaw
rep["sweep"]["contact_N"]       # peak net granular reaction on the blade
rep["sweep"]["track_mm"]        # worst deviation from the commanded path
rep["sweep"]["tilt_deg"]        # worst blade tilt away from vertical
rep["sweep"]["saturated_frac"]  # fraction of steps at plate.max_force
rep["force_limit_N"]            # the limit those are measured against
```

Costs **+1.8 %** per action: every quantity is a running max in GPU tensors, read
back exactly once at the end. A per-step `.item()` would reinstate the GPU sync
that §1.2 of the scaling doc removed.

### Read the `sweep` row, not `lower` or `lift`

The phases are not comparable, and this is not a detail. The descent and lift
drive the plate by teleport while its PD servo still holds an older target, so
the servo commands full force against its own motion. Measured:

| n | phase | actuator | granular reaction | saturated |
|---|---|---|---|---|
| 50 | lower | **30.00 N** | 0.000 N | **100 %** |
| 50 | **sweep** | **1.78 N** | 0.026 N | 0 % |
| 50 | lift | 22.98 N | 0.006 N | 0 % |
| 200 | lower | **30.00 N** | 0.024 N | **100 %** |
| 200 | **sweep** | **1.77 N** | 0.212 N | 0 % |
| 200 | lift | 22.98 N | 0.070 N | 0 % |

So a single figure across phases would report a machine permanently at its
structural limit, when the actual push load is 1.8 N. That 30 N is a control
artifact of the `pinned` teleport, not load.

### What the sweep numbers say

Pushing 200 objects costs the actuator **1.8 N and 0.001 Nm** against a granular
reaction of **0.21 N**, with tracking error 0.34 mm — well under one particle.
Actuator force is *identical* at 50 and 200 objects (1.78 vs 1.77 N) while the
reaction grows 8× (0.026 → 0.212 N), so the tool is nowhere near pile-limited:
roughly **17× headroom** against the 30 N budget.

## 5. Known limitation: solver errors are unreportable during a push

`RigidSolver.set_dofs_position` zeroes `_errno`, and `Simulator.step` only reads
it when `_cur_substep_global % RATE_CHECK_ERRNO == 0` with `RATE_CHECK_ERRNO = 10`
— every second step at `substeps: 5`. Because `pinned` clears it immediately
before each step, the error bit raised by step *N* is wiped before step *N+1*'s
check can read it, for the whole duration of a lower, sweep or lift.

The bits discarded are `OVERFLOW_COLLISION_PAIRS`, `OVERFLOW_CANDIDATE_CONTACTS`,
`OVERFLOW_CONTACTS`, `OVERFLOW_HIBERNATION_ISLANDS`, `INVALID_CONTACT_NAN`,
`INVALID_FORCE_NAN` and `INVALID_ACC_NAN`. This is why `contact_budget_usage()`
reads the collider counters directly instead of relying on Genesis to complain.

**Audited: nothing is actually being hidden in the shipped configuration.**
`_errno` was read after every step of every phase at n=50 and n=200 and was 0
throughout. But that is luck rather than detection, on two particle counts with
one action and one seed — so any *new* measurement should read `_errno` directly
rather than trust that an exception would have been raised. A measurement taken
while per-step DOF writes suppress the flag may be measuring a simulation that
has already failed; that is exactly how a 57× speedup came to be reported from a
run producing NaN.

## 6. Torsional friction

`enable_torsional_friction: True`. It resists spin about the contact normal,
which for a cube resting on the tray is the vertical axis — so what it really
resists is cubes twisting in place under an off-centre blade hit. The blade is
thin and strikes most particles off their centre of mass, so induced spin is a
large part of what a push does, and it was previously unresisted.

Measured at n=100 over 4 replicates against a noise floor of COM 0.04 mm /
displaced 6 mm: enabling it raises transport ~4 % (COM 3.84 → 4.01 mm broadside,
displaced 389.6 → 406.7 mm) and cuts peak penetration 23–30 % (0.61 → 0.47 mm),
with every replicate identical. Transitions are therefore **not comparable** to
any collected before it.

## 7. Where the plate is used

Only inside `Genesis/sandbox_manipulation_clean.py` (14 call sites).
`Genesis/tmp.py` and `Genesis/sandbox_manipulation_single_env.py` also carry a
plate but are the legacy single-env path. Nothing in `env/`, `simple_mpc/` or
`training/` touches it directly — they go through `execute_action` /
`push_and_record`.
