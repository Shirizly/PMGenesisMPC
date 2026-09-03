# Pile manipulation — design

What this project is, how the pieces fit, and how to run them. Describes the
code as it stands.

Related documents:

| | |
|---|---|
| [`PORT_NOTES.md`](PORT_NOTES.md) | why the simulator behaves as it does — every non-obvious default, with the measurement behind it |
| [`docs/plate_model.md`](plate_model.md) | the tool's actuator model in detail |
| [`Genesis/layered/README.md`](../Genesis/layered/README.md) | stacked spawn, and when you need it |
| [`MPC/README.md`](../MPC/README.md) | the MPC stack's layout and how it reaches the simulator |
| [`MPC/docs/ARCHITECTURE.md`](../MPC/docs/ARCHITECTURE.md) | model/training/registry design |
| [`tests/benchmarks/README.md`](../tests/benchmarks/README.md) | measuring anything on your own machine |

---

## 1. What the project does

A rigid-body simulation of a **flat tool pushing a pile of small objects** in a
square tray, used to generate transition data `(s, a, s')` for learning
dynamics models, and an MPC stack that plans pushes with those models.

The tool is a thin blade on a Cartesian gantry. One **action** is a push:
descend at a chosen `(x, y, yaw)`, sweep in a straight line to a stop point,
lift. One **transition** is the pile state before and after, recorded once the
pile has come to rest.

Three parts, in dependency order:

```
Genesis/          the simulator and data collection      (no dependencies)
  layered/          stacked-spawn variant, self-contained
GranularDynamics2/  U-Net models and training            (upstream's)
MPC/              MPC / world-model research stack       (imports Genesis/)
dino_wm/          DINO world model                       (submodule)
le-wm/            LeWM world model                       (submodule)
```

`Genesis/` is **not a Python package** — no `__init__.py`, and its modules
import each other flatly (`from utilities.materials import *`). Scripts there
run from inside that directory. `MPC/` follows the same convention for itself
and puts `Genesis/` on `sys.path` via `MPC/genesis_path.py`.

---

## 2. The simulator

`Genesis/sandbox_manipulation.py` — one class, `SandboxManipulation`, wrapping
a Genesis scene. Genesis is pinned to **1.3.3**.

### The scene

A square tray (walls + floor, all static), N particles, and the plate. Every
parallel environment shares one scene; `n_envs` copies step in lockstep.

**The tray is a monolayer by construction.** `__init__` overwrites
`box.vol[2]` with `wall_thickness + max_particle_height(...)`, sized so a
resting single layer never rises above the walls — which is what makes the
top-down camera's view of the pile complete. Stacking particles therefore needs
a taller tray and a different placer; that is `Genesis/layered/`, and you
usually do not need it (a 128 mm tray holds 225 cubes of 5 mm in one layer).

**Particles are created before `scene.build()`**, so `n_particles`,
`particle_size`, `shape` and every `rigid_options` field are rebuild-only
parameters. `set_n_active(n)` avoids a rebuild when only the count changes, by
placing a prefix and parking the rest outside the tray. Friction, density and
rolling friction change in place via `set_material_properties()`.

### One action

`execute_action(p_start, p_stop, angle)` runs three phases:

1. **lower** — teleport to a clearance height above the operating height, then
   simulate the short final descent. `plate.approach_mode` decides whether that
   descent is driven by the actuator (`servo`, the default, so particles can
   resist it) or written pose-by-pose (`teleport`).
2. **sweep** — track a **trapezoidal** position + velocity reference to the
   stop point. The uncommanded DOFs (z, roll, pitch, yaw) are held by
   `plate.hold_mode`.
3. **lift** — the short ascent, then teleport clear.

`update_material_state()` then settles the pile and records the state.
`settle_steps` is a **cap** with a velocity-convergence exit, not a fixed count.

### The plate

Modelled as a gantry axis rather than a free box: reflected drivetrain inertia
via `set_dofs_armature`, gains derived from mass and bandwidth
(`kp = mω²`, `kv = 2ζmω`), and a bounded force range. Details and the reasoning
in [`docs/plate_model.md`](plate_model.md).

### Safety checks that run on their own

- **Contact budget.** Genesis signals contact-buffer overflow with an error bit
  that this scene's per-step `set_dofs_position` clears before anything can read
  it, so overflow would be silent. `_check_contact_budget()` reads the collider
  counters directly at the end of every sweep, where usage peaks, and warns
  above 90 % of either cap.
- **Packing fraction.** Checked at construction, before a build is paid for:
  past ~70 % of a layer's placement capacity the tool runs out of anywhere to
  touch down.
- **Escaped particles.** Counted per batch. A particle outside the tray means
  the contact solver failed for it, and since each transition's `s` is the
  previous `s'`, every later sample in that env is suspect.
- **Settle cap hit.** Warns loudly — the recorded state is mid-motion.

---

## 3. Collecting data

```bash
cd Genesis
python data_collection.py --num-particles 50 --particle-sizes 0.005 \
    --n-envs 4 --samples-per-env 5 --seed 0
```

Runs from inside `Genesis/`. Output goes to
`<output-root>/<shape>/n<N>/size<S>/`, one set of files per material batch:

| file | contents |
|---|---|
| `_<i>_data.pt` | successful transitions: `states`, `states_`, `p_starts`, `p_stops`, `angles` |
| `_<i>_failed.pt` | same keys, for pushes that did not reach their goal |
| `_<i>_rollout.pt` | every push, shaped `(n_envs, n_samples, …)`, plus `success_mask` and `frames` if rendering |
| `_<i>_config.yaml` | the full simulator config, the seed, and the audit statistics |
| `settled_states.pt` | the state library, if one was built |

`states` is `float32 [N, n_particles, 7]` — position + quaternion.

### Every flag

**What to simulate**

| flag | default | meaning |
|---|---|---|
| `--settings` | `basic` | config name under `Genesis/configs/` |
| `--particle-shape` | `cube` | `cube`, `sphere`, `cylinder`, `rectangle`, or `config` |
| `--num-particles` | `50` | one or more counts; **each is a separate scene build** |
| `--particle-sizes` | `0.005` | one or more sizes, metres; each is a separate build |
| `--n-envs` | `10` | parallel environments |
| `--samples-per-env` | `5` | pushes per env per batch |
| `--seed` | `None` | seeds **both** numpy and torch; recorded in the saved config |

**How pushes are chosen**

| flag | default | meaning |
|---|---|---|
| `--start-sampling` | `auto` | `auto`, `uniform`, `density`, `free`, `composed` — see below |
| `--center-bias` | `0.0` | >0 pulls each push's stop toward the tray centre by a random fraction in `[0, center_bias]`, producing gathering demonstrations |
| `--shared-travel-distance` | off | one push *length* per batch; each env keeps its own start, direction and yaw |

`--start-sampling` picks the touchdown pose. The two mechanisms are
complementary: density-weighted answers *where is there material worth
pushing*, free-space answers *where can the tool actually come down*.

| value | behaviour | touchdown overlap |
|---|---|---|
| `auto` | density-weighted when particle positions are available, else uniform | 95 % |
| `uniform` | uniform over the legal box | 89 % |
| `density` | density-weighted; forced | 95 % |
| `free` | drawn from the tool's free configuration space | **16 %** |
| `composed` | density picks the neighbourhood, then the pose moves the shortest distance that makes it collision-free | **28 %**, keeping density's spatial spread |

*Overlap* is the fraction of touchdowns whose blade footprint contains a
particle centre. Descending **into** a particle is resolved by the solver
ejecting it — an artifact recorded as though it were a push. `free` and
`composed` fall back per sample when the tray is too full for any legal pose,
so they are refinements, not guarantees.

**Resetting**

| flag | default | meaning |
|---|---|---|
| `--state-library N` | `0` | settle N piles once per build, expand by the tray's symmetry group, and reset by restoring instead of re-settling |
| `--state-library-damping` | `0.0` | temporary viscous damping during those settles only — a convergence aid, never applied to the post-push settle |

`shuffle_particles()` runs zero simulation steps, so all of a reset's cost is
the settle after it. A square tray admits the full dihedral group **D4**, so
each settle yields 8 variants, and each settle randomises every env
independently: the bank is `N × n_envs × 8` states for the cost of `N` settles.
All envs in a batch share one drawn state, without replacement.

**Output and rendering**

| flag | default | meaning |
|---|---|---|
| `--output-root` | `data/corl` | relative to `Genesis/` |
| `--render-images` / `--no-render-images` | on | one RGB frame per env per snapshot, into `_rollout.pt` |
| `--render-resolution W H` | `128 128` | |
| `--viewer-type` | `None` | `observer`, `bird`, `leveled` — opens a live window |
| `--debug` | off | verbose, and draws debug frames |

**DINO-WM export**

| flag | default | meaning |
|---|---|---|
| `--export-dino-wm` | off | also write DINO-WM format after each build |
| `--dino-wm-output-root` | `<output-root>_dino_wm` | |
| `--dino-wm-obs-types` | `occupancy rendered` | `rendered` requires `--render-images` |
| `--dino-wm-resolution-scale` | `1.0` | occupancy grid pixels per mm |

### Sweeping several pile sizes

Each particle count needs its own build, so a sweep is one subprocess per size:

```bash
cd Genesis
python run_collection.py --plan configs/collection_dry_run.yaml --preflight-only
python run_collection.py --plan configs/collection_dry_run.yaml
```

**Preflight** checks placement feasibility against the same capacity the
simulator enforces (so "fits" means the reshuffle will not fail on batch two)
and free VRAM. Infeasible counts are skipped with a reason instead of failing
two minutes into a build.

**Postflight** validates what was written: shapes, dtype, non-finite values,
plausible displacement, the recorded audit fields, the state library's size,
and — above all — that **`s'` differs from `s`**. A run that "succeeds" while
nothing moves is the silent failure worth guarding against. Exits nonzero if
any size failed, and writes `run_collection_report.json`.

### Stacked spawns

For counts that will not fit one layer, `Genesis/layered/` is a self-contained
copy with a layered placer and its own config. It is rarely needed — see its
[README](../Genesis/layered/README.md) for the measured per-layer ceilings and the
two costs (partial observability, and material above the blade's reach).

---

## 4. Configuration

`Genesis/configs/basic.yaml` is the only config the current code loads; the
others are stale (older nested schema) and raise. Every value below is read;
anything a key does not cover is a code default.

Values marked `SET IN LOOP` are overwritten by `data_collection.py`.

```yaml
simulation:
  dt: 4.0e-3                      # seconds per step
  substeps: 5
  backend: gpu                    # gpu | cpu — falls back to cpu SILENTLY
  precision: '32'
  performance_mode: True          # specialises kernels; every distinct scene
                                  # shape then pays a recompile

  # Settling. update_material_state steps until the pile is at rest;
  # settle_steps is a CAP, not a duration.
  settle_steps: 500
  settle_check_every: 10          # steps between rest checks, and an effective
                                  # minimum settle
  settle_velocity_threshold: 1.0e-3   # m/s
  settle_rest_quantile: 0.995     # fraction of particles that must be at rest;
                                  # a plain max gets harder the more envs are
                                  # batched
  # settle_angular_velocity_threshold: derived from the linear threshold and
  # the particle's half body diagonal if absent, so both express the same
  # surface speed. Set it only to override.

  pos_ctrl_steps: 100             # interpolation steps for a full lower/lift
  sweep_settle_steps: 12          # steps held after the sweep reference ends

rigid_options:                    # ANY RigidOptions field is forwarded;
                                  # unknown keys warn and are dropped
  iterations: 10
  ls_iterations: 10
  tolerance: 1.0e-4
  ls_tolerance: 0.05
  box_box_detection: True         # also sets the contact-point budget
  use_contact_island: True        # load-bearing; do not turn off
  use_hibernation: False
  enable_torsional_friction: True # resists spin about the contact normal
  constraint_solver: Newton       # Newton | CG
  # Two more are decided in code unless you set them, and there is no "auto"
  # value to write - OMIT the key to get the derived behaviour:
  #   enable_rolling_friction   on for sphere/cylinder, off otherwise
  #   max_collision_pairs       max(150, n_particles // 2)
  # Setting either to a literal (a bool / an int) overrides the derivation.

box:
  vol: [0.128, 0.128, 0.032]      # z is OVERWRITTEN: wall_thickness + one
                                  # particle height
  wall_thickness: 0.02
  friction: None                  # SET IN LOOP

material:
  vol: [0.127, 0.127, 0.05]       # spawn volume, inside the tray
  shape: None                     # SET IN LOOP
  particle_size: None             # SET IN LOOP — scalar, [min,max], or a list
  n_particles: None               # SET IN LOOP
  density: None                   # SET IN LOOP
  friction: None                  # SET IN LOOP
  rolling_friction: None          # None -> 0.3 when rolling friction is on

safety_margin: 0.02               # clearance between the tool footprint and
                                  # the wall when drawing a touchdown pose

plate:
  speed: 0.125                    # m/s, trapezoid cruise
  size: [0.04, 0.002, 0.01]       # long axis, thickness, height
  friction: 0.3                   # must be set: unset means Genesis' 1.0, and
                                  # contacts combine as max(mu_a, mu_b)

  # Actuator: a heavy carriage on a Cartesian gantry, not a 2.4 g box.
  moving_mass: 0.5                # kg reflected onto x/y/z, via armature
  acceleration: 2.0               # m/s^2 trapezoid ramp
  control_bandwidth_hz: 15.0      # -> kp = m*w^2, kv = 2*m*w
  max_force: 30.0                 # N per axis

  hold_mode: pinned               # pinned | servo — how z/roll/pitch/yaw are
                                  # held. servo was measured and REJECTED
                                  # (3-6x more particle penetration)
  approach_mode: servo            # servo | teleport — how descent/lift is driven
  arrival_steps: 12               # steps holding the final descent target
  orientation_inertia: 2.0e-4     # kg m^2, rotary axis
  orientation_bandwidth_hz: 30.0
  max_torque: 2.0                 # N m

data_collection:                  # written by data_collection.py; also readable
  render_images: true
  render_resolution: [128, 128]
```

`MPC/` adds two keys under `data_collection` for its own recording:
`record_transitions` (default true) and `transitions_dir`.

**A caution on `backend`.** Genesis falls back from GPU to CPU **silently**,
printing one warning line. A CPU run is otherwise indistinguishable from a GPU
one, and the two differ by more than an order of magnitude. Every script in
`tests/benchmarks/` prints the backend it actually used, for this reason.

---

## 5. Models and training

Two model stacks, deliberately separate.

**`GranularDynamics2/`** — upstream's U-Net dynamics models and their training
scripts, operating on occupancy grids. `Genesis/training/dataset.py`'s
`PileSweepData` rasterises transitions into `(input_grid, physics) -> output_grid`,
where `input_grid` is `[occupancy, action projection]` at 128×128.

**`MPC/`** — the research stack: a model registry, a dataset registry, a
config-driven trainer, and several model families. See
[`MPC/docs/ARCHITECTURE.md`](../MPC/docs/ARCHITECTURE.md).

### On physics conditioning

`PileSweepData` emits a 3-vector `(friction, density, box_friction)` normalised
over the endpoints of the original collection sweep. **Physics is held fixed
per collection run by default**, so that vector is a constant.

That has one consequence worth stating plainly: **the FiLM-conditioned models
are only meaningful when physics is varied.** With a constant conditioning
vector there is nothing to modulate on — use the plain U-Nets. The FiLM
variants are kept for the case where a sweep is turned back on.

### Splits

Splits are stratified by nominal physics, so no physics setting appears in both
train and validation. When a folder has fewer than three distinct physics
groups — which is the normal case, since physics is fixed per run — the split
falls back to whole **runs**. Runs are independent; samples *within* a run are
a trajectory, so splitting finer would put the two ends of one transition on
opposite sides.

---

## 6. MPC

`MPC/` plans pushes with a learned or oracle dynamics model. In outline:

- **`env/genesis_env.py`** wraps the simulator as a step-and-observe
  environment, rendering a top-down occupancy or RGB observation.
- **`simple_mpc/`** holds the planners. `genesis_oracle.py` is the oracle
  variant: it plans by rolling out *the simulator itself* in parallel
  environments, which sets an upper bound on what a learned model could
  achieve. `oracle_mpc.py` and `human_mpc.py` are the driver loops.
- **`env/recording_sandbox.py`** subclasses the simulator to record
  transitions incrementally — one at a time as a planner produces them, tagged
  by whether they were an executed step or a candidate rollout. The simulator's
  own batch-oriented recording is the wrong shape for that.
- **`registry/`** maps config strings to models and datasets, so an experiment
  is a YAML file rather than an import.

The planner's objective is an occupancy-based reward — pushing material toward
a target shape, with target shapes stored as distance fields under
`MPC/env/target_shapes/`.

Detail lives in `MPC/docs/`:
[`ARCHITECTURE.md`](../MPC/docs/ARCHITECTURE.md) (module map, data flow,
extension points, config structure),
[`INTERFACES.md`](../MPC/docs/INTERFACES.md) (the contracts between model,
dataset and trainer),
[`UTILITIES.md`](../MPC/docs/UTILITIES.md),
[`oracle_mpc_design.md`](../MPC/docs/oracle_mpc_design.md) (why the oracle exists
and what it records),
[`human_demo_design.md`](../MPC/docs/human_demo_design.md).

---

## 7. World models

`dino_wm/` and `le-wm/` are git submodules. Fetch them with:

```bash
git submodule update --init --recursive
```

### DINO-WM

`data_collection.py --export-dino-wm` writes DINO-WM's format alongside the
native one:

```
<root>/<shape>/n<N>/size<S>/<obs_type>/
    states.pth        (n_episodes, T, n_particles * 7)
    actions.pth       (n_episodes, T, 5)   [x0, y0, x1, y1, yaw]
    proprios.pth      (n_episodes, T, 3)   tool pose [x, y, yaw]
    seq_lengths.pth   (n_episodes,)
    obses/episode_%06d.pth   (T, H, W, 3) uint8
```

`<obs_type>` is `occupancy` (rasterised from particle state, no camera needed)
or `rendered` (the RGB frames, which needs `--render-images`). Blade yaw is
carried as its own action dimension because it is sampled independently of the
push direction and is not recoverable from start and stop alone.

To train, point DINO-WM at that directory. Its shipped config hardcodes an
absolute path, so override it:

```bash
cd dino_wm
python train.py --config-name train_granular     env.dataset.data_path=/abs/path/to/<shape>/n<N>/size<S>     env.dataset.obs_type=occupancy
```

**DINO-WM requires a GPU.** `models/vit.py` builds its attention mask with a
hardcoded `.to('cuda')`, so there is no CPU path.

### LeWM

`Genesis/training/export_leworldmodel_dataset.py` converts a DINO-WM-format
directory into the single HDF5 file LeWM reads. It is a separate step, not a
`data_collection.py` flag:

```bash
cd Genesis
python training/export_leworldmodel_dataset.py     --input-dir <dino_wm_root>/<shape>/n<N>/size<S>/occupancy     --output-path ~/.stable_worldmodel/datasets/granular.h5
```

It also rewrites each episode's final action row from DINO-WM's zero padding to
`NaN`, which is LeWM's convention for "no action taken from this frame" and
what its normaliser filters on.

---

## 8. Verifying

```bash
python -m pytest tests/ -q          # fast, no simulator needed
python tests/verify_fixes.py        # builds a scene; asserts every fix
```

`verify_fixes.py` is the single "does this still work" gate. It asserts rather
than prints, and every check is a step count, a distance, an angle or an exact
equality — never a timing — so it should reproduce on any machine and a failure
is a real regression. Exits nonzero on any failure.

For performance on your own hardware, `tests/benchmarks/` — those print numbers
and leave the judging to you, which is the right shape for exploring and the
wrong shape for a gate. Start with the backend line in their header.
