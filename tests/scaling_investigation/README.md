# Scaling investigation — probes, measurements, and the settling diagnosis

Archived working material from the investigation that produced the simulator
changes documented in **`docs/scaling_to_200_objects.md`**. Read that first; this
directory is the evidence behind it.

Nothing here runs as part of the test suite. These are one-shot measurement
scripts, kept so every number in the scaling document can be re-derived and so
the same questions can be re-asked after future changes. The fast, Genesis-free
regression tests for the code this work added live in `tests/test_state_library.py`
and `tests/test_placement_sampling.py`.

## Running them

All of them need a GPU and the `pme` environment, and must be run **as modules
from the repo root** — `sandbox_manipulation_clean` uses relative imports that
only resolve inside its package:

```
conda activate pme
python -m tests.scaling_investigation.<script> --help
```

Most take a subprocess-per-cell approach so an OOM in one configuration does not
abort the sweep.

## What each script measures

| script | question it answers |
|---|---|
| `benchmark_scaling.py` | cost and VRAM per `(n_particles, n_envs)`; `--vram-only` finds the OOM ceiling quickly |
| `probe_step_cost.py` | what one simulation step costs, split into raw physics vs. control/sync overhead, against `max_collision_pairs` and `box_box_detection` |
| `probe_contact_counts.py` | how many collision pairs and contact points a pile really uses, so `max_collision_pairs` can be set from data |
| `probe_plate_dynamics.py` | how faithfully the plate tracks its commanded path, and how much the pile perturbs it; compares the current actuator against the previous one in a single build |
| `probe_settle_convergence.py` | how many steps a pile needs to reach rest, per particle count |
| `probe_settle_truncation.py` | whether the settle can be cut short and velocities zeroed — distance from the converged configuration, and whether a zeroed state stays put |
| `probe_settle_ab.py` | **the decisive settling experiment**: isolates warm-start discard from solver budget by settling the same snapshotted spawn under four variants |
| `probe_persistent_movers.py` | *which* particles never stop, where they are, and whether they are vibrating in place or genuinely travelling; `--mode push` vs `--mode respawn` |
| `verify_fixes.py` | asserts each correctness fix against a live scene (12 checks) |
| `verify_new_features.py` | asserts the state library and placement-aware sampling against a live scene (10 checks) |

The two `verify_*` scripts are the ones worth re-running after touching
`sandbox_manipulation_clean.py`; they are end-to-end assertions, not benchmarks,
and they fail loudly.

## Contents

- `settling_investigation.md` — the full settling diagnosis: what was measured,
  what it implied, and the root cause. Kept in narrative form deliberately,
  including the wrong turns, because the measurement traps it documents (extreme-
  value statistics over batched envs, angular-vs-linear threshold units,
  velocity as a proxy for position convergence) are easy to fall into again.
- `results/` — raw JSON and log output from the runs quoted in the reports.

## The three headline findings

1. **`max_collision_pairs` was throttling parallelism, not protecting physics.**
   Measured need is `≈0.26 × n_particles`; an oversized cap costs environments
   directly because the constraint Jacobian scales with it.
2. **The pusher plate's trajectory error was the control law, not granular
   reaction.** Reaction displaced the tool by ≤0.46 mm; the endpoint-target PD
   put it up to 23.9 mm off its commanded path.
3. **Piles were not failing to settle — the solver was being reset every step.**
   Holding the plate with a per-step `set_dofs_position` discarded the constraint
   warm start on every settle step. Removing it dropped the worst particle from
   25.1 mm/s to 1.0 mm/s *and* cost 25 % less per step.
