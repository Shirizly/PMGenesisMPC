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
| `probe_push_cost.py` | why a push step costs what it does: settle-vs-push ratio, blade orientation, and the `iterations` / `use_contact_island` / `box_box_detection` knobs. Variant outcomes are recorded in its `VARIANTS` dict so they need not be re-run |
| `benchmark_throughput.py` | **transitions/second** per `(n_objects, n_envs)` — the operating point, as opposed to `benchmark_scaling.py --vram-only`'s memory ceiling. Driven by `Genesis/configs/throughput_benchmark.yaml` |
| `probe_contact_islands.py` | **why a transition costs 85x more at 200 objects than at 100**: reads the solver's contact-island decomposition during a push and correlates island size with step cost. Also sweeps tray size at fixed particle count, which is what identifies packing fraction as the lever |
| `probe_collection_health.py` | **whether a realistic collection produces sensible data**: goal-reach rate over the real action distribution, per-env settledness at the instant the pooled criterion trips, action-space bias from placement-aware sampling, travel truncation, escaped particles, `s' == s` |
| `record_simulation_video.py` | renders the spawn, settle, descent and pushes to video from three camera angles, for the failures no metric was written to catch |
| `verify_fixes.py` | asserts each correctness fix against a live scene (12 checks) |
| `verify_new_features.py` | asserts the state library and placement-aware sampling against a live scene (10 checks) |

The two `verify_*` scripts are the ones worth re-running after touching
`sandbox_manipulation_clean.py`; they are end-to-end assertions, not benchmarks,
and they fail loudly.

`probe_collection_health.py` and `record_simulation_video.py` are the pair to run
before trusting a long collection, or before porting these changes elsewhere.
They answer the two halves of "is the output sound?" that the assertions cannot:
the probe checks the statistics of a realistic run, and the video is there for
whatever nobody thought to measure. Neither asserts — what counts as acceptable
is a judgement about the dataset, not a property of the code — so both are read,
not just run.

## Contents

- `settling_investigation.md` — the full settling diagnosis: what was measured,
  what it implied, and the root cause. Kept in narrative form deliberately,
  including the wrong turns, because the measurement traps it documents (extreme-
  value statistics over batched envs, angular-vs-linear threshold units,
  velocity as a proxy for position convergence) are easy to fall into again.
- `results/` — raw JSON and log output from the runs quoted in the reports.

## Reusing settled-state libraries

Probes that need a settled pile seed it from a recorded library
(`Genesis/data/dry_run/<shape>/n<N>/size<S>/settled_states.pt`) via
`StateLibrary.apply_per_env`, rather than settling a fresh two-layer respawn —
which costs ~1500 steps at 200 particles and dominates a probe's runtime while
contributing nothing to what it measures. Use `apply_per_env`, not `apply`:
the latter broadcasts one state to every env, and identical piles across envs
are both unrepresentative and measurably cheaper to solve.

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
4. **Push cost is dominated by blade orientation, and two solver knobs are
   load-bearing.** Broadside vs edge-on differs by 2-9x; `use_contact_island:
   False` is ~1100x slower; `box_box_detection: False` corrupts memory because
   it silently cuts the contact-point budget below what the pile needs.
