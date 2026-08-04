# Oracle MPC — Genesis-as-Model Sampling MPC (CEM / MPPI)

This is a reference doc for the `simple_mpc.oracle_mpc` subsystem: what it is,
how it's built, and why the non-obvious pieces are shaped the way they are.
See `docs/ARCHITECTURE.md` for where its files sit in the overall module map,
and follow that file's documentation policy — update this doc in the same
change that touches the design decisions recorded here.

## Purpose

A *ceiling baseline* for MPC performance on granular pile manipulation: the
MPC's prediction step **is the Genesis simulator itself**, removing dynamics-
model error entirely. Any remaining gap between plan and outcome then
measures the limits of the optimizer, horizon, and action parameterization —
not the dynamics model. Learned-model MPC runs (`simple_mpc.mpc`) are meant
to be compared against this ceiling.

Because the simulator is not differentiable through `execute_action`,
gradient descent (`simple_mpc.mpc.run_simple_mpc`'s approach) isn't usable
here. `simple_mpc.oracle_mpc` uses sampling-based optimization instead — CEM
and MPPI, both implemented, selectable via `mpc.optimizer` (`'cem' | 'mppi'`)
since they share one `ask`/`tell`/`best` skeleton and differ only in the
distribution-update rule.

## Architecture

### One Genesis scene, `n_envs` parallel copies, env 0 plays two roles

Genesis allows `gs.init()` only once per process and a scene's `n_envs` is
fixed at `build()`, so a separate 1-env "real" sim plus a K-env "planning" sim
isn't possible. `GenesisOracleEnv` (`simple_mpc/genesis_oracle.py`) instead
builds **one** `SandboxManipulation(n_envs=K)` (mirroring
`Genesis/data_collection_clean.py`'s multi-env batching):

- **Env 0 is "reality".** Its particle state is ground truth; the overhead
  camera is bound to it via `scene.add_camera(env_idx=0)` +
  `VisOptions(rendered_envs_idx=[0])`, so `render()`/`step()` only ever see
  env 0.
- **All K envs, including env 0, are rollout workers during planning.** Each
  optimizer iteration assigns a distinct candidate action to every one of the
  K envs and executes them in parallel via the same batched
  `execute_action(p_start[K,3], p_stop[K,3], angle[K])` path
  `Genesis/data_collection_clean.py` uses. Env 0 is not special-cased out of
  this — it gets a real candidate action like any other env, which is exactly
  why the state management below is load-bearing rather than a nicety.

### Snapshot / restore: why env 0's live state can't be the broadcast source

Env 0 participates in the candidate pool (previous section), so its live
particle state drifts after the very first rollout of an MPC step — it
reflects whatever candidate action happened to land in its slot, not the true
current state. If `GenesisOracleEnv` broadcast that live state as "ground
truth" for the next candidate batch, the state every candidate is evaluated
from would silently drift with each optimizer iteration, and the final
"predicted vs. actual" comparison would be checking against a moved target
rather than the real push.

The fix: `GenesisOracleEnv.snapshot_particles()` captures env 0's true state
**once**, at the top of each MPC step's planning phase, into a plain
`{'pos': ..., 'quat': ...}` dict. Every `rollout_candidates(act_seqs,
snapshot, ...)` call — every candidate batch, and the final re-roll of the
winning sequence — restores from this frozen snapshot
(`restore_snapshot` → `SandboxManipulation.set_particle_state`), never from
env 0's live state. Before the real `env.step()` executes the winning action,
the caller (`run_oracle_mpc`) calls `env.restore_snapshot(step_snapshot)`
again, undoing whatever the final re-roll did to env 0.

`SandboxManipulation.set_particle_state(pos, quat)` (in
`Genesis/sandbox_manipulation_clean.py`) is the low-level primitive this
builds on: it sets every env's particle pose directly from given tensors and
zeroes particle velocities. `broadcast_state_from_env(src_env)` is a thin
wrapper for the simpler one-off case (`__init__`, `reset()`) where the source
env's *live* state is known to be correct (right after `shuffle_particles()`
+ settle, nothing has mutated it yet).

### Full-fidelity re-roll for the predicted-vs-actual check

`rollout_candidates(..., use_rollout_fidelity=True)` (the default, used while
optimizing) uses reduced settle/clearance step budgets for speed — see
"Reduced-fidelity planning rollouts" below. The final re-roll of the winning
sequence — used only to populate `states_pred` and compute `predicted_reward`
— passes `use_rollout_fidelity=False`, so it uses the *same* full-fidelity
budgets as the real `env.step()` that follows. With an identical snapshot,
identical action, and identical step budgets on both sides, the two should
agree almost exactly (same deterministic sim), which is the whole point of
the check: it validates state-sync and determinism, not a genuine model gap
(there isn't one — the model is the simulator).

### Reduced-fidelity planning rollouts

Per phase (lower / sweep / lift / settle), only the settle and clearance
(lower/lift) step counts are reduced during planning — the sweep itself is
already the physical minimum given plate speed and travel distance, not an
independently tunable "step count" knob. `GenesisOracleEnv` computes
`rollout_settle_steps` / `rollout_clearance_steps` as half of the real
budgets by default (`mpc.rollout_settle_steps` in config overrides the
settle half); the real executed step (`env.step()`) always uses the full
`dataset.settle_steps` / full clearance budget.

### Cost: the same loss registry, not a parallel implementation

Every candidate's rollout terminal (or per-step, see below) occupancy is
scored using `training.losses.build_loss`, exactly as training does, with
`per_sample=True` forced by `run_oracle_mpc` regardless of what the config
says (candidate ranking always needs one cost per candidate, never a
batch-reduced scalar — see `training/losses.py`'s module docstring and
INTERFACES.md §3.5). `mpc.loss` in config selects the loss type and weights;
any loss registered in `training/losses.py` is usable here for free,
including ones added later for training. Two losses are relevant in
particular:

- `eulerian_combined` — the usual mse/dice/bce/... terms, scored against the
  binary goal occupancy mask (`OccupancyReward.goal_occupancy_mask`).
- `score_map_weighted` — `loss = -sum(occupancy · score_map)`, i.e. literally
  the negative of the occupancy reward used for reporting
  (`OccupancyReward.compute_score_tensor`), so optimization and reporting can
  share the exact same objective if desired.

`_per_sample_cost`/`_discounted_cost` (`simple_mpc/oracle_mpc.py`) build the
`(prediction, batch)` pair the loss contract expects: predicted occupancy
becomes `ModelOutput(logits=torch.logit(occ), probabilities=occ)`, and the
batch dict carries `current_occupancy`/`target_occupancy` and `score_map`
together so whichever loss is configured finds what it needs.

`cost_mode: 'terminal'` (default) scores only the final rollout state;
`'discounted'` sums per-step costs with `gamma**step` weighting (only
meaningful when `n_look_ahead > 1`).

### Occupancy representation: footprint-splat vs. dense depth-render

Two different occupancy measurements exist in this pipeline, deliberately:

- **Optimization / planning** (`GenesisOracleEnv.particles_to_occ`) splats
  each of the (few) particle *positions* into the occupancy grid via a
  hard-disk footprint, since a rollout only ever has particle positions, not
  a rendered image. `env.current_particles_world()` +
  `particles_to_occ` gives a footprint-based reading of env 0's *actual*
  state too, for a representation-consistent comparison against
  `predicted_reward` (see `oracle_mpc.py`'s `same-repr actual` log line).
- **Reward reporting** (`rewards[]`/`occ_rewards[]`,
  `_report_occupancy_from_obs`) uses the full foreground point cloud
  extracted from the rendered depth image — the same source
  `EulerianAdapter.obs_to_state` uses — so oracle-MPC rewards are on the same
  scale as learned-model MPC runs, which is the entire point of a ceiling
  baseline.

These two will not match exactly even given perfect physics, since they're
different measurements of the same state. `footprint_radius_voxels`'s
`shape_factor` (`transforms/functional.py`) narrows the gap for cube-shaped
material specifically: the naive half-edge radius makes two face-touching
cubes' disks exactly tangent (zero overlap margin) and under-covers a
yaw-rotated cube's corners, so the disk-union under-counts area precisely as
a push clusters material tightly — which is exactly when it matters most,
since goal-region voxels carry far more reward weight than empty ones under
`empty_penalty > 0`. `shape_factor=sqrt(2)` (the circumscribed/half-diagonal
radius) closes this — `GenesisOracleEnv` sets it automatically from
`material.shape == 'cube'`. Some residual gap between the two measurements is
still expected and not a bug; `oracle_mpc.py`'s per-step log prints both the
reward gap and the raw occupied-voxel-count deficit (`dense - footprint`) so
it stays visible and diagnosable rather than silently absorbed.

### Video: intermediate action frames via a generic hook

`SandboxManipulation.execute_action(..., on_phase=callback)`
(`Genesis/sandbox_manipulation_clean.py`) fires `callback('post_lower')` and
`callback('post_sweep')` at the two natural mid-action points (plate reached
push-start; plate reached push-stop), in addition to the pre/post-action
frames every real step already captures. This is a generic, purpose-agnostic
hook on the simulator core — see `ARCHITECTURE.md`'s Design Philosophy and
`UTILITIES.md` §1.1 for why it isn't camera/video-aware itself.
`GenesisOracleEnv.step()` (and `env.genesis_env.GenesisEnv.step()`, shared)
wire it to `utils.write_video_frame`, so a recorded video shows
before → plate-at-push-start → plate-at-push-stop → after, making the actual
action visible rather than just the before/after box state. Rollout/planning
calls never pass `on_phase`, so this adds no overhead there.

### Transition recording: real steps *and* candidate rollouts, tagged apart

`GenesisOracleEnv` records every push it executes as a training-usable
transition (before/after particle state + action), via
`SandboxManipulation.push_and_record` — see `UTILITIES.md` §1.2 for the
general mechanism shared with `env/genesis_env.py::GenesisEnv`. Two things
are specific to the oracle path:

- **Real steps** (`step()`) broadcast the *identical* winning action to all
  `n_envs` envs, so only env 0's sample is appended
  (`push_and_record(..., record_all_envs=False)`) — recording all `n_envs`
  would just duplicate the same transition `n_envs` times.
- **Candidate rollouts** (`rollout_candidates()`, `record=True` default)
  append one sample per env per horizon step, tagged `is_candidate=True` —
  every one of the `n_envs` candidates evaluated per optimizer iteration is a
  genuinely distinct, physically-simulated push, essentially free additional
  training data since it's already being simulated to compute the cost. The
  **final re-roll of the winning sequence**
  (`use_rollout_fidelity=False` in `oracle_mpc.py`) passes `record=False`:
  every env there executes the identical winning action from the identical
  snapshot that `step()` is about to execute and record for real, so
  recording it here too would be pure duplication.

`mpc_step` (which real MPC step's planning phase produced a sample) comes
from `GenesisOracleEnv`'s own step counter, incremented only in `step()` —
`oracle_mpc.py` never passes this through, so no signature in the MPC loop
itself changed to support recording.

Flushing is **incremental, per real step** (`step()` passes
`flush_after=True`), not deferred to episode end: each real step's flush
writes that step's transition plus every candidate rollout accumulated
during its own planning phase. Given real steps can each take minutes at
`n_envs=256`, this is what makes data appear on disk throughout a long
episode rather than only after it completes — see UTILITIES.md §1.2.
Episode identity (source, episode index, seed, optimizer) is set once,
*before* the episode runs, via `env.set_recording_context({...})` in
`run_oracle_mpc.py`; every per-step flush during that episode picks it up
automatically. Reward/success are deliberately not part of this context
(unknown until the episode ends) — they're saved separately in that
episode's `metrics.json` and joinable by `source` + `episode_idx`.

### Sampling optimizers

`simple_mpc/sampling_optimizers.py`: `SamplingOptimizer` base class
(`reset(init_mean)` / `ask(n)` / `tell(candidates, costs)` / `best()`), with
`CEMOptimizer` and `MPPIOptimizer` implementations and a
`make_sampling_optimizer(name, ...)` factory.

- **CEM**: keeps the `cem.n_elite` lowest-cost candidates each iteration,
  refits a diagonal Gaussian (mean, std) from them with `cem.momentum`
  smoothing, and floors the std at `cem.std_floor` to avoid premature
  collapse.
- **MPPI**: reweights candidates by `softmax(-cost / mppi.lambda)`, sets the
  mean to the weighted average, and low-pass-filters the mean *across MPC
  steps* with `mppi.beta_filter` (applied in `reset()`, blending the previous
  step's converged mean with the freshly warm-started one).
- **Warm start** (`warm_start_mean`): between MPC steps, the mean sequence is
  time-shifted by one action; the freed-up tail slot is re-randomized within
  bounds rather than repeating the last action (for `n_look_ahead == 1` this
  correctly degenerates to a fully fresh random restart every step, since
  there's nothing to shift).
- Initial/fresh samples come from `simple_mpc/action_sampler.py`'s existing
  samplers (default `physics_aware`), clipped to the same workspace bounds
  `simple_mpc.mpc.run_simple_mpc` uses.

`n_envs` candidates are evaluated per optimizer iteration (one per env,
including env 0 — see above). If `mpc.n_sample > mpc.n_envs`,
`run_oracle_mpc` runs `ceil(n_sample / n_envs)` sequential candidate batches
per iteration before calling `optimizer.tell()` once on the concatenated
result; set `n_sample == n_envs` to avoid that extra cost.

### Horizon

`mpc.n_look_ahead` is held **constant** across an entire run — unlike
`run_simple_mpc`'s `min(n_look_ahead, n_mpc - i)` shrinking horizon near the
end of an episode, the sampling optimizers' mean/std tensors are shaped by
`n_look_ahead` at construction and aren't resized mid-run. Planning slightly
past the nominal number of remaining real steps near the end of an episode is
harmless (only `act_seq[0]` is ever executed for real).

### Action convention: 4D everywhere here, 5D available for other callers

Every action this module and its optimizers produce (`action_sampler.py`,
`sampling_optimizers.py`) is 4D — `[sx, sy, ex, ey]` — and
`GenesisOracleEnv.step()`/`.rollout_candidates()` derive the plate yaw as
perpendicular to travel direction, exactly as before. That derivation now
lives in one shared place, `transforms.functional.action_to_pose`, which
also accepts an optional 5th component (`angle_norm`, a normalized `[0, 1)`
plate yaw independent of travel direction) — used by the human-demonstration
subsystem (`docs/human_demo_design.md`), not by anything in this module.
Nothing here constructs or expects a 5th component; this is purely a note
that `GenesisOracleEnv`'s real-step/rollout call sites are shared with that
other subsystem, so a change to `action_to_pose` affects both.

## File map

| File | Responsibility |
|---|---|
| `simple_mpc/genesis_oracle.py` | `GenesisOracleEnv` — batched multi-env wrapper: real-env surface (`render`/`step`/`reset`/`get_cam_params`) + planning surface (`snapshot_particles`/`restore_snapshot`/`rollout_candidates`/`particles_to_occ`) |
| `simple_mpc/oracle_mpc.py` | `run_oracle_mpc` (the MPC loop) + `load_oracle_config`; returns the same result-dict schema as `run_simple_mpc` |
| `simple_mpc/sampling_optimizers.py` | `SamplingOptimizer`, `CEMOptimizer`, `MPPIOptimizer`, `make_sampling_optimizer` |
| `simple_mpc/occupancy_reward.py` | `OccupancyReward` — `compute_score_tensor` (reward map) and `goal_occupancy_mask` (binary loss target), shared with the learned-model MPC path |
| `transforms/functional.py` | `particles_to_occupancy(..., footprint_radius=...)`, `footprint_radius_voxels(..., shape_factor=...)`, `genesis_particles_to_cam3d`, `action_to_pose` (4D-derived-yaw / 5D-explicit-yaw action convention — see below and `docs/human_demo_design.md`) |
| `training/losses.py` | `EulerianCombinedLoss`'s `per_sample` mode, `ScoreMapWeightedLoss` |
| `Genesis/sandbox_manipulation_clean.py` | `execute_action(..., on_phase=...)`, `set_particle_state`, `broadcast_state_from_env`, `push_and_record`/`flush_transitions` — shared, not oracle-specific |
| `Genesis/transition_buffer.py` | `TransitionBuffer` — accumulates/saves the transitions `push_and_record` records (see UTILITIES.md §1.2) |
| `Genesis/benchmark_n_envs.py` | Throughput sweep (`python -m Genesis.benchmark_n_envs`) for picking `mpc.n_envs` |
| `simple_mpc/config/config_oracle.yaml` | General-purpose single-episode config |
| `simple_mpc/config/config_oracle_test.yaml` | Multi-episode sanity-check config (`episodes.n_episodes`) |
| `run_oracle_mpc.py` | Entry point: builds one `GenesisOracleEnv`, runs `episodes.n_episodes` episodes, saves per-episode data + a run-level summary |
| `utils.py` | `write_video_frame` (shared with the learned-model MPC path) |

## Config reference

`mpc` section keys (see `simple_mpc/config/config_oracle.yaml` for the full
annotated default and `config_oracle_test.yaml` for a multi-episode variant):

| Key | Meaning |
|---|---|
| `optimizer` | `'cem'` \| `'mppi'` |
| `n_mpc` | Real MPC steps executed per episode |
| `n_look_ahead` | Planning horizon (held constant — see above) |
| `n_envs` | Parallel Genesis envs = candidates evaluated per optimizer iteration; pick via `Genesis/benchmark_n_envs.py` |
| `n_sample` | Candidates per optimizer iteration; `> n_envs` triggers sequential re-batching (see above) |
| `n_opt_iter` | Optimizer iterations per MPC step |
| `rollout_settle_steps` | Settle-phase step count during planning rollouts only (default: half of `dataset.settle_steps`) |
| `cost_mode` | `'terminal'` \| `'discounted'` |
| `gamma` | Discount factor (only used when `cost_mode='discounted'`) |
| `cem.*` / `mppi.*` | Optimizer-specific hyperparameters (see class docstrings) |
| `grid_res` | Occupancy grid resolution, e.g. `[64, 64]` |
| `loss` | Passed to `training.losses.build_loss` (`per_sample` is forced on regardless of what's set here) |
| `reward.empty_penalty` | Reporting-only reward shaping, same convention as `simple_mpc.mpc` |
| `action_sampler` | `'physics_aware'` (default) or any other registered sampler |
| `save_mpc_transitions` | Whether `run_oracle_mpc.py` tags each episode's incremental per-step flushes with episode identity via `set_recording_context(...)` (default True); recording itself (`dataset.record_transitions`) happens regardless — see UTILITIES.md §1.2 |

`dataset` section keys relevant to transition recording (read by
`GenesisOracleEnv`/`GenesisEnv` via `env/genesis_env.py::_build_genesis_config`,
not `mpc`): `record_transitions` (default True), `transitions_dir` (default
`'data/mpc_runs'`, relative to `Genesis/`).
| `task` | Goal spec (`target_shape`/`target_control`), same schema as `simple_mpc.mpc` |

`episodes` section (read by `run_oracle_mpc.py`, not by `run_oracle_mpc()`
itself): `n_episodes`, `random_seed_base`.

## Usage

```bash
python -m Genesis.benchmark_n_envs                      # pick mpc.n_envs first
python run_oracle_mpc.py --config simple_mpc/config/config_oracle.yaml
python run_oracle_mpc.py --config simple_mpc/config/config_oracle_test.yaml \
       --save-video --n-episodes 5
```

CLI overrides available: `--n-envs`, `--optimizer`, `--n-mpc`, `--n-episodes`,
`--seed`, `--output-dir`, `--save-video`.

### Saved data (per episode)

`raw_obs.npy` (RGB + material mask + depth per real step — the most direct
route to recreating frames offline), `states.npy` (depth-derived foreground
point clouds per step), `states_pred.npy` (each chosen action's own predicted
occupancy rollout, for the predicted-vs-actual check), `rewards.npy` /
`occ_rewards.npy` / `actions.npy`, `episode_data.npz` (optimizer convergence
traces), `rewards.png`, `metrics.json`, and (with `--save-video`) an `.avi`.
Run-level: `run_config.yaml` (exact config used), `rewards_all.npy`,
`rewards_summary.png`, `summary.json`.

Separately (not under this run's own output directory), every real step and
every candidate rollout is also saved to the shared, ever-growing
`Genesis/data/mpc_runs/` directory — see "Transition recording" above and
UTILITIES.md §1.2. This is training-ready data (states/states_/p_starts/
p_stops/angles + success/is_candidate/mpc_step), distinct from the
per-episode files above, which are for inspecting/replaying a specific run.

## Known limitations

- **Residual dense-vs-footprint reward gap.** Even after the `shape_factor`
  fix, the two occupancy measurements described above are not identical —
  watch the `same-repr actual` vs. `dense/report` gap in `oracle_mpc.py`'s
  log; the former should be ≈0 (physics/determinism check), the latter is
  expected to show some gap.
- **Constant horizon.** No shrinking-horizon behavior near the end of an
  episode (see above) — harmless but worth knowing if comparing step-by-step
  against `run_simple_mpc`.
- **Not integrated into `run_experiments.py`'s batch runner.** `run_oracle_mpc.py`
  is a standalone entry point; wiring oracle runs into the same
  experiment-suite YAML format as learned-model comparisons is future work.
- **Compute cost.** A single MPC step at `n_envs=256`, `n_opt_iter=5-10` has
  been observed in the range of tens of minutes on the reference GPU setup —
  budget accordingly, and prefer `Genesis/benchmark_n_envs.py` plus small
  `n_mpc`/`n_opt_iter` sanity runs before scaling up episode count.
- **Transition-recording volume is unthrottled.** Every candidate rollout
  gets recorded (not just executed steps) — cheap per-sample (particle
  position tensors, not images) but the sample *count* per episode scales
  with `n_opt_iter * n_envs * n_mpc`, so `Genesis/data/mpc_runs/` grows
  quickly across many episodes/runs. No sub-sampling is applied by default;
  disable with `dataset.record_transitions: false` or point
  `transitions_dir` elsewhere if this becomes unwieldy.
