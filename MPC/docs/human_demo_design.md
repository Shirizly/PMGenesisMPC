# Human-Demonstration MPC — Interactive Ceiling Baseline

This is a reference doc for the human-demonstration subsystem: what it is,
how it's built, and why the non-obvious pieces are shaped the way they are.
See `docs/ARCHITECTURE.md` for where its files sit in the overall module
map, and `docs/oracle_mpc_design.md` for the Genesis-oracle machinery it
builds directly on top of. Follow `ARCHITECTURE.md`'s documentation policy —
update this doc in the same change that touches the design decisions
recorded here.

## Purpose

`simple_mpc.oracle_mpc.run_oracle_mpc` (CEM/MPPI) already removes dynamics-
model error from the ceiling-baseline comparison by using the Genesis
simulator itself as the predictive model — but its remaining gap from
"perfect" is still bounded by the sampling optimizer's coverage, its action-
sampling prior, and the horizon/budget it's given. This subsystem replaces
the *optimizer* with a human, to measure a second, tighter ceiling: what
reward is achievable when action selection is driven by a person who can see
the state and reason about it directly, not by CEM/MPPI sampling. The
oracle simulator remains the source of truth for both — only who picks the
candidate action changes.

Two pieces make a human usable in this loop without either wasting a
person's time on typing exact coordinates or wasting compute re-running the
sim on every mouse movement:

- A **local grid search** (`simple_mpc.human_grid_search`) that treats a
  human's drawn action as *approximately* right and searches a small
  neighborhood around it with the same oracle-simulator rollouts
  `run_oracle_mpc`'s optimizers use — a human is good at picking roughly the
  right push, not at hitting the exact sub-millimeter/sub-degree optimum.
- A **GUI** (`human_mpc_gui.py`) for drawing that approximate action by
  drag-and-drop, modeled on `debug_mpc_gui.py`'s canvas interaction.

## Architecture

### A parallel, interactive control flow — not a variant of `run_oracle_mpc`

`run_oracle_mpc`'s loop is fully automated: it decides `n_look_ahead`,
samples/refines candidates for `n_opt_iter` iterations, and executes the
winner, all without pausing. A human needs to *see* the current occupancy
before choosing an action, and a GUI needs to block on a mouse click, not
run headlessly for `n_mpc` steps in a row. Rather than bending
`run_oracle_mpc`'s loop (and its `SamplingOptimizer` `ask`/`tell`/`best`
skeleton, built for exactly that automated shape) to accept an interactive
callback, `simple_mpc.human_mpc.HumanDemoSession` is a separate, small
session object over the *same* `GenesisOracleEnv` — `run_oracle_mpc` and its
optimizers are completely untouched by this subsystem.

`HumanDemoSession` exposes two calls per real step, deliberately split apart:

```
propose(action5)   -> grid-search-refine a drawn action, WITHOUT committing
                      it (safe to call repeatedly — e.g. redraw, re-propose)
commit(action5=None) -> execute an action for real (default: the last
                      propose()'s winner), advancing the episode
```

A GUI (or any other front end — a script, a notebook) drives these two calls
directly; there's no shared automated loop to plug into.

### The action is 5D: `[sx, sy, ex, ey, angle_norm]`

Every existing action representation in this codebase (`simple_mpc.mpc`,
`simple_mpc.action_sampler`, `simple_mpc.sampling_optimizers`,
`env.genesis_env.GenesisEnv`) is 4D — `[sx, sy, ex, ey]` — and always derives
the plate's yaw as perpendicular to the travel direction
(`atan2(ey-sy, ex-sx) + pi/2`). This is a convention every one of those
callers happens to use, not a constraint of the simulator:
`SandboxManipulation.execute_action`'s own `angle` parameter is already
independent of the `p_start`/`p_stop` travel path (see
`plate_velocity_translation` — the sweep direction comes purely from
`p_end - p_start`, while `angle` only sets `_horizontal_dof_fix`'s yaw DOF).
Human demonstrations want that decoupling exposed, since a plate's
orientation relative to the pile can matter independently of which way it
travels — so this subsystem adds a genuine 5th action component.

`transforms.functional.action_to_pose(act)` is the one place this branch
lives: given a `(..., 4)` action it derives yaw from direction (byte-for-byte
the old behavior); given `(..., 5)` it uses `act[..., 4] * pi` directly.
`angle_norm` is normalized to `[0, 1)` rather than raw radians because the
plate is symmetric under a 180-degree rotation (matching
`simple_mpc.action_sampler.PlateCollisionChecker`'s `k * pi / n_angles`
convention) — `[0, 1)` maps onto exactly one physically-distinct orientation
each, with no redundancy. `GenesisOracleEnv.step()` and
`.rollout_candidates()` both call this shared function (they previously
inlined the 4D-only derivation), so every automated CEM/MPPI run's behavior
is byte-for-byte unchanged — they only ever pass 4D actions — while the
human-demo path gets the 5th DOF for free at the one shared call site.

### Local grid search, not a sampling optimizer

`simple_mpc.human_grid_search.build_action_grid(center, grid_n, delta,
clip_lo, clip_hi)` builds the full `grid_n ** 5` Cartesian grid around a
center action. `delta` is a *normalized* half-width — a fraction of each
dimension's own `[clip_lo, clip_hi]` range — specifically so one scalar can
mean something sensible across dimensions with wildly different native
units (metres for `sx/sy/ex/ey`, a `[0, 1)` fraction of pi for the angle);
pass a length-5 array to vary the fraction per dimension instead. `grid_n
== 1` degenerates to the single center point with no exploration, so
"evaluate exactly what I drew" isn't a special case anywhere.

`grid_search_refine(...)` (`simple_mpc.human_grid_search`) evaluates that
grid through the oracle simulator in two passes, mirroring
`run_oracle_mpc`'s own optimize-then-re-roll structure (see
`oracle_mpc_design.md` "Full-fidelity re-roll" for the shared rationale):

1. Every grid candidate at reduced (`rollout_*`) fidelity, batched in chunks
   of `env.n_envs` (candidates beyond `env.n_envs` run as sequential
   batches — the grid is padded up to a multiple of `env.n_envs` with
   repeats of the last candidate; the padding's cost is discarded, only
   `n_real` candidates count). Cost is the *same* `model_training.losses`
   per-sample loss contract `run_oracle_mpc` uses
   (`_per_sample_cost`/`_occupancy_reward`, imported lazily — see "Why the
   oracle_mpc import is deferred" below) — a human-refined action's
   cost/reward is computed identically to an automated CEM/MPPI candidate's.
   Every candidate is recorded (`rollout_candidates`'s default
   `record=True`) as free additional training data, same convention as
   automated candidates (`oracle_mpc_design.md` "Transition recording").
2. The single winning candidate re-rolled at full fidelity
   (`use_rollout_fidelity=False, record=False` — the real step about to
   follow records it for real) so the reported `predicted_reward` isn't
   confounded by a fidelity gap on top of the genuine "will the real step
   match the plan" question.

Grid search is deliberately *not* implemented as a new `SamplingOptimizer`
plugged into `run_oracle_mpc`'s existing loop, even though the `ask`/`tell`
skeleton could technically be bent to fit (`ask()` returning a fixed grid
instead of random samples, `n_opt_iter=1`). `run_oracle_mpc`'s loop decides
`init_mean` itself every step (action-sampler bootstrap or warm-start) with
no pause for external input — accommodating a human's "look, then decide"
requirement there would have meant threading an interactive callback through
code whose entire shape assumes full automation. Keeping
`HumanDemoSession`/`grid_search_refine` separate keeps that loop untouched.

#### Why the `oracle_mpc` import is deferred

`simple_mpc.oracle_mpc` transitively imports `simple_mpc.genesis_oracle` ->
`Genesis.sandbox_manipulation` -> `genesis`. `build_action_grid` has
no such dependency (pure NumPy), so `simple_mpc.human_grid_search` imports
`_occupancy_reward`/`_per_sample_cost` from `simple_mpc.oracle_mpc` *inside*
`grid_search_refine` rather than at module level — this keeps
`build_action_grid` (and hence its unit tests, `tests/test_human_grid_search.py`)
importable and fast-testable without the `genesis` package installed, same
as every other Genesis-free module (see `ARCHITECTURE.md`'s Genesis-
dependent-modules note).

### GUI: drag-and-drop input, no automatic prediction

`human_mpc_gui.py` reuses `debug_mpc_gui.py`'s canvas-drag interaction and
tile-image helpers directly (`_heatmap_bgr`, `_stamp`, `_bgr_to_photo`) but
differs in two ways that both follow from one fact: the "model" here is the
*real simulator*, so every rollout costs real physics compute (tens of
candidates), unlike `debug_mpc_gui.py`'s near-instant learned-model forward
pass:

- **No auto-evaluate-on-drag-release.** `debug_mpc_gui.py` re-predicts on
  every handle release because a learned-model forward pass is cheap. Doing
  that here would trigger a `grid_n ** 5`-candidate physics rollout on every
  mouse-up, making the UI unusable. Dragging only updates the drawn arrow
  and orientation tick (both free); physics only runs on an explicit
  **Refine** (a `grid_n ** 5`-candidate search) or **Submit** (one real
  step) click.
- **Two tiles, not five.** Only current occupancy and the goal score map are
  shown — no predicted-occupancy heatmap tile (`grid_search_refine` computes
  one internally as `occ_pred`, cheap to add later; it's just not rendered
  yet). **Refine** still reports a predicted-reward *number* in the info
  bar (from the grid search's own full-fidelity re-roll) for whoever wants
  it before deciding whether to Submit.

**Refinement is strictly optional.** `on_submit` always executes exactly
the action currently shown in the fields — `HumanDemoSession.commit(act5)`
is called with that explicit action every time, regardless of whether
`propose()` was ever called for it. Clicking **Refine** first updates the
fields to the grid search's winner (and reports its predicted reward), so a
subsequent Submit executes that refined action; skipping Refine and
clicking Submit directly executes precisely what was drawn, no grid search
involved. `HumanDemoGUI` tracks whether the currently-shown action matches
the last `propose()` result purely to label the status message ("refined"
vs. "as drawn") — it is not a gate, and editing fields after a Refine never
blocks Submit.

The angle (5th action component) is a `Scale` slider next to the position
fields, not a second canvas drag gesture — v1 keeps canvas dragging 2D-only
(matching "start with just visualization of state and the ability to draw
the action") and exposes orientation numerically instead, mirroring how
`debug_mpc_gui.py`'s `lr` field is already an editable text control
alongside canvas-driven inputs. A yellow tick through the start handle,
oriented at the current angle, gives visual feedback independent of the
green travel arrow. An **Auto** button resets the angle to the usual
perpendicular-to-travel default (`HumanDemoSession.default_angle_norm`) and
re-enables auto-follow whenever the start/end handles are dragged again —
the default only stops auto-updating once the slider (or a Refine) has set
it explicitly.

The GUI drives **full multi-step episodes**, not one-shot evaluation: each
Submit advances `HumanDemoSession`'s internal step counter and re-renders
the post-action state for the next Refine/Submit cycle, until either the
episode's soft step cap (`mpc.n_mpc`, `HumanDemoSession.finished()`) is hit
(auto-saved) or the user clicks **Finish Episode** to stop early (also
saved, at whatever length was reached) — episode length isn't fixed
up-front, unlike `run_oracle_mpc`'s preallocated arrays, since a human isn't
obligated to use exactly `n_mpc` steps. **New Episode** resets the Genesis
scene, re-seeds, retags the recording context, and rebuilds the goal.

Like `debug_mpc_gui.py`, everything here is synchronous — Refine and Submit
block the Tkinter main loop for as long as their rollouts take (`_status()`
messages are flushed via `update_idletasks()` before each blocking call so
the user sees "running..." rather than a frozen window). No threading is
used, to avoid Tkinter/CUDA contention risk; this matches
`debug_mpc_gui.py`'s existing pattern.

### Output schema and data recording — directly comparable to automated runs

`HumanDemoSession.finalize()` + `simple_mpc.human_mpc.save_episode(...)`
produce the *same* per-episode files `run_oracle_mpc.py`'s
`run_one_episode` does (`rewards.npy`, `occ_rewards.npy`, `actions.npy`,
`states.npy`, `states_pred.npy`, `rewards.png`, `metrics.json`) so a human
episode's reward curve is directly comparable to an automated oracle-MPC
episode's, offline, with no format translation. `actions.npy` here is
`(n_steps, 5)` (vs. automated runs' `(n_mpc, 5)`, where the 5th column is
always the *derived* angle) — the difference is only in whether the 5th
component was chosen or derived, not in shape.

Transition recording reuses `GenesisOracleEnv`'s existing mechanism
unchanged (`oracle_mpc_design.md` "Transition recording"): `human_mpc_gui.py`
tags each episode via `env.set_recording_context({'source': 'human_demo',
'episode_idx': ..., 'seed': ..., 'optimizer': 'human_grid_search'})` before
building the session, so real steps and (tagged `is_candidate=True`) grid-
search rollouts land in the same shared `Genesis/data/mpc_runs/` pool as
automated runs, distinguishable by `source` for anyone assembling a
training or evaluation set that should include or exclude human
demonstrations specifically.

## File map

| File | Responsibility |
|---|---|
| `transforms/functional.py` | `action_to_pose(act)` — shared 4D-derived-yaw / 5D-explicit-yaw action convention, used by `GenesisOracleEnv.step`/`rollout_candidates` and this subsystem alike |
| `simple_mpc/human_grid_search.py` | `build_action_grid`, `grid_search_refine` — local 5D grid search around a drawn action, via the oracle simulator |
| `simple_mpc/human_mpc.py` | `HumanDemoSession` (`propose`/`commit`/`finished`/`finalize`), `save_episode` |
| `simple_mpc/genesis_oracle.py` | `GenesisOracleEnv` — unchanged in behavior for automated (4D-action) callers; now also accepts 5D actions in `step()`/`rollout_candidates()` via `action_to_pose` |
| `human_mpc_gui.py` | Tkinter GUI: drag-and-drop action input, angle slider, Refine/Submit/Finish Episode/New Episode |
| `simple_mpc/config/config_human_demo.yaml` | Config: same `dataset`/`mpc` shape as `config_oracle.yaml` (smaller default `mpc.n_envs`, sized for grid-search batching rather than optimizer population) plus a `human:` section (`grid_n`, `grid_delta`) |
| `tests/test_action_to_pose.py` | `action_to_pose` 4D/5D branch, batching, Genesis-free |
| `tests/test_human_grid_search.py` | `build_action_grid` shape/centering/bounds/broadcast, Genesis-free |

## Config reference

`config_human_demo.yaml`'s `dataset`/`mpc` sections follow
`config_oracle.yaml`'s schema (see `oracle_mpc_design.md`'s config
reference) with two differences: `mpc.n_envs` defaults much smaller (16 vs.
32+) since it's sized for grid-search candidate batching in an interactive
session, not automated-optimizer population coverage; and `mpc.n_mpc` is a
*soft* cap (`HumanDemoSession.finished()`), not enforced by
`GenesisOracleEnv` itself. The new `human:` section:

| Key | Meaning |
|---|---|
| `grid_n` | Grid points per dimension; total candidates per `propose()` call = `grid_n ** 5` |
| `grid_delta` | Normalized half-width of the search window (see "Local grid search" above) — a scalar (broadcasts to all 5 dims) or a length-5 list |

## Usage

```bash
python human_mpc_gui.py
python human_mpc_gui.py --config simple_mpc/config/config_human_demo.yaml
```

Per step: drag the ORANGE (start) and CYAN (end) handles, optionally set the
angle slider (or leave it on **Auto**), then either click **Submit** directly
(executes exactly the drawn action) or click **Refine** first (optional —
waits for a `grid_n ** 5`-candidate physics rollout, then updates the fields
to the refined action and shows the predicted reward) and Submit that
instead. **Finish Episode** saves whatever was collected so far; **New
Episode** resets and starts the next one.

## Known limitations

- **No predicted-occupancy heatmap tile yet.** `grid_search_refine` already
  computes `occ_pred`; only the GUI's tile rendering is deferred (see "GUI"
  above) — a natural next increment once the "just state + drag" baseline
  is validated.
- **Grid search is exhaustive, not adaptive.** Unlike CEM's iterative
  refit, one `propose()` call is a single fixed-resolution grid pass — a
  human re-drawing/re-proposing after seeing the result is the only
  "iteration" available. A zoom-in second pass (recentre a finer grid on
  the first pass's winner) would be a natural extension if `grid_n ** 5` at
  a single resolution proves too coarse or too slow.
- **Synchronous/blocking GUI.** Refine and Submit freeze the window for
  their rollout's duration (see "GUI" above) — acceptable at
  `config_human_demo.yaml`'s default small `n_envs`/`grid_n`, but will feel
  worse if either is scaled up.
- **Not wired into `run_experiments.py`.** Like `run_oracle_mpc.py`
  (`oracle_mpc_design.md`'s own known limitation), this is a standalone
  entry point.
