# Oracle MPC — Genesis-as-Model Sampling MPC (MPPI / CEM)

**Goal:** establish a *ceiling baseline* for MPC performance on granular pile
manipulation by removing model error entirely: the MPC's prediction step **is
the Genesis simulator itself**. Any remaining gap to the goal then measures the
limits of the optimizer / horizon / action parameterization — not the dynamics
model. Learned-model MPC runs can be compared against this ceiling.

Because the simulator is not differentiable end-to-end through
`execute_action`, gradient-descent optimization (as in
`simple_mpc/mpc.py::run_simple_mpc`) is replaced by **sampling-based
optimization**. Both **MPPI and CEM** will be implemented — their skeleton is
identical (sample candidates → evaluate costs → update sampling distribution),
they differ only in the distribution-update rule (~15 lines each), so
supporting both costs almost nothing and lets us report whichever is stronger.

---

## 1. Key design decisions

### 1.1 One Genesis scene, `n_envs` parallel copies, env 0 is "reality"

Genesis allows `gs.init()` only once per process and a scene's `n_envs` is
fixed at `build()`. Therefore we cannot have a separate 1-env "real" sim plus a
K-env "planning" sim. Instead (mirroring `Genesis/data_collection_clean.py`):

- Build **one** `SandboxManipulation(config, n_envs=K)`.
- **Env 0 is the canonical "real" environment.** Its particle state is the
  ground truth; the overhead camera is placed above env 0's grid offset
  (`scene.envs_offset[0]`) so `render()` sees only env 0 (env spacing =
  2 × box size, same as data collection, keeps other envs out of frustum).
- **All K envs serve as rollout workers during planning.** Before each
  optimizer iteration, env 0's state is broadcast to all envs; each env then
  executes a *different* candidate action (this is exactly the per-env batched
  action path `execute_action(p_start[K,3], p_stop[K,3], angle[K])` already
  used by data collection).
- **Executing the chosen action:** restore the snapshot, broadcast the winning
  action to *all* envs, and step them in lockstep. All envs stay in identical
  states between MPC steps (deterministic sim), and the realized outcome equals
  the winning rollout — so `predicted_cost == actual_cost` becomes a built-in
  sanity check that state sync is correct.

### 1.2 State snapshot / restore

A snapshot is `(pos [K, n_p, 3], quat [K, n_p, 4])` read exactly as
`SandboxManipulation.update_material_state` does
(`rigid_solver.get_links_pos/quat(links_idx=_particle_links_idx)`), taken
*after* the settle phase so velocities ≈ 0.

Restore broadcasts env 0's slice to all envs:

- `particle.set_pos(pos0.expand(K, 3), envs_idx)` / `set_quat(...)` per
  particle entity (same API `shuffle_particles` uses),
- zero particle velocities via
  `rigid_solver.set_dofs_velocity(zeros, dofs_idx=_particle_dofs_idx)`,
- park the plate at its lifted pose (as `execute_action` leaves it).

This is a new method on the wrapper, not a change to `SandboxManipulation`
(only additions there if strictly needed, e.g. exposing a
`get/set_particle_snapshot` pair — see §2.1).

### 1.3 Cost = loss functions from `training/losses.py`

The MPC cost reuses the **existing loss registry** (`build_loss`) unchanged:

- After a candidate rollout, particle positions are converted to a normalized
  camera-frame occupancy grid (same convention as the Eulerian pipeline:
  `x_n = x/global_scale`, `y_n = −y/global_scale`, `z_n = 0.5` at the table;
  bounds from `EulerianModelWrapper.default_bounds(cfg, convention='genesis')`).
- The loss is called with the standard contract:

  ```python
  prediction = ModelOutput(
      logits=torch.logit(occ_pred.clamp(eps, 1 - eps)),  # losses expect logits
      probabilities=occ_pred,
  )
  batch = {
      "input":  occ_cur.unsqueeze(1),   # (K, 1, H, W) — current occupancy
      "target": occ_goal.expand(K, H, W),  # binary goal-region mask
  }
  total, components = loss_fn(prediction, batch)   # cost per candidate
  ```

  One wrinkle: `EulerianCombinedLoss` reduces over the batch (returns a
  scalar). For MPC we need **per-candidate** costs, so the loop calls the loss
  once per candidate is too slow; instead a thin `PerSampleLoss` wrapper in
  `training/losses.py` evaluates the registered loss with reduction over
  spatial dims only. Concretely: add an optional `reduction: "none"` pathway
  (`cfg["per_sample"] = True`) to `EulerianCombinedLoss` that returns a
  `(B,)` tensor — backward compatible (default unchanged, existing tests
  untouched), and every future loss added to the registry gets used by both
  training and oracle MPC.
- Loss type + weights come straight from config
  (`mpc.loss: {type: eulerian_combined, mse: 1.0, dice: 0.5, ...}`). New loss
  terms you request later get added to `losses.py` and become available to
  both training and this MPC automatically.
- No gradients are needed — everything runs under `torch.no_grad()`; logits of
  a hard 0/1 grid are fine since only relative ranking matters.

**Occupancy density caveat:** the Eulerian pipeline derives occupancy from a
dense depth image, while particle *centers* fill ≤ n_p voxels. To make the two
comparable, particle splatting will use each particle's **footprint** (known
`particle_size`) — a hard disk of radius `size/2` in grid units — implemented
as a small extension in `transforms/functional.py`
(`particles_to_occupancy(..., footprint_radius=r)`), reusing the existing
scatter machinery.

**Reporting stays on the existing path:** after each *real* step, env 0 is
rendered and `rewards` / `occ_rewards` are computed exactly as in
`run_simple_mpc` (occupancy × score map with `empty_penalty`), so oracle runs
are directly comparable with the learned-model MPC result dicts.

### 1.4 MPPI and CEM share one skeleton

```python
class SamplingOptimizer(ABC):
    def reset(self, init_mean):            # (H, 4) — warm-startable
    def ask(self, n) -> Tensor[n, H, 4]:   # sample candidates, clipped to bounds
    def tell(self, candidates, costs):     # update distribution
    def best(self) -> Tensor[H, 4]         # current best sequence
```

- **CEM:** keep `n_elite` lowest-cost candidates, refit diagonal Gaussian
  (mean, std) with optional momentum; std floor to avoid collapse.
- **MPPI:** weights `softmax(−cost / λ)`, mean ← weighted average of
  candidates, β-filter smoothing across MPC steps (config already carries
  `mppi.beta_filter`); fixed exploration σ per action dim.
- Initial mean/samples come from the existing samplers in
  `simple_mpc/action_sampler.py` (physics-aware bounds:
  `v = wkspc_w − plate_length/2 − safety_margin`), and all samples are clipped
  to those bounds — same clipping as `run_simple_mpc`.
- **Warm start:** between MPC steps the mean sequence is time-shifted by one
  action (standard receding-horizon warm start), refreshed with a fresh random
  tail.

### 1.5 Candidate budget vs. env count

`n_envs = K` is fixed at build; per optimizer iteration we evaluate exactly
`K` candidates (one per env). If config asks for more samples per iteration
(`n_sample > K`), the iteration runs `ceil(n_sample / K)` sequential batches
with a snapshot-restore between batches. Default config keeps
`n_sample == n_envs` to avoid that cost.

### 1.6 Horizon > 1

A candidate is a *sequence* `(n_look_ahead, 4)`. Rollout = sequential
`execute_action` calls per env (each env runs its own sequence; already
batched). Cost is evaluated on the **terminal** state by default; an optional
per-step discounted sum (`cost_mode: terminal | discounted`, `gamma`) is a
cheap addition since we settle + read state after every push anyway.

### 1.7 Rollout-time physics shortcuts (config knobs, not defaults)

Each simulated push costs roughly: clearance lower (~14 steps) + sweep
(~200–600 steps depending on distance at 0.125 m/s, dt=4 ms) + clearance lift
(~14) + settle (100). To trade fidelity for speed **during planning only**:

- `rollout_settle_steps` (default = `settle_steps`): reduced settle after
  candidate pushes; the real step always uses the full `settle_steps`.
- Failed candidates (plate never reached its target — `reached_goal=False`)
  are **not** discarded: the partial push is a real, reproducible outcome, so
  its cost is evaluated normally. (Optionally log the failure rate.)

---

## 2. Files

### 2.1 New: `simple_mpc/genesis_oracle.py`

`GenesisOracleEnv` — batched analogue of `env/genesis_env.py::GenesisEnv`
(shares its config-builder and camera/obs conventions; refactor
`_build_genesis_config` into a shared helper rather than duplicating it):

```python
class GenesisOracleEnv:
    def __init__(self, cfg: dict, n_envs: int): ...
    # -- real-env API (env 0), signature-compatible with GenesisEnv --
    def render(self) -> np.ndarray                      # (H, W, 5), env 0
    def step(self, action_4d, video_recorder=None)      # broadcast to all envs
    def reset(self)                                     # shuffle env 0, broadcast
    def get_cam_params(self) / get_cam_extrinsics(self)
    # -- planning API --
    def snapshot(self) -> dict                          # env-0 particle pos/quat
    def restore_broadcast(self, snap)                   # env0 state → all envs
    def rollout_candidates(self, act_seqs: Tensor[K, H, 4],
                           settle_steps: int) -> Tensor[K, n_p, 3]
        # restore → for s in range(H): execute_action(per-env) → settle
        # returns world-frame particle positions per env (terminal or per-step)
    def particles_to_occ(self, pos_world: Tensor[K, n_p, 3]) -> Tensor[K, Nx, Ny]
        # world → normalized cam coords → footprint splat
```

Note: `reset()` shuffles env 0 only (or shuffles all and then broadcasts
env 0 — simpler and equivalent), then broadcasts so all envs start identical.

### 2.2 New: `simple_mpc/sampling_optimizers.py`

`SamplingOptimizer` base + `CEMOptimizer` + `MPPIOptimizer` +
`make_sampling_optimizer(name, cfg, act_lo, act_hi, sampler)`. Pure
torch/numpy — no Genesis import — so it is unit-testable on synthetic cost
functions.

### 2.3 New: `simple_mpc/oracle_mpc.py`

`run_oracle_mpc(env: GenesisOracleEnv, subgoal, cfg, video_recorder=None, ...)`
— mirrors `run_simple_mpc`'s structure and **returns the same result-dict
schema** (`rewards`, `occ_rewards`, `raw_obs`, `states`, `actions`,
`states_pred`, `rew_means`, `rew_stds`, `best_rewards_per_step`, timing
fields) so all existing metric extraction / plotting in `run_experiments.py`
works unchanged. Per-step loop:

```
snapshot ← env.snapshot()
optimizer.reset(warm_started_mean)
for it in range(n_opt_iter):
    cand ← optimizer.ask(K)                     # (K, n_ahead, 4)
    pos  ← env.rollout_candidates(cand, rollout_settle_steps)
    occ  ← env.particles_to_occ(pos)
    cost ← loss_fn_per_sample(occ, occ_cur, occ_goal)     # (K,)
    optimizer.tell(cand, cost); log mean/std → rew_means/rew_stds
best ← optimizer.best()
env.restore_broadcast(snapshot)
obs ← env.step(best[0])                          # broadcast winner, full settle
report rewards via existing occupancy-reward path; loop
```

`states_pred` stores the winning candidate's terminal occupancy (from its
planning rollout), giving the same predicted-vs-actual diagnostics as the
learned models — for the oracle this gap should be ≈ 0 and is the primary
correctness check.

### 2.4 New: `simple_mpc/config/config_oracle.yaml`

Copy of `config_simple.yaml`'s `dataset` section plus:

```yaml
mpc:
  optimizer: 'cem'          # 'cem' | 'mppi'
  n_mpc: 8
  n_look_ahead: 1
  n_envs: 32                # parallel Genesis envs = candidates per iteration
  n_sample: 32              # per optimizer iteration (> n_envs ⇒ sequential batches)
  n_opt_iter: 4             # optimizer iterations per MPC step
  rollout_settle_steps: 40  # settle during planning rollouts (real step uses dataset.settle_steps)
  cost_mode: 'terminal'     # 'terminal' | 'discounted'
  gamma: 0.9
  cem:  {n_elite: 8, momentum: 0.5, std_floor: 0.005}
  mppi: {lambda: 0.1, sigma: 0.02, beta_filter: 0.7}
  loss:                     # passed to training.losses.build_loss (per_sample: true)
    type: 'eulerian_combined'
    mse: 1.0
    dice: 0.0
    bce: 0.0
  reward:                   # reporting only — same as existing runs
    empty_penalty: 0.2
  action_sampler: 'physics_aware'
  task: {type: 'target_shape', target_char: 'T'}
  output_dir: 'outputs/oracle_mpc'
```

### 2.5 New entry point: `run_oracle_mpc.py` (repo root)

Thin script: load config → build `GenesisOracleEnv` → build subgoal via the
same `gen_goal_shape` / `gen_subgoal` + `scale_subgoal_to_material_pixels`
path used by `run_experiments.py` → `run_oracle_mpc` → save result dict +
video. Integration into `run_experiments.py`'s batch machinery can follow
later once the baseline is validated (kept out of scope to avoid touching the
experiment runner in the first PR).

### 2.6 Modified files (minimal)

| File | Change |
|---|---|
| `training/losses.py` | optional `per_sample` (reduction-free) mode on `EulerianCombinedLoss`; future requested losses land here |
| `transforms/functional.py` | `footprint_radius` option in `particles_to_occupancy` (hard-disk splat) |
| `env/genesis_env.py` | extract `_build_genesis_config` so the oracle env reuses it (no behavior change) |
| `Genesis/sandbox_manipulation_clean.py` | only if needed: small `get_particle_state()` / `set_particle_state()` helpers mirroring existing read/write code |

### 2.7 Tests: `tests/test_sampling_optimizers.py`, `tests/test_oracle_cost.py`

- Optimizers converge on a known quadratic cost over the action box (no
  Genesis; fast).
- CEM std never below floor; MPPI weights sum to 1; warm-start shift correct.
- `per_sample` loss mode: `(B,)` output matches per-item scalar calls; scalar
  mode unchanged.
- Footprint splat: known particle layout → expected voxel count; matches
  depth-derived occupancy on a rendered frame within tolerance (integration
  test, marked slow / requires GPU + Genesis).
- Snapshot→restore→zero-action→state-unchanged round-trip (slow test).

---

## 3. Runtime estimate (for choosing defaults)

Per candidate push ≈ 350–700 scene steps (lower + sweep + lift + reduced
settle), executed batched across `n_envs`. Per MPC step with defaults
(`n_opt_iter=4`, `n_look_ahead=1`): ~1.5–3k batched scene steps, plus one real
step. At typical Genesis rigid-solver throughput for ~32 envs × ~25 bodies
this is expected to be minutes-per-MPC-step territory, not seconds — fine for
a ceiling baseline, and the reason `rollout_settle_steps` and `n_envs` are the
primary tuning knobs. A `--benchmark` flag in the entry script will print
steps/sec after the first iteration so budgets can be set empirically.

---

## 4. Open questions for review with answers after each one

1. **`n_envs` default (32?)** — bounded by GPU memory / build time. Happy to benchmark 16/32/64 first and bake in the best default.
 - Please write a small script to benchmark, starting from 4 envs.
2. **Reduced settle during rollouts** (`rollout_settle_steps: 40` vs the full 100) — acceptable fidelity trade for planning, or keep full settle for a "pure" ceiling?
 - I would suggest reducing steps for all of the phases except sweep, by a factor of 2.
3. **Cost target**: default plan uses the binary goal mask as `target` for the losses. Do you also want a *score-map-weighted* loss (distance-transform + `empty_penalty`, i.e. the existing reward as a registered loss in `losses.py`) so optimization and reporting use literally the same objective?
 - Not sure, add the option if not costly to implement.
4. **Both optimizers ship** per §1.4 (shared skeleton makes this cheap) —
   confirm you want both wired to config rather than CEM only.
 - I want both.
5. **Entry point**: standalone `run_oracle_mpc.py` first, `run_experiments.py` integration as a follow-up — OK?
 - OK.
