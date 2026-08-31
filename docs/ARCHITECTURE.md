# Architecture

This is the repository map and extension guide. Data contracts and coordinate conventions are documented in `INTERFACES.md`. Utility ownership and transform guidance are documented in `UTILITIES.md`. The Genesis-as-model sampling MPC (CEM/MPPI ceiling baseline) has its own reference doc, `oracle_mpc_design.md`, since its design has enough moving parts (batched multi-env rollout, snapshot/restore state management, sampling optimizers) to warrant a dedicated write-up; this file only maps where its pieces live. The human-piloted variant of that same ceiling baseline (a person picks the action each step instead of CEM/MPPI, refined by a local grid search) has its own doc, `human_demo_design.md`, for the same reason. `scaling_to_200_objects.md` is the reference for the simulator's non-obvious settings — why the solver budget, contact-pair cap, settling criterion, particle spawn and pusher-plate actuator model hold their current values, and what large piles cost — and should be consulted before changing anything under `Genesis/configs/` or `SandboxManipulation`'s physics setup. `plate_model.md` covers the pusher tool specifically — its actuator model, the three phases of a push, and `reaction_report()`, which gives the peak actuator force, torque and granular reaction per action for setting real-robot limits. `genesis_world_migration.md` records the move from Genesis 0.4.5 to Genesis World 1.3.3 and the breaking changes it required.

> **Documentation is part of the change, not a follow-up.** Any change to a
> major information flow, a module's responsibility, a default value, or an
> added feature must update the relevant doc(s) below — `ARCHITECTURE.md`
> (module map, philosophy), `INTERFACES.md` (data contracts), `UTILITIES.md`
> (utility ownership), `oracle_mpc_design.md` (oracle MPC internals) — in the
> same change that makes the change, not as a separately-requested pass
> afterward. See `.claude/skills/project-overview/SKILL.md` for the
> project-level pointer to all of these.

## Design Philosophy

Recurring patterns worth preserving when extending this codebase:

- **Stateless transforms, stateful wrappers.** `transforms/functional.py` holds
  pure, dependency-light conversion functions (particle↔occupancy, action↔camera
  coordinates); heavier stateful classes (`model/eulerian_wrapper.py`,
  `simple_mpc/`) call into it, never the other way around. New geometry/representation
  conversions belong in `transforms/`, not copy-pasted into whichever wrapper needs them first.
- **Adapter pattern decouples the optimizer from the model.** `simple_mpc/mpc.py`'s
  gradient-descent loop never branches on model type; `simple_mpc/adapters.py`
  exposes a uniform `obs_to_state` / `predict_step` / `compute_reward` surface
  (INTERFACES.md §3.4) per model family. `simple_mpc/oracle_mpc.py` has no
  adapter at all — the "model" is the Genesis simulator itself, so there is
  nothing to abstract over — but it reuses the same loss registry and occupancy
  utilities as the adapters do, so the two MPC variants stay comparable rather
  than diverging into parallel implementations.
- **One loss registry serves both training and MPC cost.** `training/losses.py`
  losses are the single source of truth for "how good is this occupancy
  relative to a target" — used with a batch-mean reduction during training and
  a `per_sample=True` reduction when ranking MPC candidates. A loss added for
  one use case is available to the other for free; don't hand-roll a
  cost-specific duplicate of a training loss.
- **The simulator core stays purpose-agnostic.** `Genesis/sandbox_manipulation_clean.py`
  exposes generic hooks (`execute_action(..., on_phase=callback)`,
  `set_particle_state`/`broadcast_state_from_env`) rather than baking in
  camera, video, or MPC-specific logic. Anything that needs a camera or cares
  about *why* a push is happening lives one layer up, in the env wrappers
  (`env/genesis_env.py`, `simple_mpc/genesis_oracle.py`) or MPC loops — never
  in the simulator wrapper itself. `push_and_record`/`flush_transitions`
  (automatic MPC transition recording, `Genesis/transition_buffer.py`) are the
  one exception the core owns directly rather than exposing as a hook —
  justified because recording only touches state the core already tracks
  (`_particle_state`) and needs zero downstream-specific knowledge (no
  camera, no reward, no "MPC step" concept beyond an opaque tag the caller
  supplies) — it's generic bookkeeping, not a purpose-specific feature.
- **Shared utilities over duplicated logic.** Cross-cutting operations used by
  more than one MPC variant or script (`write_video_frame`,
  `particles_to_occupancy`, `footprint_radius_voxels`, `OccupancyReward`) live
  in `utils.py` / `transforms/functional.py` / `simple_mpc/occupancy_reward.py`
  and are imported, not re-derived. If you find yourself copying a snippet a
  second time, that's the signal to extract it instead.
- **Config-driven defaults.** Hyperparameters and physical constants live in
  YAML (`simple_mpc/config/*.yaml`), not hardcoded in Python, so swapping
  models, optimizers, or physical setups doesn't require code edits.
- **Docs mirror code boundaries.** Each doc file's scope matches a real code
  boundary (module map, data contracts, utility ownership, one subsystem's
  design) rather than being a single unstructured wiki page — this is what
  makes "update the docs" a small, targeted diff instead of a chore, and why
  it belongs in the same change as the code.

## Module Map

```
physics/
  normalization.py      PhysicsBounds (canonical raw<->normalized mapping)

registry/
  model_registry.py     register_model, build_model, ModelTrainingWrapper
                        (+ EulerianTrainingWrapper, LagrangianTrainingWrapper)
  dataset_registry.py   register_dataset, build_dataset, EulerianDatasetWrapper,
                        LagrangianDatasetWrapper, GenesisParticlePushDataset

model/
  model_card.py         ModelCard save/load, load_model_from_card (MPC wrapper),
                        load_net_from_card (bare network for eval scripts)
  NFDUNetFilm.py        NFDUNetFiLM / NFDUNetFiLMShallow — the U-Net
                        implementations built by the "unetfilm*" factories
  eulerian_wrapper.py   EulerianModelWrapper (MPC-facing occupancy wrapper),
                        heuristic push models (splat/spread/spread2/cumulative/
                        fluid) plus register_push_model/build_push_model — a
                        parallel, checkpoint-free registry (these have no
                        learned weights, so they don't go through
                        registry/model_registry.py or model_card.py),
                        UNetFiLMPushModel, particle<->occupancy helpers
  diff_mass_push.py     differentiable mass-push kernels used by the
                        heuristic push models
  gnn_dyn.py            PropNetDiffDenModel (Lagrangian GNN dynamics)
  futureintegration/    salvaged architectures; three are now registered
                        (see below), the rest await consolidation/are
                        skipped/broken — see its README for the breakdown
    NCAModels.py            NCAWithPhysics       → model type "nca"
    SpatTransNet.py         EulerianSTN          → model type "spatial-transformer"
    UNetModels_modular.py   UNet (config-driven) → model type "unet-modular"

transforms/
  functional.py         particles_to_occupancy (+ footprint_radius hard-disk
                        splat, shape_factor for non-spherical particles),
                        footprint_radius_voxels, draw_plate_soft,
                        genesis_action_to_cam3d, genesis_particles_to_cam3d,
                        build_action_delta, action_to_pose (shared 4D-derived-
                        yaw / 5D-explicit-yaw action convention — see
                        docs/human_demo_design.md)
  representation.py     Compose, EnsureRepresentation, occupancy/particle
                        alias transforms, build_transforms

training/
  types.py              TrainingBatch and ModelOutput contracts
  losses.py             register_loss, build_loss, EulerianCombinedLoss
                        (per_sample=True reduction mode for MPC-candidate
                        costs, in addition to the scalar training reduction),
                        ScoreMapWeightedLoss (loss = -occupancy·score_map, so
                        MPC optimization can share the exact reward-reporting
                        objective), LagrangianMSELoss
  metrics.py            build_metrics, EulerianMetrics, LagrangianMetrics
  trainer.py            Trainer loop and config include resolution
  train.py              CLI entry point

simple_mpc/
  mpc.py                run_simple_mpc gradient-descent MPC loop (learned/
                        heuristic models, via adapters)
  adapters.py           EulerianAdapter, GNNAdapter, make_adapter
                        (the adapter surface defined in INTERFACES.md §3.4)
  action_sampler.py     candidate-action samplers (uniform, physics-aware,
                        collision-aware, OT-guided)
  ot_planner.py         OTPlannerSparse (Sinkhorn OT action initializer)
  occupancy_reward.py   OccupancyReward (goal-mask scoring: compute_score_tensor
                        for the reward map, goal_occupancy_mask for a plain
                        binary loss target)
  benchmark.py          adapter/push throughput benchmarks
  debug_vis.py          MPC debug visualization panels/videos
  genesis_oracle.py     GenesisOracleEnv — batched multi-env Genesis wrapper
                        for oracle MPC (see docs/oracle_mpc_design.md).
                        step()/rollout_candidates() record via
                        push_and_record (real steps flush immediately; tagged
                        is_candidate=True planning rollouts accumulate until
                        the next real step); set_recording_context(...) tags
                        an episode's flushes before it runs
  oracle_mpc.py         run_oracle_mpc, load_oracle_config — sampling MPC
                        where the Genesis simulator itself is the prediction
                        model; no adapter needed (see docs/oracle_mpc_design.md)
  sampling_optimizers.py CEMOptimizer, MPPIOptimizer, make_sampling_optimizer
                        (shared ask/tell/best skeleton; gradient-free)
  human_grid_search.py  build_action_grid, grid_search_refine — local 5D grid
                        search around a human-drawn action, via the oracle
                        simulator (see docs/human_demo_design.md)
  human_mpc.py          HumanDemoSession (propose/commit/finished/finalize),
                        save_episode — interactive human-piloted episodes
                        over GenesisOracleEnv, parallel to (not built on)
                        run_oracle_mpc's automated loop
  config/               base MPC config, config_oracle*.yaml,
                        config_human_demo.yaml, experiments/

RealData/
  dataset.py            RealPileSweepData (real camera data; mirrors the
                        PileSweepData output format)

configs/
  model/                one YAML per architecture
  dataset/              one YAML per data source
  training/             experiment YAMLs composing model+dataset+training

Genesis/
  sandbox_manipulation_clean.py  SandboxManipulation — low-level multi-env
                        simulator wrapper (build/reset/execute_action/
                        update_material_state). execute_action's on_phase
                        callback (fired at 'post_lower' / 'post_sweep',
                        no-op if omitted), set_particle_state /
                        broadcast_state_from_env, and push_and_record /
                        flush_transitions (automatic transition recording,
                        see below) are generic, purpose-agnostic primitives —
                        reused by env/genesis_env.py,
                        simple_mpc/genesis_oracle.py, and this module's own
                        data_collection_clean.py, not oracle-specific.
  state_library.py       StateLibrary / build_state_library — a bank of
                        *settled* pile states, so a reset is a pose write
                        rather than a re-settle (measured ~6 ms vs ~37 s at
                        200 particles). Settles are amplified by the
                        container's symmetry group (D4, x8, for a square
                        tray), which is why a dozen settles yield a hundred
                        distinct starts. Genesis-free (torch only),
                        unit-tested in tests/test_state_library.py. Opt-in
                        from data_collection_clean.py via --state-library.
  placement_sampling.py  configuration-space sampling of tool touchdown
                        poses that do not collide with the pile: occupancy
                        grid -> rotated-rectangle dilation per yaw bin
                        (Minkowski sum) -> free set, with a Euclidean
                        distance transform for clearance-biased draws.
                        Consumed by SandboxManipulation.generate_action_
                        samples(placement_aware=True), which falls back to
                        the blind draw per sample when the free set is empty.
                        Genesis-free (torch/numpy/scipy), unit-tested in
                        tests/test_placement_sampling.py.
  action_sampling.py     batch-aware action shaping AND action-space
                        restriction, both pure torch (no `import genesis`),
                        unit-tested in tests/test_action_sampling.py.
                        Shaping: equalize_travel_distance /
                        shared_batch_distance share one push length across a
                        lockstep batch (the sweep is sized from the LONGEST
                        travel, so independent lengths make every env run for
                        the longest one's duration).
                        Restriction: blade_normal / sampling_box /
                        constrain_push / relative_blade_angle implement the
                        perpendicular-push and fixed-push-length constraints
                        that the switched-linear visual-foresight baseline
                        needs (docs/linear_visual_foresight_baseline.md §7),
                        exposed as generate_action_samples(
                        perpendicular_pushes=, push_length=) and as
                        data_collection_clean.py's --perpendicular-pushes /
                        --push-length. ray_box_max_travel is the shared
                        primitive. Where a constrained push would leave the
                        tray the START is moved, never the push length —
                        shortening would silently drop a transition out of
                        its length bin.
  transition_buffer.py   TransitionBuffer — accumulates and saves the
                        before/after/action transitions push_and_record
                        records, in the same on-disk format
                        data_collection_clean.py's dataset files use.
                        Genesis-free (no `import genesis`), independently
                        unit-tested.
  data_collection_clean.py  batched random-push dataset collection; the
                        multi-env execute_action(p_start[K,3], p_stop[K,3],
                        angle[K]) pattern that GenesisOracleEnv's rollout
                        batching mirrors. Does not call push_and_record (has
                        its own save path), so transition recording is a
                        no-op here regardless of config. Action-space
                        restrictions (--perpendicular-pushes, --push-length)
                        are recorded in each batch's saved config, so a
                        restricted dataset is identifiable on disk.
  configs/collection_foresight_single_operator.yaml
                        run_collection.py plan for the targeted collection of
                        ONE switched-linear transition operator (one push
                        length, perpendicular, one pile size / material).
  benchmark_n_envs.py    throughput sweep for picking simple_mpc.oracle_mpc's
                        n_envs default; run as `python -m Genesis.benchmark_n_envs`
                        (must run as a module from the repo root — see its
                        docstring)
  training/dataset.py    PileSweepData — raw Genesis sim dataset, wrapped by
                        the "genesis" entry in dataset_registry.py; also
                        loads push_and_record's output (superset schema,
                        extra keys ignored) if pointed at its output directory

env/
  genesis_env.py         GenesisEnv — single-env bridge from
                        SandboxManipulation to the MPC-facing
                        observation/action interface used by
                        simple_mpc/mpc.py. step() records each real push via
                        push_and_record and flushes it to disk immediately
                        (flush_after=True); set_recording_context(...),
                        called once per episode before it runs, tags those
                        per-step flushes with episode identity.

tests/
  test_configurable_unet.py       registry/model smoke tests (pytest)
  test_futureintegration_models.py  nca/spatial-transformer/unet-modular smoke tests
  test_model_card.py             model_card save/load round-trip
  test_push_model_registry.py    build_push_model + EulerianModelWrapper
                                  end-to-end forward for all 5 heuristics
  test_losses_per_sample.py      EulerianCombinedLoss/ScoreMapWeightedLoss
                                  per_sample reduction mode (Genesis-free)
  test_footprint_splat.py        particles_to_occupancy footprint_radius +
                                  shape_factor, genesis_particles_to_cam3d
                                  (Genesis-free)
  test_sampling_optimizers.py    CEM/MPPI convergence, bounds, warm-start
                                  (Genesis-free, synthetic cost function)
  test_transition_buffer.py      TransitionBuffer append/save schema
                                  round-trip (Genesis-free)
  test_action_to_pose.py         transforms.functional.action_to_pose 4D vs
                                  5D branch, batching (Genesis-free)
  test_human_grid_search.py      build_action_grid shape/centering/bounds/
                                  broadcast (Genesis-free)
  test_state_library.py          symmetry-expansion maths for the settled-state
                                  bank: D4 group size, mirrored quaternions are
                                  proper rotations, rigidity (Genesis-free)
  test_placement_sampling.py     C-space free-placement maths: occupancy,
                                  rotated-rectangle dilation, distance
                                  transform, empty-free-set degradation
                                  (Genesis-free)
  scaling_investigation/         archived probes, raw measurements and the
                                  settling diagnosis behind
                                  docs/scaling_to_200_objects.md. NOT part of
                                  the pytest suite — one-shot GPU measurement
                                  scripts, run as
                                  `python -m tests.scaling_investigation.<x>`.
                                  verify_fixes.py and verify_new_features.py
                                  there are the end-to-end assertions worth
                                  re-running after touching
                                  Genesis/sandbox_manipulation_clean.py;
                                  probe_collection_health.py and
                                  record_simulation_video.py are the pair to
                                  run before trusting a long collection —
                                  the first checks a realistic run's output
                                  statistics (goal-reach rate, per-env
                                  settledness, action-space bias, escaped
                                  particles), the second renders it to video
                                  for whatever nobody thought to measure.
                                  Findings from both are in section 8 of
                                  docs/scaling_to_200_objects.md.

utils.py                shared root-level helpers (YAML I/O, action geometry,
                        point-cloud ops, goal-shape generation,
                        write_video_frame) used by model/eulerian_wrapper.py,
                        env/genesis_env.py, and simple_mpc/
```

Entry points beyond `training/train.py`:

```
run_experiments.py       MPC experiment-suite runner; loads trained models via
                         model cards, or heuristic push models via
                         model.eulerian_wrapper.build_push_model (model.type:
                         eulerian, need_weights: false); writes
                         outputs/experiments/ (per-episode/experiment data;
                         real-step transitions additionally flow to the
                         shared Genesis/data/mpc_runs/ pool — see UTILITIES.md §1.2)
run_experiment_batch.py  subprocess driver running run_experiments.py over
                         multiple suite YAMLs
run_oracle_mpc.py        Oracle (Genesis-as-model) MPC entry point — builds
                         one GenesisOracleEnv, runs episodes.n_episodes
                         episodes, saves full per-episode trajectories
                         (raw frames, point clouds, predicted-vs-actual
                         occupancy, video) — see docs/oracle_mpc_design.md
human_mpc_gui.py         Human-demonstration GUI over the same
                         GenesisOracleEnv — drag-and-drop action input,
                         local grid-search refinement, full multi-step
                         episodes saved in run_oracle_mpc.py's schema — see
                         docs/human_demo_design.md
debug_mpc_gui.py         Interactive learned/heuristic-model MPC debugger
                         (single-env GenesisEnv, gradient-descent action
                         refinement) — human_mpc_gui.py reuses its
                         tile-image helpers and canvas-drag interaction
visualize.py             dataset / occupancy / prediction visualization
```

Genesis-dependent modules (`env/genesis_env.py`, `simple_mpc/genesis_oracle.py`,
`Genesis/*`) are only importable where the `genesis` package is installed;
everything else in this map is Genesis-free. `simple_mpc/human_grid_search.py`
is a partial exception: `build_action_grid` is plain NumPy and Genesis-free,
but `grid_search_refine` needs a real `GenesisOracleEnv` — the `genesis`-
requiring import is deferred inside that one function so the module itself
stays importable without `genesis` installed (see docs/human_demo_design.md).

## Data Flow

```
Training config YAML
  ├─ model:   configs/model/*.yaml      ──► build_model(cfg)   ──► ModelTrainingWrapper
  └─ dataset: configs/dataset/*.yaml    ──► build_dataset(cfg) ──► Dataset

Trainer.run()
  for batch in DataLoader:
      prediction = model_wrapper(batch)
      loss, comps = loss_fn(prediction, batch)
      metrics.update(prediction, batch)

  -> saves checkpoint + model_card.yaml
```

## Extension Points

### Add a Model

1. Implement an `nn.Module`.
2. Register a factory in `registry/model_registry.py`.
3. Add a config in `configs/model/`.
4. Reference it from a training config.

Example:

```python
@register_model("mymodel")
def _build_mymodel(cfg: dict) -> EulerianTrainingWrapper:
    from mypackage import MyModel
    model = MyModel(in_channels=cfg.get("in_channels", 2))
    return EulerianTrainingWrapper(model, uses_physics=False)
```

### Add a Dataset

1. Implement a builder in `registry/dataset_registry.py`.
2. Register it with `@register_dataset("name")`.
3. Ensure outputs match the standard batch contract in `INTERFACES.md`.
4. Add a dataset config in `configs/dataset/`.

### Add a Loss

1. Subclass `LossFn` in `training/losses.py`.
2. Register it with `@register_loss("name")`.
3. Reference it under `training.loss` in your training config.

## Config Structure

Top-level training config schema:

```yaml
model: configs/model/unetfilm.yaml      # must resolve to a dict with a `type` key
dataset: configs/dataset/genesis_cube.yaml

training:
  epochs: 100
  batch_size: 64
  lr: 1e-4
  lr_scheduler:            # optional StepLR params
    type: StepLR
    step_size: 100
    gamma: 0.75
  augmentation: true
  patience: 100            # early stopping
  mixed_precision: true
  grad_clip_norm: 1.0
  save_every_n_epochs: 10
  num_workers: 4
  loss:
    type: eulerian_combined   # default depends on model.type
                              # (lagrangian_mse for gnn-propnet)
    mse: 1.0
    mass: 0.2

# The inference block is NOT read during training: it is copied verbatim
# into model_card.yaml and consumed later by load_model_from_card
# (grid/plate geometry and raw physics values for normalization).
inference:
  representation: eulerian
  grid_n: 128

output:
  log_dir: runs/example
```

Notes:
- `training/trainer.py` resolves top-level string paths ending in `.yaml` and
  inlines them (project root first, then the config file's directory).
- The `model` dict must contain a `type` key matching a registered factory.
- `model_card.yaml` is written as a sidecar to best checkpoints.
- `pretrained_checkpoint` (top-level, optional) seeds training from an
  existing checkpoint (see `configs/training/unetfilm_finetune.yaml`).

## Running

```bash
python -m training.train configs/training/unetfilm_genesis.yaml
python -m training.train configs/training/unetfilm_genesis.yaml --eval-only
python -m training.train configs/training/unetfilm_genesis.yaml --override training.epochs=200
```

Oracle (Genesis-as-model) MPC — see `docs/oracle_mpc_design.md` for the design:

```bash
python -m Genesis.benchmark_n_envs                      # pick mpc.n_envs first
python run_oracle_mpc.py --config simple_mpc/config/config_oracle_test.yaml --save-video
```
