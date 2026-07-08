# Architecture

This is the repository map and extension guide. Data contracts and coordinate conventions are documented in `INTERFACES.md`. Utility ownership and transform guidance are documented in `UTILITIES.md`.

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
                        heuristic push models (splat/fluid/spread/cumulative),
                        UNetFiLMPushModel, particle<->occupancy helpers
  diff_mass_push.py     differentiable mass-push kernels used by the
                        heuristic push models
  gnn_dyn.py            PropNetDiffDenModel (Lagrangian GNN dynamics)
  futureintegration/    salvaged architectures awaiting registry integration
                        (NCA, STN, MultiExit/plain/conditioned/modular U-Nets,
                        differentiable renderer — see its README)

transforms/
  functional.py         particles_to_occupancy, draw_plate_soft,
                        genesis_action_to_cam3d, build_action_delta
  representation.py     Compose, EnsureRepresentation, occupancy/particle
                        alias transforms, build_transforms

training/
  types.py              TrainingBatch and ModelOutput contracts
  losses.py             register_loss, build_loss, EulerianCombinedLoss,
                        LagrangianMSELoss
  metrics.py            build_metrics, EulerianMetrics, LagrangianMetrics
  trainer.py            Trainer loop and config include resolution
  train.py              CLI entry point

simple_mpc/
  mpc.py                run_simple_mpc gradient-descent MPC loop
  adapters.py           EulerianAdapter, GNNAdapter, make_adapter
                        (the adapter surface defined in INTERFACES.md §3.4)
  action_sampler.py     candidate-action samplers (uniform, physics-aware,
                        collision-aware, OT-guided)
  ot_planner.py         OTPlannerSparse (Sinkhorn OT action initializer)
  occupancy_reward.py   OccupancyReward (goal-mask scoring)
  benchmark.py          adapter/push throughput benchmarks
  debug_vis.py          MPC debug visualization panels/videos
  config/               base MPC config + experiments/ suite YAMLs

RealData/
  dataset.py            RealPileSweepData (real camera data; mirrors the
                        PileSweepData output format)

configs/
  model/                one YAML per architecture
  dataset/              one YAML per data source
  training/             experiment YAMLs composing model+dataset+training

tests/
  test_configurable_unet.py  registry/model smoke tests (pytest)

utils.py                shared root-level helpers (YAML I/O, action geometry,
                        point-cloud ops, goal-shape generation) used by
                        model/eulerian_wrapper.py and simple_mpc/
```

Entry points beyond `training/train.py`:

```
run_experiments.py       MPC experiment-suite runner; loads models via
                         model cards, writes outputs/experiments/ and
                         mpc_transitions/
run_experiment_batch.py  subprocess driver running run_experiments.py over
                         multiple suite YAMLs
visualize.py             dataset / occupancy / prediction visualization
```

External dependencies imported by the pipeline but living outside this map:

```
Genesis/training/dataset.py   PileSweepData — raw Genesis sim dataset,
                              wrapped by the "genesis" entry in
                              dataset_registry.py
```

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
