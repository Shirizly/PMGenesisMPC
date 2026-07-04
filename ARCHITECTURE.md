# Architecture

This is the repository map and extension guide. Data contracts and coordinate conventions are documented in `INTERFACES.md`. Utility ownership and transform guidance are documented in `UTILITIES.md`.

## Module Map

```
physics/
  normalization.py      PhysicsBounds (canonical raw<->normalized mapping)

registry/
  model_registry.py     register_model, build_model, ModelTrainingWrapper
  dataset_registry.py   register_dataset, build_dataset, EulerianDatasetWrapper

model/
  model_card.py         ModelCard save/load and load_model_from_card

training/
  types.py              TrainingBatch and ModelOutput contracts
  losses.py             register_loss, build_loss, EulerianCombinedLoss
  metrics.py            EulerianMetrics
  trainer.py            Trainer loop and config include resolution
  train.py              CLI entry point

configs/
  model/                one YAML per architecture
  dataset/              one YAML per data source
  training/             experiment YAMLs composing model+dataset+training
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
2. Ensure outputs match the standard batch contract in `INTERFACES.md`.
3. Add a dataset config in `configs/dataset/`.

### Add a Loss

1. Subclass `LossFn` in `training/losses.py`.
2. Register it with `@register_loss("name")`.
3. Reference it under `training.loss` in your training config.

## Config Structure

Top-level training config schema:

```yaml
model: configs/model/unetfilm.yaml
dataset: configs/dataset/genesis_cube.yaml

training:
  epochs: 100
  batch_size: 64
  lr: 1e-4
  augmentation: true
  loss:
    type: eulerian_combined
    mse: 1.0
    mass: 0.2

inference:
  representation: eulerian
  grid_n: 128

output:
  log_dir: runs/example
```

Notes:
- `training/trainer.py` resolves top-level string paths ending in `.yaml` and inlines them.
- `model_card.yaml` is written as a sidecar to best checkpoints.

## Running

```bash
python -m training.train configs/training/unetfilm_genesis.yaml
python -m training.train configs/training/unetfilm_genesis.yaml --eval-only
python -m training.train configs/training/unetfilm_genesis.yaml --override training.epochs=200
```
