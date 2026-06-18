# Codebase Architecture

## Module Map

```
physics/
  normalization.py      PhysicsBounds — canonical (raw→[0,1]) normalisation

registry/
  model_registry.py     @register_model, build_model, ModelTrainingWrapper
  dataset_registry.py   @register_dataset, build_dataset, EulerianDatasetWrapper

model/
  model_card.py         ModelCard (save/load YAML sidecar), load_model_from_card

training/
  losses.py             @register_loss, build_loss, EulerianCombinedLoss
  metrics.py            EulerianMetrics
  trainer.py            Trainer (generic loop)
  train.py              CLI entry point

configs/
  model/                one YAML per architecture
  dataset/              one YAML per data source
  training/             one YAML per experiment (references model + dataset YAMLs)
```

## Data flow

```
Training config YAML
  ├─ model:   configs/model/unetfilm.yaml   ──► build_model(cfg)  ──► ModelTrainingWrapper
  └─ dataset: configs/dataset/genesis.yaml  ──► build_dataset(cfg, split) ──► Dataset

Trainer.run()
  for batch in DataLoader:
      prediction = model_wrapper(batch)          # forward(batch) → Tensor[B,1,H,W]
      loss, comps = loss_fn(prediction, batch)   # (total, dict[str,float])
      metrics.update(prediction, batch)
  
  → saves state_dict + model_card.yaml
```

## Standard batch dict

```python
{
    "input":   Tensor[B, C, H, W],   # model input (e.g. occupancy + action)
    "target":  Tensor[B, H, W],      # prediction target
    "physics": Tensor[B, P],         # normalised [0,1]; KEY ABSENT if unused
}
```

`"physics"` is absent (not `None`) when the model doesn't use it — check with `"physics" in batch`.

---

## How to integrate a new model

1. **Write the nn.Module** (anywhere; no base class required).

2. **Register a factory** in `registry/model_registry.py`:

```python
@register_model("mymodel")
def _build_mymodel(cfg: dict) -> EulerianTrainingWrapper:
    from mypackage import MyModel
    model = MyModel(in_channels=cfg.get("in_channels", 2))
    return EulerianTrainingWrapper(model, uses_physics=False)
```

3. **Add a config file** `configs/model/mymodel.yaml`:

```yaml
type: mymodel
in_channels: 2
uses_physics: false
```

4. Reference it from a training config: `model: configs/model/mymodel.yaml`.

If the model doesn't consume physics, set `uses_physics: false` — the `"physics"` key will simply be absent from batches and the wrapper won't pass it to `forward`.

---

## How to integrate a new dataset

1. **Write a builder function** and register it in `registry/dataset_registry.py`:

```python
def _build_mydataset(cfg: dict, split: str) -> EulerianDatasetWrapper:
    from mypackage import MyRawDataset
    raw = MyRawDataset(cfg["data_root"], split=split)
    return EulerianDatasetWrapper(raw, include_physics=cfg.get("include_physics", False))

_DATASET_REGISTRY["mydataset"] = _build_mydataset
```

2. `EulerianDatasetWrapper` expects `__getitem__` to return `((input_grid, physics), target)`.  
   If your dataset returns raw physics, subclass it and call `self.bounds.normalize(raw_physics)` (see `_RealEulerianDatasetWrapper`).

3. **Add a config file** `configs/dataset/mydataset.yaml`:

```yaml
type: mydataset
data_root: path/to/data
include_physics: false
```

---

## How to integrate a new loss function

1. **Subclass `LossFn`** and register it in `training/losses.py`:

```python
@register_loss("my_loss")
class MyLoss(LossFn):
    def __init__(self, cfg: dict):
        self.w = float(cfg.get("weight", 1.0))

    def __call__(self, prediction: Tensor, batch: dict):
        # prediction: Tensor[B,1,H,W] raw logit
        # batch: standard batch dict
        loss = ...
        return loss, {"my_term": loss.item()}
```

2. Reference it in a training config:

```yaml
training:
  loss:
    type: my_loss
    weight: 2.0
```

---

## How to run training

```bash
# Train from scratch
python -m training.train configs/training/unetfilm_genesis.yaml

# Resume an in-progress run (default)
python -m training.train configs/training/unetfilm_genesis.yaml

# Override config keys
python -m training.train configs/training/unetfilm_genesis.yaml \
    --override output.log_dir=runs_cubes/my_run training.epochs=200

# Evaluate only (loads unet_best.pth from output.log_dir)
python -m training.train configs/training/unetfilm_genesis.yaml --eval-only
```

Checkpoints are written to `output.log_dir` as raw `state_dict` `.pth` files.  
A `model_card.yaml` sidecar is written whenever a best checkpoint is saved.

---

## Model card (inference)

After training, load a model for MPC with just the card path:

```python
from model.model_card import load_model_from_card
model = load_model_from_card("runs_cubes/my_run/model_card.yaml", env=env)
```

In experiment configs, add `model_card: path/to/model_card.yaml` to the model spec and `load_model` in `run_experiments.py` will dispatch automatically.

---

## Physics normalisation

Use `PhysicsBounds` everywhere instead of inline arithmetic:

```python
from physics.normalization import PhysicsBounds
bounds = PhysicsBounds.default()          # or PhysicsBounds.from_config(cfg["physics"])
norm = bounds.normalize(raw_tensor)       # Tensor[3] or [B,3] → [0,1]
raw  = bounds.denormalize(norm_tensor)
```

Default bounds: friction [0.05, 0.50], density [750, 5000 kg/m³], box_friction [0.05, 0.50].
