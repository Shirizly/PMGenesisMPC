# Refactor Plan — Modular Training & Inference Framework

**Date:** 2026-06-17  
**Scope:** `train_unet_genesis.py`, `train_unet_cg.py`, `train_GNN_genesis.py`, `run_experiments.py`, `simple_mpc/`, `model/`, config files.  
**Constraint:** `GranularDynamics2/myClasses/`, `Genesis/training/dataset.py`, `RealData/dataset.py`, `model/gnn_dyn.py`, `model/eulerian_wrapper.py` see minimal changes.

---

## 1. Current Pain Points

| Problem | Where it hurts |
|---|---|
| Three training scripts share ~80 % identical train-loop boilerplate | Any bug fix or new metric must be applied in 3 places |
| Physics normalization is defined in `PileSweepData._det_physics` but **silently re-implemented** (wrongly, until recently) in every inference callsite | `run_experiments.py`, `diag_wrapper.py`, `sysid_unet_cg.py` each had their own copy |
| Model loading is a long `if/elif` chain in `run_experiments.py` | Adding a new model type requires editing the runner |
| No standard "model card" — saved checkpoints carry no record of the architecture, physics bounds, or grid config used to train them | Loading a checkpoint requires hunting through run configs manually |
| Config is a single flat `dataset:` section in `config_simple.yaml` | MPC, environment, model, and training settings are all entangled |
| `simple_mpc/adapters.py` has two parallel adapter classes (`EulerianAdapter`, `GNNAdapter`) with no shared base contract | A third model type requires understanding the full adapters file |
| `train_unet_cg.py` is a copy-paste of `train_unet_genesis.py` from a much earlier version — diverged and unmaintained | Real-data fine-tuning broken |

---

## 2. Design Principles

1. **Config is the source of truth.** Every run, checkpoint, and loaded model can be fully reconstructed from a YAML config. Nothing important lives only in code constants.
2. **Single physics-normalization definition.** Bounds live in the dataset config; a shared utility reads them. Every callsite — training, inference, sysid — uses the same function.
3. **Adapters are hierarchical.** Representation-level adapters (Eulerian, Lagrangian) handle the coordinate/state bridge. Model-level sub-adapters handle model-specific forward calls. Adding a new model requires only a new sub-adapter, not touching the runner.
4. **Training is a Trainer class, not a script.** One entry point (`train.py`) instantiates the right dataset, model, optimizer, and loss from a config and calls `Trainer.run()`.
5. **Models, datasets, loss functions are registered.** A registry maps string names to constructors so configs can drive composition without `if/elif`.
6. **Saved checkpoints embed their model card.** Every `*.pth` or sibling `model_card.yaml` records everything needed to reload and use the model.

---

## 3. Target File Layout

```
pile_manipulation/
│
├── configs/                          ← NEW: canonical configs (no more scattered YAMLs)
│   ├── model/
│   │   ├── unetfilm.yaml             full NFDUNetFiLM
│   │   ├── unetfilm_shallow.yaml     NFDUNetFiLMShallow
│   │   └── gnn_propnet.yaml          PropNetDiffDenModel
│   ├── dataset/
│   │   ├── genesis_cube.yaml         PileSweepData + physics bounds
│   │   └── real_chickpeas.yaml       RealPileSweepData + physics bounds
│   └── training/
│       ├── unetfilm_genesis.yaml     full training run config (composes model + dataset + training)
│       ├── unetfilm_finetune.yaml    fine-tune from pretrained
│       └── gnn_genesis.yaml
│
├── training/                         ← NEW: unified training framework
│   ├── __init__.py
│   ├── train.py                      entry point: python -m training.train configs/training/unetfilm_genesis.yaml
│   ├── trainer.py                    Trainer class (model-agnostic train/val/test loop)
│   ├── losses.py                     loss registry + combined_loss helpers
│   └── metrics.py                    metric registry + EulerianMetrics, GNNMetrics
│
├── registry/                         ← NEW: factories
│   ├── __init__.py
│   ├── model_registry.py             build_model(cfg) → nn.Module + ModelCard
│   └── dataset_registry.py           build_dataset(cfg, split) → Dataset
│
├── physics/                          ← NEW: single source of truth for normalization
│   ├── __init__.py
│   └── normalization.py              normalize_physics(), denormalize_physics(), PhysicsBounds
│
├── model/
│   ├── eulerian_wrapper.py           UNCHANGED
│   ├── gnn_dyn.py                    UNCHANGED
│   └── model_card.py                 NEW: ModelCard dataclass + save/load
│
├── simple_mpc/
│   ├── adapters.py                   MINOR: adapters read physics bounds from ModelCard
│   └── ...                           rest unchanged
│
├── Genesis/training/dataset.py       MINOR: _det_physics calls physics.normalization
├── RealData/dataset.py               MINOR: same
│
├── train_unet_genesis.py             DEPRECATED (kept for reference, delegates to training/)
├── train_unet_cg.py                  DEPRECATED
├── train_GNN_genesis.py              DEPRECATED
└── run_experiments.py                MINOR: load_model reads ModelCard
```

---

## 4. Config Schema

### 4.1 Training config (top-level, composes the others)

```yaml
# configs/training/unetfilm_genesis.yaml
_description: "NFDUNetFiLM on Genesis cube data, MSE + mass loss"

model:
  type: unetfilm                 # → registry.model_registry
  in_channels: 2
  cond_dim: 3
  base_features: 8
  input_mode: standard           # standard | sweep-removed-input | sweep-removed-residual

dataset:
  type: genesis                  # → registry.dataset_registry  (= PileSweepData)
  paths:
    - corl/cube/n10
    - corl/cube/n20
  val_pct: 5
  test_pct: 5
  resolution_scale: 1.0
  physics:                       # SINGLE definition; used by training and inference
    normalization:
      friction:     {min: 0.05, max: 0.50}
      density:      {min: 750.0, max: 5000.0}
      box_friction: {min: 0.05, max: 0.50}

training:
  epochs: 100
  batch_size: 64
  lr: 1e-4
  lr_scheduler:
    type: StepLR
    step_size: 100
    gamma: 0.75
  augmentation: true             # ×8 in-loop rotation + flip
  patience: 100
  mixed_precision: true          # bfloat16 AMP
  grad_clip_norm: 1.0
  save_every_n_epochs: 10
  loss:
    mse: 1.0
    mass: 0.2
    dice: 0.0
    bce: 0.0
    sharpness: 0.0
    tv: 0.0
    add: 0.0
    remove: 0.0

inference:                       # used when building EulerianModelWrapper at MPC time
  representation: eulerian
  grid_n: 128
  plate_length_m: 0.04
  plate_width_m: 0.002
  plate_sigma_px: 1.5

output:
  log_dir: runs_cubes/unetfilm_genesis
```

### 4.2 Model card (written alongside every checkpoint)

```yaml
# runs_cubes/unetfilm_genesis/model_card.yaml
# Auto-generated by Trainer.  Do not edit manually.
model:
  type: unetfilm
  in_channels: 2
  cond_dim: 3
  base_features: 8
  checkpoint: unet_best.pth       # relative to this file

dataset:
  physics:
    normalization:
      friction:     {min: 0.05, max: 0.50}
      density:      {min: 750.0, max: 5000.0}
      box_friction: {min: 0.05, max: 0.50}

inference:
  representation: eulerian
  grid_n: 128
  plate_length_m: 0.04
  plate_width_m: 0.002
  plate_sigma_px: 1.5
  wkspc_w: 0.064
  global_scale: 0.6
```

Loading a model becomes:
```python
from model.model_card import load_model_from_card
model_wrapper = load_model_from_card("runs_cubes/unetfilm_genesis/model_card.yaml", env)
```
No more `if mtype == 'unetfilm': ... elif mtype == 'unetfilm-shallow': ...` in `run_experiments.py`.

### 4.3 Dataset config (standalone, can be shared across training runs)

```yaml
# configs/dataset/genesis_cube.yaml
type: genesis
paths:
  - corl/cube/n10
  - corl/cube/n20
  - corl/cube/n30
val_pct: 5
test_pct: 5
resolution_scale: 1.0
physics:
  normalization:
    friction:     {min: 0.05, max: 0.50}
    density:      {min: 750.0, max: 5000.0}
    box_friction: {min: 0.05, max: 0.50}
```

The training config can include this inline or reference it as `dataset: !include configs/dataset/genesis_cube.yaml`.

---

## 5. Physics Normalization — Single Definition

**New file:** `physics/normalization.py`

```python
from dataclasses import dataclass
import torch

@dataclass
class PhysicsBounds:
    friction_min: float = 0.05;  friction_max: float = 0.50
    density_min:  float = 750.0; density_max: float = 5000.0
    box_friction_min: float = 0.05; box_friction_max: float = 0.50

    @classmethod
    def from_config(cls, cfg: dict) -> "PhysicsBounds":
        n = cfg["physics"]["normalization"]
        return cls(
            friction_min=n["friction"]["min"], friction_max=n["friction"]["max"],
            density_min=n["density"]["min"],   density_max=n["density"]["max"],
            box_friction_min=n["box_friction"]["min"],
            box_friction_max=n["box_friction"]["max"],
        )

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: (3,) or (B,3) [friction, density, box_friction] → [0,1]"""
        lo = torch.tensor([self.friction_min, self.density_min, self.box_friction_min])
        hi = torch.tensor([self.friction_max, self.density_max, self.box_friction_max])
        return (raw - lo) / (hi - lo)

    def denormalize(self, norm: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor([self.friction_min, self.density_min, self.box_friction_min])
        hi = torch.tensor([self.friction_max, self.density_max, self.box_friction_max])
        return norm * (hi - lo) + lo
```

**`PileSweepData._det_physics`** becomes a one-liner call to `PhysicsBounds.from_config(dataset_cfg).normalize(raw)`.  
**`run_experiments.py` `load_model`** replaces its inline formula with the same call.  
**`sysid_unet_cg.py` `PHYSICS_BOUNDS`** is replaced by `PhysicsBounds.from_config(model_card["dataset"])`.

---

## 6. Model Registry

**New file:** `registry/model_registry.py`

```python
_MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def build_model(cfg: dict) -> nn.Module:
    mtype = cfg["type"]
    cls = _MODEL_REGISTRY[mtype]
    return cls.from_config(cfg)
```

Each model class gets a `from_config(cfg)` classmethod and a `@register_model("unetfilm")` decorator (added in `registry/`, not inside the model class itself, to keep model files unchanged).

Initial registry entries:
- `"unetfilm"` → `NFDUNetFiLM`
- `"unetfilm-shallow"` → `NFDUNetFiLMShallow`
- `"gnn-propnet"` → `PropNetDiffDenModel`

---

## 7. Dataset Registry

**New file:** `registry/dataset_registry.py`

```python
def build_dataset(cfg: dict, split: str) -> Dataset:
    dtype = cfg["type"]
    if dtype == "genesis":
        return PileSweepData(cfg["paths"], split=split, ...)
    elif dtype == "real":
        return RealPileSweepData(...)
    ...
```

Because the dataset classes are not changing, the registry is a simple factory function (no decorator needed).

---

## 8. Adapter Hierarchy

The existing `simple_mpc/adapters.py` has `EulerianAdapter` and `GNNAdapter` but no shared ABC. The refactor formalises:

```
ModelAdapter (ABC — simple_mpc/adapters.py, already implicit)
│   obs_to_state(obs_np) → state
│   expand_state(state, n) → batch
│   predict_step(state_batch, action_batch) → state_batch
│   compute_reward(state_batch) → reward_batch
│
├── EulerianAdapter                          (unchanged logic)
│   Handles: occ grid ↔ particles, grid bounds, OccupancyReward
│   └── (sub-adapter logic currently in UNetFiLMPushModel.forward — stays there)
│
└── LagrangianAdapter                        (rename of GNNAdapter)
    Handles: particle cloud, s_delta computation, cam_extrinsic
    └── (sub-adapter logic in PropNetDiffDenModel.predict_one_step — stays there)
```

**No new sub-adapter classes are needed** unless a new model requires representation-level changes. The `UNetFiLMPushModel` is already the Eulerian sub-adapter for learned models. What changes is that `make_adapter` reads from the **ModelCard** rather than `isinstance` checks:

```python
def make_adapter(model_dy, cfg, env, subgoal, model_card=None):
    rep = (model_card or {}).get("inference", {}).get("representation", None)
    if isinstance(model_dy, EulerianModelWrapper) or rep == "eulerian":
        return EulerianAdapter(model_dy, cfg, env, subgoal)
    elif isinstance(model_dy, PropNetDiffDenModel) or rep == "lagrangian":
        return LagrangianAdapter(model_dy, cfg, env, subgoal)
    raise ValueError(f"Cannot infer adapter for {type(model_dy)}")
```

---

## 9. Unified Trainer

**New file:** `training/trainer.py`

```python
class Trainer:
    """Model-agnostic training loop.

    Handles:  dataset loading, model construction, optimizer, scheduler,
              AMP scaler, augmentation, logging (TensorBoard), early stopping,
              checkpoint saving, model card writing.

    Does NOT know about: specific loss terms, specific metrics — these come
    from the loss and metric registries keyed by cfg["training"]["loss"].
    """

    @classmethod
    def from_config(cls, config_path: str) -> "Trainer":
        cfg = yaml.safe_load(open(config_path))
        model = build_model(cfg["model"])
        train_ds = build_dataset(cfg["dataset"], "train")
        val_ds   = build_dataset(cfg["dataset"], "val")
        test_ds  = build_dataset(cfg["dataset"], "test")
        loss_fn  = build_loss(cfg["training"]["loss"])     # from losses registry
        metrics  = build_metrics(cfg["model"]["type"])     # per representation
        return cls(model, train_ds, val_ds, test_ds, loss_fn, metrics, cfg)

    def run(self) -> None:
        # standard train / val loop with early stopping
        # writes model_card.yaml next to each checkpoint
        ...
```

**`training/losses.py`** — `build_loss(loss_cfg)` returns a callable `loss_fn(logits, targets, inputs) → (total, components_dict)`. The existing `combined_loss` function from `train_unet_genesis.py` becomes a registered entry:

```python
register_loss("eulerian_mse_mass")(EulerianMseMassLoss)
register_loss("gnn_position_mse")(GNNPositionMSELoss)
```

**`training/metrics.py`** — `build_metrics("unetfilm")` returns `EulerianMetrics()` which wraps the current `update_metric_totals` / `average_metrics` logic.

---

## 10. Entry Point

```
python -m training.train configs/training/unetfilm_genesis.yaml
python -m training.train configs/training/unetfilm_genesis.yaml --eval-only --checkpoint runs_cubes/unetfilm_genesis/unet_best.pth
python -m training.train configs/training/unetfilm_genesis.yaml --override training.epochs=200
```

The three existing scripts (`train_unet_genesis.py`, `train_unet_cg.py`, `train_GNN_genesis.py`) become thin wrappers or are deprecated:

```python
# train_unet_genesis.py (after refactor, delegates)
from training.train import main
main(default_config="configs/training/unetfilm_genesis.yaml")
```

---

## 11. `run_experiments.py` — `load_model` simplification

Current state: ~120-line `if/elif` chain per model type.

After refactor:
```python
from model.model_card import load_model_from_card

def load_model(model_spec: dict, cfg: dict, env) -> EulerianModelWrapper | PropNetDiffDenModel:
    card_path = model_spec.get("model_card")
    if card_path:
        return load_model_from_card(card_path, env, cfg)
    # fallback: old-style mtype dispatch kept for backwards compat
    ...
```

`load_model_from_card` reads the model card, constructs the model, loads weights, applies physics normalization from the card's `dataset.physics.normalization`, and returns the correct wrapper.

---

## 12. Migration Path (what changes vs. what stays)

### Unchanged
- `GranularDynamics2/myClasses/NFDUNetFilm.py` and siblings
- `Genesis/training/dataset.py` — only `_det_physics` delegates to `PhysicsBounds`
- `RealData/dataset.py` — same single-line change
- `model/eulerian_wrapper.py`
- `model/gnn_dyn.py`
- `simple_mpc/mpc.py`, `adapters.py` (logic unchanged; `make_adapter` gains model card awareness)

### New files
- `physics/normalization.py`
- `registry/model_registry.py`
- `registry/dataset_registry.py`
- `model/model_card.py`
- `training/trainer.py`
- `training/losses.py`
- `training/metrics.py`
- `training/train.py`
- `configs/training/unetfilm_genesis.yaml` (and companions)
- `configs/dataset/genesis_cube.yaml`

### Modified files
- `Genesis/training/dataset.py` — `_det_physics` → `PhysicsBounds.normalize()`
- `run_experiments.py` — `load_model` reads `ModelCard`; existing dispatch kept as fallback
- `sysid_unet_cg.py` — `PHYSICS_BOUNDS` replaced by `PhysicsBounds.from_config(model_card)`
- `simple_mpc/adapters.py` — `make_adapter` gains `model_card` argument

### Deprecated (not deleted yet)
- `train_unet_genesis.py`
- `train_unet_cg.py`
- `train_GNN_genesis.py`

---

## 13. Open Questions / Decisions Needed

1. **Config composition strategy** — should dataset configs be inline in training configs, or referenced by path (`dataset: !include configs/dataset/genesis_cube.yaml`)? YAML anchors/includes require PyYAML extensions; inline is simpler to start.
Referenced by path.

2. **`train_unet_cg.py` scope** — this script uses `UNetConditioned` (the old concatenation-based model), not `NFDUNetFiLM`. Should it be retired in favour of real-data fine-tuning via the unified trainer (same model, different `dataset.type: real`)?
yes.

3. **GNN adapter physics normalisation** — `train_GNN_genesis.py` passes raw `density` (not normalised) to `PropNetDiffDenModel`. The GNN uses it differently (as a per-particle scalar, not a FiLM condition). Should GNN physics go through the same normalization pipeline, or stay raw?
Don't touch it yet, I'll figure out this issue later.

4. **Checkpoint format** — currently `unet_best.pth` is a raw `state_dict`. Should the refactor switch to saving `{"model_state_dict": ..., "model_card": {...}}` in one file, or keep the model card as a sidecar YAML? Sidecar is more human-readable and easier to inspect.
Sidecar.

5. **Backwards compatibility** — `run_experiments.py` experiment YAMLs currently reference `model.type: unetfilm` and `model.weights_path`. Should the old format remain supported indefinitely via a compatibility shim, or is a one-time migration of all experiment YAMLs acceptable?
One time migration is fine, but if heavy, it's also fine to leave them in the past, none work very well.


