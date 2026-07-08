# GNN Propagation Model Integration Brief

This note summarizes how the particle GNN in `model/gnn_dyn.py` is initialized, what it consumes and produces, how it interacts with data outside the module, and what is needed to train it with the current Genesis data format.

## 1) Model Initialization

The trainable model entry-point is `PropNetDiffDenModel(config, use_gpu=False)`.

It expects these config keys:

```yaml
train:
  particle:
    nf_effect: <int>      # latent width for encoders/propagators
    add_delta: <bool>     # currently stored but not used in forward path
    adj_thresh: <float>   # distance threshold (in state units) for graph edges
```

Internally it creates `PropModuleDiffDen`, then in `predict_one_step(...)`:
- builds dynamic relations (`Rr`, `Rs`) from current particle positions,
- runs message passing,
- predicts next particle positions.

## 2) External Interface (Inputs / Outputs)

`PropNetDiffDenModel.predict_one_step(a_cur, s_cur, s_delta, particle_dens, particle_nums=None)`

Input tensors:
- `a_cur`: `(B, N)` particle attributes/types per node.
- `s_cur`: `(B, N, 3)` current particle positions.
- `s_delta`: `(B, N, 3)` per-particle action displacement features.
- `particle_dens`: `(B,)` scalar density per sample (the model normalizes by `/5000`).
- `particle_nums` (optional): `(B,)` valid particle counts for padded batches.

Output tensor:
- predicted next positions `s_pred`: `(B, N, 3)`.

## 3) Interaction With Current Genesis Data

Current run files already contain what GNN training needs:
- `states`: `(num_samples, N, 7)`
- `states_`: `(num_samples, N, 7)`
- `p_starts`, `p_stops`: tool start/end positions per sample
- physics values in each run config (`material.density`, etc.)

This is sufficient to build supervised one-step transitions without occupancy rendering.

Recommended mapping:
- `s_cur = states[..., :3]`
- `target = states_[..., :3]`
- `particle_dens = config.material.density`
- `a_cur = ones(N)` for material particles
- `s_delta`: derived from `(p_start, p_stop)` and particle locations (for example distance-weighted push displacement along action segment).

## 4) Batch / Padding Requirements

`predict_one_step` supports optional `particle_nums` masking. For mixed runs with different `N`:
- pad `a_cur`, `s_cur`, `s_delta`, `target` to batch `N_max`,
- pass `particle_nums` so relation construction masks padded nodes.

Loss should be computed only on valid particles (`< particle_nums[b]`).

## 5) Changes Needed In Training Script

The previous `train_GNN_genesis.py` had UNet-only assumptions. To make it valid for GNN:

1. Replace occupancy-grid dataset usage with a particle-transition dataset directly reading `*_data.pt` and configs.
2. Build GNN config (`nf_effect`, `adj_thresh`, `add_delta`) and instantiate `PropNetDiffDenModel`.
3. Convert each sample to `(a_cur, s_cur, s_delta, particle_dens, target)`.
4. Add custom `collate_fn` for variable particle counts and `particle_nums` masking.
5. Train with masked MSE on predicted vs target particle positions.
6. Save/load checkpoints using GNN names (`gnn_best.pth`, `gnn_epoch_*.pth`, `gnn.pth`) and include optimizer/scheduler states.
7. Keep deterministic train/val/test split over run files and support `--eval-only`.

## 6) Integration Guidance

To align with current data construction while keeping the model simple:
- continue saving particle states and actions as-is,
- keep one-step supervision (`states -> states_`),
- derive `s_delta` from push geometry in the training script,
- log copy-baseline MSE (`s_cur` vs `states_`) alongside model MSE to verify learned gain over no-motion baseline.

This gives a training path that is consistent with existing Genesis exports and does not depend on UNet occupancy channels.
