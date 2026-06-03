# Plan: MPC Transition Collection (`mpc_transitions`)

## Overview

During the MPC loop, every executed push is a real simulation transition: a
known particle state before the action, a known action, and a known particle
state after.  This data is already being produced for free by the Genesis sim
— the goal is to capture it and save it in exactly the same on-disk format
used by `Genesis/data_collection_clean.py` so it can be mixed with training
data without any changes to the dataloaders.

---

## Saved format (unchanged from training data)

```
<exp_dir>/mpc_transitions/
    _{ep_idx}_data.pt      # torch.save'd dict — all transitions for one episode
    _{ep_idx}_config.yaml  # simulation config for that episode
```

`_{ep_idx}_data.pt` keys — identical to what `SandboxManipulation._save_data` writes:

| Key        | Shape                       | Dtype   | Description                              |
|------------|-----------------------------|---------|------------------------------------------|
| `states`   | `(n_mpc, n_particles, 7)`   | float32 | Particle state BEFORE action (xyz + quat)|
| `states_`  | `(n_mpc, n_particles, 7)`   | float32 | Particle state AFTER action (xyz + quat) |
| `p_starts` | `(n_mpc, 3)`                | float32 | 3-D world plate start pos `[sx, sy, z]`  |
| `p_stops`  | `(n_mpc, 3)`                | float32 | 3-D world plate stop pos  `[ex, ey, z]`  |
| `angles`   | `(n_mpc,)`                  | float32 | Plate yaw angle (radians)                |

No `_failed.pt` file is written; every MPC transition is executed and therefore valid.

`_{ep_idx}_config.yaml` is `env._sim._config` dumped as YAML (same content as
`SandboxManipulation._save_config` writes).

---

## Data sources in the MPC loop

All required data is already present at the call sites in `simple_mpc/mpc.py`:

| Datum          | Where to read it                                 |
|----------------|--------------------------------------------------|
| `state_before` | `env._sim._particle_state[0]` before `env.step()`; shape `[n_particles, 7]` — always up-to-date because `SandboxManipulation.update_material_state()` is called at the end of every `env.step()` and after `env.reset()` |
| `state_after`  | `env._sim._particle_state[0]` after `env.step()` |
| `p_start` (3D) | `[sx, sy, env._sim._operation_height]` — `sx, sy` come from `best_action_np` |
| `p_stop`  (3D) | `[ex, ey, env._sim._operation_height]` |
| `angle`        | `_angle` (already computed from `atan2(ey-sy, ex-sx) + π/2`) |

`env._sim._particle_state` has shape `[n_envs, n_particles, 7]`; for MPC
the env always has `n_envs=1`, so we index `[0]`.

The guard `hasattr(env, '_sim') and hasattr(env._sim, '_particle_state')`
keeps this feature inert when the env is not a `GenesisEnv`.

---

## Code changes

### 1. `simple_mpc/mpc.py` — accumulate transitions

Add a `collect_mpc_transitions: bool = False` parameter to `run_simple_mpc`.

Inside the MPC loop, **before** `env.step(best_action_np)`:
```python
if _collect_trans:
    _trans_states.append(
        env._sim._particle_state[0].detach().cpu().clone()
    )
```

**After** `env.step(best_action_np)`:
```python
if _collect_trans:
    _sx, _sy, _ex, _ey = best_action_np
    _z = env._sim._operation_height
    _trans_states_.append(
        env._sim._particle_state[0].detach().cpu().clone()
    )
    _trans_p_starts.append(
        torch.tensor([_sx, _sy, _z], dtype=torch.float32)
    )
    _trans_p_stops.append(
        torch.tensor([_ex, _ey, _z], dtype=torch.float32)
    )
    _trans_angles.append(_angle)   # already computed above env.step()
```

Include in the returned dict:
```python
'mpc_transitions': {
    'states':   torch.stack(_trans_states),    # (n_mpc, n_particles, 7) CPU
    'states_':  torch.stack(_trans_states_),   # (n_mpc, n_particles, 7) CPU
    'p_starts': torch.stack(_trans_p_starts),  # (n_mpc, 3) CPU
    'p_stops':  torch.stack(_trans_p_stops),   # (n_mpc, 3) CPU
    'angles':   torch.tensor(_trans_angles, dtype=torch.float32),  # (n_mpc,) CPU
} if _collect_trans else None,
```

`_collect_trans = collect_mpc_transitions and hasattr(env, '_sim') and hasattr(env._sim, '_particle_state')`

If the loop terminates early (`obs_next is None`) the already-accumulated
transitions (for completed steps) are still saved, so the output tensors may
have fewer than `n_mpc` rows.

### 2. `run_experiments.py` — save transitions

**In `run_episode()`**:

1. Read flag: `_save_trans = bool(output_cfg.get('save_mpc_transitions', False))`
2. Pass `collect_mpc_transitions=_save_trans` to `run_simple_mpc(...)`.
3. After the call, if `result.get('mpc_transitions') is not None`:
   ```python
   trans_dir = os.path.join(exp_dir, 'mpc_transitions')
   os.makedirs(trans_dir, exist_ok=True)
   torch.save(result['mpc_transitions'],
              os.path.join(trans_dir, f'_{ep_idx}_data.pt'))
   # Save sim config alongside (matches training data convention)
   with open(os.path.join(trans_dir, f'_{ep_idx}_config.yaml'), 'w') as fh:
       yaml.dump(env._sim._config, fh, default_flow_style=False)
   ```

`exp_dir` (the per-experiment results folder, e.g.
`outputs/experiments/<suite>/<exp_name>/`) is already available inside
`run_experiment → run_episode`.

### 3. Experiment YAML configs

Add `save_mpc_transitions: true` under the `output` section of any experiment
YAML where data collection is desired, e.g.:

```yaml
output:
  save_mpc_transitions: true
  save_prediction_videos: false
  ...
```

The field defaults to `false`, so existing configs are unaffected.

---

## What is NOT changed

- `SandboxManipulation` / `GenesisEnv` — no modifications needed; the particle
  state is read non-destructively after each step.
- Data loaders (`Genesis/training/`, `RealData/dataset.py`) — the on-disk
  format is unchanged, so `SandboxManipulation.load_data()` and
  `PileSweepData` can load MPC-transition files without modification.
- The MPC optimisation or action selection — collection is purely passive.
- Non-Genesis environments — the feature is silently disabled.

---

## review items

1. **Directory layout**: transitions for all episodes of one experiment go into
   `mpc_transitions/<exp_dir>/` (flat, one `.pt` per episode). 

2. **Config YAML**:  dump `env._sim._config` as-is (includes sampled
   per-particle sizes / friction arrays).  Dumping as-is mirrors the existing
   `_save_config` behaviour and is safest.

3. **Early-termination handling**: if `obs_next is None` part-way through, we
   save only the transitions collected so far (fewer than `n_mpc` rows).
   This is consistent with how `rewards` / `actions` are handled. 


