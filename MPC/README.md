# MPC

The MPC / world-model research stack, ported from the `GenesisWorld` branch to
run against the current simulator in `Genesis/`.

Kept in one directory deliberately: it is a separate deliverable from the
simulator-fidelity work, and the `Genesis/` PR should be reviewable without it.
Eventually this and `GranularDynamics2/` will likely merge; not yet.

## Documentation

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | module map, data flow, extension points, config structure — start here |
| [`docs/INTERFACES.md`](docs/INTERFACES.md) | the contracts between model, dataset and trainer |
| [`docs/UTILITIES.md`](docs/UTILITIES.md) | shared helpers and the recording hooks |
| [`docs/oracle_mpc_design.md`](docs/oracle_mpc_design.md) | the oracle planner: why it exists and what it records |
| [`docs/human_demo_design.md`](docs/human_demo_design.md) | human-in-the-loop demonstrations |

These were written against the previous layout and have been updated for this
one — `model_training/` rather than `training/`, the simulator's current file
names, and the recording API's move into `env/recording_sandbox.py`. Anything
they say about `Genesis/` internals is better checked against
[`../PORT_NOTES.md`](../docs/PORT_NOTES.md), which is maintained alongside the code.

## Running

Everything runs **from inside this directory**:

```bash
cd MPC
python run_oracle_mpc.py --config simple_mpc/config/config_oracle.yaml
python run_experiments.py --config simple_mpc/config/config_simple.yaml
python model_training/train.py --config configs/training/<name>.yaml
```

`genesis_path.py` puts both `Genesis/` and this directory on `sys.path`, and is
imported by the handful of modules that reach the simulator. Nothing here is
importable from the repo root - the packages are addressed as `simple_mpc.*`,
`model.*`, `env.*`, and that only resolves with `MPC/` on the path.

## What the port changed

Structure is otherwise unchanged from `GenesisWorld`. Five things differ:

**1. `training/` became `model_training/`.**
`Genesis/training/` also exists - it holds the dataset and the DINO/LeWM
exporters - and with both directories on `sys.path` a plain `import training`
resolves to whichever came first. This package's had an `__init__.py`, so it
won every time and `training.dataset` became unreachable. The rename removes an
ambiguity that would otherwise have been decided by import order. ~20 call
sites, mechanical.

**2. `from Genesis.x import y` became flat imports.**
`Genesis/` is not a package - no `__init__.py`, and its modules import each
other flatly. That is upstream's convention and the port keeps it, so reaching
the simulator means putting the directory on `sys.path`. All of that is in
`genesis_path.py`; four stale hacks that computed the path relative to the old
file locations (and pointed at a non-existent `MPC/Genesis`) were removed.

The path has to *stay* on `sys.path` rather than be added around one import:
`sandbox_manipulation` imports `placement_sampling`, `action_sampling` and
`state_library` lazily, inside the methods that use them.

**3. Transition recording moved into a subclass.**
`push_and_record`, `flush_transitions`, `set_transition_context` and
`broadcast_state_from_env` were methods on the historical simulator and are
**not** in the current one. They are now
[`env/recording_sandbox.py`](env/recording_sandbox.py)'s `RecordingSandbox`,
which subclasses `SandboxManipulation`, plus
[`env/transition_buffer.py`](env/transition_buffer.py).

They live here rather than in the simulator because the simulator already has a
recording path - `collect_data_samples` + `_save_rollout` + the exporters - and
a second, overlapping one in that file would be exactly what the simulator PR
should not carry. Subclassing works cleanly because none of it runs before
`scene.build()`.

**4. `reset_warmup_steps` raised from 10 to 500** in all four
`simple_mpc/config/config*.yaml`.

This one is a semantic change, not a tuning preference. `settle_steps` used to
be a fixed loop count; it is now a **cap with a convergence exit**. Under the
old semantics a bigger number was pure cost, which is why the config said
`reset_warmup_steps: 10  # was 500 but was a no-op`. Under cap semantics 10 is
far too small - the pile is recorded mid-motion, and the simulator says so:

```
WARNING: pile still moving after the full 10-step settle. At the q=0.995 rest
quantile: 9.95 mm/s linear (threshold 1.0) ... The recorded state is
mid-motion, and because each transition's s comes from the previous s', that
error propagates.
```

Measured steps-to-rest for a fresh spawn is ~34, so 500 is the same safety
margin the collection path uses and costs nothing when the pile settles early.

**5. `model/futureintegration/` lost its four `.py` files.**
All were byte-identical to `GranularDynamics2/myClasses/{Diff_Renderer,
MultiExitUnet,UNetModels,UNetModels_conditioned}.py`, which upstream maintains.
The consolidation notes are still there; the code is not, so it cannot drift.
`Diff_Renderer.py` also does not parse - an unfinished `tool_mask = ` at line
33 - and a copy of it here broke this package's import check for a defect that
is not this package's to fix. Import them as
`GranularDynamics2.myClasses.<module>` from the repo root if needed.

## Known gaps

**`koopman_skeleton.py` does not import.** Its module-level training loop
references `num_phase1_epochs` and `pretrain_loader`, neither of which is
defined. It is a design sketch and was never importable, on this branch or the
old one. Left exactly as it was rather than altered.

**Physics normalisation is fixed, not configurable** (for simulated data).
`registry/dataset_registry.py` no longer passes `physics_bounds=` to
`PileSweepData`; the dataset normalises over friction 0.05-0.50, density
750-5000, box friction 0.05-0.50, which are the endpoints of the original
collection sweep. A `physics.normalization` block in a dataset config is read
for the model card only.

Physics is held fixed per collection run by default, so the conditioning
vector is a constant. **That makes the FiLM models pointless** - there is
nothing for them to modulate on - so use the plain U-Nets unless you are
actually sweeping physics. The FiLM variants are kept for that case.

The real-data path is different and does normalise configurably:
`RealPileSweepData` returns raw physical units and
`_RealEulerianDatasetWrapper` applies `bounds.normalize()`. If simulated data
ever needs the same, that wrapper is the pattern to copy - it keeps one source
of truth for the bounds and leaves upstream's dataset alone.

## Verifying

```bash
cd MPC
python - <<'EOF'
import importlib, sys
from pathlib import Path
sys.path.insert(0, '.')
import genesis_path
bad = []
for f in sorted(Path('.').rglob('*.py')):
    if '__pycache__' in f.parts or f.name == 'genesis_path.py':
        continue
    m = str(f.with_suffix('')).replace('/', '.')
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append((m, e))
print(f"{len(bad)} failed")
for m, e in bad:
    print(' ', m, type(e).__name__, e)
EOF
```

Expect exactly one failure, `koopman_skeleton`, for the reason above.

Recording and the simulator handshake are exercised by constructing a
`GenesisEnv` and calling `push_and_record` - see the smoke test in the port
notes. The simulator's own fixes are gated separately by
`python tests/verify_fixes.py` from the repo root.
