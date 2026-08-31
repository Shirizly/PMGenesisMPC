---
name: project-overview
description: "Project map for pile_manipulation: purpose, major subsystems, and which doc file owns what. Use before working on this repo for the first time in a session, before any change that touches a major information flow, a module's responsibility, a default value, or adds a feature, or whenever it's unclear which module/doc owns something."
argument-hint: "Optionally name the subsystem you're touching, e.g. 'oracle mpc', 'losses', 'Genesis wrapper', 'adapters'"
user-invocable: true
---

# Pile Manipulation — Project Overview

## Purpose

Research codebase for granular pile manipulation: pushing granular material
(cubes/spheres of e.g. chickpeas) toward a target shape/region with a
plate-style pusher, in the Genesis simulator. It supports **learned dynamics
models driven by gradient-descent MPC**, a **Genesis-as-model sampling MPC
(CEM/MPPI) ceiling baseline** that removes dynamics-model error entirely,
and a **human-piloted variant of that same ceiling baseline** (grid-search-
refined manual actions via a GUI) that also removes optimizer/sampling
limitations from the comparison, plus the shared training/loss/transform
infrastructure all three depend on.

## Major parts

| Directory / file | Role |
|---|---|
| `Genesis/` | Low-level simulator wrapper (`SandboxManipulation`) — build/reset/execute a push/read state; batched multi-env data collection; the `n_envs` throughput benchmark |
| `env/genesis_env.py` | `GenesisEnv` — single-env bridge from `SandboxManipulation` to the MPC-facing observation/action interface |
| `model/` | Dynamics models: learned (`NFDUNetFilm.py`, `gnn_dyn.py`, `futureintegration/`) and checkpoint-free heuristics (`eulerian_wrapper.py`'s push-model registry) |
| `training/` | Model training loop (`trainer.py`), the loss registry (`losses.py`), typed batch/output contracts (`types.py`) |
| `transforms/` | Stateless, dependency-light representation conversions (particle↔occupancy, action↔camera coords) shared by datasets, models, and both MPC variants |
| `registry/` | `register_model`/`build_model`, `register_dataset`/`build_dataset` factories |
| `physics/` | `PhysicsBounds` — the one normalization source of truth |
| `simple_mpc/` | Three MPC variants: `mpc.py`/`adapters.py` (gradient-descent, learned/heuristic models via the adapter pattern); `oracle_mpc.py`/`genesis_oracle.py`/`sampling_optimizers.py` (Genesis-as-model CEM/MPPI ceiling baseline); `human_mpc.py`/`human_grid_search.py` (human-piloted variant of the same ceiling baseline, grid-search-refined) |
| `run_experiments.py` / `run_oracle_mpc.py` / `human_mpc_gui.py` | Batch/entry-point drivers for the three MPC variants |
| `tests/` | pytest suite (Genesis-free tests run without a GPU; a few require Genesis) |

## Doc map — read (and update) the one that owns what you're touching

| Doc | Owns |
|---|---|
| `docs/ARCHITECTURE.md` | Repository module map, entry points, the **Design Philosophy** (modularity / division-of-responsibility patterns to preserve), extension recipes (add a model/dataset/loss), training config schema |
| `docs/INTERFACES.md` | Data contracts: batch dict keys per representation, `ModelOutput`, the MPC adapter surface (§3.4) and its `per_sample` loss-cost variant (§3.5), coordinate conventions |
| `docs/UTILITIES.md` | Utility ownership boundaries: what belongs in `transforms/functional.py` vs `utils.py` vs a scoped module, and the on_phase-hook / write_video_frame pattern as the reference example |
| `docs/oracle_mpc_design.md` | Full design reference for the oracle MPC subsystem: snapshot/restore state management, sampling optimizers, occupancy-representation caveats, config schema, known limitations |
| `docs/linear_visual_foresight_baseline.md` | Suh & Tedrake 2020 switched-linear visual foresight as a comparison baseline: paper summary, what the repo already supports, the integration plan, and the perpendicular-push / fixed-length action restriction (§7, implemented) |
| `docs/piled_collection.md` | Piled (multi-layer, centred) particle spawns and pile-aware action sampling: why they exist, what they guarantee, and every flag/config that activates them |
| `docs/human_demo_design.md` | Full design reference for the human-demonstration subsystem: the 5D action convention, local grid-search refinement, GUI interaction model, output-schema/recording parity with `run_oracle_mpc.py` |
| `.github/skills/mpc-experiments/SKILL.md` | Deep-dive operational guide for the learned-model MPC framework specifically (adapters, reward types, running/debugging experiments) |

If you're not sure where something belongs, it's almost certainly one of
these five `docs/` files, not a new one — check the Design Philosophy in
`ARCHITECTURE.md` first; it explains *why* the boundaries are drawn where
they are, which usually settles where a change's documentation belongs too.

## Documentation policy (do this as part of the change)

**Any change to a major information flow, a module's responsibility, a
default value, or an added feature updates the relevant doc(s) above in the
same change — not as a follow-up, and not only when explicitly asked.**

Concretely, before considering such a change done:
- New module, class, or function that another part of the codebase is meant
  to reuse → add it to `ARCHITECTURE.md`'s module map (and `UTILITIES.md` if
  it's a reusable utility rather than a subsystem-specific piece).
- New or changed batch dict key, model output shape, or adapter method →
  update `INTERFACES.md`.
- New config default, changed hyperparameter meaning, or a new knob →
  update the relevant config's doc comments *and* `ARCHITECTURE.md` /
  `oracle_mpc_design.md` if it's structural rather than purely tunable.
- Anything touching the oracle MPC subsystem specifically → update
  `docs/oracle_mpc_design.md`'s relevant section (it's organized by design
  decision, so most changes map to one existing section or a clearly-scoped
  new one).
- A genuinely new subsystem on the scale of oracle MPC → give it its own
  `docs/<name>_design.md` following that file's shape (purpose, architecture
  with *why* alongside *what*, file map, config reference, usage, known
  limitations), and link it from `ARCHITECTURE.md` and this skill's doc map.

Small, targeted diffs are the goal — each doc's scope is narrow by design
(see `ARCHITECTURE.md`'s Design Philosophy), so a real change should only
ever touch one or two of them.

## Environment

Genesis-dependent code (`Genesis/*`, `env/genesis_env.py`,
`simple_mpc/genesis_oracle.py`, and anything importing them) requires the
`genesis` package and a GPU; everything else in the module map is
Genesis-free and covered by the fast pytest suite. Activate the project's
conda environment before running Python: `conda activate pme`.
