# Migrating from Genesis 0.4.5 to Genesis World 1.3.x

Working document for the `GenesisWorld` branch. "Genesis World" is a **rename**
of the same PyPI package `genesis-world`, not a different project — the repo
`Genesis-Embodied-AI/Genesis` became `Genesis-Embodied-AI/genesis-world` and the
import stays `import genesis as gs`. So there is nothing new to install, only a
version bump: **0.4.5 (2026-04-05) → 1.3.3 (2026-08-13)**.

## Rollback

`quadrants` is pinned with a hard `==` in both versions and the two cannot
coexist, so upgrading in place *replaces* it. To return to the pinned baseline:

```bash
conda activate pme
pip install genesis-world==0.4.5 quadrants==0.5.2 gs-madrona==0.0.7.post2
rm -rf ~/.cache/quadrants        # kernel cache was built by the other version
```

Environment as it stood before the upgrade:

```
genesis-world==0.4.5     quadrants==0.5.2      gs-madrona==0.0.7.post2
torch==2.11.0            torchvision==0.26.0   trimesh==4.12.2
numpy==2.2.6             numpy-quaternion==2024.0.13
```

The 0.4.5 baseline measurements are committed under
`tests/scaling_investigation/results/`, so a physics A/B does not require a live
0.4.5 install — compare against those numbers rather than re-measuring both.

## Breaking changes that affect this codebase

Verified against the 1.3.3 source and the upstream PRs, not inferred.

| # | Change | Lands in | What breaks here |
|---|---|---|---|
| 1 | `ConstraintSolverIsland` / `ContactIsland` internals **deleted**, replaced by a new `island.py` | 1.2.0 (PR #2972) | `probe_contact_islands.py` reads `constraint_solver.contact_island.{n_islands, island_entity.n, entity_id}`. Needs a **rewrite**, not a patch. |
| 2 | Rigid DOFs parsed **depth-first** | 1.2.0 (PR #2972) | Positional DOF access maps to different joints. **Silent wrong answers, no crash.** Audit every hardcoded `dofs_idx` / `dofs_idx_local`. |
| 3 | `set_pos`/`get_pos` `relative` default flipped to `True` | 1.1.2 (PR #2934) | Pass `relative=False` explicitly to keep world-frame behaviour. Likely a no-op for plain `Box` entities with no pose offset — verify, do not assume. |
| 4 | Camera recording args moved from stop to start: `start_recording(save_to_filename, fps)` / `stop_recording()` | 1.3.0 | `record_simulation_video.py`. |
| 5 | `use_contact_island` default `False` → **`True`**; `tolerance` default → `None` | 1.2.0 / 0.4.7 | We set both explicitly, so neutral — but this is why `constraint_solver` is now explicit in `basic.yaml` too. |
| 6 | `hibernation_thresh_acc`, `prefer_parallel_linesearch` removed | — | Not used here. |
| 7 | Contact normal forces decoupled from friction coefficient and sliding speed | 1.3.1 | Physics change at the tool interface. Expect the recorded transitions to differ. |

## Why the upgrade is worth it for this workload

PR #2930 (merged 2026-06-12, released 1.1.2 — *after* the 0.4.5 pinned here)
opens: *"The contact island constraint solver was unusable: its main kernel did
not even compile"*. It fixes three bugs, two of which are **hibernation-gated**
(`set_pos`/`set_quat` bypassing wake-up when hibernation is on; stale
hibernated-island chains leaving separated bodies merged).

That scoping matters for reading our own results:

* §8.7's cost law (`push cost ~ island_size^2.64`) was measured with hibernation
  **off**, on a path that compiled and produced coherent island decompositions
  across ten configurations. **It stands.**
* §8.8's hibernation result — a systematic −4 % transport with *exactly zero*
  variance — is what failure-to-wake looks like. **Re-test hibernation here.**

PR #2879 (1.1.0) reports a 100-box tower + duck (606 DOF, CPU) going 69.0 →
15.9 ms/step, 4.3x. Two caveats: the win is **CPU-only** (*"GPU behaviour is
unchanged"*), and it is attributed to a tight skyline band, which a *tower* has
and a **dense pile does not**. There is no published single-env ~200-body pile
benchmark. Treat 4.3x as directional, and measure our own scene.

## Order of work

1. Upgrade, confirm `import genesis as gs` still resolves and the version reports 1.3.x.
2. Mechanical breaks first (camera args; `probe_contact_islands.py`).
3. Silent breaks (DOF ordering audit; `relative=` on `get_pos`/`set_pos`).
4. Re-run the task-3 checks: `pytest`, `verify_fixes`, `verify_new_features`,
   `probe_collection_health`, the dry run.
5. Re-baseline throughput — the island solver is a different implementation, so
   `Genesis/configs/measured/throughput_optimal.yaml` does not transfer.
6. Re-test `use_hibernation` and, separately, whether `constraint_solver=CG`
   works with `use_contact_island=True` (unverified upstream; CG's per-island
   linear solve appears Newton-only, so islands may partition without
   accelerating CG).

## Measured on 1.3.3 (same GPU, same scenes, same seeds)

### Integration changes required

Three, all now in the code and all written to work on **both** versions so the
0.4.5 results in `tests/scaling_investigation/results/` stay reproducible:

1. **`contact_budget_usage()`** — 1.2.x publishes the contact-point cap directly
   as `collider._collider_info.max_contacts`, and it is no longer
   `max_collision_pairs * n_contacts_per_pair`: the buffer is sized per regime
   (convex vs nonconvex pairs have different per-pair caps) and then reduced by
   link-pair contact pruning. Recomputing the old product would *overstate* the
   cap and hide a real overflow — the one thing that check exists to catch.
   Now reads `max_contacts` when present, falls back to the product otherwise.
2. **`record_simulation_video.py`** — camera recording arguments moved from
   `stop_recording` to `start_recording` in 1.3.0. Detected with
   `inspect.signature` rather than a version string.
3. **`probe_contact_islands.py`** — the island structures were replaced.
   `contact_island` (membership per *entity*, sizes from `island_entity.n`)
   became `constraint_state.island` (membership per *link*, via
   `links_island_idx`, sizes counted with `bincount`). For this scene the new
   layout is if anything more direct: every particle is a single-link entity,
   so a link count is a particle count.

The import path did **not** change (`import genesis as gs`), so despite the
rename there were no imports to rewrite.

**DOF depth-first reordering does not affect this codebase.** Every entity here
is a single-link box with a 6-DOF free joint, and depth-first parsing only
reorders multi-link kinematic trees. Confirmed empirically: `verify_fixes`
passes 12/12 including plate cruise speed (124.3 mm/s) and 0.01 mm sweep
tracking, both of which exercise `dofs_idx_local=[0,1,2]` and the vertical /
horizontal DOF fixes.

### Verification

`pytest` 133/133 · `verify_fixes` 12/12 · `verify_new_features` 10/10 ·
`probe_collection_health` clean (no flags) · dry run resolves.

### What actually got faster

| | 0.4.5 | 1.3.3 | |
|---|---|---|---|
| push, 50 objects x 16 envs | 30.29 s | **5.64 s** | 5.4x |
| post-push settle, same | 0.78 s | **0.21 s** | 3.7x |
| shuffle+settle, n=200 | 116,891 ms | **49,824 ms** | 2.3x |
| state-library build | 273.2 s | **104.8 s** | 2.6x |
| settled-state residual (peak) | 49.6 mm/s, 14.4 rad/s | **1.5 mm/s, 0.6 rad/s** | 33x quieter |
| contact-point cap at mcp=150 | 2400 | **6000** | more headroom |

### What did NOT change: the scaling law

Single env, identical broadside push, same library states:

| n | island (0.4.5 -> 1.3.3) | push ms 0.4.5 | push ms 1.3.3 | speedup |
|---|---|---|---|---|
| 50 | 11 -> 11 | 109.3 | 39.9 | 2.7x |
| 100 | 24 -> 23 | 436.8 | 212.0 | 2.1x |
| 150 | 41 -> 36 | 2355.9 | 981.9 | 2.4x |
| 200 | 56 -> 61 | 7835.1 | 3521.8 | 2.2x |

Island sizes are unchanged, as they must be — the contact graph is a property of
the pile, not of the solver. Fitting cost against island size gives an exponent
of **2.62 on 1.3.3 against 2.64 on 0.4.5**: identical within noise.

So the upgrade is a solid **constant-factor** win of roughly 2-3x in the solver
and up to 5x end to end, and it changes nothing structural. §8.7's diagnosis
stands, and **packing fraction remains the only lever on the scaling itself**.
A transition at 200 objects goes from ~896 s to roughly 400 s — better, but
still ~4.4 GPU-days per 1000 transitions, so open decision 5 in
`scaling_to_200_objects.md` is unchanged in kind.

### Still outstanding

* `Genesis/configs/measured/throughput_optimal.yaml` was measured on 0.4.5 and
  does **not** transfer — the optimum env count per object count has to be
  re-benchmarked (`benchmark_throughput.py`).
* Transitions collected on 1.3.3 are **not comparable** to 0.4.5 data. Beyond
  the solver rewrite, 1.3.1 decoupled contact normal forces from the friction
  coefficient and sliding speed, which changes the tool interface directly.

### The upgrade reopens both escapes — and they are large

Both configurations that were unusable on 0.4.5 now run. At n=200, one env,
identical broadside push:

| config | settle ms/step | push ms/step | largest island | contact points |
|---|---|---|---|---|
| Newton + islands (baseline) | 65.6 | 3521.8 | 61 | 1053 |
| **hibernation** | 50.2 | **62.0** (57x) | **7** | 770 |
| **CG + islands** | 69.3 | **116.6** (30x) | 62 | 1049 |

Two different mechanisms, both consistent with §8.7:

* **Hibernation attacks the island size.** It is the physically principled form
  of "split the island when it grows too large": bodies below the velocity /
  acceleration threshold are frozen and dropped from the active island rather
  than carried in its dense block, so during a push the active island shrinks to
  the neighbourhood of the blade — 61 entities down to **7**. Cost follows,
  because cost was never about particle count.
* **CG ignores the island size.** Its island stays at 62 and it is still 30x
  cheaper, because it never forms the dense Hessian whose assembly and
  factorization is what island size actually prices.

On 0.4.5 neither was available: CG+islands did not run at all (two bugs in
`solver_island.py`), and hibernation was affected by the wake-up bugs PR #2930
fixed in 1.1.2 — which is very likely what our -4 %-transport-with-zero-variance
measurement was actually detecting.

**These numbers are speed only and must not be acted on yet.** Hibernation
records 770 contact points against the baseline's 1053, i.e. it is solving a
different contact set by construction, and freezing bodies that should still be
moving is exactly the failure mode measured on 0.4.5. Re-run
`probe_solver_equivalence.py` against these two configurations before adopting
either — the noise-floor and replicate discipline in §8.8 exists precisely
because a 57x speedup is the most tempting possible moment to skip it.
