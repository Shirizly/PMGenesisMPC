# Linear Visual Foresight on Granular Piles — Findings and Handoff

**Branch:** `VisualForesight` · **Last updated:** 2026-08-31
**Paper under test:** Suh & Tedrake 2020, *The Surprising Effectiveness of Linear
Models for Visual Foresight in Object Pile Manipulation*, arXiv:2002.09093
([`docs/2002.09093v3.pdf`](2002.09093v3.pdf))

**Companion documents**
- [`docs/linear_visual_foresight_baseline.md`](linear_visual_foresight_baseline.md) — the paper summary and the original integration plan
- [`docs/piled_collection.md`](piled_collection.md) — the piled-spawn and pile-aware-sampling subsystem, and every flag
- [`reports/linear_foresight_report.md`](../reports/linear_foresight_report.md) — the chronological experiment log, including superseded results
- [`reports/linear_foresight_numbers.json`](../reports/linear_foresight_numbers.json) — machine-readable results

**This document is the one to read first.** It supersedes the running commentary
in `reports/linear_foresight_report.md` wherever they disagree.

---

## 1. Executive summary

We set out to reproduce the paper's method as a comparison baseline: predict how a
top-down occupancy image changes under a push, using a **linear** operator, and
use it for greedy control.

**The headline: the method's central claim is regime-dependent, and the regime is
set by how you sample actions — not by the model, the physics, or the
representation.**

Seven findings, in order of confidence:

1. **Blind action sampling was the problem.** Drawing push start points uniformly
   over the tray produces pushes spanning a huge range of contact with the pile,
   and *that variation is the nonlinearity*. Conditioned on contact amount, the
   dynamics are essentially linear. The paper collected 1000 examples per
   discretised action on purpose; our sampler did not.

2. **With contact-aware sampling, linear nearly closes the gap on displacement.**
   The share of predictable per-push displacement that a linear model captures
   rises from **69% → 92–94%**. On the control-relevant quantity (`dV`) the
   picture is **mixed and weaker** — see finding 8 and §3.4.

3. **This holds for *summary quantities*, not for per-pixel images.** A
   well-determined pixel operator (parameters cut to 256/row, 8× more examples
   than unknowns) still does not beat "predict nothing moved." The insight does
   not transfer to the paper's actual model target.

4. **Per-pixel image error was the wrong yardstick anyway.** The paper's claim is
   comparative (its own numbers span ~10–35%) and control-based, and it reports
   **no persistence baseline**. Persistence — which wins every per-pixel
   comparison — predicts `dV = 0` for every action and therefore *cannot rank
   actions at all*, making it useless as a control baseline. On ranking, the
   linear operator captures 43–79% of an oracle's advantage.

5. **Linearity costs real control performance.** A linear model turns a harmful
   random push into a helpful one; a nonlinear model is 3–6× better in realised
   `dV`. Linear is *sufficient*, not optimal.

6. **Three plausible failure explanations were tested and refuted:** the plate is
   not skimming (it engages 100% of cubes and moves them ~19 mm per 40 mm push);
   binary occupancy is not destroying height information (the pile is one layer
   deep, so there is none); displacement is not stochastic (it is ~84%
   predictable).

7. **The pile itself was not the lever.** Denser in-plane packing does not improve
   the linear share (76 / 84 / 75% across density strata), and 30 cubes cannot
   form a deep heap in a 128 mm tray regardless — they settle into a dense
   monolayer.

8. **The remaining `dV` gap was the push DIRECTION, and expressing it closes the
   gap.** Contact-aware sampling linearises *how much* material moves (92%) but
   not immediately *where it goes relative to a goal* (68% for a centred target).
   Adding push-direction features — `cos/sin` of the heading, the push's
   projection onto the pile's outward radial, and their interaction — raises the
   linear share to **82% (centre), 90% (corner), 91% (point)**. Corner nearly
   doubles, from 44%.

   **This is the same mechanism as the paper's warp.** Their SE(2) canonical push
   frame factors direction out of the representation by construction, which is
   precisely what these features do by hand. So the paper's two ingredients
   (canonical frame + switch on push length) plus one our data needs
   (**condition on contact amount**) give a model class that is ~90% linear.
   That is the cleanest statement of the whole investigation.

**Status of the baseline:** implemented, characterised, and honestly negative *at
the pixel level*; positive *at the summary level* and *as a controller*. That is a
publishable comparison result rather than a failed reproduction.

---

## 2. What exists in the code

All new collection modes are **off by default**; existing configs, datasets and
scripts are unaffected. 169 tests pass, none require a GPU.

### 2.1 Infrastructure

| File | What it provides |
|---|---|
| `transforms/functional.py` | The SE(2) push-frame warp the method requires: `push_frame_transform`, `to_push_frame`, `from_push_frame`, `push_frame_validity_mask`, `blend_push_prediction`, `invert_affine`. Supports a `scale` argument that **crops** the canonical window instead of warping the whole image. Verified mass-preserving. |
| `Genesis/action_sampling.py` | Action-space geometry, pure torch, GPU-free tests. `blade_normal`, `sampling_box`, `constrain_push` (perpendicular / fixed-length), `relative_blade_angle`, `pile_contact_starts` (aim the blade at the pile), `ray_box_max_travel`. |
| `Genesis/sandbox_manipulation_clean.py` | `shuffle_particles(pile_extent=, pile_layers=)` for compact multi-layer spawns; `generate_action_samples(perpendicular_pushes=, push_length=, pile_aware=, ...)`; per-step action redraw in pile-aware mode. |
| `Genesis/data_collection_clean.py` | Flags: `--perpendicular-pushes`, `--push-length`, `--pile-extent`, `--pile-layers`, `--pile-aware-actions`, `--pile-clearance`, `--min-swath-particles`. All recorded in each batch's saved config. |

### 2.2 Analysis tools

| Script | Question it answers |
|---|---|
| `fit_linear_foresight.py` | Fits the pixel operator; compares against persistence, an **identity-operator** control (warp with `A = I`, isolating resampling cost from model error), and the geometric heuristics. Self-checks the world→pixel mapping and aborts on mismatch. |
| `loro_foresight.py` | Leave-one-run-out CV of the above. **Use this, not single splits** — see §6. |
| `variance_decomposition.py` | How much of per-push displacement is predictable, and how much of that *linearly*. The core scientific instrument. |
| `deltav_predictability.py` | Same, for the control-relevant `dV`, computed on particles (their eq. 3) so it is free of grid resolution and the warp. Also converts R² into realised control performance. |
| `density_stratified.py` | Stratifies by local packing (`--by packing`) or contact amount (`--by contact`) — the two hypothesis tests. |
| `control_utility_test.py` | Can the operator **rank** actions? Correlations, sign agreement, best-of-K slate utility, and a partial correlation controlling for state difficulty. |
| `interpret_foresight.py` | Restates errors in physical units (cube-equivalents, mm) and as a share of the change that actually occurred. |
| `run_pile_analysis.py` | Runs the three main analyses on a dataset and prints the scattered reference alongside. |

### 2.3 Datasets

| Config | Contents |
|---|---|
| `configs/dataset/genesis_foresight_L040.yaml` | 2560 scattered 50-cube transitions, blind perpendicular 40 mm pushes |
| `configs/dataset/genesis_foresight_L040_full.yaml` | The above pooled with a second run — 7680 transitions |
| `configs/dataset/genesis_foresight_pile30.yaml` | ~1870 piled 30-cube transitions (29 files), pile-aware 40 mm pushes. **Collected before the start-clamp fix** — ~36% of starts were outside the workspace box and ~3% of pushes travelled ~0 mm. Still usable; the contamination biases *against* the linear result, so findings from it are conservative. |
| `Genesis/configs/collection_pile30.yaml` | The corrected collection plan: 20 mm pushes, clamped starts |

---

## 3. The scientific narrative

### 3.1 The method, in one paragraph

Represent the scene as a top-down occupancy image. Warp it into a canonical frame
where every push looks identical (origin at the push midpoint, push direction
along +x), apply a single matrix `A`, warp back, and blend with the original where
the warp lost information. One `A` per push length. Control greedily by choosing
the action that most reduces `V = dᵀy/‖y‖₁`, a distance-transform-weighted average
of where the mass sits.

### 3.2 What failed, and the false leads

For most of this work the operator never beat persistence. Restating the error in
physical units made the scale clear: **the operator's error was 99% of the change
that actually occurred**, i.e. it explained under 1% of what happened. Three
explanations were tested and refuted:

- **"The plate is skimming."** No. Recorded 3-D state shows the plate bottom at
  cube-centre height with +2.49 mm vertical overlap, engaging 100% of cubes, and
  band particles move a mean of 18.9 mm per 40 mm push, sliding not tipping.
- **"Binary occupancy destroys height."** No. The pile was 100% one layer deep, so
  there was no height information to lose. (An earlier recommendation to un-clamp
  `particles_to_occupancy` is **withdrawn**.)
- **"The dynamics are stochastic."** No. Per-push displacement is ~84% predictable
  from what the occupancy grid already contains. Exact sub-pixel particle
  positions add only +0.04; cube orientation adds nothing.

One real obstruction *was* found: on a raw binary occupancy of 5 mm cubes, the
SE(2) rotation the method requires costs more accuracy than one push changes,
because the field has features at the pixel scale (`|∇²occ|/|occ| = 3.05`). A
σ ≈ 1 px blur removes the cost entirely. The paper's colour-thresholded diced
carrots were already in that regime, so it never hit this. **Treat σ≈1 smoothing
as a precondition of the method, not a hyperparameter.**

### 3.3 The actual cause

Stratifying by an exogenous measure of how much material the blade meets:

| contact stratum | particles in swath | linear R² | boosted R² | linear share |
|---|---|---|---|---|
| smallest | 1.24 | 0.708 | 0.781 | 91% |
| middle | 3.51 | 0.822 | 0.797 | 103% |
| middle | 5.48 | 0.810 | 0.737 | 110% |
| largest | 8.30 | 0.782 | 0.718 | 109% |

**Within a contact stratum, linear matches nonlinear.** The 26-point gap measured
on the pooled data is almost entirely dependence on one scalar. Blind sampling
produced a huge spread in that scalar; the paper's protocol did not.

### 3.4 Confirmation on piled, contact-sampled data

| target | scattered + blind | piled + contact-sampled |
|---|---|---|
| mean band displacement | 0.576 / 0.836 = **69%** | 0.785 / 0.852 = **92%** |
| forward displacement | — | 0.774 / 0.839 = **92%** |
| max displacement | — | 0.856 / 0.908 = **94%** |

n = 1872, 5-fold grouped CV by run.

The control-relevant `dV` target improves **less, and unevenly** (n = 1856):

| goal | scattered share | piled + contact share |
|---|---|---|
| point | 75% | **86%** |
| stripe | 33% | **73%** |
| centre | 70% | 68% — *no improvement* |
| corner | 38% | 44% |

So the sampling change makes *how much material moves* close to linear, while
*where it ends up relative to a goal* is only partly linearised in these features.

**Resolved.** The missing ingredient is the push **direction relative to the
goal**. Adding `cos/sin` of the heading, the push's projection onto the pile's
outward radial, and their interaction:

| goal | OCC features | + direction features |
|---|---|---|
| centre | 68% | **82%** |
| corner | 44% | **90%** |
| point | 86% | **91%** |

This is not a trick — it is the paper's own mechanism, arrived at from the other
direction. Their SE(2) canonical push frame *removes direction-dependence by
construction*, which is exactly what these hand features do. Our descriptor
feature set lacked any representation of heading, so a linear model could not
express "pushing north-east helps a north-east goal". Once it can, `dV` is
~90% linear.

**An earlier draft of this document reported 89–99% for `dV`.** That came from the
first 464 transitions and did not survive 4× more data. Treat any result here at
n < 1000 as provisional.

**Attribution is to the sampling, not the pile**, on three pieces of evidence:
conditioning scattered data on contact already gave the same range (§3.3); packing
density made no difference to the linear share; and the piled data reached only
1.09 cube-width packing and 1.1 layers, inside the regime the density test covered.

### 3.5 What does not transfer

A contact-switched *pixel* operator at `D = 256`, `M/D = 8.3`, properly determined,
6-fold LORO: every variant sits within 0.0003 rms of persistence against a fold
standard deviation of 0.014 — indistinguishable. **The insight is level-specific:
it holds for scalar summaries of a push, not for per-pixel image prediction.**

### 3.6 Control utility

Persistence predicts `dV = 0` for every action and cannot rank at all. Against
that floor, on held-out scattered data:

| goal | linear operator | `cumulative` heuristic |
|---|---|---|
| centre | 0.299 | 0.416 |
| corner | 0.472 | 0.737 |

(best-of-16 slate utility; 1.0 = oracle, 0.0 = random). Realised `dV` under
best-of-16 selection, mm, negative = toward the goal:

| goal | random | linear | boosted | oracle |
|---|---|---|---|---|
| centre | +0.353 | +0.040 | +0.014 | −0.040 |
| corner | +0.453 | −0.438 | −1.424 | −1.546 |
| point | +0.818 | −0.104 | −0.579 | −0.750 |

Note the inversion against per-pixel error: **the geometric heuristic ranks better
than the fitted operator** despite losing to it on pixels. A transport model gets
the *direction* of mass flow right, and direction is what `dV` depends on.

---

## 4. Why did per-pixel linear work for them and not for us?

This is the most important open question, since the paper's Table 1 shows their
pixel operator beating two deep models. Hypotheses, ranked by how much they would
explain and how testable they are.

### H1 — Object count and the continuum limit *(most likely, most testable)*

Their figures show on the order of 100–200 carrot pieces forming a near-continuous
mass. We ran 30–50 cubes. With enough pieces the occupancy image becomes a smooth
**density field**, and pushing it is advection of a continuum — which is genuinely
a linear operation on the density. With few pieces, individual objects move
discretely: a cube either gets caught by the blade or does not, tumbles or slides,
and those are threshold events no linear map represents.

This also subsumes the smoothness observation in §3.2: many small pieces produce a
blob-like thresholded image (low `|∇²occ|/|occ|`), which is exactly the regime
where the SE(2) warp is free.

**Test:** collect at 100–150 particles in a compact spawn with contact-aware
20 mm pushes, and re-run `fit_linear_foresight.py` / `loro_foresight.py` at the
pixel level. If the pixel operator starts beating persistence, H1 is confirmed and
the whole negative result was about particle count. **This is the single highest-value
experiment remaining.** Budget it as an overnight run (see §5).

### H2 — 2-D versus 3-D

Their simulation was Pymunk, strictly 2-D. Objects cannot ride over one another
and mass is *exactly* conserved in the image, so a transport operator can be exact.
Our Genesis scene is 3-D: cubes can climb, tumble, and leave the top-down
footprint. Measured vertical displacement is small but nonzero and rises in piles
(0.01 mm scattered → 0.032 mm piled).

Partially tested and *not* the main story — world-frame mass is conserved to 0.4%
in our data, so their `‖I‖₁ ≈ const` premise holds. But out-of-plane *rearrangement*
without mass loss is still a nonlinearity a 2-D sim cannot produce.

**Test:** compare the linear share between our single-layer scattered data (which
is effectively 2-D) and genuinely multi-layer data. Requires H1's larger particle
count to get real depth.

### H3 — They never compared against persistence *(cheap to reason about, impossible to test directly)*

Their Table 1 reports 1.858 (linear) against 2.062 and 2.537 for the deep models.
There is **no "predict no change" column**. It is entirely possible their operator
also failed to beat persistence and nobody checked — in which case our result is
not a contradiction of theirs at all, and the honest comparison table should carry
a persistence row that theirs lacks.

**Test:** not directly testable without reimplementing their simulator. But it
should be *stated* in any write-up, because it changes what "reproducing their
result" means.

### H4 — Object shape and rotational degrees of freedom

Their carrots are flat, irregular convex hulls — effectively 2-D discs that slide.
Our cubes are 3-D and can tumble about any axis. Cube tumbling is a discrete,
strongly nonlinear event with no linear-transport analogue.

**Test:** collect with `--particle-shape sphere` (spheres roll but do not tumble
discontinuously) or with flat cylinders, and compare the linear share. Cheap —
the shape flag already exists.

### H5 — Push length relative to pile size

A linear operator is a first-order approximation, so it should suit small
perturbations. Our 40 mm pushes displaced a large fraction of a ~57 mm pile.
Their 5 length bins on a board-sized pile may have been proportionally gentler.

**Partially addressed** — push length is now 20 mm (4 cube widths). The
contact-stratified result (§3.3) suggests perturbation *size* is not the operative
variable once contact is conditioned on, so this is likely secondary. Testable for
free by comparing the 20 mm and 40 mm piled datasets once both exist.

### H6 — Image resolution relative to object size

Theirs: 32×32 over a cutting board, carrot pieces ~1 px each. Ours: 64×64 over a
128 mm tray, 5 mm cubes ≈ 2.5 px. Similar in ratio, so this is unlikely to be the
difference on its own — but it interacts with H1, since what matters is pieces per
image, not pixels per piece.

---

## 5. Practical constraints discovered

### Piled collection is ~100× more expensive per transition

A compact heap is one large contact island, so the **post-push settle** dominates:
~4 minutes per 2-push batch, against ~0.18 s/transition for scattered collection
at 64 envs. The settle cannot be damped without biasing `s'` toward smaller
displacements. The *library* settle **can** be (`--state-library-damping 15` took
it from ~25 min to seconds) — always use it.

`n_envs` is the only real lever, since the settle amortises across envs. Run as
high as VRAM allows; the optima in `Genesis/configs/measured/throughput_optimal.yaml`
were measured on **scattered** piles and are optimistic here.

**Plan any piled collection at useful volume as an overnight job.**

### The pile outgrows the blade's workspace box

A settled, pushed-around 30-cube pile reaches particle radius p95 34.6 mm / max
54 mm, while the blade's allowed box is 23.5–42.5 mm half-extent (yaw-dependent).
Placing the blade one particle-width behind such a pile's near face lands
**outside** the box 35.8% of the time, and such a push has nowhere to travel —
3.3% came out at ~0 mm. Fixed by clamping the start into the box (a spread pile is
entered rather than approached) and shortening the push to 20 mm.

### 30 cubes cannot form a deep pile

3750 mm³ of cubes at a ~30° angle of repose settles into a dense **monolayer**
(90% layer 0, z span one cube) whatever the spawn does. Depth needs more particles
or a smaller container. This is why H1 and H2 need the same experiment.

---

## 6. Methodological traps (read before running anything)

1. **Never trust a single train/test split here.** With 8 data files the holdout is
   one file and the fold-to-fold standard deviation of rms is ~0.004 — several
   times the effects being compared. Two "wins" were noise and did not survive
   seeding (a gating variant won on seed 0 and lost 3 of 6 seeds). Use
   `loro_foresight.py`, and compare **paired, fold by fold**.
2. **Occupancy-per-pixel differences are uninterpretable.** Always restate as a
   share of the change that actually occurred (`interpret_foresight.py`), where
   100% = no better than predicting nothing moved.
3. **Score against the identity operator, not only persistence.** `A = I` pushed
   through the same warp isolates resampling cost from model error; without it,
   resampling loss is silently attributed to the model.
4. **Whole-image error is ~95% pixels that could not have changed**, where
   persistence is exact by construction. Use the swept-region metric.
5. **Regularise toward `A = I`, not `A = 0`.** Plain ridge shrinks toward "the pile
   vanishes," which is wrong for a transport operator. (Measured caveat: at
   λ = 1e-2 the target makes no difference — the data term dominates.)
6. **The non-negative fit needs FISTA and ~4000 iterations.** Plain projected
   gradient at 300 was not converged and biased the operator toward erasing mass.
7. **The dataset grid convention appears transposed** relative to
   `INTERFACES.md` §4.1: brute-forcing all 24 orientation hypotheses against the
   dataset's own rasterised plate channel puts `dim0 = world_x` at 3.4 px, with the
   documented reading at 9.7 px. `fit_linear_foresight.py` re-checks this every run
   and aborts rather than fitting a transposed operator. **Unresolved** — needs a
   second look at which tensor the doc describes before editing a contract.
8. **`variance_decomposition.py`'s `PART` feature sets are numerically unstable**
   at larger n — sentinel values for empty swaths produce outliers that make the
   ridge fit return negative R². The `OCC` set is clean and is the one that
   matters (it is the grid-visible set). Fix the sentinels before using `PART`.
9. **A piled state library is not interchangeable with a scattered one.** The
   compatibility check covers particle count, size and shape but **not** spawn
   geometry, so a piled run pointed at a scattered library silently reuses the
   wrong states. Keep separate `--output-root` trees.

---

## 7. Recommended next steps

1. **Run H1 (100–150 particles, compact pile, contact-aware 20 mm pushes) and
   re-test the pixel operator.** This is the experiment that decides whether the
   paper's per-pixel result is reproducible at all in 3-D. Overnight job.
2. **Isolate sampling from piling** with one cheap control: contact-aware sampling
   on *scattered* geometry (`--pile-aware-actions` without `--pile-extent`).
   ~15 minutes, since scattered collection is ~100× cheaper. This turns §3.4's
   inferential attribution into a measured one.
3. **Move the working model to the descriptor level.** `dmdc_baseline.py` already
   fits per-action linear maps over ~50 analytic descriptors — well-determined at
   current data volumes, unlike a million-parameter pixel operator. Two additions:
   switch the operator on **contact amount** (§3.3), and express the Lyapunov cost
   as a linear functional in the descriptor basis (idea I1 in `ideas_log.md`), so
   MPC never reconstructs a grid.
4. **Then build the closed-loop controller** and score it on the Lyapunov descent
   curve — the paper's actual metric. Do not score it on one-step image error.
5. **Test H4 cheaply** with `--particle-shape sphere`.
6. Resolve trap 7 (the `INTERFACES.md` convention) and trap 8 (the `PART`
   sentinels).

---

## 8. Reproduction

```bash
# Scattered reference (data already collected)
python variance_decomposition.py                       # 69% linear share
python deltav_predictability.py                        # control-relevant target
python density_stratified.py --by packing              # pile hypothesis: no support
python density_stratified.py --by contact --bins 4     # THE finding: 91-110%
python control_utility_test.py --dataset configs/dataset/genesis_foresight_L040.yaml

# Pixel operator, leave-one-run-out
python loro_foresight.py --dataset configs/dataset/genesis_foresight_L040_full.yaml \
    --res 32 --crop 0.5 --blur 1.0 --bins 3 --folds 6 --ridge 1.0

# Piled collection (corrected: 20 mm, clamped starts). OVERNIGHT.
python -m Genesis.run_collection --plan configs/collection_pile30.yaml

# Piled analysis once >= 6 files exist
python run_pile_analysis.py
```
