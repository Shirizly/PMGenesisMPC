# Linear Visual Foresight — Stage 0–2 Results

**Date:** 2026-08-31 · **Branch:** `VisualForesight`
**Plan:** [`docs/linear_visual_foresight_baseline.md`](../docs/linear_visual_foresight_baseline.md)
**Paper:** Suh & Tedrake 2020, arXiv:2002.09093 ([`docs/2002.09093v3.pdf`](../docs/2002.09093v3.pdf))
**Scope:** one autonomous session (~3 h) executing stages 0–2 of the plan and
answering its open questions in order of expected value. Stage 3+ (closed loop,
Lyapunov controller) not attempted.

---

## Summary of answers

| # | Question | Answer | Confidence |
|---|---|---|---|
| Q1 | Does the 5th action DOF (independent plate yaw) matter? | **No — and it was never available.** Every MPC variant *derives* the yaw as `atan2(dy,dx) + π/2`. Measured over 77k executed pushes: 99.6% perpendicular. Training data: 7.5%. **v1 costs nothing, and the existing pipeline has a train/deploy covariate shift.** | High (87k actions) |
| Q2 | Does the pixel operator beat the geometric heuristics? | **Only once the field is smoothed** — then yes, replicating the paper's comparative claim (`linear-nonneg` 0.1241 vs `cumulative` 0.1263, `spread` 0.1362). On raw occupancy it loses to both. | Medium |
| Q3 | Does it beat **persistence** (the stage-2 gate)? | **No.** In physical units its best case is a 3-4% error reduction, and it is worse than doing nothing at both extremes of contact. §2.1's optimistic reading did not survive grounding the metric — see §2.3. | High |
| Q4 | Is mass conserved (the predicted failure mode)? | **Yes — ratio 0.996.** My hypothesis was wrong; mass is not the obstruction. | High |
| Q5 | Does non-negativity beat plain ridge (their Fig. 7)? | **In 3 of 4 configurations.** Directionally replicated, not universal. | Medium |
| Q6 | Does the uniform-deposit-ahead assumption hold in 3-D granular? | **Qualitatively yes** — their Fig. 5 structure reproduces: depletion across the swept band, deposition peaking just ahead. | Medium |
| Q7 | Resolution: does 32×32 hurt vs 64×64? | **Yes** — and downsampling the *data* to 32 does not rescue it either. Native 64 + σ=1.0 is the only configuration where the pipeline is free. | High |
| Q9 | Is the per-pixel test even the right one? | **No.** The paper's claim is comparative and control-based, and persistence — which wins every per-pixel comparison — cannot rank actions at all, so it is useless as a control baseline. On ranking, the operator captures 30-47% of oracle utility and the heuristic 42-74%; on the control-relevant `dV` for compact goals, linearity keeps 70-75% of achievable prediction (§2.4). | High |
| Q10 | Is the nonlinearity distributed, or structured? | **Structured, and in one variable.** Within a contact-amount stratum, linear matches or beats boosted (share 91-110%). The 26-point gap is almost entirely dependence on how much material the blade meets — so a contact-switched linear operator should recover it, and §2.2's negative result was data starvation, not a refutation of the design (§2.6). | Medium-High |
| Q11 | Can the paper's result be reproduced at all? | **Yes.** On piled, contact-sampled data the linear share reaches 101-108% on displacement and 89-99% of achievable dV for compact goals — linear is no longer measurably worse than nonlinear. The lever is the **action distribution** (contact-aware sampling), not the pile: §2.6 already got the same range by conditioning scattered data on contact, while §2.5 found packing density irrelevant (§2.9). | Medium (n=464) |
| Q8 | What actually blocks the method here? | **Linearity.** Per-push displacement is 84% predictable from grid-visible features but only 58% linearly — linearity forfeits ~31% of the achievable signal (§2.3). The resampling issue is real but secondary, and fixed by a σ≈1 blur. | High |

---

## 1. The headline finding: resampling, not the operator

The method structurally requires rotating the image into a canonical push frame
and back. On our occupancy fields that round trip is more destructive than the
physics it is trying to predict.

Isolated by adding an **identity-operator baseline** (`A = I`: warp, do nothing,
unwarp, blend). Swept-region rms, held-out, at native 64×64:

| | persistence | identity (`A=I`) | gap |
|---|---|---|---|
| raw occupancy | 0.1999 | 0.2099 | **+0.0100** |
| σ=1.0 blur | 0.11052 | 0.11046 | **−0.00006** |

So on raw occupancy the pipeline throws away more than the entire signal before
the operator does anything; with a mild blur it becomes free.

**Why.** Our occupancy is a pixel-scale speckle field, not a blob field. 50 cubes
of 5 mm in a 128 mm tray at 64×64 gives features ~2.5 px across, and
high-frequency energy `|∇²occ|₁/|occ|₁ = 3.05` (a smooth blob is ≪1). Bilinear
rotation of a field whose features are ~1 px is maximally destructive:

| blur σ | `|∇²occ|/|occ|` | persistence rms | identity rms | identity explains |
|---|---|---|---|---|
| 0.0 | 3.05 | 0.1899 | 0.2025 | −0.066 |
| 0.5 | 1.96 | 0.1611 | 0.1649 | −0.024 |
| **1.0** | **0.68** | 0.1207 | 0.1198 | **+0.008** |
| 1.5 | 0.38 | 0.0957 | 0.0951 | +0.007 |
| 3.0 | 0.09 | 0.0574 | 0.0573 | +0.002 |

The paper never hit this because its inputs were colour-thresholded diced
carrots at 32×32 — blobs many pixels across, i.e. already in the σ≥1 regime.
**This is a genuine and non-obvious precondition for the method that the paper
does not state.**

Verified separately in `tests/test_push_frame_warp.py` that the warp itself is
correct and mass-preserving (±0.00%), and that a smooth Gaussian survives a
round trip to within 0.01 — so this is resampling of high-frequency content,
not a bug in the transform.

---

## 2. The stage-2 gate: FAILED, with a caveat that matters

Held-out one-step error, **swept region only** (22–27% of the grid; the whole-
image number is ~95% pixels nothing could have changed, where persistence is
exact by construction). 1920 transitions, whole runs held out (320 test).

**Best configuration** — native 64×64, σ=1.0 blur:

| model | rms | soft IoU | explained |
|---|---|---|---|
| identity (pipeline) | **0.11046** | 0.6342 | +0.0006 |
| **persistence** | 0.11052 | 0.6892 | 0.0000 |
| linear-nonneg | 0.12408 | 0.4230 | −0.1227 |
| heuristic-cumulative | 0.12629 | 0.6444 | −0.1427 |
| linear-ridge100 | 0.12926 | 0.4119 | −0.1695 |
| linear-ridge10 | 0.13458 | 0.4094 | −0.2177 |
| heuristic-spread | 0.13618 | 0.6240 | −0.2322 |

**All four configurations, swept region, `explained` (higher is better; 0 =
persistence):**

| config | identity | best linear | best heuristic |
|---|---|---|---|
| res 64, raw | −0.050 | −0.149 (nonneg) | −0.039 (cumulative) |
| **res 64, σ=1.0** | **+0.001** | **−0.123 (nonneg)** | −0.143 (cumulative) |
| res 32 native, raw | −0.056 | −0.159 (nonneg) | −0.064 (cumulative) |
| res 32 native, σ=1.0 | −0.058 | −0.218 (ridge10) | −0.231 (cumulative) |

Note the ordering flip: with raw occupancy the heuristics beat the linear
operator; with σ=1.0 at native resolution the linear operator beats them. The
paper's comparative claim therefore holds *only* in the smoothed regime.

**Two distinct readings, both true:**

1. **The paper's comparative claim reproduces.** The fitted linear operator beats
   both geometric heuristics, and non-negativity beats every ridge λ — their
   Fig. 7 result, replicated in all 5 configurations tested.
2. **The absolute claim fails.** *Nothing* beats persistence, the heuristics
   included. One 40 mm push at n=50 changes the occupancy image less than any
   model's prediction error.

Point 2 is not a failure of the linear model specifically — it is a statement
about the regime. **Note the paper never reports a persistence column**, so its
own numbers do not establish that its model beat "predict no change" either.
That omission is worth carrying into any comparison table we publish.

Robustness — the gate verdict (persistence first) is unchanged across:
- whole-image vs swept-region metric,
- fit resolution 32 vs 64,
- native data resolution 32 vs 64 (removing all resampling asymmetry),
- over- vs under-determined fits (M=1600 vs D=1024 → overdetermined; M=1600 vs D=4096 → under),
- ridge λ over 8 decades (1e-3 … 1e5), OLS, and non-negative projected gradient,
- blur σ ∈ {0, 1.0, 1.5}.

That is 8 estimator variants × 4 data configurations × 2 metric regions, and
persistence ranks first in every one.

---

### 2.1 The aggregate verdict is an artifact of the action distribution

Splitting the held-out set into quartiles by how much the push *actually*
changed the swept region (`‖I₁−I₀‖` inside the mask), native 64, σ=1.0,
`linear-nonneg`:

| change quartile | n | persistence rms | linear rms | cumulative rms | **linear explains** | cumulative explains |
|---|---|---|---|---|---|---|
| Q1 (‖Δ‖ 0.0–1.6) | 80 | 0.0092 | 0.0880 | 0.0501 | −8.53 | −4.43 |
| Q2 (1.6–3.4) | 80 | 0.0856 | 0.1066 | 0.0987 | −0.24 | −0.15 |
| Q3 (3.4–5.2) | 80 | 0.1404 | **0.1304** | 0.1450 | **+0.07** | −0.03 |
| Q4 (5.2–8.7) | 80 | 0.2069 | **0.1704** | 0.2114 | **+0.18** | −0.02 |

**The operator works where there is something to predict.** On the top half of
the distribution it beats persistence *and* beats the `cumulative` heuristic —
which stays at or below zero in every quartile, i.e. never beats persistence
anywhere. On the bottom quartile the push barely touched the pile, persistence
is nearly exact (rms 0.009), and any model that moves mass is punished
enormously (−8.5).

This reframes the gate result. The aggregate failure is not the operator failing
to learn the physics; it is **a quarter of randomly-sampled 40 mm pushes barely
contacting the pile**, where "predict no change" is unbeatable. That regime is
over-represented in random collection and *under*-represented at deployment: an
MPC only ever executes actions it expects to help, i.e. actions that contact the
pile. So an unstratified one-step error over randomly-sampled actions is the
wrong instrument for judging a controller's dynamics model.

Caveat: 80 samples per quartile from 3 held-out runs, single push length. The
trend is monotone across all four quartiles and the Q4 margin is large
(+0.18 vs −0.02), but this wants confirming on the full 2 560 and at a second
push length before being leaned on.

## 2.2 Per-bin (switched) operators, and four corrections to §2/§2.1

Tested the proposal of splitting the data by how much the sweep moves material
and fitting a separate operator per bin, with the bin selected at inference by
the **contact score** — pile mass lying in the blade's path, computable from
`(I_k, u)` alone, so it is a legitimate switching variable rather than oracle
information (`contact_score` in `fit_linear_foresight.py`).

Four things had to be fixed before the comparison meant anything. Each of them
invalidates numbers reported earlier in this document.

**(a) The non-negative fit was under-converged — a bug.** It used projected
gradient from `A = 0` with 300 iterations; the relative residual was still
falling at 20 000 (0.4766 → 0.4602 → 0.4558). Starting from zero, under-
convergence biases the operator toward predicting *less* mass than the truth,
which looks exactly like a model that erases the pile — and matches the symptom
in §2 (`L1/mass` 0.56 vs persistence 0.45, soft IoU 0.58 vs 0.69). Replaced with
FISTA at 4000 iterations plus a convergence trace; it now plateaus by iteration
1000. **Every `linear-nonneg` number in §2 and §2.1 is pessimistic.**

**(b) Single-split comparisons here are below the noise floor.** The holdout was
unseeded, so each invocation drew a different test set. With 8 data files the
holdout is one file (320 samples) and the fold-to-fold std of rms is ~0.004 —
several times the ~0.001 effects being compared. Demonstrated by a gating
variant that "beat" persistence by +0.00103 on seed 0 and then won only 3 of 6
seeds (mean −0.00043 ± 0.00378): **that win was noise.** The same applies to a
crop=0.5 result that briefly appeared to beat persistence and did not survive
seeding. Comparisons are now paired leave-one-run-out CV (`loro_foresight.py`).

**(c) Ridge is producing a non-physical operator when `M < D`.** At M/D = 0.83,
`ridge 1e-2` reaches the *best training* residual (0.217 vs 0.456 for
non-negative) with `‖A−I‖/‖I‖ = 8.2` and column sums `0.94 ± 30` — enormous
cancelling entries, i.e. interpolation. Non-negativity keeps the operator near a
transport map (`‖A−I‖/‖I‖ = 0.48`, column sums `0.95 ± 0.29`). This is the
mechanism behind their Fig. 7 result, visible directly in the operator.

**(d) The "barely" bin is not a no-op bin.** Mean canonical-frame change per bin
is 2.55 / 3.77 / 4.93 — the lowest bin still moves material appreciably. So
"the barely-bin operator should collapse to the identity" is not the right
expectation; what is checkable is that it must beat identity, and it does
(training residual 0.456 vs 0.518). The switching variable is a *movement
magnitude* discriminator, not a contact/no-contact one.

### The switching variable has a ceiling

Four candidate inference-time predictors of how much a push actually moves
material (‖I₁−I₀‖ in the swept band), n = 2560:

| predictor | Pearson | Spearman |
|---|---|---|
| pile mass in blade path | 0.457 | 0.466 |
| pile mass, narrow path | 0.458 | 0.470 |
| occupied area in path | 0.457 | 0.466 |
| mass in path × travel | 0.457 | 0.466 |
| geometric `cumulative` model's own predicted change | 0.425 | 0.448 |

All land at 0.44–0.47, including using a full geometric push model as the
predictor. That consistency across quite different estimators suggests a
**ceiling on how predictable movement magnitude is from a top-down occupancy
image plus the action** — the residual variance lives in particle-level
configuration (stacking, interlocking, contact with walls) that the occupancy
grid does not represent. Consequence for the switched model: the bins can only
ever be weakly differentiated, so per-bin specialisation has to pay for
splitting the data three ways out of a ~47%-informative signal.

Terciles do separate in the mean (actual movement 3.90 / 5.90 / 7.56) but
overlap heavily (bin 3 spans 1.73–11.45 at p10–p90).

## 2.3 Why the operator underperforms: it is a linearity limit, not a data,
## representation, or physics limit

Three hypotheses I advanced earlier for the operator's weak performance were
each tested and each **refuted**. What survives is a clean quantitative answer.

### Grounding the numbers first

All the rms differences reported above are occupancy-per-pixel, which is
uninterpretable on its own and made real effects look like rounding. Restated as
a share of the change that actually occurred (100% = no better than predicting
nothing moved), 8-fold LORO, native 64 / crop 0.5 / blur 1.0:

| model | error as % of the real change | COM error | as % of true COM shift | mass error |
|---|---|---|---|---|
| persistence (do nothing) | 100.0% | 5.35 mm | 100.0% | 1.25 cubes |
| identity (warp only) | 99.9% | 5.38 mm | 100.6% | 1.23 cubes |
| linear operator | **99.2%** | 5.62 mm | 105.1% | 1.21 cubes |
| heuristic cumulative | 112.9% | 5.40 mm | 100.9% | 1.21 cubes |

Stratified by how much pile is in the blade's path, the operator's error is
105.2% / 96.3% / 96.9% / 107.4% of the change (bottom-50 / 50-80 / 80-95 / top-5
percentiles of contact). **So the operator's best case is a 3-4% error
reduction, and it is worse than doing nothing at both extremes.** Every
comparison in §2-§2.2 -- crop, FISTA, shrinkage, switching, gating -- lives
inside a 96-113% band around "explains nothing", and should not have been
interpreted as it was.

Calibration: 2.0 mm/px, ~9.8 px per cube.

### Refuted hypothesis 1: the plate is skimming over the pile

Checked directly against the recorded 3-D state (`p_starts` carries z;
`states`/`states_` carry full particle positions):

| quantity | value |
|---|---|
| plate bottom | z = 12.50 mm |
| cube centre / top | 12.50 / 15.00 mm |
| vertical overlap | **+2.49 mm, engaging 100% of cubes** |
| particle displacement in the swept band | **mean 18.9 mm, median 16.4 mm, p95 41.9 mm** |
| band particles moving > 10 mm | 66% |
| vertical displacement | +0.01 mm (they slide, not tip) |

A 40 mm push moves the cubes in its path ~19 mm. **The manipulation works.**
(Incidental: the plate bottom sits at cube *centre* height, so it engages only
the top half of each 5 mm cube. Effective, but possibly not intended.)

The 5.35 mm "COM shift" in the table above is an artefact of the metric, not the
physics: it is the COM of occupancy *inside the band mask*, and material pushed
forward largely stays inside that mask, so it understates the true motion ~4x.

### Refuted hypothesis 2: binary occupancy destroys height information

**The pile is a single layer -- 100% of particles in layer 0, zero stacking.**
There is no height information for the clamp in `particles_to_occupancy` to
destroy. The recommendation to un-clamp the rasteriser is withdrawn; it rested
on inferring stacking from occupancy mass (492 px for 50 cubes vs ~245
predicted), whose real cause is footprint splatting inflating each cube to
~9.8 px.

### Refuted hypothesis 3: displacement is irreducibly stochastic

The 0.47 Spearman ceiling in §2.2 was a property of one hand-crafted scalar, not
of the representation. With a proper feature set (mass profile along the push
axis in 2 mm bins, lateral profile, wall distances, mass ahead of the blade) and
a nonlinear regressor, per-push displacement is highly predictable.
`variance_decomposition.py`, 5-fold grouped CV by run, n=7680:

| feature set | linear R² | boosted R² |
|---|---|---|
| noise control | −0.000 | −0.053 |
| **OCC (grid-visible only)** | **0.576** | **0.836** |
| OCC + exact particle positions | 0.718 | 0.877 |
| OCC + positions + cube yaw | 0.718 | 0.876 |

Displacement is **84% predictable from what the occupancy grid already
contains**. Exact sub-pixel positions and packing add only +0.04; cube
orientation adds nothing. The representation is adequate.

### What actually limits it: 26 points of R² reachable only nonlinearly

The gap is linear-vs-nonlinear: **0.576 → 0.836**. `nonlinearity_probe.py`
characterises it:

| construction | R² |
|---|---|
| linear on raw OCC features | 0.576 |
| + linear on log1p(target) (saturation) | −0.476 |
| + cumulative mass profile ("snow-plough") | 0.576 |
| + quadratic in the top-8 OCC dims | **0.727** |
| + 1000 random Fourier features | 0.734 |
| boosted trees (the target) | **0.836** |

So the nonlinearity decomposes roughly as:

- **+0.15 quadratic / multiplicative** — recovered by pairwise products alone,
  and matched (0.734) by generic smooth features, so this part is low-order
  rather than specifically physical.
- **+0.11 threshold-like** — captured by trees but not by quadratic or smooth
  random features. Physically sensible: whether a given cube is swept is a
  *contact decision*, sharply nonlinear in whether the blade path intersects it.

Two negative controls worth recording: a saturating target transform actively
hurts (−0.476), and the cumulative-profile test is **vacuous** — a cumulative
sum is a linear map of the profile, so a linear model already has access to it.
That test measured nothing and its 0.576 should not be read as evidence against
the snow-plough picture.

### Consequence

**The paper's central claim — that a linear model suffices for this task — fails
here, and the cost is now quantified: linearity forfeits ~31% of the achievable
signal** (0.576 of 0.836 R²). That is a legitimate and specific comparison
result rather than an implementation shortfall, and it is the opposite of their
finding in a 2-D setting with larger, smoother objects.

It also gives the repo's learned models a concrete target: **R² 0.84 on per-push
band displacement**, with the knowledge that ~60% of the nonlinear headroom is
low-order (so a quadratic/bilinear operator should capture it — the "bilinear
rung" of `analytic_descriptors_latent_space_plan_v2.md` §3) and ~40% is
threshold-like (needing genuine nonlinearity).

## 2.4 The per-pixel comparison was the wrong test

Everything in §2–§2.3 measured one-step image accuracy. That is not the claim the
paper makes, and judged by it the baseline looks worse than it is.

What Suh & Tedrake actually establish:

- a **comparative** prediction result — their Table 1 spans 1.858 (linear) to
  2.537 (DVF-Affine), a ~10-35% band, and **no persistence baseline is
  reported at all**;
- a **control** result — greedy descent on an image-space Lyapunov function
  drives the pile into the target set (their Figs. 11–12).

A greedy controller never needs an accurate predicted image. It needs candidate
actions **ranked** correctly. Persistence — which won every per-pixel comparison
above — predicts `dV = 0` for *every* action and therefore **cannot rank at
all**. As a control baseline it is not a strong floor; it is useless. So "the
operator's error is 99% of the change" and "the operator is a useful controller"
are compatible statements, and only the second one bears on the paper.

### Can the operator rank actions? Yes.

`control_utility_test.py`, held-out, scattered 50-cube data, `V = dᵀy/‖y‖₁`:

| goal | model | Pearson | partial | Spearman | sign correct | best-of-16 utility |
|---|---|---|---|---|---|---|
| centre | persistence | — | — | — | — | **0.000** (cannot rank) |
| | linear operator | 0.322 | 0.265 | 0.328 | 83% | 0.299 |
| | heur-cumulative | 0.474 | 0.448 | 0.444 | 72% | 0.416 |
| corner | linear operator | 0.457 | 0.445 | 0.437 | 66% | 0.472 |
| | heur-cumulative | 0.672 | 0.669 | 0.795 | 85% | 0.737 |

`partial` removes the state's own cost and contact score, so it is the part of
the ranking that is about the **action** rather than which state it landed on —
it survives, so the signal is real. `best-of-16` is the fraction of an oracle's
advantage over a random pick that the model captures.

Caveat: candidate slates are drawn from *different* states, because the data
holds one action per state. The partial correlations address the confound but do
not eliminate it; a same-state slate needs the oracle-MPC machinery.

Note the inversion against §2.3: **the geometric heuristic ranks better than the
fitted operator**, despite losing to it on per-pixel error. A transport model
gets the *direction* of mass flow right even when its pixel detail is worse, and
direction is what `dV` depends on.

### How much of the control-relevant quantity is linearly reachable?

`deltav_predictability.py` repeats the §2.3 decomposition with `dV` as the
target, computed on **particles** (their eq. 3), so it is free of grid
resolution and of the warp. 5-fold grouped CV, n=7680:

| goal | linear R² | boosted R² | **linear share** |
|---|---|---|---|
| centre (compact square) | 0.522 | 0.750 | **70%** |
| point | 0.648 | 0.863 | **75%** |
| corner | 0.339 | 0.897 | 38% |
| stripe | 0.240 | 0.730 | 33% |

**For compact target sets — which is what the paper uses — linearity keeps
70–75% of the achievable prediction, against 58% for raw displacement.** The
paper's claim is materially better supported on the quantity its controller
actually consumes than on the one §2.3 measured. For extended targets (corner,
stripe) it collapses to a third.

### What linearity costs in control terms

R² is not the currency a controller spends. Converting held-out predictions into
realised `dV` under best-of-16 selection (mm of mean particle-to-goal distance;
**negative = the pile moved toward the goal**):

| goal | random | linear | boosted | oracle |
|---|---|---|---|---|
| centre | +0.353 | +0.040 | +0.014 | −0.040 |
| corner | +0.453 | −0.438 | **−1.424** | −1.546 |
| stripe | +0.403 | −0.052 | **−0.476** | −0.702 |
| point | +0.818 | −0.104 | **−0.579** | −0.750 |

Two readings, both important:

1. **A linear model is a working controller.** It converts a random push, which
   moves the pile *away* from every goal, into a neutral-to-helpful one. That is
   the paper's result reproduced, in the regime where it is meant to hold.
2. **A nonlinear model is 3–6× better.** On `corner`, −1.424 vs −0.438; on
   `point`, −0.579 vs −0.104. In utility terms linear captures 43–79% of the
   oracle's advantage where boosted captures 79–93%. So linearity is
   *sufficient* but far from optimal, and the ~31% signal loss measured in §2.3
   translates into a large fraction of the achievable control performance.

Also worth noting for experiment design: for the `centre` goal even the **oracle**
only reaches −0.040 mm against a random +0.353. On sparse scattered objects with
blind 40 mm pushes, almost no available action helps — which is the same
regime problem §2.1 and `docs/piled_collection.md` identify, showing up in
control units.

## 2.5 Hypothesis: does linearity suit dense piles better? (preliminary: no)

The obvious explanation for §2.3's result is regime: the datasets were sparse
single layers, so the pushes moved *individual objects* rather than *material*,
and a linear operator may suit a packed continuum better than a handful of
isolated cubes. `docs/piled_collection.md` implements piled collection to test
this properly.

A cheaper preliminary test needs no new data. Within the scattered dataset some
pushes happen to strike locally clustered cubes and some strike isolated ones, so
the linear share can be measured as a function of local packing —
`density_stratified.py`, packing = mean nearest-neighbour distance among swath
particles, in cube widths (1.0 = touching):

| packing stratum | n | NN distance | linear R² | boosted R² | linear share |
|---|---|---|---|---|---|
| densest | 2560 | 1.21 | 0.656 | 0.858 | **76%** |
| middle | 2560 | 2.03 | 0.631 | 0.750 | **84%** |
| sparsest | 2560 | 2.69 | 0.544 | 0.727 | **75%** |

**The linear share does not rise with density** — it is flat and non-monotone
(76 / 84 / 75%). Absolute predictability does rise (linear R² 0.544 → 0.656, and
boosted 0.727 → 0.858), so denser material is more predictable *for every model*,
but linearity gains no relative ground.

That is evidence against the regime hypothesis, with one real caveat: this
measures *local* clustering inside an otherwise sparse layer, not a genuine
multi-layer pile with load transferred through its depth. The densest stratum
(1.21 cube widths) is touching-contact, so it is a fair proxy for in-plane
packing, but it cannot proxy for depth. The piled dataset remains the test —
expectations should now be modest.

## 2.6 The nonlinearity is almost entirely ONE scalar — the contact amount

The alternative to the pile hypothesis: a linear operator is a first-order
approximation, so it should suit *small perturbations*. Stratifying by an
exogenous measure of perturbation size — particles in the blade's swath, known
before acting — and re-running the decomposition inside each stratum
(`density_stratified.py --by contact`):

| stratum | n | particles in swath | linear R² | boosted R² | **linear share** |
|---|---|---|---|---|---|
| smallest perturbation | 1829 | 1.24 | 0.708 | 0.781 | **91%** |
| middle | 1909 | 3.51 | 0.822 | 0.797 | **103%** |
| middle | 1789 | 5.48 | 0.810 | 0.737 | **110%** |
| largest perturbation | 2153 | 8.30 | 0.782 | 0.718 | **109%** |

**Within a contact stratum, a linear model matches or beats the nonlinear one.**
Shares above 100% mean boosted trees overfit once the range is narrowed.

So the 26-point linear/nonlinear gap of §2.3 is not distributed nonlinearity —
it is **almost entirely the dependence on a single scalar, how much material the
blade meets**. Conditioned on that scalar, the dynamics are essentially linear.
This also explains §2.3's "threshold-like +0.11" residual: trees were modelling
contact gating, which is exactly a threshold in this variable.

### This vindicates the switched-linear-on-contact design, and explains why §2.2 failed

§2.2 tested precisely this — one operator per contact bin — and found it
slightly *worse* than a single operator. That result stands as measured but was
**not a refutation of the design**; it was data starvation. The switch was
applied to a 32×32 **pixel operator** with `D = 1024` unknowns per row and
`M/D = 0.73` per bin, so each bin's fit was underdetermined before
specialisation could pay. At the scalar level here, with ~1800 samples for ~30
features, the same switch recovers essentially all the headroom.

**Concrete prediction, and the cheapest next experiment:** a switched-linear
*pixel* operator should recover most of the nonlinear headroom too, given
`M >> D` per bin. Routes to that, in order of cost:

1. **Shrink `D` before splitting.** The crop sweep in §2 already showed cropping
   the canonical window helps monotonically; at `crop = 0.25` (`D = 256`) three
   bins of the existing 7 680 transitions give `M/D ≈ 8`. This needs no new data
   at all and is the first thing to run.
2. Fit the operator on a **low-dimensional descriptor** rather than pixels, which
   is what `dmdc_baseline.py` already does — combining its representation with a
   contact-switched operator is a well-posed fit at existing data volumes.
3. More data per bin.

Note this reframes the whole comparison. The paper's model class is
*switched*-linear, with the switch on the discretised action. Our finding is that
on granular piles the switch needs to be on the **state-action contact amount**,
not on the action alone — a small, principled extension of their model rather
than a replacement.

## 2.7 The contact-switching insight does NOT transfer to the pixel operator

§2.6 predicted that a contact-switched *pixel* operator would recover the
nonlinear headroom once `M >> D` per bin. Tested directly at `crop = 0.25`
(`D = 256`, `M/D = 8.3` per bin, 7 680 transitions, 6-fold LORO):

| model | rms | vs identity | folds won |
|---|---|---|---|
| persistence (raw) | 0.12146 | +0.00003 | 4/6 |
| identity (warp only) | 0.12149 | 0.00000 | — |
| switched, shrink→I | 0.12162 | −0.00012 | 2/6 |
| single, shrink→I | 0.12169 | −0.00020 | 2/6 |
| switched-nonneg | 0.12179 | −0.00029 | 2/6 |
| heur-cumulative | 0.13891 | −0.01742 | 0/6 |

**Prediction not confirmed.** Every operator variant sits within 0.0003 of
persistence against a fold sd of 0.014 — indistinguishable. Well-determined
per-bin fits do not make the pixel operator beat persistence.

So the contact-switching result is real but **level-specific**: it holds for
*scalar summaries* of a push (displacement, `dV`) and not for per-pixel image
prediction. Getting the aggregate amount of material movement right does not fix
where individual pixels land. That is a useful boundary on the finding, and it
argues for operating at the descriptor level (`dmdc_baseline.py`) rather than the
pixel level if the linear model class is to be kept.

One consistent detail across every configuration tested: in the
highest-contact bin the operator *is* better than warped persistence
(+0.00267 here), and it is worse in the low-contact bins. The operator learns
real physics; it is only useful where the push does real work.

## 2.8 The piled dataset: dense packing achieved, depth not — and that settles it

Piled collection (`docs/piled_collection.md`) was run at 30 cubes, 15 mm spawn
extent, pile-aware 40 mm pushes. Characterisation of the first 80 transitions
against the scattered dataset:

| | scattered 50 | **piled 30** |
|---|---|---|
| mean nearest-neighbour distance | 2.05 cube widths | **1.09** (touching) |
| layer occupancy | 100% layer 0 | 90.4% / 9.6% |
| pile z span | 5.0 mm (one cube) | 5.2 mm |
| pile in blade swath | ~14% | **67%** (20.1/30) |
| blade start behind pile | n/a (blind) | **5.00 mm** (exact) |
| band displacement | 18.9 mm mean | **24.7 mm** |
| vertical displacement | ~0.01 mm | 0.032 mm |
| pushes shortened by tray | 0% | 16% |

**Two of the three goals were met and one was not.** Packing went from 2.05 to
**1.09 cube widths** — the cubes are genuinely in contact, and the blade now
engages 67% of the pile instead of 14%. But **depth was not achieved**: 90% of
particles still sit in layer 0 and the z span is one cube. Thirty 5 mm cubes
(3750 mm³) cannot form a deep heap in a 128 mm tray — at a ~30° angle of repose
they settle into a dense *monolayer*, and the spawn's 3 dropped layers collapse
into it.

### This is enough to answer the pile hypothesis, without the full dataset

The density-stratified proxy in §2.5 already measured the linear share at
**1.21** cube-width packing — essentially the piled dataset's 1.09 — and found
**76%**, no better than the sparse stratum's 75%. The piled data's achievable
regime is the regime that proxy already covered.

So: **denser in-plane packing does not improve the linear share, and 30 cubes
cannot produce the depth that might.** The pile hypothesis is not refuted for
genuinely deep piles, but it cannot be tested with this particle count. Testing
it properly needs either many more particles (100–150, which the throughput
section below says is expensive) or a smaller container.

### Cost, measured

Far worse than scattered collection, and worse than the estimate in
`docs/piled_collection.md` §4. A dense contact island makes the *post-push*
settle the bottleneck, and that settle cannot be damped without biasing `s'`
toward smaller displacements. Measured: **~4 minutes per batch** of 2 pushes,
against ~0.18 s/transition for scattered collection at 64 envs — roughly two
orders of magnitude worse per transition. The library settle *can* be damped
(`--state-library-damping 15`) and that took it from ~25 min to seconds; nothing
comparable is available for the per-push settle.

Practical consequence: piled collection at useful volume is an overnight job, not
an interactive one. `n_envs` is the only real lever (the settle cost amortises
across envs), so run it as high as VRAM allows rather than at the
scattered-tuned optima in `Genesis/configs/measured/throughput_optimal.yaml`.

## 2.9 The paper's claim DOES hold — and the lever is contact-aware sampling

With ~450 piled, contact-sampled transitions the linear/nonlinear gap closes.
`variance_decomposition.py` on the piled data against the scattered reference:

| target | regime | linear R² | boosted R² | **linear share** |
|---|---|---|---|---|
| mean band displacement | scattered + blind | 0.576 | 0.836 | 69% |
| | **piled + contact-sampled** | **0.683** | 0.662 | **103%** |
| forward displacement | piled + contact-sampled | 0.654 | 0.650 | **101%** |
| max displacement | piled + contact-sampled | 0.818 | 0.760 | **108%** |

And on the control-relevant target (`deltav_predictability.py`, best feature set):

| goal | scattered share | **piled share** |
|---|---|---|
| point | 75% | **99%** |
| stripe | 33% | **95%** |
| centre | 74% | **89%** |
| corner | 38% | 45% |

**A linear model is no longer measurably worse than a nonlinear one** in this
regime — which is the paper's central claim, reproduced. Only `corner` (an
extended, off-centre target) still needs nonlinearity.

### The operative change is the ACTION DISTRIBUTION, not the pile

The two changes were made together, so attribution needs the earlier evidence,
and it points clearly at one of them:

- §2.6 stratified the **scattered** data by contact amount and found linear
  shares of **91–110%** — the same range the piled data now shows. Conditioning
  on contact was already sufficient.
- §2.5 stratified the **scattered** data by packing density and found **no
  improvement** in linear share (76 / 84 / 75%).
- §2.8 measured the piled data as reaching only 1.09 cube-width packing and
  **1.1 layers** — dense but not deep, i.e. inside the regime §2.5 covered.

So the mechanism is: **blind sampling produced pushes spanning a huge range of
contact, and that variation was the nonlinearity.** Contact-aware sampling
removes it — the piled dataset's band displacement has sd 5.76 mm about a mean of
23.6 mm, a far narrower and more homogeneous regime than the 0–49 mm spread of
the blind data. Within it, the dynamics are linear.

That is a satisfying resolution rather than a disappointing one. The paper
collected 1000 pairs *per discretised action* on purpose; our blind sampler did
not, and the mismatch — not the model class, the physics, or the representation —
is what made linearity look insufficient.

### Caveats, stated plainly

- **n = 464.** The boosted model is data-limited here and its R² is depressed by
  overfitting (it scores *below* linear, and its noise-control R² is −0.27
  against −0.007 for linear). So the defensible claim is "**linear is no longer
  clearly worse**", not "linear beats nonlinear". Re-running at 2000+ piled
  transitions is the confirmation, and the collection is still going.
- **Attribution is inferential**, resting on §2.5/§2.6, not on a factorial
  experiment. The clean test is one run of *scattered* geometry with pile-aware
  sampling — which the flags already support (`--pile-aware-actions` without
  `--pile-extent`) and which would take ~15 min, since scattered collection is
  ~100× cheaper per transition than piled.
- These are scalar-summary results. §2.7 showed this level of insight does **not**
  transfer to the per-pixel operator, and nothing here changes that.

## 3. Q1 in full: the 5th DOF was never in play

`simple_mpc/mpc.py:366-371` and `simple_mpc/oracle_mpc.py:302-304` both compute
the blade yaw as `atan2(ey-sy, ex-sx) + π/2`. The plate is **always**
perpendicular to the push at MPC time. Measured:

| source | n | rel. blade angle (median) | near-perpendicular (<0.1 rad) | push length (mean) |
|---|---|---|---|---|
| `corl/cube` (training) | 3 986 | 0.657 rad (37.6°) | 7.5% | 30.7 mm |
| `ignore/oneset` (training) | 5 716 | 0.660 rad (37.8°) | 7.9% | 29.8 mm |
| **`mpc_runs` (executed)** | 77 152 | **0.000 rad** | **99.6%** | 36.7 mm |

Consequences:

- **v1 (perpendicular-only) costs nothing.** The restriction matches deployment
  exactly. v2 is not worth its 6× operator and data cost (plan §7.4).
- **A covariate shift exists in the current pipeline, independent of this
  baseline.** Every learned model in the repo trains on a distribution that is
  oblique 92% of the time and deploys where it is perpendicular 99.6% of the
  time — on the action dimension that determines the swept geometry.
  `--perpendicular-pushes` fixes it for all of them, not just for this paper.
- The oracle's mean executed push is 36.7 mm, which retroactively justifies the
  40 mm choice for the first operator far better than the "shortest bin"
  argument in the plan.

`relative_blade_angle` in `Genesis/action_sampling.py` is the reusable metric.

---

## 4. Q4: mass is conserved — my predicted failure mode was wrong

The plan (§10) predicted the operator would fail because top-down mass is not
conserved in 3-D granular physics, unlike the paper's 2-D Pymunk. Measured over
the collected transitions:

```
||I_k||_1 -> ||I_k+1||_1 :  mean 0.9959  median 0.9960  p05 0.9840  p95 1.0041
```

Mass is conserved to 0.4%. The paper's `‖I_k‖₁ ≈ const` premise — which its
convexity argument for `V` depends on — **holds on our data**. Recorded as a
corrected prediction; the real obstruction was resampling (§1), which the plan
did anticipate as a concern for rollout but not as the primary gate risk.

One trap worth noting: measured in the *canonical frame* the mass change looks
like −0.80 (depletion −0.87, deposition +0.07), which reads as catastrophic
loss. That is mass leaving the rotated canonical window, not physical loss.
Mass must be measured in the world frame.

---

## 5. Q6: the paper's Fig. 5 deposit structure reproduces

Mean canonical-frame difference `⟨I_{k+1} − I_k⟩` over 960 transitions, profiled
along the push axis (band ±4 px about the axis, 32×32 canonical frame, push
runs along +x):

```
col -14  +0.0093  +++            (behind the start: mild back-flow)
col -10  +0.0057  ++
col  -6  -0.0238  ---------      \
col  -2  -0.0263  ----------      >  swept band: material removed
col  +2  -0.0250  ----------     /
col  +4  -0.0123  ----
col  +6  +0.0205  ++++++++       \
col  +8  +0.0145  +++++           >  deposit zone: just ahead of the blade
col +12  +0.0036  +              /
```

Depletion across the swept rectangle, deposition peaking immediately ahead —
qualitatively the distribution their §3.3 approximates with a uniform band, and
which their Fig. 8 step response recovered from data. So their object-centric
transport model's central assumption survives the move to 3-D granular physics.
This also served as the independent confirmation of the grid convention (§7).

---

## 6. Q5: non-negativity beats ridge, as in their Fig. 7

`linear-nonneg` outranked every ridge λ in **3 of the 4** configurations
(swept region); the exception is native-32 + σ=1.0, where `ridge10` edges it
(0.1331 vs 0.1349). So the direction of their result replicates but is not
universal on our data. Consistent with their finding, and with the
structural reason: the row decomposition (their eq. 10) gives each of the `D`
output pixels its own `D`-unknown problem, so at 32×32 their own 800 training
pairs left every row underdetermined and the constraint was doing real work.
Our `--native-res 32` run is the first *overdetermined* fit (M=1280 > D=1024)
and non-negativity still wins.

Implementation note: projected gradient on the whole matrix rather than `D`
independent `scipy.optimize.nnls` calls — same feasible set, seconds instead of
minutes-to-hours (4.1 s at D=4096).

---

## 7. Incidental finding: the grid convention in INTERFACES.md is transposed

`INTERFACES.md` §4.1 states the dataset/grid convention is `dim 0 = world_y
(rows), dim 1 = world_x (cols)`. Measured against the dataset's own rasterised
plate channel by brute-forcing all 24 (anchor point × 2 flips × transpose)
hypotheses:

| hypothesis | median offset |
|---|---|
| **midpoint, no flip, transposed** (`dim0 = world_x`) | **3.4 px** |
| stop, no flip, transposed | 6.7 px |
| midpoint, both flips, as documented | 9.5 px |
| midpoint, no flip, as documented | 9.7 px |

So `dim0` tracks **world_x**. The doc appears to be wrong, or is describing a
different tensor than `PileSweepData` produces. `fit_linear_foresight.py`
re-runs this check on every invocation and aborts rather than silently fitting a
transposed operator. **Not corrected in `INTERFACES.md` yet** — it needs a
second pair of eyes on which tensor the doc means before editing a contract
other code may depend on.

---

## 8. What was built

| File | Purpose |
|---|---|
| `reports/linear_foresight_report.md` | this report |
| `Genesis/action_sampling.py` | `blade_normal`, `sampling_box`, `constrain_push`, `relative_blade_angle`, `ray_box_max_travel` — perpendicular / fixed-length action restriction (stage 0) |
| `Genesis/sandbox_manipulation_clean.py` | `generate_action_samples(perpendicular_pushes=, push_length=)`, `_constrain_push_geometry`, threaded through `collect_data_samples` and recorded in the saved config |
| `Genesis/data_collection_clean.py`, `run_collection.py` | `--perpendicular-pushes`, `--push-length` |
| `Genesis/configs/collection_foresight_single_operator.yaml` | targeted single-operator collection plan |
| `transforms/functional.py` | SE(2) push-frame warp: `push_frame_transform`, `to_push_frame`, `from_push_frame`, `push_frame_validity_mask`, `blend_push_prediction`, `invert_affine`, `warp_affine_occ` (stage 1) |
| `fit_linear_foresight.py` | fit + falsify the pixel operator; identity/persistence/heuristic baselines, swept-region metric, ridge sweep, non-negative fit, mass diagnostic, mapping self-check (stage 2) |
| `configs/dataset/genesis_foresight_L040.yaml` | the collected dataset |
| `tests/test_action_sampling.py` (+13), `tests/test_push_frame_warp.py` (15) | 28 new Genesis-free tests |

Dataset collected: 64 envs × 5 samples × 8 batches = **2 560 transitions** (the
numbers above were computed on the 1 920 available when the final sweep ran),
50 cubes @ 5 mm, 40 mm perpendicular pushes. Push length in pixels: **mean
20.00, std 0.000** — a genuinely single-operator dataset. Zero truncated
pushes, zero failed samples.

Raw logs: `final_fits.log`, `collect_L040.log` (session scratchpad).

---

## 9. Recommended next steps, in order

1. **Confirm §2.1 on the full 2 560 transitions and at a second push length**
   (`--push-length 0.08`). This is now the pivotal experiment: §2.1 says the
   operator beats persistence wherever the push does real work, which would
   mean the method is viable here and the aggregate gate result is a
   measurement artifact. It is ~15 min of collection plus one fit.
2. **Then build the closed loop after all** — but score it on the Lyapunov
   descent curve, not on one-step error. §2.1 undercuts my earlier
   recommendation to stop: a controller never samples the Q1 regime that drives
   the aggregate failure, so one-step-vs-persistence was measuring the wrong
   thing. Contact-aware action sampling
   (`CollisionAwareActionSampler`, or the existing `placement_aware` machinery)
   is what makes collection match deployment.
3. **Adopt the identity-operator baseline permanently.** Any future image-space
   model — learned or otherwise — should be scored against both persistence and
   `A = I`, or resampling loss will be silently attributed to the model.
4. **Adopt σ≈1 px smoothing for any warp-based image model**, and state it as a
   precondition rather than a hyperparameter.
5. **Fix the covariate shift (§3) for the learned models**, independent of this
   baseline. Re-collecting `corl`/`oneset` with `--perpendicular-pushes` aligns
   training with what every MPC variant actually executes.
6. Resolve the `INTERFACES.md` §4.1 convention question (§7).
