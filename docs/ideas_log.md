# Ideas Log — Pre-implementation research for the descriptor-latent plan

**Status:** LIVE — maintained during the research/ideation task (2026-07-14).
**Anchors:** [`analytic_descriptors_latent_space_plan_v2.md`](analytic_descriptors_latent_space_plan_v2.md) ·
[`reports/dmdc_baseline_report.md`](../reports/dmdc_baseline_report.md) ·
Suh & Tedrake 2020 ([`2002.09093v3.pdf`](2002.09093v3.pdf), already partially exploited — descriptor-level switched-linear DMDc is implemented).
**Constraints (every idea must respect):** ≤14k non-sequential single-physics transitions
(or ~35k wide-physics) · 64×64 grids · one-step supervision only · end use MPC in z-space ·
closed-form/convex fits now, learned components later.

**Entry format:** provenance · mechanism · for/against · evidence status
(**probed** / literature-supported / speculative) · gates affected (G1–G6, plan-v2 §5) ·
verdict (*adopt now / small experiment / defer—needs new data / reject + why*).

---

## 1. Ideas from the inspiring paper (unexploited remainder)

### I1. Task-functional descriptors: the Lyapunov cost is (approximately) linear in z
- **Provenance:** paper §2.2 eq. (4): `V(y) = dᵀy / ‖y‖₁` — distance-transform-weighted
  mass, a *linear functional of the occupancy image*; CLF argument + greedy descent
  converged even for non-convex targets (their Fig. 12).
- **Mechanism:** our descriptors are inner products of ρ with fixed basis images
  (moments ↔ polynomial basis, DFT/projections ↔ Fourier basis, rings/SDF-bands ↔
  radial basis). Project the goal's distance transform `d` onto that basis:
  `V ≈ cᵀz` with `c` computed **in closed form per goal** (least-squares projection of
  `d` onto the basis images). MPC cost becomes a linear functional of predicted z —
  no grid reconstruction, and prediction error of V decomposes over descriptor groups.
  Optionally, for a fixed goal, append the exact scalar `⟨d,ρ⟩` as one extra z-dim and
  refit (closed-form, seconds) — then the dynamics predict the task cost *directly*.
- **For:** turns "task-aligned" from aspiration into arithmetic; principled weighting
  of the MPC cost across descriptor groups; the per-goal refit variant is exact and
  cheap; directly supports G5.
- **Against:** basis projection quality depends on goal-shape smoothness (fine goals
  need more basis dims); per-goal refit means the operator is goal-specific (mitigated:
  refit is seconds, and the projection variant needs no refit).
- **Evidence:** literature-supported (their CLF results) + our basis blocks already beat
  persistence (report §3.3–3.4). Projection quality is probe-able (P3, §4).
- **Gates:** G5 (cost candidate), G3 (V-error in physical units).
- **Verdict:** **adopt now** (projection variant); per-goal extra-dim refit as small experiment.

### I2. Physically-constrained least squares (their strongest estimator was constrained)
- **Provenance:** paper §3.1 + Fig. 7 — non-negative LS **beat** plain OLS for their
  pixel operator; row-sum-constrained over-regularized. Their eq. (10): row-wise
  decomposition keeps constrained QP tractable.
- **Mechanism:** in descriptor space the analogs are *physics* constraints, imposed
  row-wise on `A` (QP per row, D≈50 vars — trivial): mass row = conservation
  (`Δmass ≤ 0` inequality, or equality up to leakage); COM shift bounded by push
  length; ring/corridor rows non-negative where descriptors are non-negative masses;
  optional spectral bound ‖A‖ ≤ 1+ε (quasi-static ⇒ non-expansive dynamics).
- **For:** their empirical result says meaningful constraints beat generic shrinkage;
  constraints encode physics the 3-dim physics vector can't; row-wise QP is
  closed-form-adjacent (OSQP/active-set, milliseconds).
- **Against:** wrong constraints (their row-sum case) actively hurt — each constraint
  must be validated by G1-style ablation; ridge already gives some of this benefit.
- **Evidence:** literature-supported (their Fig. 7); mass near-conservation is
  probed in our data (mass ratio ≈ 1 vs persistence, report §3.3).
- **Gates:** G4 (estimator variant), G1.
- **Verdict:** **small experiment** — mass-conservation row + non-negativity on mass-like
  rows first; skip spectral bound until multi-step matters.

### I3. Action-geometric sparsity: persistence outside the "touched" descriptor set
- **Provenance:** paper Fig. 8 — their learned kernels are the **identity outside the
  push rectangle**; that sparsity is *why* a 10⁶-parameter operator learned from 5k
  samples generalized (sim→real).
- **Mechanism:** descriptor analog: under a push `u`, only descriptors overlapping the
  swept corridor + deposit zone can change; distant corridor bins, far rings,
  opposite-wall bands are persistence. Enforce as a **sparsity pattern on (A − I)**
  (or on B/Nᵢ columns in the continuous-u fit): rows/entries for "untouched" dims are
  frozen to persistence. Pattern computed from action geometry in closed form.
- **For:** massive effective-parameter reduction (the scarce-data regime collapse in
  report §3.5 is exactly what this prevents); the tool-frame corridor blocks make the
  pattern easy to define; complements (not duplicates) v2's persistence-prior.
- **Against:** long-range effects exist (mass pushed against a far wall piles back);
  pattern must be conservative (allow a halo); interacts with whitening (pattern on
  whitened Δφ rows is still well-defined).
- **Evidence:** speculative for descriptor space → **probe P1** (§4): measure |Δφᵢ|
  vs descriptor-to-push distance on oneset.
- **Gates:** G4 (scarcity curve is where it should shine), G1.
- **Verdict:** pending P1 → expected **small experiment**.

### I4. Lyapunov descent-curve as the closed-loop metric
- **Provenance:** paper §4.3/Fig. 11 — they benchmark models by **V-vs-iteration descent
  curves**, not success rates; steepness separates models that tie on one-step MSE.
- **Mechanism:** adopt descent curves (and area-under-curve) as G5's primary readout,
  against the heuristic-push surrogate sim; secondary: their "stuck-loop" failure check
  (predicted-best action changes nothing — their DVF failure mode).
- **For:** scalar, task-relevant, sensitive; directly comparable to the paper's figures.
- **Against:** none material.
- **Evidence:** literature-supported.
- **Gates:** G5 (sharpens its definition).
- **Verdict:** **adopt now** (evaluation-protocol change only).

### Paper assumptions our data violates (for the record)
- Push surface ⊥ push direction (their §2.1) — **violated**: plate angle independent of
  push direction (probed, plan-v2 §6). Their 1-scalar action reduction does not apply; u is 2-D.
- Data collected per-canonical-action on purpose (1000 pairs per length bin) — ours is
  random; bin occupancy is uneven (report §3.4: 26/72 bins underdetermined).
- Full-image prediction target (32×32 px) — we deliberately predict descriptors;
  their identity-outside-rectangle finding transfers as I3, not as pixel operators.

---

## 2. Literature sweep — ideas derived

### S1. Errors-in-variables fix: TLS-DMD / forward-backward debiasing
- **Provenance:** TLS-DMD (Hemati et al.; strong-consistency: Springer JJIAM 2022), fbDMD
  (Dawson et al. 2016) — noise in **both** φ_t and φ_{t+1} biases OLS/ridge operators.
- **Mechanism:** our descriptors are computed from rasterized particle states — both
  sides of the regression carry the same rasterization/discretization noise, so plain
  ridge suffers attenuation bias (systematically shrunk A beyond what λ intends).
  Total-least-squares DMD, or averaging forward and backward fits (`A_fb = (A_f·A_b⁻¹)^{1/2}`
  variants), debias in closed form.
- **For:** free accuracy at zero data cost; explains part of why fitted operators
  under-move mass; TLS is a one-line SVD change.
- **Against:** TLS is less numerically stable at D≈50 with collinear dims (needs the
  whitening + dedup already planned); backward fit assumes invertibility (pushes lose
  information — backward operator may be ill-posed).
- **Evidence:** literature-supported. **Gates:** G4 (estimator variant), G6.
- **Verdict:** **small experiment** (TLS vs ridge on oneset, one config each).

### S2. Analytic no-training nonlinearity: wavelet scattering block
- **Provenance:** Bruna & Mallat 2013 (1203.1513); rigid-motion scattering (1403.1687).
- **Mechanism:** first-order scattering coefficients (|ρ ∗ ψ_λ| averaged) are
  deterministic, smooth, deformation-stable descriptors — an *analytic* nonlinear
  block that needs no training, directly addressing the high-frequency regime where
  linear-in-DFT prediction broke down (report §3.3). A small block (2 scales × 4
  orientations ≈ 8 dims, tool-frame-orientable) fits the D budget.
- **For:** the only known way to add *nonlinear* image features while keeping the
  dynamics fit closed-form; deformation stability is exactly the right inductive bias
  for pushed piles.
- **Against:** modulus nonlinearity makes the dims non-reconstructible and their
  *dynamics* not obviously linear (must pass G1 like everyone else); partially
  redundant with |DFT| magnitudes.
- **Evidence:** literature-supported (as features); speculative (as linearly-predictable
  dynamics dims). **Gates:** G1, G4.
- **Verdict:** **small experiment** (8-dim block, G1 ablation).

### S3. Sliced-OT steering supports the projection-cost design
- **Provenance:** sliced-Wasserstein steering between measures (2604.22807);
  Wasserstein-space ensemble/swarm control (2303.15638).
- **Mechanism:** control-theoretic literature validates steering distributions by
  solving 1-D OT problems on projections — our corridor/lateral profiles are exactly
  such projections, so a sliced-OT MPC cost in z is not just a heuristic, it has a
  steering-theory pedigree (strengthens plan-v2's SW cost and I1).
- **Verdict:** **adopt now** (no new machinery — evidence upgrade for the planned cost).

### Sweep notes
- ACD-EDMD (2111.12256): dictionary construction should follow the system's
  configuration-space topology — v2's blocks already do this implicitly (mass/COM/cov
  = rigid modes, corridor/lateral = contact geometry, wall bands = workspace
  boundary); the transferable discipline is *completeness auditing*: check what
  state variation the descriptor set cannot express (feeds the learned-dims triggers, §6).
- Bilinear-Koopman MPC literature (2105.08036 etc.) confirms control-affine → lifted
  bilinear as the standard; v2's bilinear rung is well-grounded; note their warning
  that bilinear MPC costs ~10× linear MPC compute.
- Hankel/delay-embedding DMD variants are **inapplicable** (need trajectories; we have
  one-step pairs only).

---

## 3. Subagent proposals (condensed; full arguments in the critic round)

### Granular-physics lens (GP1–GP5)
- **GP1 Residual-DMDc over the snow-plow prior** — fit the ridge operator on the residual
  after applying `differentiable_push_cumulative/_spread2` in descriptor space
  (φ̂′ = encode(heuristic(ρ,u)) + Aφ + Bu + c); heuristic does the nonlinear part, DMDc
  corrects. *For:* only known convex route to nonlinear capacity; upgrades the plan's
  strongest baseline into the mean function. *Against:* bad heuristic bias could make
  the residual harder; inherits heuristic hyperparams. Evidence: residual-physics
  literature (Kloss/Schaal/Bohg 2020, Zeng 2019). Gates: G4, G5. **Probe P4:** per-group
  Var(φ′−φ̂′_heur) vs Var(φ′−φ) on ~2k transitions.
- **GP2 Swept-mass effective action** — replace/augment u with physically scaled
  features: m_swept (corridor partial sums up to depth L), saturating length basis,
  ×θ_rel products; the physics-selected sparse subset of bilinear terms (~10 params
  vs 2·D²). Evidence: plow-load scaling (Reece; Albert 1999; Gravish 2010). Gates: G4,
  G3, G1. **Probe P5:** corr(ΔCOM∥, m_swept) vs corr(ΔCOM∥, L).
- **GP3 Contact-line gate** — g = soft interface mass at the plate face at t; add gated
  copies g·φ, g·u as columns (varying-coefficient regression, closed form). Unilateral
  contact: g≈0 ⇒ Δφ≡0 exactly; current fit averages dead and loaded pushes. Gates: G1,
  G4, G5, G3. **Probe P6:** persistence MSE + model ratio per g-quartile; expect
  bottom quartile ≈ pure persistence.
- **GP4 Upwind-triangular corridor operator** — constrain corridor→corridor sub-block
  to forward-transport triangular (+ optional NNLS nonnegativity, mass-budget coupling
  to lateral/rings). Physics: a plow never pulls mass backward. *Against:* marginalized
  one-step operators may legitimately need backward entries (frame re-centering).
  Gates: G4 (scarcity), G5 (rollout monotonicity), G6. **Probe P7:** fraction of ℓ₁
  mass in the forward triangle of the fitted unconstrained sub-block (≥80% ⇒ free
  regularization; ~50% ⇒ kill).
- **GP5 Packing-fraction descriptor** — one smooth dim mass/soft-area (critical-state
  dilatancy: dense piles spread, loose piles gather under identical pushes). Gates:
  G1, G3. **Probe P8:** Var(φ_pack) nontrivial? corr(Δtr(cov), φ_pack) sign-crossing?
  Dead if grids are effectively binary.

### MPC/control lens (MC1–MC5)
- **MC1 Frame-then-QP** — per candidate frame the one-step cost is a 2-D box-QP in
  u = (L, θ_rel) with closed-form solution; batch over frames, argmax over achieved
  optima. Finally uses the plate angle. Evidence: Korda & Mezić 2018. Gates: G5, G2,
  G4. **Probe P9:** cond(BᵀB) of a global fit; rank-2 and mostly-interior u* ⇒ lives.
- **MC2 Adjoint frame refinement** — the whole cost a→frame→φ(a),u(a)→residual is
  smooth by v2's own design; Gauss-Newton on the world action from top-k seeds.
  Needs soft binning (bin-edge kinks). Gates: G5, G1. **Probe P10:** autodiff φ(θ)
  smoothness sweep on 20 grids.
- **MC3 Controllability audit / reachable-cone screening** — SVD of B: which z-groups
  are commandable at all (audit → G1 demotion criterion) and per-frame
  ‖proj_B e‖/‖e‖ as pre-QP frame score + infeasibility certificate. Gates: G1, G5.
  **Probe P11:** singular values + group loadings of fitted B.
- **MC4 OT-geodesic reference governor** — no model rollouts ever: Sinkhorn
  displacement interpolation gives waypoints spaced to one-push scale; receding
  one-step MPC tracks waypoints; long-horizon reasoning delegated to OT geometry
  (module exists in simple_mpc/ot_planner.py). Evidence: reference governors
  (Garone 2017), displacement interpolation (McCann). Gates: G5, G3. **Probe P12:**
  are geodesic increments within the observed one-push ‖Δφ‖ envelope, and inside
  span(B)?
- **MC5 Noise-calibrated decrease gate + drift certificate** — per-group residual
  covariance Σ_r (closed-form) → accept actions only when predicted decrease clears
  κ·√(gᵀΣ_r g); aggregate drift E[ΔV] ≤ −α gives an expected-steps-to-goal bound
  (supermartingale). Gates: G5 (pass criterion), G4, G3. **Probe P13:** fraction of
  (state, best-action) pairs whose predicted improvement exceeds 1σ.

### Operator-theory lens (OT1–OT5)
- **OT1 Moment-graded block-triangular operator** — restrict each moment row to
  regress on equal-or-lower-degree blocks + u (affine transport leaves the moment
  filtration invariant); ~6× parameter cut on the COM row; upward-coupling energy
  becomes a non-affineness diagnostic. Gates: G4 (scarcity), G1, G6. **Probe P14:**
  energy fraction of high-degree→low-degree entries in the fitted A; restricted-COM
  refit vs 0.53.
- **OT2 Perron–Frobenius structural constraints (piDMD-style)** — mass row structural
  (persistence + u-leak), profile-block sums tied to mass via equality constraints
  (KKT ridge), optional NNLS. Occupancy is a transported density; the operator should
  be sub-stochastic on profile blocks. Evidence: piDMD (Baddoo 2023), PF-EDMD (Klus
  2016). Gates: G1, G3, G4, G5. **Probe P15:** regress Δmass on L and wall proximity
  (fixes the leak-term form; R²≲0.05 ⇒ hard conservation safe).
- **OT3 Errors-in-variables (TLS/fbDMD/checkerboard-IV)** — rasterization noise sits
  in both φ_t and φ_{t+1} ⇒ attenuation bias + spurious mean-reversion that inflates
  "beats persistence" on low-SNR dims. Closed-form fixes: TLS-DMD, fbDMD, or 2SLS with
  checkerboard-split descriptor copies as instruments. (Merged with sweep item S1.)
  Gates: G3, G4, G6. **Probe P16:** checkerboard-split noise variance per dim vs
  Var(Δφ); EIV justified where noise ≥ ~10% of signal.
- **OT4 Reduced-rank residual operator + verified spectrum** — closed-form RRR on the
  whitened Δ-operator (rank swept on held-out); eigenvalues of I+A flag rollout
  instability (|λ|>1) before MPC compounds it; ResDMD-style residuals certify which
  eigenpairs are real. Protect span(B) from truncation. Gates: G4, G1, G5.
  **Probe P17:** singular values vs Marchenko–Pastur edge; count |λ|>1.
- **OT5 Mondrian split-conformal intervals** — distribution-free per-regime one-step
  prediction intervals (calibrated in wall-proximity × push-length bins); MPC risk
  term; quantifies the wall-regime worry. Evidence: Lei 2018, Lindemann 2023. Gates:
  G5, G3, §7. **Probe P18:** 90th-pct residuals by wall-proximity quartile (≥2×
  spread validates Mondrian + regime switch).

### Representation-learning lens (RL1–RL5)
- **RL1 RRR residual-direction mining** — closed-form CCA/RRR from grid-PCA scores to
  DMDc residuals; top-k left singular vectors are THE grid functionals linearly
  predictive of what the analytic set misses; append ~4–8, refit. Gates: G1, G2, G4,
  G6. **Probe P19:** RRR spectrum vs shuffled-pairs noise floor on 2k subsample.
- **RL2 Scattering block** (merged with sweep item S2) — tool-frame-steerable
  first/second-order scattering coefficients (~8–16 dims): deformation-stable,
  training-free access to the high-frequency regime where DFT-linear prediction
  failed. Risk: near-conserved energies ⇒ mass-like ratios ≈1. Gates: G1, G2, G3,
  G5. **Probe P20:** poor-man's Gabor-modulus block, per-dim ratio given (φ,u).
- **RL3 Random-feature pool + elastic-net selection** — 1–2k fixed nonlinear
  observables (random Fourier/wavelet-modulus/soft local pools), convex sparse
  selection against group residuals, stability-selection on bootstrap halves; ≤8
  survivors. Nonlinear complement to RL1. Gates: G1, G4, G6. **Probe P21:** LASSO
  out-of-fold R² on residuals from a 512-feature pool (<2–3% everywhere ⇒ kill).
- **RL4 Closure-gated admission protocol (G1+)** — any new dim must (a) beat a
  matched-variance dummy-dim conditioning floor, (b) itself be ≥ persistence-
  predictable (closure), (c) survive at the 3k scarcity point; failures of (b) demote
  to input-only observables (first-MPC-step features, never propagated). This is the
  sharpened G1 that Ideas RL1–RL3/GP5/S2 all pass through. Gates: extends G1,
  consumes G4/G6. **Probe P22:** measure the dummy-dim degradation floor (3 seeds).
- **RL5 Ridge-in-the-loop residual encoder (learned-phase design)** — small CNN
  g_θ(occ)→ℝ^k appended to φ; train by backprop through the closed-form ridge solve
  on split batches (Morton 2018/Bertinetto 2019 pattern), decorrelation penalty
  against analytic φ, jitter-smoothness penalty; deploy by freezing θ and refitting
  A,B,c closed-form. Trigger criteria (all three): RL1–RL3 saturated per RL4; some
  group still fails G3/G5 in physical units; **Probe P23** (nonlinear headroom:
  gradient-boosting vs linear R² gap on residuals from grid-PCA ≥5%). Gates: RL4
  admission per dim, G4, G5, G6.

---

## 4. Probes run on our data

All run on a seeded 2k-transition oneset subsample (14,246 available), nf=6, D=55.
Full script: session scratchpad `probes.py`; reproducible from `dmdc_baseline.py`
helpers. NOTE: the operator fitted *inside the probes* used only 1k train / 32 bins
(deliberately small) — its ratios >1 are the expected scarcity artifact (report §3.2),
not a contradiction of the report; probes read *structure*, not headline ratios.

- **P0a (done, plan-v2 §6):** plate angle ⊥ push? **No** — median relative angle
  0.93 rad, independent. u must be 2-D.
- **P0b (done, plan-v2 §6):** wall-proximity support? **Yes** — ~20% of mass within
  8 mm of a wall in essentially every frame.
- **P4 heuristic residual (GP1): VALIDATED for geometry, mass must be excluded.**
  Var(φ′−φ̂′_heur)/Var(φ′−φ_t): com **0.80**, moments2 **0.71**, dft_real **0.77**,
  dft_imag **0.74** — the oriented-splat heuristic (crude: perpendicular-plate
  fallback, unfitted width/sigma) already makes the residual 20–30% easier than
  persistence on every geometry group. **mass 11.6** — the heuristic's mass handling
  (clamping/deposit) is catastrophic; the mass row must stay persistence/structurally
  anchored (compose GP1 with OT2). Expect better with the cumulative variant + tuned
  width.
- **P6 contact gate (GP3): premise CONFIRMED.** 10.1% of pushes have ~zero interface
  mass; per-g-quartile persistence MSE rises monotonically 8.1e-5 → 1.68e-4 (~2×) —
  one-step signal concentrates in loaded pushes exactly as predicted.
- **P8 packing fraction (GP5): KILLED.** Grids are strictly binary (2 unique values);
  φ_pack = 0.9945 ± 0.0004 — a dead constant on this data.
- **P14 moment grading (OT1): AMBIGUOUS.** 67.8% of the low-degree rows' energy sits
  in high-degree→low-degree entries — either genuine non-affine transport (heaping)
  or noise absorbed by an underdetermined fit (probe's fit is 1k-sample). Does not
  kill OT1; the decider is the restricted-COM-refit vs unrestricted comparison at
  full data.
- **P16 checkerboard noise (OT3/S1): justified ONLY for mass.** Measurement-noise /
  Δ-signal (median per group): mass **13.2%** (above the 10% EIV threshold), com 1.9%,
  moments2 2.6%, dft ~3%. EIV/TLS correction is a mass-row fix, not a global one.
  Also independently confirms "the mass operator is mostly fitting noise" (report §3.5).
- **P17 spectrum (OT4/MC4): rank & instability CONFIRMED.** Whitened Δ-operator has
  24/55 singular values above the noise edge (effective rank ≈ 24 → rank constraint
  has real room); **10/55 eigenvalues of I+A exceed 1** (max |λ| = 1.079) — naive
  multi-step rollouts of the fitted operator diverge. Validates OT4's stability
  concern and MC4's no-rollout governor design.
- **P19 RRR spectrum (RL1): small but real headroom.** 4 cross-covariance singular
  values above the shuffled-pairs null (top: 0.97 vs null 0.70) — admit at most
  ~4 RRR dims, not 8.
- **P23 nonlinear headroom (RL5 trigger): re-audited 2026-07-16 — original design had
  a real bug, but the null conclusion survives a corrected rerun, with a quantified
  power limit.**
  - **Bug found:** the original probe regressed the residual **averaged across all
    dims in a group** (`res[:, s].mean(1)`). For the 24-dim DFT groups this is a
    genuine flaw — frequency bins are near-independent, so averaging cancels signal.
    Measured directly: var(group-mean target) / avg per-dim variance is **0.58%**
    for dft_real and **0.50%** for dft_imag, vs. the 4.17% (=1/24) expected under
    pure independence — i.e. the target was engineered by sign-cancellation to carry
    almost no residual signal, near-zero R² was close to guaranteed regardless of
    ground truth. (com/moments2 showed a milder version, ratios 0.52/0.42 vs.
    1/k=0.50/0.33; mass, k=1, is unaffected.)
  - **Corrected rerun** (regress each dim separately, average R² *after*): gap stays
    ≤ 0 for every dim in every group (mass −0.05, com −0.04, moments2 −0.09,
    dft_real −0.06, dft_imag −0.06; GBM beats linear on **0% of dims**). The bug did
    not change the conclusion.
  - **Power check (positive control):** injected synthetic nonlinear signal into the
    same feature matrix at known SNR fractions through the identical pipeline. Gap
    is clearly positive at ≥30% signal fraction (+0.13 to +0.67) but collapses into
    the same noise floor P23 measured (≈ −0.02 to −0.03) below ~15%. **The honest
    reading is "no nonlinear headroom above ~15–20% of residual variance is
    detectable here," not "no headroom exists."**
  - **Two further limitations:** (1) the residual came from a deliberately small
    1k-sample/32-bin fit *inside the probe* (25/32 bins underdetermined) — part of
    its variance is the operator's own scarcity noise, not unmodeled dynamics;
    (2) the input features never include the action `u` — any residual variance
    from within-bin continuous-action variation (continuous-u DMDc / GP2 / MC1's
    target) is structurally invisible to this test and would misreport as "no
    headroom" even if fixable by non-encoder means.
  - **Revised verdict:** trigger criterion **still not met** for building RL5 today,
    but on weaker grounds than originally stated. Before relying on a re-probe to
    green-light RL5, fix both the aggregation (per-dim, not group-mean) and the
    residual source (full-data fit, not the toy 1k fit), and treat detected effects
    below ~15–20% as inconclusive rather than "no".
- **P3 (open):** projection quality of goal distance transforms onto the descriptor
  basis (backs I1) — run during implementation of the MPC cost.

---

## 5. Considered-papers table

| # | Paper | Verdict (1 line) |
|---|---|---|
| 1 | Suh & Tedrake 2020 (2002.09093) — anchor | Deep-read; unexploited remainder extracted as I1–I4 |
| 2 | Neural Field Dynamics for granular piles (2311.00802) | Already the repo's learned baseline (NFDUNetFiLM); the bar the descriptor track undercuts |
| 3 | Dynamic-Resolution pile model, GNN (2306.16700) | Needs trajectories + big data; learned-phase context only |
| 4 | Schenck et al. CoRL 2017 (granular media) | Precedent: learned per-action conv predictor on height maps; superseded by anchor |
| 5 | Gaussian-Splatting visual MPC for granular (2410.09740) | Heavy learned pipeline; out of scope for closed-form phase |
| 6 | AdaptiGraph material-adaptive GNN (2024) | Physics-conditioning precedent for the learned phase only |
| 7 | ACD-EDMD analytic dictionaries (2111.12256) | Deep-read; completeness-auditing discipline adopted into §6 triggers |
| 8 | KEEDMD learned eigenfunctions (1911.08751) | Learned-phase candidate mechanism (eigenfunction-constrained encoder) |
| 9 | Koopman soft robotics review (2301.09708) | Confirms bilinear-lifting standard; nothing new to take |
| 10 | Koopman in robot learning survey (2408.04200) | Coverage check — no missed family relevant at our data scale |
| 11 | Stable Koopman via Hankel DMD (2408.06607) | INAPPLICABLE — needs trajectories (Hankel/delay embedding) |
| 12 | Koopman NMPC bilinear lifting (2105.08036) | Deep-read; bilinear-MPC compute warning (~10× linear) noted for G4/G5 |
| 13 | Koopman for interactive envs (2306.11941) | Abstract too thin to extract; skipped |
| 14 | CCA = RRR in high dims (2405.19539) | Supports RL1's estimator choice; regularized-RRR guidance |
| 15 | Invariant scattering convolution networks (1203.1513) | Basis for S2/RL2 scattering block (demoted behind RL1 by Critic A) |
| 16 | Rigid-motion scattering (1403.1687) | Rotation-steerable variant needed for tool-frame S2/RL2 |
| 17 | Wasserstein-space swarm tracking (2303.15638) | Steering-theory pedigree for OT costs (S3) |
| 18 | Sliced-Wasserstein steering (2604.22807) | Direct support for projection-based sliced-OT cost + MC4 |
| 19 | Distributed OT swarm deployment (Krishnan & Martínez) | Context for MC4's OT-guided waypoints |
| 20 | TLS-DMD strong consistency (JJIAM 2022) | EIV fix theory; asymptotic/trajectory regime — evidence-inflation flagged |
| 21 | fbDMD / noise-corrected DMD (Dawson et al. 2016) | Alternative debias; backward operator ill-posed for pushes (info destroyed) |
| 22 | DMD multiverse survey (2312.00137) | Algorithm-family map; confirms no missed estimator variant |
| 23 | Elliott & Cakmak 2018 (dirt rearrangement, RF over grid) | Anchor's related work; discrete transition model — superseded |
| 24 | Wilson & Hermans 2019 (grounded state reps for piles) | Anchor's ref [25]; latent-space stuck-loop failure mode informs I4's check |
| 25 | Zeng et al. 2018 (pushing/grasping synergies) | Anchor's ref [28]; the affine-canonicalization precedent (already in v2) |

Deep-reads: #1 (full), #7, #12, plus working knowledge of #14/#15/#20/#21.
Skims: all others. Cap respected (25 skimmed / ≤10 deep).

---

## 3b. Critic round (two independent critics; verdicts adjudicated with §4 probes)

**Critic A (statistics / data-limits / duplication)** and **Critic B (systems / MPC
soundness / closed-loop failure modes)** reviewed all 27 entries. Convergent calls:

- **Merges:** I2 + OT2 + GP4-constraint + OT1-constraint → one **constrained-estimation
  track** (row-wise/KKT ridge machinery built once; mass row first). GP3 → joint
  experiment with GP2 (collinearity of g and m_swept must be measured, not assumed).
  MC5 → OT5 (one uncertainty-calibration layer, conformal preferred; drift bound is a
  dashboard number, not a certificate). S2 = RL2 (already merged).
- **Kills:** GP5 (confirmed by probe P8 — binary grids). S1/OT3 *as an accuracy play* —
  Critic B's argument is decisive: at deployment the model always feeds *measured*
  descriptors, for which the ridge-attenuated operator is already the MMSE predictor;
  EIV debiasing helps only in rollout regimes this system never operates in. P16
  survives as a noise-floor measurement for calibration.
- **Demotions:** I3 (motivating regime absent: 2.5k params vs 14k samples; soft
  persistence-prior already does this), OT1 (affine-filtration premise wrong for local
  pushes; P14 = diagnostic only), OT4 (diagnostic + day-one span(B) truncation
  protection; RRR optional), MC2 (Gauss-Newton deferred — but its load-bearing extract,
  **soft-binned descriptors**, is promoted to a day-one foundation), MC4 governor →
  **seeder only** (code already exists in `ot_planner.extract_push_candidates`;
  waypoint feasibility replaced by a model-free nearest-neighbor check over observed
  (φ, Δφ) pairs), OT5-as-cost-term (regime-switch trigger only; a wall-uncertainty
  penalty would make near-wall goals systematically unreachable), RL3 (sequenced
  after RL1 + P23).
- **Modifications that survive into the shortlist:** I1 — ranking-fidelity probe
  (Spearman of ΔV_proj vs ΔV_true over candidate actions) replaces raw projection R²;
  `c` is per-(goal, frame), recomputed per candidate (cheap). GP1 — one-step-MPC arm
  only (mean function needs the grid; multi-step in pure z is off the table), heuristic
  hyperparams frozen from tool geometry before fitting, mass row excluded (probe P4:
  heuristic mass residual ×11.6), judged only by a *different* heuristic family than
  it embeds. MC1 — θ_rel folded into frame enumeration so GP3's gate and bilinear
  terms don't break QP-ness; inner problem = 1-D box-QP in L (closed form always).
  MC3 — per-frame B is rank-2, so the audit must be *frame-swept* (stack frame-mapped
  B images, SVD that). I2/OT2 — "mass" is occupied fraction, NOT conserved material
  (spreading creates it, stacking destroys it): no sign constraints before P15
  (extended with packing/spread covariates); soft penalties only.
- **Cross-idea interactions (Critic A):** (1) **GP1×G5 circularity** — see §7 risk;
  (2) bilinear dynamics stay *linear in u* at one step, so MC1's QP survives the whole
  model ladder; (3) EIV-debias and the persistence prior point in opposite directions
  — the report's own zero-shot-beats-in-domain mass result says the shrinkage is
  beneficial; (4) conformalize the single scalar cᵀΔz, not 48 per-dim intervals;
  (5) I3 and GP3 are the same physics probed spatially vs temporally; (6) **global
  D-ledger**: individually-admitted dims can quietly recreate the D=87 regime — hard
  cap **D_dyn ≤ 60**, trades not just admissions; (7) offline-calibrated uncertainty
  degrades under MPC covariate shift — G5 must log empirical coverage online.

---

## 6. Learned latent dims (next phase) — trigger criteria & mechanisms

**Status: the trigger is currently NOT met, but on re-audited (weaker) grounds
than first stated (see §4's revised P23 entry, 2026-07-16).** The original P23 had a
real aggregation bug (averaged residuals across group dims before regressing, which
for the 24-dim DFT groups cancelled ~99.4% of the per-dimension signal by
construction); a corrected per-dimension rerun reached the same null result, and a
positive-control power check showed the pipeline reliably detects nonlinear signal
≥~30% of residual variance but loses power below ~15%. So the honest claim is:
**no nonlinear headroom above ~15–20% of residual variance is detectable, from
state-only features, on a scarcity-degraded 1k-sample residual.** It is not "zero
headroom proven." This must be re-measured — with the aggregation fix, on the
well-conditioned full-data residual, and ideally with `u` included as a control
input — after GP1/GP2/RL1 change the residual (they consume the easy structure first).

**Trigger criteria — all three required before any learned dim is built:**
1. **Closed-form saturation:** RL1 (RRR mining — P19 says ≤4 dims of linear headroom
   exist today), RL2 (scattering, P20-gated), and RL3 (random features, P21-gated)
   have all been run and their surviving dims admitted; the best remaining candidate
   fails RL4's rent test.
2. **A physical-units gap remains:** some group still fails G3/G5 needs (e.g., COM
   error > ~15% of a 40 mm push, or letter-goal shape terms demonstrably
   representation-limited in closed loop).
3. **Headroom re-probe ≥ 5%:** P23 rerun **with the aggregation bug fixed (per-dim,
   not group-mean) and on the full-data residual, not a toy fit** (same seeded
   splits as everything else, G6) shows ≥5% predictable-but-not-linearly-predictable
   residual variance. Given the measured ~15–20% power floor, a "pass" result is
   trustworthy; a "fail" result below that floor should be treated as inconclusive,
   not as proof of absence — re-run with a larger held-out set before concluding.

**Candidate mechanisms, in order of preference:**
- **RL5 ridge-in-the-loop residual encoder** (primary): small CNN g_θ(occ)→ℝ^k (k≤4),
  trained by backprop through the closed-form ridge solve on split batches
  (Morton 2018 / Bertinetto 2019 pattern), with (a) decorrelation penalty against the
  analytic φ (forces it onto the residual), (b) jitter-smoothness penalty
  (‖g_θ(occ) − g_θ(shift₁px occ)‖ — plan-v2 principle 1), (c) dihedral 8×
  augmentation (mandatory at 14k samples). **Deployment contract:** freeze θ, refit
  A, B, c closed-form on all data — the training objective is the deployed
  estimator's held-out loss, so no train/deploy mismatch by construction.
- **KEEDMD-style eigenfunction encoder** (alternative): learn Koopman eigenfunctions
  anchored to the fitted operator's verified spectrum (P17 gives the candidate
  eigenvalues); more structure, less capacity — preferable if the residual is
  low-dimensional and slow.
- **RL3 random features** (pre-neural fallback): if headroom exists but is simple,
  convex selection over a fixed nonlinear pool may capture it without any training
  loop at all — try before RL5.

**Admission:** every learned dim passes RL4 individually (rent vs dummy-dim floor,
closure ≥ persistence-predictability, scarcity survival, no degradation of existing
groups) and respects the global D_dyn ≤ 60 ledger. Dims failing closure are demoted
to input-only observables (first-MPC-step features, never propagated).

---

## 7. Final ranked shortlist (EV ÷ implementation-cost, descending)

**#1 — Foundations bundle: soft-binned descriptors + seeded splits + RL4 admission
harness + D-ledger.** *(~2 days; prerequisite for everything)*
Soft binning (Gaussian bin memberships for corridor/lateral/rings) must exist before
any fitting — retrofitting it invalidates every fitted operator and probe. The RL4
harness (dummy-dim floor via P22, closure gate, no-degradation check, D_dyn ≤ 60 cap)
is what keeps all subsequent admissions honest. First step: modify the three profile
blocks in the descriptor spec to soft bins; wire seeded splits into
`dmdc_baseline.py`; implement the dummy-dim floor measurement.

**#2 — I1 + I4 + S3: linear-in-z task cost and the evaluation stack.** *(~2 days;
zero new dims)*
Project each goal's distance transform onto the descriptor basis per candidate frame
→ MPC cost = cᵀz; validate by **ranking fidelity** (Spearman of ΔV_proj vs ΔV_true
over candidate batches on ~20 states), not projection R². Adopt Lyapunov descent
curves as G5's readout with the **dual-surrogate rule** (a heuristic family inside a
model never judges that model). First step: implement the basis-projection of D and
the ranking probe.

**#3 — GP2+GP3 joint: contact-gated swept-mass action features.** *(~1 day; ~12
params; probes passed)*
Soft cumulative m_swept over corridor bins + contact gate g (P6 confirmed: signal
concentrates in loaded pushes; 10% of pushes are dead) + saturating length basis, fit
jointly with a collinearity ablation (partial-R² probe P5-revised first). The
physics-selected sparse subset of the bilinear rung. First step: run P5-revised
(partial R² of m_swept given L on ΔCOM∥), then add the feature columns to the
continuous-u fit.

**#4 — MC1 + MC3 + MC4-seeder: the planner.** *(~3–4 days)*
Frame enumeration (position, direction, θ_rel) × closed-form 1-D box-QP in L per
frame; frames screened by ‖proj_B e‖ (MC3, frame-swept audit) and seeded by the
existing Sinkhorn candidate extractor. Compatible with every model rung (bilinear
stays u-linear at one step). First step: P9-revised (fit global continuous-u model,
check cond(BᵀB) and u*-sensitivity-to-state).

**#5 — GP1: residual-DMDc over the frozen heuristic prior (one-step arm).**
*(~1 day; biggest single accuracy lever measured: 20–30% residual reduction pre-fit,
probe P4)*
Mean function = `differentiable_push_cumulative` with geometry-frozen hyperparams;
fit the ridge operator on the residual; **mass row stays persistence/structural**
(P4: heuristic mass ×11.6 worse); judged only on held-out one-step data and by the
*other* surrogate family in closed loop. First step: rerun P4 with the cumulative
variant + tuned-by-geometry width to confirm the splat-probe numbers improve.

**#6 (honorable) — I2/OT2 constrained mass row.** *(~1–2 days, after P15)* The only
idea aimed at the one failing group; soft penalties only; P15 (Δmass regressed on L,
wall proximity, packing/spread covariates) fixes the leak-term form first.

**Deferred with dignity:** RL1 (run after foundations; P19 caps it at ~4 dims),
RL2/S2 (P20 first; expect input-only demotion), RL3/RL5 (§6 triggers), OT4/OT1/P14/
P17 (free diagnostics on existing fits), GP4 (P7 wall-split probe only), MC2-GN, I3.

**The one risk that outranks all ideas (both critics, independently):
G5 is circular.** The only closed-loop judge is a surrogate built from
`diff_mass_push.py` heuristics; GP1 puts that family inside the model, GP2's
m_swept is the surrogate's sufficient statistic, and I4's curves are scored by it.
Mitigations adopted into the plan: (a) hard cross-surrogate rule (`cumulative`-based
models judged on `spread2` dynamics and vice versa); (b) **flagged as needs-new-data**
(cheap): collect ~50 short *real Genesis* rollouts as a one-time, non-circular
closed-loop test set — the only genuine multi-step signal this project can obtain,
and the difference between "our MPC works" and "our MPC works against its own
teacher". Additionally: all offline-calibrated uncertainty (OT5) must log empirical
coverage online during G5 runs (covariate shift).
