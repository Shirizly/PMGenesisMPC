# State Representation Plan v2: Task-Aligned Latent Vector (z)

**Status:** refined design — supersedes `analytic_descriptors_latent_space_plan.md` (kept for reference).
**Grounding:** empirical results in [`reports/dmdc_baseline_report.md`](../reports/dmdc_baseline_report.md)
(per-action linear maps over analytic descriptors beat persistence: COM 0.53–0.61,
moments 0.79–0.82, low-freq shape ~0.85–0.90; mass ≈ tie; transfer via ridge prior works).
**Data constraints this plan must respect:** up to ~14k **non-sequential** single-push
transitions in one physics regime (oneset), or ~35k across a wide physics range
(corl/cube). No trajectories exist → all validation is one-step + closed-loop
against a surrogate; multi-step rollout evaluation is not possible with current data.

---

## 0. Design principles (lessons already paid for)

1. **Smoothness is a hard requirement for the dynamics latent.** A ridge-fit linear
   operator predicts every dim from every dim; discontinuous or wildly-scaled targets
   corrupt the shared fit. Non-smooth shape summaries are still useful — but only as
   monitoring/reward features, never as propagated state.
   → `z` splits into **`z_dyn`** (smooth, propagated) and **`z_mon`** (reward/monitoring only).
2. **Every dim must earn its conditioning cost.** Cutting D 87→55 improved *every*
   descriptor group on 11.4k transitions (report §3.3). Target **D ≈ 45–60** for `z_dyn`.
3. **Whiten before fitting.** Per-dim z-scoring on train statistics; ridge is not
   scale-invariant. Fit **Δφ (persistence-centered)**, which makes the ridge prior
   shrink toward the *identity/persistence* operator rather than toward zero — the
   correct prior for near-conserved quantities and the fix for the scarce-data
   collapse seen in report §3.5.
4. **Include the constant observable** (affine operators; EDMD convention; report §2.1).
5. **Transform descriptors, not pixels.** Tool-frame values of moments, projections,
   and polar profiles are computable exactly from the original grid by rotating the
   integration coordinates. Never resample/warp the 64×64 grid (corner loss,
   interpolation blur, mass leakage).

---

## 1. Frames and the affine "warp" (revised)

Tool-relative canonicalization is kept, with three corrections to v1:

- **The action does NOT reduce to one scalar.** Verified in data: the plate angle is
  sampled independently of push direction (median plate-vs-push relative angle
  0.93 rad; only ~8% of pushes near-perpendicular). Tool-frame action is
  **u = (push_length, relative_plate_angle)** — and note the current baseline ignores
  plate angle entirely, so exposing it is an improvement in either frame.
- **The fence is not a corner case.** Measured: ~20% of pile mass sits within 8 mm of
  a wall on average, in essentially every frame. Tool-frame invariance of the dynamics
  is therefore violated routinely, not rarely. → **Mixed-frame latent by design**:
  canonical blocks in tool frame, wall block always in world/wall frame. The
  warped-vs-unwarped choice is an A/B experiment (§5), not an assumption.
- **No grid warping** — see principle 5.

---

## 2. Revised latent layout

### 2.1 `z_dyn` — propagated dynamics state (≈ 47 dims + const)

| # | Block | Dim | Definition | Frame | Rationale / changes vs v1 |
|---|---|---|---|---|---|
| 0 | const | 1 | literal 1 (fit-side) | — | affine operators (was missing in v1) |
| 1 | mass | 1 | Σρ / (H·W) | invariant | unchanged |
| 2 | com | 2 | density-weighted centroid | tool | unchanged |
| 3 | cov | 3 | central 2nd moments (μ_yy, μ_xx, μ_xy) | tool | **replaces v1's A-spread + B-eigen/θ** — same information, smooth, no π-ambiguity / eigen-swap discontinuities; already beats persistence at 0.79 |
| 4 | moments3 | 4 | central 3rd moments (μ₃₀, μ₂₁, μ₁₂, μ₀₃) | tool | **replaces Hu moments** — smooth asymmetry/skew (where mass heaps up), the differentiable ancestor of Hu |
| 5 | corridor profile | 12 | mass ahead of the tool, inside the swept corridor (tool width), binned along the push axis | tool | **the keeper from v1's D/E**: exactly the "cumulative mass ahead" statistic used by `CumulativePushModel`'s snow-plow formula |
| 6 | lateral profile | 8 | mass binned perpendicular to the push axis (spillover to the sides) | tool | v1's 90° projection, kept |
| 7 | radial rings | 8 | mass in concentric rings centered at the tool contact point | tool | v1's E-radial, halved (16→8; wedges dropped — duplicated the projections) |
| 8 | wall block | 8 | mass in SDF distance-bands from the fence (6) + soft-min distance to walls, x and y (2) | **world** | v1's F pruned: SDF-band histogram is smooth; soft-min replaces the non-differentiable hard min; per-wall 4-vector folded into 2 by symmetry |
| 9 | entropy | 1 | Shannon entropy of normalized ρ | invariant | global dispersion, complements variance |

Total: **48** (incl. const). Radon at 45°/135° and the 2-D DFT block are dropped from
`z_dyn`: by the projection-slice theorem the projections and the low-freq 2-D FFT
carry overlapping information, and the tool-aligned 0°/90° pair (blocks 5–6) is the
push-relevant subset. The DFT block remains available as an optional standard-frame
alternative for the unwarped A/B arm (§5).

### 2.2 `z_mon` — monitoring / reward-only (never propagated)

| Feature | Dim | Note |
|---|---|---|
| Hu moments (log-scaled) | 7 | shape classification vs goal; log|h|, h7 sign caveat |
| Solidity, compactness | 2 | binarization- and perimeter-noise-prone at 64² — fine for reward |
| Euler characteristic | 1 | integer; event flag (pile split), useless in a linear fit |
| Ellipse (λ₁, λ₂, θ) | 3 | derived from cov for display/reward |
| swept-path entropy | 1 | action-conditioned → belongs with (state, action) logging, not state |

`z_mon` supports MPC costs and event detection; the transition model never sees it.

### 2.3 v1 → v2 dimension reconciliation

v1: 104 dims, all propagated. v2: 48 propagated + 14 monitoring.
Removed from the propagated state: eigen-parametrization (3), duplicated spread (2),
45°/135° projections (16), angular wedges (16), SDF profile surplus (8), hard
min-distances (2), Euler (1), Hu/solidity/compactness (9), swept entropy (1).
Added: const (1), third-order moments (4).

---

## 3. Transition-model family (the missing payoff of canonicalization)

In order of increasing capacity — each is a closed-form or convex fit:

1. **Switched-linear (current baseline):** per-action-bin `A_b`. Data-hungry: splits
   N across bins; needs samples/bin ≫ D.
2. **Continuous-control DMDc (primary v2 target):**
   `Δφ = A φ + B u + c`, with `u = (push_length, rel_plate_angle)` after
   canonicalization. **One fit uses all 11–35k transitions** (~30× more data per
   parameter than 32-bin switching); MPC optimizes continuous `u` by gradient descent
   instead of argmin over bins. At D≈50: ~2·D² ≈ 5k parameters vs ~700k scalar
   equations at 14k transitions — very well determined.
3. **Bilinear (Koopman-with-input):** add `Σᵢ uᵢ Nᵢ φ` terms — captures
   action-magnitude-dependent dynamics (a long push transforms the pile differently,
   not just more). Still closed-form; doubles parameters, still well-determined.
4. **Physics-conditioned (for corl/cube):** append normalized physics p (3 dims,
   already in every batch) to `u` / bilinear terms. Directly targets the one group
   that is physics-limited: mass (1.019 unseen-physics vs 0.950 single-physics,
   report §3.4).

All variants keep: whitened Δφ fitting, ridge prior → persistence (or → source
operators for transfer, report §3.5), float64 solve.

**Symmetry exploitation (free data):** descriptor transforms under the dihedral group
(4 rotations × 2 flips) are closed-form for every `z_dyn` block → either augment
(φ, u) pairs 8× or constrain the operators to be equivariant. In tool frame, y-flip
parity for the symmetric plate gives 2×.

---

## 4. Baselines the latent model must beat

1. **Persistence** (`Δφ = 0`) — per descriptor group, as in the report.
2. **Current DMDc baseline** — the switched-linear numbers in report §3.3–3.4
   (COM 0.53/0.61, moments 0.79/0.82, DFT ~0.87).
3. **Heuristic push in descriptor space** *(new, zero-fit, nonlinear)*: apply
   `SpreadPushModel` / `CumulativePushModel` to the occupancy grid, encode the
   result → φ̂′. Since block 5 is the sufficient statistic of the snow-plow formula,
   the linear model can in principle match this baseline — this test shows whether
   it actually does, and where the heuristics' pixel-space nonlinearity still wins.

---

## 5. Evaluation protocol (gates before implementation is "done")

| Gate | Test |
|---|---|
| G1 block ablation | each `z_dyn` block must improve ≥1 group's held-out ratio when added (and hurt when removed); blocks that don't are demoted to `z_mon` |
| G2 frame A/B | warped (tool-frame z, continuous u) vs unwarped (world-frame z + DFT block, binned actions) on identical splits — the warp must win, not be assumed |
| G3 physical units | report COM error in mm (vs ~40 mm typical push), mass error in particle-equivalents — ratios alone hide practical (in)significance |
| G4 model family sweep | switched vs DMDc-control vs bilinear at matched D; scarcity curve (300 / 1k / 3k / 11k transitions) with persistence-prior on |
| G5 closed-loop sanity | greedy/gradient MPC in z-space against a heuristic-push surrogate sim (no trajectories exist in data, so closed-loop is the only multi-step signal available) |
| G6 reproducibility | seeded splits (current by-file holdout is unseeded; report caveat) |

**MPC cost note:** blocks 5–6 give 1-D marginals → a sliced-Wasserstein-style
EMD proxy between z(state) and z(goal) comes essentially for free and is the
task-aligned cost candidate for G5.

---

## 6. Pre-implementation data checks (cheap, do first)

| Check | Status |
|---|---|
| plate angle independent of push direction → u is 2-D | **done — confirmed** (median rel. angle 0.93 rad) |
| wall proximity has data support | **done — confirmed** (~20% of mass within 8 mm of a wall, every frame) |
| per-dim variance & persistence-MSE of every proposed `z_dyn` dim on oneset | pending — any near-constant dim is dropped before fitting |
| corridor-profile bin count (12) vs typical push length in px | pending — bins should roughly match the ~40 mm push at 2 mm/px |
| dihedral-transform correctness of each descriptor block (unit test) | pending |

---

## 7. Risks / open questions

- **Wall-regime nonlinearity:** ~20% near-wall mass means wall contact is inside the
  operating envelope, and no linear-in-z model represents contact saturation well.
  If G1 shows the wall block underperforming, the fallback is regime-switched
  operators (interior / wall-contact) — still closed-form, modest bin split (2–4).
- **Plate-angle marginalization in the current baseline** may explain part of the
  residual gap to persistence in high-frequency shape; the u=2-D fit will answer this.
- **Descriptor completeness for MPC:** z is not invertible to ρ; the task cost must be
  expressible in z (G5 tests this). If letter-shaped goals need finer shape detail,
  extend block 5–7 resolution before reaching for learned encoders — same
  falsification logic as the report.
