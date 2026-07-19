# DMDc Falsification Baseline — Integration & Results

**Date:** 2026-07-14
**Script:** [`dmdc_baseline.py`](../dmdc_baseline.py)
**Question:** Are per-action **linear** maps over cheap **analytic** descriptors of the
occupancy grid already predictive of pile evolution — i.e. do they beat a
persistence baseline, with zero training loops? (DMDc / switched-linear visual
foresight, Suh & Tedrake 2020, at descriptor level rather than raw pixels.)

If they do, the latent-linear ("Koopman") hypothesis has legs and every learned
encoder/operator must beat these numbers. If they don't, the per-group error
report says *what* is not linearly predictable.

---

## 1. What was done

### 1.1 Integration into the codebase
`dmdc_baseline.py` shipped with the descriptor/binning/ridge-fit/eval math fully
implemented but the data bridge stubbed. The following was added:

| Area | Change |
|---|---|
| **Data bridge** | Implemented `load_transition_arrays()` — builds a `genesis` dataset via `registry.dataset_registry.build_dataset` and extracts, per transition: `occ_t` (`input[0]`), `occ_t1` (`target`), the raw world-frame push `[sx,sy,ex,ey]`, and a run id. |
| **Dataset accessors** | Added `workspace_bounds`, `get_raw_action(idx)`, `get_run_index(idx)` to `PileSweepData` ([`Genesis/training/dataset.py`](../Genesis/training/dataset.py)) — the raw action is *not* in the rasterised batch, so these recover it cleanly (no private-attr access). |
| **Split modes** | `--split {auto,registry,by-file}` + `--holdout-frac`. `registry` uses the physics-group split; `by-file` holds out whole files (correct for single-physics datasets where the registry leaves an empty test set); `auto` picks registry, falling back to by-file. |
| **Contiguity guard** | Auto-detects whether per-run samples form a contiguous rollout; the multi-step report runs only if they do (they never do in this data — see §3.1). |
| **Transfer learning** | Generalised `fit_per_action_operators` with a **ridge prior** (shrink toward a source operator set) and added `--transfer-from`, `--transfer-weight`, `--target-max-train`. |
| **Tests** | `tests/test_dmdc_baseline.py` — 9 fast, data-free unit tests (descriptors, binning, closed-form fit recovery, ridge-prior behaviour, contiguity, episode grouping). Full suite: **41 passed**. |

A train/test **leakage bug** in the first transfer implementation (the target was
re-loaded with an independent random split, overlapping the eval set) was found
and fixed by reusing `main()`'s single split.

### 1.2 Config files added
| Config | Dataset | Size |
|---|---|---|
| [`configs/dataset/genesis_dmdc_cube_n50.yaml`](../configs/dataset/genesis_dmdc_cube_n50.yaml) | `ignore/cube/n50/size0.005` | ~140 transitions (smoke test) |
| [`configs/dataset/genesis_dmdc_oneset.yaml`](../configs/dataset/genesis_dmdc_oneset.yaml) | `ignore/oneset/cube/n30/size0.012` | ~14,300 transitions, single physics group |
| [`configs/dataset/genesis_dmdc_corl_cube.yaml`](../configs/dataset/genesis_dmdc_corl_cube.yaml) | `corl/cube` (2006 files) | ~35,000 transitions, wide physics range |

---

## 2. Method notes

### 2.1 Descriptors `φ(occupancy)` and how `D` is distributed

The descriptor vector concatenates six contiguous slices. Let
`nf = n_fourier · (n_fourier//2 + 1)` be the size of the low-frequency 2-D FFT
block. Then:

| Slice | dims | nf=8 | nf=6 | nf=4 |
|---|---|---|---|---|
| `const` | 1 | 1 | 1 | 1 |
| `mass` | 1 | 1 | 1 | 1 |
| `com` | 2 | 2 | 2 | 2 |
| `moments2` | 3 | 3 | 3 | 3 |
| `dft_real` | nf | 40 | 24 | 12 |
| `dft_imag` | nf | 40 | 24 | 12 |
| **`D` = 7 + 2·nf** | | **87** | **55** | **31** |

Only **7 dims are the interpretable low-order geometry** (`const, mass, com,
moments2`); the DFT block (`2·nf`) dominates `D`. This is why lowering
`n_fourier` both improves conditioning (smaller `D`) *and* drops the least
linearly-predictable high frequencies — the two effects that sharpened every
ratio in §3.3–3.4.

**Grid convention:** `dim0 = world_y` (rows), `dim1 = world_x` (cols); pixel
coordinates are normalized to `[0,1]`. What each slice is:

- **`const` (1)** — the literal constant `1`. It makes each operator **affine,
  not purely linear**: because `φ[0] ≡ 1`, the matching column of `A_b` acts as
  an additive bias, so an operator can (e.g.) shift the COM by a fixed offset
  independent of the current COM. This is the EDMD "include the constant
  observable" convention, required for structure *creation* from a spread state.
- **`mass` (1)** — `Σ occ / (H·W)`, the occupied fraction of the grid (proxy for
  total material). Near-conserved under a single push, so persistence is a very
  strong baseline for it.
- **`com` (2)** — the mass-weighted centroid `(com_y, com_x)` in `[0,1]`: where
  the pile sits. `com_y = Σ(occ·y)/Σocc`, likewise `com_x`.
- **`moments2` (3)** — the second **central** moments about the COM:
  `(μ_yy, μ_xx, μ_xy)` = the entries of the occupancy distribution's `2×2`
  covariance (`μ_yy = Σ occ·(y−com_y)² / Σocc`, etc.). They encode the pile's
  spatial **spread and orientation** — how elongated and how tilted the blob is.
- **`dft_real` / `dft_imag` (`nf` each)** — real and imaginary parts of the
  **low-frequency block of the 2-D real FFT** (`torch.fft.rfft2`,
  `norm="forward"`): the lowest `n_fourier` vertical × `n_fourier//2+1`
  horizontal frequencies, flattened. Low frequencies summarize coarse blob
  shape/layout; the high frequencies (fine edges/texture) are deliberately
  excluded. Two subtleties: `norm="forward"` keeps coefficients O(mean
  occupancy) so no pre-scaling is needed before the fit; and the DC coefficient
  `F[0,0].real` **equals** `mass/(H·W)` — a deliberate redundancy that has a
  consequence for the fit (see §2.3).

### 2.2 Action binning
Coarse start-cell grid (`n_start_bins²` position cells) × push-angle bins
(`n_angle_bins`), giving `n_bins = n_start_bins² · n_angle_bins`. One linear
operator `A_b` is fit per bin. Actions are the raw world-frame push
`[sx, sy, ex, ey]`; the start cell comes from `(sx, sy)`, the angle from
`atan2(ey−sy, ex−sx)`.

### 2.3 The per-action operators `A_b`: constraints and the ridge fit
Each `A_b` is an **unconstrained** `D×D` real matrix — one per bin, shape
`[n_bins, D, D]`. **No** orthogonality, stability, symmetry, or non-negativity
is imposed. The only structure is:

- the **affine** behaviour induced by the `const` observable (§2.1);
- **ridge regularization** toward a target `A₀` (`0` by default, or the source
  operators in transfer mode), which bounds `‖A‖`;
- **empty / unseen bins** fall back to the **identity** (⇒ persistence) or, in
  transfer mode, to the prior operator;
- fits are **per-bin independent** and solved in float64 (stored float32).

Ridge with `λ > 0` is **required**, not just stabilising: because the DC Fourier
term duplicates `mass` (§2.1), the descriptors are linearly dependent, so the
normal-equation matrix `X Xᵀ` is singular and needs the `λI` term to be
invertible.

Per bin `b`, let the columns of `X ∈ ℝ^{D×n_b}` be the current descriptors of the
`n_b` transitions in the bin and `Y` the corresponding next descriptors:

```
A_b = argmin_A  ‖A X − Y‖²_F  +  λ ‖A − A₀‖²_F

Setting the gradient to zero:   (A X − Y) Xᵀ + λ (A − A₀) = 0
                          ⇒   A_b = (Y Xᵀ + λ A₀) (X Xᵀ + λ I)⁻¹

With the default A₀ = 0:            A_b = Y Xᵀ (X Xᵀ + λ I)⁻¹
```

Interpretation: `X Xᵀ` is the descriptor auto-covariance (Gram matrix) within the
bin, `Y Xᵀ` the current→next cross-covariance; `A_b` = cross-cov × (auto-cov +
λI)⁻¹, the standard DMDc/least-squares operator, ridge-stabilized. The code
solves the equivalent transposed system `(X Xᵀ + λ I) Aᵀ = X Yᵀ + λ A₀ᵀ` (valid
because `X Xᵀ + λ I` is symmetric). The same `λ` weights both the shrink term and
the `I`, so the transfer prior strength is set by `--transfer-weight`.

### 2.4 Metric & reliability
- **Metric**: per descriptor group, **ratio = model MSE / persistence MSE** on
  held-out data. Persistence = predicting `φ_{t+1} = φ_t`. **Ratio < 1 ⇒ the
  linear map beats persistence** (has signal).
- **Reliability rule of thumb**: want samples/bin ≫ `D`; below `D` the per-bin
  operator is underdetermined (ridge regularises but the estimate is
  low-confidence — the source of the >1 ratios in §3.2 and the 72-bin run in §3.4).

---

## 3. Results

### 3.1 The data is single-push transitions, not trajectories
Across all datasets the per-run samples are **independent** single-push transitions,
not contiguous rollouts (`states_[i] ≠ states[i+1]`). The contiguity diagnostic
(≈0 ⇒ trajectories, ≈1 ⇒ independent) confirms this and the multi-step report is
skipped everywhere:

| Dataset | contiguity ratio |
|---|---|
| oneset | 3.5 |
| corl/cube | 12.3 |

One-step prediction is therefore *the* falsification test here.

### 3.2 One-step falsification — small data is inconclusive
On the tiny n50 smoke set (~140 transitions, ~27 samples/bin ≪ D), every operator
is underdetermined and the fit is **worse than persistence** across the board —
a data-starvation artifact, not a verdict:

| Group | n50, single global operator (nf=4, 118 train) |
|---|---|
| mass | 2.176 |
| com | 3.120 |
| moments2 | 3.803 |
| dft_real | 2.399 |
| dft_imag | 2.857 |

*(The unit test `test_fit_recovers_linear_map` confirms the estimator recovers a
true linear map exactly — so these >1 ratios are underdetermination, not a code bug.)*

### 3.3 One-step falsification — with enough data, linear beats persistence
On **oneset** (single physics, ~11.4k train, 32 bins, well-conditioned), the verdict
flips. Restricting the Fourier block (`nf=8 → nf=6`, i.e. dropping the least
linearly-predictable high frequencies and improving conditioning) strengthens
every beat:

| Group | oneset nf=8 (D=87) | oneset nf=6 (D=55) |
|---|---|---|
| mass | 1.056 | **0.950** |
| com | **0.582** | **0.533** |
| moments2 | **0.909** | **0.790** |
| dft_real | **0.973** | **0.903** |
| dft_imag | 0.994 | **0.871** |

`< 1` in **bold**. At `nf=6` all five groups beat persistence.

### 3.4 Full dataset (corl/cube) — physics generalization
`corl/cube` has 2006 files, **every file a distinct physics group** (friction
0.05–0.5, density 750–5000), so the registry split puts **disjoint physics** in
train vs test — a harder test of a *physics-agnostic* linear map. Bin conditioning
matters: 72 bins left 26 bins underdetermined (median 126/bin); 32 bins were clean
(median 389/bin) and sharpened every beat:

| Group | 72 bins (26/72 underdetermined) | 32 bins (clean) |
|---|---|---|
| mass | 1.108 | 1.019 |
| com | **0.602** | **0.608** |
| moments2 | **0.944** | **0.824** |
| dft_real | **0.948** | **0.895** |
| dft_imag | **0.930** | **0.854** |

**Cross-dataset summary (nf=6, 32 bins, clean fits):**

| Group | oneset (single physics) | corl/cube (disjoint physics) |
|---|---|---|
| mass | 0.950 | 1.019 |
| com | **0.533** | **0.608** |
| moments2 | **0.790** | **0.824** |
| dft_real | **0.903** | **0.895** |
| dft_imag | **0.871** | **0.854** |

Marginalising over a wide physics range costs a little on every metric (as
expected — a physics-blind operator cannot know friction/density), but the
qualitative result holds: **COM and low-order shape are linearly predictable per
action, even on unseen physics; mass is a wash** (near-conserved under a push, so
persistence is very hard to beat and the residual variance is physics-dependent).

### 3.5 Transfer learning (corl → oneset)
Implemented as a **ridge prior**: shrink each oneset operator toward the
corl-fitted operator. Three regimes on the *same* oneset held-out test — `in-domain`
(oneset only), `zero-shot` (corl operators applied cold), `transfer` (oneset fit
with a corl prior, weight `1e-2`).

**Full target data (11.4k oneset transitions):**

| Group | in-domain | zero-shot | transfer |
|---|---|---|---|
| mass | 0.940 | 0.907 | **0.860** |
| com | 0.519 | 0.650 | **0.502** |
| moments2 | 0.754 | 0.877 | **0.709** |
| dft_real | 0.887 | 0.885 | **0.825** |
| dft_imag | 0.866 | 0.858 | **0.808** |

→ Transfer improves **all 5 groups by 3–8%** even when the target is data-rich.

**Scarce target data (300 oneset transitions, ~9/bin ≪ D=55):**

| Group | in-domain | zero-shot | transfer |
|---|---|---|---|
| mass | 2.972 | 0.919 | 0.972 |
| com | 1.444 | 0.649 | **0.600** |
| moments2 | 2.352 | 0.885 | **0.835** |
| dft_real | 2.362 | 0.887 | 0.949 |
| dft_imag | 2.277 | 0.848 | 0.925 |

→ In-domain **collapses** (overfits underdetermined bins, worse than doing
nothing); transfer stays near zero-shot and still wins on COM.

**Mechanism:** the corl prior is **variance reduction (shrinkage)**, not
data-borrowing. corl's physics range *encompasses* oneset's regime, so the prior
is informative but biased; blending it with the noisy target fit yields lower MSE
than either alone (note zero-shot mass 0.907 already beats in-domain mass 0.940 —
oneset's own mass operator is mostly fitting noise). The benefit is small when data
is abundant and large when data is scarce — the textbook transfer regime.

---

## 4. Conclusions

1. **The latent-linear/Koopman hypothesis clears its bar.** With adequate data and
   well-conditioned bins, per-action linear maps over analytic descriptors beat
   persistence on COM, spatial spread (moments), and low-frequency shape (DFT) —
   including generalization to unseen physics.
2. **Concrete numbers a learned model must beat** (corl/cube, unseen physics):
   COM **0.61**, moments **0.82**, low-freq DFT **~0.87**.
3. **Where linear breaks down / a learned model should add value:** (a) **mass**
   (physics-dependent, likely needs physics conditioning), and (b) **high-frequency
   shape detail** (dropped at nf=6, where linearity fails).
4. **Transfer learning helps** — modestly at full data, dramatically under scarcity —
   via shrinkage. Practical guidance: for a new physics regime with limited data,
   fit operators on corl and use them as a prior rather than fitting cold.

**Caveats:** transfer weight was fixed at `1e-2` (not tuned — direction is robust,
magnitude will shift with the weight); all results are one-step (the data has no
trajectories to test multi-step); by-file splits use an unseeded random holdout, so
absolute numbers vary by ~0.01–0.02 between runs (comparisons within a single run
are exact).

---

## 5. Reproduction

```bash
# Single-physics (oneset), well-conditioned:
python3 dmdc_baseline.py configs/dataset/genesis_dmdc_oneset.yaml \
    --split by-file --n-fourier 6 --n-start-bins 2 --n-angle-bins 8

# Full corl/cube (disjoint-physics test):
python3 dmdc_baseline.py configs/dataset/genesis_dmdc_corl_cube.yaml \
    --n-fourier 6 --n-start-bins 2 --n-angle-bins 8

# Transfer corl -> oneset (add --target-max-train 300 for the scarcity regime):
python3 dmdc_baseline.py configs/dataset/genesis_dmdc_oneset.yaml --split by-file \
    --n-fourier 6 --n-start-bins 2 --n-angle-bins 8 \
    --transfer-from configs/dataset/genesis_dmdc_corl_cube.yaml --transfer-weight 1e-2
```

## 6. Files added / changed

- `dmdc_baseline.py` — data bridge, split modes, contiguity guard, ridge-prior transfer
- `Genesis/training/dataset.py` — `workspace_bounds`, `get_raw_action`, `get_run_index`
- `configs/dataset/genesis_dmdc_{cube_n50,oneset,corl_cube}.yaml`
- `tests/test_dmdc_baseline.py` — 9 unit tests (suite total 41 passed)
