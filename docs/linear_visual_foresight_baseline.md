# Linear Visual Foresight (Suh & Tedrake 2020) as a Comparison Baseline

**Status:** PLAN — nothing implemented yet. Intended to be executed on a **new
fork/branch** off the current work (the `GenesisWorld` line is busy with the
plate/settling investigation).
**Source paper:** [`2002.09093v3.pdf`](2002.09093v3.pdf) — H.J.T. Suh & R. Tedrake,
*The Surprising Effectiveness of Linear Models for Visual Foresight in Object Pile
Manipulation*, arXiv:2002.09093v3 (16 Jun 2020).
**Prior partial exploitation in this repo:** descriptor-level switched-linear DMDc
([`dmdc_baseline.py`](../dmdc_baseline.py), [`reports/dmdc_baseline_report.md`](../reports/dmdc_baseline_report.md),
ideas I1–I4 in [`ideas_log.md`](ideas_log.md)). **This plan is different:** it
reproduces the paper's *actual* pipeline — pixel-space operators, the affine warp,
and their Lyapunov greedy controller — as a runnable end-to-end baseline, rather
than mining it for ideas at descriptor level.

---

## 1. The paper, summarized

### 1.1 Problem setup (§2.1)

- Top-down greyscale image of the workspace, `I_k ∈ R^{N×N}` (they use `N = 32`),
  vectorized as `y_k ∈ R^{N²}`. Obtained by colour thresholding / background
  subtraction, so it is effectively a **2-D occupancy image of the pile**.
- Action `u ∈ R⁴`: push start `p_i ∈ R²`, push orientation `θ`, push length `l`.
  **Assumption: the push surface is always perpendicular to the push direction**
  (so `u` is fully determined by start point + end point).
- Target: learn the one-step image dynamics `I_{k+1} = f̄(I_k, u)`. Writing it
  one-step-Markov in the image encodes the **quasi-static** assumption (friction
  dominates; no momentum carried between pushes).

### 1.2 The controller: an image-space Lyapunov function + greedy 1-step search (§2.2)

This is the part that makes the whole thing a *closed-loop* baseline rather than a
prediction benchmark, and it is strikingly cheap:

1. Precompute, **once per goal**, a distance matrix `D ∈ R^{N×N}` where
   `D[i,j] = min_{p ∈ S_d} ‖center(pixel i,j) − p‖_p` — i.e. the Euclidean (or
   `p`-norm; `p` is a hyperparameter) distance transform of the target set `S_d`.
   Vectorize to `d`.
2. Lyapunov function = mass-normalized weighted average distance:

   ```
   V̄(I_k) = (1 / ‖I_k‖_{1,1}) · Σ (D ⊙ I_k)        V(y_k) = dᵀ y_k / ‖y_k‖_1
   ```

   `V̄ = 0` ⟺ all non-zero pixels sit where `D = 0` ⟺ all material inside the goal.
   This is the average Chamfer distance from the pile to the target set. Intuition
   (their Fig. 2): the closed loop deforms the flat cutting board into a **bowl**
   whose flat bottom is the target set, and the carrots roll in.
3. It is a **Control Lyapunov Function**: for any image outside the goal set there
   exists some push that decreases it (you can always find one small piece to nudge
   goalward). So the greedy 1-step policy suffices — no horizon, no trajectory
   optimization:

   ```
   u* = argmin_u  V̄( f̄(I_k, u) )
   ```
4. `u` is **discretized on a fixed grid** and the argmin is done by direct
   enumeration. Their stated reason is important and matches our experience:
   planar pushing is severely non-convex and *most actions miss the pile entirely
   and produce zero gradient*. Enumeration also makes the model comparison fair —
   every model is scored over the identical action set.
5. Non-convexity note: for a non-convex `S_d`, `V(X)` over particle positions is
   non-convex, but `V(y)` is **linear (hence convex) in the image** as long as
   `‖I_k‖_1` is roughly constant — a convex-relaxation-in-the-space-of-measures
   effect. Empirically greedy descent stabilizes even non-convex targets (their
   Fig. 12).

### 1.3 Model A — the switched-linear operator (§3.1, the paper's contribution)

```
y_{k+1} = A_i y_k ,   A_i ∈ R^{N²×N²},   i ∈ {1 … |U|}
```

The action *selects the matrix*; there is no `u` input to the map. Because object
position lives in *pixel indices* rather than pixel values, `A_i` acts as a
(soft) **permutation/transport matrix on indices**.

**Fit (their eq. 8–10).** Collect `M` pairs subject to the same action, stack into
`Y_k, Y_{k+1} ∈ R^{N²×M}`, solve matrix OLS `A* = argmin ‖Y_{k+1} − A Y_k‖_F`
(closed form). They then test constrained variants as regularization:

| Variant | Result (their Fig. 7) |
|---|---|
| plain OLS | baseline |
| **non-negativity `A ≥ 0`** | **best — adopted as default** |
| row-sums = 1 | over-regularized, worse |
| column-sums = 1 (Markov stochastic) | loses row decomposition, too big for a QP solver, deferred |

The tractability trick: `Y_{k+1}[i,:] = A[i,:] Y_k`, so the `N⁴`-variable QP
**decomposes into `N²` independent `N²`-variable QPs**, one per row — each row's
optimum is the global optimum for that row. (Column constraints break this.)

**Interpretation (§3.1, tensor view).** Reshape `A ∈ R^{N²×N²}` into
`Ā ∈ R^{N×N×N×N}`; then `Ā[i,j,:,:]` is the **kernel image of output pixel (i,j)**,
element-wise multiplied with `I_k` and summed. Their Fig. 8 shows the learned
structure: **identity outside the push rectangle**, ≈zero inside it (material
leaves), and at the leading edge a kernel that gathers the swept mass and deposits
it ahead. The step response of `A` on a uniform `0.5` image recovers the deposit
distribution — learned purely from data, and near-identical to their hand-written
transport model's distribution.

### 1.4 The affine warp — how the action space collapses (§3.1, their Fig. 4)

Instead of one `A` per `(p_i, θ, l)` cell, exploit that the action lives in a
Cartesian frame:

1. From `u`, construct the **push rectangle** (area swept by the pusher).
2. Compute `T ∈ SE(2)`: image-center frame → push-rectangle frame.
3. Predict in the canonical frame: `Î_{k+1} = T⁻¹( A_l · T(I_k) )`.
4. Warping loses image content at the borders, so also warp a ones-image to get a
   validity mask `M = T⁻¹(T(1_{N×N}))` and use it to **blend the prediction with
   the original image** (prediction where valid, `I_k` elsewhere).

Result: **one `A` per push-length bin only**. They use **5 length bins, 1000
`(I_k, I_{k+1})` pairs each** (~5000 samples total, 32×32 → ~1.05M parameters per
operator).

### 1.5 The two baselines they compare against

- **Deep visual foresight (§3.2).** Conv autoencoder → MLP latent dynamics →
  decoder, with skip connections. Two variants: **DVF-Original** (action appended
  to the latent, `R^{N×N} × R⁴ → R^{N×N}`, 2.38M params, trained on 23 000 tuples)
  and **DVF-Affine** (same warp trick as the linear model, so it is pure
  image→image; 5 separate networks, one per length bin, mixture-of-experts style,
  trained on the *same 5000 samples* as the linear model). Loss = Frobenius error;
  Adam, >1000 epochs, step LR from 0.01, γ=0.1 every 100 epochs.
- **Object-centric transport model (§3.3).** First-principles, near-zero-fit:
  treat every non-zero pixel as a particle; split into affected `X_a` (inside the
  push rectangle) and unaffected `X_u`; resample `|X_a|` coordinates from a
  bivariate distribution `P(u)` — approximated as a **small uniform distribution
  immediately in front of the push rectangle** — and evaluate the *particle*
  Lyapunov function directly (it never synthesizes an image). They validated the
  uniform-deposit choice by warping real transitions and thresholding
  `I_{k+1} − I_k` (their Fig. 5, right).

### 1.6 Results worth reproducing (§4–5)

| Model | Dim | Params | Samples | One-step test error `‖I_{k+1} − f(I_k,u)‖_F` |
|---|---|---|---|---|
| Switched-Linear | `R^{N×N} → R^{N×N}` | 1 048 576 | 5 000 | **1.858** |
| DVF-Original | `R^{N×N} × R⁴ → R^{N×N}` | 2 382 721 | 23 000 | 2.062 |
| DVF-Affine | `R^{N×N} → R^{N×N}` | 2 317 185 | 5 000 | 2.537 |

- **Closed loop (Fig. 11):** linear, transport, and DVF-Affine all converge;
  **DVF-Original fails** — its argmin action often changes nothing in the real
  scene, so the same image is fed back and the loop is stuck. (Same failure mode
  reported by Wilson & Hermans for spatial-autoencoder latents.) Deep models were
  observed to *make carrots disappear* instead of pushing them.
- Advantage grows with pile size: with many pieces, linear descends visibly
  faster; DVF-Affine typically needs ~2 more iterations.
- **Compute per MPC step:** switched-linear **1.0 s CPU**, DVF-Affine 4.0 s CPU,
  object-centric **27.0 s CPU**, DVF-Original 0.17 s GPU.
- **Sim→real (Fig. 14):** operators fitted in a Pymunk sim, applied to real diced
  carrots with **no fine-tuning**. Linear and object-centric still converge;
  **DVF-Affine fails** — it had overfit the simulator's images. This is the
  paper's strongest claim: the linear model's inductive bias (mostly: identity
  outside the swept rectangle) is what generalizes.
- Their headline caveat, stated plainly: a linear model *is* a subclass of the
  deep model, so a better deep model must exist — but it is not findable with the
  data/architecture search budget, whereas the linear fit is globally optimal in
  its parameter space and cleanly constrainable.

**Why this is a good baseline for us:** it is a genuinely strong, ~zero-training,
seconds-to-fit method on exactly our task (push a pile of small objects into a
target set from a top-down occupancy image), with a published closed-loop metric
that our MPC already collects the ingredients for.

---

## 2. What already exists in this repo

Substantially more than half the machinery is already here. Ordered by how directly
it is reusable.

### 2.1 Directly reusable, no changes needed

| Paper component | Where it already lives |
|---|---|
| Occupancy image `I_k` (their thresholded greyscale) | [`transforms/functional.py:51`](../transforms/functional.py#L51) `particles_to_occupancy`; grids are `[64, 64]` by config ([`simple_mpc/config/config_oracle.yaml:92`](../simple_mpc/config/config_oracle.yaml#L92)) |
| Action `u ∈ R⁴` | `[sx, sy, ex, ey]` is the action convention through the whole Eulerian path — [`model/eulerian_wrapper.py:371`](../model/eulerian_wrapper.py#L371) `predict_one_step_occ`, [`simple_mpc/mpc.py`](../simple_mpc/mpc.py). Start+end ⟺ their `(p_i, θ, l)`. |
| Distance transform of the target set (their `D`) | [`simple_mpc/occupancy_reward.py`](../simple_mpc/occupancy_reward.py) `OccupancyReward.compute_score_tensor` already back-projects the pixel-space subgoal to an occupancy goal and runs `scipy.ndimage.distance_transform_edt`. It returns a *shifted reward*, not the raw distance — one small accessor away from `d`. |
| `argmin` over enumerated candidates | [`simple_mpc/mpc.py:271-275`](../simple_mpc/mpc.py#L271-L275) — `run_simple_mpc` already computes `rew_seqs = reward_fn_opt(state_batch)` and tracks the argmax candidate *before* any gradient step. With `n_look_ahead: 1` and `n_update_iter: 1` the loop **is** their greedy 1-step direct search. |
| Occupancy-space model plug-in point | `_PUSH_MODEL_REGISTRY` / `build_push_model` ([`model/eulerian_wrapper.py:1027-1059`](../model/eulerian_wrapper.py#L1027-L1059)) — the checkpoint-free path for models that are not trained by `training/trainer.py`. Signature is exactly `forward(occ (B,Nx,Ny), start_grid (B,3), end_grid (B,3)) -> occ`. |
| Loss/cost registry shared by training and MPC ranking | [`training/losses.py`](../training/losses.py) `register_loss` + the `per_sample=True` reduction (INTERFACES §3.5) |
| Action-space enumeration + plate-collision filtering | [`simple_mpc/action_sampler.py`](../simple_mpc/action_sampler.py) — `ActionSampler` ABC, `make_action_sampler` factory, and `PlateCollisionChecker` (line 176) for rejecting infeasible pushes |
| Local Cartesian-product grid over actions | [`simple_mpc/human_grid_search.py:34`](../simple_mpc/human_grid_search.py#L34) `build_action_grid` — a `grid_n**5` product grid; the *global* version is the paper's `U` |
| Ground-truth transition data `(I_k, u, I_{k+1})` | [`dmdc_baseline.py:309`](../dmdc_baseline.py#L309) `load_transition_arrays` already extracts `occ_t`, `occ_t1`, raw world push, run id via `registry.dataset_registry.build_dataset`; accessors `workspace_bounds` / `get_raw_action` / `get_run_index` live in [`Genesis/training/dataset.py`](../Genesis/training/dataset.py). Datasets: `genesis_dmdc_oneset.yaml` (~14.3k transitions, single physics) and `genesis_dmdc_corl_cube.yaml` (~35k, wide physics). |
| Their deep baseline ≈ our existing model | [`model/NFDUNetFilm.py`](../model/NFDUNetFilm.py) is an action-conditioned (FiLM) image→image occupancy predictor — the structural analogue of **DVF-Original**, and already trainable/evaluable through `training/` + `run_experiments.py` |
| Their object-centric transport model ≈ our heuristics | `_PUSH_MODEL_REGISTRY` entries `splat`, `spread`, `spread2`, `cumulative`, `fluid` ([`model/eulerian_wrapper.py:1062-1140`](../model/eulerian_wrapper.py#L1062-L1140)). `spread`/`cumulative` are richer than their uniform-deposit model; a faithful uniform-deposit variant is a ~40-line addition to the same registry. |
| Sampling-based candidate ranking with a non-differentiable model | [`simple_mpc/sampling_optimizers.py`](../simple_mpc/sampling_optimizers.py) (CEM/MPPI) — not needed for pure enumeration, but the precedent for gradient-free ranking |

### 2.2 Already-established findings that de-risk the plan

From [`reports/dmdc_baseline_report.md`](../reports/dmdc_baseline_report.md) and
[`ideas_log.md`](ideas_log.md) §1, at **descriptor** level (D≈55–87, not pixels):

- Per-action linear maps **beat persistence** on COM (0.61), 2nd moments (0.82),
  and low-frequency DFT (~0.87) — including across unseen physics. So the
  switched-linear hypothesis is already known to hold on our data in the coarse
  regime.
- Where it breaks: **mass** (physics-dependent — leakage/stacking, which their
  2-D Pymunk sim does not have) and **high-frequency shape**.
- **Ridge-prior transfer** (fit on `corl`, shrink `oneset` toward it) helps
  modestly at full data and dramatically under scarcity — directly applicable to
  per-length-bin operators with uneven bin occupancy.
- Idea I4 (adopt their **Lyapunov descent curve** as the primary closed-loop
  metric) is already recorded as *adopt now*; this plan is what makes it runnable.
- Idea I2 (their **non-negative LS beat OLS**) is recorded as a small experiment
  at descriptor level; this plan does it at the level the paper actually did.

### 2.3 Gaps — what does not exist yet

| Missing piece | Notes |
|---|---|
| **SE(2) affine warp into the push-rectangle frame** and its inverse + validity mask | The single biggest gap. Nothing in [`transforms/functional.py`](../transforms/functional.py) warps a grid; the only `warpAffine` in the repo is [`utils.py:328`](../utils.py#L328) (goal-mask scaling, unrelated). Tool-frame canonicalization is an explicit `TODO(canonicalization)` in [`dmdc_baseline.py:122`](../dmdc_baseline.py#L122) and a *planned, unimplemented* item in [`analytic_descriptors_latent_space_plan_v2.md`](analytic_descriptors_latent_space_plan_v2.md) §1. |
| **Pixel-space operator** `A ∈ R^{N²×N²}` (fit, storage, apply) | Only the descriptor-space `D×D` version exists (`fit_per_action_operators`, `dmdc_baseline.py:163`). |
| **Row-decomposed non-negative least squares** | Descriptor fit is ridge/OLS only. Needs `scipy.optimize.nnls` per row (or a batched projected-gradient/OSQP path). |
| **The image Lyapunov cost `V = dᵀy/‖y‖₁`** | `compute_score_tensor` is a *reward* (`occ_goal − dist`, shifted, unnormalized). The paper's `V` is mass-normalized and is a cost with an exact zero. Not in `training/losses.py`, and not in `EulerianAdapter._REWARD_METHODS` (`'default'` / `'iou'` only) — the gradient-MPC path reads its cost from that enum, not from the loss registry. |
| **Global enumerating action sampler** | `RandomUniformSampler`/`PhysicsAwareActionSampler`/`CollisionAwareActionSampler` all *sample*; `build_action_grid` enumerates only *locally* around a center. |
| **A gradient-free path through `run_simple_mpc`** | Achievable purely by config (`n_update_iter: 1`), but nothing asserts it, and the Adam optimizer is constructed unconditionally ([`mpc.py:247`](../simple_mpc/mpc.py#L247)). |
| **Per-length-bin model artifacts + a loader** | `load_model_from_card` is checkpoint/`model_card.yaml`-oriented; a fitted `.npz` of 5 operators has no home. |
| **Faithful uniform-deposit transport model** | Nearest existing is `spread`/`cumulative`; neither is the paper's model. |
| **DVF-Affine variant** | Needs the warp + 5 per-bin NFDUNetFilm checkpoints with the action input removed. |
| **Descent-curve reporting/plotting** | MPC results record `rewards`/`occ_rewards`; no `V` series, and no stuck-loop detector. |

---

## 3. Fidelity: where our setup differs from the paper, and what to do about it

These are recorded in [`ideas_log.md`](ideas_log.md) §1 ("Paper assumptions our data
violates") and matter for claiming the baseline is *identical*.

| Divergence | Impact | Decision for this plan |
|---|---|---|
| **Plate angle is sampled independently of push direction** (median plate-vs-push relative angle 0.93 rad; only ~8% near-perpendicular) — their §2.1 assumes ⊥ | Their action reduction to *push length alone* after the warp does not hold for our full action set | **V1 chosen — see §7 for the collection flag and the full analysis.** **V1 (faithful):** constrain the action set and the fit data to perpendicular pushes — this is the literal reproduction, and note that the existing Eulerian path *already ignores plate angle*, so `predict_one_step_occ(occ, [sx,sy,ex,ey])` needs no change. **V2 (angle-aware):** switch over `(length bin × relative-plate-angle bin)`. V2 needs the plate angle threaded into the push-model signature — a real interface change (touches every heuristic + INTERFACES §3.4), so keep it strictly optional and do V1 first. |
| **Grid is 64×64, theirs is 32×32** | `A` is `64⁴ = 16.8M` floats = 67 MB per operator (×5 bins = 335 MB), and the row-wise NNLS is 4096 problems of 4096 variables | **Fit and predict at 32×32** (downsample the 64×64 occupancy, predict, upsample for reporting) for parameter parity with their Table 1. Expose `foresight_res` as a config knob. Ablation protocol and the cross-resolution metric hazard: **§8**. |
| **Our data is randomly-actioned; theirs is 1000 pairs per length bin by design** | Uneven bin occupancy (report §3.4: 26/72 bins underdetermined in the *unwarped* descriptor fit) | The warp is exactly what fixes this: after canonicalization every push falls into one of ~5 length bins, so ~14k `oneset` transitions give ~2800/bin — **more than their 1000**. Bin-occupancy reporting is still required. Ridge-prior transfer (already implemented for descriptors) is the fallback for thin bins. |
| **3-D granular sim (Genesis) vs 2-D Pymunk**: mass leaves the top-down footprint via stacking, and can leak past the plate | Their `‖I_k‖_1 ≈ const` premise (which makes `V` linear/convex) is weaker for us; the report already flags **mass** as the group where linearity fails | Measure and report `‖I_k‖_1` drift over closed-loop episodes as a first-class diagnostic. It is a finding about our task, not a bug in the baseline. |
| **Walls/fence**: ~20% of pile mass sits within 8 mm of a wall in essentially every frame | Tool-frame invariance is routinely violated (plan-v2 §1) — mass piling against a wall is a long-range effect the warp cannot represent | Report warped-vs-unwarped as an A/B (plan-v2's gate **G2**), and stratify closed-loop results by whether the goal is wall-adjacent. Do **not** silently add a wall term — that stops being their model. |
| **No trajectories in the existing datasets** (single-push transitions only, report §3.1) | Multi-step open-loop prediction cannot be evaluated offline | Irrelevant to this baseline — their controller is 1-step greedy, and the closed loop supplies the multi-step signal through the simulator. |

---

## 4. Integration plan

Placement follows the Design Philosophy in [`ARCHITECTURE.md`](ARCHITECTURE.md):
stateless geometry into `transforms/`, the model into the push-model registry, the
cost into the one loss registry, the enumeration into the sampler factory, nothing
new in the MPC loop.

### Step 1 — the affine warp, in `transforms/functional.py`

The one genuinely new primitive, and shared by the linear model, DVF-Affine, the
transport model, *and* the deferred plan-v2 canonicalization work. Stateless and
`torch`-only, so it belongs here (UTILITIES.md's ownership rule) rather than inside
whichever model needs it first.

```python
def push_rectangle_transform(start_grid, end_grid, grid_res, plate_w_px) -> Tensor  # (B,2,3) SE(2), grid coords
def warp_to_tool_frame(occ, T, out_res=None) -> Tensor                              # differentiable grid_sample
def warp_from_tool_frame(occ_tool, T, out_res) -> Tensor
def tool_frame_validity_mask(T, grid_res) -> Tensor                                 # M = T⁻¹(T(1))
def blend_with_mask(pred, occ_orig, mask) -> Tensor                                 # their Fig. 4 recombination
```

Implementation notes:
- Use `torch.nn.functional.affine_grid` + `grid_sample` (bilinear, `align_corners`
  fixed and documented) so the whole path stays differentiable — that keeps the
  gradient-descent MPC usable on this model for free, even though the paper only
  enumerates.
- Canonical frame convention: origin at the push-rectangle center, `+x` along the
  push direction, extent covering `[−pad, l + deposit_pad]`. Fix and document it
  once; every downstream user reads it from here.
- Grid convention hazard: `dim 0 = world_y` for dataset/grid tensors but the
  `EulerianWrapper` state uses `dim 0 = world_x, dim 1 = −world_y`
  (INTERFACES §4.1). The warp takes **grid coords** (as the push models already
  do) so it sits on the `_cam3d_to_grid` side of that boundary and never has to
  know. Assert the convention in the tests.

### Step 2 — the switched-linear model, `model/linear_foresight.py` (new)

Two concerns, one file, mirroring how `eulerian_wrapper.py` keeps its heuristics
next to their factories:

**(a) `SwitchedLinearForesight(nn.Module)` — inference.**
`forward(occ, start_grid, end_grid)` implementing exactly their Fig. 4:
`downsample → warp → select A by length bin → matvec → unwarp → mask-blend → upsample`.
Registered as `@register_push_model("linear_foresight")` so it reaches MPC through
the existing checkpoint-free path with **zero changes to
`eulerian_wrapper.EulerianModelWrapper`, `simple_mpc/adapters.py`, or
`simple_mpc/mpc.py`** — `run_experiments.py` already dispatches
`heuristic_type` (line 111 / 159–171). The `.npz` operator bundle path becomes one
more `cfg` key (`operator_path`), read by the factory. This is the smallest
possible seam.

**(b) `fit_switched_linear(...)` — the estimator.**
Closed-form matrix OLS by default; `--constraint {none,nonneg,rowsum}` implementing
their eq. (9) via the eq. (10) row decomposition (`scipy.optimize.nnls` per row,
`N²` independent problems — trivially parallel with `joblib`/`multiprocessing`).
Ridge-prior shrinkage toward a source operator set, reusing the semantics already
implemented in `dmdc_baseline.fit_per_action_operators` so thin length bins behave.
Saves `{A_0…A_{L−1}, bin_edges, res, plate_w_px, provenance}` to `.npz`.

**Not** a `registry/model_registry.py` entry and **not** a `training/trainer.py`
client: there is no gradient loop, no epochs, no `ModelOutput`. Forcing it into the
training wrapper contract would be the wrong abstraction (same reasoning that keeps
the geometric heuristics out of it — see the comment at
[`eulerian_wrapper.py:1015`](../model/eulerian_wrapper.py#L1015)).

### Step 3 — the Lyapunov cost, in two places (one definition)

There is a wrinkle worth stating up front: **the two MPC variants get their cost
from different places.** `simple_mpc/oracle_mpc.py` consumes the loss registry
directly (`per_sample=True`, INTERFACES §3.5), but `simple_mpc/mpc.py` gets its
cost from the adapter's own reward enum —
`EulerianAdapter._REWARD_METHODS` ([`adapters.py:130-133`](../simple_mpc/adapters.py#L130-L133),
currently just `'default'` and `'iou'`), selected by `mpc.reward.opt_type`
([`adapters.py:649-655`](../simple_mpc/adapters.py#L649-L655)). So the cost has to
be reachable from both, without being written twice.

**(a) The definition — `training/losses.py`:**

```python
@register_loss("image_lyapunov")
class ImageLyapunovLoss(LossFn):   # V = dᵀ y / ‖y‖₁ ,  supports per_sample=True
```

Config: `p_norm` (their `p` hyperparameter), `eps` for the mass normalization,
`mass_floor` behaviour when the grid is empty.

**(b) The adapter hook — `simple_mpc/adapters.py`:** add
`'lyapunov': ('_reward_lyapunov', True, True)` to `EulerianAdapter._REWARD_METHODS`,
with `_reward_lyapunov` a thin **delegation to the registered loss** (negated —
the adapter maximizes reward while `V` is a cost) rather than a reimplementation.
It needs the distance grid `d` cached in `__init__` next to the existing
`score_tensor`, so also add one accessor beside the existing distance-transform
call:

```python
# simple_mpc/occupancy_reward.py
def distance_to_goal_grid(self, subgoal, p: float = 2.0) -> torch.Tensor:   # the paper's D
```

Keeping the *definition* in the loss registry is the point (Design Philosophy:
"one loss registry serves both training and MPC cost") — it is then available to
`oracle_mpc`, `human_mpc`, `run_experiments.py` reporting, and as a training loss,
from one definition, with the adapter entry as a 3-line bridge. It also makes the
paper's descent curves directly comparable **across all three existing MPC
variants**, which is arguably worth more than the baseline itself.

### Step 4 — the enumerating sampler, in `simple_mpc/action_sampler.py`

```python
class ActionGridSampler(ActionSampler):
    """The paper's fixed discretization of U: n_start_x × n_start_y × n_theta × n_length."""
```

- Returns the **same** deterministic candidate set on every call — that is what
  makes the model comparison fair (their §2.2).
- `n_sample` from the config is honoured as a cap; if the grid exceeds it, fail
  loudly with the required `n_sample` rather than silently subsampling (the
  "no silent caps" habit).
- Reuse `PlateCollisionChecker` to drop candidates whose plate start pose is
  infeasible, and report how many were dropped.
- Register in `make_action_sampler` as `'action_grid'`.

### Step 5 — the controller: config only

No new MPC module. A new config `simple_mpc/config/config_linear_foresight.yaml`:

```yaml
mpc:
  n_mpc: 20            # their closed-loop episodes run to convergence
  n_look_ahead: 1      # greedy, per their CLF argument
  n_update_iter: 1     # ⇒ pure enumeration; the Adam step never runs
  n_sample: <grid size>
  action_sampler: action_grid
  reward:
    opt_type: lyapunov      # EulerianAdapter reward entry added in Step 3b
    report_type: lyapunov
```

One small guard in [`simple_mpc/mpc.py`](../simple_mpc/mpc.py): skip constructing
the Adam optimizer / the `requires_grad` machinery when `n_update_iter == 1`, and
log "enumeration mode". A few lines, no behaviour change for existing configs —
without it, "identical to theirs" rests on an undocumented config coincidence.

### Step 6 — the fit driver, `fit_linear_foresight.py` (repo root, next to `dmdc_baseline.py`)

CLI mirroring `dmdc_baseline.py`'s: dataset config, `--split {auto,registry,by-file}`,
`--res`, `--n-length-bins`, `--constraint`, `--transfer-from/--transfer-weight`,
`--out`. Reports per-bin sample counts, per-bin condition numbers, and their
Table-1 metric (`‖I_{k+1} − f(I_k,u)‖_F`, held-out mean) against persistence and
against the `spread`/`cumulative` heuristics.

**Shared data bridge.** `TransitionArrays` + `load_transition_arrays` +
`split_by_episode` currently live inside `dmdc_baseline.py`. Both drivers need
them, and copying is what the Design Philosophy explicitly names as the signal to
extract: move them to **`Genesis/training/transition_arrays.py`** (they are
dataset-registry-coupled, so they belong next to `Genesis/training/dataset.py`) and
re-import from `dmdc_baseline.py`. Small, mechanical, and covered by the existing
`tests/test_dmdc_baseline.py`.

### Step 7 — the paper's own baselines (for the full comparison table)

- **`@register_push_model("uniform_transport")`** — their §3.3 model: mask the
  push rectangle in the tool frame, count affected mass, redeposit it uniformly in
  a band immediately ahead of the rectangle, unwarp. ~40 lines given Step 1, and it
  reuses the same warp. Also reproduce their Fig. 5 validation (warp real
  transitions, threshold `I_{k+1} − I_k`, compare to the assumed uniform band) —
  cheap, and it tells us whether the assumption survives 3-D granular physics.
- **DVF-Original** ≈ existing `NFDUNetFilm` runs; no new code, just an entry in
  the comparison table using the same held-out split and the same `V` metric.
- **DVF-Affine** — `model/dvf_affine.py`: a thin wrapper applying Step 1's warp
  around a per-length-bin `NFDUNetFilm` with the action input dropped, plus 5
  training configs. **Lowest priority** — it is the variant the paper reports as
  failing sim→real, so it is the least informative for us. Defer until the
  linear-vs-transport-vs-existing-deep comparison is in hand.

### Step 8 — evaluation

Adopt idea **I4** as the protocol:

1. **Descent curves.** Record `V` at every MPC step for every variant; plot
   `V` vs iteration with per-variant spread over ≥10 initial conditions (their
   Fig. 11). Primary scalar: **area under the descent curve**; secondary: steps to
   `V ≤ V_threshold`.
2. **Stuck-loop check** (their DVF-Original failure mode): fraction of steps where
   the executed action produced `‖I_{k+1} − I_k‖_1 < ε`. This is a *model-quality*
   signal and applies to our learned models too.
3. **One-step table** mirroring their Table 1: model, dimension, parameter count,
   training samples, held-out Frobenius error — plus a persistence column, which
   they omit and which the DMDc report established as the honest floor.
4. **Predicted-vs-realized gap.** `run_simple_mpc` already prints
   `predicted r` vs realized (`mpc.py:423`); aggregate it per variant.
5. **Mass-drift diagnostic** (`‖I_k‖_1` over the episode), per §3 above.
6. **Timing**, to compare against their 1.0 s / 4.0 s / 27.0 s / 0.17 s figures.
7. **Sim→real, if the fixtures allow it.** [`RealData/`](../RealData) and
   `configs/dataset/real_chickpeas.yaml` exist; their most interesting result is
   sim-fitted-operators-applied-to-real-with-no-fine-tuning. Scope this as a
   stretch goal after the sim comparison closes.

Output: `reports/linear_foresight_report.md`, following the shape of
`reports/dmdc_baseline_report.md` (what was done / method notes / results /
conclusions / reproduction commands / files changed).

### Step 9 — tests

New `tests/test_linear_foresight.py`, Genesis-free (so it runs in the fast suite):

- Warp round-trip: `warp_from_tool_frame(warp_to_tool_frame(occ, T), T) ≈ occ`
  inside the validity mask, for random `T`.
- Mask blend is exactly the identity when `A = I`.
- Grid-convention assertion: a push along `+world_x` maps to `+x` in the tool
  frame at the documented resolution.
- OLS recovery: synthesize `Y_{k+1} = A_true Y_k` with a random permutation
  `A_true`, check the fit recovers it (mirrors the existing DMDc closed-form test).
- NNLS respects `A ≥ 0` and its objective is ≥ the unconstrained OLS objective.
- `image_lyapunov` is `0` iff all mass is inside the goal; is invariant to a
  uniform scaling of `I`; and its `per_sample=True` shape is `(B,)`.
- `ActionGridSampler` returns a deterministic, correctly-sized, in-bounds set and
  raises (not truncates) when the grid exceeds `n_sample`.

### Step 10 — documentation obligations (same change, per the project doc policy)

- **New** `docs/linear_foresight_design.md` — the subsystem design doc (purpose,
  architecture with *why*, file map, config reference, usage, known limitations),
  following `oracle_mpc_design.md`'s shape. This planning doc is superseded by it.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) module map — `model/linear_foresight.py`,
  `fit_linear_foresight.py`, `Genesis/training/transition_arrays.py`; link the new
  design doc.
- [`INTERFACES.md`](INTERFACES.md) — the `image_lyapunov` loss (§3.5 neighbourhood),
  the new `'lyapunov'` entry in the §3.4 adapter reward enum, and, **only if V2 is
  pursued**, the push-model signature change (§3.4).
- [`UTILITIES.md`](UTILITIES.md) — the new `transforms/functional.py` warp helpers
  and their frame convention.
- `.claude/skills/project-overview/SKILL.md` doc map — one row for the new design doc.
- [`ideas_log.md`](ideas_log.md) — mark I2/I4 as executed at pixel level, and cross-link.

---

## 5. Suggested ordering

Each stage ends at something runnable and reportable, so the baseline can be
abandoned early if the numbers say to.

| # | Stage | Deliverable | Rough size |
|---|---|---|---|
| 0 | §7.2–7.3 (`--perpendicular-pushes`) + the oracle/human relative-angle probe (§7.5) | v1 collection flag, recorded in dataset metadata; evidence on whether the 5th DOF matters | ~0.5 day |
| 1 | Steps 1 + 9 (warp + its tests) | `transforms/functional.py` warp helpers, green fast suite | ~0.5 day |
| 2 | Step 6's extraction + Step 2b (fit) + Table-1 numbers | `fit_linear_foresight.py` produces operators; one-step held-out error vs persistence vs heuristics | ~1 day |
| 3 | Steps 2a + 3a/3b + 4 + 5 (closed loop) | `heuristic_type: linear_foresight` runs end-to-end under `run_experiments.py`; first descent curve | ~1 day |
| 4 | Steps 7 (uniform_transport) + 8 | Full comparison table + `reports/linear_foresight_report.md` | ~1 day |
| 5 | Step 10 | Docs landed | ~0.5 day |
| 6 | *Optional* | V2 angle-aware operators; DVF-Affine; sim→real | open-ended |

Gate after stage 2 — see **§10** for what it decides and why it sits there.

Note that stage 0 needs **no new data collection to be useful**: the relative-angle
probe runs on existing oracle-MPC and human-demo logs, and the fit can proceed on
the existing `oneset`/`corl_cube` datasets by *filtering* to near-perpendicular
pushes. Collecting a purpose-built v1 dataset is only necessary if that filter
leaves too few transitions per length bin (~8% of pushes are near-perpendicular
per plan-v2 §1, so 14.3k `oneset` transitions leave ~1 100 — right at the paper's
per-bin budget in total, i.e. **too thin**, which is the argument for collecting
with the flag on rather than filtering).

---

## 6. Decisions taken

1. **V1 (perpendicular-only) is the default.** See §7 for the collection-flag
   design and the v1→v2 analysis.
2. **Fit and predict at 32×32**, with a resolution ablation as part of the
   evaluation rather than an afterthought — see §8.
3. **Fit on `oneset` first, `corl_cube` as the generalization test** (mirrors the
   DMDc report's structure).
4. **The Lyapunov axis is the deliverable, not just the baseline** — see §9.

---

## 7. The perpendicular-push constraint

### 7.1 Where the action is actually sampled

One function, one place:
[`Genesis/sandbox_manipulation_clean.py:1576`](../Genesis/sandbox_manipulation_clean.py#L1576)
`generate_action_samples`. Today it draws three things **independently**:

```python
angles       = (-π/2) + rand(n_total) * π                    # plate yaw
start_samples = (high - low) * rand(n_total, 2) + low        # start point
stop_samples  = (high - low) * rand(n_total, 2) + low        # end point → direction AND length
```

so the push direction is `stop − start`, unrelated to the plate yaw — exactly the
0.93 rad median relative angle measured in plan-v2 §1. Note also that
`sample_space_x/y` shrinks the sampling box by `cos(θ)·tool_length/2 +
|sin(θ)|·tool_width/2`, which tells us the convention: **the plate's long axis is
`(cos θ, sin θ)`**, so its face normal — the only direction a perpendicular push
can travel — is `(−sin θ, cos θ)`, up to sign.

### 7.2 The minimal change (~8 lines + flag threading)

Add one kwarg to `generate_action_samples` and rewrite only the *direction* of the
push, keeping everything else — including the existing push-length distribution —
untouched:

```python
def generate_action_samples(self, n_samples, ..., perpendicular_pushes: bool = False):
    ...
    # (existing angles / start_samples / stop_samples draws unchanged)

    if perpendicular_pushes:
        # Plate long axis is (cos θ, sin θ) — see how sample_space_x is built
        # above — so a push normal to the blade face travels along
        # ±(−sin θ, cos θ). Reuse the already-drawn stop only for its DISTANCE,
        # so the push-length marginal is exactly what it was before; the ray-box
        # truncation in equalize_travel_distance keeps the push inside its box.
        sign  = torch.where(torch.rand(n_total, device=gs.device) < 0.5, -1.0, 1.0)
        n_hat = sign.unsqueeze(-1) * torch.stack(
            [-torch.sin(angles), torch.cos(angles)], dim=1)
        target = (stop_samples - start_samples).norm(dim=-1, keepdim=True)
        stop_samples, _ = equalize_travel_distance(
            start_samples, start_samples + n_hat, low, high, target)
```

Why this shape rather than sampling a length directly:

- It **reuses `equalize_travel_distance`**
  ([`Genesis/action_sampling.py`](../Genesis/action_sampling.py)) for the ray-box
  truncation, which is already written and tested. No new geometry code.
- It **preserves the push-length distribution exactly** — the existing
  "difference of two uniforms in a yaw-dependent box" marginal is kept, so v1
  datasets stay length-comparable to every dataset already collected. Sampling
  `l ~ U(0, t_max)` instead would silently change it.
- `sign` matters: without it, every push travels along `+n̂`, and since
  `θ ∈ (−π/2, π/2)` the normal always has `cos θ > 0` — **every push would go in
  the `+y` half-plane**. That would be a badly biased dataset, and the bug would
  be invisible in aggregate statistics.

**Ordering hazard — the one thing to get right.** `placement_aware`
(`_apply_placement_aware_starts`, line 1687) **overrides both the start point and
the yaw**. So the perpendicular rewrite must run *after* it, or the yaw changes
underneath and perpendicularity silently breaks. `shared_travel_distance` is safe
either side, since it rescales along the existing direction. Correct order:

```
draw angles / starts / stops  →  placement_aware  →  perpendicular_pushes  →  shared_travel_distance
```

A test asserting `|cos(angle_between(push_dir, blade_normal))| < 1e-5` under all
four flag combinations is what keeps this honest.

### 7.3 Flag threading — follow the `--placement-aware` pattern exactly

| File | Change |
|---|---|
| `Genesis/sandbox_manipulation_clean.py:1576` | the kwarg + the 8 lines above |
| `Genesis/sandbox_manipulation_clean.py:1935` | `collect_data_samples(..., perpendicular_pushes=False)` → pass through, **and add it to `self._config["data_collection"]`** (line ~1961) so the constraint is recorded in the saved dataset metadata — otherwise a v1 and a non-v1 dataset are indistinguishable on disk |
| `Genesis/data_collection_clean.py:116` | one `parser.add_argument("--perpendicular-pushes", action="store_true")` + pass at line 271 |
| `Genesis/run_collection.py:339` | one `if spec["perpendicular_pushes"]: cmd += ["--perpendicular-pushes"]` |

The MPC side gets the same restriction for free from `ActionGridSampler` (Step 4):
enumerate `(start_x, start_y, θ, l)` and *derive* the plate yaw as `θ ± 90°`. The
other samplers are untouched, so nothing existing changes behaviour.

### 7.4 v1 → v2: what the extra DOF actually costs

| | **v1 — perpendicular** | **v2 — angle-aware** |
|---|---|---|
| World action DOF | 4 (`start_xy`, `θ`, `l`) — **identical to the paper's `u ∈ R⁴`** | 5 (plate yaw independent) |
| Tool-frame action after the warp | **1-D: push length** | 2-D: `(length, relative plate angle)` |
| Operators to fit | **5** (their 5 length bins) | 5 × K. At K=6 angle bins: **30** |
| Transitions for their ~1000/bin conditioning | **5 000** — `oneset` (14.3k) gives ~2 800/bin, *above* the paper's budget | **30 000** — `oneset` gives ~480/bin (**below** their 1000); `corl_cube` (35k) gives ~1 170/bin, but smeared across physics regimes |
| Enumerated MPC grid | 4-D | 5-D → K× the candidates → their 1.0 s/step becomes ~6 s |
| Push-model interface | **no change** — `predict_one_step_occ(occ, [sx,sy,ex,ey])` already carries no yaw | plate yaw must be threaded into the push-model signature (INTERFACES §3.4), touching **every** registered heuristic |

So v2 is not "a bit more data" — at fixed per-bin conditioning it is **6× the
operators and 6× the data**, plus an interface change and a 6× slower controller.

**The smarter v2, if you ever want it, is not more bins.** Partitioning the data
by angle is the brute-force route. The alternative already sketched in
[`analytic_descriptors_latent_space_plan_v2.md`](analytic_descriptors_latent_space_plan_v2.md) §3
is to keep **one** operator set and make the angle a *continuous input*:
`y′ = A_l y + B_l u_θ` (DMDc) or the bilinear rung `y′ = (A_l + Σ u_θ,i N_i) y`.
That shares all 14k transitions across angles instead of splitting them into 30
piles, at the cost of leaving the paper's model class (their `A_i` takes no input
by construction). Worth knowing that the escape hatch exists; not worth building
for a baseline.

### 7.5 What the restriction means for the paper you'd write

This is the part worth being deliberate about, because the flag is trivial and the
claim it licenses is not.

**The assumption itself is uncontroversial.** Push surface ⊥ push direction is the
standard planar-pushing convention (Mason 1986 — the paper's own reference [18]),
and it is what Agrawal et al. and Zeng et al. use. No reviewer will object to a
*baseline* being configured that way.

**What matters is whether your method gets the same restriction.** Three
possibilities, only two of which are defensible:

1. **Whole benchmark is perpendicular-only** (your method included). Clean,
   apples-to-apples, and the restriction is a stated scope limitation. This is the
   safe framing and it costs you nothing you can't add later.
2. **Your method uses all 5 DOF, the baseline gets 4.** Your method is then
   optimizing over a *strictly larger action space* than the baseline. A reviewer
   will find this, and it invalidates the comparison — this is the trap, and it is
   easy to fall into by accident, since the collection flag makes v1 datasets and
   the MPC action space diverge independently.
3. **You claim your method exploits the extra DOF.** This is the interesting claim
   — an oblique blade *shears* material sideways rather than plowing it forward,
   and the linear baseline structurally cannot represent that (its warp assumes the
   swept rectangle's normal *is* the push direction). But claiming it requires
   showing the baseline can't, which means v2 or full-5-DOF baselines.

**Cheap probe that decides this before you commit — run it on data you already
have.** Histogram `|relative angle between chosen push direction and blade
normal|` over the actions the **Genesis oracle MPC** selected
([`oracle_mpc_design.md`](oracle_mpc_design.md)) — that optimizer has no
restriction and no model error, so what it *chooses* is evidence about whether the
5th DOF is worth anything on this task:

- **Concentrated near perpendicular** → the extra DOF buys nothing, v1 costs you
  nothing, and that null result is itself worth one sentence in the paper
  ("the oracle converges to near-perpendicular pushes, so we restrict the action
  space without loss").
- **Spread out** → oblique pushes are doing real work, and framing 3 becomes
  available *and* necessary. You would then need v2 to make the comparison fair.

The same histogram over the **human-piloted** actions
([`human_demo_design.md`](human_demo_design.md)) is a second, independent read on
the same question, and arguably the more interesting one — it says whether *people*
reach for oblique pushes when solving this task.

---

## 8. Resolution: what to check, and the metric trap

Fitting at 32×32 is the default (§3), but it must be verified rather than assumed,
and there is one way to do it wrong.

**The trap: `‖·‖_F` is not comparable across resolutions.** The paper's Table 1
metric sums over `N²` pixels, so a 64×64 error is ~2× a 32×32 error for identical
physical accuracy. Any cross-resolution table must report a
**resolution-independent** metric — per-pixel RMS, or better, error normalized by
pile mass (`‖Î−I‖_F / ‖I‖_1`) plus **COM error in mm** as plan-v2's gate G3
already demands. Report the raw `‖·‖_F` too, but only for comparison against the
paper's own numbers at 32×32.

**One-step ablation.** Fit and evaluate the operator at 32 and 64, and evaluate
every other model (`spread`, `cumulative`, `NFDUNetFilm`, persistence) at both, on
the identical held-out split. Two distinct questions fall out: does the *operator*
lose accuracy at 32, and does the *task* need 64?

**Sequential rollout — the effect that only shows up in closed loop.** The
existing datasets are single-push transitions (report §3.1), so multi-step is only
observable by running the closed loop in Genesis. There the resolution choice
compounds in a way one-step error cannot reveal: a 32×32 model inside a 64×64
pipeline applies a **downsample→upsample low-pass filter once per MPC step**, so
blur accumulates over the ~20 steps of an episode even if each single step looks
fine. Diagnostics to record per step:

- occupancy entropy / effective support area (does the predicted pile diffuse?)
- `‖I_k‖_1` drift (§3's mass diagnostic — separates *physical* mass loss from
  *resampling* mass loss)
- predicted-vs-realized `V` gap, which `run_simple_mpc` already prints
  ([`mpc.py:423`](../simple_mpc/mpc.py#L423))

If blur accumulation turns out to dominate, the fix is not a bigger operator: keep
the 32×32 fit but **re-derive the input from the true 64×64 observation every
step** rather than feeding predictions back — which the closed loop already does,
making this mostly a concern for the open-loop rollout figures.

---

## 9. Why the Lyapunov axis is the real deliverable

This repo already has two reference points that most papers do not:

- [`simple_mpc/oracle_mpc.py`](../simple_mpc/oracle_mpc.py) — Genesis *is* the
  model, so there is **zero dynamics-model error**. Whatever it achieves is the
  ceiling for any model-based method on this task.
- [`simple_mpc/human_mpc.py`](../simple_mpc/human_mpc.py) — the same ceiling with
  the *optimizer* limitation also removed (a human proposes, grid search refines).

Once `image_lyapunov` is a registered loss (Step 3a), `oracle_mpc` and `human_mpc`
consume it for free, because they rank candidates through the loss registry. That
means **one plot** with every method on a common axis:

```
V vs MPC iteration:   human-piloted ceiling
                      Genesis-oracle ceiling
                      ── linear foresight (this baseline)
                      ── NFDUNetFilm / learned models
                      ── spread / cumulative heuristics
                      ── persistence floor (do nothing)
```

The value is the **bracketing**: a learned model's descent curve means little in
isolation, but "it recovers 70% of the gap between the heuristic floor and the
Genesis ceiling, and beats published linear foresight" is a claim that needs no
further context. Adopting their metric is what makes the two ceilings you already
have legible — which is a larger payoff than the baseline number itself, and it is
~30 lines of loss code.

---

## 10. The stage-2 gate, spelled out

Stage 2 (§5) exists to produce **one cheap number before any closed-loop machinery
is built**: the held-out one-step error of the fitted pixel operator, against
(a) persistence and (b) the existing `spread` / `cumulative` heuristics, on the
same split. That is ~1.5 days of work; stages 3–4 are ~2 more.

**Why gate there.** The paper's entire result — CLF, greedy search, sim→real —
rests on `A_l` being a *good predictor*. If it is not, the closed loop cannot
descend, and building the controller first would mean discovering that after
double the work.

**The specific way I expect it could fail, and why stage 2 is the right place to
find out.** Their `V` is mass-normalized, and their convexity argument needs
`‖I_k‖_1 ≈ const`. In 2-D Pymunk that is nearly exact — carrots cannot leave the
image. In 3-D Genesis, top-down mass is *not* conserved: material stacks (leaving
the footprint), and can escape past the plate edges. The DMDc report already found
**mass** to be precisely the descriptor group where linear prediction fails, and
flagged it as physics-dependent. An operator that cannot predict total mass cannot
predict a mass-normalized `V`.

**"Kill" is the wrong word — the result becomes the finding.** If the operator
loses to the heuristics, you do not abandon the work: you report it. *"Published
linear visual foresight degrades on 3-D granular piles because top-down mass is
not conserved, unlike the 2-D setting it was validated in"* is a legitimate and
useful comparison result, obtained for 1.5 days instead of 4, and it strengthens
the case for whatever your method does instead. What the gate actually decides is
whether to spend the remaining ~2.5 days on the **closed-loop** half — and you only
spend it if the one-step number says the closed loop has a chance.

Concretely, proceed to stage 3 if the operator beats persistence on mass-normalized
one-step error **and** is within noise of (or better than) `cumulative`. Stop and
write up if it loses to persistence on mass.
