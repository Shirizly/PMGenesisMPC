# Settling investigation — why does a pile take seconds of simulated time to stop?

Status: **root cause identified and fixed** (§1.7); residual characterised
(§1.8); the cost model corrected (§1.9). Remaining items are secondary. This document records what has been measured so far, what the
measurements imply, what is still unexplained, and how to find out.

Context: settling dominates data-collection cost at large particle counts. One
settle of 200 particles across 32 environments takes **2041 s of wall clock**,
against roughly 208 simulation steps for the push itself — so settling is on the
order of 85–90 % of the cost of producing a transition. It is also a *fidelity*
question, not only a throughput one: `collect_data_samples` takes each
transition's `s` from the previous transition's `s'`, so a state read before the
pile has stopped contaminates the next sample's input as well as its own output.

The physical expectation is that a tray of 5 mm cubes dropped ~5 mm comes to rest
in well under one second. The simulation takes several seconds and, as shown
below, never fully stops.

---

## 1. What has been measured

All numbers below are from files in `outputs/scaling_benchmark/`, produced by
scripts in `Genesis/`. Where a file is named, the numbers are copied from it
verbatim.

### 1.1 Steps to reach rest, single environment

`tests/scaling_investigation/probe_settle_convergence.py` → `settle_convergence.json`.
Fresh respawn (`shuffle_particles`), 5 mm cubes, `n_envs=1`, 3 trials, cap 3000.
Rest criterion at the time: max over all particles, thresholds 1 mm/s linear and
0.1 rad/s angular.

| n_particles | steps to rest | wall seconds | max linear at exit | max angular at exit |
|---|---|---|---|---|
| 50 | 40, 40, 40 | 1.9–2.2 | 4.1e-4 m/s | 4.4e-4 … 0.068 rad/s |
| 100 | 40, 40, 40 | 3.7–3.9 | 4.1e-4 m/s | 4.6e-4 … 0.089 rad/s |
| 200 | 1020, 640, 1260 | 134–250 | 3.7e-4 … 8.4e-4 m/s | 0.057 … 0.099 rad/s |

Two observations. Small piles settle in **40 steps = 0.16 s**, which is
physically sensible. The jump to 640–1260 steps at 200 particles is large,
variable (2× spread across three identical trials), and coincides exactly with
the switch from a single-layer spawn to a **two-layer spawn** that has to
collapse.

At one environment the exit state is genuinely quiet: max angular velocity at
exit is ≤ 0.1 rad/s.

### 1.2 Steps to reach rest, batched

`tests/scaling_investigation/probe_settle_convergence.py --n-envs 32` → `settle_batched.json`.
Same setup, `n_envs=32`, quantile-based criterion (q=0.995), derived angular
threshold (0.231 rad/s at 5 mm).

| n_particles | steps to rest | wall seconds | max linear at exit | max angular at exit |
|---|---|---|---|---|
| 150 | 570 | 545 | 0.029 m/s | **8.62 rad/s** |
| 200 | 1460 | 2041 | 0.048 m/s | **18.65 rad/s** |

**This is the most alarming number in the investigation.** At the moment the
criterion declares the pile settled, the worst particle is rotating at 18.6 rad/s
— about three revolutions per second — and translating at 48 mm/s. That is not a
settled particle by any reading. The quantile criterion (0.5 % of 6400 particles
= up to 32 particles permitted above threshold) is hiding it.

Contrast with §1.1: at one environment, the *same* pile size exits with max
angular ≤ 0.1 rad/s. So the fast-moving particles are not an inevitable feature
of a 200-cube pile; something about the batched runs, or simply the larger
sample of particles, surfaces them.

### 1.3 Decay of motion over simulated time

`tests/scaling_investigation/probe_settle_truncation.py`, n=200, `n_envs=8`, cap 3000 steps (12 s).
`q99.5` = the 99.5th-percentile particle speed across all envs.

| sim time | steps | q99.5 linear | q99.5 angular |
|---|---|---|---|
| 0.5 s | 125 | 22.59 mm/s | 8.53 rad/s |
| 1.0 s | 250 | 4.48 mm/s | 1.12 rad/s |
| 2.0 s | 500 | 1.37 mm/s | 0.29 rad/s |
| 4.0 s | 1000 | **1.36 mm/s** | **0.34 rad/s** |
| 6.0 s | 1500 | 1.15 mm/s | 0.30 rad/s |
| 8.0 s | 2000 | 0.94 mm/s | 0.20 rad/s |
| 12.0 s | 3000 | **1.46 mm/s** | **0.46 rad/s** |

**This is the key result so far, and the last row settles it.** Motion falls by a
factor of ~16 between 0.5 s and 2 s, and then stops falling. It is *higher* at
12 s than at 2 s, and higher at 4 s than at 2 s. Over the whole interval 2 s →
12 s the metric wanders between 0.94 and 1.46 mm/s with no downward trend.

A physically settling pile dissipates energy monotonically toward zero. This does
not decay — it reaches a **noise floor at about 2 s and then fluctuates around it
indefinitely**. That is a persistent source of excitation which the solver
re-injects every step, faster than contact dissipation can remove it.

Two direct consequences:

- **There is no convergence to detect after ~2 s.** Any absolute rest threshold
  placed inside the floor (and 1.0 mm/s / 0.231 rad/s is inside it) converges
  only when a random fluctuation happens to dip below the line. That fully
  explains the 2× trial-to-trial spread in §1.1, the erratic 640–1460 step
  results, and why the criterion previously appeared to "never converge".
- **Settling longer than ~2 s buys nothing.** Every step past the knee is pure
  cost. Whether it is *safe* to stop there depends on whether positions have
  converged, which is what §4.1 measures.

### 1.4 Per-step cost and where it goes

`tests/scaling_investigation/probe_step_cost.py` → `step_cost.log`, n=50, `n_envs=1`.

| max_collision_pairs | box_box_detection | raw `scene.step()` | full sweep-loop step | control/sync overhead |
|---|---|---|---|---|
| 200 | on | 31.0 ms | 52.8 ms | 21.8 ms (41 %) |
| 200 | off | 45.6 ms | 58.1 ms | 12.5 ms |
| 800 | on | 30.8 ms | 96.9 ms | **66.0 ms (68 %)** |
| 800 | off | 45.8 ms | 69.1 ms | 23.3 ms |

Raw physics cost is **independent of `max_collision_pairs`** (31.0 vs 30.8 ms).
The *overhead* is not: it triples from 21.8 to 66.0 ms as the cap goes from 200
to 800. That overhead is the per-step `set_dofs_position` used to hold the plate
still, which calls `collider.reset()` and `constraint_solver.reset()` — both
O(preallocated buffer size) — plus a whole-scene forward-kinematics pass, on
*every* step of the settle.

This matters to the settling question for a reason beyond speed: those resets
discard the constraint solver's **warm start** every step. With
`iterations: 10` and `tolerance: 1e-4` (Genesis defaults are 50 and 1e-6), a
contact solve that begins from scratch every step has little chance of
converging, which is a strong candidate for the noise floor in §1.3.

### 1.5 Related settings (values in force *at the time of §1.1–1.4*)

From `Genesis/configs/basic.yaml`:

| setting | value | Genesis default | note |
|---|---|---|---|
| `dt` | 4e-3 | — | fixed; **no adaptive timestep exists in Genesis** |
| `substeps` | 5 | — | fixed; substep is 0.8 ms |
| `iterations` | 10 → **50** | 50 | raised in §1.7 |
| `tolerance` | 1e-4 → **1e-6** | 1e-6 | raised in §1.7 |
| `ls_iterations` | 10 → **50** | 50 | raised in §1.7 |
| `ls_tolerance` | 0.05 → **0.01** | 0.01 | raised in §1.7 |
| `constraint_timeconst` | (default) 0.01 s | 0.01 s | 2.5× `dt`; soft contacts |
| `box_box_detection` | true | false | 16 contact points/pair instead of 5 |

Also relevant: contact restitution is effectively zero in this solver
configuration, so collisions are already maximally dissipative — the pile is not
failing to settle because it is bouncing elastically.

### 1.6 Jitter or real displacement? Both — in a ~99 / 1 split

`tests/scaling_investigation/probe_settle_truncation.py` → `truncation.json`. n=200, 8 envs. For each
candidate truncation point: distance from the fully-settled (12 s) configuration,
and — the direct test of "truncate and zero the velocities" — restore that
checkpoint *with velocities zeroed* and measure how far the pile then moves over
250 free steps.

| truncate at | dist from converged p50 / p99 / max (mm) | moves after zeroing p50 / p99 / max (mm) |
|---|---|---|
| 0.5 s | 0.001 / 6.674 / 7.472 | 0.000 / 5.651 / 6.057 |
| 1.0 s | 0.000 / 6.478 / 7.360 | 0.000 / 5.450 / 5.639 |
| 2.0 s | 0.000 / 6.336 / 7.196 | 0.000 / 2.597 / 6.513 |
| 4.0 s | 0.000 / 5.992 / 6.896 | 0.000 / **0.600** / 6.601 |
| 6.0 s | 0.000 / 3.347 / 6.677 | 0.000 / 0.458 / 0.686 |
| 8.0 s | 0.000 / 0.544 / 6.433 | 0.000 / 0.498 / 5.210 |
| converged (12 s) | (reference) | 0.000 / **0.566** / 2.823 |

Three things follow, and they refine the §1.3 picture rather than confirming it.

**The pile is not uniformly jittering — it is ~99 % still and ~1 % genuinely
moving.** The *median* particle is 0.000 mm from its final position at every
checkpoint, including 0.5 s. Meanwhile the p99 is 6.7 mm — more than a full
particle diameter — so roughly 1 % of particles (about 2 per env at n=200) are
still relocating properly, for seconds. That is real rearrangement, not
numerical vibration.

**Zeroing velocity becomes sound at about 4 s.** The "moves after zeroing" p99
falls to 0.600 mm at 4 s and 0.458 mm at 6 s, against a converged-state baseline
of 0.566 mm. From 4 s onward, a velocity-zeroed truncated state is as stable as
the fully-settled one.

**"Distance from converged" is the wrong soundness criterion.** At 4 s the state
is stable under zeroing (0.600 mm) yet still 5.99 mm from the 12 s state. Both
can be true because a granular pile is multistable: zeroing velocity settles
those few mobile particles into a *different but equally valid* resting
arrangement rather than the one they would have reached. For a dataset that is
acceptable — every state is a legitimate pile — provided the recorded `s'` is
stable, which is what the hold test measures. The verdict logic in the probe is
stricter than this and should be relaxed to the hold test alone.

### 1.7 RESOLVED — the cause, via A/B on an identical spawn

`tests/scaling_investigation/probe_settle_ab.py` → `settle_ab.json`. n=200, 4 envs, 2000 steps. All
four variants settle the *same* snapshotted spawn, same seed.

| variant | worst particle linear | worst angular | q99.5 floor | ms/step | rel. cost |
|---|---|---|---|---|---|
| baseline (per-step teleport, 10 iters) | 25.1 mm/s | 7.22 rad/s | 0.90 mm/s | 190.9 | 1.00× |
| **pd_hold** (no teleport, 10 iters) | **1.0 mm/s** | **0.28 rad/s** | 0.21 mm/s | 143.8 | **0.75×** |
| more_solver (teleport, 50 iters) | 0.8 mm/s | 0.25 rad/s | 0.68 mm/s | 203.4 | 1.07× |
| **both** | **0.2 mm/s** | **0.07 rad/s** | **0.11 mm/s** | 154.1 | **0.81×** |

**The per-step `set_dofs_position` that held the plate was the dominant cause.**
Removing it alone drops the worst particle 25× in linear speed and 26× in
angular, *and* costs 25 % less per step. Restoring Genesis's default solver
budget on top gives an 8× lower noise floor at 0.81× the baseline cost — the
removed collider/constraint resets cost more than five times the solver
iterations do.

> **Stale as written — corrected.** Of the two changes, only the plate one
> survives, and neither config key above is current:
>
> * Holding the plate by control is now **unconditional**; the
>   `hold_plate_by_control` flag no longer exists.
> * `iterations: 50` / `tolerance: 1e-6` were **reverted to 10 / 1e-4**. This
>   A/B was settle-only, where extra iterations are nearly free because the
>   plate is clear of the pile and nothing is moving. Measured on a broadside
>   *push* — the regime that actually dominates collection cost — 50 iterations
>   is 1.6-1.8x slower per step. See `probe_push_cost.py`.
>
> The plate finding itself stands, and it is worth noting what it cleared:
> §3's Q10 suspected the plate's 0.5 kg armature against a 0.125 g particle
> (4000:1) of being the cause. It was not — the per-step `set_dofs_position`
> was. The gantry actuator model is in fact what made the fix possible, since
> holding a 2.4 g box by control is only trustworthy given a real `kp`.

### 1.8 What the remaining movers actually are

`tests/scaling_investigation/probe_persistent_movers.py` → `movers_respawn.json`, `movers_push.json`.
n=200, 2 envs, 1 s snapshots, after the §1.7 fixes.

In both modes the pile is at rest by any reasonable reading: **median particle
speed 0.001 mm/s**, and only **0–3 of 400 particles** above 1 mm/s at any
snapshot. The entire "noise floor" of §1.3 is those few particles. They fall
into two populations, and identity persistence is 67–100 %, so they are the same
particles snapshot to snapshot — a localized effect, not global noise.

**Population A — corner/floor jitterers** (e.g. idx 177 respawn, idx 129 push).
Height above floor 2.38–2.55 mm, i.e. resting *on* the floor (a 5 mm cube at
rest has its centre 2.5 mm up), often with a wall gap of ~2.5 mm, i.e. touching a
wall. Net displacement 0.01–0.24 mm against a path length of 2.3–3.3 mm — a
**path-to-net ratio of 10–100×**. These vibrate in place and go nowhere; they
spike to 5–37 mm/s intermittently, which is what drives the max statistic, while
contributing nothing to the state. Numerical.

**Population B — second-layer creepers** (idx 119, 153, 157, 189, 53). Height
above floor 7.13–7.34 mm, i.e. resting on top of another cube (2.5 + 5 = 7.5).
Net ≈ path (0.1–0.6 mm per second window), so this is genuine slow translation,
not vibration. Speed grows slowly: idx 157 goes 0.14 → 0.17 → 0.20 → 0.24 →
0.42 mm/s over four seconds.

Is population B physical sliding? No. A cube on a slope steeper than the friction
angle (atan 0.3 = 16.7°) accelerates at `g(sin θ − µ cos θ)`; even at 20° that is
~0.6 m/s², reaching 0.6 m/s within a second. The observed acceleration is
(0.42 − 0.14) mm/s over 4 s ≈ **7×10⁻⁵ m/s², four orders of magnitude too small
to be gravity-driven sliding**. It is numerical creep — the signature of a
compliant-contact solver under sustained load, where `constraint_timeconst`
(0.01 s, 2.5× the 4 ms step) lets stacked bodies slowly sink and slide.

**No genuine interpenetration.** Nearest-neighbour distances are 5.0–6.9 mm
against an inscribed sum of 5.0 mm; the only `OVERLAPPING` flags sit exactly at
5.0 mm, i.e. at the classification boundary rather than meaningfully inside it.
That rules out the "particle wedged inside another and continuously ejected"
hypothesis.

### 1.9 Post-push settling is ~8× cheaper than post-respawn

Same probe, `--mode push`: settle fully, execute one sweep, then watch.

| | q99.5 linear at t=1 s | particles > 1 mm/s | time to be at rest |
|---|---|---|---|
| after a **push** | **0.99 mm/s** | 2 / 400 | **~1 s (250 steps)** |
| after a **respawn** | see §1.3 | — | ~8 s (2000 steps) |

**Every settling cost figure in this document before this section was the
respawn case**, which only occurs at reset. The settle that runs after each
recorded push — the one that actually multiplies per-transition cost — reaches
the same at-rest state in about **one second of simulated time**.

This matters more than any other number here: the early-exit criterion already
takes advantage of it, so per-transition settling is roughly 250 steps rather
than the 1460 measured after a respawn. The expensive respawn settle is paid
once per state-library entry, not once per transition.

### 1.10 What has *not* been measured

Stated explicitly because it is easy to assume otherwise:

- Whether the two-layer spawn is the trigger for the 40 → 1000+ step jump on
  *respawn*, or whether the particle count alone is. §1.8 makes the spawn the
  strong suspect — the creepers all sit at second-layer height — but it has not
  been tested directly (see §4.4).
- Whether reducing `constraint_timeconst` (currently 0.01 s against a 4 ms step)
  suppresses the numerical creep of population B.
- Post-push settling has now been measured (§1.9). Post-respawn settling with the
  §1.7 fixes in place has not been re-measured at 32 envs; the 2041 s figure
  predates them and is expected to fall substantially.

---

## 2. What the evidence currently supports

1. **The long settle is not physical, and it is not settling.** The pile reaches
   a floor at ~2 s and then fluctuates around it out to at least 12 s, ending
   *higher* than it started the plateau (§1.3). There is no slow convergence
   happening after 2 s that we are waiting for — there is nothing to wait for.
2. **There is a persistent excitation floor** at roughly 0.9–1.5 mm/s and
   0.2–0.46 rad/s at the 99.5th percentile (§1.3), and the rest thresholds sit
   inside it, which makes convergence detection a coin flip rather than a
   measurement. Every threshold-tuning fix so far — the quantile, the derived
   angular threshold — has been adjusting where the line sits within the noise,
   not removing the noise.
3. **A small number of particles are moving far faster than the floor** — 48 mm/s
   and 18.6 rad/s at the moment of declared convergence (§1.2). Sustained
   rotation at 3 rev/s is not jitter; jitter oscillates about zero. Something is
   continuously driving those particles.
4. **That small number is about 1 % of the pile, and it is doing real work.**
   §1.6 shows the median particle is exactly at its final position from 0.5 s
   onward, while the 99th percentile is more than a particle diameter away and
   stays there for seconds. So the "noise floor" of §1.3 is not the whole pile
   vibrating — it is a couple of particles per environment still relocating,
   dominating a percentile statistic computed over an otherwise-static pile.
   This reframes the question from "why won't the pile settle" to **"which few
   particles never stop, and why"**.
4. **The most likely mechanism is contact-solver non-convergence**, from
   discarding the warm start every step (§1.4) combined with a solver budget five
   times looser than Genesis's default (§1.5).
5. **Batching does not change the physics**, but it changes what is observed:
   32 envs sample 32× more particles, so the tail of the distribution is far
   better sampled. This has already caused one measurement error in this
   investigation (a max-based criterion that could not converge at 32 envs), and
   the same effect may explain why fast particles appear at 32 envs and not at 1.

---

## 3. Open questions

**Q1. Is the residual motion jitter in place, or genuine displacement?**
Velocity alone cannot distinguish a particle vibrating about a fixed point from
one slowly creeping. The discriminator is net displacement versus path length: if
total path length over a window greatly exceeds net displacement, it is jitter.

**Q2. Which particles are still moving, and are they always the same ones?**
If a stable handful of particles accounts for all the fast motion, it is a
localized defect (a wedged cube, a bad contact pair) rather than a global
property of the pile.

**Q3. Where are the fast particles?** Top of the pile, against a wall, in a
corner, underneath others, or outside the tray. Each implicates a different
mechanism.

**Q4. Is the noise floor caused by discarding the constraint warm start every
step?** This is the leading hypothesis and it is directly testable.

**Q5. Is the noise floor caused by the loose solver budget** (10 iterations,
1e-4) rather than, or in addition to, the warm-start discard?

**Q6. Does the two-layer spawn cause the 40 → 1000+ step jump**, or is it the
particle count? At 5 mm, n≤147 is one layer and n=200 is two.

**Q7. How expensive is a *post-push* settle**, as opposed to a post-respawn one?
This is the number that actually determines collection cost, and it has never
been measured.

**Q8. Is the configuration already converged long before the velocity criterion
trips?** If positions stop changing at ~2 s while velocities plateau above
threshold, truncation plus velocity zeroing is justified. (Probe running; see
§4.1.)

**Q9. Are contacts penetrating significantly?** Persistent penetration would
explain continuous constraint forces and hence continuous excitation.

**Q10. Does the plate participate?** It is held by a per-step `set_dofs_position`
during the settle. Its armature is 0.5 kg against a 0.125 g particle — a mass
ratio of 4000:1, which is hard on a constraint solver if they are in contact.

---

## 4. Suggested investigation plan

Ordered by information gained per unit of GPU time.

### 4.1 Already running — jitter vs. displacement, and truncation safety

`tests/scaling_investigation/probe_settle_truncation.py` (n=200, 8 envs). For checkpoints at 0.5–8 s
it reports each checkpoint's distance from the fully-settled configuration as a
distribution over particles, and — the decisive test for the truncation
proposal — restores each checkpoint *with velocities zeroed* (exactly what
`set_particle_state` does) and measures how far the pile then moves, against the
converged state's own drift as a baseline.

Answers **Q8**, and partly **Q1**.

### 4.2 Characterize the residual motion (cheap, high value)

New probe. After settling, over a window of ~250 steps, per particle record:
net displacement, total path length, mean speed, and the index/position/height of
the fastest particles at each checkpoint.

Report: ratio of path length to net displacement (the jitter discriminator);
whether the set of fast particles is stable across checkpoints; and a histogram
of speed against height above the tray floor and distance to the nearest wall.

Answers **Q1, Q2, Q3**. This is the single most informative next step, and it is
one short run.

### 4.3 A/B the leading mechanism (decisive for Q4/Q5)

Four settles of the same seeded initial state, changing one thing at a time:

| variant | change |
|---|---|
| baseline | as shipped |
| no warm-start discard | hold the plate without a per-step `set_dofs_position` — e.g. give it a stiff PD target once, or reposition it every K steps instead of every step |
| more solver budget | `iterations: 50`, `tolerance: 1e-6`, `ls_iterations: 50` |
| both | |

Compare the decay curves from §1.3. If the noise floor drops in the
"no warm-start discard" variant, the fix is architectural and cheap. If it drops
only with more iterations, it is a cost trade. Note the second variant also
removes the 21.8–66.0 ms/step overhead measured in §1.4, so it may pay for
itself twice.

Answers **Q4, Q5**.

### 4.4 Isolate the two-layer spawn (Q6)

Settle n=147 (one layer) and n=160 (two layers) at 5 mm, and separately n=200 at
4 mm (one layer, since capacity scales as 1/size²). If steps-to-rest tracks the
layer count rather than the particle count, the spawn geometry is the trigger and
a gentler spawn (smaller inter-layer gap, or per-layer sequential settling) fixes
it.

### 4.5 Measure the post-push settle (Q7)

Settle a pile fully, snapshot, execute one push, then measure steps-to-rest from
that snapshot. Repeat for several push magnitudes. This is the number that
belongs in the cost model; everything measured so far is the respawn worst case.

### 4.6 Contact penetration (Q9)

Read penetration depth from the collider after settling and report its
distribution. Non-trivial persistent penetration would confirm that constraint
forces never stop, and would point at `constraint_timeconst` (0.01 s against a
4 ms step) as a contributor.

### 4.7 If the floor turns out to be irreducible

Then the practical options, in order of preference:

1. **Truncate on a plateau criterion rather than an absolute threshold** — stop
   when the motion metric has not improved by more than X % over the last N
   checks. This adapts to whatever the floor is, instead of requiring a threshold
   that happens to sit above it, and it degrades gracefully as pile size changes.
2. **Truncate at a fixed time and zero velocities**, justified by §4.1's result
   and applied only where it is sound: the library/reset settle needs only *a*
   valid resting configuration, whereas the post-push settle produces the
   recorded `s'` and truncating it biases displacement downward.
3. Settle-phase-only damping (already implemented as
   `--state-library-damping`, off by default). Explicitly a numerical device —
   real air drag on a 5 mm cube at 50 mm/s is ~3e-5 of its weight, so it is not
   a physical justification.

---

## 5. Cost context

Why this is worth the effort, from `settle_batched.json` and the dry run:

- one settle, 200 particles, 32 envs: **2041 s**
- one settle, 150 particles, 32 envs: **545 s**
- a push sweep: ~208 steps, i.e. a few percent of the above

If §4.3 removes the noise floor and settling returns to the ~40-step behaviour
seen at 50–100 particles, per-transition cost at 200 objects falls by more than
an order of magnitude. That is a larger win than anything else currently on the
table, including the state library.
