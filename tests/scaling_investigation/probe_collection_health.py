#!/usr/bin/env python3
"""
probe_collection_health.py — does a realistic collection run produce data that
makes sense?

Every other probe here measures one mechanism in isolation under a hand-picked
action. This one runs the collection path as configured — random actions,
placement-aware touchdown, shared travel distance, state library, batched envs —
and asks whether the *output* is sound. It is the check that would have caught
each of the following, none of which any unit test can see:

  goal-reach rate         ``reached_goal`` decides which transitions land in the
                          kept file and which go to ``_failed``. Its meaning
                          changed during this work (it is now "final tracking
                          error < goal_threshold", not "reached at some point
                          during the sweep"), and it has only ever been checked
                          against hand-picked reachable targets. If it fails
                          often on the real random action distribution, the
                          dataset is being silently halved.

  per-env settledness     ``settle_rest_quantile`` is evaluated over every
                          particle in every env POOLED. That is deliberate — a
                          max over 6400 particles is an extreme-value statistic
                          that never converges — but it does not guarantee each
                          env is individually at rest. At 20 particles x 128
                          envs, one env with 10 particles still moving is 0.39 %
                          of the pool, under a 0.5 % tolerance. This reports the
                          WORST env at the instant the criterion trips, which is
                          the number that would expose it.

  action-space coverage   Placement-aware sampling draws p_start from free
                          space, and free space is wherever the pile is not — so
                          touchdowns may drift systematically toward the tray
                          edges. That is a bias in the action distribution the
                          policy will be trained on, and nothing looks for it.

  travel truncation       Shared travel distance re-points each stop to a common
                          push length, truncating at the tray boundary where
                          that would leave the box. A high truncation rate means
                          pushes near walls are systematically shorter.

  state validity          Particles outside the tray (escaped through a wall),
                          and transitions where s' == s (the push did nothing,
                          i.e. a null training example).

Nothing here asserts — thresholds for "acceptable" are a judgement call about
the dataset, not a property of the code. It prints numbers and flags the ones
that look wrong.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_collection_health
    python -m tests.scaling_investigation.probe_collection_health \
        --n-particles 200 --n-envs 32 --n-samples 8
    python -m tests.scaling_investigation.probe_collection_health \
        --cells 20:128 50:128 200:1 --n-samples 5
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GENESIS_DIR = REPO_ROOT / "Genesis"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _config(n_particles, particle_size):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------

def per_env_motion(sim):
    """Worst particle speed WITHIN each env — the statistic the pooled rest
    test cannot see. Returns (lin [n_envs] m/s, ang [n_envs] rad/s)."""
    import torch
    vel = sim._scene.rigid_solver.get_dofs_velocity(
        dofs_idx=sim._particle_dofs_idx).reshape(sim._n_envs, -1, 6)
    vel = vel[:, :getattr(sim, "_n_active", vel.shape[1])]
    return (vel[..., :3].norm(dim=-1).amax(dim=1),
            vel[..., 3:].norm(dim=-1).amax(dim=1))


class TruncationSpy:
    """Records how often shared travel distance had to truncate a push.

    Patches the module attribute rather than the method: ``_equalize_batch_travel``
    imports ``equalize_travel_distance`` from the module at call time, so this
    catches the real call without reimplementing the geometry — which would make
    the probe test its own copy of the logic instead of the shipped one.
    """

    def __init__(self):
        self.clipped = 0
        self.total = 0
        self.target_m = None
        self._orig = None

    def __enter__(self):
        from Genesis import action_sampling
        self._orig = action_sampling.equalize_travel_distance

        def spy(starts, stops, low, high, target):
            new_stops, clipped = self._orig(starts, stops, low, high, target)
            self.clipped += int(clipped.sum())
            self.total += int(clipped.numel())
            self.target_m = float(target.flatten()[0])
            return new_stops, clipped

        action_sampling.equalize_travel_distance = spy
        return self

    def __exit__(self, *exc):
        from Genesis import action_sampling
        action_sampling.equalize_travel_distance = self._orig
        return False


class PlacementSpy:
    """Records how often placement-aware sampling actually found free space."""

    def __init__(self):
        self.placed = 0
        self.total = 0
        self._orig = None

    def __enter__(self):
        from Genesis import placement_sampling
        self._orig = placement_sampling.sample_free_placements

        def spy(*a, **kw):
            xy, yaw, ok = self._orig(*a, **kw)
            self.placed += int(ok.sum())
            self.total += int(ok.numel())
            return xy, yaw, ok

        placement_sampling.sample_free_placements = spy
        return self

    def __exit__(self, *exc):
        from Genesis import placement_sampling
        placement_sampling.sample_free_placements = self._orig
        return False


def edge_fraction(starts_xy, sim):
    """Fraction of touchdowns in the outer third of the tray.

    Reported against the SAME quantity measured on a blind draw rather than
    against a closed-form uniform baseline: the blind sampler already shrinks
    its own bounds by the tool footprint and the safety margin, and by a yaw-
    dependent amount, so "what uniform would give" is not the tray's area ratio
    and is not worth deriving. What matters is only whether placement-awareness
    moves the distribution, which the paired draw answers directly.
    """
    import torch
    w, d, _ = sim._box_params["vol"]
    half = torch.tensor([w / 2, d / 2], device=starts_xy.device)
    # Chebyshev radius normalised to the tray half-width: 1.0 is the wall.
    r = (starts_xy.abs() / half).amax(dim=-1)
    return float((r > 2 / 3).float().mean()), float(r.mean())


def unsettled_particles(sim):
    """How many individual particles are above the rest thresholds.

    The env-level count says a batch has stragglers; this says how many. The
    pooled quantile permits a fixed FRACTION of particles to be moving, so the
    particle count is the number that criterion is actually tolerating."""
    import torch
    vel = sim._scene.rigid_solver.get_dofs_velocity(
        dofs_idx=sim._particle_dofs_idx).reshape(sim._n_envs, -1, 6)
    vel = vel[:, :getattr(sim, "_n_active", vel.shape[1])]
    over = ((vel[..., :3].norm(dim=-1) > sim._settle_vel_threshold)
            | (vel[..., 3:].norm(dim=-1) > sim._settle_angvel_threshold))
    return int(over.sum()), int(over.numel())


def hold_displacement(sim, steps):
    """Net particle displacement over ``steps`` extra steps after the settle
    criterion has already tripped.

    This is the question the velocity threshold cannot answer: a particle at
    15 mm/s is either travelling — in which case s' was recorded mid-motion and
    is not the settled state the dataset claims — or vibrating in place against
    its neighbours, in which case the velocity is real but nothing is actually
    moving and the recorded state is fine. Net displacement separates the two.
    """
    import torch
    n_active = getattr(sim, "_n_active", None)
    before = sim._get_particle_positions()[:, :n_active].clone()
    for _ in range(steps):
        sim._step_scene()
    after = sim._get_particle_positions()[:, :n_active]
    disp = (after - before).norm(dim=-1)
    return {"steps": steps, "sim_time_s": steps * sim._scene.dt,
            "max_mm": float(disp.max()) * 1000,
            "mean_mm": float(disp.mean()) * 1000,
            "n_over_0.1mm": int((disp > 1e-4).sum()),
            "n_total": int(disp.numel())}


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def run_cell(n_particles, n_envs, n_samples, particle_size, library_root, seed,
             placement_aware, shared_travel, hold_steps):
    import numpy as np
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation
    from Genesis.state_library import StateLibrary, default_library_path

    torch.manual_seed(seed)
    np.random.seed(seed)
    out = {"n_particles": n_particles, "n_envs": n_envs, "n_samples": n_samples,
           "placement_aware": placement_aware, "shared_travel": shared_travel}

    sim = SandboxManipulation(config=_config(n_particles, particle_size),
                              n_envs=n_envs, debug=False)
    sim.build()

    try:
        # Seed from the library exactly as collection does: ONE state broadcast
        # to every env. Falls back to a real spawn if no library exists, since
        # the spawn is itself worth exercising.
        lib_path = default_library_path(GENESIS_DIR / library_root, "cube",
                                        n_particles, particle_size)
        rng = np.random.default_rng(seed)
        if lib_path.exists():
            lib = StateLibrary.load(lib_path)
            lib.apply(sim, rng=rng)
            out["seeded_from"] = "library"
        else:
            sim.shuffle_particles()
            out["seeded_from"] = "spawn"

        t0 = time.perf_counter()
        sim.update_material_state()
        out["settle_s"] = time.perf_counter() - t0

        # --- per-env settledness at the instant the pooled criterion tripped
        lin, ang = per_env_motion(sim)
        thr_lin, thr_ang = sim._settle_vel_threshold, sim._settle_angvel_threshold
        unsettled = ((lin > thr_lin) | (ang > thr_ang))
        n_over, n_tot_p = unsettled_particles(sim)
        out["settle"] = {
            "threshold_lin_mm_s": thr_lin * 1000,
            "threshold_ang_rad_s": thr_ang,
            "quantile": sim._settle_rest_quantile,
            "worst_env_lin_mm_s": float(lin.max()) * 1000,
            "worst_env_ang_rad_s": float(ang.max()),
            "median_env_lin_mm_s": float(lin.median()) * 1000,
            "n_envs_over_threshold": int(unsettled.sum()),
            "frac_envs_over_threshold": float(unsettled.float().mean()),
            "n_particles_over_threshold": n_over,
            "frac_particles_over_threshold": n_over / max(n_tot_p, 1),
            "hold": hold_displacement(sim, hold_steps) if hold_steps else None,
        }
        sim.update_material_state()      # re-sync _particle_state after the hold

        # --- actions, with the two samplers instrumented.
        # Draw blind first from the SAME rng state, purely as the reference the
        # placement-aware distribution is compared against; it is then discarded.
        rng_state = torch.get_rng_state()
        blind_starts, _, _ = sim.generate_action_samples(
            n_samples, placement_aware=False, shared_travel_distance=shared_travel)
        torch.set_rng_state(rng_state)
        with TruncationSpy() as trunc, PlacementSpy() as place:
            starts, stops, angles = sim.generate_action_samples(
                n_samples, placement_aware=placement_aware,
                shared_travel_distance=shared_travel)
        out["truncation"] = {
            "clipped": trunc.clipped, "total": trunc.total,
            "frac": trunc.clipped / trunc.total if trunc.total else None,
            "shared_target_mm": trunc.target_m * 1000 if trunc.target_m else None,
        }
        out["placement"] = {
            "placed": place.placed, "total": place.total,
            "frac_free": place.placed / place.total if place.total else None,
        }
        obs_edge, obs_r = edge_fraction(starts[..., :2].reshape(-1, 2), sim)
        blind_edge, blind_r = edge_fraction(blind_starts[..., :2].reshape(-1, 2), sim)
        dist = (stops[..., :2] - starts[..., :2]).norm(dim=-1)
        out["actions"] = {
            "edge_frac_observed": obs_edge,
            "edge_frac_blind": blind_edge,
            "mean_radius_observed": obs_r,
            "mean_radius_blind": blind_r,
            "travel_mm_min": float(dist.min()) * 1000,
            "travel_mm_mean": float(dist.mean()) * 1000,
            "travel_mm_max": float(dist.max()) * 1000,
            # std over envs is undefined with a single env, and NaN in a report
            # reads as a bug rather than as "not applicable"
            "travel_mm_std_within_batch": (
                float(dist.std(dim=0).mean()) * 1000 if n_envs > 1 else None),
        }

        # --- the transitions themselves
        per_sample = []
        for i in range(n_samples):
            s = sim._particle_state.clone()
            p_start, p_stop = starts[:, i, :], stops[:, i, :]
            t_push = time.perf_counter()
            reached, final = sim.execute_action(p_start, p_stop, angles[:, i])
            torch.cuda.synchronize()
            t_settle = time.perf_counter()
            sim.update_material_state()
            torch.cuda.synchronize()
            settle_s = time.perf_counter() - t_settle
            push_s = t_settle - t_push
            s_ = sim._particle_state

            disp = (s_[..., 0:3] - s[..., 0:3])[:, :getattr(sim, "_n_active", None)]
            disp = disp.norm(dim=-1)
            lin_i, ang_i = per_env_motion(sim)
            n_over_i, _ = unsettled_particles(sim)
            err = (final[:, :2] - p_stop[:, :2]).norm(dim=-1)
            rec = {
                "push_s": push_s, "post_push_settle_s": settle_s,
                "reached_frac": float(reached.float().mean()),
                "tracking_err_mm_max": float(err.max()) * 1000,
                "tracking_err_mm_median": float(err.median()) * 1000,
                "moved_particles_frac": float((disp > 1e-4).float().mean()),
                "max_disp_mm": float(disp.max()) * 1000,
                "mean_disp_mm": float(disp.mean()) * 1000,
                "n_envs_unchanged": int((disp.amax(dim=1) < 1e-6).sum()),
                # the shipped check, not a copy of it — a probe that
                # reimplements what it is testing stops testing it
                "escaped": sim.escaped_particle_count(),
                "worst_env_lin_mm_s": float(lin_i.max()) * 1000,
                "n_envs_over_threshold": int(((lin_i > thr_lin) | (ang_i > thr_ang)).sum()),
                "n_particles_over_threshold": n_over_i,
            }
            if hold_steps:
                # Costs a little extra settling that collection would not do,
                # which only ever makes the following s MORE settled — it cannot
                # manufacture the motion this is looking for.
                rec["hold"] = hold_displacement(sim, hold_steps)
                sim.update_material_state()
            per_sample.append(rec)
        out["samples"] = per_sample

        n_tot = n_samples * n_envs
        out["summary"] = {
            "goal_reached_frac": sum(s["reached_frac"] for s in per_sample) / n_samples,
            "unchanged_transitions": sum(s["n_envs_unchanged"] for s in per_sample),
            "unchanged_frac": sum(s["n_envs_unchanged"] for s in per_sample) / n_tot,
            "escaped_final": per_sample[-1]["escaped"],
            "worst_tracking_err_mm": max(s["tracking_err_mm_max"] for s in per_sample),
            "worst_post_push_env_lin_mm_s": max(s["worst_env_lin_mm_s"] for s in per_sample),
            "envs_unsettled_after_push_max": max(
                s["n_envs_over_threshold"] for s in per_sample),
            "particles_unsettled_after_push_max": max(
                s["n_particles_over_threshold"] for s in per_sample),
            "worst_hold_disp_mm": (max(s["hold"]["max_mm"] for s in per_sample)
                                   if hold_steps else None),
            "mean_push_s": sum(s["push_s"] for s in per_sample) / n_samples,
            "mean_post_push_settle_s": sum(
                s["post_push_settle_s"] for s in per_sample) / n_samples,
        }
        u = sim.contact_budget_usage()
        out["contacts"] = {
            "broad_pairs": u["broad_pairs"], "broad_cap": u["broad_cap"],
            "contact_points": u["contact_points"], "contact_cap": u["contact_cap"],
            "worst_frac": max(u["broad_pairs"] / max(u["broad_cap"], 1),
                              u["contact_points"] / max(u["contact_cap"], 1)),
        }
        out["ok"] = True
    except Exception as e:
        import traceback
        out.update(ok=False, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1500:])
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def flag(cond, msg):
    return f"  {'!!' if cond else '  '} {msg}"


def report(cells):
    print("\n" + "=" * 78)
    print("COLLECTION HEALTH")
    print("=" * 78)
    for c in cells:
        head = f"n={c['n_particles']:>3}  envs={c['n_envs']:>3}"
        if not c.get("ok"):
            print(f"\n{head}   FAILED: {c.get('error')}")
            continue
        s, st, a = c["summary"], c["settle"], c["actions"]
        print(f"\n{head}   seeded from {c['seeded_from']}, "
              f"settle {c['settle_s']:.1f} s, {c['n_samples']} pushes/env")
        print(f"    cost/sample           push {s['mean_push_s']:.2f} s + "
              f"settle {s['mean_post_push_settle_s']:.2f} s")

        print("  settling (at the instant the pooled criterion tripped)")
        print(f"    thresholds            {st['threshold_lin_mm_s']:.1f} mm/s, "
              f"{st['threshold_ang_rad_s']:.2f} rad/s "
              f"at q={st['quantile']}")
        print(f"    worst env             {st['worst_env_lin_mm_s']:.2f} mm/s, "
              f"{st['worst_env_ang_rad_s']:.2f} rad/s")
        print(flag(st["frac_envs_over_threshold"] > 0.05,
                   f"envs over threshold   {st['n_envs_over_threshold']}/{c['n_envs']} "
                   f"({st['frac_envs_over_threshold']*100:.0f} %), "
                   f"{st['n_particles_over_threshold']} particles "
                   f"({st['frac_particles_over_threshold']*100:.2f} %)"))
        if st.get("hold"):
            h = st["hold"]
            print(flag(h["max_mm"] > 0.5,
                       f"net drift in {h['sim_time_s']:.2f} s     "
                       f"max {h['max_mm']:.3f} mm, mean {h['mean_mm']:.4f} mm "
                       f"({h['n_over_0.1mm']}/{h['n_total']} moved >0.1 mm)"))

        print("  actions")
        print(flag(c["placement"]["frac_free"] is not None
                   and c["placement"]["frac_free"] < 0.5,
                   f"free-space placements {c['placement']['placed']}/"
                   f"{c['placement']['total']}"
                   if c["placement"]["total"] else "placement-aware off"))
        print(flag(c["truncation"]["frac"] is not None and c["truncation"]["frac"] > 0.25,
                   f"truncated pushes      {c['truncation']['clipped']}/"
                   f"{c['truncation']['total']}"
                   + (f" (shared target {c['truncation']['shared_target_mm']:.0f} mm)"
                      if c["truncation"]["shared_target_mm"] else "")
                   if c["truncation"]["total"] else "shared travel off"))
        print(flag(abs(a["edge_frac_observed"] - a["edge_frac_blind"]) > 0.15,
                   f"touchdowns near rim   {a['edge_frac_observed']*100:.0f} % vs "
                   f"{a['edge_frac_blind']*100:.0f} % blind "
                   f"(mean radius {a['mean_radius_observed']:.2f} vs "
                   f"{a['mean_radius_blind']:.2f})"))
        sd = a["travel_mm_std_within_batch"]
        print(f"    travel                {a['travel_mm_min']:.0f}-"
              f"{a['travel_mm_max']:.0f} mm, mean {a['travel_mm_mean']:.0f}, "
              + (f"within-batch sd {sd:.1f}" if sd is not None
                 else "single env, no within-batch spread"))

        print("  transitions")
        print(flag(s["goal_reached_frac"] < 0.9,
                   f"goal reached          {s['goal_reached_frac']*100:.1f} % "
                   f"(worst tracking error {s['worst_tracking_err_mm']:.2f} mm)"))
        print(flag(s["unchanged_frac"] > 0.1,
                   f"s' == s               {s['unchanged_transitions']}/"
                   f"{c['n_samples']*c['n_envs']} "
                   f"({s['unchanged_frac']*100:.0f} %)"))
        print(flag(s["escaped_final"] > 0,
                   f"escaped particles     {s['escaped_final']}"))
        print(flag(s["envs_unsettled_after_push_max"] > 0.05 * c["n_envs"],
                   f"unsettled after push  up to "
                   f"{s['envs_unsettled_after_push_max']}/{c['n_envs']} envs, "
                   f"{s['particles_unsettled_after_push_max']} particles "
                   f"(worst {s['worst_post_push_env_lin_mm_s']:.2f} mm/s)"))
        if s.get("worst_hold_disp_mm") is not None:
            print(flag(s["worst_hold_disp_mm"] > 0.5,
                       f"  net drift after s'    max "
                       f"{s['worst_hold_disp_mm']:.3f} mm — "
                       + ("REAL MOTION: s' was recorded mid-travel"
                          if s["worst_hold_disp_mm"] > 0.5
                          else "jitter in place, s' is a valid resting state")))
        print(flag(c["contacts"]["worst_frac"] > 0.9,
                   f"contact budget        {c['contacts']['worst_frac']*100:.0f} % of cap "
                   f"({c['contacts']['contact_points']}/{c['contacts']['contact_cap']} points)"))

    print("\n('!!' marks a value worth looking at, not a hard failure.)\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", default=None,
                   help="n_particles:n_envs pairs, e.g. 20:128 200:1. "
                        "Overrides --n-particles/--n-envs.")
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=5,
                   help="pushes per env; more gives a better goal-reach rate")
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--library-root", default="data/dry_run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold-steps", type=int, default=50,
                   help="extra steps to run after each settle, to tell a "
                        "particle that is travelling from one that is only "
                        "vibrating. 0 disables.")
    p.add_argument("--no-placement-aware", action="store_true")
    p.add_argument("--no-shared-travel", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--cell", default=None, help=argparse.SUPPRESS)  # subprocess entry
    return p.parse_args()


def main():
    args = parse_args()

    if args.cell:                                   # child process: one cell
        n_particles, n_envs = (int(v) for v in args.cell.split(":"))
        res = run_cell(n_particles, n_envs, args.n_samples, args.particle_size,
                       args.library_root, args.seed,
                       not args.no_placement_aware, not args.no_shared_travel,
                       args.hold_steps)
        print("__RESULT__" + json.dumps(res))
        return

    cells = args.cells or [f"{args.n_particles}:{args.n_envs}"]
    out = Path(args.out) if args.out else RESULTS_DIR / "collection_health.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for cell in cells:
        print(f"\n>>> cell {cell}", flush=True)
        # Subprocess per cell: an OOM or a CUDA fault in one configuration must
        # not take the whole sweep with it, and Genesis cannot be re-initialised
        # in-process.
        proc = subprocess.run(
            [sys.executable, "-m", "tests.scaling_investigation.probe_collection_health",
             "--cell", cell, "--n-samples", str(args.n_samples),
             "--particle-size", str(args.particle_size),
             "--library-root", args.library_root, "--seed", str(args.seed),
             "--hold-steps", str(args.hold_steps)]

            + (["--no-placement-aware"] if args.no_placement_aware else [])
            + (["--no-shared-travel"] if args.no_shared_travel else []),
            cwd=REPO_ROOT, capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines()
                     if l.startswith("__RESULT__")), None)
        if line:
            results.append(json.loads(line[len("__RESULT__"):]))
        else:
            n_particles, n_envs = (int(v) for v in cell.split(":"))
            results.append({"n_particles": n_particles, "n_envs": n_envs, "ok": False,
                            "error": (proc.stderr or proc.stdout)[-400:]})
        # Write after EVERY cell, and print its report immediately. A full sweep
        # is over an hour of GPU time; losing all of it because the last cell
        # was interrupted is not an acceptable failure mode, and it is exactly
        # what happened the first time this was run.
        out.write_text(json.dumps(results, indent=2))
        report(results[-1:])

    print("\n" + "#" * 78)
    print("# FULL SWEEP")
    report(results)
    print(f"raw results -> {out}")


if __name__ == "__main__":
    main()
