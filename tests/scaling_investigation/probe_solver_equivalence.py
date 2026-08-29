#!/usr/bin/env python3
"""
probe_solver_equivalence.py — do the cheaper solver configurations produce the
SAME physics as the baseline, or just faster physics?

Why this gates everything else
------------------------------
probe_contact_islands.py found large speedups available: CG with islands off is
5.5x faster during a push, hibernation is 5.9x faster during a settle, and a
dual-scene scheme that uses each where it wins would be ~4.7x end to end. Every
one of those numbers is worthless — worse than worthless, actively harmful — if
the configurations do not agree on the contacts.

Three ways a mismatch would hurt, in increasing order of subtlety:

  1. A dataset collected under one solver is not comparable to one collected
     under another.
  2. Switching solver by object count (Newton below some n, CG above) entangles
     particle count with solver in the training data, so a model can learn a
     spurious property of "many objects".
  3. Dual-scene switching runs the PUSH under one solver and the SETTLE under
     another *within a single transition*, so s' would be produced by different
     physics than the dynamics that created it. That is worse than either
     solver alone.

The hard part: chaos is not error
---------------------------------
A granular pile is chaotic. Two solvers started from an identical state under an
identical action WILL diverge in particle positions, and so will one solver run
twice against itself. Comparing final positions and declaring a difference would
therefore "prove" a mismatch that is not there.

So this probe never interprets a difference in isolation. It measures a **noise
floor** from configurations that are known to be physically identical, and only
differences exceeding that floor count:

  newton_repeat   bit-identical rerun. Isolates GPU/run-to-run non-determinism;
                  the absolute lower bound on any difference.
  newton_eps      push start shifted by 1 um — 1/5000 of a particle, physically
                  meaningless. How far chaos alone carries the pile apart.

And it leads with a metric that chaos does not corrupt:

  penetration     how far bodies overlap at contacts, read from the solver's own
                  contact data. This is a property of the STATE, not of the
                  trajectory, so it stays meaningful even once two runs have
                  diverged. A solver that under-resolves contacts lets bodies
                  sink into each other, and that shows here regardless of where
                  the particles ended up. Reported separately for particle-
                  particle and plate-particle contacts, the latter being the
                  interface the action acts through.

Bulk outcome (centre-of-mass shift, displaced mass, particles moved) is reported
too, but as corroboration, and always beside the floor.

Configurations compared
-----------------------
  newton          Newton + contact islands. The baseline everything else is
                  measured against, and what every dataset so far used.
  newton_repeat   identical to newton (noise floor, see above)
  newton_eps      identical to newton, action perturbed 1 um (noise floor)
  cg              CG, islands off, iterations 10
  cg_iter30       CG, islands off, iterations 30. Included because "CG differs
                  from Newton" and "CG is under-converged" look identical from
                  the outside: Newton converges quadratically near the solution
                  and CG does not, so matching iteration counts compares
                  CONFIGURATIONS, not solvers. If cg_iter30 agrees and cg does
                  not, the answer is "CG, but with more iterations" — which is
                  still far cheaper than Newton.
  hibernation     Newton + islands + hibernation. Note this is not a clean 2x2:
                  hibernation REQUIRES islands and CG (in Genesis 0.4.5)
                  requires them off, so these are three configurations, not two
                  orthogonal factors.

Fairness rules, which matter more than they look
------------------------------------------------
* Every configuration is seeded from the SAME library state by explicit index,
  never through an rng, so they start bit-identical.
* Pre-push and post-push settles run a FIXED number of steps, identical across
  configurations. A convergence-based settle exits at a different step in each,
  which would compare configurations in different states and silently favour
  whichever declares victory soonest. The convergence-based step count is
  recorded separately, as a metric rather than as the schedule.
* Hibernation zeroes the velocities of bodies it freezes
  (``func_hibernate_entity_and_zero_dof_velocities``), which can make a pile
  look settled when it is merely asleep. The post-settle hold measures net
  DISPLACEMENT over further steps, which that trick cannot fake.
* Actions are fixed and deterministic, spanning blade orientations, because push
  cost and contact structure differ 2-9x between broadside and edge-on.

Acceptance criterion, fixed before running
------------------------------------------
A configuration is interchangeable with ``newton`` iff, for every action and
object count:
  (a) its bulk metrics differ from newton by no more than ``newton_eps`` does,
      and
  (b) its penetration is no worse than newton's, allowing the eps spread.
Stated up front so the verdict cannot be rationalised after seeing the numbers.

Usage
-----
From the REPO ROOT::

    python -m tests.scaling_investigation.probe_solver_equivalence
    python -m tests.scaling_investigation.probe_solver_equivalence \
        --n-particles 50 --configs newton newton_eps cg
    python -m tests.scaling_investigation.probe_solver_equivalence \
        --n-particles 200 --configs newton cg --actions broadside
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GENESIS_DIR = REPO_ROOT / "Genesis"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# name -> (rigid_options overrides, action perturbation in metres)
CONFIGS = {
    "newton":        ({}, 0.0),
    "newton_repeat": ({}, 0.0),
    "newton_eps":    ({}, 1e-6),
    "cg":            ({"constraint_solver": "CG", "use_contact_island": False}, 0.0),
    "cg_iter30":     ({"constraint_solver": "CG", "use_contact_island": False,
                       "iterations": 30}, 0.0),
    "hibernation":   ({"use_hibernation": True}, 0.0),
    # THE MISSING CONTROL. `cg` differs from `newton` in two ways at once:
    # solver AND island decomposition. Any difference attributed to CG could
    # equally be caused by running without islands. This isolates that.
    "newton_islands_off": ({"use_contact_island": False}, 0.0),
    # Genesis >=1.2.x only. Included as a direct test of the upstream claim in
    # rigid_solver.py that CG "carries a search history that the moving
    # objective invalidates, leaving friction short" -- with torsional friction
    # enabled there is MORE friction to leave short, so any deficit should grow.
    "cg_islands_on": ({"constraint_solver": "CG"}, 0.0),
    # Torsional friction is ON in basic.yaml as of this branch, so `newton` now
    # includes it. This turns it back OFF, which is the only way to see what
    # enabling it actually did: it resists spin about the contact normal, and
    # for a cube resting on the tray that normal is vertical, so what it really
    # resists is cubes twisting in place under an off-centre blade hit.
    "no_torsional": ({"enable_torsional_friction": False}, 0.0),
    # The solver budget was set to 10/1e-4 on COST evidence alone -- 50
    # iterations measured 1.6-1.8x slower on a broadside push. The fidelity side
    # of that trade was only ever checked on a SETTLE, where the plate is clear
    # of the pile. These price it on a push, against penetration.
    "iter50": ({"iterations": 50, "ls_iterations": 50}, 0.0),
    # Iterations alone may change nothing if tolerance 1e-4 is already met
    # inside 10, so this is the full Genesis default budget.
    "genesis_budget": ({"iterations": 50, "ls_iterations": 50,
                        "tolerance": 1e-6, "ls_tolerance": 0.01}, 0.0),
    # The plate redesign itself, as a candidate to be judged against the
    # reference rather than swapped in blind. Overrides plate.hold_mode, not
    # rigid_options -- see PLATE_OVERRIDES.
    "servo_plate": ({}, 0.0),
    # The descent/lift driving ALONE, with the sweep hold left as-is. Separated
    # from servo_plate because that config changed both at once, so its failure
    # could not be attributed to either half.
    "servo_approach": ({}, 0.0),
}

# name -> (start_xy, stop_xy, yaw). Fixed, not sampled: every configuration must
# see the identical action, and the set spans blade orientation because contact
# structure differs sharply between broadside and edge-on.
ACTIONS = {
    "broadside": ((-0.030, 0.000), (0.030, 0.000), math.pi / 2),
    "edge_on":   ((-0.030, 0.000), (0.030, 0.000), 0.0),
    "offset":    ((-0.030, 0.025), (0.030, 0.025), math.pi / 2),
    "diagonal":  ((-0.025, -0.025), (0.025, 0.025), math.pi / 4),
}

# config name -> plate:{} overrides, for candidates that change the tool model
# rather than the solver.
PLATE_OVERRIDES = {
    "servo_plate": {"hold_mode": "servo"},
    "servo_approach": {"approach_mode": "servo"},
}

PRE_SETTLE_STEPS = 50     # fixed, identical across configs
POST_SETTLE_STEPS = 300   # fixed, identical across configs
HOLD_STEPS = 50           # after the fixed settle, to catch faked rest


_config_name = [""]      # set by run_cell; keeps _config's signature stable


def _config(n_particles, particle_size, overrides, hold_mode=None):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    cfg.setdefault("rigid_options", {}).update(overrides)
    for k, v in PLATE_OVERRIDES.get(_config_name[0], {}).items():
        cfg.setdefault("plate", {})[k] = v
    if hold_mode:
        # Applied to EVERY config in the run, including the reference and the
        # noise-floor replicas. hold_mode is not the variable under test here --
        # it is the platform the test runs on, and hibernation cannot run on
        # `pinned` at all (it NaNs), so a comparison that left the reference
        # pinned would be comparing two different plate models at once.
        cfg.setdefault("plate", {})["hold_mode"] = hold_mode
    return cfg


def link_classes(sim):
    """Link indices split into particle / plate / static (floor + tray walls).

    Needed because "penetration" means very different things depending on what
    is touching. An early version of this probe classified everything that was
    not the plate as particle-particle, and duly reported a 97 mm particle-
    particle overlap — geometrically impossible for 5 mm cubes, and the tell
    that the classification, not the solver, was wrong. Deep penetration into
    an infinite floor plane or a tray wall is possible; deep penetration
    between two cubes is not.
    """
    particles = {int(p.links[0].idx) for p in sim.material}
    plate = {int(sim.plate.links[0].idx)}
    return particles, plate


def contact_quality(sim, classes):
    """Penetration and force at the current contacts, split by what is touching.

    Penetration is the discriminator this probe is built around: it describes
    the state rather than the trajectory, so unlike particle positions it stays
    comparable after two chaotic runs have diverged.
    """
    import torch
    try:
        c = sim._scene.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)
    except Exception as e:
        return {"available": False, "error": str(e)[:120]}
    pen = c.get("penetration")
    if pen is None or pen.numel() == 0:
        return {"available": True, "n": 0}
    pen = pen.flatten().float()
    la, lb = c["link_a"].flatten(), c["link_b"].flatten()
    force = c.get("force")

    particles, plate = classes
    pset = torch.tensor(sorted(particles), device=la.device)
    a_is_p, b_is_p = torch.isin(la, pset), torch.isin(lb, pset)
    plate_idx = next(iter(plate))
    is_plate = (la == plate_idx) | (lb == plate_idx)
    # particle-particle: both ends are particles. particle-static: exactly one
    # end is a particle and the other is neither particle nor plate (floor/wall).
    pp = a_is_p & b_is_p
    ps = (a_is_p ^ b_is_p) & ~is_plate
    pl = is_plate

    def _stats(mask):
        p = pen[mask]
        if p.numel() == 0:
            return {"n": 0}
        d = {"n": int(p.numel()), "max_mm": float(p.max()) * 1000,
             "mean_mm": float(p.mean()) * 1000}
        if force is not None and force.numel():
            f = force.reshape(-1, force.shape[-1])[mask].norm(dim=-1)
            d["force_max_N"] = float(f.max())
        return d

    out = {"available": True, "n": int(pen.numel()),
           "particle": _stats(pp), "static": _stats(ps), "plate": _stats(pl)}
    # On a gross event, record what was actually touching — the classification
    # is the whole question when a penetration looks impossible.
    worst = int(pen.argmax())
    if float(pen[worst]) > 5e-3:
        out["worst"] = {"pen_mm": float(pen[worst]) * 1000,
                        "link_a": int(la[worst]), "link_b": int(lb[worst]),
                        "a_is_particle": bool(a_is_p[worst]),
                        "b_is_particle": bool(b_is_p[worst]),
                        "involves_plate": bool(is_plate[worst])}
    return out


def run_cell(n_particles, config_name, action_name, particle_size, library_root,
             library_index, hold_mode=None):
    import numpy as np
    import torch
    from Genesis.sandbox_manipulation_clean import SandboxManipulation
    from Genesis.state_library import StateLibrary, default_library_path

    overrides, perturb = CONFIGS[config_name]
    start_xy, stop_xy, yaw = ACTIONS[action_name]
    torch.manual_seed(0)
    np.random.seed(0)
    out = {"n_particles": n_particles, "config": config_name,
           "action": action_name, "overrides": overrides, "perturb_m": perturb}

    out["hold_mode"] = hold_mode
    _config_name[0] = config_name
    sim = SandboxManipulation(
        config=_config(n_particles, particle_size, overrides, hold_mode),
        n_envs=1, debug=False)
    sim.build()

    try:
        lib_path = default_library_path(GENESIS_DIR / library_root, "cube",
                                        n_particles, particle_size)
        if not lib_path.exists():
            raise FileNotFoundError(f"no settled-state library at {lib_path}")
        # Explicit index, never an rng: every configuration must start from a
        # bit-identical state or nothing downstream is comparable.
        StateLibrary.load(lib_path).apply(sim, index=library_index)

        classes = link_classes(sim)
        n_active = getattr(sim, "_n_active", None)

        # ---- fixed pre-settle (NOT convergence-based; see module docstring) --
        frozen = sim.plate.get_dofs_position()
        sim.plate.zero_all_dofs_velocity()
        sim.plate.control_dofs_position_velocity(
            frozen, torch.zeros_like(frozen), dofs_idx_local=[0, 1, 2, 3, 4, 5])
        for _ in range(PRE_SETTLE_STEPS):
            sim._step_scene()
        pre = sim._get_particle_positions()[0, :n_active].clone()
        out["pre_push_positions_mm"] = (pre * 1000).tolist()

        # ---- the push, with contact quality sampled throughout --------------
        dev = sim._particle_state.device
        h = sim._operation_height
        p_start = torch.tensor([[start_xy[0] + perturb, start_xy[1], h]], device=dev)
        p_stop = torch.tensor([[stop_xy[0] + perturb, stop_xy[1], h]], device=dev)
        angle = torch.full((1,), yaw, device=dev)

        samples = []
        extremes = []

        def on_step(step, *_):
            if step % 5 == 0:
                samples.append(contact_quality(sim, classes))
                # Correlate any gross contact reading with where the particles
                # actually ARE. A 97 mm penetration against static geometry is
                # either a particle genuinely driven into the floor/wall — a real
                # integrity failure — or a spurious value from the collider. The
                # positions decide it, and nothing else measured here can.
                p = sim._get_particle_positions()[0, :n_active]
                extremes.append({
                    "step": step,
                    "min_z_mm": float(p[:, 2].min()) * 1000,
                    "max_xy_mm": float(p[:, :2].abs().max()) * 1000,
                    "worst_pen_mm": samples[-1].get("static", {}).get("max_mm", 0.0),
                })

        t0 = time.perf_counter()
        sim.plate_position_translation(
            p_start + sim._clearance_offset, p_start, sim._clearance_ctrl_steps)
        reached, final_pos = sim.plate_velocity_translation(
            p_start, p_stop, angle, on_step=on_step)
        torch.cuda.synchronize()
        out["push_s"] = time.perf_counter() - t0
        out["reached_goal"] = bool(reached[0])
        out["tracking_err_mm"] = float(
            (final_pos[0, :2] - p_stop[0, :2]).norm()) * 1000

        valid = [s for s in samples if s.get("available") and s.get("n")]
        # Keep the whole series, not just its peak: a peak alone cannot say
        # whether a bad penetration was one transient step or a sustained
        # failure to resolve contacts, and those mean very different things.
        out["pen_particle_series_mm"] = [
            s.get("particle", {}).get("max_mm", 0.0) for s in valid]
        out["worst_events"] = [s["worst"] for s in samples if s.get("worst")]
        # Tray floor is z=0 and walls are at +/-64 mm; a real excursion shows as
        # min_z well below 0 or max_xy well beyond 64.
        out["min_z_mm"] = min((e["min_z_mm"] for e in extremes), default=0.0)
        out["max_xy_mm"] = max((e["max_xy_mm"] for e in extremes), default=0.0)
        out["extremes_at_events"] = [e for e in extremes if e["worst_pen_mm"] > 5.0]
        for kind in ("particle", "static", "plate"):
            pk = [s[kind] for s in valid if s.get(kind, {}).get("n")]
            out[f"pen_{kind}_max_mm"] = max((d["max_mm"] for d in pk), default=0.0)
            out[f"pen_{kind}_mean_mm"] = (
                sum(d["mean_mm"] for d in pk) / len(pk) if pk else 0.0)
            out[f"force_{kind}_max_N"] = max(
                (d.get("force_max_N", 0.0) for d in pk), default=0.0)
        out["contacts_peak"] = max((s["n"] for s in valid), default=0)

        # ---- fixed post-settle, then a hold that hibernation cannot fake ----
        frozen = sim.plate.get_dofs_position()
        sim.plate.zero_all_dofs_velocity()
        sim.plate.control_dofs_position_velocity(
            frozen, torch.zeros_like(frozen), dofs_idx_local=[0, 1, 2, 3, 4, 5])
        # Where a convergence-based settle WOULD have stopped, recorded as a
        # metric rather than used as the schedule.
        converged_at = None
        for step in range(POST_SETTLE_STEPS):
            sim._step_scene()
            if converged_at is None and (step + 1) % sim._settle_check_every == 0 \
                    and sim._pile_is_at_rest():
                converged_at = step + 1
        out["would_converge_at"] = converged_at

        post = sim._get_particle_positions()[0, :n_active].clone()
        out["post_positions_mm"] = (post * 1000).tolist()

        # How much of the pile is asleep at s'. The hold test below is the usual
        # way to tell real rest from faked rest, but it is BLIND to hibernation
        # by construction: a frozen body does not drift, so "no drift" no longer
        # proves "at rest", it may just mean "still frozen". Bulk transport
        # becomes the real detector, and this number says how much of the pile
        # was excluded from the solve when s' was recorded.
        try:
            hib = sim._scene.rigid_solver.dyn_state.links.is_hibernated.to_numpy()
            out["hibernated_links"] = int(hib.reshape(hib.shape[0], -1)[:, 0].sum())
        except Exception:
            out["hibernated_links"] = None

        for _ in range(HOLD_STEPS):
            sim._step_scene()
        held = sim._get_particle_positions()[0, :n_active]
        drift = (held - post).norm(dim=-1)
        out["hold_drift_max_mm"] = float(drift.max()) * 1000
        out["hold_drift_mean_mm"] = float(drift.mean()) * 1000

        # ---- bulk outcome of the push --------------------------------------
        disp = (post - pre).norm(dim=-1)
        out["com_shift_mm"] = float((post.mean(0) - pre.mean(0)).norm()) * 1000
        out["displaced_mass_mm"] = float(disp.sum()) * 1000
        out["max_disp_mm"] = float(disp.max()) * 1000
        out["n_moved_1mm"] = int((disp > 1e-3).sum())
        out["spread_mm"] = float(post[:, :2].std(dim=0).mean()) * 1000
        out["escaped"] = sim.escaped_particle_count()
        out["ok"] = True
    except Exception as e:
        import traceback
        out.update(ok=False, error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[-1200:])
    finally:
        try:
            sim.destroy()
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def divergence(a, b, key="post_positions_mm"):
    """Mean and max per-particle distance between two runs' final states, mm."""
    import numpy as np
    pa, pb = np.array(a[key]), np.array(b[key])
    if pa.shape != pb.shape:
        return None
    d = np.linalg.norm(pa - pb, axis=-1)
    return {"mean_mm": float(d.mean()), "max_mm": float(d.max())}


def report(cells):
    by = {}
    for c in cells:
        if c.get("ok"):
            by[(c["n_particles"], c["action"], c["config"])] = c

    keys = sorted({(n, a) for n, a, _ in by})
    for n, action in keys:
        ref = by.get((n, action, "newton"))
        print("\n" + "=" * 100)
        print(f"n={n}  action={action}" + ("" if ref else "   (no newton reference)"))
        print("=" * 100)
        print(f"{'config':<15}{'push s':>8}{'pen par':>9}{'pen plate':>10}"
              f"{'COM mm':>8}{'moved':>7}{'displ mm':>10}{'hold mm':>9}"
              f"{'conv':>6}{'esc':>5}{'diverge mm':>12}")
        for cfg in CONFIGS:
            c = by.get((n, action, cfg))
            if c is None:
                continue
            dv = divergence(ref, c) if (ref and cfg != "newton") else None
            print(f"{cfg:<15}{c['push_s']:>8.1f}"
                  f"{c['pen_particle_max_mm']:>9.3f}{c['pen_plate_max_mm']:>10.3f}"
                  f"{c['com_shift_mm']:>8.2f}{c['n_moved_1mm']:>7}"
                  f"{c['displaced_mass_mm']:>10.0f}{c['hold_drift_max_mm']:>9.3f}"
                  f"{str(c['would_converge_at'] or '-'):>6}{c['escaped']:>5}"
                  + (f"{dv['mean_mm']:>12.2f}" if dv else f"{'—':>12}"))

    verdict_table(by, keys)
    if any(c.get("replicate", 0) for c in cells):
        distribution_table(cells)


def distribution_table(cells):
    """Compare configurations as DISTRIBUTIONS, not as points.

    Necessary because Genesis is not bit-deterministic: a single run per cell is
    one draw, and a 3.4 mm penetration difference was observed once and then
    failed to reproduce on an identical rerun. Point comparisons at the few-
    percent level are therefore not trustworthy, however carefully the noise
    floor is estimated from one sample.

    Reports mean and half-range per configuration and flags a difference only
    when the intervals do not overlap — crude with a handful of replicates, but
    it cannot manufacture a difference out of run-to-run scatter, which is the
    failure mode that matters here.
    """
    runs = {}
    for c in cells:
        if c.get("ok"):
            runs.setdefault((c["n_particles"], c["action"], c["config"]), []).append(c)

    def interval(rs, key):
        v = [r[key] for r in rs]
        return (sum(v) / len(v), (max(v) - min(v)) / 2, len(v))

    print("\n" + "#" * 100)
    print("# DISTRIBUTIONS over replicates (mean +/- half-range); "
          "'DIFFERS' only when intervals do not overlap")
    print("#" * 100)
    for n, action in sorted({(k[0], k[1]) for k in runs}):
        ref = runs.get((n, action, "newton"))
        if not ref or len(ref) < 2:
            continue
        print(f"\n  n={n} {action}")
        for metric, label in (("com_shift_mm", "COM mm"),
                              ("displaced_mass_mm", "displaced mm"),
                              ("pen_particle_max_mm", "penetration mm")):
            rm, rh, rn = interval(ref, metric)
            line = f"    {label:<16} newton {rm:8.2f} +/-{rh:6.2f} (n={rn})"
            for cfg in _candidates(runs, n):
                rs = runs.get((n, action, cfg))
                if not rs:
                    continue
                cm, ch, cn = interval(rs, metric)
                overlap = abs(cm - rm) <= (rh + ch) or abs(cm - rm) <= REL_TOL * abs(rm)
                line += (f"   {cfg} {cm:8.2f}+/-{ch:5.2f}"
                         + ("" if overlap else " DIFFERS"))
            print(line)


REL_TOL = 0.02          # 2 % of the reference value

# The reference and the two known-identical replicas that define the noise
# floor; everything else present in a run is a candidate to be judged.
_FLOOR_CONFIGS = ("newton", "newton_repeat", "newton_eps")


def _candidates(keyed, n):
    """Configs actually present for this particle count, minus the reference and
    its noise-floor replicas. Derived from the data rather than hardcoded: an
    earlier version listed the candidates literally and silently omitted every
    config added afterwards from the report, having measured them all."""
    return [c for c in dict.fromkeys(k[2] for k in keyed if k[0] == n)
            if c not in _FLOOR_CONFIGS]


def verdict_table(by, keys):
    """Apply the acceptance criterion, with the amendment below.

    The criterion as first written — "differ by no more than the noise floor
    does" — is DEGENERATE, and the first full run exposed it. In gentle cells
    (edge_on, diagonal) the pile barely moves, chaos never gets going, and both
    noise-floor replicas reproduce the reference exactly. The floor is then
    0.000, so the criterion silently becomes "must be bit-identical" and flags a
    0.001 mm penetration difference as a physics mismatch.

    Two fixes, neither of which relaxes the intent:

    * **Pool the floor across actions** at a given particle count. A per-cell
      floor is a one-sample estimate of a random quantity — it came out 0.00 in
      one cell and 0.37 mm in another *for the same physics*. The largest
      deviation any known-identical configuration produces at this n is the
      honest estimate of what identical physics can look like.
    * **Add a relative tolerance** (2 % of the reference). Below that, a
      difference is not physically meaningful for a dataset regardless of how
      quiet the replicas happened to be.

    The threshold is therefore max(pooled floor, 2 % of reference), and the
    verdict reports the actual percentages so it can be second-guessed.
    """
    for n in sorted({k[0] for k in keys}):
        actions = [a for (nn, a) in keys if nn == n]
        # pooled floor: the worst any known-identical configuration deviates
        f_com = f_displ = f_pen = 0.0
        for a in actions:
            ref = by.get((n, a, "newton"))
            if not ref:
                continue
            for name in ("newton_repeat", "newton_eps"):
                c = by.get((n, a, name))
                if not c:
                    continue
                f_com = max(f_com, abs(c["com_shift_mm"] - ref["com_shift_mm"]))
                f_displ = max(f_displ, abs(c["displaced_mass_mm"]
                                           - ref["displaced_mass_mm"]))
                f_pen = max(f_pen, abs(c["pen_particle_max_mm"]
                                       - ref["pen_particle_max_mm"]))
        print("\n" + "#" * 100)
        print(f"# VERDICT, n={n}   pooled noise floor: COM {f_com:.2f} mm, "
              f"displaced {f_displ:.0f} mm, penetration {f_pen:.3f} mm "
              f"(+{REL_TOL*100:.0f} % relative)")
        print("#" * 100)
        print(f"{'config':<14}{'action':<11}{'dCOM %':>9}{'dDispl %':>10}"
              f"{'dPen mm':>10}{'hold mm':>9}  verdict")
        for cfg in _candidates(by, n):
            for a in actions:
                ref, c = by.get((n, a, "newton")), by.get((n, a, cfg))
                if not ref or not c:
                    continue
                d_com = abs(c["com_shift_mm"] - ref["com_shift_mm"])
                d_displ = abs(c["displaced_mass_mm"] - ref["displaced_mass_mm"])
                d_pen = c["pen_particle_max_mm"] - ref["pen_particle_max_mm"]
                ok_com = d_com <= max(f_com, REL_TOL * abs(ref["com_shift_mm"]))
                ok_displ = d_displ <= max(f_displ,
                                          REL_TOL * abs(ref["displaced_mass_mm"]))
                # only WORSE penetration counts against a config
                ok_pen = d_pen <= max(f_pen, REL_TOL * ref["pen_particle_max_mm"])
                bad = [x for x, b in (("COM", not ok_com), ("displaced", not ok_displ),
                                      ("PENETRATION", not ok_pen)) if b]
                pct = lambda v, r: (100 * v / r) if r else 0.0
                print(f"{cfg:<14}{a:<11}"
                      f"{pct(d_com, ref['com_shift_mm']):>9.1f}"
                      f"{pct(d_displ, ref['displaced_mass_mm']):>10.1f}"
                      f"{d_pen:>+10.3f}{c['hold_drift_max_mm']:>9.3f}  "
                      + ("ok" if not bad else "DIFFERS: " + ", ".join(bad)))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-particles", nargs="+", type=int, default=[50, 100])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   choices=list(CONFIGS))
    p.add_argument("--actions", nargs="+", default=list(ACTIONS),
                   choices=list(ACTIONS))
    p.add_argument("--particle-size", type=float, default=0.005)
    p.add_argument("--library-root", default="data/dry_run")
    p.add_argument("--library-index", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--hold-mode", choices=["pinned", "servo"], default=None,
                   help="plate.hold_mode for EVERY config in the run. Required "
                        "as 'servo' to test hibernation at all, since "
                        "hibernation produces NaN under 'pinned'.")
    p.add_argument("--replicates", type=int, default=1,
                   help="run each configuration N times. Genesis is not "
                        "bit-deterministic, so a single run per cell is one "
                        "draw from a distribution — enough to spot a gross "
                        "mismatch, not enough to resolve a few-percent one. "
                        "With N>1 the report compares distributions.")
    p.add_argument("--report-only", action="store_true",
                   help="re-report from saved results without re-running; the "
                        "measurements are expensive and the analysis is not")
    p.add_argument("--cell", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()

    if args.report_only:
        out = Path(args.out) if args.out else RESULTS_DIR / "solver_equivalence.json"
        report(json.load(open(out)))
        return

    if args.cell:
        n, cfg, action, rep = args.cell.split(":")
        res = run_cell(int(n), cfg, action, args.particle_size,
                       args.library_root, args.library_index, args.hold_mode)
        res["replicate"] = int(rep)
        print("__RESULT__" + json.dumps(res))
        return

    out = Path(args.out) if args.out else RESULTS_DIR / "solver_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for n in args.n_particles:
        for action in args.actions:
            for cfg in args.configs:
              for rep in range(args.replicates):
                cell = f"{n}:{cfg}:{action}:{rep}"
                print(f">>> {cell}", flush=True)
                proc = subprocess.run(
                    [sys.executable, "-m",
                     "tests.scaling_investigation.probe_solver_equivalence",
                     "--cell", cell, "--particle-size", str(args.particle_size),
                     "--library-root", args.library_root,
                     "--library-index", str(args.library_index)]
                    + (["--hold-mode", args.hold_mode] if args.hold_mode else []),
                    cwd=REPO_ROOT, capture_output=True, text=True)
                line = next((l for l in proc.stdout.splitlines()
                             if l.startswith("__RESULT__")), None)
                if line:
                    results.append(json.loads(line[len("__RESULT__"):]))
                else:
                    results.append({"n_particles": n, "config": cfg,
                                    "action": action, "replicate": rep,
                                    "ok": False,
                                    "error": (proc.stderr or proc.stdout)[-400:]})
                    print(f"    FAILED: {results[-1]['error'][-200:]}", flush=True)
                out.write_text(json.dumps(results, indent=2))
    report(results)
    print(f"\nraw results -> {out}")


if __name__ == "__main__":
    main()
