#!/usr/bin/env python3
"""
tests/scaling_investigation/verify_new_features.py — live-scene checks for the two collection
features added alongside the 200-object scaling work:

  * the settled-state library (Genesis/state_library.py) — reset by restoring a
    pre-settled arrangement instead of re-shuffling and re-settling
  * placement-aware action sampling (Genesis/placement_sampling.py) — draw the
    tool's touchdown pose from its free configuration space

The unit tests in tests/ cover the maths without a GPU; this covers the parts
that only exist against a built Genesis scene: that a restored state really is
at rest, that the library is a real speedup, and that placement-aware sampling
actually stops the plate landing on a particle (and degrades instead of failing
when it cannot).

Run from the REPO ROOT::

    python -m tests.scaling_investigation.verify_new_features
    python -m tests.scaling_investigation.verify_new_features --n-particles 200 --n-envs 4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Genesis.sandbox_manipulation_clean import SandboxManipulation
from Genesis.state_library import build_state_library, StateLibrary, box_symmetries

# This script lives outside Genesis/, so paths to the simulator's configs are
# resolved explicitly rather than relative to this file.
GENESIS_DIR = Path(__file__).resolve().parents[2] / "Genesis"
REPO_ROOT = Path(__file__).resolve().parents[2]


_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"{'  PASS' if ok else '  FAIL'}  {name}" + (f"  --  {detail}" if detail else ""),
          flush=True)


def _config(n_particles, particle_size):
    with open(GENESIS_DIR / "configs" / "basic.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["material"].update(shape="cube", particle_size=particle_size,
                           n_particles=n_particles, density=1000.0, friction=0.3)
    cfg["box"]["friction"] = 0.3
    cfg.setdefault("data_collection", {})["record_transitions"] = False
    return cfg


def _pile_motion(sim):
    """Peak (linear m/s, angular rad/s). Kept separate: a free joint's dofs are
    [x,y,z,roll,pitch,yaw], so a single max over all six conflates m/s with
    rad/s and makes a mildly spinning cube look like a 24 m/s projectile."""
    return sim._pile_motion()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-particles", type=int, default=200)
    ap.add_argument("--particle-size", type=float, default=0.005)
    ap.add_argument("--n-envs", type=int, default=2)
    ap.add_argument("--n-settles", type=int, default=3)
    args = ap.parse_args()

    sim = SandboxManipulation(config=_config(args.n_particles, args.particle_size),
                              n_envs=args.n_envs, debug=False)
    try:
        sim.build()

        # ---------------- state library ----------------
        print(f"\n--- settled-state library ({args.n_settles} settles, "
              f"n_envs={args.n_envs}) ---")
        t0 = time.perf_counter()
        lib = build_state_library(sim, n_settles=args.n_settles, verbose=False)
        t_build = time.perf_counter() - t0

        expected = args.n_settles * args.n_envs * len(box_symmetries(sim._box_params["vol"]))
        check("library size = settles x envs x symmetries",
              len(lib) == expected,
              f"{args.n_settles} x {args.n_envs} x "
              f"{len(box_symmetries(sim._box_params['vol']))} = {len(lib)}")

        check("library particle count matches the scene",
              lib.n_particles == len(sim.material),
              f"{lib.n_particles} particles")

        # every stored state must sit inside the tray
        half_x = sim._box_params["vol"][0] / 2
        half_y = sim._box_params["vol"][1] / 2
        inside = (lib.states[..., 0].abs() <= half_x + 1e-3).all() and \
                 (lib.states[..., 1].abs() <= half_y + 1e-3).all()
        check("all library states lie inside the tray", bool(inside),
              f"|x|max={float(lib.states[...,0].abs().max()):.4f} "
              f"|y|max={float(lib.states[...,1].abs().max()):.4f} m")

        # quaternions must stay unit after the symmetry transforms
        qn = lib.states[..., 3:7].norm(dim=-1)
        check("library quaternions are unit", bool((qn - 1).abs().max() < 1e-4),
              f"max deviation {float((qn-1).abs().max()):.2e}")

        # THE point of the library: a restored state is already at rest, so no
        # settle is needed. Restore, step a little, and confirm nothing moves.
        # Baseline: how much does a FRESHLY settled pile drift? A restored
        # state only has to be as quiet as that -- an absolute threshold would
        # be measuring the settle criterion, not the restore.
        sim.shuffle_particles()
        sim.update_material_state()
        fresh_before = sim._get_particle_positions().clone()
        for _ in range(20):
            sim._step_scene()
        fresh_drift = float((sim._get_particle_positions() - fresh_before).abs().max())

        lib.apply(sim, index=0)
        pos_before = sim._get_particle_positions().clone()
        for _ in range(20):
            sim._step_scene()
        drift = float((sim._get_particle_positions() - pos_before).abs().max())
        lin, ang = _pile_motion(sim)
        check("restored state is as quiet as a freshly settled one",
              drift <= max(fresh_drift * 1.5, 5e-4),
              f"restored drifts {drift*1000:.3f} mm over 20 steps vs "
              f"{fresh_drift*1000:.3f} mm for a fresh settle; "
              f"peak {lin*1000:.1f} mm/s linear, {ang:.1f} rad/s angular")

        # speed: restore vs shuffle+settle
        t0 = time.perf_counter(); lib.apply(sim, index=1); torch.cuda.synchronize()
        t_restore = time.perf_counter() - t0
        t0 = time.perf_counter(); sim.shuffle_particles(); sim.update_material_state()
        torch.cuda.synchronize()
        t_shuffle = time.perf_counter() - t0
        check("restore is much cheaper than shuffle+settle",
              t_restore < t_shuffle / 5,
              f"restore {t_restore*1000:.0f} ms vs shuffle+settle "
              f"{t_shuffle*1000:.0f} ms ({t_shuffle/max(t_restore,1e-9):.0f}x); "
              f"library build cost {t_build:.1f} s once")

        # save/load round-trip against a real library
        out = Path("/tmp/state_lib_check"); saved = lib.save(out)
        reloaded = StateLibrary.load(saved)
        check("library saves and reloads",
              len(reloaded) == len(lib)
              and torch.allclose(reloaded.states, lib.states.cpu()),
              f"{saved} ({saved.stat().st_size/2**20:.1f} MiB)")

        # ---------------- placement-aware sampling ----------------
        print("\n--- placement-aware action sampling ---")
        sim.shuffle_particles()
        sim.update_material_state()

        n_samples = 8
        blind_starts, _, blind_ang = sim.generate_action_samples(n_samples)
        aware_starts, _, aware_ang = sim.generate_action_samples(
            n_samples, placement_aware=True)

        check("placement-aware sampling returns the same shapes",
              aware_starts.shape == blind_starts.shape
              and aware_ang.shape == blind_ang.shape,
              f"{tuple(aware_starts.shape)}")

        # Count how often the tool footprint would land on a particle.
        sizes = sim._sampled_params.get("particle_sizes")
        half = torch.as_tensor(sizes, dtype=torch.float32,
                               device=aware_starts.device)[:, :2] * 0.5
        n_active = getattr(sim, "_n_active", len(sim.material))
        pos = sim._particle_state[:, :n_active, 0:2]
        tool_l, tool_w, _ = sim._plate_params["size"]

        def overlap_rate(starts, angles):
            hits = 0
            for e in range(starts.shape[0]):
                for s in range(starts.shape[1]):
                    c, ang = starts[e, s, :2], float(angles[e, s])
                    d = pos[e] - c
                    ca, sa = np.cos(-ang), np.sin(-ang)
                    u = ca * d[:, 0] - sa * d[:, 1]
                    v = sa * d[:, 0] + ca * d[:, 1]
                    hit = ((u.abs() <= tool_l / 2 + half[:n_active, 0]) &
                           (v.abs() <= tool_w / 2 + half[:n_active, 1]))
                    hits += int(hit.any())
            return hits / (starts.shape[0] * starts.shape[1])

        blind_rate = overlap_rate(blind_starts, blind_ang)
        aware_rate = overlap_rate(aware_starts, aware_ang)
        check("placement-aware lands on fewer particles than blind",
              aware_rate <= blind_rate,
              f"blind {blind_rate*100:.0f}% of touchdowns overlap a particle, "
              f"placement-aware {aware_rate*100:.0f}%")

        # Degradation: with the tray fully covered there is no free placement,
        # and sampling must fall back rather than fail.
        saved_state = sim._particle_state.clone()
        flooded = saved_state.clone()
        g = torch.linspace(-0.05, 0.05, 16, device=flooded.device)
        gx, gy = torch.meshgrid(g, g, indexing="ij")
        flat = torch.stack((gx.flatten(), gy.flatten()), dim=-1)
        k = min(flat.shape[0], flooded.shape[1])
        flooded[:, :k, 0:2] = flat[:k].unsqueeze(0)
        sim.set_particle_state(flooded[0:1, :, 0:3], flooded[0:1, :, 3:7])
        try:
            fs, _, fa = sim.generate_action_samples(4, placement_aware=True)
            ok = fs.shape == (sim._n_envs, 4, 3) and torch.isfinite(fs).all()
        except Exception as e:
            ok = False
            print(f"      raised: {e}")
        check("falls back to blind sampling when no free placement exists", ok,
              "returned finite blind samples instead of raising")
        sim.set_particle_state(saved_state[0:1, :, 0:3], saved_state[0:1, :, 3:7])

    finally:
        sim.destroy()

    n_pass = sum(1 for _, ok in _results if ok)
    print(f"\n=== {n_pass}/{len(_results)} checks passed ===")
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
