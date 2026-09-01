"""Size a dense-pile collection before committing a night to it.

The H1 continuum-limit run died with CUDA OOM because the constraint Jacobian is
O(max_collision_pairs x contacts x n_dofs x n_envs), and going from 30 to 150
particles raises n_dofs 5x while a dense pile needs a much larger contact budget
than the scattered-tuned default of max(150, n/2). Guessing that budget is
dangerous in the other direction too: past the cap Genesis silently stops adding
contacts and the recorded state comes from incomplete physics with no error.

So this measures rather than guesses. It builds the scene, settles a pile,
executes a few real pushes, and reports `contact_budget_usage()` -- peak
broad-phase pairs and contact points against their actual caps -- plus wall-clock
per transition, so both the memory ceiling and the throughput can be read off
before scheduling.

    python scripts/probe_dense_pile.py --n 80 --size 0.003 --envs 4 --mcp 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Run as a plain script (`python scripts/probe_dense_pile.py`) and Python puts
# scripts/ on sys.path, not the repo root, so `import Genesis` fails.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="particles")
    ap.add_argument("--size", type=float, default=0.003, help="cube edge, m")
    ap.add_argument("--density", type=float, default=4000.0,
                    help="kg/m^3. 3 mm cubes at the default 1000 have a mass "
                         "Genesis flags as too small for solver stability; 4000 "
                         "puts them alongside the 5 mm cubes at 1000 that every "
                         "earlier dataset used, and is inside this project's "
                         "750-5000 normalisation range.")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--mcp", type=int, default=400, help="max_collision_pairs")
    ap.add_argument("--pile-extent", type=float, default=0.016)
    ap.add_argument("--pushes", type=int, default=3)
    ap.add_argument("--debug", action="store_true",
                   help="verbose sim logs. NOTE: SandboxManipulation._step_scene "
                        "treats debug as 'show viewer' (_show = debug or "
                        "viewer_type is not None), so this renders at 2880x2160 "
                        "every step and destroys the throughput measurement -- "
                        "measured 627 s/transition with it on. Off by default "
                        "for exactly that reason; only pass it to debug geometry, "
                        "never to time anything.")
    ap.add_argument("--push-length", type=float, default=0.02)
    args = ap.parse_args()

    import yaml
    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    cfg = yaml.safe_load(open("Genesis/configs/basic.yaml"))
    cfg["material"]["shape"] = "cube"
    cfg["material"]["particle_size"] = args.size
    cfg["material"]["n_particles"] = args.n
    cfg["material"]["particle_friction"] = 0.3
    cfg["material"]["density"] = args.density
    cfg["box"]["friction"] = 0.3
    cfg.setdefault("rigid_options", {})["max_collision_pairs"] = args.mcp
    cfg["spawn"] = {"pile_extent": args.pile_extent, "pile_layers": None}

    print(f"probe: n={args.n} size={args.size*1000:.0f}mm density={args.density} "
          f"envs={args.envs} mcp={args.mcp} extent={args.pile_extent}", flush=True)

    t0 = time.time()
    sim = SandboxManipulation(config=cfg, n_envs=args.envs, debug=args.debug,
                              viewer_type=None)
    sim.build()
    print(f"  build: {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    sim.shuffle_particles()
    sim.update_material_state()
    print(f"  spawn+settle: {time.time()-t1:.0f}s", flush=True)

    def report(tag):
        u = sim.contact_budget_usage()
        bp, bc = u["broad_pairs"], u["broad_cap"]
        cp, cc = u["contact_points"], u["contact_cap"]
        flag = lambda a, b: "OVERFLOW" if a >= b else ("tight" if a >= 0.9 * b else "ok")
        print(f"  [{tag}] broad {bp}/{bc} ({100*bp/max(bc,1):.0f}%, {flag(bp,bc)})  "
              f"points {cp}/{cc} ({100*cp/max(cc,1):.0f}%, {flag(cp,cc)})", flush=True)
        return u

    report("settled")

    # Real pushes through the normal pile-aware path.
    peak = {"broad_pairs": 0, "contact_points": 0}
    t2 = time.time()
    for i in range(args.pushes):
        starts, stops, angles = sim.generate_action_samples(
            1, pile_aware=True, push_length=args.push_length,
            min_swath_particles=3)
        sim.execute_action(starts[:, 0, :], stops[:, 0, :], angles[:, 0])
        sim.update_material_state()
        u = report(f"push {i+1}")
        for k in peak:
            peak[k] = max(peak[k], u[k])
    dt = time.time() - t2
    n_trans = args.pushes * args.envs

    st = sim._particle_state
    z = st[..., 2]
    z0 = float(z.reshape(-1).quantile(0.02))
    layers = ((z - z0) / args.size).round()
    print(f"\n  pile: z span {1000*float(z.max()-z.min()):.1f} mm, "
          f"layer occupancy " +
          ", ".join(f"L{L}={100*float((layers==L).float().mean()):.0f}%"
                    for L in range(5)
                    if float((layers == L).float().mean()) > 0.02))
    xy = st[..., :2].reshape(-1, 2)
    print(f"  footprint {1000*float(xy.max(0).values.sub(xy.min(0).values).max()):.0f} mm")

    print(f"\n  THROUGHPUT: {dt:.0f}s for {n_trans} transitions = "
          f"{dt/max(n_trans,1):.1f} s/transition")
    print(f"  -> 6 h would yield ~{int(6*3600/max(dt/max(n_trans,1),1e-9))} transitions")
    print(f"  PEAK BUDGET: broad {peak['broad_pairs']}, points {peak['contact_points']}")
    print(f"  headroom suggests mcp could be "
          f"{'RAISED' if peak['contact_points'] > 0.8*report('final')['contact_cap'] else 'kept or lowered'}")
    sim.destroy()


if __name__ == "__main__":
    main()
