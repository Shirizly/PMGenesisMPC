"""Can we get a pile that is actually several layers deep?

Two probes have now shown that dropping stacked layers onto an empty tray yields
a dense MONOLAYER at both particle sizes (90% layer 0 at 30x5mm, 94% at 80x3mm):
the cubes bounce outward before they rest. Depth is the untested half of both H1
(continuum limit) and H2 (2-D vs 3-D), so it needs a different mechanism.

This tries the two cheapest ideas, on the smallest particle counts that could
physically support the depth:

  friction   raise particle/box friction so cubes grip instead of sliding apart.
             Nothing else changes -- if this alone gives depth it is free.
  pyramid    place the cubes AS a stable stepped pyramid instead of dropping
             layers, with a small jitter. A pyramid is already at rest under
             gravity, so it has no bounce energy to spread with. Two readings:
             how it settles on its own, and how it settles after one push.

Counts are the minimum that can support the target depth, not more:
a k-layer square pyramid with a 1-cube top needs sum of (2i-1)^2. That is 1, 10,
35, 84 for 1..4 layers, so ~35 cubes support 3 layers and ~84 support 4. We test
50 (asked: can it work at all?) and 80 (the cap).

    python scripts/probe_pile_depth.py
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def pyramid_positions(n, size, gap=1.15):
    """Stepped square pyramid using ALL n cubes, densest layer at the bottom.

    Base width is the smallest w whose descending-square stack w^2 + (w-1)^2 +
    ... reaches n; layers are then filled from the base until the cubes run out.
    That maximises depth while parking nothing:

        n=30 -> 16+9+4+1  = 30, four layers
        n=50 -> 25+16+9   = 50, three layers

    Parking the remainder was the first version's approach and it wrecked the
    measurement: parked cubes went to z=0, which is BELOW the tray floor, so
    they were ejected violently and dominated both the reported footprint
    (751 mm) and the layer indices.

    Every cube rests on the one below, so the structure starts at rest under
    gravity -- unlike a dropped stack, it has no bounce energy to spread with.
    """
    pitch = size * gap
    w = 1
    while sum(i * i for i in range(1, w + 1)) < n:
        w += 1
    pos, left = [], n
    for i in range(w):
        side = w - i
        take = min(side * side, left)
        if take <= 0:
            break
        z = size * (0.5 + i) * 1.001
        off = (side - 1) / 2.0
        cells = list(itertools.product(range(side), repeat=2))[:take]
        for a, b in cells:
            pos.append(((a - off) * pitch, (b - off) * pitch, z))
        left -= take
    return torch.tensor(pos, dtype=torch.float32), len(pos), i + 1


def layer_report(state, size, label, park_x=0.4):
    # Exclude parked particles. A k-layer pyramid uses sum(i^2) cubes, which is
    # fewer than n, and the remainder is parked outside the tray -- left in, they
    # register as a spurious bottom layer (measured: 40% of a 50-cube run) and
    # inflate the footprint to half a metre.
    keep = state[..., 0].abs() < park_x
    state = state[keep].unsqueeze(0) if keep.any() else state
    z = state[..., 2]
    z0 = float(z.reshape(-1).quantile(0.02))
    lay = ((z - z0) / size).round()
    occ = {L: float((lay == L).float().mean()) for L in range(6)}
    occ = {L: v for L, v in occ.items() if v > 0.02}
    xy = state[..., :2].reshape(-1, 2)
    foot = 1000 * float(xy.max(0).values.sub(xy.min(0).values).max())
    span = 1000 * float(z.max() - z.min())
    depth = sum(L * v for L, v in occ.items())          # mean layer index
    print(f"    {label:22s} z span {span:5.1f} mm  footprint {foot:5.1f} mm  "
          f"mean layer {depth:4.2f}  " +
          " ".join(f"L{L}={100*v:.0f}%" for L, v in sorted(occ.items())),
          flush=True)
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=[50, 80])
    ap.add_argument("--size", type=float, default=0.005)
    ap.add_argument("--frictions", type=float, nargs="+", default=[0.3, 0.9])
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--mcp", type=int, default=250)
    ap.add_argument("--density", type=float, default=1000.0)
    args = ap.parse_args()

    import yaml
    from Genesis.sandbox_manipulation_clean import SandboxManipulation

    print("PILE DEPTH PROBE — can any cheap change give >1 layer?\n"
          "'mean layer' is the mass-weighted mean layer index: 0.0 = flat "
          "monolayer, 1.0 = two full layers.\n", flush=True)

    for n, fric in itertools.product(args.counts, args.frictions):
        pos, n_used, k = pyramid_positions(n, args.size)
        print(f"=== n={n} (pyramid uses {n_used}, {k} layers)  friction={fric} ===",
              flush=True)

        cfg = yaml.safe_load(open("Genesis/configs/basic.yaml"))
        cfg["material"].update({"shape": "cube", "particle_size": args.size,
                                "n_particles": n, "particle_friction": fric,
                                "density": args.density})
        cfg["box"]["friction"] = fric
        cfg.setdefault("rigid_options", {})["max_collision_pairs"] = args.mcp
        # Extent sized so a dropped spawn is comparable to the pyramid footprint.
        cfg["spawn"] = {"pile_extent": args.size * k * 1.6, "pile_layers": None}

        sim = SandboxManipulation(config=cfg, n_envs=args.envs, debug=False,
                                  viewer_type=None)
        sim.build()
        sim.set_material_properties({"particle_friction": fric,
                                     "particle_density": args.density,
                                     "box_friction": fric,
                                     "sampled_particle_friction": None,
                                     "sampled_particle_density": None})

        # --- A: the existing dropped-layer spawn, as a control ---
        t = time.time()
        sim.shuffle_particles()
        sim.update_material_state()
        d_drop = layer_report(sim._particle_state, args.size,
                              f"dropped ({time.time()-t:.0f}s)")

        # --- B: placed as a pyramid, then settled ---
        p = pos.to(sim._particle_state.device)
        full = torch.zeros((args.envs, n, 3), device=p.device)
        full[:, :n_used] = p.unsqueeze(0)
        if n_used < n:            # should not happen now; keep them in-tray if it does
            full[:, n_used:, 2] = args.size * 0.5
        quat = torch.zeros((args.envs, n, 4), device=p.device)
        quat[..., 0] = 1.0
        # A small jitter so the pyramid is not perfectly symmetric (a perfect
        # one can sit in unstable equilibrium and then topple all at once).
        full[:, :n_used, :2] += (torch.rand_like(full[:, :n_used, :2]) - 0.5) \
            * args.size * 0.08
        sim.set_particle_state(full, quat)
        d_pyr_pre = layer_report(sim._particle_state, args.size, "pyramid as placed")
        t = time.time()
        sim.update_material_state()
        d_pyr = layer_report(sim._particle_state, args.size,
                             f"pyramid settled ({time.time()-t:.0f}s)")

        # --- C: one push on the settled pyramid ---
        try:
            s, e, a = sim.generate_action_samples(1, pile_aware=True,
                                                 push_length=0.02,
                                                 min_swath_particles=3)
            sim.execute_action(s[:, 0, :], e[:, 0, :], a[:, 0])
            sim.update_material_state()
            layer_report(sim._particle_state, args.size, "pyramid + 1 push")
        except Exception as exc:                              # noqa: BLE001
            print(f"    push failed: {exc}", flush=True)

        print(f"    --> best mean layer: dropped {d_drop:.2f} | "
              f"pyramid {d_pyr:.2f} (placed {d_pyr_pre:.2f})\n", flush=True)
        sim.destroy()


if __name__ == "__main__":
    main()
