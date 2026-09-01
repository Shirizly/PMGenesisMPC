"""
Genesis/spawn_geometry.py — particle spawn layouts that the RSA draw cannot make.

Why this exists
---------------
The default spawn is rejection sampling over an xy region, optionally split into
stacked layers that are then dropped. That reliably produces a *dense monolayer*,
however hard it is pushed: measured, 90% of particles end in layer 0 at 30 cubes
of 5 mm and 94% at 80 cubes of 3 mm, and raising friction from 0.3 to 0.9 made it
slightly flatter rather than deeper. The cause is the drop itself — cubes released
above the floor bounce and spread outward before they come to rest, and lighter
cubes bounce more.

Depth is the untested half of two hypotheses about why the paper's per-pixel
linear operator worked and ours did not (docs/linear_foresight_findings.md §4,
H1 and H2), so it needs a mechanism that does not involve dropping anything.

A stepped pyramid does: every cube rests on the one below, so the structure starts
at rest under gravity and has no bounce energy to spread with. Measured at
50 cubes of 5 mm, mean layer index (mass-weighted, 0 = flat):

    dropped spawn      0.05 - 0.09     footprint 65-67 mm
    pyramid            0.68            footprint 23 mm
    pyramid, settled   0.68            footprint 23 mm   (identical -- nothing moves)
    pyramid + 1 push   0.67 - 0.68     footprint 38-48 mm

So it is ~7x deeper than the dropped spawn by mean layer, a third of the
footprint, does not relax when gravity is applied, and keeps its layering through
a push.

Caveat on those numbers: they were measured with a [25, 16, 9] layout for n=50.
`pyramid_layer_plan` was then changed to fix a non-monotonicity (see its
docstring), and now gives [36, 9, 4, 1] for n=50 -- one layer *deeper*, so it
should be at least as good, but it has not been re-measured. Re-run
`scripts/probe_pile_depth.py --counts 50` to confirm before relying on the exact
figures.

Pure torch, no `genesis` import, so it is unit testable without a GPU
(tests/test_spawn_geometry.py).
"""

from __future__ import annotations

import itertools

import torch


def pyramid_layer_plan(n: int) -> list[int]:
    """How many cubes go in each layer of an n-cube pyramid, bottom first.

    Builds the tallest *complete* pyramid that fits — layers ``k^2, (k-1)^2, ...,
    1`` for the largest ``k`` with ``sum(i^2) <= n`` — then adds whatever is left
    over to the **bottom** layer, widening the base rather than perching an
    unstable partial layer on the apex:

        n=14 -> [9, 4, 1]           three layers
        n=30 -> [16, 9, 4, 1]       four layers
        n=50 -> [36, 9, 4, 1]       four layers  (30 complete + 20 into the base)
        n=55 -> [25, 16, 9, 4, 1]   five layers
        n=80 -> [50, 16, 9, 4, 1]   five layers

    The layer count is then **monotone** in ``n``, which the obvious alternative
    is not: choosing the base from the smallest full-pyramid sum that *reaches*
    ``n`` and truncating the top gives 5 layers at n=55 but only 2 at n=56
    (``[36, 20]``), because the base jumps a whole width to absorb one extra
    cube. A unit test pins the monotonicity.

    Placing only a complete pyramid and parking the remainder was tried first and
    is a trap: parked cubes have to go somewhere, and anywhere outside the tray
    is either below the floor (they are ejected violently) or inside the
    measurement (they register as a spurious bottom layer — observed at 40% of a
    50-cube run, with the reported footprint inflated to 751 mm).
    """
    if n <= 0:
        return []
    k = 1
    while sum(i * i for i in range(1, k + 2)) <= n:
        k += 1
    plan = [i * i for i in range(k, 0, -1)]
    plan[0] += n - sum(plan)
    return plan


def pyramid_positions(n: int, size: float, gap: float = 1.15,
                      centre: tuple[float, float] = (0.0, 0.0),
                      floor_z: float = 0.0,
                      device=None, dtype=torch.float32):
    """Centre positions for an n-cube stepped pyramid resting on ``floor_z``.

    Parameters
    ----------
    n       : cubes to place. All of them are placed (see `pyramid_layer_plan`).
    size    : cube edge length, metres.
    gap     : lateral pitch as a multiple of ``size``. Slightly above 1 so
              neighbours are not born in contact, which the solver dislikes.
    centre  : xy centre of the pyramid.
    floor_z : z of the surface the bottom layer sits on.

    Returns
    -------
    ``(positions, n_layers)`` with positions ``(n, 3)``.

    Layers are square blocks, largest at the bottom, each centred on ``centre``,
    so the whole structure is symmetric. A caller wanting to avoid a perfectly
    symmetric (and therefore unstably balanced) start should add a small xy
    jitter -- `SandboxManipulation.shuffle_particles` adds 8% of a cube width.
    """
    plan = pyramid_layer_plan(n)
    pitch = size * gap
    pos = []
    for i, take in enumerate(plan):
        side = int(round(take ** 0.5))
        if side * side < take:
            side += 1
        z = floor_z + size * (0.5 + i) * 1.001
        off = (side - 1) / 2.0
        cells = list(itertools.product(range(side), repeat=2))[:take]
        for a, b in cells:
            pos.append((centre[0] + (a - off) * pitch,
                        centre[1] + (b - off) * pitch,
                        z))
    return torch.tensor(pos, dtype=dtype, device=device), len(plan)
