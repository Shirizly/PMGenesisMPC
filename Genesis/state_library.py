"""
Genesis/state_library.py — pre-generated libraries of *settled* pile states, so
resetting an environment costs a pose write instead of a re-settle.

Why
---
``SandboxManipulation.shuffle_particles()`` itself runs zero simulation steps;
all of a reset's cost is the settle that must follow it before the pile is at
rest. Measured on an RTX 4070 at 5 mm cubes:

    n_particles   shuffle + settle   set_particle_state   speedup
            50            5.08 s            0.094 s          54x
           100           12.77 s            0.159 s          80x
           200           42.26 s            0.763 s          55x

``set_particle_state`` needs no settle at all *provided the state it restores
was already settled* — which is exactly what this module manufactures. Generate
a library once per build, then draw from it for the rest of the run.

Augmentation
------------
Settling is the expensive part, so each settled state is amplified by the
symmetry group of the container: a state reflected or rotated into another
orientation of a square box is still a valid settled state (the walls, floor
and gravity are all invariant under those maps), and it is a genuinely
different arrangement from the sampler's point of view. A square box admits the
full dihedral group D4 — 4 rotations x optional mirror = **8 variants per
settled state**; a rectangular box admits only the 4 that preserve its aspect
ratio. So 15 settles become 120 distinct initial states for the price of 15.

Mirroring is legitimate here because the particles are achiral (cubes, spheres,
cylinders): the mirror of a valid resting arrangement is itself realizable. The
transform is applied to orientations too, not just positions — see
``_mirror_quat`` for why a reflection still maps rotations to rotations.

Usage
-----
    from Genesis.state_library import build_state_library, StateLibrary

    lib = build_state_library(sim, n_settles=15)      # generate + augment
    lib.save(out_dir)                                 # -> states.pt
    ...
    lib = StateLibrary.load(out_dir / "states.pt")
    lib.apply(sim, rng)                               # reset, no settle needed
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

STATE_LIBRARY_FILENAME = "settled_states.pt"


# --------------------------------------------------------------------------
# symmetry transforms
# --------------------------------------------------------------------------

def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (..., 4) quaternions in (w, x, y, z) order."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def _mirror_quat(q: torch.Tensor) -> torch.Tensor:
    """Orientation after mirroring the world through the xz-plane (y -> -y).

    A reflection M has det(M) = -1, so it is not itself a rotation — but the
    conjugation ``M R M`` is, since det(M R M) = det(R) = 1. For a rotation of
    angle t about axis a, ``M R(a,t) M = R(-M a, t)``, i.e. the axis maps
    (ax, ay, az) -> (-ax, ay, -az) with the angle unchanged. In quaternion
    terms that is simply (w, x, y, z) -> (w, -x, y, -z).

    Sanity check: a pure yaw q = (cos(t/2), 0, 0, sin(t/2)) maps to
    (cos(t/2), 0, 0, -sin(t/2)) — yaw t becomes yaw -t, which is what mirroring
    a top-down view does.
    """
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, y, -z), dim=-1)


def _yaw_quat(angle: float, device, dtype) -> torch.Tensor:
    return torch.tensor(
        (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)),
        device=device, dtype=dtype)


def box_symmetries(box_vol) -> list[tuple[float, bool]]:
    """(yaw, mirror) pairs that map the container onto itself.

    A square footprint admits the full dihedral group D4 (8 elements). A
    rectangular one admits only half-turns and axis mirrors (4 elements),
    because a 90 deg rotation would not fit back inside the walls.
    """
    width, depth = float(box_vol[0]), float(box_vol[1])
    square = abs(width - depth) < 1e-9
    yaws = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2] if square else [0.0, math.pi]
    return [(yaw, mirror) for yaw in yaws for mirror in (False, True)]


def apply_symmetry(pos: torch.Tensor, quat: torch.Tensor,
                   yaw: float, mirror: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a settled state through one container symmetry.

    pos  : (..., n_particles, 3)
    quat : (..., n_particles, 4) in (w, x, y, z)
    """
    p, q = pos.clone(), quat.clone()

    if mirror:
        p[..., 1] = -p[..., 1]
        q = _mirror_quat(q)

    if yaw:
        c, s = math.cos(yaw), math.sin(yaw)
        x, y = p[..., 0].clone(), p[..., 1].clone()
        p[..., 0] = c * x - s * y
        p[..., 1] = s * x + c * y
        rot = _yaw_quat(yaw, q.device, q.dtype).expand_as(q)
        q = _quat_mul(rot, q)

    return p, q


# --------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------

class StateLibrary:
    """A bank of settled particle states, restorable without re-settling."""

    def __init__(self, states: torch.Tensor, meta: dict | None = None):
        if states.ndim != 3 or states.shape[-1] != 7:
            raise ValueError(
                f"states must be (n_states, n_particles, 7), got {tuple(states.shape)}")
        self.states = states
        self.meta = meta or {}

    def __len__(self) -> int:
        return int(self.states.shape[0])

    @property
    def n_particles(self) -> int:
        return int(self.states.shape[1])

    def sample_index(self, rng=None) -> int:
        if rng is None:
            return int(torch.randint(len(self), (1,)).item())
        return int(rng.integers(len(self)))

    def apply(self, sim, rng=None, index: int | None = None) -> int:
        """Reset ``sim`` to one library state. Returns the index used.

        No settle is required afterwards: the stored state is already at rest.
        This is the whole point of the library — it replaces
        ``shuffle_particles() + update_material_state()``.
        """
        if index is None:
            index = self.sample_index(rng)
        state = self.states[index].to(sim._particle_state.device)
        if state.shape[0] != len(sim.material):
            raise ValueError(
                f"library holds {state.shape[0]} particles but the scene has "
                f"{len(sim.material)}; libraries are specific to a build")
        sim.set_particle_state(state[None, :, 0:3], state[None, :, 3:7])
        return index

    def save(self, out_dir: str | Path, filename: str = STATE_LIBRARY_FILENAME) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        torch.save({"states": self.states.cpu(), "meta": self.meta}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StateLibrary":
        blob = torch.load(Path(path), weights_only=False)
        return cls(blob["states"], blob.get("meta", {}))


def build_state_library(sim, n_settles: int = 15, *, augment: bool = True,
                        damping: float = 0.0, verbose: bool = True) -> StateLibrary:
    """Settle ``n_settles`` fresh random piles and expand them by symmetry.

    Each ``shuffle_particles()`` randomizes *every* parallel env independently,
    so one settle yields ``n_envs`` distinct states, not one — the library is
    ``n_settles * n_envs * len(box_symmetries)`` states for the cost of
    ``n_settles`` settles.

    ``damping`` adds temporary viscous damping to the particles for the duration
    of these settles only, and removes it afterwards. It is a **numerical**
    device, not a physical one: real air drag on a 5 mm cube at 50 mm/s is about
    3e-5 of its weight, far too small to influence settling, and damping strong
    enough to matter would be a sizeable fraction of gravity. What justifies it
    here is scope — this settle only has to reach *a* valid resting
    configuration for the library, and damping changes how fast that is reached
    rather than what counts as resting. The same argument does NOT extend to the
    post-push settle, where the relaxation being cut short would bias the
    recorded s' toward smaller displacements, so nothing applies damping there.
    """
    if n_settles <= 0:
        raise ValueError("n_settles must be positive")

    dofs = getattr(sim, "_particle_dofs_idx", None)
    damped = damping > 0.0 and dofs is not None and dofs.numel() > 0
    if damped:
        sim._scene.rigid_solver.set_dofs_damping(
            torch.full((dofs.numel(),), float(damping), device=dofs.device),
            dofs_idx=dofs)

    base = []
    try:
        for i in range(n_settles):
            sim.shuffle_particles()
            sim.update_material_state()      # the expensive part, paid once
            base.append(sim._particle_state.detach().clone().cpu())
            if verbose:
                print(f"  settled pile {i + 1}/{n_settles}", flush=True)
    finally:
        if damped:
            sim._scene.rigid_solver.set_dofs_damping(
                torch.zeros(dofs.numel(), device=dofs.device), dofs_idx=dofs)

    base_states = torch.cat(base, dim=0)      # (n_settles * n_envs, n_p, 7)

    syms = box_symmetries(sim._box_params["vol"]) if augment else [(0.0, False)]
    variants = []
    for yaw, mirror in syms:
        p, q = apply_symmetry(base_states[..., 0:3], base_states[..., 3:7], yaw, mirror)
        variants.append(torch.cat((p, q), dim=-1))
    states = torch.cat(variants, dim=0)

    meta = {
        "n_settles": n_settles,
        "n_envs": int(sim._n_envs),
        "n_base_states": int(base_states.shape[0]),
        "n_symmetries": len(syms),
        "n_states": int(states.shape[0]),
        "n_particles": int(states.shape[1]),
        "augmented": bool(augment),
        "symmetries": [(float(y), bool(m)) for y, m in syms],
        "box_vol": list(sim._box_params["vol"]),
        "particle_size": sim._material_params.get("particle_size"),
        "shape": sim._material_params.get("shape"),
    }
    if verbose:
        print(f"  state library: {base_states.shape[0]} settled x {len(syms)} "
              f"symmetries = {states.shape[0]} states", flush=True)
    return StateLibrary(states, meta)
