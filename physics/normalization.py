"""
Physics parameter normalisation.

Defines PhysicsBounds, the single source of truth for mapping raw simulator
physics values to the normalised [0, 1] range expected by FiLM-conditioned
models.  Every callsite — training, inference, sysid — should use this class
rather than duplicating the arithmetic inline.

Standard physics vector layout (length 3, dtype float32):
    index 0:  material friction     (dimensionless)
    index 1:  material density      (kg/m³)
    index 2:  box / surface friction (dimensionless)

The bounds are stored in dataset config YAMLs under a ``physics.normalization``
sub-dict and in model cards so that inference always uses the same scale as
training:

    physics:
      normalization:
        friction:     {min: 0.05, max: 0.50}
        density:      {min: 750.0, max: 5000.0}
        box_friction: {min: 0.05, max: 0.50}

Usage::

    from physics.normalization import PhysicsBounds

    bounds = PhysicsBounds.from_config(dataset_cfg["physics"])
    normalised = bounds.normalize(raw_tensor)   # → [0, 1]
    raw        = bounds.denormalize(normalised)  # → physical units
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PhysicsBounds:
    """
    Min/max bounds for the three-element physics vector.

    Inputs / outputs of normalize() and denormalize():
        Tensor[3]    — single sample   [friction, density, box_friction]
        Tensor[B, 3] — batch of samples

    All values are clamped to [0, 1] after normalisation to handle
    out-of-training-distribution inputs gracefully.
    """

    friction_min: float = 0.05
    friction_max: float = 0.50
    density_min: float = 750.0
    density_max: float = 5000.0
    box_friction_min: float = 0.05
    box_friction_max: float = 0.50

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "PhysicsBounds":
        """
        Build from the ``physics`` or ``physics.normalization`` sub-dict of a
        dataset / model-card config.

        Accepts either::

            {"normalization": {"friction": {"min": ..., "max": ...}, ...}}

        or the inner normalisation dict directly::

            {"friction": {"min": ..., "max": ...}, ...}
        """
        n = cfg.get("normalization", cfg)
        return cls(
            friction_min=float(n["friction"]["min"]),
            friction_max=float(n["friction"]["max"]),
            density_min=float(n["density"]["min"]),
            density_max=float(n["density"]["max"]),
            box_friction_min=float(n["box_friction"]["min"]),
            box_friction_max=float(n["box_friction"]["max"]),
        )

    @classmethod
    def default(cls) -> "PhysicsBounds":
        """Default bounds matching the Genesis cube training data."""
        return cls()

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    def _lo_hi(self, device=None, dtype=torch.float32):
        lo = torch.tensor(
            [self.friction_min, self.density_min, self.box_friction_min],
            dtype=dtype, device=device,
        )
        hi = torch.tensor(
            [self.friction_max, self.density_max, self.box_friction_max],
            dtype=dtype, device=device,
        )
        return lo, hi

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Map raw physics values → [0, 1].

        Parameters
        ----------
        raw : Tensor[3] or Tensor[B, 3]
            [friction, density, box_friction] in physical units.

        Returns
        -------
        Tensor, same shape, values clamped to [0, 1].
        """
        lo, hi = self._lo_hi(device=raw.device, dtype=raw.dtype)
        return ((raw - lo) / (hi - lo)).clamp(0.0, 1.0)

    def denormalize(self, norm: torch.Tensor) -> torch.Tensor:
        """
        Map [0, 1] values → raw physical units.

        Parameters
        ----------
        norm : Tensor[3] or Tensor[B, 3]  — values in [0, 1].

        Returns
        -------
        Tensor, same shape, in physical units.
        """
        lo, hi = self._lo_hi(device=norm.device, dtype=norm.dtype)
        return norm * (hi - lo) + lo

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the ``normalization`` sub-dict for embedding in YAML configs."""
        return {
            "friction":     {"min": self.friction_min,     "max": self.friction_max},
            "density":      {"min": self.density_min,      "max": self.density_max},
            "box_friction": {"min": self.box_friction_min, "max": self.box_friction_max},
        }
