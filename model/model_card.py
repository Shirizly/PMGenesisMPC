"""
Model card: everything needed to reload and use a trained checkpoint.

A model card is a YAML sidecar file written alongside each checkpoint by the
Trainer.  Loading a model for inference or evaluation requires only the card
path — no training config needed at that point.

File layout (one training run):

    runs_cubes/my_run/
        unet_best.pth      ← raw state_dict (torch.save / torch.load)
        model_card.yaml    ← auto-written by Trainer after each best checkpoint
        run_config.yaml    ← full training config (for reference / reproducibility)

Model card YAML schema
----------------------

    model:
      type:       unetfilm          # registry key
      checkpoint: unet_best.pth    # relative to this card file
      in_channels: 2
      cond_dim:    3
      uses_physics: true
      # ... other model config keys

    physics:
      normalization:
        friction:     {min: 0.05, max: 0.50}
        density:      {min: 750.0, max: 5000.0}
        box_friction: {min: 0.05, max: 0.50}

    inference:
      representation:  eulerian     # "eulerian" | "lagrangian"
      grid_n:          128          # occupancy grid side length (px)
      plate_length_m:  0.04         # plate long axis (m)
      plate_width_m:   0.002        # plate short axis (m)
      plate_sigma_px:  1.5          # soft-plate render sigma
      wkspc_w:         0.064        # workspace half-width (m)
      global_scale:    0.6          # = 2 × cam_height
      particle_friction: 0.05       # raw physics values for normalisation
      particle_density:  750.0
      box_friction:      0.05

Usage::

    from model.model_card import load_model_from_card
    model = load_model_from_card("runs_cubes/my_run/model_card.yaml", env)
    # model is an EulerianModelWrapper ready for use in run_simple_mpc()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from physics.normalization import PhysicsBounds


@dataclass
class ModelCard:
    """
    Self-contained description of a trained model checkpoint.

    Attributes
    ----------
    path           : Path          — location of this card's YAML file
    model_cfg      : dict          — model config (same structure as configs/model/*.yaml),
                                     including "checkpoint" key (relative path to .pth)
    physics_bounds : PhysicsBounds — normalisation bounds used at training time
    inference_cfg  : dict          — inference geometry and raw physics values
    """

    path: Path
    model_cfg: dict
    physics_bounds: PhysicsBounds
    inference_cfg: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, card_path: str | Path) -> "ModelCard":
        """
        Load a ModelCard from a YAML file on disk.

        Parameters
        ----------
        card_path : str | Path — path to model_card.yaml
        """
        card_path = Path(card_path)
        raw = yaml.safe_load(card_path.read_text())
        bounds = (
            PhysicsBounds.from_config(raw["physics"])
            if "physics" in raw
            else PhysicsBounds.default()
        )
        return cls(
            path=card_path,
            model_cfg=raw.get("model", {}),
            physics_bounds=bounds,
            inference_cfg=raw.get("inference", {}),
        )

    @classmethod
    def from_training_config(
        cls,
        training_cfg: dict,
        run_dir: Path,
        checkpoint_name: str,
    ) -> "ModelCard":
        """
        Build a ModelCard from a resolved training config dict.

        Called by the Trainer after saving a checkpoint.

        Parameters
        ----------
        training_cfg    : dict  — full resolved training YAML (has "model",
                                  "dataset", and "inference" sections)
        run_dir         : Path  — directory where checkpoints are written
        checkpoint_name : str   — filename of the checkpoint (e.g. "unet_best.pth")
        """
        model_cfg = dict(training_cfg.get("model", {}))
        model_cfg["checkpoint"] = checkpoint_name

        physics_sub = training_cfg.get("dataset", {}).get("physics", {})
        bounds = (
            PhysicsBounds.from_config(physics_sub)
            if physics_sub
            else PhysicsBounds.default()
        )
        return cls(
            path=run_dir / "model_card.yaml",
            model_cfg=model_cfg,
            physics_bounds=bounds,
            inference_cfg=training_cfg.get("inference", {}),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write this card to ``self.path`` as YAML."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "model":     self.model_cfg,
            "physics":   {"normalization": self.physics_bounds.to_dict()},
            "inference": self.inference_cfg,
        }
        with open(self.path, "w") as f:
            yaml.dump(raw, f, sort_keys=False, default_flow_style=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def checkpoint_path(self) -> Path:
        """Absolute path to the checkpoint .pth file."""
        ckpt = self.model_cfg.get("checkpoint", "model.pth")
        p = Path(ckpt)
        return p if p.is_absolute() else self.path.parent / p

    @property
    def representation(self) -> str:
        """'eulerian' or 'lagrangian'."""
        return self.inference_cfg.get("representation", "eulerian")


# ---------------------------------------------------------------------------
# Inference model loader
# ---------------------------------------------------------------------------

def load_model_from_card(
    card_path: str | Path,
    env: Any = None,
    mpc_cfg: dict | None = None,
) -> Any:
    """
    Load an inference-ready model wrapper from a model card.

    For Eulerian models:   returns an EulerianModelWrapper wrapping a
                           UNetFiLMPushModel (ready for ptcl_model_rollout).
    For Lagrangian models: returns a PropNetDiffDenModel (direct).

    Parameters
    ----------
    card_path : str | Path — path to model_card.yaml
    env       : GenesisEnv  — needed for cam_extrinsics; can be None for tests
                              (cam_extrinsic will be None on the wrapper)
    mpc_cfg   : dict        — override inference config values (optional)

    Returns
    -------
    EulerianModelWrapper | PropNetDiffDenModel  ready for use in MPC.
    """
    card = ModelCard.load(card_path)
    if card.representation == "eulerian":
        return _load_eulerian(card, env, mpc_cfg)
    elif card.representation == "lagrangian":
        return _load_lagrangian(card)
    else:
        raise ValueError(f"Unknown representation: {card.representation!r}")


def _load_eulerian(card: ModelCard, env: Any, mpc_cfg: dict | None):
    from registry.model_registry import build_model
    from model.eulerian_wrapper import EulerianModelWrapper, UNetFiLMPushModel

    # --- Build and load weights ---
    training_wrapper = build_model(card.model_cfg)
    state = torch.load(card.checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        training_wrapper.load_state_dict(state["model_state_dict"])
    else:
        training_wrapper.load_state_dict(state)
    unet = training_wrapper.model
    unet.eval()

    # --- Inference geometry ---
    ic = {**card.inference_cfg, **(mpc_cfg or {})}
    grid_n       = int(ic.get("grid_n", 128))
    plate_L_m    = float(ic.get("plate_length_m", 0.04))
    plate_W_m    = float(ic.get("plate_width_m", 0.002))
    sigma_px     = float(ic.get("plate_sigma_px", 1.5))
    wkspc_w      = float(ic.get("wkspc_w", 0.064))
    global_scale = float(ic.get("global_scale", 0.6))

    # --- Normalised physics vector ---
    raw_physics = torch.tensor([
        float(ic.get("particle_friction", 0.05)),
        float(ic.get("particle_density",  750.0)),
        float(ic.get("box_friction",       0.05)),
    ], dtype=torch.float32)
    physics_vec = card.physics_bounds.normalize(raw_physics)

    # --- Grid bounds ---
    px_per_m   = grid_n / (wkspc_w * 2.0)
    plate_L_px = plate_L_m * px_per_m
    plate_W_px = plate_W_m * px_per_m

    push_model = UNetFiLMPushModel(
        unet_film=unet,
        physics=physics_vec,
        grid_size=(grid_n, grid_n),
        plate_length_px=plate_L_px,
        plate_width_px=plate_W_px,
        sigma=sigma_px,
    )
    fake_config = {"dataset": {"wkspc_w": wkspc_w, "global_scale": global_scale}}
    bounds = UNetFiLMPushModel.default_bounds(fake_config)

    cam_extrinsic = env.get_cam_extrinsics() if env is not None else None
    return EulerianModelWrapper(
        push_model,
        bounds,
        (grid_n, grid_n),
        cam_extrinsic,
        global_scale,
        action_convention="genesis",
    )


def _load_lagrangian(card: ModelCard):
    from model.gnn_dyn import PropNetDiffDenModel

    model_cfg = {"train": {"particle": {
        "nf_effect": int(card.model_cfg.get("nf_effect", 150)),
        "add_delta":  bool(card.model_cfg.get("add_delta", False)),
        "adj_thresh": float(card.model_cfg.get("adj_thresh", 0.08)),
    }}}
    model = PropNetDiffDenModel(model_cfg, use_gpu=torch.cuda.is_available())
    state = torch.load(card.checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model
