"""
TransitionBuffer — accumulates before/after particle-state transitions
produced by SandboxManipulation.push_and_record and saves them in the same
on-disk format Genesis/data_collection_clean.py's dataset files use (a
strict superset — see docs/ARCHITECTURE.md and docs/oracle_mpc_design.md).

Genesis-free (no ``import genesis``) so it's testable without a GPU or the
genesis package installed; SandboxManipulation is the only caller.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml


class TransitionBuffer:
    """
    Accumulates push transitions in memory; ``save()`` writes them to disk
    and clears the buffer.

    On-disk schema (torch.save'd dict, ``<prefix>_data.pt``):

        states       Tensor[N, n_particles, 7]  float32 — pose before the push
        states_      Tensor[N, n_particles, 7]  float32 — pose after the push
        p_starts     Tensor[N, 3]               float32
        p_stops      Tensor[N, 3]               float32
        angles       Tensor[N]                  float32
        success      Tensor[N]                  bool    — plate reached p_stop
                                                  (mechanical; from
                                                  execute_action's reached_goal —
                                                  NOT a task-reward success)
        is_candidate Tensor[N]                  bool    — False: a real executed
                                                  step (part of a sequential
                                                  trajectory); True: an
                                                  optimizer-exploration rollout
        mpc_step     Tensor[N]                  int64   — which real MPC step's
                                                  planning phase produced this
                                                  sample (-1 if unknown)

    The first 5 keys exactly match Genesis/data_collection_clean.py's
    ``states``/``states_``/``p_starts``/``p_stops``/``angles``, so existing
    loaders (Genesis/training/dataset.py's ``PileSweepData``,
    registry/dataset_registry.py's ``GenesisParticlePushDataset``) load these
    files unchanged — they only read those 5 keys and ignore extras.

    Unlike ``collect_data_samples``'s valid/failed file split, everything
    goes in **one** file: splitting would fragment a sequential trajectory
    across two files. ``success`` carries the same information per-sample
    instead.

    Sidecars: ``<prefix>_config.yaml`` (the producing ``SandboxManipulation``'s
    live config — same convention as ``_save_config``) and, when a context
    dict is supplied, ``<prefix>_context.yaml`` (episode-level info: rewards,
    success, source MPC variant, seed, etc. — free-form, YAML-serializable
    values only).
    """

    def __init__(self):
        self.clear()

    def is_empty(self) -> bool:
        return len(self._states) == 0

    def __len__(self) -> int:
        return len(self._states)

    def append(
        self,
        state: torch.Tensor,      # (n_particles, 7) — before push
        state_: torch.Tensor,     # (n_particles, 7) — after push
        p_start: torch.Tensor,    # (3,)
        p_stop: torch.Tensor,     # (3,)
        angle: float,
        success: bool,
        is_candidate: bool,
        mpc_step: int | None,
    ) -> None:
        """Append a single transition."""
        self._states.append(state.detach().cpu())
        self._states_.append(state_.detach().cpu())
        self._p_starts.append(p_start.detach().cpu().float())
        self._p_stops.append(p_stop.detach().cpu().float())
        self._angles.append(float(angle))
        self._success.append(bool(success))
        self._is_candidate.append(bool(is_candidate))
        self._mpc_step.append(int(mpc_step) if mpc_step is not None else -1)

    def append_batch(
        self,
        states: torch.Tensor,     # (K, n_particles, 7) — before push, one per env
        states_: torch.Tensor,    # (K, n_particles, 7) — after push, one per env
        p_starts: torch.Tensor,   # (K, 3)
        p_stops: torch.Tensor,    # (K, 3)
        angles: torch.Tensor,     # (K,)
        success: torch.Tensor,    # (K,) bool
        is_candidate: bool,
        mpc_step: int | None,
    ) -> None:
        """
        Append a whole batch (e.g. one rollout call's ``n_envs`` candidates)
        in one call — avoids a K-iteration Python loop when K is large (e.g.
        256 parallel envs).
        """
        K = states.shape[0]
        states_cpu   = states.detach().cpu()
        states__cpu  = states_.detach().cpu()
        p_starts_cpu = p_starts.detach().cpu().float()
        p_stops_cpu  = p_stops.detach().cpu().float()
        angles_cpu   = angles.detach().cpu().float().tolist()
        success_cpu  = success.detach().cpu().bool().tolist()

        self._states.extend(states_cpu[k] for k in range(K))
        self._states_.extend(states__cpu[k] for k in range(K))
        self._p_starts.extend(p_starts_cpu[k] for k in range(K))
        self._p_stops.extend(p_stops_cpu[k] for k in range(K))
        self._angles.extend(float(a) for a in angles_cpu)
        self._success.extend(bool(s) for s in success_cpu)
        self._is_candidate.extend([bool(is_candidate)] * K)
        self._mpc_step.extend([int(mpc_step) if mpc_step is not None else -1] * K)

    def clear(self) -> None:
        self._states:       list = []
        self._states_:      list = []
        self._p_starts:     list = []
        self._p_stops:      list = []
        self._angles:       list = []
        self._success:      list = []
        self._is_candidate: list = []
        self._mpc_step:     list = []

    def save(
        self,
        out_dir: str | Path,
        sim_config: dict,
        context: dict | None = None,
    ) -> str | None:
        """
        Write the accumulated samples to ``<out_dir>/<prefix>_data.pt`` (plus
        ``_config.yaml`` / optional ``_context.yaml`` sidecars) and clear the
        buffer. Returns the data file path, or ``None`` if the buffer was
        empty (no file written).

        ``prefix`` is a nanosecond timestamp (collision-free across
        concurrent/sequential runs into the same shared, ever-growing
        directory), optionally suffixed with ``context['source']`` /
        ``context['episode_idx']`` for readability.
        """
        if self.is_empty():
            return None

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = str(time.time_ns())
        if context:
            source     = context.get("source")
            episode_idx = context.get("episode_idx")
            if source is not None:
                prefix += f"_{source}"
            if episode_idx is not None:
                prefix += f"_ep{episode_idx}"

        data = {
            "states":       torch.stack(self._states),
            "states_":      torch.stack(self._states_),
            "p_starts":     torch.stack(self._p_starts),
            "p_stops":      torch.stack(self._p_stops),
            "angles":       torch.tensor(self._angles, dtype=torch.float32),
            "success":      torch.tensor(self._success, dtype=torch.bool),
            "is_candidate": torch.tensor(self._is_candidate, dtype=torch.bool),
            "mpc_step":     torch.tensor(self._mpc_step, dtype=torch.int64),
        }
        data_path = out_dir / f"{prefix}_data.pt"
        torch.save(data, data_path)

        with open(out_dir / f"{prefix}_config.yaml", "w") as f:
            yaml.dump(sim_config, f, default_flow_style=False)

        if context:
            with open(out_dir / f"{prefix}_context.yaml", "w") as f:
                yaml.dump(context, f, default_flow_style=False)

        self.clear()
        return str(data_path)
