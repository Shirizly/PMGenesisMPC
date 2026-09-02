"""`SandboxManipulation` plus MPC-side transition recording.

Why a subclass rather than four more methods on the simulator
-------------------------------------------------------------
The simulator already has a recording path: `collect_data_samples` writes
`_N_data.pt` / `_N_rollout.pt` and the DINO/LeWM exporters read them. That path
is batch-oriented - allocate buffers for `n_samples`, fill them, save once -
which is the right shape for a collection sweep and the wrong shape for MPC,
where transitions arrive one at a time from a planner and candidate rollouts
have to be distinguished from executed steps.

So MPC needs an incremental path, and putting it in
`Genesis/sandbox_manipulation.py` would leave two overlapping recording
mechanisms in a file whose whole purpose right now is to be reviewable as a
simulator PR. Subclassing keeps it out.

This works cleanly here - unlike the layered spawn, which had to be a copy -
because none of it runs before `scene.build()`. Everything the recording needs
is set up after `super().__init__()` returns.

Use `RecordingSandbox` anywhere the MPC stack would have used
`SandboxManipulation`.
"""
from pathlib import Path

import genesis_path  # noqa: F401  (puts Genesis/ and MPC/ on sys.path)
from sandbox_manipulation import SandboxManipulation

from env.transition_buffer import TransitionBuffer


class RecordingSandbox(SandboxManipulation):
    """The simulator, with an incremental transition recorder attached.

    Recording is controlled by the `data_collection` block of the config:

        data_collection:
          record_transitions: true          # default
          transitions_dir: mpc_transitions  # relative to MPC/
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dc = self._config.get("data_collection", {}) or {}
        self._record_transitions = bool(dc.get("record_transitions", True))
        self._transitions_dir = (
            Path(__file__).resolve().parent.parent
            / dc.get("transitions_dir", "mpc_transitions"))
        self._transition_buffer = (
            TransitionBuffer() if self._record_transitions else None)
        # Episode-level context, set once per episode and reused by every
        # real-step flush until it is replaced. This is what lets flushes carry
        # episode-identifying information *incrementally* rather than waiting
        # for the episode's outcome to be known.
        self._transition_context: dict | None = None

    # ------------------------------------------------------------------ #
    def set_transition_context(self, context: dict | None) -> None:
        """Tag subsequent flushes with this episode's context.

        Call once per episode, right after resetting. Per-episode outcome
        (reward, success) is written separately by the driver script and joined
        on source + episode_idx, so nothing here has to wait for it.
        """
        self._transition_context = context

    def flush_transitions(self, context: dict | None = None) -> str | None:
        """Write buffered transitions to `transitions_dir` and clear the buffer.

        Returns the saved file's path, or None if recording is off or nothing
        is buffered. `context=None` falls back to whatever `set_transition_context`
        last set, so the safety-net calls from `shuffle_particles` and
        `destroy` still tag their flushes.
        """
        if not self._record_transitions or self._transition_buffer.is_empty():
            return None
        return self._transition_buffer.save(
            self._transitions_dir, self._config,
            context=context if context is not None else self._transition_context)

    def push_and_record(self, p_start, p_stop, angle, on_phase=None,
                        is_candidate=False, mpc_step=None,
                        record_all_envs=True, flush_after=False):
        """`execute_action` + settle, with the transition appended to the buffer.

        Replaces the `execute_action(...)` + `update_material_state()` pair at
        every real-step and candidate-rollout call site. Callers set
        `_settle_steps` / `_clearance_ctrl_steps` before calling, exactly as
        they already did before that pair - neither is read by
        `execute_action`, so any time before the settle is equivalent.

        Args:
            is_candidate: False for a real executed step; True for an
                optimizer-exploration rollout evaluated during planning. The
                distinction matters downstream: candidate rollouts are
                off-policy and vastly outnumber real steps.
            mpc_step: which real MPC step's planning produced this push. The
                simulator has no notion of an MPC step, so the caller supplies
                its own counter.
            record_all_envs: True records one sample per env, for when every
                env ran a genuinely distinct action. Pass False when all envs
                execute the same broadcast action from the same state, where
                recording every env would just duplicate one transition
                `n_envs` times.
            flush_after: flush immediately after appending. True for real
                steps, so data reaches disk step by step - an episode can run
                for a long time, and losing it all to a crash (or seeing no
                output while it runs) is the failure this avoids. False for
                candidate rollouts, which accumulate between real steps.

        Returns (reached_goal, final_pos), identical to `execute_action`.
        """
        before = self._particle_state.clone() if self._record_transitions else None
        reached_goal, final_pos = self.execute_action(
            p_start, p_stop, angle, on_phase=on_phase)
        self.update_material_state()
        if self._record_transitions:
            if record_all_envs:
                self._transition_buffer.append_batch(
                    before, self._particle_state, p_start, p_stop, angle,
                    reached_goal, is_candidate=is_candidate, mpc_step=mpc_step,
                )
            else:
                self._transition_buffer.append(
                    before[0], self._particle_state[0], p_start[0], p_stop[0],
                    float(angle[0]), bool(reached_goal[0]),
                    is_candidate=is_candidate, mpc_step=mpc_step,
                )
            if flush_after:
                self.flush_transitions(context=self._transition_context)
        return reached_goal, final_pos

    def broadcast_state_from_env(self, src_env: int = 0) -> None:
        """Copy env `src_env`'s live particle pose to every env.

        For one-off resyncs where `src_env`'s live state is the correct
        reference. For repeated use across rollouts where `src_env` may itself
        be mutated in between - env 0 doubling as a rollout worker during
        planning, as in `GenesisOracleEnv` - capture a snapshot once and pass
        it to `set_particle_state` directly instead of relying on this live
        read.
        """
        state = self._particle_state[src_env, :, :].clone()
        self.set_particle_state(state)

    # ------------------------------------------------------------------ #
    def shuffle_particles(self, *args, **kwargs):
        """Flush before wiping particle state, so buffered data is never lost.

        A safety net: a real step's `flush_after=True` should already have
        drained the buffer, but a new episode is starting and whatever is still
        buffered belongs to the one being overwritten. The context is cleared
        too - the caller is expected to set a fresh one right after reset.
        """
        if getattr(self, "_record_transitions", False):
            self.flush_transitions()
            self.set_transition_context(None)
        return super().shuffle_particles(*args, **kwargs)

    def destroy(self, *args, **kwargs):
        if getattr(self, "_record_transitions", False):
            self.flush_transitions()
        return super().destroy(*args, **kwargs)
