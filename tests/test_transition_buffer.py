"""
Fast, Genesis-free unit tests for Genesis.transition_buffer.TransitionBuffer —
the accumulate/save mechanism behind automatic MPC transition recording (see
docs/ARCHITECTURE.md / docs/oracle_mpc_design.md).
"""

import torch

from Genesis.transition_buffer import TransitionBuffer


def _particle_state(n_particles=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_particles, 7, generator=g)


def test_empty_buffer_save_is_noop(tmp_path):
    buf = TransitionBuffer()
    assert buf.is_empty()
    assert len(buf) == 0
    result = buf.save(tmp_path, sim_config={"a": 1})
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_append_and_save_round_trip(tmp_path):
    buf = TransitionBuffer()
    before, after = _particle_state(seed=1), _particle_state(seed=2)
    buf.append(before, after, torch.tensor([0.1, 0.2, 0.3]),
               torch.tensor([0.4, 0.5, 0.3]), angle=1.23,
               success=True, is_candidate=False, mpc_step=0)
    assert len(buf) == 1

    path = buf.save(tmp_path, sim_config={"material": {"n_particles": 5}})
    assert path is not None
    assert buf.is_empty()   # save() clears the buffer

    data = torch.load(path, weights_only=False)
    for key in ("states", "states_", "p_starts", "p_stops", "angles",
                "success", "is_candidate", "mpc_step"):
        assert key in data

    assert data["states"].shape == (1, 5, 7)
    assert data["states_"].shape == (1, 5, 7)
    assert data["p_starts"].shape == (1, 3)
    assert data["p_stops"].shape == (1, 3)
    assert data["angles"].shape == (1,)
    assert torch.allclose(data["states"][0], before)
    assert torch.allclose(data["states_"][0], after)
    assert abs(data["angles"][0].item() - 1.23) < 1e-6
    assert data["success"][0].item() is True
    assert data["is_candidate"][0].item() is False
    assert data["mpc_step"][0].item() == 0
    assert data["success"].dtype == torch.bool
    assert data["is_candidate"].dtype == torch.bool
    assert data["mpc_step"].dtype == torch.int64

    # sidecar config always written
    assert list(tmp_path.glob("*_config.yaml"))
    # no context passed -> no context sidecar
    assert not list(tmp_path.glob("*_context.yaml"))


def test_append_batch_matches_iterative_append(tmp_path):
    K, n_particles = 4, 3
    states  = torch.randn(K, n_particles, 7)
    states_ = torch.randn(K, n_particles, 7)
    p_starts = torch.randn(K, 3)
    p_stops  = torch.randn(K, 3)
    angles   = torch.randn(K)
    success  = torch.tensor([True, False, True, True])

    buf_batch = TransitionBuffer()
    buf_batch.append_batch(states, states_, p_starts, p_stops, angles,
                            success, is_candidate=True, mpc_step=2)

    buf_loop = TransitionBuffer()
    for k in range(K):
        buf_loop.append(states[k], states_[k], p_starts[k], p_stops[k],
                         float(angles[k]), bool(success[k]),
                         is_candidate=True, mpc_step=2)

    p1 = buf_batch.save(tmp_path / "batch", sim_config={})
    p2 = buf_loop.save(tmp_path / "loop", sim_config={})
    d1 = torch.load(p1, weights_only=False)
    d2 = torch.load(p2, weights_only=False)

    for key in ("states", "states_", "p_starts", "p_stops", "angles"):
        assert torch.allclose(d1[key], d2[key])
    assert torch.equal(d1["success"], d2["success"])
    assert torch.equal(d1["is_candidate"], d2["is_candidate"])
    assert torch.equal(d1["mpc_step"], d2["mpc_step"])
    assert (d1["is_candidate"] == True).all()
    assert (d1["mpc_step"] == 2).all()


def test_save_with_context_writes_context_sidecar_and_names_file(tmp_path):
    buf = TransitionBuffer()
    buf.append(_particle_state(), _particle_state(), torch.zeros(3),
               torch.zeros(3), 0.0, True, False, 3)

    context = {"source": "oracle_mpc", "episode_idx": 7, "rewards": [1.0, 2.0]}
    path = buf.save(tmp_path, sim_config={}, context=context)

    assert "oracle_mpc" in path
    assert "ep7" in path

    context_files = list(tmp_path.glob("*_context.yaml"))
    assert len(context_files) == 1
    import yaml
    with open(context_files[0]) as f:
        loaded = yaml.safe_load(f)
    assert loaded == context


def test_multiple_saves_do_not_collide(tmp_path):
    paths = []
    for i in range(3):
        buf = TransitionBuffer()
        buf.append(_particle_state(seed=i), _particle_state(seed=i + 1),
                    torch.zeros(3), torch.zeros(3), 0.0, True, False, i)
        paths.append(buf.save(tmp_path, sim_config={}))
    assert len(set(paths)) == 3   # all distinct filenames
    assert len(list(tmp_path.glob("*_data.pt"))) == 3
