"""
Fast, Genesis-free unit tests for transforms.functional.action_to_pose —
the 4-vs-5-component push-action convention shared by
simple_mpc.genesis_oracle.GenesisOracleEnv (real steps and planning
rollouts) and the human-demo grid search / GUI (simple_mpc.human_mpc,
simple_mpc.human_grid_search). See docs/human_demo_design.md.
"""

import math

import torch

from transforms.functional import action_to_pose


def test_4d_action_derives_perpendicular_yaw():
    act = torch.tensor([0.0, 0.0, 1.0, 0.0])   # travel along +x
    sx, sy, ex, ey, angle = action_to_pose(act)
    assert (float(sx), float(sy), float(ex), float(ey)) == (0.0, 0.0, 1.0, 0.0)
    assert math.isclose(float(angle), math.pi / 2, abs_tol=1e-6)


def test_4d_zero_length_push_has_zero_angle():
    act = torch.tensor([0.3, 0.3, 0.3, 0.3])   # start == end
    *_, angle = action_to_pose(act)
    assert float(angle) == 0.0


def test_5d_action_uses_explicit_normalized_angle():
    act = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.25])
    *_, angle = action_to_pose(act)
    assert math.isclose(float(angle), 0.25 * math.pi, abs_tol=1e-6)


def test_5d_angle_independent_of_travel_direction():
    # Same start/end (and hence same implied travel direction) but different
    # explicit angle_norm -> different yaw. This is the whole point of the
    # 5th component: orientation decoupled from where the plate travels.
    act_a = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    act_b = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.9])
    *_, angle_a = action_to_pose(act_a)
    *_, angle_b = action_to_pose(act_b)
    assert not math.isclose(float(angle_a), float(angle_b), abs_tol=1e-3)


def test_batched_actions():
    act = torch.tensor([
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.5],
    ])
    sx, sy, ex, ey, angle = action_to_pose(act)
    assert sx.shape == (2,)
    assert math.isclose(float(angle[0]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(angle[1]), 0.5 * math.pi, abs_tol=1e-6)


def test_horizon_batched_actions_3d():
    # (n_envs, n_ahead, 5) — the shape simple_mpc.genesis_oracle.
    # GenesisOracleEnv.rollout_candidates slices per horizon step.
    act = torch.zeros(4, 3, 5)
    act[:, :, 2] = 1.0    # ex = 1 for every env/step
    act[:, :, 4] = 0.5    # angle_norm = 0.5 for every env/step
    sx, sy, ex, ey, angle = action_to_pose(act)
    assert sx.shape == (4, 3)
    assert torch.allclose(angle, torch.full((4, 3), math.pi / 2))
