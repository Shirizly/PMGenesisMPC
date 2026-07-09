"""
Verify the diff_mass_push heuristic models are reachable through
model/eulerian_wrapper.py's push-model registry and produce correct-shaped
predictions end-to-end through EulerianModelWrapper (as used for MPC).
"""

import pytest
import torch

from model.eulerian_wrapper import (
    EulerianModelWrapper,
    SplatPushModel,
    SpreadPushModel,
    SplatPushModel2,
    CumulativePushModel,
    FluidPushModel,
    build_push_model,
)

REGISTERED_TYPES = {
    "splat": SplatPushModel,
    "spread": SpreadPushModel,
    "spread2": SplatPushModel2,
    "cumulative": CumulativePushModel,
    "fluid": FluidPushModel,
}


@pytest.mark.parametrize("heuristic_type,expected_cls", REGISTERED_TYPES.items())
def test_build_push_model_returns_expected_class(heuristic_type, expected_cls):
    model = build_push_model({"heuristic_type": heuristic_type})
    assert isinstance(model, expected_cls)


def test_build_push_model_default_is_spread():
    model = build_push_model({})
    assert isinstance(model, SpreadPushModel)


def test_build_push_model_applies_overrides():
    model = build_push_model({"heuristic_type": "splat", "width": 7.0, "sigma": 2.0})
    assert model.width == 7.0
    assert model.sigma == 2.0


def test_build_push_model_unknown_type_raises():
    with pytest.raises(KeyError):
        build_push_model({"heuristic_type": "not-a-real-heuristic"})


@pytest.mark.parametrize("heuristic_type", REGISTERED_TYPES)
def test_push_model_forward_shape(heuristic_type):
    model = build_push_model({"heuristic_type": heuristic_type})
    occ = torch.rand(2, 32, 32)
    action_start = torch.tensor([[10.0, 10.0, 0.0], [5.0, 20.0, 0.0]])
    action_end = torch.tensor([[20.0, 10.0, 0.0], [15.0, 20.0, 0.0]])
    with torch.no_grad():
        out = model(occ, action_start, action_end)
    assert out.shape == occ.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("heuristic_type", REGISTERED_TYPES)
def test_eulerian_model_wrapper_predict_one_step_occ(heuristic_type):
    """End-to-end MPC deployment path: build via registry, wrap, roll one step."""
    push_model = build_push_model({"heuristic_type": heuristic_type})
    cfg = {"dataset": {"global_scale": 0.6, "wkspc_w": 0.064}}
    bounds = EulerianModelWrapper.default_bounds(cfg, convention="genesis")
    grid_n = 32
    wrapper = EulerianModelWrapper(
        push_model,
        bounds,
        (grid_n, grid_n),
        cam_extrinsic=None,
        global_scale=cfg["dataset"]["global_scale"],
        action_convention="genesis",
    )

    occ = torch.zeros(2, grid_n, grid_n)
    occ[:, 10:20, 10:20] = 1.0
    action = torch.tensor([
        [0.0, 0.0, 0.02, 0.0],
        [-0.01, 0.01, 0.01, 0.01],
    ])
    with torch.no_grad():
        occ_pred = wrapper.predict_one_step_occ(occ, action)
    assert occ_pred.shape == occ.shape
    assert torch.isfinite(occ_pred).all()
