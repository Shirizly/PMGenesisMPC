"""Registry smoke tests for the newly integrated futureintegration models."""

import torch

from registry.model_registry import build_model

B, C, H, W = 2, 2, 64, 64
COND_DIM = 3


def test_nca_forward():
    wrapper = build_model({"type": "nca"})
    batch = {
        "input": torch.randn(B, C, H, W),
        "physics": torch.randn(B, COND_DIM),
    }
    with torch.no_grad():
        output = wrapper(batch)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (B, 1, H, W)


def test_spatial_transformer_forward():
    wrapper = build_model({"type": "spatial-transformer"})
    batch = {
        "input": torch.randn(B, C, H, W),
        "physics": torch.randn(B, COND_DIM),
    }
    with torch.no_grad():
        output = wrapper(batch)
    assert output.shape == (B, 1, H, W)
    # Jacobian regularizer side effect is populated after forward.
    assert hasattr(wrapper.model, "last_j_loss")


def test_unet_modular_forward_no_physics():
    wrapper = build_model({"type": "unet-modular"})
    with torch.no_grad():
        output = wrapper({"input": torch.randn(B, C, H, W)})
    assert output.shape == (B, 1, H, W)


def test_unet_modular_bottleneck_variants():
    for bottleneck_type, kwargs in [
        ("SE", {}),
        ("FC_channel", {}),
        ("Transformer", {"num_heads": 2}),
    ]:
        wrapper = build_model({
            "type": "unet-modular",
            "features": [4, 8],
            "bottleneck_type": bottleneck_type,
            "bottleneck_kwargs": kwargs,
        })
        with torch.no_grad():
            output = wrapper({"input": torch.randn(B, C, H, W)})
        assert output.shape == (B, 1, H, W), bottleneck_type
