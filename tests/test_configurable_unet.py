"""Registry smoke tests: configurable-depth NFDUNetFiLM built via build_model."""

import pytest
import torch

from registry.model_registry import build_model

B, C, H, W = 4, 2, 64, 64
COND_DIM = 3


def _config(depth: int, model_type: str = "unetfilm") -> dict:
    return {
        "type": model_type,
        "cond_dim": COND_DIM,
        "uses_physics": True,
        "input_mode": "standard",
        "depth": depth,
        "in_channels": C,
        "residual_channel": 0,
    }


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_unetfilm_configurable_depth_forward(depth):
    wrapper = build_model(_config(depth))
    batch = {
        "input": torch.randn(B, C, H, W),
        "physics": torch.randn(B, COND_DIM),
    }
    with torch.no_grad():
        output = wrapper(batch)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (B, 1, H, W)


def test_unetfilm_missing_physics_fails_loudly():
    wrapper = build_model(_config(depth=2))
    with pytest.raises(ValueError, match="physics"):
        wrapper({"input": torch.randn(B, C, H, W)})


def test_unetfilm_default_channels_double():
    wrapper = build_model(_config(depth=3))
    assert wrapper.model.channels == [8, 16, 32]


def test_unetfilm_explicit_channels_override_depth():
    cfg = _config(depth=2)
    cfg["channels"] = [4, 8, 16, 32]
    wrapper = build_model(cfg)
    assert wrapper.model.channels == [4, 8, 16, 32]
    assert wrapper.model.depth == 4
    batch = {
        "input": torch.randn(B, C, H, W),
        "physics": torch.randn(B, COND_DIM),
    }
    with torch.no_grad():
        output = wrapper(batch)
    assert output.shape == (B, 1, H, W)


def test_unetfilm_loads_legacy_state_dict():
    # Pre-configurable checkpoints use enc{n}/dec{n} names; the model remaps
    # them onto encoder_blocks/decoder_blocks on load.
    wrapper = build_model(_config(depth=3))
    legacy = {}
    for key, value in wrapper.model.state_dict().items():
        legacy_key = (
            key.replace("encoder_blocks.", "enc__").replace("decoder_blocks.", "dec__")
        )
        if "__" in legacy_key:
            tag, rest = legacy_key.split("__", 1)
            idx, tail = rest.split(".", 1)
            legacy_key = f"{tag}{int(idx) + 1}.{tail}"
        legacy[legacy_key] = value.clone()
    assert any(k.startswith("enc1.") for k in legacy)

    fresh = build_model(_config(depth=3))
    fresh.model.load_state_dict(legacy)
    for key, value in fresh.model.state_dict().items():
        assert torch.equal(value, wrapper.model.state_dict()[key])


def test_unetfilm_shallow_builds():
    wrapper = build_model(_config(depth=2, model_type="unetfilm-shallow"))
    batch = {
        "input": torch.randn(B, C, H, W),
        "physics": torch.randn(B, COND_DIM),
    }
    with torch.no_grad():
        output = wrapper(batch)
    assert output.shape == (B, 1, H, W)
