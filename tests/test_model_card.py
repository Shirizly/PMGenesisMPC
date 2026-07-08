"""Round-trip test: save a checkpoint + model card, reload via load_net_from_card."""

import torch
import yaml

from model.model_card import ModelCard, load_net_from_card
from physics.normalization import PhysicsBounds
from registry.model_registry import build_model

MODEL_CFG = {
    "type": "unetfilm",
    "in_channels": 2,
    "cond_dim": 3,
    "uses_physics": True,
    "depth": 2,
}


def test_load_net_from_card_roundtrip(tmp_path):
    wrapper = build_model(MODEL_CFG)
    ckpt = tmp_path / "unet_best.pth"
    torch.save(wrapper.state_dict(), ckpt)

    card = ModelCard(
        path=tmp_path / "model_card.yaml",
        model_cfg={**MODEL_CFG, "checkpoint": "unet_best.pth"},
        physics_bounds=PhysicsBounds.default(),
        inference_cfg={"representation": "eulerian", "grid_n": 64},
    )
    card.save()

    net, loaded_card = load_net_from_card(card.path)
    assert loaded_card.representation == "eulerian"
    assert not net.training  # eval mode

    x, p = torch.randn(2, 2, 64, 64), torch.randn(2, 3)
    with torch.no_grad():
        out = net(x, p)
    assert out.shape == (2, 1, 64, 64)

    # Weights actually came from the checkpoint
    orig = wrapper.state_dict()
    for k, v in net.state_dict().items():
        assert torch.equal(v, orig[k])


def test_card_yaml_schema(tmp_path):
    card = ModelCard(
        path=tmp_path / "model_card.yaml",
        model_cfg={**MODEL_CFG, "checkpoint": "unet_best.pth"},
        physics_bounds=PhysicsBounds.default(),
        inference_cfg={"representation": "eulerian"},
    )
    card.save()
    raw = yaml.safe_load(card.path.read_text())
    assert set(raw) == {"model", "physics", "inference"}
    assert raw["model"]["type"] == "unetfilm"
