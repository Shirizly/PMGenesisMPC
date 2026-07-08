import torch
from GranularDynamics2.myClasses.NFDUNetFilm import NFDUNetFiLM
from registry.model_registry import build_model

# --- Test Setup ---
# Use the new configurable model and a sample config dictionary
CONFIG = {
    "type": "unetfilm",
    "cond_dim": 3,
    "uses_physics": True,
    "input_mode": "standard",
    "depth": 2, # Test with depth=2
    "in_channels": 2,
    "residual_channel": 0,
}

# --- Model Initialization and Testing ---
try:
    print("--- Initializing Configurable NFDUNetFiLM model (Depth=2) ---")
    model = build_model(CONFIG)
    print("Model successfully built.")

    # Dummy input data for testing the forward pass
    B, C, H, W = 4, 2, 64, 64 # Batch size, Channels, Height, Width
    dummy_input = torch.randn(B, C, H, W)
    dummy_props = torch.randn(B, CONFIG["cond_dim"])

    print("--- Running forward pass test ---")
    with torch.no_grad():
        output = model(dummy_input, dummy_props)

    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.shape == (B, 1, H, W)
    print(f"Forward pass successful! Output shape: {output.shape}")

except Exception as e:
    print(f"--- Test FAILED ---")
    print(f"An error occurred during model initialization or forward pass: {e}")