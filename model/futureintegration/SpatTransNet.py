"""
Spatial Transformer Network (STN) for Eulerian Granular Dynamics Prediction
--------------------------------------------------------------
This module implements a Spatial Transformer Network (STN) designed to predict the next state of a granular system given the current state and a tool action. The STN learns to generate a dense flow
field that warps the current density map to produce the predicted next state. The architecture consists of:
1. Localization Network: A convolutional network that takes the current density and tool action as input
    and outputs a dense displacement field (dx, dy) for each pixel.
2. Grid Generator: Creates a sampling grid by adding the predicted displacements to a base coordinate grid.
3. Bilinear Sampler: Uses the sampling grid to warp the current density map, producing the predicted next state.
The STN is trained with a combination of reconstruction loss (MSE between predicted and true next state) and a regularization loss that encourages the predicted flow to be smooth and physically plausible.
Inputs:
    current_density: (B, 1, H, W) - Grayscale image representing the current state of the granular system.
    tool_action:     (B, 1, H, W) - Grayscale image representing the swept area of the tool action.
    props:           (B, P)       - Additional properties (e.g., pusher width, density) that can be concatenated to the input channels if needed.
Outputs:
    next_density:    (B, 1, H, W) - Predicted next state of the granular system after applying the tool action.
    jacobian_loss:   Scalar       - Regularization loss based on the Jacobian determinant of the flow field to encourage incompressibility.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EulerianSTN(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        depth: int = 3,
        flow_scale: float = 0.1,
    ):
        """
        Args:
            in_channels:   Number of input channels (default 2: density + action).
            base_channels: Channel width of the first hidden conv layer.  Subsequent
                           hidden layers double up to depth//2 then halve back down,
                           giving an hourglass profile.  E.g. depth=3, base=32 →
                           [32, 64, 32]; depth=4, base=16 → [16, 32, 32, 16].
            depth:         Number of hidden conv layers (not counting the final 2-ch
                           flow output layer).  Must be >= 1.
            flow_scale:    Scalar applied to the raw flow output so the model starts
                           near identity.  Smaller values → more conservative initial
                           displacements.
        """
        super().__init__()
        self.flow_scale = flow_scale

        # Build hourglass channel schedule: ramp up to peak at depth//2 then back down.
        # Always start and end at base_channels; peak is base_channels * 2^(depth//2).
        half = depth // 2
        channels = [base_channels * (2 ** min(i, depth - 1 - i)) for i in range(depth)]

        layers: list[nn.Module] = []
        in_ch = in_channels
        for out_ch in channels:
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_ch = out_ch
        # Final layer: outputs the 2-channel (dx, dy) flow field
        layers.append(nn.Conv2d(in_ch, 2, kernel_size=3, padding=1))

        self.localization = nn.Sequential(*layers)

        # Zero-initialize the final layer so initial predictions cause zero movement
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.zero_()

    def create_base_grid(self, x):
        """Generates a standard normalized (-1 to 1) coordinate grid."""
        B, _, H, W = x.size()
        # meshgrid generates coordinates matching grid_sample expectations
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        # Shape: [1, H, W, 2] -> replicated to match batch size [B, H, W, 2]
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        return base_grid.repeat(B, 1, 1, 1)

    def compute_jacobian_loss(self, flow, base_grid):
        """
        Calculates how much the transformation stretches or compresses space.
        Ideally, Jacobian Determinant (det(J)) == 1 for area-preserving flow.
        """
        # Total sampling grid map
        grid = base_grid + flow
        
        # Compute spatial gradients of the lookup coordinates map
        # dy represents change along rows, dx along columns
        dy, dx = torch.gradient(grid, dim=[1, 2])
        
        dx_dx = dx[..., 0]
        dx_dy = dx[..., 1]
        dy_dx = dy[..., 0]
        dy_dy = dy[..., 1]
        
        # Determinant of the 2D Jacobian matrix.
        # grid has shape (B, H, W, 2); dy = gradient along H (rows), dx = gradient along W (cols).
        # J maps (row, col) -> (x_sample, y_sample), so:
        #   det(J) = (∂x/∂row)(∂y/∂col) - (∂x/∂col)(∂y/∂row) = dy_dx*dx_dy - dx_dx*dy_dy
        det_J = dy_dx * dx_dy - dx_dx * dy_dy

        # |det| == 1 → area-preserving (incompressible) backward warp.
        # Using abs makes the loss symmetric with respect to orientation flips.
        jacobian_loss = F.mse_loss(det_J.abs(), torch.ones_like(det_J))
        return jacobian_loss

    def forward(self, x: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
        """
        Matches the training framework API used by NFDUNetFiLM and NCAWithPhysics.

        Args:
            x:     (B, 2, H, W) — channel 0: current density, channel 1: tool action
            props: (B, P)       — material/physics properties (not used; accepted for API
                                  compatibility; add FiLM conditioning here to use them)
        Returns:
            logit: (B, 1, H, W) — raw logit; apply sigmoid to obtain occupancy probability.

        Side-effect:
            self.last_j_loss is updated to the Jacobian regularisation loss of this
            forward pass so the training loop can add JAC_WEIGHT * model.last_j_loss
            to the total loss.
        """
        current_density = x[:, 0:1]   # (B, 1, H, W)
        tool_action     = x[:, 1:2]   # (B, 1, H, W)

        # Stack inputs along the channel dimension
        inp = torch.cat([current_density, tool_action], dim=1)  # (B, 2, H, W)

        # Predict local displacements (dx, dy) for backward lookup.
        # flow_scale prevents large unphysical jumps early in training.
        flow_displacement = self.localization(inp) * self.flow_scale

        # Permute from (B, 2, H, W) to (B, H, W, 2) to match grid_sample convention
        flow_displacement = flow_displacement.permute(0, 2, 3, 1)

        # 2. Grid Generator
        base_grid = self.create_base_grid(current_density)
        sampling_grid = base_grid + flow_displacement
        # NOTE: do NOT clamp sampling_grid here — out-of-bounds coords are already
        # handled by padding_mode='zeros', which sets sampled values to 0.
        # Clamping would defeat that behaviour by silently projecting onto the border.

        # 3. Bilinear Sampler
        next_density = F.grid_sample(
            current_density,
            sampling_grid,
            mode='bilinear',
            padding_mode='zeros',   # mass sampled outside the map is treated as zero
            align_corners=True,
        )

        # Jacobian regularisation loss (stored for the training loop; not returned)
        self.last_j_loss = self.compute_jacobian_loss(flow_displacement, base_grid)

        # Convert probability in [0,1] to an unbounded logit so combined_loss can
        # apply sigmoid and compute MSE/BCE against the target the same way as UNet/NCA.
        logit = torch.logit(next_density.clamp(1e-5, 1.0 - 1e-5))
        return logit