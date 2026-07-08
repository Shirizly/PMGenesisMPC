import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffRenderer(nn.Module):
    """
    Differentiable renderer for particle-based states and tool masks.
    
    Args:
        height (int): Height of the output image in pixels.
        width (int): Width of the output image in pixels.
        sigma (float): Standard deviation of the Gaussian splat for particles.
        
    TODO:
      - Support shape outline input as a matrix (for future extension).
      - Tool mask rendering: apply scaling, rotation, and blur.
    """
    def __init__(self, height: int, width: int, sigma: float, tool_source):
        super().__init__()
        self.height = height
        self.width = width
        self.sigma = sigma
        
        # Pre-compute grid for Gaussian splats
        ys = torch.linspace(0, height - 1, height)
        xs = torch.linspace(0, width - 1, width)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        # Register as buffers so they move to device with the module
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

        self.tool_source = tool_source 
        tool_mask = 
        
    def forward(self, particle_positions: torch.Tensor, tool_mask: torch.Tensor, tool_params: torch.Tensor):
        """
        Render a multi-channel mask of particles and tool.
        
        Args:
            particle_positions (Tensor[N, 2]): x, y positions of N particles in pixel coordinates.
            tool_mask (Tensor[H_tool, W_tool]): binary mask of the tool shape.
            tool_params (Tensor[3]): (x, y, theta) for tool placement in pixel coords and radians.
        
        Returns:
            Tensor[3, H, W]: three-channel rendered image (particle concentration, tool mask before, tool mask after).
        """
        # 1) Render particles via Gaussian splats
        # particle_positions: (N, 2)
        px = particle_positions[:, 0].unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        py = particle_positions[:, 1].unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        
        # Compute squared distance from each grid point to each particle
        dist2 = (self.grid_x.unsqueeze(0) - px) ** 2 + (self.grid_y.unsqueeze(0) - py) ** 2  # (N, H, W)
        gauss = torch.exp(-dist2 / (2 * self.sigma ** 2))  # (N, H, W)
        particle_img = gauss.sum(dim=0, keepdim=True)  # (1, H, W)
        
        # 2) Placeholder for tool rendering
        # TODO: apply scaling, rotation (theta), translation (x,y) to tool_mask
        # and then blur the edges for differentiability.
        tool_img = torch.zeros_like(particle_img)
        
        # Combine layers
        rendered = particle_img 
        # Optionally normalize or clamp
        rendered = rendered.clamp(0.0, 1.0)
        return rendered

# Example usage
if __name__ == "__main__":
    # Create renderer
    renderer = DiffRenderer(height=64, width=64, sigma=2.0)
    
    # Dummy particles
    particles = torch.tensor([[20.0, 30.0], [40.0, 10.0]])
    # Dummy tool mask (to be implemented)
    tool_mask = torch.zeros(16, 16)
    # Dummy tool params: x, y, theta
    tool_params = torch.tensor([32.0, 32.0, 0.0])
    
    # Forward render
    img = renderer(particles, tool_mask, tool_params)
    print("Rendered image shape:", img.shape)
