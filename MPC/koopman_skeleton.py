### First: define the Phase 1 Geometric Encoder and its loss function

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Standard CNN backbone to extract structural features
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # [B, 32, H/2, W/2]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # [B, 64, H/4, W/4]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, H/8, W/8]
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # Project to the target latent dimension
        self.fc = nn.Linear(128 * 8 * 8, latent_dim) # Assuming 64x64 input grid

    def forward(self, x):
        features = self.backbone(x)
        z = self.fc(features)
        return z

def compute_phase1_loss(z_batch, shape_targets=None):
    """
    Enforces non-collapse and properties like geometric variance.
    shape_targets: Optional labels for specific axes (e.g., circularity, left/right mass)
    """
    B, D = z_batch.shape
    
    # 1. Variance Loss: Force each latent dimension to vary across the batch (prevents zero-collapse)
    std_z = torch.sqrt(z_batch.var(dim=0) + 1e-04)
    variance_loss = torch.mean(F.relu(1.0 - std_z))
    
    # 2. Covariance Loss: Penalize off-diagonal correlations to ensure distinct, orthogonal axes
    z_centered = z_batch - z_batch.mean(dim=0)
    cov_matrix = (z_centered.T @ z_centered) / (B - 1)
    diag_mask = torch.eye(D, device=z_batch.device)
    cov_loss = (cov_matrix * (1 - diag_mask)).pow(2).sum() / D
    
    # 3. Task-Specific Supervised Alignment (Optional)
    # E.g., if you have pre-calculated metrics like Center of Mass or Fourier descriptors
    target_loss = 0.0
    if shape_targets is not None:
        # Align the first N dimensions with your known explicit properties
        target_loss = F.mse_loss(z_batch[:, :shape_targets.shape[1]], shape_targets)
        
    total_loss = variance_loss + 0.01 * cov_loss + 1.0 * target_loss
    return total_loss


### Next: Define the Koopman Transition Model and Koopman-JEPA loss for Phase 2

class KoopmanTransitionModel(nn.Module):
    def __init__(self, encoder: GeometricEncoder, latent_dim=512, action_dim=4):
        super().__init__()
        self.encoder = encoder  # Composition: Embeds the Phase 1 encoder
        self.latent_dim = latent_dim
        
        # Koopman Linear Operators
        # x_next = K_z * z + K_u * u
        self.K_z = nn.Parameter(torch.eye(latent_dim))
        self.K_u = nn.Parameter(torch.randn(latent_dim, action_dim) * 0.01)

    def forward_one_step(self, z, u):
        """Advances the latent state by one micro-step linearly."""
        # z: [B, latent_dim], u: [B, action_dim]
        z_next = (self.K_z @ z.T).T + (self.K_u @ u.T).T
        return z_next

    def forward_multi_step(self, init_density, tool_trajectory):
        """
        Rolls out the simulation entirely in the latent space over M micro-steps.
        tool_trajectory: [B, M_steps, action_dim]
        """
        B, M, _ = tool_trajectory.shape
        
        # 1. Encode the initial true state into the Koopman Space
        z_current = self.encoder(init_density)
        
        latent_predictions = []
        for step in range(M):
            u_t = tool_trajectory[:, step, :]
            # Linear evolution
            z_current = self.forward_one_step(z_current, u_t)
            latent_predictions.append(z_current)
            
        return latent_predictions # List of M predicted states in latent space





def compute_koopman_jepa_loss(model, density_sequence, tool_trajectory):
    """
    Args:
        density_sequence: List or tensor of real observed frames [B, M_steps, 1, H, W]
        tool_trajectory:  Actions executed at each micro-step [B, M_steps, action_dim]
    """
    B, M, _, H, W = density_sequence.shape
    
    # Extract the initial state to seed the model
    init_density = density_sequence[:, 0, :, :, :]
    
    # 1. Get linear predictions from the model rollout
    predicted_latents = model.forward_multi_step(init_density, tool_trajectory)
    
    multi_step_loss = 0.0
    variance_loss = 0.0
    
    # 2. Evaluate each step against the target encoded frame
    for step in range(M):
        z_pred = predicted_latents[step]
        
        # Encode the true observed state at time t+1 (Detached target for JEPA setup)
        with torch.no_grad():
            z_true = model.encoder(density_sequence[:, step + 1, :, :, :])
            
        # Linear tracking metric
        multi_step_loss += F.mse_loss(z_pred, z_true)
        
        # JEPA Variance Anchor: Keep the online encoder dynamic during rollout optimization
        std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-04)
        variance_loss += torch.mean(F.relu(1.0 - std_pred))
        
    return (multi_step_loss / M) + (variance_loss / M)





### The training loop is divided into two phases:
# ==========================================
# STEP 1: PRETRAIN THE ENCODER
# ==========================================
encoder = GeometricEncoder(latent_dim=512)
optimizer_p1 = torch.optim.AdamW(encoder.parameters(), lr=1e-4)

for epoch in range(num_phase1_epochs):
    for batch_density, batch_geometric_targets in pretrain_loader:
        optimizer_p1.zero_grad()
        z = encoder(batch_density)
        loss = compute_phase1_loss(z, batch_geometric_targets)
        loss.backward()
        optimizer_p1.step()

# ==========================================
# STEP 2: TRAIN KOOPMAN TRANSITION OPERATORS
# ==========================================
# Pass the pretrained encoder into the transition model
model = KoopmanTransitionModel(encoder=encoder, latent_dim=512, action_dim=4)

# Fine-tune the encoder alongside the operators with a lower learning rate
optimizer_p2 = torch.optim.AdamW([
    {'params': [model.K_z, model.K_u], 'lr': 1e-3},
    {'params': model.encoder.parameters(), 'lr': 1e-5} 
])

for epoch in range(num_phase2_epochs):
    for density_seq, tool_seq in transition_loader: # Sequential time-series batches
        optimizer_p2.zero_grad()
        loss = compute_koopman_jepa_loss(model, density_seq, tool_seq)
        loss.backward()
        optimizer_p2.step()