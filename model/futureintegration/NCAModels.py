import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# NCA single-step module
# -------------------------
class NCAUpdate(nn.Module):
    """
    One local NCA update step.
    Input: grid (B, C_in, H, W) where C_in includes current state channel(s) + action channels.
    Output: delta to be added to the state channel(s) (same spatial size).
    Implementation: small conv net with 3x3 perception and 1x1 output.
    """
    def __init__(self, in_ch, hidden_ch=8, out_ch=1, dt=0.5):
        super().__init__()
        self.out_ch = out_ch
        self.dt = dt  # step size for residual update

        # perception conv: shareable 3x3 that extracts local neighborhood info
        self.perception = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # compute delta for state channels
        self.delta_layer = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)

        # optional gating for stability
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1),
            nn.Tanh()  # gate in [-1,1]
        )

    def forward(self, grid):
        """
        grid: (B, C_in, H, W)
        returns: delta (B, state_channels, H, W)
        """
        feat = self.perception(grid)
        raw_delta = self.delta_layer(feat)
        gate = self.gate(feat)  # optional multiplicative gating
        delta = self.dt * raw_delta * (1.0 + gate)  # scale and gate
        return delta


# -------------------------
# Stack K NCA steps
# -------------------------
class NCAStack(nn.Module):
    """
    Apply the same NCAUpdate K times.
    Optionally mask updates (stochastic or deterministic).
    """
    def __init__(self, in_ch, out_ch=1, hidden_ch=64, steps=8, dt=0.5):
        super().__init__()
        self.steps = steps
        self.nca = NCAUpdate(in_ch, hidden_ch=hidden_ch, out_ch=out_ch, dt=dt)

    def forward(self, input, aux_channels=None, steps=None):
        """
        state: (B, state_ch, H, W)
        action_grid: (B, A, H, W)  -- tool geometry / action encodings
        aux_channels: optional (B, M, H, W) extra channels (swept mask, distance, etc.)
        steps: override the number of update steps (defaults to self.steps).
        returns: cumulative_delta (B, state_ch, H, W), state_after_nca (state + cumulative_delta)
        """
        _steps = self.steps if steps is None else steps
        state = input[:,0:1,:,:]
        action_grid = input[:,1:,:,:] if input.shape[1]>1 else None
        B, sc, H, W = state.shape
        cum_delta = torch.zeros_like(state)

        # build static part of grid that doesn't change across steps (actions, aux)
        static_parts = [action_grid] if action_grid is not None else []
        if aux_channels is not None:
            static_parts.append(aux_channels)

        # At each step we provide the current state plus static channels
        cur_state = state
        for t in range(_steps):
            grid_inputs = [cur_state] + static_parts  # current dynamic state + static fields
            grid = torch.cat(grid_inputs, dim=1)  # (B, C_in, H, W)
            delta = self.nca(grid)  # (B, state_ch, H, W)
            cur_state = cur_state + delta
            cum_delta = cum_delta + delta

        return cum_delta, cur_state


# -------------------------
# Small residual UNet correction
# -------------------------
class SmallUNetRes(nn.Module):
    """
    Small UNet-like residual correction head with variable depth.
    Input: state_after_nca concatenated with action + aux -> (B, C, H, W)
    Output: delta_correction (B, state_ch, H, W)
    `features` controls both depth and width: len(features) encoder stages are
    built, each halving spatial resolution, with a symmetric decoder and additive
    skip connections.
    """
    def __init__(self, in_ch, out_ch=1, features=[8, 16, 32]):
        super().__init__()
        n = len(features)

        # encoder: first block is a double-conv stem (no pool), rest pool+conv
        self.enc_layers = nn.ModuleList()
        self.enc_layers.append(nn.Sequential(
            nn.Conv2d(in_ch, features[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], 3, padding=1),
            nn.ReLU(inplace=True),
        ))
        for i in range(1, n):
            self.enc_layers.append(nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(features[i - 1], features[i], 3, padding=1),
                nn.ReLU(inplace=True),
            ))

        # bottleneck
        self.bot = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # decoder: len(features)-1 upsample stages (no upsample after bottleneck)
        self.dec_layers = nn.ModuleList()
        for i in range(n - 1, 0, -1):
            self.dec_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(features[i], features[i - 1], 3, padding=1),
                nn.ReLU(inplace=True),
            ))

        self.outc = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, state_after, action_grid, aux_channels=None):
        # build input
        parts = [state_after, action_grid] if action_grid is not None else [state_after]
        if aux_channels is not None:
            parts.append(aux_channels)
        x = torch.cat(parts, dim=1)

        # encoder — store skip features
        skips = []
        for enc in self.enc_layers:
            x = enc(x)
            skips.append(x)

        x = self.bot(x)

        # decoder — add skips in reverse order (skip from matching encoder level)
        for i, dec in enumerate(self.dec_layers):
            x = dec(x)
            x = x + skips[-(i + 2)]  # skips[-1] was the deepest encoder output (= bot input)

        delta_corr = self.outc(x)
        return delta_corr


# -------------------------
# Full Model: NCA + UNet Correction
# -------------------------
class NCAPlusUNet(nn.Module):
    """
    state: (B, state_ch, H, W)
    action_grid: (B, A, H, W)  -- tool rasterization channels, etc.
    aux_channels: (B, M, H, W) optional
    """
    def __init__(self, in_ch=1, out_ch=1,
                 nca_hidden=8, nca_steps=8, unet_features=[8, 16, 32]):
        super().__init__()
        self.in_ch = in_ch

        self.nca_stack = NCAStack(in_ch=in_ch, out_ch=out_ch,
                                  hidden_ch=nca_hidden, steps=nca_steps, dt=0.5)

        # UNet correction input channels = state_after + action + aux
        self.corr_unet = SmallUNetRes(in_ch=in_ch, out_ch=out_ch, features=unet_features)

    def forward(self, input, steps=None):
        """
        returns: next_state_pred, dict with internals
        steps: optional override for NCA step count (passed to NCAStack).
        """
        # NCA local iterative updates
        cum_delta, state_after = self.nca_stack(input, steps=steps)
        action = input[:,1:,:,:] 
        # UNet residual correction (one global pass)
        delta_corr = self.corr_unet(state_after, action)

        # Final prediction (residual)
        next_state = state_after + delta_corr  # equivalently: state + cum_delta + delta_corr

        return next_state#, {"delta_nca": cum_delta, "delta_corr": delta_corr}


# -------------------------
# Physics-conditioned wrapper (matches NFDUNetFiLM API)
# -------------------------
class NCAWithPhysics(nn.Module):
    """
    Wraps NCAPlusUNet to accept the same call signature as NFDUNetFiLM:
        forward(x: Tensor[B, in_channels, H, W], props: Tensor[B, physics_dim])
            -> Tensor[B, 1, H, W]

    Physics conditioning is done by broadcasting the physics vector to spatial
    feature maps and concatenating with the input, rather than FiLM modulation.
    The NCA sees physics at every local step as an additional set of static channels.

    The output is a raw (unbounded) value, treated as a logit by the training
    pipeline: apply torch.sigmoid() to obtain an occupancy probability in (0, 1).

    Design notes vs. NFDUNetFiLM:
    - No FiLM: physics enters as extra input channels, not as scale/bias of hidden
      features. This is simpler but gives the NCA local update rule direct access
      to physics values rather than learned modulation of intermediate activations.
    - Residual structure: the NCA starts from the input state and accumulates
      delta updates, so the output is anchored near the input occupancy at
      initialisation. The training loss (MSE between sigmoid(output) and target)
      still produces well-scaled gradients because sigmoid'(x) > 0 everywhere.
    - For MPC use (UNetFiLMPushModel equivalent): apply sigmoid to the output
      before using it as an occupancy probability, exactly as described in
      CODEBASE_OVERVIEW.md §11.4. The gradient will flow through sigmoid cleanly
      since the output is in a range where sigmoid is not saturated early in
      training.
    """

    def __init__(
        self,
        in_channels: int = 2,
        physics_dim: int = 3,
        nca_hidden: int = 64,
        nca_steps: int = 8,
        unet_features: list = None,
    ):
        super().__init__()
        if unet_features is None:
            unet_features = [32, 64, 128]
        self.physics_dim = physics_dim
        total_in_ch = in_channels + physics_dim
        self.nca = NCAPlusUNet(
            in_ch=total_in_ch,
            out_ch=1,
            nca_hidden=nca_hidden,
            nca_steps=nca_steps,
            unet_features=unet_features,
        )

    def forward(self, x: torch.Tensor, props: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        """
        x:     (B, in_channels, H, W)  — occupancy + tool action channels
        props: (B, physics_dim)         — [friction, density, box_friction]
        steps: optional per-batch NCA step override (passed through to NCAStack).
        returns: (B, 1, H, W)           — raw logit (apply sigmoid for probs)
        """
        B, C, H, W = x.shape
        physics_spatial = props[:, :, None, None].expand(B, self.physics_dim, H, W)
        x_cond = torch.cat([x, physics_spatial], dim=1)  # (B, C + physics_dim, H, W)
        return self.nca(x_cond, steps=steps)


# -------------------------
# Utilities / quick test
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, W = 2, 128, 128
    in_channels = 2   # occupancy + action
    physics_dim = 3   # friction, density, box_friction

    # Test NCAWithPhysics (the API-compatible wrapper used by train_NCA_genesis.py)
    model = NCAWithPhysics(
        in_channels=in_channels,
        physics_dim=physics_dim,
        nca_hidden=64,
        nca_steps=8,
        unet_features=[32, 64, 128],
    ).to(device)

    x = torch.rand(B, in_channels, H, W, device=device)
    props = torch.rand(B, physics_dim, device=device)

    with torch.no_grad():
        out = model(x, props)
    print("out.shape:", out.shape)             # (B, 1, H, W)
    print("probs range:", torch.sigmoid(out).min().item(), torch.sigmoid(out).max().item())
    print("num params:", sum(p.numel() for p in model.parameters()))

    # Test bare NCAPlusUNet
    model2 = NCAPlusUNet(in_ch=in_channels, nca_hidden=48, nca_steps=12,
                         unet_features=[32, 64, 128]).to(device)
    inp = torch.rand(B, in_channels, H, W, device=device)
    with torch.no_grad():
        out2 = model2(inp)
    print("NCAPlusUNet out.shape:", out2.shape)
    print("num params:", sum(p.numel() for p in model2.parameters()))
