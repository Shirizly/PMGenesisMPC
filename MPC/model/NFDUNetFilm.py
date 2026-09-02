"""
NFD Shallow U-Net + FiLM conditioning
Xue et al., CoRL 2023  —  extended with Feature-wise Linear Modulation.

FiLM conditions every encoder and decoder conv block on a vector of
object/material properties (e.g. friction coefficient, object size, pusher width).

    FiLM(x | z) = γ(z) ⊙ x + β(z)

where γ, β ∈ R^C are produced by a small MLP from the property vector z,
and applied channel-wise after each conv+ReLU block.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.
"""

from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# FiLM generator  (one per block — each block has its own γ/β head)
# ─────────────────────────────────────────────────────────────────────────────

class FiLMGenerator(nn.Module):
    """
    Maps a property vector z ∈ R^{cond_dim} to per-channel scale γ and shift β.

    Args:
        cond_dim : dimensionality of the conditioning vector z
        num_ch   : number of feature-map channels C to modulate
        hidden   : hidden size of the 1-layer MLP
    """
    def __init__(self, cond_dim: int, num_ch: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_ch * 2),   # outputs [γ | β]
        )
        # Init γ ≈ 1, β ≈ 0  so FiLM is identity at the start of training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias[:num_ch],  1.0)   # γ = 1
        nn.init.constant_(self.mlp[-1].bias[num_ch:],  0.0)   # β = 0

    def forward(self, z: torch.Tensor):
        """
        Args:
            z : (B, cond_dim)
        Returns:
            gamma : (B, C, 1, 1)
            beta  : (B, C, 1, 1)
        """
        out   = self.mlp(z)                           # (B, 2C)
        C     = out.shape[1] // 2
        gamma = out[:, :C].unsqueeze(-1).unsqueeze(-1)
        beta  = out[:, C:].unsqueeze(-1).unsqueeze(-1)
        return gamma, beta


# ─────────────────────────────────────────────────────────────────────────────
# Conv block  +  FiLM modulation
# ─────────────────────────────────────────────────────────────────────────────

class FiLMConvBlock(nn.Module):
    """
    3×3 Conv → ReLU → FiLM(γ, β)

    FiLM is applied after ReLU so the modulation acts on the activated features.
    """
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, hidden: int = 64):
        super().__init__()
        self.conv      = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.relu      = nn.ReLU(inplace=True)
        self.film_gen  = FiLMGenerator(cond_dim, out_ch, hidden)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_ch, H, W)  feature map
            z : (B, cond_dim)     conditioning vector
        Returns:
            out : (B, out_ch, H, W)  modulated feature map
        """
        x             = self.relu(self.conv(x))
        gamma, beta   = self.film_gen(z)
        return gamma * x + beta


# ─────────────────────────────────────────────────────────────────────────────
# NFD U-Net  with FiLM at every encoder + decoder block (Configurable Depth)
# ─────────────────────────────────────────────────────────────────────────────

def _remap_legacy_state_dict(state_dict: dict) -> dict:
    """
    Map pre-configurable checkpoint keys onto the ModuleList layout:
    ``enc{n}.*`` → ``encoder_blocks.{n-1}.*`` and ``dec{n}.*`` →
    ``decoder_blocks.{n-1}.*`` (``bottleneck``/``head`` are unchanged).
    Weight shapes match because the default channel scheme reproduces the
    original doubling widths.
    """
    remapped = {}
    for key, value in state_dict.items():
        m = re.match(r"^(enc|dec)(\d+)\.(.+)$", key)
        if m:
            prefix = "encoder_blocks" if m.group(1) == "enc" else "decoder_blocks"
            remapped[f"{prefix}.{int(m.group(2)) - 1}.{m.group(3)}"] = value
        else:
            remapped[key] = value
    return remapped


class NFDUNetFiLM(nn.Module):
    """
    NFD dynamics model conditioned on material properties via FiLM, with configurable depth.

    The encoder widths double per level by default (base * 2**i — the original
    fixed-depth architecture at depth 3), or can be given explicitly.

    Args:
        in_channels : input channels  (default 2)
        out_channels: output channels (default 1)
        cond_dim    : dimensionality of the property/conditioning vector z
        base        : base channel width (default 8)
        depth       : number of pooling levels (encoder stages). Must be >= 1.
        channels    : optional explicit per-level encoder widths, e.g.
                      [8, 16, 32]. Overrides base/depth when given.
        film_hidden : hidden size of each FiLM MLP (default 64)
        residual_channel: which input channel to use as residual skip (default 0)
    """

    def __init__(
        self,
        in_channels:  int = 2,
        out_channels: int = 1,
        cond_dim:     int = 3,
        base:         int = 8,
        depth:        int = 3,
        channels:     Sequence[int] | None = None,
        film_hidden:  int = 64,
        residual_channel: int = 0,
    ):
        super().__init__()
        self.residual_channel = residual_channel

        if channels is not None:
            channels = [int(c) for c in channels]
            if not channels or any(c < 2 for c in channels):
                raise ValueError(f"channels must be a non-empty list of ints >= 2, got {channels!r}")
            self.depth = len(channels)
        else:
            self.depth = max(1, depth)
            channels = [base * 2 ** i for i in range(self.depth)]
        self.channels = channels

        kw = dict(cond_dim=cond_dim, hidden=film_hidden)

        # --- Encoder Stages ---
        self.encoder_blocks = nn.ModuleList()
        current_in_ch = in_channels
        for out_ch in channels:
            self.encoder_blocks.append(FiLMConvBlock(current_in_ch, out_ch, **kw))
            current_in_ch = out_ch

        # Pool layer is always used after the first encoder stage (i=0) up to depth-2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck Stage ---
        self.bottleneck = FiLMConvBlock(channels[-1], channels[-1], **kw)

        # --- Decoder Stages (Dynamic Construction) ---
        # Built into pre-sized lists indexed by level `i` (not appended) so that
        # forward()'s `self.decoder_blocks[i]` / `self.up_layers[i]` lookups line
        # up with the level they were constructed for.
        decoder_blocks_by_level = [None] * self.depth
        up_layers_by_level = [None] * self.depth

        prev_out_ch = channels[-1]  # channel count feeding into the next upsample

        for i in range(self.depth - 1, -1, -1):
            up_layers_by_level[i] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            # Input channels for the decoder block: upsampled feature map (the
            # bottleneck for the first level, the previous decoder block
            # otherwise) + the same-level skip connection.
            decoder_in_ch = prev_out_ch + channels[i]

            # Decoder width at level i mirrors half the encoder width at that
            # level — reproduces the original b*2 / b / b//2 scheme at depth 3.
            out_ch = max(1, channels[i] // 2)

            decoder_blocks_by_level[i] = FiLMConvBlock(decoder_in_ch, out_ch, **kw)
            prev_out_ch = out_ch

        self.decoder_blocks = nn.ModuleList(decoder_blocks_by_level)
        self.up_layers = nn.ModuleList(up_layers_by_level)

        # --- Output head (plain 1×1 conv, no FiLM) ───────────────────────────
        self.head = nn.Conv2d(max(1, channels[0] // 2), out_channels, kernel_size=1)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        if any(k.startswith("enc1.") for k in state_dict):
            state_dict = _remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward(
        self,
        x:  torch.Tensor,
        props:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x      : (B, in_channels, H, W)  input tensor
            props  : (B, cond_dim)        material/object property vector z
                                          e.g. [friction, obj_size, pusher_width, density]
        Returns:
            pred   : (B, 1, H, W)        predicted next state ŝ_{t+1}
        """

        # Encoder
        s = [] # List to hold all encoder outputs for skip connections
        for i in range(self.depth):
            block = self.encoder_blocks[i]
            if i == 0:
                output = block(x, props)
            else:
                output = block(self.pool(s[-1]), props)
            s.append(output)

        # Bottleneck
        b_feat = self.bottleneck(self.pool(s[-1]), props)

        # Decoder (Iterating backwards from depth-1 down to 0)
        d_features = b_feat # Start with bottleneck features

        for i in range(self.depth - 1, -1, -1):
            up_layer = self.up_layers[i]
            dec_block = self.decoder_blocks[i]
            skip_s = s[i]

            # Upsample the previous decoder output (d_features)
            up_feat = up_layer(d_features)

            # Concatenate with skip connection
            concat_input = torch.cat([up_feat, skip_s], dim=1)

            # Pass through the decoder block
            d_features = dec_block(concat_input, props)


        current_state = x[:, self.residual_channel:self.residual_channel + 1, :, :]
        return self.head(d_features) + current_state                       # (B, 1, H, W)


class NFDUNetFiLMShallow(nn.Module):
    """
    Less-deep FiLM U-Net variant.

    Keeps the same input/output resolution and residual current-state skip, but
    removes the deepest encoder/decoder stage. This gives two pooling levels
    instead of three.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        cond_dim: int = 3,
        base: int = 8,
        film_hidden: int = 64,
        residual_channel: int = 0,
    ):
        super().__init__()
        b = base
        self.residual_channel = residual_channel
        kw = dict(cond_dim=cond_dim, hidden=film_hidden)

        self.enc1 = FiLMConvBlock(in_channels, b, **kw)
        self.enc2 = FiLMConvBlock(b, b * 2, **kw)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = FiLMConvBlock(b * 2, b * 2, **kw)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = FiLMConvBlock(b * 2 + b * 2, b, **kw)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = FiLMConvBlock(b + b, b // 2, **kw)

        self.head = nn.Conv2d(b // 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x, props)
        s2 = self.enc2(self.pool(s1), props)

        b_feat = self.bottleneck(self.pool(s2), props)

        d2 = self.dec2(torch.cat([self.up2(b_feat), s2], dim=1), props)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1), props)

        current_state = x[:, self.residual_channel:self.residual_channel + 1, :, :]
        return self.head(d1) + current_state