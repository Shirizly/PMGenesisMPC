import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# -------------------------
# Activation factory
# -------------------------
_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(),
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(),
    "mish": lambda: nn.Mish(),
    "elu": lambda: nn.ELU(),
    "identity": lambda: nn.Identity(),
}

_ACTIVATIONS_mixed = {
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(0.2),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
}

# -------------------------
# Small helper blocks
# -------------------------
class DoubleConv(nn.Module):
    """Two conv layers (stride=1), each followed by BN + activation."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, activation: str):
        super().__init__()
        pad = kernel_size // 2
        act = _ACTIVATIONS.get(activation, _ACTIVATIONS["relu"])
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            act(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            act(),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    """Double conv (stride=1) then MaxPool2d(2). Returns (skip, downsampled)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, activation: str):
        super().__init__()
        self.DConv = DoubleConv(in_ch, out_ch, kernel_size, activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.DConv(x)
        down = self.pool(skip)
        return skip, down


class UpBlock(nn.Module):
    """Upsample via ConvTranspose2d (stride=2) then DoubleConv on concat(skip, up)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, activation: str):
        """
        in_ch: channels of incoming (bottleneck or previous decoder)
        out_ch: channels to produce (matches encoder skip channels)
        """
        super().__init__()
        # ConvTranspose reduces channels from in_ch -> out_ch (so that concat makes 2*out_ch)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.DConv = DoubleConv(in_ch=out_ch * 2, out_ch=out_ch, kernel_size=kernel_size, activation=activation)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary (odd sizes)
        if x.shape[2:] != skip.shape[2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.DConv(x)
        return x

class MixedActivationConv(nn.Module):
    def __init__(self, in_ch, out_ch, act_list, kernel_size=3):
        """
        Convolution + BN + split activations.
        Args:
            in_ch: input channels
            out_ch: output channels (must be divisible by len(act_list))
            act_list: list of activation names to mix
        """
        super().__init__()
        self.num_groups = len(act_list)
        assert out_ch % self.num_groups == 0, "out_ch must divide evenly by len(act_list)"
        # self.group_ch = out_ch // self.num_groups
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.acts = nn.ModuleList([_ACTIVATIONS_mixed[a] for a in act_list])

    def forward(self, x):
        out = self.bn(self.conv(x))  # (B, out_ch, H, W)
        chunks = torch.chunk(out, self.num_groups, dim=1)
        activated = [act(c) for act, c in zip(self.acts, chunks)]
        return torch.cat(activated, dim=1)
    
class MixedActivationConv2(nn.Module): #this one applies all activations to the same features and concatenates
    def __init__(self, in_ch, out_ch, act_list, kernel_size=3):
        """
        Convolution + BN + mixed activations applied to the *same* features.
        Args:
            in_ch: input channels
            out_ch: output channels (same as normal conv)
            act_list: list of activation names to apply in parallel
        """
        super().__init__()
        self.num_groups = len(act_list)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch//self.num_groups, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch//self.num_groups)
        self.acts = nn.ModuleList([_ACTIVATIONS_mixed[a] for a in act_list])

    def forward(self, x):
        out = self.bn(self.conv(x))  # (B, out_ch, H, W)
        # apply each activation to the same feature map
        activated = [act(out) for act in self.acts]
        return torch.cat(activated, dim=1)  # (B, out_ch * num_groups, H, W)
    
class DownMixedBlock(nn.Module):
    """Mixed activation Double conv (stride=1) then MaxPool2d(2). Returns (skip, downsampled)."""
    def __init__(self, in_ch: int, out_ch: int, act_list: List[str], kernel_size: int = 3, mixed_type: str = "split"):
        super().__init__()
        if mixed_type == "split":
            self.DConv = nn.Sequential(MixedActivationConv(in_ch, out_ch, act_list, kernel_size),
                                   MixedActivationConv(out_ch, out_ch, act_list, kernel_size))
        elif mixed_type == "parallel": #applies all activations to the same features and concatenates
            self.DConv = nn.Sequential(MixedActivationConv2(in_ch, out_ch, act_list, kernel_size),
                                   MixedActivationConv2(out_ch, out_ch, act_list, kernel_size))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.DConv(x)
        down = self.pool(skip)
        return skip, down

class UpMixedBlock(nn.Module):
    """Upsample via ConvTranspose2d (stride=2) then MixedActivationConv on concat(skip, up)."""
    def __init__(self, in_ch: int, out_ch: int, act_list: List[str], kernel_size: int = 3, mixed_type: str = "split"):
        """
        in_ch: channels of incoming (bottleneck or previous decoder)
        out_ch: channels to produce (matches encoder skip channels)
        """
        super().__init__()
        # ConvTranspose reduces channels from in_ch -> out_ch (so that concat makes 2*out_ch)
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        if mixed_type == "split":
            self.DConv = nn.Sequential(MixedActivationConv(in_ch=out_ch * 2, out_ch=out_ch, act_list=act_list, kernel_size=kernel_size),
                                       MixedActivationConv(in_ch=out_ch, out_ch=out_ch, act_list=act_list, kernel_size=kernel_size))
        elif mixed_type == "parallel": #applies all activations to the same features and concatenates
            self.DConv = nn.Sequential(MixedActivationConv2(in_ch=out_ch * 2, out_ch=out_ch, act_list=act_list, kernel_size=kernel_size),
                                   MixedActivationConv2(in_ch=out_ch, out_ch=out_ch, act_list=act_list, kernel_size=kernel_size))

    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary (odd sizes)
        if x.shape[2:] != skip.shape[2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.DConv(x)
        return x
# -------------------------
# Bottleneck variants
# -------------------------
class BottleneckFCN(nn.Module):
    """Plain conv-based bottleneck (default)."""
    def __init__(self, in_ch: int, kernel_size: int, activation: str):
        super().__init__()
        self.conv = DoubleConv(in_ch, in_ch, kernel_size, activation)

    def forward(self, x):
        return self.conv(x)


class BottleneckSE(nn.Module):
    """Squeeze-and-Excitation applied to the bottleneck features."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # channel-wise scaling


class BottleneckFCChannel(nn.Module):
    """Channel-wise fully-connected MLP via global pooling (lightweight FC across channels)."""
    def __init__(self, channels: int, hidden_ratio: float = 0.25, activation: str = "relu"):
        super().__init__()
        self.channels = channels
        h = max(1, int(channels * hidden_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        act = _ACTIVATIONS.get(activation, _ACTIVATIONS["relu"])
        self.mlp = nn.Sequential(
            nn.Linear(channels, h, bias=True),
            act(),
            nn.Linear(h, channels, bias=True),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.pool(x).view(b, c)          # (B, C)
        v = self.mlp(v).view(b, c, 1, 1)     # (B, C, 1, 1)
        return x + v  # residual channel-wise correction


class BottleneckFCFlat(nn.Module):
    """
    Full flatten FC bottleneck. WARNING: heavy. Requires image_size in structure params.
    This flattens (C*H*W) -> FC(hidden) -> FC(C*H*W) and reshapes back.
    """
    def __init__(self, channels: int, image_size: int, hidden_dim: Optional[int] = None, activation: str = "relu"):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        in_dim = channels * image_size * image_size
        if hidden_dim is None:
            hidden_dim = max(in_dim // 4, 512)
        act = _ACTIVATIONS.get(activation, _ACTIVATIONS["relu"])
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.image_size and w == self.image_size, "BottleneckFCFlat requires fixed image_size"
        v = x.view(b, -1)
        v = self.net(v)
        v = v.view(b, c, h, w)
        return v

class BottleneckTransformer(nn.Module):
    """
    Transformer bottleneck: applies global self-attention over spatial positions.
    Operates at the bottleneck resolution (low H, W).
    """
    def __init__(self, channels: int, num_heads: int = 4, num_layers: int = 1, dim_feedforward: int = 256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        b, c, h, w = x.shape
        # flatten spatial dims
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
        x_enc = self.transformer(x_flat)        # (B, N, C)
        x_out = x_enc.permute(0, 2, 1).view(b, c, h, w)
        return x_out

# -------------------------
# Unified UNet class
# -------------------------
class UNet(nn.Module):
    """
    Modular UNet controlled by structure_parameters dictionary.
    Required keys in structure_parameters:
      - features: list of ints (channels for encoder blocks)
      - in_channels, out_channels (ints)
    Optional keys:
      - kernel_size (int, default=3)
      - final_kernel_size (int, default=kernel_size)
      - activation (str, default='relu'), must be in _ACTIVATIONS
      - residual (bool, default=False) -> if True, return input_first_channel + model_delta
      - bottleneck_type (str): one of {'FCN', 'SE', 'FC_channel', 'FC_flat'} (default 'FCN')
      - bottleneck_kwargs (dict): extra args for bottleneck classes (e.g., hidden_ratio, image_size)
    """
    def __init__(self, structure_parameters: Dict):
        super().__init__()
        # --- parse params ---
        features: List[int] = structure_parameters["features"]
        in_channels: int = structure_parameters.get("in_channels", 1)
        out_channels: int = structure_parameters.get("out_channels", 1)
        kernel_size: int = structure_parameters.get("kernel_size", 3)
        final_kernel_size: int = structure_parameters.get("final_kernel_size", kernel_size)
        activation: str = structure_parameters.get("activation", "relu")
        self.residual_mode: bool = bool(structure_parameters.get("residual", False))
        bottleneck_type: str = structure_parameters.get("bottleneck_type", "None")
        bottleneck_kwargs: Dict = structure_parameters.get("bottleneck_kwargs", {})
        self.mixed_blocks: List[int] = structure_parameters.get("mixed_blocks", [])
        act_list: List[str] = structure_parameters.get("activation_list", ['relu','silu','gelu','mish'])
        mixed_type: str = structure_parameters.get("mixed_type", "split") #or "parallel"
        self.structure_parameters = structure_parameters.copy()

        # --- encoder ---
        self.down_blocks = nn.ModuleList()
        prev_ch = in_channels
        for idx,f in enumerate(features):
            if idx in self.mixed_blocks:
                self.down_blocks.append(DownMixedBlock(prev_ch, f, act_list, kernel_size,mixed_type))
            else:
                self.down_blocks.append(DownBlock(prev_ch, f, kernel_size, activation))
            prev_ch = f

        # --- bottleneck ---
        # typical UNet doubles channels in bottleneck
        bottleneck_in = prev_ch
        bottleneck_ch = bottleneck_in * 2
        # one conv to increase channels, then apply chosen bottleneck block which returns same channels
        self.bottleneck_pre = DoubleConv(bottleneck_in, bottleneck_ch, kernel_size, activation)

        if bottleneck_type == "FCN":
            self.bottleneck = BottleneckFCN(bottleneck_ch, kernel_size, activation)
        elif bottleneck_type == "SE":
            self.bottleneck = BottleneckSE(bottleneck_ch, reduction=bottleneck_kwargs.get("reduction", 16))
        elif bottleneck_type == "FC_channel":
            self.bottleneck = BottleneckFCChannel(bottleneck_ch, hidden_ratio=bottleneck_kwargs.get("hidden_ratio", 0.25), activation=activation)
        elif bottleneck_type == "FC_flat":
            # requires image_size
            image_size = bottleneck_kwargs.get("image_size", None)
            if image_size is None:
                raise ValueError("bottleneck_type='FC_flat' requires bottleneck_kwargs['image_size']")
            self.bottleneck = BottleneckFCFlat(bottleneck_ch, image_size, hidden_dim=bottleneck_kwargs.get("hidden_dim", None), activation=activation)
        elif bottleneck_type == "Transformer":
            self.bottleneck = BottleneckTransformer(
                bottleneck_ch,
                num_heads=bottleneck_kwargs.get("num_heads", 4),
                num_layers=bottleneck_kwargs.get("num_layers", 1),
                dim_feedforward=bottleneck_kwargs.get("dim_feedforward", 256),
            )
        elif bottleneck_type == "None":
            self.bottleneck = nn.Identity()
        else:
            raise ValueError(f"Unknown bottleneck_type {bottleneck_type}")

        # --- decoder ---
        self.up_blocks = nn.ModuleList()
        rev_features = list(reversed(features))
        prev_ch = bottleneck_ch
        for f in rev_features:
            if f in features and features.index(f) in self.mixed_blocks:
                self.up_blocks.append(UpMixedBlock(prev_ch, f, act_list, kernel_size,mixed_type))
            else:
                self.up_blocks.append(UpBlock(prev_ch, f, kernel_size, activation))
            
            prev_ch = f

        # --- final conv ---
        pad_final = final_kernel_size // 2
        self.final_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=final_kernel_size, padding=pad_final)

    def forward(self, x):
        """
        x: (B, in_channels, H, W)
        returns: (B, out_channels, H, W)
        If residual_mode True, model predicts delta and we return x[:,0:1,:,:] + delta
        """
        # encode
        skips = []
        cur = x
        for down in self.down_blocks:
            skip, cur = down(cur)
            skips.append(skip)

        # bottleneck
        cur = self.bottleneck_pre(cur)
        cur = self.bottleneck(cur)

        # decode
        for up, skip in zip(self.up_blocks, reversed(skips)):
            cur = up(cur, skip)

        raw = self.final_conv(cur)  # (B, out_ch, H, W)

        if self.residual_mode:
            # add residual to input first channel (broadcast if necessary)
            base = x[:, 0:1, :, :].to(raw.dtype)
            # If out_channels != 1, attempt to broadcast: if out_channels==in_channels use full input; else add base to channel 0
            if raw.shape[1] == 1:
                return base + raw
            else:
                # prepend base as channel 0 and leave other raw channels as-is
                out = raw.clone()
                out[:, 0:1, :, :] = out[:, 0:1, :, :] + base
                return out
        else:
            return raw

    # convenience: save / load structure + weights
    def save_checkpoint(self, path: str):
        data = {
            "model_state": self.state_dict(),
            "structure_parameters": self.structure_parameters
        }
        torch.save(data, path)

    @staticmethod
    def load_checkpoint(path: str, map_location=None) -> "UNet":
        ckpt = torch.load(path, map_location=map_location,weights_only=False)
        params = ckpt["structure_parameters"]
        model = UNet(params)
        model.load_state_dict(ckpt["model_state"])
        return model