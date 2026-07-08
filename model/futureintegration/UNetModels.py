import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== #
#   UNET ARCHITECTURE   #
# ===================== #
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNetSmall(nn.Module):
    """
    Shallow U-Net as described in the paper (Sec. B).
    Input: concatenated state and action fields [s_t, a_t] -> shape (B, 3, H, W)
    Output: predicted next state -> shape (B, 1, H, W)
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64]):
        super().__init__()
        self.down1 = ConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(features[0], features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.upconv1 = ConvBlock(features[1], features[0])
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))

        u1 = self.up1(d2)
        concat1 = torch.cat([u1, d1], dim=1)
        u1 = self.upconv1(concat1)

        return self.final(u1)

class UNetMedium(nn.Module):
    """
    Medium U-Net with an additional down and up layer.
    Input: concatenated state and action fields [s_t, a_t] -> shape (B, 3, H, W)
    Output: predicted next state -> shape (B, 1, H, W)
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128]):
        super().__init__()
        self.down1 = ConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(features[1], features[2])

        self.up1 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.upconv1 = ConvBlock(features[2], features[1])
        self.up2 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.upconv2 = ConvBlock(features[1], features[0])
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)                       # (B, features[0], H, W)
        d2 = self.down2(self.pool1(d1))          # (B, features[1], H/2, W/2)
        d3 = self.down3(self.pool2(d2))          # (B, features[2], H/4, W/4)

        u1 = self.up1(d3)                        # (B, features[1], H/2, W/2)
        concat1 = torch.cat([u1, d2], dim=1)     # (B, features[1]*2, H/2, W/2)
        u1 = self.upconv1(concat1)               # (B, features[1], H/2, W/2)

        u2 = self.up2(u1)                        # (B, features[0], H, W)
        concat2 = torch.cat([u2, d1], dim=1)     # (B, features[0]*2, H, W)
        u2 = self.upconv2(concat2)               # (B, features[0], H, W)
        raw_out = self.final(u2)                 # (B, out_channels, H, W)
        zero_bias_out = raw_out - raw_out.mean(dim=[2,3], keepdim=True)  # zero-mean output
        return zero_bias_out + x[:,0:1,:,:]  # add back current state for residual prediction

class UNetLarge(nn.Module):
    """
    Medium U-Net with an additional down and up layer.
    Input: concatenated state and action fields [s_t, a_t] -> shape (B, 3, H, W)
    Output: predicted next state -> shape (B, 1, H, W)
    """
    def __init__(self, in_channels=3, out_channels=1, features=[8, 16, 32, 64]):
        super().__init__()
        # Encoder
        self.enc1 = self.double_conv(in_channels, features[0])   # 3 -> 4
        self.pool1 = nn.MaxPool2d(2)                   # 128 -> 64

        self.enc2 = self.double_conv(features[0], features[1])             # 4 -> 8
        self.pool2 = nn.MaxPool2d(2)                   # 64 -> 32

        self.enc3 = self.double_conv(features[1], features[2])            # 8 -> 16
        self.pool3 = nn.MaxPool2d(2)                   # 32 -> 16

        self.enc4 = self.double_conv(features[2], features[3])           # 16 -> 32
        self.pool4 = nn.MaxPool2d(2)                   # 16 -> 8

        # Bottleneck
        self.bottleneck = self.double_conv(features[3], features[3])

        # Decoder
        self.up4 = nn.ConvTranspose2d(features[3], features[3], kernel_size=2, stride=2)   # 8 -> 16
        self.dec4 = self.double_conv(features[3] + features[3], features[2])

        self.up3 = nn.ConvTranspose2d(features[2], features[2], kernel_size=2, stride=2)   # 16 -> 32
        self.dec3 = self.double_conv(features[2] + features[2], features[1])

        self.up2 = nn.ConvTranspose2d(features[1], features[1], kernel_size=2, stride=2)     # 32 -> 64
        self.dec2 = self.double_conv(features[1] + features[1], features[0])

        self.up1 = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2)     # 64 -> 128
        self.dec1 = self.double_conv(features[0] + features[0], 4)

        # Final output
        self.final = nn.Conv2d(4, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def triple_conv(self, in_channels, out_channels): # actually triple conv, just checking
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        raw = self.final(d1)
        zero_bias_out = raw - raw.mean(dim=[2,3], keepdim=True)  # zero-mean output
        return zero_bias_out + x[:,0:1,:,:]  # add back current state for residual prediction
    
class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.depthwise(x))
        x = self.act(self.pointwise(x))
        return x

class SepDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = DWSeparableConv(in_ch, out_ch)
        self.c2 = DWSeparableConv(out_ch, out_ch)
    def forward(self, x):
        return self.c2(self.c1(x))

class ResSepDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = SepDoubleConv(in_ch, out_ch)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        return self.conv(x) + self.proj(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze: (B,C,H,W) -> (B,C,1,1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)          # (B, C)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)          # (B, C, 1, 1)
        # Scale
        return x * y.expand_as(x)
# --------------------------
# Deeper UNet with ResSepDoubleConv
# --------------------------
class UNetDeepSmall(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[4, 8, 16, 32, 64]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder (Down path)
        for feat in features:
            self.downs.append(ResSepDoubleConv(in_channels, feat))
            in_channels = feat

        # Pooling between downs
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder (Up path)
        for feat in reversed(features[:-1]):
            self.ups.append(
                nn.ConvTranspose2d(in_channels, feat, kernel_size=2, stride=2)
            )
            self.ups.append(ResSepDoubleConv(in_channels, feat))
            in_channels = feat

        # Final output conv
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:  # don’t pool after last encoder block
                skip_connections.append(x)
                x = self.pool(x)

        skip_connections = skip_connections[::-1]  # reverse for decoding

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # upsample
            skip = skip_connections[i // 2]

            # Pad if needed (for odd spatial dims)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)  # concat skip
            x = self.ups[i + 1](x)  # conv block

        return self.final(x)


class UNetOriginal(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetOriginal, self).__init__()

        # Encoder
        self.enc1 = self.double_conv(in_channels, 4)   # 1 -> 4
        self.pool1 = nn.MaxPool2d(2)                   # 128 -> 64

        self.enc2 = self.double_conv(4, 8)             # 4 -> 8
        self.pool2 = nn.MaxPool2d(2)                   # 64 -> 32

        self.enc3 = self.double_conv(8, 16)            # 8 -> 16
        self.pool3 = nn.MaxPool2d(2)                   # 32 -> 16

        self.enc4 = self.double_conv(16, 32)           # 16 -> 32
        self.pool4 = nn.MaxPool2d(2)                   # 16 -> 8

        # Bottleneck
        self.bottleneck1 = self.double_conv(32, 64)
        self.seblock = SEBlock(64,reduction=8)
        self.bottleneck2 = self.double_conv(64, 32)
        # self.bottleneck = self.double_conv(32, 32)
        # self.FCL = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(32*8*8, 32*8*8),
        #     nn.ReLU(inplace=True),
        #     nn.Unflatten(1, (32,8,8))
        # )

        # Decoder
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)   # 8 -> 16
        self.dec4 = self.double_conv(32 + 32, 16)

        self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)   # 16 -> 32
        self.dec3 = self.double_conv(16 + 16, 8)

        self.up2 = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)     # 32 -> 64
        self.dec2 = self.double_conv(8 + 8, 4)

        self.up1 = nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2)     # 64 -> 128
        self.dec1 = self.double_conv(4 + 4, 4)

        
        # Final output
        self.final = nn.Conv2d(4, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels, kernel_size=5):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
    def triple_conv(self, in_channels, out_channels, kernel_size=3): # actually triple conv, just checking
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        # b1 = self.bottleneck1(p4)
        b = self.bottleneck1(p4)
        b = self.seblock(b)
        b = self.bottleneck2(b)

        # Decoder
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        # raw = self.final(d1)
        # zero_bias_out = raw - raw.mean(dim=[2,3], keepdim=True)  # zero-mean output
        # return zero_bias_out + x[:,0:1,:,:]  # add back current state for residual prediction

        out = self.final(d1)
        return out

# ------------------------
# Hybrid Residual Block
# ------------------------
class ResHybridConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        if in_channels <= 32:
            # Use standard conv (efficient at low channel counts)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            # Use depthwise separable conv (saves FLOPs at high channel counts)
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Residual projection if needed
        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)

# ------------------------
# Down & Up blocks
# ------------------------
class DownRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResHybridConv(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))

class UpRes(nn.Module):
    def __init__(self, in_channels, out_channels, mode="nearest"):
        super().__init__()
        self.mode = mode
        self.conv = ResHybridConv(in_channels, out_channels)

    def forward(self, x, skip):
        # Upsample
        if self.mode in ["nearest", "bilinear"]:
            x = F.interpolate(x, size=skip.shape[2:], mode=self.mode, align_corners=False if self.mode=="bilinear" else None)
        elif self.mode == "convtranspose":
            x = nn.ConvTranspose2d(x.size(1), x.size(1), kernel_size=2, stride=2)(x)
        else:
            raise ValueError(f"Unknown upsampling mode: {self.mode}")
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ------------------------
# UNet deep model
# ------------------------
class UNetDeep(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[4, 8, 16, 32, 64], up_mode="nearest"):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        self.downs.append(ResHybridConv(in_channels, features[0]))
        for i in range(1, len(features)):
            self.downs.append(DownRes(features[i-1], features[i]))

        # Decoder
        for i in range(len(features)-1, 0, -1):
            self.ups.append(UpRes(features[i] + features[i-1], features[i-1], mode=up_mode))

        # Final conv
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                skips.append(x)

        for i, up in enumerate(self.ups):
            skip = skips[-(i+1)]
            x = up(x, skip)

        return self.final(x)
# ------------------------
# UNet diffusion model
# ------------------------

# -----------------------------
# Basic UNet Building Blocks
# -----------------------------
class DoubleConv(nn.Module):
    """(Conv => ReLU => Conv => ReLU)"""
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed to handle odd sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# -----------------------------
# UNet with Refinement Head
# -----------------------------
class UNetDelta(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, features=[64, 128, 256, 512]):
        """
        Args:
            in_ch: number of input channels (state + action, usually 2 or 3)
            out_ch: number of output channels (1 for density field)
            features: channels at each scale
        """
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Encoder
        self.inc = DoubleConv(in_ch, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck
        self.bot = DoubleConv(features[3], features[3])

        # Decoder
        self.up1 = Up(features[3] + features[2], features[2])
        self.up2 = Up(features[2] + features[1], features[1])
        self.up3 = Up(features[1] + features[0], features[0])

        # Raw delta output
        self.outc = nn.Conv2d(features[0], out_ch, kernel_size=1)

        # Refinement head: lightweight residual block
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        state = x[:, 0:1, :, :]  # assume first channel is current state

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x5 = self.bot(x4)

        # Decoder
        x = self.up1(x5, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        delta = self.outc(x)
        delta_refined = delta + self.refine(delta)
        delta_refined = delta_refined - delta_refined.mean(dim=[2,3], keepdim=True)  # zero-mean delta
        # Residual prediction: state + delta
        next_state = state + delta_refined
        return next_state


# Available activations
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(0.2),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
}


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
        self.acts = nn.ModuleList([ACTIVATIONS[a] for a in act_list])

    def forward(self, x):
        out = self.bn(self.conv(x))  # (B, out_ch, H, W)
        chunks = torch.chunk(out, self.num_groups, dim=1)
        activated = [act(c) for act, c in zip(self.acts, chunks)]
        return torch.cat(activated, dim=1)


class DoubleConv(nn.Module):
    """ Standard double conv block with one activation """
    def __init__(self, in_ch, out_ch, act="relu", kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            ACTIVATIONS[act],
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            ACTIVATIONS[act],
        )

    def forward(self, x):
        return self.block(x)


class UNetMixed(nn.Module):
    def __init__(self, in_ch=3, out_ch=1,
                 features=[4, 8, 16, 32, 64],
                 mixed_blocks=None,
                 act_list=["relu", "gelu"]):
        """
        Args:
            in_ch: input channels
            out_ch: output channels
            features: channel sizes for each stage
            mixed_blocks: indices of blocks that should use MixedActivationConv
                          e.g. [2, 3] means the 3rd and 4th encoder blocks use mixed activations
            act_list: activations to mix inside mixed blocks
        """
        super().__init__()
        self.mixed_blocks = mixed_blocks or []

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_ch
        for idx, f in enumerate(features):
            if idx in self.mixed_blocks:
                self.enc_blocks.append(nn.Sequential(MixedActivationConv(prev_ch, f, act_list),
                                                     MixedActivationConv(f, f, act_list)))
            else:
                self.enc_blocks.append(DoubleConv(prev_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(prev_ch, prev_ch * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        rev_feats = features[::-1]
        prev_ch = prev_ch * 2
        for f in rev_feats:
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2))
            if f in features and features.index(f) in self.mixed_blocks:
                self.dec_blocks.append(nn.Sequential(MixedActivationConv(prev_ch, f, act_list),
                                                     MixedActivationConv(f, f, act_list)))
            else:
                self.dec_blocks.append(DoubleConv(prev_ch, f))
            prev_ch = f

        self.final_conv = nn.Conv2d(prev_ch, out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for up, dec, skip in zip(self.upconvs, self.dec_blocks, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final_conv(x)
    


class DoubleConvDown(nn.Module):
    """Downsampling block: conv(stride=1) -> conv(stride=2)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=pad),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=2, padding=pad),
            activation()
        )

    def forward(self, x):
        return self.block(x)

class DoubleConvUp(nn.Module):
    """Upsampling block: upconv(stride=2) -> conv(stride=1) -> conv(stride=1)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU):
        super().__init__()
        pad = kernel_size // 2
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size, stride=1, padding=pad)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=pad)
        self.act2 = activation()

    def forward(self, x, skip):
        x = self.up(x)
        x = F.pad(x, self._match_padding(x, skip))  # handle odd sizes
        x = torch.cat([skip, x], dim=1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x

    def _match_padding(self, x, ref):
        """Pads tensor x so that its H,W match ref (due to odd sizes)."""
        diffY = ref.size(2) - x.size(2)
        diffX = ref.size(3) - x.size(3)
        return (diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2)

class UNetStrided(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, kernel_size=3, activation=nn.ReLU):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch=in_channels, out_ch=base_channels, act="relu", kernel_size=kernel_size)
        self.enc2 = DoubleConvDown(base_channels, base_channels*2, kernel_size, activation)
        self.enc3 = DoubleConvDown(base_channels*2, base_channels*4, kernel_size, activation)
        self.enc4 = DoubleConvDown(base_channels*4, base_channels*8, kernel_size, activation)

        # Bottleneck
        pad = kernel_size // 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*16, kernel_size, stride=1, padding=pad),
            activation(),
            nn.Conv2d(base_channels*16, base_channels*8, kernel_size, stride=1, padding=pad),
            activation()
        )

        # Decoder
        self.up3 = DoubleConvUp(base_channels*8, base_channels*4, kernel_size, activation)
        self.up2 = DoubleConvUp(base_channels*4, base_channels*2, kernel_size, activation)
        self.up1 = DoubleConvUp(base_channels*2, base_channels, kernel_size, activation)


        # Final conv
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)   # 1/1 res
        x2 = self.enc2(x1)  # 1/2 res
        x3 = self.enc3(x2)  # 1/4 res
        x4 = self.enc4(x3)  # 1/8 res

        # Bottleneck
        xb = self.bottleneck(x4)

        # Decoder
        d3 = self.up3(xb, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)

        # Final
        raw = self.final(d1)
        zero_bias_out = raw - raw.mean(dim=[2,3], keepdim=True)
        out = zero_bias_out + x[:,0:1,:,:]
        return out
