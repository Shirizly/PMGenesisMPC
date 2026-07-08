import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        return self.pool(self.block(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.block = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetMultiExit(nn.Module):
    def __init__(self, base_ch=8, path = 'datasets/weights/multi_exit_unet', residual_mode=True):
        super().__init__()
        self.path = path
        os.makedirs(self.path, exist_ok=True)

        self.inc = ConvBlock(2, base_ch)              # input conv
        self.down1 = Down(base_ch, base_ch*2)         # 1/2 res
        self.down2 = Down(base_ch*2, base_ch*4)       # 1/4 res
        self.down3 = Down(base_ch*4, base_ch*8)       # 1/8 res

        self.up1 = Up(base_ch*8, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up3 = Up(base_ch*2, base_ch)

        # ---------------------
        # Multi-exit heads
        # ---------------------
        self.exit_low  = nn.Conv2d(base_ch*2, 1, 1)   # after down1
        self.exit_mid  = nn.Conv2d(base_ch*4, 1, 1)   # after down2
        self.exit_high = nn.Conv2d(base_ch,    1, 1)  # final output
        self.residual_mode = residual_mode # whether to add output as residual to input

    def exit_prep(self,x,raw):
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

    def forward(self, x):
        # encoder
        x0 = self.inc(x)        # full res
        x1 = self.down1(x0)     # 1/2 res
        x2 = self.down2(x1)     # 1/4 res
        x3 = self.down3(x2)     # 1/8 res bottleneck

        # exits from encoder
        # resize to input size
        low_raw = F.interpolate(self.exit_low(x1), scale_factor=2, mode='bilinear', align_corners=False)
        mid_raw = F.interpolate(self.exit_mid(x2), scale_factor=4, mode='bilinear', align_corners=False)

        low_pred = self.exit_prep(x,low_raw)
        mid_pred = self.exit_prep(x,mid_raw)

        # decoder
        u1 = self.up1(x3, x2)
        u2 = self.up2(u1, x1)
        u3 = self.up3(u2, x0)

        high_raw = self.exit_high(u3)
        high_pred = self.exit_prep(x,high_raw)

        return {
            "low":  low_pred,
            "mid":  mid_pred,
            "high": high_pred
        }
    
    def save_checkpoint(self, epoch):
        torch.save(self.state_dict(), os.path.join(self.path, f'unet_epoch_{epoch}.pth') )
