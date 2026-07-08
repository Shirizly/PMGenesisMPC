import torch.nn as nn
import torch
from .UNetModels import ConvBlock as DoubleConvBlock

############################################
# A SIMPLTE PHYSICS CONDITIONED UNET MODEL #
############################################

class UNetConditioned(nn.Module):
    
    def __init__(
        self,
        in_channels=2, # state(i) + action(i) layers
        physics_dim=3, # scene physics
        out_channels=1 # state(i+1) layer
    ):
        super().__init__()
        
        total_input_channels = in_channels + physics_dim
        
        # Encoder
        self.enc1 = DoubleConvBlock(in_channels=total_input_channels, out_channels=64)
        self.enc2 = DoubleConvBlock(in_channels=64, out_channels=128)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.bottle = DoubleConvBlock(in_channels=128, out_channels=256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        
    def forward(self, x, physics):
        B, _, H, W = x.size()
        physics = physics[:, :, None, None]
        physics = physics.expand(B, physics.size(1), H, W)
        
        x = torch.cat([x, physics], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1) # down
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2) # down
        
        # Bottleneck
        b = self.bottle(p2) # middle
        
        # Decoder
        u1 = self.up1(b) # up
        c1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(c1)
        
        u2 = self.up2(d1)
        c2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(c2)
        
        return self.out(d2)
        
###############################
# FiLM CONDITIONAL UNET MODEL #
###############################

class FiLMGenerator(nn.Module):
    def __init__(
            self,
            physics_dim,
            hidden=128,
            film_dims=(64, 128, 256
    )):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(physics_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        self.to_film = nn.ModuleDict({
            "enc1": nn.Linear(hidden, film_dims[0] * 2),
            "enc2": nn.Linear(hidden, film_dims[1] * 2),
            "bottle": nn.Linear(hidden, film_dims[2] * 2),
            "dec1": nn.Linear(hidden, film_dims[1] * 2),
            "dec2": nn.Linear(hidden, film_dims[0] * 2),
        })
        
    def forward(self, physics):
        h = self.mlp(physics)
        
        film = {}
        
        for key, layer in self.to_film.items():
            gb = layer(h)
            gamma, beta = gb.chunk(2, dim=-1)
            film[key] = (gamma, beta)
        return film
    
class UNetFiLM(nn.Module):
    def __init__(
        self,
        in_channels=2,
        physics_dim=3,
        out_channels=1,
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.enc2 = ConvBlock(in_channels=64, out_channels=128)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.bottle = ConvBlock(in_channels=128, out_channels=256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        
        self.film = FiLMGenerator(physics_dim=physics_dim, film_dims=(64, 128, 256))
    
    def apply_film(self, x, gamma, beta):
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * gamma + beta
    
    def forward(self, x, physics):
        
        film_params = self.film(physics)
        
        # Encoder
        e1 = self.apply_film(self.enc1(x), *film_params['enc1'])
        p1 = self.pool(e1) # down
        
        e2 = self.apply_film(self.enc2(p1), *film_params['enc2'])
        p2 = self.pool(e2) # down
        
        # Bottleneck
        b = self.bottle(p2) # middle
        b = self.apply_film(b, *film_params['bottle'])
        
        # Decoder
        u1 = self.up1(b) # up
        c1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(c1)
        d1 = self.apply_film(d1, *film_params['dec1'])
        
        u2 = self.up2(d1)
        c2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(c2)
        d2 = self.apply_film(d2, *film_params['dec2'])

        # residual: model predicts delta on top of current particle state (logit space)
        current_state = x[:, 0:1, :, :]
        return self.out(d2) + current_state
    
#####################################
# FiLM Conditioned FNC (from Paper) #
#####################################

class ConvBlock(nn.Module):
    """Single 3×3 conv + ReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))
    

class UNetFiLMNFD(nn.Module):
 
    def __init__(
        self,
        in_channels:  int  = 3,
        out_channels: int  = 1,
        base:         int  = 8,
        physics_dim:  int  = 3,
    ):
        super().__init__()
        b = base
 
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b*2)
        self.enc3 = ConvBlock(b*2, b*4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
 
        
        self.bottleneck = ConvBlock(b*4, b*4)
 
        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(b*4 + b*4, b*2)
 
        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(b*2 + b*2, b)
 
        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(b   + b,   b//2)
 
        # ── Output head ──────────────────────────────────────────────────────
        self.head = nn.Conv2d(b//2, out_channels, kernel_size=1)

        # film layer
        self.film = FiLMGenerator(physics_dim=physics_dim, film_dims=(64, 128, 256))
 
    def apply_film(self, x, gamma, beta):
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * gamma + beta
    
    def forward(self, x: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
        
        film_params = self.film(physics)

        # Encoder + skip connections
        e1 = self.apply_film(self.enc1(x), *film_params['enc1'])
        p1 = self.pool(e1)
        e2 = self.apply_film(self.enc2(p1), *film_params['enc2'])
        p2 = self.pool(e2)
        e3 = self.apply_film(self.enc3(p2), *film_params['enc3'])
        p3 = self.pool(e3)
 
        # Bottleneck
        b = self.bottleneck(p3)    # (B, b*4, H/8, W/8)
        b = self.apply_film(b, *film_params['bottle'])

 
        # Decoder with skip connections
        u3 = self.up3(b)
        c1 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(c1, dim=1)
        d3 = self.apply_film(d3, *film_params['dec3'])

        u2 = self.up2(d3)
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(c2, dim=1)
        d2 = self.apply_film(d2, *film_params['dec2'])

        u1 = self.up1(d2)
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec2(c1, dim=1)
        d1 = self.apply_film(d1, *film_params['dec1'])
 
        return self.head(d1)