from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model


# NOTE: here each domwsampling is in both time and freq check with only freq

#------------------------ Block elements ---------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class DownAnysotropic(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UpAnysotropic(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=(2, 1), stride=(2, 1))
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size(2) - x1.size(2)   # frequency axis
        diffX = x2.size(3) - x1.size(3)   # time axis
        x1 = F.pad(x1,[diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TemporalHead1D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.out = nn.Conv1d(in_ch, num_classes, 1)
    def forward(self, x): return self.out(x)

#------------------------ Full Architectures ---------------------
@register_model("unet_temporal")
class UNetTemporal(nn.Module):
    """U-Net over (F,T) then pool over F and T and classify over time."""
    def __init__(self, num_classes: int = 10, base_ch: int = 32):
        super().__init__()
        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.up1 = Up(base_ch*8, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up3 = Up(base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, base_ch, 1)
        self.head = TemporalHead1D(base_ch, num_classes)

    def forward(self, x):
        # x: (B,1,F,T)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)     # (B,base_ch,F,T)
        x = x.mean(dim=2)    # pool F -> (B,base_ch,T)
        return self.head(x)

@register_model("unet_anysotropic_temporal")
class UNetAnysotropicTemporal(nn.Module):
    """U-Net over (F,T) then pool over F and classify over time."""
    def __init__(self, num_classes: int = 10, base_ch: int = 32):
        super().__init__()
        self.inc = DoubleConv(1, base_ch)
        self.down1 = DownAnysotropic(base_ch, base_ch*2)
        self.down2 = DownAnysotropic(base_ch*2, base_ch*4)
        self.down3 = DownAnysotropic(base_ch*4, base_ch*8)
        self.up1 = UpAnysotropic(base_ch*8, base_ch*4)
        self.up2 = UpAnysotropic(base_ch*4, base_ch*2)
        self.up3 = UpAnysotropic(base_ch*2, base_ch)
        self.outc = nn.Conv2d(base_ch, base_ch, 1)
        self.head = TemporalHead1D(base_ch, num_classes)

    def forward(self, x):
        # x: (B,1,F,T)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)     # (B,base_ch,F,T)
        x = x.mean(dim=2)    # pool F -> (B,base_ch,T)
        return self.head(x)
