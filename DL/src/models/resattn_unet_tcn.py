from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model

class ResBlock2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)

class AttentionGate2D(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.Wg = nn.Conv2d(g_ch, inter_ch, 1)
        self.Wx = nn.Conv2d(x_ch, inter_ch, 1)
        self.psi = nn.Conv2d(inter_ch, 1, 1)
    def forward(self, g, x):
        # g: gating, x: skip
        a = F.relu(self.Wg(g) + self.Wx(x))
        a = torch.sigmoid(self.psi(a))
        return x * a

class Down2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResBlock2D(out_ch),
        )
    def forward(self, x): return self.conv(self.pool(x))

class UpAttn2D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.attn = AttentionGate2D(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch//2, 8))
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch+skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResBlock2D(out_ch),
        )
    def forward(self, x, skip):
        x = self.up(x)
        # pad
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        skip2 = self.attn(x, skip)
        x = torch.cat([skip2, x], dim=1)
        return self.conv(x)

class TemporalHeadTCN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, hidden: int , layers: int , k: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks = []
        c = in_ch
        for i in range(layers):
            d = 2 ** i
            blocks.append(nn.Conv1d(c, hidden, kernel_size=k, padding=d*(k-1)//2, dilation=d))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(dropout))
            c = hidden
        self.tcn = nn.Sequential(*blocks)
        self.out = nn.Conv1d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.tcn(x))
    

@register_model("resattn_aniso_unet_temporal_tcn")
class ResAttnAnisoUNetTemporalTCN(nn.Module):
    def __init__(self, num_classes: int = 10, base_ch: int = 32, tcn_hidden: int = 128, tcn_layers: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            ResBlock2D(base_ch),
        )
        self.down1 = Down2D(base_ch, base_ch*2)
        self.down2 = Down2D(base_ch*2, base_ch*4)
        self.down3 = Down2D(base_ch*4, base_ch*8)
        self.up1 = UpAttn2D(base_ch*8, base_ch*4, base_ch*4)
        self.up2 = UpAttn2D(base_ch*4, base_ch*2, base_ch*2)
        self.up3 = UpAttn2D(base_ch*2, base_ch, base_ch)
        self.proj = nn.Conv2d(base_ch, base_ch, 1)
        self.head = TemporalHeadTCN(base_ch, num_classes, tcn_hidden, tcn_layers)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.proj(x)
        x = x.mean(dim=2) # can be replace by mean + max for instance
        return self.head(x)
