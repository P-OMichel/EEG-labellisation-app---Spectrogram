from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model


# ------------------------ Block elements ---------------------

class DoubleConv1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """
        in_ch = channels after concatenation logic:
                upsampled branch has in_ch//2 channels,
                skip branch also has in_ch//2 channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_ch, out_ch)

    def forward(self, x1, x2):
        """
        x1: decoder input
        x2: skip connection
        """
        x1 = self.up(x1)

        # pad if needed to match temporal length
        diffT = x2.size(2) - x1.size(2)
        if diffT != 0:
            x1 = F.pad(x1, [diffT // 2, diffT - diffT // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TemporalHead1D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.out = nn.Conv1d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        return self.out(x)


# ------------------------ TCN block elements ---------------------

class ResidualTCNBlock(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        out = out + residual
        out = self.relu2(out)
        return out


class TCNBottleneck(nn.Module):
    def __init__(self, ch: int, dilations=(1, 2, 4, 8), dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualTCNBlock(ch=ch, dilation=d, dropout=dropout) for d in dilations]
        )

    def forward(self, x):
        return self.blocks(x)


# ------------------------ Full Architecture ---------------------

@register_model("unet1d_tcn")
class UNet1DTCN(nn.Module):
    """
    1D U-Net + TCN bottleneck for sequence segmentation.

    Input:
        x: (B, 1, T)

    Output:
        logits: (B, num_classes, target_len)
    """
    def __init__(
        self,
        num_classes: int = 10,
        base_ch: int = 32,
        target_len: int = 297,
        tcn_dilations=(1, 2, 4, 8),
        tcn_dropout: float = 0.0,
    ):
        super().__init__()
        self.target_len = target_len

        # U-Net encoder
        self.inc = DoubleConv1D(1, base_ch)
        self.down1 = Down1D(base_ch, base_ch * 2)
        self.down2 = Down1D(base_ch * 2, base_ch * 4)
        self.down3 = Down1D(base_ch * 4, base_ch * 8)

        # TCN bottleneck
        self.tcn = TCNBottleneck(
            ch=base_ch * 8,
            dilations=tcn_dilations,
            dropout=tcn_dropout,
        )

        # U-Net decoder
        self.up1 = Up1D(base_ch * 8, base_ch * 4)
        self.up2 = Up1D(base_ch * 4, base_ch * 2)
        self.up3 = Up1D(base_ch * 2, base_ch)

        # projection + classifier
        self.outc = nn.Conv1d(base_ch, base_ch, kernel_size=1)
        self.head = TemporalHead1D(base_ch, num_classes)

    def forward_features(self, x):
        # x: (B,1,T)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.tcn(x4)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.outc(x)  # (B, base_ch, T)

        # ensure correct temporal length
        if x.size(-1) != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)

        return x  
    
    def forward_logits_from_features(self, feats):
        return self.head(feats)
    
    def forward(self, x):
        feats = self.forward_features(x)
        return  self.forward_logits_from_features(feats)