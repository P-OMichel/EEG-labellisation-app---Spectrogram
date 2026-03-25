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
        self.net = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            DoubleConv1D(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        """
        in_ch is the number of channels of the decoder input.
        After upsampling, channels become in_ch // 2.
        Then we concatenate with the skip connection, so the conv sees in_ch channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_ch, out_ch)

    def forward(self, x1, x2):
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
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)
        self.relu_out = nn.ReLU(inplace=True)

        # project residual if channel count changes
        if in_ch != out_ch:
            self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        out = out + residual
        out = self.relu_out(out)
        return out


class TCNPostUNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        dilations=(1, 2, 4, 8),
        dropout: float = 0.0,
    ):
        super().__init__()

        blocks = []
        ch_in = in_ch
        for d in dilations:
            blocks.append(
                ResidualTCNBlock(
                    in_ch=ch_in,
                    out_ch=hidden_ch,
                    dilation=d,
                    dropout=dropout,
                )
            )
            ch_in = hidden_ch

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ------------------------ Full Architecture ---------------------

@register_model("unet1d_then_tcn")
class UNet1DThenTCN(nn.Module):
    """
    1D U-Net followed by a TCN refinement head.

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
        tcn_hidden_ch: int | None = None,
        tcn_dilations=(1, 2, 4, 8),
        tcn_dropout: float = 0.0,
    ):
        super().__init__()
        self.target_len = target_len

        if tcn_hidden_ch is None:
            tcn_hidden_ch = base_ch

        # U-Net
        self.inc = DoubleConv1D(1, base_ch)
        self.down1 = Down1D(base_ch, base_ch * 2)
        self.down2 = Down1D(base_ch * 2, base_ch * 4)
        self.down3 = Down1D(base_ch * 4, base_ch * 8)

        self.up1 = Up1D(base_ch * 8, base_ch * 4)
        self.up2 = Up1D(base_ch * 4, base_ch * 2)
        self.up3 = Up1D(base_ch * 2, base_ch)

        # U-Net output feature projection
        self.outc = nn.Conv1d(base_ch, base_ch, kernel_size=1)

        # TCN after U-Net
        self.tcn = TCNPostUNet(
            in_ch=base_ch,
            hidden_ch=tcn_hidden_ch,
            dilations=tcn_dilations,
            dropout=tcn_dropout,
        )

        # classifier
        self.head = TemporalHead1D(tcn_hidden_ch, num_classes)

    def forward(self, x):
        # x: (B, 1, T)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.outc(x)      # (B, base_ch, ~T)
        x = self.tcn(x)       # (B, tcn_hidden_ch, ~T)
        x = self.head(x)      # (B, num_classes, ~T)

        if x.size(-1) != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode="linear", align_corners=False)

        return x