from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_model


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

class TemporalHead1D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.out = nn.Conv1d(in_ch, num_classes, 1)
    def forward(self, x): return self.out(x)

class FreqAttentionBlock(nn.Module):
    """
    Analyzes low and high frequency bands separately and produces 
    a temporal attention mask for the bottleneck.
    """
    def __init__(self, bottleneck_ch: int, freq_split_idx: int):
        super().__init__()
        self.split_idx = freq_split_idx
        
        # Branch for Low Freq
        self.low_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, None)), # Flexible sizing
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)) # Collapse Freq to 1, keep Time
        )
        
        # Branch for High Freq
        self.high_branch = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )
        
        # Mix the information to create a temporal attention filter
        self.mixer = nn.Sequential(
            nn.Conv1d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, bottleneck_ch, 1),
            nn.Sigmoid() # Scale between 0 and 1
        )

    def forward(self, x_raw, bottleneck_feat):
        # x_raw: (B, 1, F, T)
        low_f = x_raw[:, :, :self.split_idx, :]
        high_f = x_raw[:, :, self.split_idx:, :]
        
        # Extract 1D temporal features (B, 8, 1, T) -> (B, 8, T)
        low_proj = self.low_branch(low_f).squeeze(2)
        high_proj = self.high_branch(high_f).squeeze(2)
        
        # Combine (B, 16, T)
        combined = torch.cat([low_proj, high_proj], dim=1)
        
        # Generate Attention Mask (B, bottleneck_ch, T)
        attn = self.mixer(combined)
        
        # Match bottleneck dimensions (B, C, F_small, T_small)
        # We need to interpolate the attention mask to match the bottleneck's T dimension
        attn = F.interpolate(attn, size=bottleneck_feat.shape[3], mode='linear', align_corners=False)
        
        # Apply attention (Unsqueeze to add F dimension back for broadcasting)
        return bottleneck_feat * attn.unsqueeze(2)


#------------------------ Full Architectures ---------------------

@register_model("unet_freq_gate_attention")
class UNetFreqAttention(nn.Module):
    def __init__(self, num_classes: int = 10, base_ch: int = 32, split_idx: int = 16):
        super().__init__()
        self.split_idx = split_idx
        
        # Standard U-Net Parts
        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        
        # New Attention Block
        self.freq_attn = FreqAttentionBlock(base_ch*8, split_idx)
        
        self.up1 = Up(base_ch*8, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up3 = Up(base_ch*2, base_ch)
        
        self.outc = nn.Conv2d(base_ch, base_ch, 1)
        self.head = TemporalHead1D(base_ch, num_classes)

    def forward(self, x):
        # Save raw input for the attention block
        raw_x = x 
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Apply the logic: Use global freq knowledge to gate the bottleneck
        x4 = self.freq_attn(raw_x, x4)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        x = self.outc(x)
        x = x.mean(dim=2) 
        return self.head(x)