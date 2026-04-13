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

class SelfAttention1DBlock(nn.Module):
    """
    Transformer-style self-attention block for 1D sequences.

    Input:  (B, C, T)
    Output: (B, C, T)

    Internally uses MultiheadAttention with shape (B, T, C).
    """
    def __init__(
        self,
        ch: int,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
    ):
        super().__init__()

        if ch % num_heads != 0:
            raise ValueError(f"ch={ch} must be divisible by num_heads={num_heads}")

        self.norm1 = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(
            embed_dim=ch,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(attn_dropout)

        self.norm2 = nn.LayerNorm(ch)
        self.ff = nn.Sequential(
            nn.Linear(ch, ff_mult * ch),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_mult * ch, ch),
        )
        self.drop2 = nn.Dropout(ff_dropout)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x_seq = x.transpose(1, 2)

        # Pre-norm attention
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_seq = x_seq + self.drop1(attn_out)

        # Pre-norm feed-forward
        x_norm = self.norm2(x_seq)
        ff_out = self.ff(x_norm)
        x_seq = x_seq + self.drop2(ff_out)

        # back to (B, C, T)
        return x_seq.transpose(1, 2)


class AttentionBottleneck(nn.Module):
    def __init__(
        self,
        ch: int,
        num_heads: int = 4,
        num_layers: int = 1,
        attn_dropout: float = 0.0,
        ff_mult: int = 4,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                SelfAttention1DBlock(
                    ch=ch,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    ff_mult=ff_mult,
                    ff_dropout=ff_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x input: (B, C, F, T) -> Bottleneck is 4D
        b, c, f, t = x.shape
        
        # --- OPTION 1: Mean Pooling ---
        # Collapse frequency so we can do 1D temporal self-attention
        # (B, C, F, T) -> (B, C, T)
        x_1d = x.mean(dim=2)
        
        # Apply the Multi-Head Attention blocks
        # (B, C, T) -> (B, C, T)
        x_1d = self.blocks(x_1d)
        
        # Restore to 4D so it can be concatenated with U-Net skip connections
        # (B, C, T) -> (B, C, 1, T) -> (B, C, F, T)
        x_4d = x_1d.unsqueeze(2).expand(-1, -1, f, -1)
        
        return x_4d

#------------------------ Full Architectures ---------------------

@register_model("attn_unet_freq_gate_attention")
class UNetFreqAttention(nn.Module):
    def __init__(self, num_classes: int = 10, base_ch: int = 32, split_idx: int = 16,
                 attn_num_heads: int = 4,
                 attn_num_layers: int = 1,
                 attn_dropout: float = 0.0,
                 attn_ff_mult: int = 4,
                 attn_ff_dropout: float = 0.0):
        super().__init__()
        self.split_idx = split_idx
        
        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        
        # 1. Frequency Gating Logic
        self.freq_attn = FreqAttentionBlock(base_ch*8, split_idx)
        
        # 2. Fixed Attention Bottleneck (Option 1)
        self.attn = AttentionBottleneck(
            ch=base_ch*8, # Correctly matches Down3 output
            num_heads=attn_num_heads,
            num_layers=attn_num_layers,
            attn_dropout=attn_dropout,
            ff_mult=attn_ff_mult,
            ff_dropout=attn_ff_dropout,
        )

        self.up1 = Up(base_ch*8, base_ch*4)
        self.up2 = Up(base_ch*4, base_ch*2)
        self.up3 = Up(base_ch*2, base_ch)
        
        self.outc = nn.Conv2d(base_ch, base_ch, 1)
        self.head = TemporalHead1D(base_ch, num_classes)

    def forward(self, x):
        raw_x = x 
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Step A: Inform the channels using high/low freq info (Local)
        x4 = self.freq_attn(raw_x, x4)

        # Step B: Analyze temporal sequences (Global)
        x4 = self.attn(x4)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        x = self.outc(x)
        x = x.mean(dim=2) 
        return self.head(x)