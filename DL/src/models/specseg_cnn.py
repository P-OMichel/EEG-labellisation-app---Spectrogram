from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model

class TemporalHeadLinear(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.proj(x)

class TemporalHeadTCN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, hidden: int = 128, layers: int = 3, k: int = 3, dropout: float = 0.1):
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

class SpecEncoder2D(nn.Module):
    def __init__(self, in_ch: int = 1, enc_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, enc_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,F,T) -> (B,enc_ch,F',T')
        return self.net(x)

@register_model("specseg_2dcnn_linear")
class SpecSeg_2DCNN_LinearHead(nn.Module):
    """2D CNN encoder -> pool over F -> linear (1x1 conv) over time."""
    def __init__(self, num_classes: int = 10, enc_ch: int = 128):
        super().__init__()
        self.encoder = SpecEncoder2D(in_ch=1, enc_ch=enc_ch)
        self.head = TemporalHeadLinear(in_ch=enc_ch, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,F,T)
        z = self.encoder(x)
        z = z.mean(dim=2)  # pool over frequency -> (B,enc_ch,T')
        return self.head(z)

@register_model("specseg_2dcnn_tcn")
class SpecSeg_2DCNN_TCNHead(nn.Module):
    def __init__(self, num_classes: int = 10, enc_ch: int = 128, tcn_hidden: int = 128, tcn_layers: int = 3):
        super().__init__()
        self.encoder = SpecEncoder2D(in_ch=1, enc_ch=enc_ch)
        self.head = TemporalHeadTCN(in_ch=enc_ch, num_classes=num_classes, hidden=tcn_hidden, layers=tcn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.mean(dim=2)
        return self.head(z)

@register_model("baseline_specseg")
class BaselineSpecSeg(nn.Module):
    """Light baseline: 2D conv then temporal 1D conv."""
    def __init__(self, num_classes: int = 10, ch: int = 64):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Conv1d(ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv2d(x)      # (B,ch,F,T)
        z = z.mean(dim=2)       # (B,ch,T)
        return self.out(z)

# NOTE: ASTFeatureSeg / CNN14_PANNs_FeatureSeg in notebooks rely on external pretrained models.
# Here we keep placeholders so you can still register them once you plug your feature extractor.
@register_model("ast_feature_seg_stub")
class ASTFeatureSeg(nn.Module):
    def __init__(self, num_classes: int = 10, feat_ch: int = 768):
        super().__init__()
        self.head = nn.Conv1d(feat_ch, num_classes, 1)
        raise NotImplementedError("ASTFeatureSeg requires AST feature extractor; plug it in here.")

@register_model("cnn14_panns_feature_seg_stub")
class CNN14_PANNs_FeatureSeg(nn.Module):
    def __init__(self, num_classes: int = 10, feat_ch: int = 2048):
        super().__init__()
        self.head = nn.Conv1d(feat_ch, num_classes, 1)
        raise NotImplementedError("CNN14_PANNs_FeatureSeg requires PANNs CNN14 feature extractor; plug it in here.")

