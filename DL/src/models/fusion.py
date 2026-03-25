import torch
import torch.nn as nn
import torch.nn.functional as F



class LateFusionSegmentation(nn.Module):
    """
    Late fusion of logits from:
      - 1D model: input x1d -> logits1d (B, C, T)
      - 2D model: input x2d -> logits2d (B, C, T)

    Final logits = alpha * logits1d + (1 - alpha) * logits2d
    """
    def __init__(
        self,
        model_1d: nn.Module,
        model_2d: nn.Module,
        num_classes: int = 10,
        target_len: int = 297,
        learnable_alpha: bool = True,
        init_alpha: float = 0.5,
    ):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d
        self.num_classes = num_classes
        self.target_len = target_len

        if learnable_alpha:
            self.alpha_logit = nn.Parameter(torch.tensor(self._inv_sigmoid(init_alpha), dtype=torch.float32))
        else:
            self.register_buffer("alpha_const", torch.tensor(init_alpha, dtype=torch.float32))
            self.alpha_logit = None

    @staticmethod
    def _inv_sigmoid(p: float) -> float:
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        return torch.log(torch.tensor(p / (1 - p))).item()

    def get_alpha(self):
        if self.alpha_logit is not None:
            return torch.sigmoid(self.alpha_logit)
        return self.alpha_const

    def forward(self, x1d, x2d):
        logits1d = self.model_1d(x1d)   # (B, C, T1)
        logits2d = self.model_2d(x2d)   # (B, C, T2)

        if logits1d.size(-1) != self.target_len:
            logits1d = F.interpolate(logits1d, size=self.target_len, mode="linear", align_corners=False)

        if logits2d.size(-1) != self.target_len:
            logits2d = F.interpolate(logits2d, size=self.target_len, mode="linear", align_corners=False)

        alpha = self.get_alpha()
        fused_logits = alpha * logits1d + (1.0 - alpha) * logits2d

        return {
            "logits": fused_logits,
            "logits_1d": logits1d,
            "logits_2d": logits2d,
            "alpha": alpha,
        }
    


import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: Is fusion code here training all or 1d and 2d networks are already trained ?

class IntermediateFusionSegmentation(nn.Module):
    """
    Intermediate fusion:
      feat1d = model_1d.forward_features(x1d)   -> (B, C1, T)
      feat2d = model_2d.forward_features(x2d)   -> (B, C2, T)

    Then:
      proj1d -> hidden
      proj2d -> hidden
      concat
      fusion conv block
      classifier
    """
    def __init__(
        self,
        model_1d: nn.Module,
        model_2d: nn.Module,
        feat1d_ch: int,
        feat2d_ch: int,
        fusion_ch: int = 128,
        num_classes: int = 10,
        target_len: int = 297,
        use_aux_heads: bool = True,
    ):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d
        self.target_len = target_len
        self.use_aux_heads = use_aux_heads

        self.proj1d = nn.Conv1d(feat1d_ch, fusion_ch, kernel_size=1)
        self.proj2d = nn.Conv1d(feat2d_ch, fusion_ch, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv1d(2 * fusion_ch, fusion_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(fusion_ch, fusion_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_ch),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv1d(fusion_ch, num_classes, kernel_size=1)

        if self.use_aux_heads:
            self.aux_head_1d = nn.Conv1d(feat1d_ch, num_classes, kernel_size=1)
            self.aux_head_2d = nn.Conv1d(feat2d_ch, num_classes, kernel_size=1)

    def forward(self, x1d, x2d):
        feat1d = self.model_1d.forward_features(x1d)  # (B, C1, T1)
        feat2d = self.model_2d.forward_features(x2d)  # (B, C2, T2)

        if feat1d.size(-1) != self.target_len:
            feat1d = F.interpolate(feat1d, size=self.target_len, mode="linear", align_corners=False)
        if feat2d.size(-1) != self.target_len:
            feat2d = F.interpolate(feat2d, size=self.target_len, mode="linear", align_corners=False)

        z1 = self.proj1d(feat1d)  # (B, fusion_ch, T)
        z2 = self.proj2d(feat2d)  # (B, fusion_ch, T)

        fused = torch.cat([z1, z2], dim=1)   # (B, 2*fusion_ch, T)
        fused = self.fusion(fused)           # (B, fusion_ch, T)
        logits = self.head(fused)            # (B, num_classes, T)

        out = {
            "logits": logits,
            "feat_1d": feat1d,
            "feat_2d": feat2d,
        }

        if self.use_aux_heads:
            out["logits_1d_aux"] = self.aux_head_1d(feat1d)
            out["logits_2d_aux"] = self.aux_head_2d(feat2d)

        return out
    

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedIntermediateFusionSegmentation(nn.Module):
    """
    Same as intermediate fusion, but learns a per-time-step, per-channel gate
    between 1D and 2D projected features.
    """
    def __init__(
        self,
        model_1d: nn.Module,
        model_2d: nn.Module,
        feat1d_ch: int,
        feat2d_ch: int,
        fusion_ch: int = 128,
        num_classes: int = 10,
        target_len: int = 297,
    ):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d
        self.target_len = target_len

        self.proj1d = nn.Conv1d(feat1d_ch, fusion_ch, kernel_size=1)
        self.proj2d = nn.Conv1d(feat2d_ch, fusion_ch, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Conv1d(2 * fusion_ch, fusion_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(fusion_ch, fusion_ch, kernel_size=1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv1d(fusion_ch, fusion_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(fusion_ch),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv1d(fusion_ch, num_classes, kernel_size=1)

    def forward(self, x1d, x2d):
        feat1d = self.model_1d.forward_features(x1d)
        feat2d = self.model_2d.forward_features(x2d)

        if feat1d.size(-1) != self.target_len:
            feat1d = F.interpolate(feat1d, size=self.target_len, mode="linear", align_corners=False)
        if feat2d.size(-1) != self.target_len:
            feat2d = F.interpolate(feat2d, size=self.target_len, mode="linear", align_corners=False)

        z1 = self.proj1d(feat1d)  # (B, fusion_ch, T)
        z2 = self.proj2d(feat2d)  # (B, fusion_ch, T)

        gate = self.gate(torch.cat([z1, z2], dim=1))  # (B, fusion_ch, T)
        fused = gate * z1 + (1.0 - gate) * z2
        fused = self.refine(fused)

        logits = self.head(fused)
        return {
            "logits": logits,
            "gate": gate,
            "feat_1d": feat1d,
            "feat_2d": feat2d,
        }