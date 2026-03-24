from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,C,T); target: (B,T)
        B, C, T = logits.shape
        logits2 = logits.permute(0,2,1).reshape(-1, C)   # (B*T, C)
        target2 = target.reshape(-1)                     # (B*T,)
        ce = F.cross_entropy(logits2, target2, weight = self.class_weights, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        # mask ignore_index
        if self.ignore_index is not None:
            m = (target2 != self.ignore_index).float()
            loss = (loss * m).sum() / (m.sum().clamp_min(1.0))
        else:
            loss = loss.mean()
        return loss


class FocalLoss1(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,C,T); target: (B,T)
        B, C, T = logits.shape
        logits2 = logits.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)
        target2 = target.reshape(-1)                       # (B*T,)

        # Unweighted CE to get true p_t = exp(-NLL) = softmax(logits)[y]
        ce_unweighted = F.cross_entropy(
            logits2, target2,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce_unweighted)  # p_t in [0,1], ignore positions give exp(0)=1 (masked out below)

        # Weighted CE carries the class-balancing signal
        ce_weighted = F.cross_entropy(
            logits2, target2,
            weight=self.class_weights,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        # Focal modulation applied to the weighted loss
        loss = (1.0 - pt) ** self.gamma * ce_weighted

        # Mask ignored positions and normalise by the number of valid tokens
        mask = (target2 != self.ignore_index).float()
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    # logits: (B,C,T), target: (B,T)
    probs = torch.softmax(logits, dim=1)
    tgt_oh = F.one_hot(target.clamp_min(0), num_classes=num_classes).permute(0,2,1).float()  # (B,C,T)
    # if ignore_index present (-100), treat as zeros
    tgt_oh = torch.where(target[:,None,:] >= 0, tgt_oh, torch.zeros_like(tgt_oh))
    inter = (probs * tgt_oh).sum(dim=(0,2))
    denom = (probs + tgt_oh).sum(dim=(0,2))
    dice = (2*inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: torch.Tensor, gamma: float = 2.0, dice_weight: float = 1.0, focal_weight: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.focal = FocalLoss(class_weights=class_weights, gamma=gamma, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(logits, target) + self.dice_weight * multiclass_dice_loss(logits, target, self.num_classes)

class FocalDiceLoss1(nn.Module):
    def __init__(self, num_classes: int, class_weights: torch.Tensor, gamma: float = 2.0, dice_weight: float = 1.0, focal_weight: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.focal = FocalLoss1(class_weights=class_weights, gamma=gamma, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(logits, target) + self.dice_weight * multiclass_dice_loss(logits, target, self.num_classes)


# NOTE: not correct 
class HierarchicalBgThenClassLoss(nn.Module):
    """Two-stage hierarchical loss:
    (1) background vs non-background (class 0 is background)
    (2) fine-grained classes for non-background time steps

    Expects logits (B,C,T) and target (B,T) with classes in [0..C-1].
    """
    def __init__(self, num_classes: int, class_weights: torch.Tensor, gamma: float = 2.0, bg_class: int = 0, ignore_index: int = -100):
        super().__init__()
        self.num_classes = num_classes
        self.bg_class = bg_class
        self.ignore_index = ignore_index
        self.focal = FocalLoss(class_weights=class_weights, gamma=gamma, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B,C,T = logits.shape
        # stage1: bg vs non-bg
        bg_logit = logits[:, self.bg_class:self.bg_class+1, :]
        nonbg_logit = torch.logsumexp(torch.cat([logits[:,:self.bg_class,:], logits[:,self.bg_class+1:,:]], dim=1), dim=1, keepdim=True)
        logits_bin = torch.cat([bg_logit, nonbg_logit], dim=1)  # (B,2,T)
        target_bin = torch.where(target==self.bg_class, torch.zeros_like(target), torch.ones_like(target))
        loss_bin = self.focal(logits_bin, target_bin)

        # stage2: fine classes on non-bg positions
        mask = (target != self.bg_class) & (target != self.ignore_index)
        if mask.sum() == 0:
            return loss_bin
        logits_nb = logits.permute(0,2,1)[mask]   # (N, C)
        target_nb = target[mask]                  # (N,)
        loss_nb = F.cross_entropy(logits_nb, target_nb)
        return loss_bin + loss_nb


def build_loss(cfg: Dict[str, Any], num_classes: int, class_weights: torch.Tensor) -> nn.Module:
    name = cfg.get("name", "ce")
    kwargs = cfg.get("kwargs", {}) or {}
    if name == "focal_dice":
        return FocalDiceLoss(num_classes=num_classes, class_weights=class_weights, **kwargs)
    if name == "focal_dice1":
        return FocalDiceLoss1(num_classes=num_classes, class_weights=class_weights, **kwargs)
    if name == "hierarchical":
        return HierarchicalBgThenClassLoss(num_classes=num_classes, class_weights=class_weights, **kwargs)
    if name == "focal":
        return FocalLoss(class_weights=class_weights,**kwargs)
    if name == "focal1":
        return FocalLoss1(class_weights=class_weights,**kwargs)
    if name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    raise KeyError(f"Unknown loss '{name}'")
