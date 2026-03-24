from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# NOTE: need to check with notebook if losses are just called like that, also if they need to be define at same time as model if they are not same syntax between different model libraries.

@dataclass
class MLLossSpec:
    name: str
    objective: str
    eval_metric: str
    use_sample_weights: bool = False
    notes: str = ""


def build_loss(cfg: Dict[str, Any], num_classes: int) -> MLLossSpec:
    """
    Map experiment loss names to ML-compatible objectives.

    Notes
    -----
    - Tree models naturally optimize log-loss style objectives.
    - Dice / focal_dice are approximated through weighted multiclass log-loss.
    - Hierarchical is better handled by a two-stage pipeline, but here we keep
      a weighted multiclass fallback so the generic trainer can still run.
    """
    name = cfg.get("name", "ce")
    _ = num_classes  # kept for future extension / compatibility

    if name == "ce":
        return MLLossSpec(
            name="ce",
            objective="multiclass",
            eval_metric="multi_logloss",
            use_sample_weights=False,
            notes="Standard multiclass log-loss.",
        )

    if name == "focal":
        return MLLossSpec(
            name="focal",
            objective="multiclass",
            eval_metric="multi_logloss",
            use_sample_weights=True,
            notes="Approximate focal loss via sample weighting.",
        )

    if name == "focal_dice":
        return MLLossSpec(
            name="focal_dice",
            objective="multiclass",
            eval_metric="multi_logloss",
            use_sample_weights=True,
            notes="Approximate focal+dice via weighted multiclass log-loss; report F1/IoU separately.",
        )

    if name == "hierarchical":
        return MLLossSpec(
            name="hierarchical",
            objective="multiclass",
            eval_metric="multi_logloss",
            use_sample_weights=True,
            notes="Fallback weighted multiclass objective. True hierarchical behavior is better done in a 2-stage setup.",
        )

    raise KeyError(f"Unknown ML loss '{name}'")