from __future__ import annotations
from typing import Dict, Type, Any
import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    def deco(cls: Type[nn.Module]):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model name already registered: {name}")
        MODEL_REGISTRY[name] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls
    return deco

def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Build a model from a config dict like:
    {"name": "cnn_head", "kwargs": {"num_classes": 10, ...}}
    """
    name = cfg["name"]
    kwargs = cfg.get("kwargs", {}) or {}
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
