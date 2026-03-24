from __future__ import annotations

from typing import Callable, Dict, Any


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str):
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name]


def build_model(model_cfg: dict):
    """
    Expected config:
    model:
      name: xgboost | lightgbm
      kwargs:
        num_classes: 10
        ...
    """
    name = model_cfg["name"]
    kwargs = model_cfg.get("kwargs", {}).copy()

    if "num_classes" not in kwargs:
        raise ValueError("model.kwargs must include 'num_classes'")

    num_classes = kwargs.pop("num_classes")

    model_cls = get_model_class(name)
    return model_cls(num_classes=num_classes, params=kwargs)