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






# -------------------------------------------------------------------
# 3) Build one branch from config
# -------------------------------------------------------------------

def build_branch_from_config(branch_cfg: Dict[str, Any], branch_name: str):
    """
    Builds one branch (1D or 2D) from config.

    Rules:
      - if enabled=False -> returns None
      - builds the model from registry
      - optionally loads checkpoint
      - optionally freezes the model
    """
    enabled = _safe_get(branch_cfg, "enabled", True)
    if not enabled:
        print(f"[build_branch_from_config] {branch_name}: disabled")
        return None

    model_name = branch_cfg["name"]
    model_kwargs = _safe_get(branch_cfg, "kwargs", {})

    model = build_model(model_name, **model_kwargs)
    print(f"[build_branch_from_config] {branch_name}: built '{model_name}'")

    ckpt_cfg = _safe_get(branch_cfg, "checkpoint", {})
    use_ckpt = _safe_get(ckpt_cfg, "use", False)

    if use_ckpt:
        ckpt_path = ckpt_cfg["path"]
        strict = _safe_get(ckpt_cfg, "strict", True)
        key = _safe_get(ckpt_cfg, "key", None)

        load_checkpoint_into_model(
            model=model,
            checkpoint_path=ckpt_path,
            strict=strict,
            key=key,
            map_location="cpu",
        )
        print(f"[build_branch_from_config] {branch_name}: loaded checkpoint from '{ckpt_path}'")

    freeze = _safe_get(branch_cfg, "freeze", False)
    if freeze:
        set_requires_grad(model, False)
        print(f"[build_branch_from_config] {branch_name}: frozen")
    else:
        print(f"[build_branch_from_config] {branch_name}: trainable")

    return model


# -------------------------------------------------------------------
# 4) Build fusion model from config
# -------------------------------------------------------------------

def build_fusion_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Builds the full fusion model from config.

    Expected structure:
      config["model"]
      config["branches"]["model_1d"]
      config["branches"]["model_2d"]

    It:
      - builds model_1d
      - builds model_2d
      - injects them into the fusion model constructor
    """
    if "model" not in config:
        raise KeyError("Config must contain a 'model' section.")
    if "branches" not in config:
        raise KeyError("Config must contain a 'branches' section.")

    branches_cfg = config["branches"]

    if "model_1d" not in branches_cfg:
        raise KeyError("Config['branches'] must contain 'model_1d'.")
    if "model_2d" not in branches_cfg:
        raise KeyError("Config['branches'] must contain 'model_2d'.")

    model_1d = build_branch_from_config(branches_cfg["model_1d"], branch_name="model_1d")
    model_2d = build_branch_from_config(branches_cfg["model_2d"], branch_name="model_2d")

    fusion_name = config["model"]["name"]
    fusion_kwargs = _safe_get(config["model"], "kwargs", {}).copy()

    # Inject built branches into fusion constructor
    fusion_model = build_model(
        fusion_name,
        model_1d=model_1d,
        model_2d=model_2d,
        **fusion_kwargs,
    )

    print(f"[build_fusion_from_config] built fusion model '{fusion_name}'")
    print(f"[build_fusion_from_config] trainable params: {count_trainable_parameters(fusion_model):,}")

    return fusion_model