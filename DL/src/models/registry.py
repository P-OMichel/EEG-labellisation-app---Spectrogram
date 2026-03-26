from __future__ import annotations
from typing import Dict, Type, Any
import torch.nn as nn
from .loading import load_trained_bundle, load_trained_model_from_bundle, load_config
from .utils import _as_dict, _get_first

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    def deco(cls: Type[nn.Module]):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model name already registered: {name}")
        MODEL_REGISTRY[name] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls
    return deco

# --- UTILS ---
def _get_first(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
        if hasattr(d, k):
            return getattr(d, k)
    return default

# -------------------------------------------------------------------
# SINGLE MODEL
# -------------------------------------------------------------------
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
# FUSION OF MODELS
# -------------------------------------------------------------------
def build_model_fusion(cfg: Dict[str, Any], model1, model2) -> nn.Module:
    """Build a model from a config dict like:
    {"name": "cnn_head", "kwargs": {"num_classes": 10, ...}}
    """
    name = cfg["name"]
    kwargs = cfg.get("kwargs", {}) or {}
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](model1, model2, **kwargs)

# create single model or load existing one. This model is used in the fusion
def build_branch_from_config(branch_cfg: Dict[str, Any]):
    """
    Builds one branch (1D or 2D) from config.

    Rules:
      - builds the model from registry
      - optionally loads checkpoint
      - optionally freezes the model
    """
    # load config file of branch
    branch_cfg_file = load_config(branch_cfg['config'])

    if branch_cfg['pretrained']:
        # load already trained model
        bundle_dir = branch_cfg_file['save']['out_dir']
        model = load_trained_model_from_bundle(bundle_dir, device_pref = 'cuda')
        print(f'load trained model from {bundle_dir} associated to config {branch_cfg['config']}')

    else:
        model = build_model(branch_cfg_file)
        print(f'create untrained model from {branch_cfg['config']}')

    if branch_cfg.get("freeze", False): # no more training of the branch if freeze is True
        for p in model.parameters():
            p.requires_grad = False
        print(f"Froze branch from {branch_cfg['config']}")

    return model

def build_fusion(config: Dict[str, Any]) -> nn.Module:
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

    model_1d = build_branch_from_config(branches_cfg["model_1d"])
    model_2d = build_branch_from_config(branches_cfg["model_2d"])

    # Inject built branches into fusion constructor
    fusion_model = build_model_fusion(config, model_1d, model_2d) #NOTE: to check can also enter with config['model']

    print('build fusion model')

    return fusion_model