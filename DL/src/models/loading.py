'''
File with functions to load models
'''

from __future__ import annotations

from typing import Optional, Any, NamedTuple
import torch
from .utils import _as_dict, _get_first
from pathlib import Path
import json

# --- LOAD CONFIG ---
def load_config(path: str) -> dict:
    p = Path(path)
    if p.suffix in {".json"}:
        return json.loads(p.read_text())
    if p.suffix in {".yml", ".yaml"}:
        import yaml
        return yaml.safe_load(p.read_text())
    raise ValueError("Config must be .json or .yaml")


class ModelBundle(NamedTuple):
    model: "torch.nn.Module"
    mean: float
    std: float
    spec_F: int
    spec_T: int
    num_classes: int
    arch: str
    bundle_dir: str

# --- load model + stats + ... ---
def load_trained_bundle(bundle_dir: str, device_pref: str = "cuda") -> "ModelBundle":
    """
    Load a trained model bundle from a run directory using DL.src.io.bundle.load_bundle.

    Returns
    -------
    ModelBundle
        Bundle containing:
        - model
        - normalization stats
        - spectrogram dimensions
        - num_classes
        - architecture name
        - bundle_dir

    Notes
    -----
    This function mirrors the behavior of DLBundledPredictor._load(),
    but as a reusable function for training / fusion-building pipelines.
    """

    # -----------------------------
    # Resolve device
    # -----------------------------
    dev = "cpu"
    if device_pref.lower().startswith("cuda") and torch.cuda.is_available():
        dev = "cuda"
    device = torch.device(dev)

    # -----------------------------
    # Import project loader
    # -----------------------------
    try:
        from DL.src.io.bundle import load_bundle  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import DL.src.io.bundle.load_bundle. "
            "Run this code from the repo root."
        ) from e

    # -----------------------------
    # Call load_bundle with several possible signatures
    # -----------------------------
    try:
        loaded = load_bundle(bundle_dir, device=device)
    except TypeError:
        try:
            loaded = load_bundle(bundle_dir, device_str=device.type)
        except TypeError:
            try:
                loaded = load_bundle(bundle_dir, map_location=device)
            except TypeError:
                loaded = load_bundle(bundle_dir)

    # -----------------------------
    # Normalize outputs
    # -----------------------------
    ld = _as_dict(loaded)

    model = _get_first(loaded, ["model"], None) or ld.get("model")
    if model is None:
        raise ValueError("load_bundle(...) did not return a model")

    model.to(device)
    model.eval() # NOTE in case of fusion does eval is correct ? or it blocks training ?

    stats = _get_first(loaded, ["stats"], None)
    if stats is None:
        raise ValueError("load_bundle(...) did not return stats for normalization")

    mean = float(_get_first(stats, ["mean"], 0.0))
    std = float(_get_first(stats, ["std"], 1.0))
    spec_F = int(_get_first(loaded, ["spec_F", "F"], ld.get("spec_F", 45)))
    spec_T = int(_get_first(loaded, ["spec_T", "T"], ld.get("spec_T", 297)))
    num_classes = int(_get_first(loaded, ["num_classes", "C"], ld.get("num_classes", 10)))
    arch = str(_get_first(loaded, ["arch", "model_name", "name"], ld.get("arch", "unknown")))

    bundle = ModelBundle(
        model=model,
        mean=mean,
        std=std,
        spec_F=spec_F,
        spec_T=spec_T,
        num_classes=num_classes,
        arch=arch,
        bundle_dir=bundle_dir,
    )
    return bundle

# --- Only load trained model ---
def load_trained_model_from_bundle(bundle_dir: str, device_pref: str = "cuda") -> torch.nn.Module:
    """
    Load only the trained model from a bundle directory.
    """
    bundle = load_trained_bundle(bundle_dir=bundle_dir, device_pref=device_pref)
    return bundle.model