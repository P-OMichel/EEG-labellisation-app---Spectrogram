from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import torch
import numpy as np

from ..models.registry import build_model

@dataclass
class ModelBundle:
    model: torch.nn.Module
    config: Dict[str, Any]
    stats: Dict[str, Any]
    label_map: Optional[Dict[str, Any]] = None

def save_bundle(
    out_dir: str | Path,
    model: torch.nn.Module,
    model_cfg: Dict[str, Any],
    stats: Dict[str, Any],
    label_map: Optional[Dict[str, Any]] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps({"model": model_cfg}, indent=2))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    if label_map is not None:
        (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2))

    torch.save(model.state_dict(), out_dir / "model.pt")

def load_bundle(path: str | Path, map_location: str | torch.device = "cpu") -> ModelBundle:
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    model_cfg = cfg["model"]
    model = build_model(model_cfg)
    state = torch.load(path / "model.pt", map_location=map_location)
    model.load_state_dict(state)
    model.eval()

    stats = json.loads((path / "stats.json").read_text())
    label_map_path = path / "label_map.json"
    label_map = json.loads(label_map_path.read_text()) if label_map_path.exists() else None
    return ModelBundle(model=model, config=cfg, stats=stats, label_map=label_map)
