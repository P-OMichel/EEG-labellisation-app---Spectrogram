from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .losses import build_loss


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]
    train_main_loss: List[float]
    val_main_loss: List[float]
    train_aux1_loss: List[float]
    val_aux1_loss: List[float]
    train_aux2_loss: List[float]
    val_aux2_loss: List[float]


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def accuracy_time(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred == target).float().mean().item())


def _prepare_fusion_inputs(
    x1d: torch.Tensor,
    x2d: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1d = x1d.to(device)
    x2d = x2d.to(device)
    y = y.to(device)

    if x1d.ndim == 2:
        x1d = x1d.unsqueeze(1)
    if x1d.ndim != 3 or x1d.shape[1] != 1:
        raise ValueError(f"Expected x1d shape (B,1,T), got {tuple(x1d.shape)}")

    if x2d.ndim == 3:
        x2d = x2d.unsqueeze(1)
    if x2d.ndim != 4 or x2d.shape[1] != 1:
        raise ValueError(f"Expected x2d shape (B,1,F,T), got {tuple(x2d.shape)}")

    if y.ndim != 2:
        raise ValueError(f"Expected y shape (B,T_out), got {tuple(y.shape)}")

    return x1d, x2d, y


def _get_fusion_param_groups(model: torch.nn.Module, cfg: Dict[str, Any]):
    param_groups = []

    lr_1d = float(cfg["model"]["branches"]["model_1d"].get("lr", cfg["train"].get("lr", 1e-3)))
    lr_2d = float(cfg["model"]["branches"]["model_2d"].get("lr", cfg["train"].get("lr", 1e-3)))
    lr_fusion = float(cfg["train"].get("lr", 1e-3))

    if hasattr(model, "model_1d") and model.model_1d is not None:
        params_1d = [p for p in model.model_1d.parameters() if p.requires_grad]
        if params_1d:
            param_groups.append({"params": params_1d, "lr": lr_1d})

    if hasattr(model, "model_2d") and model.model_2d is not None:
        params_2d = [p for p in model.model_2d.parameters() if p.requires_grad]
        if params_2d:
            param_groups.append({"params": params_2d, "lr": lr_2d})

    fusion_module_names = [
        "proj1d",
        "proj2d",
        "fusion",
        "head",
        "aux_head_1d",
        "aux_head_2d",
        "gate",
        "refine",
    ]

    fusion_params = []
    seen = set()

    for name in fusion_module_names:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                if p.requires_grad and id(p) not in seen:
                    fusion_params.append(p)
                    seen.add(id(p))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("model_1d.") or name.startswith("model_2d."):
            continue
        if id(p) not in seen:
            fusion_params.append(p)
            seen.add(id(p))

    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": lr_fusion})

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")

    return param_groups


def build_fusion_losses_from_configs(
    fusion_cfg: Dict[str, Any],
    num_classes: int,
    class_weights: torch.Tensor,
    device: torch.device,
):
    """
    Builds:
      - main loss from fusion config
      - 1D aux loss from model_1d config
      - 2D aux loss from model_2d config
    """
    loss_main = build_loss(
        fusion_cfg["train"]["loss"],
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    branch1_cfg_path = fusion_cfg["model"]["branches"]["model_1d"]["config"]
    branch2_cfg_path = fusion_cfg["model"]["branches"]["model_2d"]["config"]

    branch1_cfg = load_yaml_config(branch1_cfg_path)
    branch2_cfg = load_yaml_config(branch2_cfg_path)

    loss_1d_aux = build_loss(
        branch1_cfg["train"]["loss"],
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    loss_2d_aux = build_loss(
        branch2_cfg["train"]["loss"],
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    return loss_main, loss_1d_aux, loss_2d_aux, branch1_cfg, branch2_cfg


def _compute_fusion_losses(
    out: torch.Tensor | Dict[str, torch.Tensor],
    y: torch.Tensor,
    loss_main_fn,
    loss_1d_aux_fn,
    loss_2d_aux_fn,
    fusion_cfg: Dict[str, Any],
):
    aux_cfg = fusion_cfg["train"].get("aux_loss", {})
    use_aux = bool(aux_cfg.get("use", False))
    w1 = float(aux_cfg.get("weight_1d", 0.0))
    w2 = float(aux_cfg.get("weight_2d", 0.0))

    if isinstance(out, torch.Tensor):
        logits = out
        main_loss = loss_main_fn(logits, y)
        total_loss = main_loss
        return total_loss, {
            "main_loss": float(main_loss.item()),
            "aux1_loss": 0.0,
            "aux2_loss": 0.0,
            "total_loss": float(total_loss.item()),
        }

    if "logits" not in out:
        raise KeyError("Fusion model output dict must contain key 'logits'")

    logits = out["logits"]
    main_loss = loss_main_fn(logits, y)
    total_loss = main_loss

    aux1_val = 0.0
    aux2_val = 0.0

    if use_aux and "logits_1d_aux" in out:
        loss_1d = loss_1d_aux_fn(out["logits_1d_aux"], y)
        total_loss = total_loss + w1 * loss_1d
        aux1_val = float(loss_1d.item())

    if use_aux and "logits_2d_aux" in out:
        loss_2d = loss_2d_aux_fn(out["logits_2d_aux"], y)
        total_loss = total_loss + w2 * loss_2d
        aux2_val = float(loss_2d.item())

    return total_loss, {
        "main_loss": float(main_loss.item()),
        "aux1_loss": aux1_val,
        "aux2_loss": aux2_val,
        "total_loss": float(total_loss.item()),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_main_fn,
    loss_1d_aux_fn,
    loss_2d_aux_fn,
    fusion_cfg,
    device,
):
    model.train()
    tot_loss, tot_main, tot_aux1, tot_aux2, n = 0.0, 0.0, 0.0, 0.0, 0

    for x1d, x2d, y in loader:
        x1d, x2d, y = _prepare_fusion_inputs(x1d, x2d, y, device)

        out = model(x1d, x2d)
        loss, stats = _compute_fusion_losses(
            out=out,
            y=y,
            loss_main_fn=loss_main_fn,
            loss_1d_aux_fn=loss_1d_aux_fn,
            loss_2d_aux_fn=loss_2d_aux_fn,
            fusion_cfg=fusion_cfg,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = y.shape[0]
        tot_loss += stats["total_loss"] * bs
        tot_main += stats["main_loss"] * bs
        tot_aux1 += stats["aux1_loss"] * bs
        tot_aux2 += stats["aux2_loss"] * bs
        n += bs

    return {
        "loss": tot_loss / max(n, 1),
        "main_loss": tot_main / max(n, 1),
        "aux1_loss": tot_aux1 / max(n, 1),
        "aux2_loss": tot_aux2 / max(n, 1),
    }


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    loss_main_fn,
    loss_1d_aux_fn,
    loss_2d_aux_fn,
    fusion_cfg,
    device,
):
    model.eval()
    tot_loss, tot_main, tot_aux1, tot_aux2, n = 0.0, 0.0, 0.0, 0.0, 0

    for x1d, x2d, y in loader:
        x1d, x2d, y = _prepare_fusion_inputs(x1d, x2d, y, device)

        out = model(x1d, x2d)
        loss, stats = _compute_fusion_losses(
            out=out,
            y=y,
            loss_main_fn=loss_main_fn,
            loss_1d_aux_fn=loss_1d_aux_fn,
            loss_2d_aux_fn=loss_2d_aux_fn,
            fusion_cfg=fusion_cfg,
        )

        bs = y.shape[0]
        tot_loss += stats["total_loss"] * bs
        tot_main += stats["main_loss"] * bs
        tot_aux1 += stats["aux1_loss"] * bs
        tot_aux2 += stats["aux2_loss"] * bs
        n += bs

    return {
        "loss": tot_loss / max(n, 1),
        "main_loss": tot_main / max(n, 1),
        "aux1_loss": tot_aux1 / max(n, 1),
        "aux2_loss": tot_aux2 / max(n, 1),
    }


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    num_classes: int,
    class_weights: torch.Tensor,
    device: torch.device,
):
    loss_main_fn, loss_1d_aux_fn, loss_2d_aux_fn, branch1_cfg, branch2_cfg = build_fusion_losses_from_configs(
        fusion_cfg=cfg,
        num_classes=num_classes,
        class_weights=class_weights,
        device=device,
    )

    model.to(device)
    param_groups = _get_fusion_param_groups(model, cfg) # get the learning rates
    optimizer = torch.optim.Adam(param_groups)

    epochs = int(cfg["train"].get("epochs", 50))
    patience = int(cfg["train"].get("early_stop_patience", 10))

    best = float("inf")
    best_state = None
    bad = 0

    hist = TrainHistory(
        train_loss=[],
        val_loss=[],
        train_main_loss=[],
        val_main_loss=[],
        train_aux1_loss=[],
        val_aux1_loss=[],
        train_aux2_loss=[],
        val_aux2_loss=[],
    )

    print("Fusion main loss:", cfg["train"]["loss"]["name"])
    print("1D aux loss:", branch1_cfg["train"]["loss"]["name"])
    print("2D aux loss:", branch2_cfg["train"]["loss"]["name"])

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_main_fn=loss_main_fn,
            loss_1d_aux_fn=loss_1d_aux_fn,
            loss_2d_aux_fn=loss_2d_aux_fn,
            fusion_cfg=cfg,
            device=device,
        )

        va = eval_one_epoch(
            model=model,
            loader=val_loader,
            loss_main_fn=loss_main_fn,
            loss_1d_aux_fn=loss_1d_aux_fn,
            loss_2d_aux_fn=loss_2d_aux_fn,
            fusion_cfg=cfg,
            device=device,
        )

        hist.train_loss.append(tr["loss"])
        hist.val_loss.append(va["loss"])
        hist.train_main_loss.append(tr["main_loss"])
        hist.val_main_loss.append(va["main_loss"])
        hist.train_aux1_loss.append(tr["aux1_loss"])
        hist.val_aux1_loss.append(va["aux1_loss"])
        hist.train_aux2_loss.append(tr["aux2_loss"])
        hist.val_aux2_loss.append(va["aux2_loss"])

        print(
            f"Epoch {ep:03d} | "
            f"train total {tr['loss']:.4f} main {tr['main_loss']:.4f} aux1 {tr['aux1_loss']:.4f} aux2 {tr['aux2_loss']:.4f} | "
            f"val total {va['loss']:.4f} main {va['main_loss']:.4f} aux1 {va['aux1_loss']:.4f} aux2 {va['aux2_loss']:.4f}"
        )

        if va["loss"] < best - 1e-6:
            best = va["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = Path(cfg["save"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(hist.train_loss, label="train_total")
    plt.plot(hist.val_loss, label="val_total")
    plt.plot(hist.train_main_loss, label="train_main")
    plt.plot(hist.val_main_loss, label="val_main")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    torch.save(best_state, out_dir / "best_model.pt")

    return model, hist