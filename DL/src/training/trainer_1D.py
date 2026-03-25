from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .losses import build_loss


@dataclass
class TrainHistory:
    train_loss: list
    val_loss: list


def accuracy_time(pred: torch.Tensor, target: torch.Tensor) -> float:
    # pred/target: (B, T_out)
    return float((pred == target).float().mean().item())


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    tot, n = 0.0, 0

    for x, y in loader:
        x = x.to(device)   # expected (B,T) or (B,1,T)
        y = y.to(device)   # expected (B,T_out), e.g. (B,297)

        if x.ndim == 2:
            x = x.unsqueeze(1)   # (B,T) -> (B,1,T)

        if x.ndim != 3:
            raise ValueError(f"Expected x with 3 dims (B,1,T), got {tuple(x.shape)}")

        if x.shape[1] != 1:
            raise ValueError(f"Expected x shape (B,1,T), got {tuple(x.shape)}")

        # debug print
        print("x fed to model:", x.shape)

        logits = model(x)   # expected (B,C,T_out)

        if logits.ndim != 3:
            raise ValueError(f"Expected logits with 3 dims (B,C,T_out), got {tuple(logits.shape)}")

        if y.ndim != 2:
            raise ValueError(f"Expected y with 2 dims (B,T_out), got {tuple(y.shape)}")

        if logits.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch mismatch between logits and y: {tuple(logits.shape)} vs {tuple(y.shape)}"
            )

        if logits.shape[-1] != y.shape[-1]:
            raise ValueError(
                f"Time-length mismatch between logits and y: {tuple(logits.shape)} vs {tuple(y.shape)}"
            )

        loss = loss_fn(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        tot += float(loss.item()) * bs
        n += bs

    return tot / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    tot, n = 0.0, 0

    for x, y in loader:
        x = x.to(device)   # expected (B,T) or (B,1,T)
        y = y.to(device)   # expected (B,T_out)

        if x.ndim == 2:
            x = x.unsqueeze(1)   # (B,T) -> (B,1,T)

        if x.ndim != 3:
            raise ValueError(f"Expected x with 3 dims (B,1,T), got {tuple(x.shape)}")

        if x.shape[1] != 1:
            raise ValueError(f"Expected x shape (B,1,T), got {tuple(x.shape)}")

        logits = model(x)

        if logits.ndim != 3:
            raise ValueError(f"Expected logits with 3 dims (B,C,T_out), got {tuple(logits.shape)}")

        if y.ndim != 2:
            raise ValueError(f"Expected y with 2 dims (B,T_out), got {tuple(y.shape)}")

        if logits.shape[0] != y.shape[0]:
            raise ValueError(
                f"Batch mismatch between logits and y: {tuple(logits.shape)} vs {tuple(y.shape)}"
            )

        if logits.shape[-1] != y.shape[-1]:
            raise ValueError(
                f"Time-length mismatch between logits and y: {tuple(logits.shape)} vs {tuple(y.shape)}"
            )

        loss = loss_fn(logits, y)

        bs = x.shape[0]
        tot += float(loss.item()) * bs
        n += bs

    return tot / max(n, 1)


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    num_classes: int,
    class_weights: torch.Tensor,
    device: torch.device,
):
    loss_fn = build_loss(
        cfg["train"]["loss"],
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg["train"].get("lr", 1e-3)))
    epochs = int(cfg["train"].get("epochs", 50))
    patience = int(cfg["train"].get("early_stop_patience", 10))

    best = float("inf")
    best_state = None
    bad = 0
    hist = TrainHistory(train_loss=[], val_loss=[])

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, loss_fn, device)
        va = eval_one_epoch(model, val_loader, loss_fn, device)

        hist.train_loss.append(tr)
        hist.val_loss.append(va)

        print(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")

        if va < best - 1e-6:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # plot loss curves
    out_dir = cfg["save"]["out_dir"]
    plt.figure()
    plt.plot(hist.train_loss, label="train")
    plt.plot(hist.val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(__import__("pathlib").Path(out_dir) / "loss_curve.png"), dpi=150)
    plt.close()

    return model, hist