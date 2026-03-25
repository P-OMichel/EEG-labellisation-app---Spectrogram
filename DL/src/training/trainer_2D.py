from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .losses import build_loss

@dataclass
class TrainHistory:
    train_loss: list
    val_loss: list

def accuracy_time(pred: torch.Tensor, target: torch.Tensor) -> float:
    # pred/target: (B,T)
    return float((pred == target).float().mean().item())

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    tot, n = 0.0, 0
    for x,y in loader:
        x = x.to(device)  # (B,F,T)
        y = y.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B,1,F,T) - common for 2D conv
        print("x fed to model:", x.shape)    
        logits = model(x)        # expected (B,C,T)
        if x.ndim == 4:
            print("x 4D:", x.shape)
        elif x.ndim == 3:
            print("x 3D:", x.shape)
        else:
            print("x weird:", x.shape)
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bs = x.shape[0]
        tot += float(loss.item()) * bs
        n += bs
    return tot / max(n,1)

@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    tot, n = 0.0, 0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        logits = model(x)
        loss = loss_fn(logits, y)
        bs = x.shape[0]
        tot += float(loss.item()) * bs
        n += bs
    return tot / max(n,1)

def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    num_classes: int,
    class_weights: torch.Tensor,
    device: torch.device,
):
    #class_weights = class_weights.to(device) # weigths , loss and model should be on same device. not necessary to do this line if registered as buffer either manually (focal loss) or automatically (ce option) | if stored as buffer it will be saved in the state_dict. 
    loss_fn = build_loss(cfg["train"]["loss"], num_classes=num_classes, class_weights=class_weights).to(device)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg["train"].get("lr", 1e-3)))
    epochs = int(cfg["train"].get("epochs", 50))
    patience = int(cfg["train"].get("early_stop_patience", 10))

    best = float("inf")
    best_state = None
    bad = 0
    hist = TrainHistory(train_loss=[], val_loss=[])

    for ep in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, optim, loss_fn, device)
        va = eval_one_epoch(model, val_loader, loss_fn, device)
        hist.train_loss.append(tr)
        hist.val_loss.append(va)
        print(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f}")

        if va < best - 1e-6:
            best = va
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # plot
    out_dir = cfg["save"]["out_dir"]
    plt.figure()
    plt.plot(hist.train_loss, label="train")
    plt.plot(hist.val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(__import__("pathlib").Path(out_dir)/"loss_curve.png"), dpi=150)
    plt.close()

    return model, hist
