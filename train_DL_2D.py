from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from DL.src.datasets.make_dataset import SpectrogramSegDataset, compute_norm_stats, apply_norm
from DL.src.models.registry import build_model
from DL.src.io.bundle import save_bundle
from DL.src.training.trainer_2D import fit

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# NOTE:   Import model modules so they register themselves| if not registered load config will not work
from DL.src.models import specseg_cnn, unet_temporal, resattn_unet, conv2_temporal_head, resattn_unet_tcn  


def load_config(path: str) -> dict:
    p = Path(path)
    if p.suffix in {".json"}:
        return json.loads(p.read_text())
    if p.suffix in {".yml", ".yaml"}:
        import yaml
        return yaml.safe_load(p.read_text())
    raise ValueError("Config must be .json or .yaml")


# ---  evaluate + save confusion matrix ---
@torch.no_grad()
def save_test_confusion_matrix(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    out_path: Path,
    class_names: list[str],
    device: torch.device,
) -> None:
    model.eval()
    y_true_all = []
    y_pred_all = []

    for xb, yb in test_loader:
        if xb.ndim == 3:
            xb = xb.unsqueeze(1)
        xb = xb.to(device)
        yb = yb.to(device)
        if xb.ndim != 4:
            raise ValueError(f"Expected xb with 4 dims (B,1,F,T), got {tuple(xb.shape)}")

        # Handle possible swapped dims
        if xb.shape[1] != 1 and xb.shape[0] == 1:
            xb = xb.permute(1, 0, 2, 3).contiguous()

        logits = model(xb)

        if logits.ndim != 3:
            raise ValueError(f"Expected logits with 3 dims, got shape {tuple(logits.shape)}")

        # (B,C,T) or (B,T,C)
        if logits.shape[1] == len(class_names):
            pred = logits.argmax(dim=1)
        elif logits.shape[2] == len(class_names):
            pred = logits.argmax(dim=2)
        else:
            raise ValueError(
                f"Cannot infer class dimension from logits shape {tuple(logits.shape)} "
                f"and num_classes={len(class_names)}"
            )

        y_true_all.append(yb.reshape(-1).cpu().numpy())
        y_pred_all.append(pred.reshape(-1).cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    # --- row-normalized proportions ---
    cm_prop = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_prop = np.nan_to_num(cm_prop)

    # --- Plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    im = ax.imshow(cm_prop, interpolation="nearest", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Proportion")

    ax.set_title("Confusion Matrix (Test set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # --- Annotate cells with count + proportion ---
    thresh = cm_prop.max() * 0.6

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            prop = cm_prop[i, j]
            text = f"{count}\n({prop:.2f})"
            ax.text(j,i,text,ha="center",va="center",color="white" if prop > thresh else "black",fontsize=8,)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---- Compute weigths for classes | useful when imbalance dataset -------
# weights_k = 1 / log(c + freq_k)
# where freq_k = count_k / total. (c slightly > 1)
# ----------------------------
def compute_log_inv_class_weights(y_train: np.ndarray, num_classes: int, c: float = 1.02, eps: float = 1e-12) -> np.ndarray:
    flat = y_train.reshape(-1)
    counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / (np.log(c + freq) + eps)
    # normalize mean weight over present classes to 1
    present = counts > 0
    if present.any():
        weights = weights / weights[present].mean()
    # missing classes -> weight 0 (won't matter for CE)
    weights[~present] = 0.0
    return weights.astype(np.float32), counts.astype(np.int64)    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["save"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset (same files as notebooks) ----
    x_spec_path = cfg["data"].get("x_spec_path", "X_Y_dataset/X_spec_24_03_2026.npy")
    y_path = cfg["data"].get("y_path", "X_Y_dataset/Y_24_03_2026.npy.npy")
    X = np.load(x_spec_path)  # (N,F,T)
    y = np.load(y_path)       # (N,T)

    # ---- Split ----
    test_size = float(cfg["data"].get("test_size", 0.2))
    val_size = float(cfg["data"].get("val_size", 0.2))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)

    # ---- Log transform ----
    X_train = np.log1p(X_train + 0.00000000001)
    X_val   = np.log1p(X_val + 0.00000000001)
    X_test  = np.log1p(X_test + 0.00000000001)

    # ---- Normalize (dataset-level) ----
    stats = compute_norm_stats(X_train)
    X_train = apply_norm(X_train, stats)
    X_val   = apply_norm(X_val, stats)
    X_test  = apply_norm(X_test, stats)

    # ---- DataLoaders ----
    bs = int(cfg["train"].get("batch_size", 16))
    train_ds = SpectrogramSegDataset(X_train, y_train)
    val_ds   = SpectrogramSegDataset(X_val, y_val)
    test_ds  = SpectrogramSegDataset(X_test, y_test)  

    # ----------------Class weights from TRAIN ONLY (log-inverse; stable for heavy imbalance)-------------------
    num_classes = int(cfg["model"]["kwargs"]["num_classes"])
    weights_np, counts_np = compute_log_inv_class_weights(y_train, num_classes=num_classes, c=1.02)
    print("Train class counts:", counts_np)
    print("Train class weights (log-inv):", weights_np)
    class_weights = torch.tensor(weights_np, dtype=torch.float32) #.to(device)  # NOTE: remove to device here as later it is sent to devce when loss is created

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)  # --- NEW ---

    # ---- Model ----
    model = build_model(cfg["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hist = fit(model, train_loader, val_loader, cfg, num_classes=num_classes, class_weights=class_weights, device=device)

    # ---- Save HF-like bundle ----
    save_bundle(
        out_dir,
        model=model,
        model_cfg=cfg["model"],
        stats={"mean": stats.mean, "std": stats.std},
        label_map=cfg.get("label_map", None),
    )
    print(f"Saved bundle to: {out_dir}")

    # ---- confusion matrix on test set ----
    classes = ["ok", "alpha-sup", "IES", "gc", "shallow", "gamma",
               "eye artifact", "HF artifact", "large artifact", "awake"]

    if len(classes) != num_classes:
        print(f"[WARN] Provided {len(classes)} class names but model has num_classes={num_classes}. "
              f"Using numeric labels instead.")
        classes = [str(i) for i in range(num_classes)]

    cm_path = out_dir / "confusion_matrix.png"
    save_test_confusion_matrix(model, test_loader, cm_path, classes, device)
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()