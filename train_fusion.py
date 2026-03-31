from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from DL.src.datasets.make_dataset import FusionSegDataset, compute_norm_stats, apply_norm

from DL.src.models.registry import build_fusion
from DL.src.io.bundle import save_bundle
from DL.src.training.trainer_fusion import fit

# import fusion model modules so they register themselves
# 1D
from DL.src.models import unet_bottleneck_tcn_1D, unet_output_tcn_1D, unet_2tcn_1D, unet_bottleneck_tcn_att_1D
# 2D 
from DL.src.models import specseg_cnn, unet_temporal, resattn_unet, conv2_temporal_head, resattn_unet_tcn   
# fusion
from DL.src.models import fusion


def load_config(path: str) -> dict:
    p = Path(path)
    if p.suffix in {".json"}:
        return json.loads(p.read_text())
    if p.suffix in {".yml", ".yaml"}:
        import yaml
        return yaml.safe_load(p.read_text())
    raise ValueError("Config must be .json or .yaml")


@torch.no_grad()
def save_test_confusion_matrix_fusion(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    out_path: Path,
    class_names: list[str],
    device: torch.device,
) -> None:
    model.eval()
    y_true_all = []
    y_pred_all = []

    for x1d, x2d, yb in test_loader:
        if x1d.ndim == 2:
            x1d = x1d.unsqueeze(1)   # (B,T) -> (B,1,T)
        if x2d.ndim == 3:
            x2d = x2d.unsqueeze(1)   # (B,F,T) -> (B,1,F,T)

        x1d = x1d.to(device)
        x2d = x2d.to(device)
        yb = yb.to(device)

        if x1d.ndim != 3:
            raise ValueError(f"Expected x1d with 3 dims (B,1,T), got {tuple(x1d.shape)}")
        if x2d.ndim != 4:
            raise ValueError(f"Expected x2d with 4 dims (B,1,F,T), got {tuple(x2d.shape)}")

        out = model(x1d, x2d)
        logits = out["logits"] if isinstance(out, dict) else out

        if logits.ndim != 3:
            raise ValueError(f"Expected logits with 3 dims, got shape {tuple(logits.shape)}")

        # logits: (B,C,T) or (B,T,C)
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
    cm_prop = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_prop = np.nan_to_num(cm_prop)

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

    thresh = cm_prop.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            prop = cm_prop[i, j]
            text = f"{count}\n({prop:.2f})"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if prop > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_log_inv_class_weights(
    y_train: np.ndarray,
    num_classes: int,
    c: float = 1.02,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    flat = y_train.reshape(-1)
    counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / (np.log(c + freq) + eps)

    present = counts > 0
    if present.any():
        weights = weights / weights[present].mean()

    weights[~present] = 0.0
    return weights.astype(np.float32), counts.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["save"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load aligned multimodal arrays
    # -------------------------
    x_eeg_path = cfg["data"]["x_eeg_path"]
    x_spec_path = cfg["data"]["x_spec_path"]
    y_path = cfg["data"]["y_path"]

    X1 = np.load(x_eeg_path)   # raw 1D, shape (N,T)
    X2 = np.load(x_spec_path)  # spectrogram, shape (N,F,Tspec)
    y = np.load(y_path)        # mask, shape (N,T_out)

    if len(X1) != len(X2) or len(X1) != len(y):
        raise ValueError(
            f"Length mismatch: len(X1)={len(X1)}, len(X2)={len(X2)}, len(y)={len(y)}"
        )

    # -------------------------
    # Split ONCE on indices
    # -------------------------
    test_size = float(cfg["data"].get("test_size", 0.2))
    val_size = float(cfg["data"].get("val_size", 0.2))

    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(idx, test_size=test_size, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=val_size, random_state=42)

    X1_train, X1_val, X1_test = X1[idx_train], X1[idx_val], X1[idx_test]
    X2_train, X2_val, X2_test = X2[idx_train], X2[idx_val], X2[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # -------------------------
    # Preprocess 1D branch
    # -------------------------
    stats_1d = compute_norm_stats(X1_train)
    X1_train = apply_norm(X1_train, stats_1d)
    X1_val   = apply_norm(X1_val, stats_1d)
    X1_test  = apply_norm(X1_test, stats_1d)

    # -------------------------
    # Preprocess 2D branch
    # -------------------------
    X2_train = np.log1p(X2_train + 1e-11)
    X2_val   = np.log1p(X2_val + 1e-11)
    X2_test  = np.log1p(X2_test + 1e-11)

    stats_2d = compute_norm_stats(X2_train)
    X2_train = apply_norm(X2_train, stats_2d)
    X2_val   = apply_norm(X2_val, stats_2d)
    X2_test  = apply_norm(X2_test, stats_2d)

    # -------------------------
    # Datasets / loaders
    # -------------------------
    bs = int(cfg["train"].get("batch_size", 16))

    train_ds = FusionSegDataset(X1_train, X2_train, y_train)
    val_ds   = FusionSegDataset(X1_val, X2_val, y_val)
    test_ds  = FusionSegDataset(X1_test, X2_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    # -------------------------
    # Class weights from TRAIN ONLY
    # -------------------------
    num_classes = int(cfg["model"]["kwargs"]["num_classes"])
    weights_np, counts_np = compute_log_inv_class_weights(y_train, num_classes=num_classes, c=1.02)
    print("Train class counts:", counts_np)
    print("Train class weights (log-inv):", weights_np)
    class_weights = torch.tensor(weights_np, dtype=torch.float32)

    # -------------------------
    # Model
    # -------------------------
    model = build_fusion(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, hist = fit(
        model,
        train_loader,
        val_loader,
        cfg,
        num_classes=num_classes,
        class_weights=class_weights,
        device=device,
    )

    # -------------------------
    # Save bundle
    # -------------------------
    save_bundle(
        out_dir,
        model=model,
        model_cfg=cfg["model"],
        stats={
            "mean_1d": stats_1d.mean,
            "std_1d": stats_1d.std,
            "mean_2d": stats_2d.mean,
            "std_2d": stats_2d.std,
        },
        label_map=cfg.get("label_map", None),
    )
    print(f"Saved bundle to: {out_dir}")

    # -------------------------
    # Confusion matrix on test set
    # -------------------------
    classes = [
        "ok", "alpha-sup", "IES", "gc", "shallow",
        "gamma", "eye artifact", "HF artifact",
        "large artifact", "awake"
    ]

    if len(classes) != num_classes:
        print(
            f"[WARN] Provided {len(classes)} class names but model has num_classes={num_classes}. "
            f"Using numeric labels instead."
        )
        classes = [str(i) for i in range(num_classes)]

    cm_path = out_dir / "confusion_matrix.png"
    save_test_confusion_matrix_fusion(model, test_loader, cm_path, classes, device)
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()