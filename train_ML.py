from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from ML.src.features.pipeline import build_tabular
from ML.src.models.factory import build_model
from ML.src.training.trainer import train_model


def load_config(path: str) -> dict:
    p = Path(path)
    if p.suffix in {".json"}:
        return json.loads(p.read_text())
    if p.suffix in {".yml", ".yaml"}:
        import yaml
        return yaml.safe_load(p.read_text())
    raise ValueError("Config must be .json or .yaml")


# ---- SAME AS DL ----
def compute_log_inv_class_weights(y_train, num_classes, c=1.02, eps=1e-12):
    flat = y_train.reshape(-1)
    counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
    freq = counts / max(counts.sum(), 1.0)
    weights = 1.0 / (np.log(c + freq) + eps)

    present = counts > 0
    if present.any():
        weights = weights / weights[present].mean()

    weights[~present] = 0.0
    return weights.astype(np.float32), counts.astype(np.int64)


def save_confusion_matrix(y_true, y_pred, out_path, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm_prop = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_prop = np.nan_to_num(cm_prop)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    im = ax.imshow(cm_prop, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix (TEST set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n({cm_prop[i,j]:.2f})",
                    ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["save"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- LOAD DATASET ----
    X_features  =  np.load('X_Y_dataset/X_features_26_02_2026.npy')  # NOTE: put desired datset version here
    Y = np.load('X_Y_dataset/Y_26_02_2026.npy') # NOTE: put desired datset version here


    # ---- CONVERT TO TABULAR AND ADD POSSIBLE LAGS ----
    X_tab, y_tab, group_id, time_id, names = build_tabular(X_features,Y, config = cfg) # NOTE may need to send specific part of config not full one

    # ---- SCALING ---- NOTE: need to check if should be done here or after just on train ?
    scaler = None
    if cfg.get("preprocessing", {}).get("use_robust_scaler", True):
        scaler = RobustScaler()
        X_tab = scaler.fit_transform(X_tab)

    y_tab = y_tab.astype(int)
    num_classes = int(cfg["model"]["kwargs"]["num_classes"])


    # NOTE: possible part for merging class before training

    # --- TRAIN / VALIDATION / TEST SETS ---
    g_unique = pd.unique(group_id) # split by full mask
    g_unique = np.asarray(g_unique)
    rng = 42
    g_temp, g_test = train_test_split(
        g_unique,
        test_size=0.2,
        random_state=rng,
        shuffle=True,
        stratify=None
    )

    g_train, g_val = train_test_split(
        g_temp,
        test_size=0.2,   # 20% of remaining -> 16% of total
        random_state=rng,
        shuffle=True,
        stratify=None
    )

    train_mask = np.isin(group_id, g_train)
    val_mask   = np.isin(group_id, g_val)
    test_mask  = np.isin(group_id, g_test)

    X_train, y_train, gid_train = X_tab[train_mask], y_tab[train_mask], group_id[train_mask]
    X_val,   y_val,   gid_val   = X_tab[val_mask],   y_tab[val_mask],   group_id[val_mask]
    X_test,  y_test,  gid_test  = X_tab[test_mask],  y_tab[test_mask],  group_id[test_mask]

    print("Groups:", len(g_unique), "-> train/val/test:", len(g_train), len(g_val), len(g_test))
    print("Rows:", X_tab.shape[0], "-> train/val/test:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    # ---- CLASS WEIGHTS (ONLY BASED ON TRAIN) ----
    class_weights, counts = compute_log_inv_class_weights(
        y_train,
        num_classes=num_classes
    )

    print("Class counts:", counts)
    print("Class weights:", class_weights)

    # ---- MODEL ----
    model = build_model(cfg["model"]) # NOTE need change here

    # ---- TRAIN (train + val ONLY) ----
    result = train_model(model=model, X=X_train, y=y_train, config=cfg, class_weights=class_weights)
    
    model = result["model"]

    # ============================================================
    # 🔥 TEST EVALUATION (NEW)
    # ============================================================

    y_test_pred = model.predict(X_test)

    from sklearn.metrics import f1_score, accuracy_score

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_test_pred)),
        "macro_f1": float(f1_score(y_test, y_test_pred, average="macro")),
    }

    print("TEST METRICS:", test_metrics)

    # ---- SAVE ----
    joblib.dump(model, out_dir / "model.pkl")

    if scaler is not None:
        joblib.dump(scaler, out_dir / "scaler.joblib")

    # Save metrics
    (out_dir / "metrics.json").write_text(json.dumps({
        "train": result["metrics_train"],
        "val": result["metrics_val"],
        "test": test_metrics,
    }, indent=2))

    # Confusion matrix ON TEST
    class_names = cfg.get("class_names", [str(i) for i in range(num_classes)])

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        out_path=out_dir / "confusion_matrix_test.png",
        class_names=class_names,
    )

    print(f"Saved ML run to {out_dir}")


if __name__ == "__main__":
    main()