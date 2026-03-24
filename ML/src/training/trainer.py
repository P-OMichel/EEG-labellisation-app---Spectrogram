from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from ML.src.training.losses import build_loss


def make_sample_weights(y: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    y = y.astype(int).reshape(-1)
    return class_weights[y]


def _safe_predict_proba(model, X: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def train_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
    class_weights: Optional[np.ndarray] = None,
):
    """
    Generic trainer for tabular ML segmentation.

    Parameters
    ----------
    model:
        Wrapped model exposing fit / predict / optionally predict_proba
    X:
        Shape (n_samples, n_features)
    y:
        Shape (n_samples,)
    config:
        Experiment configuration
    class_weights:
        Optional class weights computed outside the trainer
        (same philosophy as your DL train.py)

    Returns
    -------
    dict with fitted model, metrics and split artifacts
    """
    train_cfg = config.get("train", {})
    loss_cfg = config.get("loss", {})

    random_state = int(train_cfg.get("random_state", 42))
    val_size = float(train_cfg.get("val_size", 0.2))
    early_stopping_rounds = train_cfg.get("early_stopping_rounds", 30)
    verbose = bool(train_cfg.get("verbose", True))
    stratify = bool(train_cfg.get("stratify", True))

    y = y.astype(int).reshape(-1)
    num_classes = int(np.max(y)) + 1
    labels = np.arange(num_classes)

    loss_spec = build_loss(loss_cfg, num_classes=num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    sample_weight_train = None
    sample_weight_val = None

    if loss_spec.use_sample_weights:
        if class_weights is None:
            raise ValueError(
                f"Loss '{loss_spec.name}' requires class_weights, but none were provided."
            )
        class_weights = np.asarray(class_weights, dtype=np.float32)
        sample_weight_train = make_sample_weights(y_train, class_weights)
        sample_weight_val = make_sample_weights(y_val, class_weights)

    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weight_train=sample_weight_train,
        sample_weight_val=sample_weight_val,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    proba_train = _safe_predict_proba(model, X_train)
    proba_val = _safe_predict_proba(model, X_val)

    metrics_train = compute_metrics(y_train, y_train_pred, labels=labels)
    metrics_val = compute_metrics(y_val, y_val_pred, labels=labels)

    return {
        "model": model,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
        "proba_train": proba_train,
        "proba_val": proba_val,
        "class_weights": None if class_weights is None else class_weights.tolist(),
        "loss_spec": {
            "name": loss_spec.name,
            "objective": loss_spec.objective,
            "eval_metric": loss_spec.eval_metric,
            "use_sample_weights": loss_spec.use_sample_weights,
            "notes": loss_spec.notes,
        },
        "split": {
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "val_size": float(val_size),
            "random_state": int(random_state),
            "stratify": bool(stratify),
        },
    }