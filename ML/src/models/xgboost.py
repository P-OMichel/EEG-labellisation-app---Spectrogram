from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from xgboost import XGBClassifier

from ML.src.models.registry import register_model

@register_model("xgboost")
@dataclass
class XGBoostSegmentationModel:
    num_classes: int
    params: Dict[str, Any]

    def __post_init__(self):
        default_params = {
            "objective": "multi:softprob",
            "num_class": self.num_classes,
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
        }
        default_params.update(self.params or {})
        default_params["objective"] = "multi:softprob"
        default_params["num_class"] = self.num_classes

        self.model = XGBClassifier(**default_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight_train: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = 30,
        verbose: bool = True,
    ):
        fit_kwargs = {}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if sample_weight_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]

        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            verbose=verbose,
            **fit_kwargs,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_booster(self):
        return self.model.get_booster()