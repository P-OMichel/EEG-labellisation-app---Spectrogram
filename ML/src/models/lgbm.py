from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from lightgbm import LGBMClassifier

from ML.src.models.registry import register_model

@register_model("lightgbm")
@dataclass
class LightGBMSegmentationModel:
    num_classes: int
    params: Dict[str, Any]

    def __post_init__(self):
        default_params = {
            "objective": "multiclass",
            "num_class": self.num_classes,
            "n_estimators": 300,
            "max_depth": -1,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 42,
            "class_weight": None,
        }
        default_params.update(self.params or {})
        default_params["objective"] = "multiclass"
        default_params["num_class"] = self.num_classes

        self.model = LGBMClassifier(**default_params)

    # NOTE: Compared to DL approach where fit method is defined in trainer.py as it is common between all architectures, here the fit function depends on the model
    # used as they have independent libraries. Consequently the fit function is define at same time as model. This way the fit method can be used later as model.fit()
    # for all ML models. 
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
            fit_kwargs["eval_metric"] = "multi_logloss"
            if sample_weight_val is not None:
                fit_kwargs["eval_sample_weight"] = [sample_weight_val]

        callbacks = []
        if early_stopping_rounds is not None:
            from lightgbm import early_stopping, log_evaluation
            callbacks.append(early_stopping(early_stopping_rounds))
            callbacks.append(log_evaluation(period=10 if verbose else 0))
            fit_kwargs["callbacks"] = callbacks

        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            **fit_kwargs,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def booster_(self):
        return self.model.booster_