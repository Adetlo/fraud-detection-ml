"""Advanced model: XGBoost gradient-boosted trees on the shared feature pipeline (ADR 0004).

Class imbalance is handled with ``scale_pos_weight``. The default uses the square root of the
negative/positive ratio: the full ratio maximises recall at the cost of badly inflated
probabilities, the square root keeps ranking quality while leaving less for the calibration
layer to undo. Early stopping on the *validation* split (never the test split) picks the number
of trees.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_ml.features.pipeline import build_pipeline

GRADIENT_BOOSTING_NAME = "xgboost"


class GradientBoostingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_estimators: int = Field(default=600, ge=10)
    max_depth: int = Field(default=4, ge=1, le=12)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    min_child_weight: float = Field(default=5.0, ge=0)
    reg_lambda: float = Field(default=1.0, ge=0)
    gamma: float = Field(default=0.0, ge=0)
    early_stopping_rounds: int = Field(default=40, ge=1)
    scale_pos_weight_mode: Literal["balanced", "sqrt", "none"] = "sqrt"
    n_jobs: int = Field(default=2, ge=1)


def scale_pos_weight(y: pd.Series | np.ndarray, mode: str) -> float:
    positives = float(np.sum(np.asarray(y, dtype=bool)))
    negatives = float(len(y) - positives)
    if mode == "none" or positives == 0:
        return 1.0
    ratio = negatives / positives
    return float(np.sqrt(ratio)) if mode == "sqrt" else ratio


def build_gradient_boosting(
    params: GradientBoostingParams, *, seed: int, pos_weight: float
) -> Pipeline:
    estimator = XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        min_child_weight=params.min_child_weight,
        reg_lambda=params.reg_lambda,
        gamma=params.gamma,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=params.early_stopping_rounds,
        tree_method="hist",
        n_jobs=params.n_jobs,
        random_state=seed,
        verbosity=0,
    )
    return build_pipeline(estimator, scale_numeric=False)


def fit_gradient_boosting(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Pipeline:
    """Fit preprocessing on the training split only, then boost with validation early stopping."""
    preprocessing: Pipeline = pipeline[:-1]
    transformed_train = preprocessing.fit_transform(x_train)
    transformed_validation = preprocessing.transform(x_validation)
    model: XGBClassifier = pipeline.named_steps["model"]
    model.fit(
        transformed_train,
        y_train.to_numpy(dtype="int64"),
        eval_set=[(transformed_validation, y_validation.to_numpy(dtype="int64"))],
        verbose=False,
    )
    return pipeline


def best_iteration(pipeline: Pipeline) -> int | None:
    model = pipeline.named_steps["model"]
    value = getattr(model, "best_iteration", None)
    return int(value) if value is not None else None
