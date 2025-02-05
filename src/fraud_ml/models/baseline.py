"""Transparent baseline: class-weighted logistic regression on the shared feature pipeline.

The baseline is mandatory (ADR 0003). It fixes the scale of the problem: any advanced model
must beat a well-regularised, class-weighted linear model on PR-AUC *and* at the operating
threshold, otherwise the added complexity is not justified.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_ml.features.pipeline import build_pipeline

BASELINE_NAME = "logistic_regression"


class BaselineParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    C: float = Field(default=0.5, gt=0)
    class_weight: str = "balanced"
    max_iter: int = Field(default=2000, ge=100)
    solver: str = "lbfgs"


def build_baseline(params: BaselineParams, *, seed: int) -> Pipeline:
    estimator = LogisticRegression(
        C=params.C,
        class_weight=params.class_weight,
        max_iter=params.max_iter,
        solver=params.solver,
        random_state=seed,
    )
    return build_pipeline(estimator, scale_numeric=True)
