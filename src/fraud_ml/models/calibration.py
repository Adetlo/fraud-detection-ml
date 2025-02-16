"""Probability calibration: measure first, calibrate only when it helps.

Class weighting and ``scale_pos_weight`` deliberately distort the probability scale, so the
raw scores of both models over-state fraud risk. A post-hoc calibrator (Platt scaling or
isotonic regression) is fitted on the validation split with the trained model frozen. It is
*kept only if* it lowers the Brier score by at least ``MIN_RELATIVE_BRIER_GAIN`` without
costing ranking quality (PR-AUC) on the same split. Otherwise the uncalibrated model is used
and the decision is recorded in the metadata.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

CalibrationMethod = Literal["sigmoid", "isotonic"]
MIN_RELATIVE_BRIER_GAIN = 0.02
MAX_PR_AUC_LOSS = 0.005
#: Isotonic regression is a step function fitted on the positives it sees; with few positives it
#: overfits the ranking (observed: -0.03 to -0.08 test PR-AUC on 75 validation positives). It is
#: only considered when the calibration split has at least this many positives.
MIN_POSITIVES_FOR_ISOTONIC = 500
#: Isotonic must beat Platt scaling by this additional relative Brier margin to be preferred.
ISOTONIC_EXTRA_GAIN = 0.10


class CalibrationDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: CalibrationMethod | None
    brier_uncalibrated: float
    brier_by_method: dict[str, float]
    pr_auc_uncalibrated: float
    pr_auc_by_method: dict[str, float]
    rationale: str


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table with quantile bins (equal-count bins suit skewed score distributions)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_prob = np.asarray(y_prob, dtype="float64")
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.asarray([0.0, 1.0])
    bins = np.clip(np.searchsorted(edges, y_prob, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "fraction_positive": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    if table.empty:
        return 0.0
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["mean_predicted"] - table["fraction_positive"]).abs()).sum())


def fit_calibrator(
    model: Any, x_cal: pd.DataFrame, y_cal: pd.Series, method: CalibrationMethod
) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    calibrated.fit(x_cal, y_cal.to_numpy(dtype="int64"))
    return calibrated


def choose_calibration(
    model: Any,
    x_cal: pd.DataFrame,
    y_cal: pd.Series,
    methods: tuple[CalibrationMethod, ...] = ("sigmoid", "isotonic"),
) -> tuple[Any, CalibrationDecision]:
    """Return ``(model_to_use, decision)``; the model is wrapped only when calibration helps.

    Platt scaling (``sigmoid``) is monotone and therefore rank-preserving, so it is the
    preferred method. Isotonic regression is only considered with enough positives and only
    kept if it beats Platt scaling by a clear extra margin.
    """
    y = y_cal.to_numpy(dtype="int64")
    raw = model.predict_proba(x_cal)[:, 1]
    brier_raw = float(brier_score_loss(y, raw))
    pr_raw = float(average_precision_score(y, raw))
    positives = int(y.sum())
    candidates: dict[str, Any] = {}
    brier: dict[str, float] = {}
    pr_auc: dict[str, float] = {}
    notes: list[str] = []
    for method in methods:
        if method == "isotonic" and positives < MIN_POSITIVES_FOR_ISOTONIC:
            notes.append(f"isotonic skipped ({positives} positives < {MIN_POSITIVES_FOR_ISOTONIC})")
            continue
        calibrated = fit_calibrator(model, x_cal, y_cal, method)
        probs = calibrated.predict_proba(x_cal)[:, 1]
        candidates[method] = calibrated
        brier[method] = float(brier_score_loss(y, probs))
        pr_auc[method] = float(average_precision_score(y, probs))

    def acceptable(method: str) -> bool:
        gain = (brier_raw - brier[method]) / brier_raw if brier_raw > 0 else 0.0
        return gain >= MIN_RELATIVE_BRIER_GAIN and pr_raw - pr_auc[method] <= MAX_PR_AUC_LOSS

    best: CalibrationMethod | None = None
    if "sigmoid" in brier and acceptable("sigmoid"):
        best = "sigmoid"
    if "isotonic" in brier and acceptable("isotonic"):
        reference = brier["sigmoid"] if best == "sigmoid" else brier_raw
        if reference > 0 and (reference - brier["isotonic"]) / reference >= ISOTONIC_EXTRA_GAIN:
            best = "isotonic"
    if best is None:
        rationale = (
            f"kept uncalibrated scores: no method improved Brier by >= "
            f"{MIN_RELATIVE_BRIER_GAIN:.0%} without losing more than {MAX_PR_AUC_LOSS} PR-AUC"
        )
        chosen: Any = model
    else:
        rationale = (
            f"applied {best} calibration: Brier {brier_raw:.4f} -> {brier[best]:.4f}, "
            f"PR-AUC {pr_raw:.4f} -> {pr_auc[best]:.4f} on the validation split"
        )
        chosen = candidates[best]
    if notes:
        rationale += "; " + "; ".join(notes)
    decision = CalibrationDecision(
        method=best,
        brier_uncalibrated=brier_raw,
        brier_by_method=brier,
        pr_auc_uncalibrated=pr_raw,
        pr_auc_by_method=pr_auc,
        rationale=rationale,
    )
    return chosen, decision


def unwrap_pipeline(model: Any) -> Pipeline:
    """Return the underlying feature pipeline, through any calibration wrapper."""
    if isinstance(model, Pipeline):
        return model
    if isinstance(model, CalibratedClassifierCV):
        inner = model.estimator
        if isinstance(inner, FrozenEstimator):
            inner = inner.estimator
        return unwrap_pipeline(inner)
    if isinstance(model, FrozenEstimator):
        return unwrap_pipeline(model.estimator)
    raise TypeError(f"cannot unwrap {type(model).__name__}")
