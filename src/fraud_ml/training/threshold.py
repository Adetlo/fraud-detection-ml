"""Threshold selection as an operational policy (ADR 0005, ``docs/THRESHOLDING.md``).

The threshold is not a modelling constant; it is the point where the business accepts a trade
between missed fraud, wasted reviews and analyst capacity. The analysis sweeps candidate
thresholds on the *validation* split and the policy picks one:

1. candidates must reach ``recall_target`` **and** stay within ``max_review_rate``;
2. among those, the lowest expected illustrative cost wins (ties: fewer alerts);
3. if no candidate satisfies both constraints, the policy is marked unsatisfied and the
   best-recall candidate within the review-rate ceiling is used instead.

A second, higher ``priority_threshold`` (first point where precision reaches
``priority_precision_target``) separates REVIEW from PRIORITY_REVIEW.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fraud_ml.config import CostModel, ThresholdPolicy
from fraud_ml.contracts.score import RiskPolicy

ANALYSIS_COLUMNS = (
    "threshold", "true_positives", "false_positives", "true_negatives", "false_negatives",
    "precision", "recall", "f1", "false_positive_rate", "review_rate", "alerts_per_1000",
    "expected_cost", "expected_cost_per_1000",
)  # fmt: skip


class ThresholdDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float = Field(ge=0, le=1)
    priority_threshold: float = Field(ge=0, le=1)
    low_upper: float = Field(ge=0, le=1)
    policy_satisfied: bool
    rationale: str
    recall: float
    precision: float
    review_rate: float
    alerts_per_1000: float
    expected_cost: float

    def to_policy(self) -> RiskPolicy:
        return RiskPolicy(
            threshold=self.threshold,
            priority_threshold=self.priority_threshold,
            low_upper=self.low_upper,
        )


def candidate_grid(y_prob: np.ndarray, steps: int = 400) -> np.ndarray:
    """Score quantiles plus a uniform grid, so sparse score regions are still covered."""
    p = np.asarray(y_prob, dtype="float64")
    quantiles = np.quantile(p, np.linspace(0, 1, steps + 1)) if len(p) else np.asarray([])
    uniform = np.linspace(0.0, 1.0, 201)
    grid = np.unique(np.round(np.concatenate([quantiles, uniform]), 6))
    return np.asarray(np.clip(grid, 0.0, 1.0), dtype="float64")


def threshold_analysis(
    y_true: np.ndarray, y_prob: np.ndarray, cost_model: CostModel
) -> pd.DataFrame:
    """Vectorised confusion counts for every candidate threshold."""
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(y_prob, dtype="float64")
    n = len(y)
    positives = int(y.sum())
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    cum_pos = np.concatenate([[0], np.cumsum(y[order])])
    grid = candidate_grid(p)
    first_flagged = np.searchsorted(sorted_p, grid, side="left")
    flagged = n - first_flagged
    tp = positives - cum_pos[first_flagged]
    fp = flagged - tp
    fn = positives - tp
    tn = n - flagged - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(flagged > 0, tp / np.maximum(flagged, 1), 0.0)
        recall = np.where(positives > 0, tp / max(positives, 1), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / np.maximum(precision + recall, 1e-12),
            0.0,
        )
        fpr = np.where(fp + tn > 0, fp / np.maximum(fp + tn, 1), 0.0)
    review_rate = flagged / n if n else np.zeros_like(grid)
    cost = (
        fn * cost_model.false_negative_cost
        + fp * cost_model.false_positive_cost
        + tp * cost_model.true_positive_cost
        + tn * cost_model.true_negative_cost
    )
    return pd.DataFrame(
        {
            "threshold": grid,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "review_rate": review_rate,
            "alerts_per_1000": 1000.0 * review_rate,
            "expected_cost": cost,
            "expected_cost_per_1000": 1000.0 * cost / n if n else 0.0,
        }
    )


def _pick(rows: pd.DataFrame) -> pd.Series:
    ordered = rows.sort_values(
        ["expected_cost", "review_rate", "threshold"], ascending=[True, True, False]
    )
    return ordered.iloc[0]


def select_threshold(analysis: pd.DataFrame, policy: ThresholdPolicy) -> ThresholdDecision:
    if analysis.empty:
        raise ValueError("threshold analysis is empty")
    within_capacity = analysis[analysis["review_rate"] <= policy.max_review_rate]
    feasible = within_capacity[within_capacity["recall"] >= policy.recall_target]
    if not feasible.empty:
        chosen = _pick(feasible)
        satisfied = True
        rationale = (
            f"recall {chosen['recall']:.3f} >= target {policy.recall_target:.2f} within "
            f"review-rate ceiling {policy.max_review_rate:.1%}; lowest expected cost among "
            "feasible thresholds"
        )
    elif not within_capacity.empty:
        best_recall = within_capacity["recall"].max()
        chosen = _pick(within_capacity[within_capacity["recall"] >= best_recall - 1e-12])
        satisfied = False
        rationale = (
            f"no threshold reaches recall {policy.recall_target:.2f} within review-rate ceiling "
            f"{policy.max_review_rate:.1%}; using best achievable recall {chosen['recall']:.3f}"
        )
    else:  # every threshold exceeds capacity (degenerate scores); fall back to the strictest
        chosen = analysis.sort_values("threshold", ascending=False).iloc[0]
        satisfied = False
        rationale = "no threshold satisfies the review-rate ceiling; using the strictest threshold"

    threshold = float(chosen["threshold"])
    above = analysis[
        (analysis["threshold"] > threshold)
        & (analysis["precision"] >= policy.priority_precision_target)
    ]
    priority = (
        float(above["threshold"].min())
        if not above.empty
        else min(1.0, threshold + (1.0 - threshold) / 2)
    )
    if priority <= threshold:
        priority = min(1.0, threshold + (1.0 - threshold) / 2)
    return ThresholdDecision(
        threshold=threshold,
        priority_threshold=priority,
        low_upper=round(threshold * policy.low_band_fraction, 6),
        policy_satisfied=satisfied,
        rationale=rationale,
        recall=float(chosen["recall"]),
        precision=float(chosen["precision"]),
        review_rate=float(chosen["review_rate"]),
        alerts_per_1000=float(chosen["alerts_per_1000"]),
        expected_cost=float(chosen["expected_cost"]),
    )


def analysis_to_records(analysis: pd.DataFrame) -> list[dict[str, Any]]:
    records = analysis.loc[:, list(ANALYSIS_COLUMNS)].to_dict(orient="records")
    return [{str(k): v for k, v in record.items()} for record in records]
