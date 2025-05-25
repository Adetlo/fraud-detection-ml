"""Evaluation metrics, segment performance, feature importance and plots.

Accuracy is intentionally absent. With a 1% positive rate, predicting "never fraud" scores
99% accuracy and catches nothing. The primary ranking metric is PR-AUC (average precision);
ROC-AUC is reported but can look optimistic under severe imbalance because the false-positive
*rate* is diluted by the huge negative class (``docs/MODEL_EVALUATION.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fraud_ml.config import CostModel
from fraud_ml.contracts.score import MetricsSummary
from fraud_ml.models.calibration import calibration_table, unwrap_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def confusion_counts(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    y = np.asarray(y_true, dtype=bool)
    flagged = np.asarray(y_prob, dtype="float64") >= threshold
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    tn = int(np.sum(~flagged & ~y))
    return tp, fp, tn, fn


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_model: CostModel,
) -> MetricsSummary:
    y = np.asarray(y_true, dtype="int64")
    p = np.asarray(y_prob, dtype="float64")
    n = len(y)
    positives = int(y.sum())
    tp, fp, tn, fn = confusion_counts(y, p, threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    review_rate = (tp + fp) / n if n else 0.0
    cost = cost_model.expected_cost(tp, fp, tn, fn)
    distinct_classes = len(np.unique(y)) == 2
    return MetricsSummary(
        pr_auc=float(average_precision_score(y, p)) if distinct_classes else 0.0,
        roc_auc=float(roc_auc_score(y, p)) if distinct_classes else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        brier_score=float(brier_score_loss(y, p)),
        review_rate=review_rate,
        alerts_per_1000=1000.0 * review_rate,
        expected_cost=cost,
        expected_cost_per_1000=1000.0 * cost / n if n else 0.0,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        n_rows=n,
        n_positives=positives,
        threshold=threshold,
    )


DEFAULT_SEGMENTS: tuple[str, ...] = ("currency", "channel", "transaction_type", "cross_border")
MIN_SEGMENT_ROWS = 50


def segment_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_model: CostModel,
    segments: tuple[str, ...] = DEFAULT_SEGMENTS,
) -> pd.DataFrame:
    """Performance per operational segment (not a demographic fairness analysis)."""
    work = frame.copy()
    bene = work["beneficiary_country"].astype("string")
    work["cross_border"] = np.where(
        bene.notna() & (bene != work["country"].astype("string")), "CROSS_BORDER", "DOMESTIC"
    )
    rows: list[dict[str, Any]] = []
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    for segment in segments:
        for value, idx in work.groupby(segment, sort=True).indices.items():
            if len(idx) < MIN_SEGMENT_ROWS:
                continue
            m = compute_metrics(y[idx], p[idx], threshold, cost_model)
            defined = 0 < m.n_positives < m.n_rows  # ranking metrics need both classes
            rows.append(
                {
                    "segment": segment,
                    "value": str(value),
                    "rows": m.n_rows,
                    "positives": m.n_positives,
                    "positive_rate": m.n_positives / m.n_rows,
                    "pr_auc": m.pr_auc if defined else np.nan,
                    "roc_auc": m.roc_auc if defined else np.nan,
                    "precision": m.precision,
                    "recall": m.recall,
                    "review_rate": m.review_rate,
                    "alerts_per_1000": m.alerts_per_1000,
                }
            )
    return pd.DataFrame(rows)


def feature_importance(model: Any) -> pd.DataFrame:
    """Model-native importance over the transformed features (XGBoost gain or LR |coef|)."""
    pipeline = unwrap_pipeline(model)
    estimator = pipeline.named_steps["model"]
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    if hasattr(estimator, "get_booster"):
        booster = estimator.get_booster()
        gains = booster.get_score(importance_type="gain")
        values = [float(gains.get(name, 0.0)) for name in names]
        method = "xgboost_gain"
    else:
        values = [float(abs(c)) for c in np.ravel(estimator.coef_)]
        method = "abs_coefficient"
    total = sum(values) or 1.0
    table = pd.DataFrame({"feature": names, "importance": values, "method": method})
    table["importance_share"] = table["importance"] / total
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance(
    model: Any, x: pd.DataFrame, y: pd.Series, *, seed: int, n_repeats: int = 3
) -> pd.DataFrame:
    """Permutation importance over the *contract input columns* that feed the feature pipeline.

    Computed by permuting raw contract columns, so the result answers "which inputs does the
    scorer rely on", which is the question the data contract owner cares about.
    """
    columns = [c for c in x.columns if c in _INPUT_COLUMNS_FOR_PERMUTATION]
    result = sk_permutation_importance(
        model, x[columns + [c for c in x.columns if c not in columns]], y.to_numpy(dtype="int64"),
        scoring="average_precision", n_repeats=n_repeats, random_state=seed, n_jobs=1,
    )  # fmt: skip
    table = pd.DataFrame(
        {
            "feature": list(x.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "method": "permutation_average_precision",
        }
    )
    table = table[table["feature"].isin(columns)]
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


_INPUT_COLUMNS_FOR_PERMUTATION: frozenset[str] = frozenset(
    {
        "amount", "currency", "transaction_type", "channel", "country", "beneficiary_country",
        "transaction_timestamp", "account_age_days", "transactions_1h", "transactions_24h",
        "amount_1h", "amount_24h", "unique_counterparties_24h", "days_since_last_transaction",
        "account_median_amount_30d", "account_mean_amount_30d", "inflow_amount_24h",
        "merchant_category", "ip_risk_band", "customer_segment", "is_new_beneficiary",
        "is_new_device",
    }
)  # fmt: skip


# --- plots ---------------------------------------------------------------------------------------


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_precision_recall(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, positive_rate: float
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, p) in curves.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    ax.axhline(
        positive_rate,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"no-skill ({positive_rate:.3%})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall (test split)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (test split; optimistic under imbalance)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion_matrix(metrics: MetricsSummary, path: Path, title: str) -> Path:
    matrix = np.asarray(
        [
            [metrics.true_negatives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_positives],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            f"{value:,}",
            ha="center",
            va="center",
            color="black" if value < matrix.max() / 2 else "white",
        )
    ax.set_xticks([0, 1], labels=["PASS", "REVIEW"])
    ax.set_yticks([0, 1], labels=["legit", "fraud (synthetic)"])
    ax.set_xlabel("Decision support")
    ax.set_ylabel("Synthetic label")
    ax.set_title(f"{title}\nthreshold={metrics.threshold:.3f}")
    return _save(fig, path)


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, path: Path, title: str) -> Path:
    table = calibration_table(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="perfect")
    ax.plot(table["mean_predicted"], table["fraction_positive"], marker="o", label="model")
    ax.set_xlabel("Mean predicted probability (quantile bins)")
    ax.set_ylabel("Observed fraud fraction")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_threshold_analysis(analysis: pd.DataFrame, threshold: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(analysis["threshold"], analysis["precision"], label="precision")
    ax.plot(analysis["threshold"], analysis["recall"], label="recall")
    ax.plot(analysis["threshold"], analysis["review_rate"], label="review rate")
    ax.axvline(
        threshold, color="black", linestyle=":", linewidth=1, label=f"selected={threshold:.3f}"
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Threshold analysis (validation split)")
    ax.legend(loc="center right")
    return _save(fig, path)
