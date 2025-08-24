from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_ml.config import CostModel, ThresholdPolicy
from fraud_ml.models.calibration import calibration_table, expected_calibration_error
from fraud_ml.training.evaluation import (
    compute_metrics,
    confusion_counts,
    plot_calibration,
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc,
    plot_threshold_analysis,
    segment_metrics,
)
from fraud_ml.training.threshold import (
    analysis_to_records,
    candidate_grid,
    select_threshold,
    threshold_analysis,
)

Y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
P = np.array([0.95, 0.80, 0.60, 0.20, 0.70, 0.30, 0.10, 0.05, 0.02, 0.01])
COSTS = CostModel(false_negative_cost=100, false_positive_cost=5, true_positive_cost=1)


def test_cost_model_expected_cost() -> None:
    assert COSTS.expected_cost(tp=2, fp=3, tn=10, fn=1) == 2 * 1 + 3 * 5 + 1 * 100
    assert CostModel(true_negative_cost=0.5).expected_cost(0, 0, 4, 0) == 2.0


def test_compute_metrics_known_values() -> None:
    m = compute_metrics(Y, P, threshold=0.5, cost_model=COSTS)
    assert (m.true_positives, m.false_positives, m.true_negatives, m.false_negatives) == (
        3,
        1,
        5,
        1,
    )
    assert m.precision == pytest.approx(0.75)
    assert m.recall == pytest.approx(0.75)
    assert m.f1 == pytest.approx(0.75)
    assert m.false_positive_rate == pytest.approx(1 / 6)
    assert m.review_rate == pytest.approx(0.4)
    assert m.alerts_per_1000 == pytest.approx(400.0)
    assert m.expected_cost == 3 * 1 + 1 * 5 + 1 * 100
    assert m.expected_cost_per_1000 == pytest.approx(108 * 100)
    assert 0 <= m.pr_auc <= 1 and 0 <= m.roc_auc <= 1 and 0 <= m.brier_score <= 1
    assert m.n_rows == 10 and m.n_positives == 4
    assert confusion_counts(Y, P, 0.99) == (0, 0, 6, 4)


def test_compute_metrics_single_class_does_not_crash() -> None:
    m = compute_metrics(np.zeros(5, dtype=int), np.linspace(0, 1, 5), 0.5, COSTS)
    assert m.pr_auc == 0.0 and m.recall == 0.0


def test_threshold_analysis_monotonic_review_count() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    analysis = threshold_analysis(y, p, COSTS)
    assert analysis["threshold"].is_monotonic_increasing
    flagged = analysis["true_positives"] + analysis["false_positives"]
    assert (np.diff(flagged.to_numpy()) <= 0).all()
    assert (analysis["recall"].diff().dropna() <= 1e-12).all()
    assert analysis["threshold"].between(0, 1).all()
    assert analysis.loc[analysis["threshold"] == 0.0, "recall"].iloc[0] == 1.0
    records = analysis_to_records(analysis)
    assert records and "expected_cost" in records[0]


def test_threshold_analysis_matches_compute_metrics() -> None:
    analysis = threshold_analysis(Y, P, COSTS)
    row = analysis[analysis["threshold"] == 0.5].iloc[0]
    m = compute_metrics(Y, P, 0.5, COSTS)
    assert row["true_positives"] == m.true_positives and row["false_positives"] == m.false_positives
    assert row["expected_cost"] == m.expected_cost


def test_candidate_grid_bounds() -> None:
    grid = candidate_grid(np.array([0.2, 0.9, 0.9, 0.4]))
    assert grid.min() >= 0 and grid.max() <= 1 and len(np.unique(grid)) == len(grid)


def test_select_threshold_satisfies_policy() -> None:
    analysis = threshold_analysis(Y, P, COSTS)
    decision = select_threshold(analysis, ThresholdPolicy(recall_target=0.75, max_review_rate=0.5))
    assert decision.policy_satisfied
    assert decision.recall >= 0.75 and decision.review_rate <= 0.5
    assert 0 <= decision.threshold <= 1
    assert decision.priority_threshold > decision.threshold
    assert decision.low_upper == pytest.approx(decision.threshold * 0.5)
    policy = decision.to_policy()
    assert policy.threshold == decision.threshold


def test_select_threshold_falls_back_when_infeasible() -> None:
    analysis = threshold_analysis(Y, P, COSTS)
    decision = select_threshold(analysis, ThresholdPolicy(recall_target=1.0, max_review_rate=0.2))
    assert not decision.policy_satisfied
    assert decision.review_rate <= 0.2
    assert "no threshold reaches" in decision.rationale


def test_select_threshold_degenerate_scores() -> None:
    analysis = threshold_analysis(np.array([1, 0, 0]), np.array([1.0, 1.0, 1.0]), COSTS)
    decision = select_threshold(analysis, ThresholdPolicy(recall_target=0.9, max_review_rate=0.01))
    assert decision.threshold == 1.0 and not decision.policy_satisfied
    with pytest.raises(ValueError):
        select_threshold(analysis.iloc[0:0], ThresholdPolicy())


def test_priority_threshold_uses_precision_target() -> None:
    analysis = threshold_analysis(Y, P, COSTS)
    decision = select_threshold(
        analysis,
        ThresholdPolicy(recall_target=0.5, max_review_rate=1.0, priority_precision_target=1.0),
    )
    above = analysis[(analysis["threshold"] > decision.threshold) & (analysis["precision"] >= 1.0)]
    assert decision.priority_threshold == pytest.approx(above["threshold"].min())


def test_segment_metrics(small_dataset: pd.DataFrame) -> None:
    x = small_dataset.drop(columns=["is_fraud", "fraud_type"])
    y = small_dataset["is_fraud"].to_numpy()
    p = np.where(y, 0.9, 0.1) + np.random.default_rng(1).random(len(y)) * 0.05
    table = segment_metrics(x, y, p, 0.5, COSTS)
    assert set(table["segment"]) == {"currency", "channel", "transaction_type", "cross_border"}
    assert (table["rows"] >= 50).all()
    assert table["recall"].between(0, 1).all()
    assert set(table.loc[table["segment"] == "cross_border", "value"]) <= {
        "DOMESTIC",
        "CROSS_BORDER",
    }
    no_positive = table[table["positives"] == 0]
    assert no_positive.empty or no_positive["pr_auc"].isna().all()
    assert table["pr_auc"].notna().any()


def test_calibration_table_and_ece() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    table = calibration_table(y, p, n_bins=10)
    assert len(table) == 10
    assert table["count"].sum() == 2000
    assert expected_calibration_error(table) < 0.08
    assert expected_calibration_error(pd.DataFrame()) == 0.0
    assert len(calibration_table(np.array([1, 0]), np.array([0.5, 0.5]))) == 1


def test_plots_write_png_files(tmp_path: Path) -> None:
    metrics = compute_metrics(Y, P, 0.5, COSTS)
    analysis = threshold_analysis(Y, P, COSTS)
    curves = {"a": (Y, P), "b": (Y, 1 - P)}
    paths = [
        plot_precision_recall(curves, tmp_path / "pr.png", 0.4),
        plot_roc(curves, tmp_path / "roc.png"),
        plot_confusion_matrix(metrics, tmp_path / "cm.png", "t"),
        plot_calibration(Y, P, tmp_path / "cal.png", "t"),
        plot_threshold_analysis(analysis, 0.5, tmp_path / "thr.png"),
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 1000
