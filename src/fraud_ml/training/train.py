"""End-to-end training: validate -> temporal split -> baseline + XGBoost -> calibrate ->
threshold policy -> compare -> reports -> versioned artifact.

Everything that *tunes* (early stopping, calibration, threshold) sees only the validation
split. The test split is scored exactly once per model, after every decision has been made.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fraud_ml.config import CostModel, ThresholdPolicy
from fraud_ml.contracts.score import MetricsSummary
from fraud_ml.contracts.transaction import LABEL_COLUMNS
from fraud_ml.data.split import TemporalSplit, temporal_split
from fraud_ml.data.validation import validate_frame
from fraud_ml.features.definitions import ALL_FEATURES, FEATURE_VERSION
from fraud_ml.logging import get_logger
from fraud_ml.models.baseline import BASELINE_NAME, BaselineParams, build_baseline
from fraud_ml.models.calibration import CalibrationDecision, calibration_table, choose_calibration
from fraud_ml.models.gradient_boosting import (
    GRADIENT_BOOSTING_NAME,
    GradientBoostingParams,
    best_iteration,
    build_gradient_boosting,
    fit_gradient_boosting,
    scale_pos_weight,
)
from fraud_ml.models.registry import (
    ModelMetadata,
    git_commit,
    library_versions,
    new_run_id,
    read_metadata,
    save_artifact,
)
from fraud_ml.training.evaluation import (
    compute_metrics,
    feature_importance,
    permutation_importance,
    plot_calibration,
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc,
    plot_threshold_analysis,
    segment_metrics,
)
from fraud_ml.training.experiment import NullTracker, Tracker
from fraud_ml.training.threshold import ThresholdDecision, select_threshold, threshold_analysis

log = get_logger(__name__)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str = "fraud-risk"
    model_version: str = "fraud-risk/1.0.0"
    seed: int = 42
    train_fraction: float = Field(default=0.70, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    baseline: BaselineParams = BaselineParams()
    gradient_boosting: GradientBoostingParams = GradientBoostingParams()
    calibration_methods: tuple[str, ...] = ("sigmoid", "isotonic")
    cost_model: CostModel = CostModel()
    threshold_policy: ThresholdPolicy = ThresholdPolicy()
    permutation_importance: bool = False
    promote_latest: bool = True


class CandidateResult(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    name: str
    calibration: CalibrationDecision
    threshold: ThresholdDecision
    validation_metrics: MetricsSummary
    test_metrics: MetricsSummary
    fit_seconds: float
    best_iteration: int | None = None


class TrainingResult(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    run_id: str
    model_version: str
    artifact_path: Path
    reports_path: Path
    winner: str
    selection_rationale: str
    candidates: dict[str, CandidateResult]
    split_summary: dict[str, Any]
    rows: int
    positives: int
    positive_rate: float
    data_version: str
    total_seconds: float
    mlflow_run_id: str | None = None

    @property
    def winning(self) -> CandidateResult:
        return self.candidates[self.winner]


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = frame.drop(columns=[c for c in LABEL_COLUMNS if c in frame.columns])
    y = frame["is_fraud"].astype(bool)
    return x, y


def _evaluate_candidate(
    name: str,
    model: Any,
    split: TemporalSplit,
    config: TrainingConfig,
    fit_seconds: float,
    iteration: int | None,
) -> tuple[Any, CandidateResult, pd.DataFrame, np.ndarray, np.ndarray]:
    x_val, y_val = _xy(split.validation)
    x_test, y_test = _xy(split.test)
    model, calibration = choose_calibration(model, x_val, y_val, tuple(config.calibration_methods))  # type: ignore[arg-type]
    p_val = model.predict_proba(x_val)[:, 1]
    analysis = threshold_analysis(y_val.to_numpy(), p_val, config.cost_model)
    decision = select_threshold(analysis, config.threshold_policy)
    p_test = model.predict_proba(x_test)[:, 1]
    result = CandidateResult(
        name=name,
        calibration=calibration,
        threshold=decision,
        validation_metrics=compute_metrics(
            y_val.to_numpy(), p_val, decision.threshold, config.cost_model
        ),
        test_metrics=compute_metrics(
            y_test.to_numpy(), p_test, decision.threshold, config.cost_model
        ),
        fit_seconds=fit_seconds,
        best_iteration=iteration,
    )
    return model, result, analysis, p_val, p_test


def select_winner(
    candidates: dict[str, CandidateResult], advanced: str, baseline: str
) -> tuple[str, str]:
    """Primary: validation PR-AUC. Operational override: a candidate that is cheaper *and* needs
    fewer reviews at its operating point wins even with slightly lower PR-AUC; a candidate that
    alone satisfies the threshold policy always wins."""
    a, b = candidates[advanced], candidates[baseline]
    if a.threshold.policy_satisfied != b.threshold.policy_satisfied:
        winner = advanced if a.threshold.policy_satisfied else baseline
        return (
            winner,
            f"{winner} is the only candidate satisfying the threshold policy on validation",
        )
    leader, other = (
        (advanced, baseline)
        if a.validation_metrics.pr_auc >= b.validation_metrics.pr_auc
        else (baseline, advanced)
    )
    lm, om = candidates[leader].validation_metrics, candidates[other].validation_metrics
    if om.expected_cost < lm.expected_cost and om.review_rate < lm.review_rate:
        return other, (
            f"{other} has lower validation PR-AUC ({om.pr_auc:.4f} vs {lm.pr_auc:.4f}) but "
            f"lower expected cost ({om.expected_cost:.0f} vs {lm.expected_cost:.0f}) and "
            "review rate at its operating point"
        )
    return leader, (
        f"{leader} has the higher validation PR-AUC ({lm.pr_auc:.4f} vs {om.pr_auc:.4f}); "
        f"expected cost {lm.expected_cost:.0f} vs {om.expected_cost:.0f}, review rate "
        f"{lm.review_rate:.2%} vs {om.review_rate:.2%}"
    )


def train_model(
    frame: pd.DataFrame,
    config: TrainingConfig,
    *,
    model_root: Path,
    reports_dir: Path,
    data_name: str,
    data_version: str,
    tracker: Tracker | None = None,
) -> TrainingResult:
    tracker = tracker or NullTracker()
    started = time.perf_counter()
    run_id = new_run_id()
    reports_dir.mkdir(parents=True, exist_ok=True)

    validate_frame(frame, require_labels=True, strict=True)
    split = temporal_split(
        frame, train_fraction=config.train_fraction, validation_fraction=config.validation_fraction
    )
    x_train, y_train = _xy(split.train)
    x_val, y_val = _xy(split.validation)
    x_test, y_test = _xy(split.test)
    log.info("training.split", **{s.name: s.rows for s in split.summaries})
    tracker.log_params(
        {
            "config": config.model_dump(mode="json"),
            "data_version": data_version,
            "feature_version": FEATURE_VERSION,
        }
    )

    # --- baseline ------------------------------------------------------------------------------
    t0 = time.perf_counter()
    baseline = build_baseline(config.baseline, seed=config.seed)
    baseline.fit(x_train, y_train.to_numpy(dtype="int64"))
    baseline_seconds = time.perf_counter() - t0
    baseline_model, baseline_result, baseline_analysis, _, baseline_p_test = _evaluate_candidate(
        BASELINE_NAME, baseline, split, config, baseline_seconds, None
    )
    log.info(
        "training.candidate",
        name=BASELINE_NAME,
        pr_auc=round(baseline_result.validation_metrics.pr_auc, 4),
        seconds=round(baseline_seconds, 2),
    )

    # --- gradient boosting ---------------------------------------------------------------------
    t0 = time.perf_counter()
    pos_weight = scale_pos_weight(y_train, config.gradient_boosting.scale_pos_weight_mode)
    boosted = build_gradient_boosting(
        config.gradient_boosting, seed=config.seed, pos_weight=pos_weight
    )
    fit_gradient_boosting(boosted, x_train, y_train, x_val, y_val)
    boosted_seconds = time.perf_counter() - t0
    boosted_model, boosted_result, boosted_analysis, _, boosted_p_test = _evaluate_candidate(
        GRADIENT_BOOSTING_NAME, boosted, split, config, boosted_seconds, best_iteration(boosted)
    )
    log.info(
        "training.candidate",
        name=GRADIENT_BOOSTING_NAME,
        pr_auc=round(boosted_result.validation_metrics.pr_auc, 4),
        seconds=round(boosted_seconds, 2),
    )

    candidates = {BASELINE_NAME: baseline_result, GRADIENT_BOOSTING_NAME: boosted_result}
    winner, rationale = select_winner(candidates, GRADIENT_BOOSTING_NAME, BASELINE_NAME)
    winning_model = boosted_model if winner == GRADIENT_BOOSTING_NAME else baseline_model
    winning = candidates[winner]
    winning_analysis = boosted_analysis if winner == GRADIENT_BOOSTING_NAME else baseline_analysis
    winning_p_test = boosted_p_test if winner == GRADIENT_BOOSTING_NAME else baseline_p_test
    log.info("training.selected", winner=winner, rationale=rationale)

    # --- reports -------------------------------------------------------------------------------
    y_test_arr = y_test.to_numpy()
    winning_analysis.to_csv(reports_dir / "threshold_analysis.csv", index=False)
    segment_metrics(
        x_test, y_test_arr, winning_p_test, winning.threshold.threshold, config.cost_model
    ).to_csv(reports_dir / "segment_metrics.csv", index=False)
    feature_importance(winning_model).to_csv(reports_dir / "feature_importance.csv", index=False)
    if config.permutation_importance:
        permutation_importance(winning_model, x_val, y_val, seed=config.seed).to_csv(
            reports_dir / "permutation_importance.csv", index=False
        )
    calibration_table(y_test_arr, winning_p_test).to_csv(
        reports_dir / "calibration.csv", index=False
    )
    curves = {
        BASELINE_NAME: (y_test_arr, baseline_p_test),
        GRADIENT_BOOSTING_NAME: (y_test_arr, boosted_p_test),
    }
    plot_precision_recall(curves, reports_dir / "precision_recall_curve.png", float(y_test.mean()))
    plot_roc(curves, reports_dir / "roc_curve.png")
    plot_confusion_matrix(
        winning.test_metrics, reports_dir / "confusion_matrix.png", f"{winner} (test split)"
    )
    plot_calibration(
        y_test_arr,
        winning_p_test,
        reports_dir / "calibration_curve.png",
        f"{winner} calibration (test split)",
    )
    plot_threshold_analysis(
        winning_analysis, winning.threshold.threshold, reports_dir / "threshold_analysis.png"
    )

    # --- artifact ------------------------------------------------------------------------------
    metadata = ModelMetadata(
        model_name=config.model_name,
        model_version=config.model_version,
        model_type=winner,
        run_id=run_id,
        training_timestamp=datetime.now(tz=UTC),
        training_data_version=data_version,
        training_data_name=data_name,
        training_rows=len(frame),
        git_commit=git_commit(),
        seed=config.seed,
        library_versions=library_versions(),
        training_config=config.model_dump(mode="json"),
        feature_names=list(ALL_FEATURES),
        policy=winning.threshold.to_policy(),
        calibration_method=winning.calibration.method,
        validation_metrics=winning.validation_metrics,
        test_metrics=winning.test_metrics,
        baseline_test_metrics=baseline_result.test_metrics if winner != BASELINE_NAME else None,
        selection_rationale=rationale,
        split_summary=split.to_dict(),
    )
    artifact_path = save_artifact(
        model_root, winning_model, metadata, promote=config.promote_latest
    )
    result = TrainingResult(
        run_id=run_id,
        model_version=config.model_version,
        artifact_path=artifact_path,
        reports_path=reports_dir,
        winner=winner,
        selection_rationale=rationale,
        candidates=candidates,
        split_summary=split.to_dict(),
        rows=len(frame),
        positives=int(y_train.sum() + y_val.sum() + y_test.sum()),
        positive_rate=float(frame["is_fraud"].mean()),
        data_version=data_version,
        total_seconds=time.perf_counter() - started,
        mlflow_run_id=tracker.run_id,
    )
    payload = result.model_dump(mode="json")
    payload["metadata"] = read_metadata(artifact_path).model_dump(mode="json")
    (reports_dir / "metrics.json").write_text(_json_dumps(payload), encoding="utf-8")

    for name, candidate in candidates.items():
        tracker.log_metrics(candidate.validation_metrics.model_dump(), prefix=f"{name}.validation.")
        tracker.log_metrics(candidate.test_metrics.model_dump(), prefix=f"{name}.test.")
    tracker.log_params(
        {
            "winner": winner,
            "threshold": winning.threshold.threshold,
            "calibration": winning.calibration.method,
            "run_id": run_id,
        }
    )
    for file in reports_dir.iterdir():
        tracker.log_artifact(file)
    tracker.log_artifact(artifact_path / "metadata.json")
    log.info(
        "training.complete", winner=winner, run_id=run_id, seconds=round(result.total_seconds, 1)
    )
    return result


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)
