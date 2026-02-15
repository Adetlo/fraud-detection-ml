from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_ml.config import DriftThresholds, Settings
from fraud_ml.contracts.enums import ReasonCode
from fraud_ml.contracts.score import RiskPolicy
from fraud_ml.contracts.transaction import TransactionFeatures
from fraud_ml.data.loader import normalise_frame, write_frame
from fraud_ml.exceptions import SchemaVersionMismatchError, ScoringError
from fraud_ml.monitoring.data_quality import data_quality_report
from fraud_ml.monitoring.drift import categorical_psi, drift_check, numeric_psi
from fraud_ml.monitoring.score_distribution import summarise_scores
from fraud_ml.scoring.batch import score_file, score_frame_batch
from fraud_ml.scoring.explanation import MAX_REASON_CODES, PRIORITY, indicator_matrix, reason_codes
from fraud_ml.scoring.scorer import SCORE_COLUMNS, Scorer
from fraud_ml.training.experiment import NullTracker, flatten, start_run
from tests.conftest import transaction_payload


def frame_of(*rows: dict) -> pd.DataFrame:
    return normalise_frame(pd.DataFrame(list(rows)))


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"amount": "500", "account_median_amount_30d": "100"}, ReasonCode.HIGH_AMOUNT_VS_HISTORY),
        (
            {"transactions_1h": 3, "transactions_24h": 3, "unique_counterparties_24h": 1},
            ReasonCode.HIGH_VELOCITY_1H,
        ),
        ({"transactions_24h": 12, "unique_counterparties_24h": 2}, ReasonCode.HIGH_VELOCITY_24H),
        ({"is_new_beneficiary": True}, ReasonCode.NEW_BENEFICIARY),
        ({"beneficiary_country": "NG"}, ReasonCode.CROSS_BORDER),
        ({"transaction_timestamp": "2025-06-30T02:15:00Z"}, ReasonCode.UNUSUAL_HOUR),
        ({"days_since_last_transaction": 60.0}, ReasonCode.DORMANT_ACCOUNT_REACTIVATED),
        (
            {"transactions_24h": 6, "unique_counterparties_24h": 5},
            ReasonCode.MANY_COUNTERPARTIES_24H,
        ),
        ({"amount": "930.00"}, ReasonCode.NEAR_THRESHOLD_AMOUNT),
        ({"account_age_days": 3}, ReasonCode.NEW_ACCOUNT),
        ({"is_new_device": True}, ReasonCode.NEW_DEVICE),
        ({"ip_risk_band": "HIGH"}, ReasonCode.HIGH_RISK_IP),
        ({"inflow_amount_24h": "200", "amount": "150"}, ReasonCode.RAPID_OUTFLOW_AFTER_INFLOW),
    ],
)
def test_each_reason_code_triggers(overrides: dict, code: ReasonCode) -> None:
    baseline = reason_codes(frame_of(transaction_payload()))[0]
    assert code not in baseline
    assert code in reason_codes(frame_of(transaction_payload(**overrides)))[0]


def test_reason_codes_are_ordered_and_limited() -> None:
    everything = transaction_payload(
        amount="930.00", account_median_amount_30d="100", transactions_1h=4, transactions_24h=12,
        unique_counterparties_24h=6, is_new_beneficiary=True, beneficiary_country="NG",
        transaction_timestamp="2025-06-30T02:15:00Z", days_since_last_transaction=60.0,
        account_age_days=3, is_new_device=True, ip_risk_band="HIGH", inflow_amount_24h="900",
    )  # fmt: skip
    matrix = indicator_matrix(frame_of(everything))
    assert matrix.iloc[0].all()
    codes = reason_codes(frame_of(everything))[0]
    assert len(codes) == MAX_REASON_CODES
    assert codes == list(PRIORITY[:MAX_REASON_CODES])


def test_scorer_outputs_contract(scorer: Scorer, payload: dict) -> None:
    response = scorer.score_one(TransactionFeatures.model_validate(payload))
    assert response.transaction_id == payload["transaction_id"]
    assert 0 <= response.risk_probability <= 1
    assert response.model_version == scorer.model_version
    assert response.threshold == scorer.policy.threshold
    assert response.risk_band == scorer.policy.band(response.risk_probability)
    assert response.decision_support == RiskPolicy.decision(response.risk_band)
    assert response.scored_at.tzinfo is not None


def test_scorer_is_deterministic(scorer: Scorer, unlabelled: pd.DataFrame) -> None:
    a = scorer.probabilities(unlabelled)
    b = scorer.probabilities(unlabelled.copy())
    np.testing.assert_array_equal(a, b)
    assert scorer.probabilities(unlabelled.iloc[0:0]).size == 0


def test_scorer_rejects_unknown_schema_version(scorer: Scorer, unlabelled: pd.DataFrame) -> None:
    bad = unlabelled.assign(schema_version="fraud-transaction/2")
    with pytest.raises(SchemaVersionMismatchError):
        scorer.probabilities(bad)


def test_scorer_wraps_model_failures(scorer: Scorer, unlabelled: pd.DataFrame) -> None:
    broken = unlabelled.drop(columns=["amount"])
    with pytest.raises(ScoringError):
        scorer.probabilities(broken)


def test_score_frame_columns(scorer: Scorer, unlabelled: pd.DataFrame) -> None:
    scored = scorer.score_frame(unlabelled.head(20))
    assert list(scored.columns) == list(SCORE_COLUMNS)
    assert scored.index.equals(unlabelled.head(20).index)
    assert set(scored["decision_support"]) <= {"PASS", "REVIEW", "PRIORITY_REVIEW"}


def test_batch_scoring_scores_only_valid_rows(
    scorer: Scorer, unlabelled: pd.DataFrame, tmp_path: Path
) -> None:
    frame = unlabelled.head(50).copy()
    frame.loc[3, "amount"] = -1.0
    frame.loc[7, "channel"] = "PIGEON"
    scores, rejects = score_frame_batch(scorer, frame)
    assert len(scores) == 48 and len(rejects) == 2
    assert set(rejects["validation_errors"]) == {"INVALID_AMOUNT", "UNSUPPORTED_CATEGORY:channel"}
    path = write_frame(frame, tmp_path / "in.parquet")
    summary = score_file(scorer, path, tmp_path / "out.parquet")
    assert summary.scored_rows == 48 and summary.rejected_rows == 2
    assert summary.rejects_path is not None and summary.rejects_path.exists()
    out = pd.read_parquet(summary.output_path)
    assert list(out.columns) == list(SCORE_COLUMNS)
    assert out["scored_at"].dt.tz is not None
    assert sum(summary.decision_counts.values()) == 48
    assert summary.to_dict()["rows_per_second"] is not None


def test_batch_scoring_all_invalid(scorer: Scorer, unlabelled: pd.DataFrame) -> None:
    frame = unlabelled.head(3).assign(amount=-1.0)
    scores, rejects = score_frame_batch(scorer, frame)
    assert scores.empty and len(rejects) == 3 and list(scores.columns) == list(SCORE_COLUMNS)


def test_numeric_and_categorical_psi() -> None:
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    same, _ = numeric_psi(ref, pd.Series(rng.normal(0, 1, 5000)), bins=10)
    shifted, detail = numeric_psi(ref, pd.Series(rng.normal(2, 1, 5000)), bins=10)
    assert same < 0.05 < 0.25 < shifted
    assert detail["current_mean"] > detail["reference_mean"]
    assert numeric_psi(pd.Series([], dtype=float), ref, 10)[0] == 0.0
    assert numeric_psi(pd.Series([1.0] * 10), pd.Series([1.0] * 10), 10)[0] == 0.0
    cats_ref = pd.Series(["A"] * 800 + ["B"] * 200)
    psi, detail = categorical_psi(cats_ref, pd.Series(["A"] * 200 + ["B"] * 700 + ["C"] * 100))
    assert psi > 0.25 and detail["new_categories"] == ["C"]
    assert categorical_psi(cats_ref, cats_ref)[0] == pytest.approx(0.0)


def test_drift_report_statuses(small_dataset: pd.DataFrame) -> None:
    ref = small_dataset.head(3000)
    same = drift_check(ref, small_dataset.tail(2500))
    assert same.features_checked > 10
    assert same.status.value in {"OK", "WARNING"}
    current = small_dataset.tail(2500).copy()
    for column in (
        "amount",
        "transactions_24h",
        "amount_24h",
        "account_age_days",
        "amount_1h",
        "transactions_1h",
    ):
        current[column] = current[column] * 50 + 1000
    current["channel"] = "API"
    current["risk_probability"] = 0.9
    ref = ref.assign(risk_probability=0.01)
    drifted = drift_check(
        ref,
        current,
        score_column="risk_probability",
        thresholds=DriftThresholds(drifted_feature_fraction_for_drift=0.2),
    )
    assert drifted.status.value == "DRIFT"
    assert {"amount", "channel"} <= set(drifted.drifted_features)
    assert drifted.score_drift is not None and drifted.score_drift.status.value == "DRIFT"
    summary = drifted.summary()
    assert summary["features_drifted"] == len(drifted.drifted_features)
    json.dumps(summary)
    assert drift_check(ref, ref).status.value == "OK"


def test_drift_thresholds_validation() -> None:
    with pytest.raises(ValueError):
        DriftThresholds(psi_warning=0.3, psi_drift=0.2)


def test_score_distribution_summary() -> None:
    policy = RiskPolicy(threshold=0.5, priority_threshold=0.8, low_upper=0.2)
    dist = summarise_scores(np.array([0.1, 0.3, 0.6, 0.9]), policy)
    assert dist.rows == 4 and dist.review_rate == 0.5 and dist.alerts_per_1000 == 500.0
    assert dist.band_rates["LOW"] == 0.25 and sum(dist.histogram.values()) == 4
    assert summarise_scores(np.array([]), policy).rows == 0
    assert "p99" in dist.quantiles


def test_data_quality_report(unlabelled: pd.DataFrame) -> None:
    frame = unlabelled.head(20).copy()
    frame.loc[0, "amount"] = -5.0
    report = data_quality_report(frame)
    assert report.rows == 20 and report.invalid_rows == 1 and report.invalid_rate == 0.05
    assert report.error_counts == {"INVALID_AMOUNT": 1}
    assert "amount" in report.numeric_ranges and "transaction_id" in report.missing_rates
    json.dumps(report.to_dict())


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_ML_FALSE_NEGATIVE_COST", "250")
    monkeypatch.setenv("FRAUD_ML_RECALL_TARGET", "0.9")
    monkeypatch.setenv("FRAUD_ML_LOG_LEVEL", "debug")
    settings = Settings()
    assert settings.cost_model.false_negative_cost == 250
    assert settings.threshold_policy.recall_target == 0.9
    assert settings.log_level == "DEBUG"


def test_tracker_helpers(tmp_path: Path) -> None:
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
    tracker = NullTracker()
    tracker.log_params({"x": 1})
    tracker.log_metrics({"m": 1.0})
    tracker.log_artifact(tmp_path)
    tracker.log_dict({"k": "v"}, "f.json")
    with start_run(enabled=False, tracking_uri="", experiment="e", run_name="r") as t:
        assert t.run_id is None
