"""Full flow: generate -> temporal split -> train -> evaluate -> threshold -> save -> load ->
API score -> batch -> drift report. Lightweight enough for CI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from fraud_ml.config import Settings
from fraud_ml.contracts.transaction import LABEL_COLUMNS
from fraud_ml.data.loader import read_frame, sha256_file, write_frame
from fraud_ml.data.split import temporal_split
from fraud_ml.data.synthetic import GeneratorConfig, generate_dataset
from fraud_ml.main import create_app
from fraud_ml.models.gradient_boosting import GradientBoostingParams
from fraud_ml.monitoring.drift import drift_check
from fraud_ml.scoring.batch import score_file
from fraud_ml.scoring.scorer import Scorer
from fraud_ml.training.train import TrainingConfig, train_model


def test_full_lifecycle(tmp_path: Path) -> None:
    frame, summary = generate_dataset(
        GeneratorConfig(transactions=5000, fraud_rate=0.02, seed=21, days=100)
    )
    assert summary.fraud_rows > 50
    dataset = write_frame(frame, tmp_path / "data" / "transactions.parquet")

    split = temporal_split(frame)
    assert split.train["transaction_timestamp"].max() <= split.test["transaction_timestamp"].min()

    config = TrainingConfig(
        seed=21,
        gradient_boosting=GradientBoostingParams(
            n_estimators=60, early_stopping_rounds=10, n_jobs=1
        ),
    )
    result = train_model(
        read_frame(dataset),
        config,
        model_root=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        data_name=dataset.name,
        data_version=sha256_file(dataset),
    )
    winning = result.winning
    assert winning.validation_metrics.pr_auc > 0.2
    assert winning.threshold.threshold == winning.validation_metrics.threshold
    assert (tmp_path / "reports" / "precision_recall_curve.png").exists()

    scorer = Scorer.from_path(tmp_path / "models" / "latest")
    assert scorer.metadata.training_data_version == sha256_file(dataset)
    assert scorer.policy.threshold == winning.threshold.threshold

    settings = Settings(
        model_dir=tmp_path / "models", mlflow_enabled=False, log_format="json", log_level="WARNING"
    )
    test_rows = split.test.drop(columns=list(LABEL_COLUMNS))
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/ready").status_code == 200
        row = test_rows.iloc[0].to_dict()
        row["transaction_timestamp"] = row["transaction_timestamp"].isoformat()
        payload = {k: v for k, v in row.items() if not (isinstance(v, float) and v != v)}
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert (
            body["model_version"] == scorer.model_version
            and body["threshold"] == scorer.policy.threshold
        )
        api_probability = body["risk_probability"]
    local = scorer.probabilities(test_rows.head(1))[0]
    assert abs(api_probability - local) < 1e-6

    scoring = write_frame(test_rows, tmp_path / "data" / "scoring.parquet")
    batch = score_file(scorer, scoring, tmp_path / "reports" / "scores.parquet")
    assert batch.scored_rows == len(test_rows) and batch.rejected_rows == 0
    scores = pd.read_parquet(batch.output_path)
    review_rate = (scores["decision_support"] != "PASS").mean()
    assert 0 < review_rate < 0.25

    reference = split.train.drop(columns=list(LABEL_COLUMNS)).assign(
        risk_probability=scorer.probabilities
    )
    current = test_rows.assign(risk_probability=scores["risk_probability"].to_numpy())
    report = drift_check(reference, current, score_column="risk_probability")
    assert report.features_checked >= 15
    payload = json.loads(report.model_dump_json())
    assert payload["status"] in {"OK", "WARNING", "DRIFT"}
    assert payload["score_drift"]["feature"] == "risk_probability"
