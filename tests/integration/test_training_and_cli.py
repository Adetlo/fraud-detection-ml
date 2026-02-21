from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fraud_ml.cli.main import app
from fraud_ml.data.loader import read_frame
from fraud_ml.models.registry import load_artifact
from fraud_ml.scoring.scorer import SCORE_COLUMNS
from fraud_ml.training.train import TrainingResult

runner = CliRunner()

REPORT_FILES = (
    "metrics.json", "threshold_analysis.csv", "segment_metrics.csv", "feature_importance.csv",
    "calibration.csv", "precision_recall_curve.png", "roc_curve.png", "confusion_matrix.png",
    "calibration_curve.png", "threshold_analysis.png",
)  # fmt: skip


def test_training_produces_artifact_and_reports(trained: TrainingResult) -> None:
    assert trained.winner in trained.candidates
    for name in REPORT_FILES:
        assert (trained.reports_path / name).exists(), name
    metrics = json.loads((trained.reports_path / "metrics.json").read_text())
    assert metrics["winner"] == trained.winner
    assert set(metrics["candidates"]) == {"logistic_regression", "xgboost"}
    assert metrics["metadata"]["artifact_sha256"]
    analysis = pd.read_csv(trained.reports_path / "threshold_analysis.csv")
    assert analysis["threshold"].between(0, 1).all()
    importance = pd.read_csv(trained.reports_path / "feature_importance.csv")
    assert importance["importance_share"].sum() == pytest.approx(1.0)
    segments = pd.read_csv(trained.reports_path / "segment_metrics.csv")
    assert {"currency", "channel", "transaction_type", "cross_border"} <= set(segments["segment"])
    for candidate in trained.candidates.values():
        assert 0 < candidate.validation_metrics.pr_auc <= 1
        assert (
            candidate.test_metrics.pr_auc
            > candidate.test_metrics.n_positives / candidate.test_metrics.n_rows
        )
        assert 0 <= candidate.threshold.threshold <= 1
    loaded = load_artifact(trained.artifact_path)
    assert loaded.metadata.model_type == trained.winner
    assert (
        loaded.metadata.split_summary["train"]["rows"]
        > loaded.metadata.split_summary["test"]["rows"]
    )


def test_training_is_reproducible(
    trained: TrainingResult, small_dataset: pd.DataFrame, fast_config, tmp_path: Path
) -> None:
    from fraud_ml.training.train import train_model

    again = train_model(
        small_dataset,
        fast_config,
        model_root=tmp_path / "m",
        reports_dir=tmp_path / "r",
        data_name="x",
        data_version="v",
    )
    for name in trained.candidates:
        a = trained.candidates[name].validation_metrics
        b = again.candidates[name].validation_metrics
        assert a.pr_auc == pytest.approx(b.pr_auc, abs=1e-9)
        assert trained.candidates[name].threshold.threshold == pytest.approx(
            again.candidates[name].threshold.threshold
        )
    assert again.winner == trained.winner


def test_training_rejects_invalid_data(
    small_dataset: pd.DataFrame, fast_config, tmp_path: Path
) -> None:
    from fraud_ml.exceptions import DataValidationError
    from fraud_ml.training.train import train_model

    bad = small_dataset.copy()
    bad.loc[bad.index[:5], "amount"] = -1.0
    with pytest.raises(DataValidationError):
        train_model(
            bad,
            fast_config,
            model_root=tmp_path,
            reports_dir=tmp_path,
            data_name="x",
            data_version="v",
        )


def test_mlflow_file_backend_logs_run(
    small_dataset: pd.DataFrame, fast_config, tmp_path: Path
) -> None:
    import mlflow

    from fraud_ml.training.experiment import start_run
    from fraud_ml.training.train import train_model

    uri = f"file:{(tmp_path / 'mlruns').as_posix()}"
    with start_run(enabled=True, tracking_uri=uri, experiment="test-exp", run_name="r1") as tracker:
        result = train_model(
            small_dataset.head(3000),
            fast_config,
            model_root=tmp_path / "m",
            reports_dir=tmp_path / "r",
            data_name="x",
            data_version="v",
            tracker=tracker,
        )
        tracker.log_dict({"k": "v"}, "extra.json")
        assert tracker.run_id == result.mlflow_run_id
    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(result.mlflow_run_id)
    assert run.data.params["winner"] == result.winner
    assert "xgboost.test.pr_auc" in run.data.metrics
    assert run.data.params["config.seed"] == str(fast_config.seed)


def test_cli_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_ML_MLFLOW_ENABLED", "false")
    data = tmp_path / "data"
    models = tmp_path / "models"
    reports = tmp_path / "reports"

    result = runner.invoke(
        app,
        [
            "generate-data",
            "-n",
            "4000",
            "--fraud-rate",
            "0.02",
            "--seed",
            "3",
            "--days",
            "90",
            "-o",
            str(data),
        ],
    )
    assert result.exit_code == 0, result.output
    dataset = data / "transactions.parquet"
    assert dataset.exists() and (data / "generation_summary.json").exists()

    result = runner.invoke(
        app,
        [
            "train",
            "-i",
            str(dataset),
            "-o",
            str(models),
            "--reports",
            str(reports),
            "--fast",
            "--no-mlflow",
            "--seed",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (models / "latest" / "metadata.json").exists()
    assert "selected threshold" in result.output

    result = runner.invoke(app, ["model-info", "-m", str(models / "latest")])
    assert result.exit_code == 0 and "fraud-risk/1.0.0" in result.output

    result = runner.invoke(
        app,
        ["evaluate", "-m", str(models / "latest"), "-i", str(dataset), "-o", str(reports / "eval")],
    )
    assert result.exit_code == 0, result.output
    assert (reports / "eval" / "metrics.json").exists()

    frame = read_frame(dataset).drop(columns=["is_fraud", "fraud_type"]).tail(300)
    frame.loc[frame.index[0], "amount"] = -9.0
    scoring = data / "scoring.jsonl"
    from fraud_ml.data.loader import write_frame

    write_frame(frame, scoring)
    result = runner.invoke(
        app,
        [
            "score-file",
            "-m",
            str(models / "latest"),
            "-i",
            str(scoring),
            "-o",
            str(reports / "scores.parquet"),
        ],
    )
    assert result.exit_code == 0, result.output
    scores = pd.read_parquet(reports / "scores.parquet")
    assert len(scores) == 299 and list(scores.columns) == list(SCORE_COLUMNS)
    assert (reports / "scores_rejects.parquet").exists()

    single = tmp_path / "one.json"
    row = frame.iloc[1].to_dict()
    row["transaction_timestamp"] = row["transaction_timestamp"].isoformat()
    single.write_text(
        json.dumps(
            {k: v for k, v in row.items() if not (isinstance(v, float) and v != v)}, default=str
        )
    )
    result = runner.invoke(app, ["score", "-m", str(models / "latest"), "-i", str(single)])
    assert result.exit_code == 0, result.output
    assert "risk_band" in result.output

    result = runner.invoke(
        app,
        [
            "drift-check",
            "-r",
            str(dataset),
            "-c",
            str(scoring),
            "-m",
            str(models / "latest"),
            "-o",
            str(reports / "drift.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    drift = json.loads((reports / "drift.json").read_text())
    assert drift["status"] in {"OK", "WARNING", "DRIFT"} and drift["score_drift"] is not None

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_cli_error_paths(tmp_path: Path) -> None:
    result = runner.invoke(app, ["model-info", "-m", str(tmp_path / "missing")])
    assert result.exit_code == 1 and "ARTIFACT_INTEGRITY" in result.output
    result = runner.invoke(
        app, ["score-file", "-m", str(tmp_path / "missing"), "-i", str(tmp_path / "x.parquet")]
    )
    assert result.exit_code == 1
    bad = tmp_path / "bad.json"
    bad.write_text('{"transaction_id": "x"}')
    result = runner.invoke(app, ["score", "-m", str(tmp_path / "missing"), "-i", str(bad)])
    assert result.exit_code == 1
    result = runner.invoke(app, ["--log-format", "json", "--log-level", "ERROR", "version"])
    assert result.exit_code == 0


def test_cli_demo_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_ML_MLFLOW_ENABLED", "false")
    result = runner.invoke(
        app,
        [
            "demo",
            "-n",
            "3000",
            "--fraud-rate",
            "0.02",
            "--seed",
            "4",
            "--fast",
            "--no-mlflow",
            "--workdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "demo complete" in result.output
    assert (tmp_path / "models" / "latest" / "metadata.json").exists()
    assert (tmp_path / "reports" / "drift_report.json").exists()
    assert (tmp_path / "reports" / "scores.parquet").exists()
