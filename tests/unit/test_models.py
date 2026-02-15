from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_ml.contracts.transaction import LABEL_COLUMNS
from fraud_ml.data.split import temporal_split
from fraud_ml.exceptions import ArtifactIntegrityError, ModelVersionMismatchError
from fraud_ml.models.baseline import BaselineParams, build_baseline
from fraud_ml.models.calibration import (
    MIN_POSITIVES_FOR_ISOTONIC,
    choose_calibration,
    fit_calibrator,
    unwrap_pipeline,
)
from fraud_ml.models.gradient_boosting import (
    GradientBoostingParams,
    best_iteration,
    build_gradient_boosting,
    fit_gradient_boosting,
    scale_pos_weight,
)
from fraud_ml.models.registry import (
    LoadedModel,
    ModelMetadata,
    check_compatibility,
    git_commit,
    library_versions,
    list_artifacts,
    load_artifact,
    read_metadata,
    save_artifact,
)
from fraud_ml.models.serialization import METADATA_FILENAME, MODEL_FILENAME, load_model, sha256_path
from fraud_ml.training.evaluation import feature_importance, permutation_importance
from fraud_ml.training.train import TrainingResult


def test_scale_pos_weight_modes() -> None:
    y = pd.Series([1] * 10 + [0] * 990)
    assert scale_pos_weight(y, "balanced") == pytest.approx(99.0)
    assert scale_pos_weight(y, "sqrt") == pytest.approx(np.sqrt(99.0))
    assert scale_pos_weight(y, "none") == 1.0
    assert scale_pos_weight(pd.Series([0, 0]), "balanced") == 1.0


def test_baseline_and_boosting_fit_and_predict(small_dataset: pd.DataFrame) -> None:
    split = temporal_split(small_dataset)
    x_tr, y_tr = split.train.drop(columns=list(LABEL_COLUMNS)), split.train["is_fraud"].astype(bool)
    x_va, y_va = (
        split.validation.drop(columns=list(LABEL_COLUMNS)),
        split.validation["is_fraud"].astype(bool),
    )
    baseline = build_baseline(BaselineParams(max_iter=300), seed=1).fit(x_tr, y_tr.astype(int))
    boosted = build_gradient_boosting(
        GradientBoostingParams(n_estimators=30, early_stopping_rounds=5, n_jobs=1),
        seed=1,
        pos_weight=5.0,
    )
    fit_gradient_boosting(boosted, x_tr, y_tr, x_va, y_va)
    for model in (baseline, boosted):
        p = model.predict_proba(x_va)[:, 1]
        assert p.shape == (len(x_va),) and ((p >= 0) & (p <= 1)).all()
    assert best_iteration(boosted) is not None and best_iteration(boosted) < 30
    assert best_iteration(baseline) is None
    importance = feature_importance(boosted)
    assert importance["importance_share"].sum() == pytest.approx(1.0)
    assert importance["method"].iloc[0] == "xgboost_gain"
    assert feature_importance(baseline)["method"].iloc[0] == "abs_coefficient"


def test_calibration_decision_and_unwrap(small_dataset: pd.DataFrame) -> None:
    split = temporal_split(small_dataset)
    x_tr, y_tr = split.train.drop(columns=list(LABEL_COLUMNS)), split.train["is_fraud"].astype(int)
    x_va, y_va = (
        split.validation.drop(columns=list(LABEL_COLUMNS)),
        split.validation["is_fraud"].astype(bool),
    )
    model = build_baseline(BaselineParams(max_iter=300), seed=1).fit(x_tr, y_tr)
    chosen, decision = choose_calibration(model, x_va, y_va)
    assert decision.method in (None, "sigmoid", "isotonic")
    assert int(y_va.sum()) < MIN_POSITIVES_FOR_ISOTONIC
    assert "isotonic" not in decision.brier_by_method
    assert "isotonic skipped" in decision.rationale
    assert unwrap_pipeline(chosen) is model
    calibrated = fit_calibrator(model, x_va, y_va, "isotonic")
    assert unwrap_pipeline(calibrated) is model
    with pytest.raises(TypeError):
        unwrap_pipeline(object())


def test_permutation_importance_runs(trained: TrainingResult, small_dataset: pd.DataFrame) -> None:
    model = load_artifact(trained.artifact_path).model
    x = small_dataset.drop(columns=list(LABEL_COLUMNS)).tail(1500)
    y = small_dataset["is_fraud"].tail(1500)
    assert y.sum() > 0
    table = permutation_importance(model, x, y, seed=0, n_repeats=1)
    assert "amount" in set(table["feature"])
    assert "transaction_id" not in set(table["feature"])


def test_artifact_round_trip_and_latest(trained: TrainingResult) -> None:
    loaded = load_artifact(trained.artifact_path)
    assert isinstance(loaded, LoadedModel)
    metadata = loaded.metadata
    assert metadata.model_version == "fraud-risk/1.0.0"
    assert metadata.artifact_sha256 == sha256_path(trained.artifact_path / MODEL_FILENAME)
    assert (
        metadata.feature_version == "features/1"
        and metadata.schema_version == "fraud-transaction/1"
    )
    assert metadata.training_data_version and metadata.seed == 11
    assert set(metadata.library_versions) >= {"numpy", "scikit-learn", "xgboost"}
    assert 0 <= metadata.policy.threshold <= 1
    assert metadata.test_metrics.n_rows > 0
    latest = load_artifact(trained.artifact_path.parent / "latest")
    assert latest.metadata == metadata
    info = metadata.info()
    assert (
        "path" not in info.model_dump() and str(trained.artifact_path) not in info.model_dump_json()
    )
    assert [m.run_id for m in list_artifacts(trained.artifact_path.parent)] == [metadata.run_id]
    assert list_artifacts(Path("/nonexistent")) == []


def test_tampered_artifact_is_refused(trained: TrainingResult, tmp_path: Path) -> None:
    import shutil

    target = tmp_path / "tampered"
    shutil.copytree(trained.artifact_path, target)
    with (target / MODEL_FILENAME).open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        load_artifact(target)
    with pytest.raises(ArtifactIntegrityError):
        load_model(target / "missing.joblib", "abc")


def test_version_mismatch_is_refused(trained: TrainingResult, tmp_path: Path) -> None:
    import shutil

    target = tmp_path / "old"
    shutil.copytree(trained.artifact_path, target)
    metadata = json.loads((target / METADATA_FILENAME).read_text())
    metadata["feature_version"] = "features/0"
    (target / METADATA_FILENAME).write_text(json.dumps(metadata))
    with pytest.raises(ModelVersionMismatchError, match="feature version"):
        load_artifact(target)
    metadata["feature_version"] = "features/1"
    metadata["schema_version"] = "fraud-transaction/0"
    with pytest.raises(ModelVersionMismatchError, match="input schema"):
        check_compatibility(ModelMetadata.model_validate(metadata))
    metadata["schema_version"] = "fraud-transaction/1"
    metadata["score_schema_version"] = "fraud-score/0"
    with pytest.raises(ModelVersionMismatchError, match="score schema"):
        check_compatibility(ModelMetadata.model_validate(metadata))


def test_missing_or_invalid_metadata(tmp_path: Path, trained: TrainingResult) -> None:
    with pytest.raises(ArtifactIntegrityError, match="not found"):
        load_artifact(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArtifactIntegrityError, match="no metadata"):
        read_metadata(empty)
    (empty / METADATA_FILENAME).write_text('{"model_name": 1}')
    with pytest.raises(ArtifactIntegrityError, match="invalid"):
        read_metadata(empty)
    metadata = read_metadata(trained.artifact_path).model_copy(update={"artifact_sha256": ""})
    (empty / METADATA_FILENAME).write_text(metadata.model_dump_json())
    with pytest.raises(ArtifactIntegrityError, match="artifact_sha256"):
        load_artifact(empty)


def test_save_artifact_without_promote(tmp_path: Path, trained: TrainingResult) -> None:
    loaded = load_artifact(trained.artifact_path)
    metadata = loaded.metadata.model_copy(update={"run_id": "manual"})
    path = save_artifact(tmp_path, loaded.model, metadata, promote=False)
    assert path.name == "manual" and not (tmp_path / "latest").exists()
    path = save_artifact(tmp_path, loaded.model, metadata, promote=True)
    assert (tmp_path / "latest" / MODEL_FILENAME).exists()
    save_artifact(tmp_path, loaded.model, metadata, promote=True)  # replaces existing latest copy


def test_git_commit_and_library_versions() -> None:
    commit = git_commit()
    assert commit is None or len(commit) == 40
    assert "pandas" in library_versions()
