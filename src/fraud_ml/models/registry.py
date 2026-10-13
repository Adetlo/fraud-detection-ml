"""Versioned model artifacts: metadata contract, save/load with compatibility checks.

An artifact directory contains ``model.joblib`` (feature pipeline + preprocessing + model, plus
any calibration wrapper) and ``metadata.json``. Loading fails safely when:

* the metadata cannot be parsed,
* the artifact's input schema or feature version differs from the running code,
* the joblib hash does not match the recorded one.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 - used only to read the git commit with a fixed argv
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fraud_ml.contracts.enums import SCHEMA_VERSION, SCORE_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS
from fraud_ml.contracts.score import MetricsSummary, ModelInfo, RiskPolicy
from fraud_ml.exceptions import ArtifactIntegrityError, ModelVersionMismatchError
from fraud_ml.features.definitions import FEATURE_VERSION
from fraud_ml.models.serialization import METADATA_FILENAME, MODEL_FILENAME, dump_model, load_model

LATEST = "latest"
TRACKED_LIBRARIES = ("numpy", "pandas", "scikit-learn", "xgboost", "pydantic", "joblib")


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover
            versions[name] = "unknown"
    return versions


def git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, no shell
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str
    model_version: str
    model_type: str
    run_id: str
    schema_version: str = SCHEMA_VERSION
    score_schema_version: str = SCORE_SCHEMA_VERSION
    feature_version: str = FEATURE_VERSION
    training_timestamp: datetime
    training_data_version: str = Field(description="SHA-256 of the training source")
    training_data_name: str
    training_rows: int
    git_commit: str | None
    seed: int
    library_versions: dict[str, str]
    training_config: dict[str, Any]
    feature_names: list[str]
    policy: RiskPolicy
    calibration_method: str | None
    validation_metrics: MetricsSummary
    test_metrics: MetricsSummary
    baseline_test_metrics: MetricsSummary | None = None
    selection_rationale: str
    split_summary: dict[str, Any]
    artifact_sha256: str = ""

    def info(self) -> ModelInfo:
        return ModelInfo(
            model_version=self.model_version,
            model_name=self.model_name,
            schema_version=self.schema_version,
            score_schema_version=self.score_schema_version,
            feature_version=self.feature_version,
            training_timestamp=self.training_timestamp,
            training_data_version=self.training_data_version,
            git_commit=self.git_commit,
            policy=self.policy,
            metrics=self.test_metrics,
            feature_count=len(self.feature_names),
        )


class LoadedModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, protected_namespaces=())

    model: Any
    metadata: ModelMetadata
    path: Path


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(tz=UTC)
    return now.strftime("%Y%m%dT%H%M%SZ")


def save_artifact(root: Path, model: Any, metadata: ModelMetadata, *, promote: bool = True) -> Path:
    """Write ``<root>/<run_id>/`` and optionally refresh ``<root>/latest/`` as a copy."""
    target = root / metadata.run_id
    target.mkdir(parents=True, exist_ok=True)
    digest = dump_model(model, target / MODEL_FILENAME)
    stamped = metadata.model_copy(update={"artifact_sha256": digest})
    (target / METADATA_FILENAME).write_text(stamped.model_dump_json(indent=2), encoding="utf-8")
    if promote:
        latest = root / LATEST
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        elif latest.is_dir():
            shutil.rmtree(latest)
        shutil.copytree(target, latest)
    return target


def read_metadata(path: Path) -> ModelMetadata:
    metadata_file = path / METADATA_FILENAME
    if not metadata_file.is_file():
        raise ArtifactIntegrityError(f"no {METADATA_FILENAME} in artifact directory")
    try:
        return ModelMetadata.model_validate(json.loads(metadata_file.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise ArtifactIntegrityError(f"invalid {METADATA_FILENAME}: {exc}") from exc


def check_compatibility(metadata: ModelMetadata) -> None:
    if metadata.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ModelVersionMismatchError(
            f"artifact expects input schema {metadata.schema_version!r}; this build supports "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if metadata.feature_version != FEATURE_VERSION:
        raise ModelVersionMismatchError(
            f"artifact feature version {metadata.feature_version!r} != code {FEATURE_VERSION!r}"
        )
    if metadata.score_schema_version != SCORE_SCHEMA_VERSION:
        raise ModelVersionMismatchError(
            f"artifact score schema {metadata.score_schema_version!r} != code "
            f"{SCORE_SCHEMA_VERSION!r}"
        )


def load_artifact(path: Path) -> LoadedModel:
    """Load a trusted local artifact, failing safely on any mismatch."""
    if not path.is_dir():
        raise ArtifactIntegrityError(f"artifact directory not found: {path.name}")
    metadata = read_metadata(path)
    check_compatibility(metadata)
    if not metadata.artifact_sha256:
        raise ArtifactIntegrityError("metadata has no artifact_sha256; refusing to load")
    model = load_model(path / MODEL_FILENAME, metadata.artifact_sha256)
    if not hasattr(model, "predict_proba"):
        raise ArtifactIntegrityError("artifact does not expose predict_proba")
    return LoadedModel(model=model, metadata=metadata, path=path)


def list_artifacts(root: Path) -> list[ModelMetadata]:
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if child.name == LATEST or not (child / METADATA_FILENAME).is_file():
            continue
        try:
            found.append(read_metadata(child))
        except ArtifactIntegrityError:
            continue
    return found
