"""Local MLflow experiment tracking (file backend, no server) with a no-op fallback."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# MLflow 3.x marks the file store as maintenance-only; it remains the right choice for a
# zero-infrastructure local demo (docs/ARCHITECTURE.md).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class Tracker(Protocol):
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def log_dict(self, payload: Mapping[str, Any], filename: str) -> None: ...
    @property
    def run_id(self) -> str | None: ...


class NullTracker:
    run_id: str | None = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        return None

    def log_artifact(self, path: Path) -> None:
        return None

    def log_dict(self, payload: Mapping[str, Any], filename: str) -> None:
        return None


class MlflowTracker:
    def __init__(self, run: Any) -> None:
        import mlflow

        self._mlflow = mlflow
        self._run = run

    @property
    def run_id(self) -> str | None:
        return str(self._run.info.run_id)

    def log_params(self, params: Mapping[str, Any]) -> None:
        flat = {k: _truncate(v) for k, v in flatten(params).items()}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        payload = {
            f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, int | float)
        }
        self._mlflow.log_metrics(payload)

    def log_artifact(self, path: Path) -> None:
        if path.exists():
            self._mlflow.log_artifact(str(path))

    def log_dict(self, payload: Mapping[str, Any], filename: str) -> None:
        self._mlflow.log_dict(dict(payload), filename)


def flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def _truncate(value: Any, limit: int = 500) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@contextmanager
def start_run(
    *, enabled: bool, tracking_uri: str, experiment: str, run_name: str
) -> Iterator[Tracker]:
    if not enabled:
        yield NullTracker()
        return
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        yield MlflowTracker(run)
