"""Scorer: a loaded artifact plus its risk policy, producing contract score responses."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_ml.contracts.enums import SUPPORTED_SCHEMA_VERSIONS
from fraud_ml.contracts.score import ModelInfo, ScoreResponse
from fraud_ml.contracts.transaction import TransactionFeatures
from fraud_ml.data.loader import normalise_frame
from fraud_ml.exceptions import SchemaVersionMismatchError, ScoringError
from fraud_ml.models.registry import LoadedModel, load_artifact
from fraud_ml.scoring.explanation import reason_codes

SCORE_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "risk_probability",
    "risk_band",
    "decision_support",
    "reason_codes",
    "model_version",
    "scored_at",
)


class Scorer:
    def __init__(self, loaded: LoadedModel) -> None:
        self._loaded = loaded
        self.metadata = loaded.metadata
        self.policy = loaded.metadata.policy

    @classmethod
    def from_path(cls, path: Path) -> Scorer:
        return cls(load_artifact(path))

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    def info(self) -> ModelInfo:
        return self.metadata.info()

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Risk probabilities for a validated contract frame."""
        if frame.empty:
            return np.asarray([], dtype="float64")
        if "schema_version" in frame.columns:
            versions = set(frame["schema_version"].dropna().astype(str).unique())
            unsupported = versions - set(SUPPORTED_SCHEMA_VERSIONS)
            if unsupported:
                raise SchemaVersionMismatchError(
                    f"unsupported schema_version values: {sorted(unsupported)}"
                )
        try:
            probabilities = self._loaded.model.predict_proba(frame)[:, 1]
        except Exception as exc:
            raise ScoringError(f"model inference failed: {type(exc).__name__}") from exc
        return np.clip(np.asarray(probabilities, dtype="float64"), 0.0, 1.0)

    def score_frame(
        self, frame: pd.DataFrame, *, scored_at: datetime | None = None
    ) -> pd.DataFrame:
        """Score a validated contract frame; returns ``SCORE_COLUMNS`` on the input index."""
        scored_at = scored_at or datetime.now(tz=UTC)
        probabilities = self.probabilities(frame)
        bands = [self.policy.band(float(p)) for p in probabilities]
        return pd.DataFrame(
            {
                "transaction_id": frame["transaction_id"].astype(str).to_numpy(),
                "risk_probability": probabilities,
                "risk_band": [b.value for b in bands],
                "decision_support": [self.policy.decision(b).value for b in bands],
                "reason_codes": [[c.value for c in codes] for codes in reason_codes(frame)],
                "model_version": self.model_version,
                "scored_at": scored_at,
            },
            index=frame.index,
        )

    def score_transactions(self, transactions: list[TransactionFeatures]) -> list[ScoreResponse]:
        frame = normalise_frame(pd.DataFrame([t.model_dump(mode="python") for t in transactions]))
        scored = self.score_frame(frame)
        responses = []
        for record in scored.to_dict(orient="records"):
            responses.append(
                ScoreResponse(
                    transaction_id=str(record["transaction_id"]),
                    model_version=self.model_version,
                    risk_probability=round(float(record["risk_probability"]), 6),
                    risk_band=record["risk_band"],
                    decision_support=record["decision_support"],
                    reason_codes=list(record["reason_codes"]),
                    threshold=self.policy.threshold,
                    scored_at=record["scored_at"],
                )
            )
        return responses

    def score_one(self, transaction: TransactionFeatures) -> ScoreResponse:
        return self.score_transactions([transaction])[0]
