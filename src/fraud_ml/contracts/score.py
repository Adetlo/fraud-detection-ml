"""Versioned scoring output contract (``fraud-score/1``) and API envelopes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fraud_ml.contracts.enums import (
    SCORE_SCHEMA_VERSION,
    DecisionSupport,
    ReasonCode,
    RiskBand,
)
from fraud_ml.contracts.transaction import TransactionFeatures


class RiskPolicy(BaseModel):
    """Operating policy persisted with a model artifact and reported with every score.

    Band boundaries (documented in ``docs/THRESHOLDING.md``)::

        LOW       risk_probability <  low_upper
        MEDIUM    low_upper        <= risk_probability < threshold
        HIGH      threshold        <= risk_probability < priority_threshold
        CRITICAL  priority_threshold <= risk_probability

    Decision support: LOW/MEDIUM -> PASS, HIGH -> REVIEW, CRITICAL -> PRIORITY_REVIEW.
    """

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(ge=0.0, le=1.0, description="Review threshold")
    priority_threshold: float = Field(ge=0.0, le=1.0)
    low_upper: float = Field(ge=0.0, le=1.0)

    def band(self, probability: float) -> RiskBand:
        if probability >= self.priority_threshold:
            return RiskBand.CRITICAL
        if probability >= self.threshold:
            return RiskBand.HIGH
        if probability >= self.low_upper:
            return RiskBand.MEDIUM
        return RiskBand.LOW

    @staticmethod
    def decision(band: RiskBand) -> DecisionSupport:
        if band is RiskBand.CRITICAL:
            return DecisionSupport.PRIORITY_REVIEW
        if band is RiskBand.HIGH:
            return DecisionSupport.REVIEW
        return DecisionSupport.PASS


class ScoreResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SCORE_SCHEMA_VERSION
    transaction_id: str
    model_version: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    decision_support: DecisionSupport
    reason_codes: list[ReasonCode]
    threshold: float = Field(ge=0.0, le=1.0)
    scored_at: datetime


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionFeatures] = Field(min_length=1)


class BatchScoreResponse(BaseModel):
    model_version: str
    count: int
    scores: list[ScoreResponse]


class MetricsSummary(BaseModel):
    """Metrics at the selected operating threshold; produced by ``training.evaluation``."""

    model_config = ConfigDict(frozen=True)

    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    brier_score: float
    review_rate: float
    alerts_per_1000: float
    expected_cost: float
    expected_cost_per_1000: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    n_rows: int
    n_positives: int
    threshold: float


class ModelInfo(BaseModel):
    """Public description of the loaded model; never exposes filesystem paths."""

    model_version: str
    model_name: str
    schema_version: str
    score_schema_version: str = SCORE_SCHEMA_VERSION
    feature_version: str
    training_timestamp: datetime
    training_data_version: str
    git_commit: str | None
    policy: RiskPolicy
    metrics: MetricsSummary | None
    feature_count: int


class ErrorDetail(BaseModel):
    code: str
    message: str
    correlation_id: str
    details: list[Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
