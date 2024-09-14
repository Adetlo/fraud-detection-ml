"""Runtime configuration (environment variables / ``.env``); see ``.env.example``.

Cost values and policy targets are illustrative synthetic units used to demonstrate
cost-sensitive threshold selection. They are not estimates of real financial loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CostModel(BaseModel):
    """Illustrative per-outcome costs in synthetic units (not currency)."""

    model_config = ConfigDict(frozen=True)

    false_negative_cost: float = Field(default=100.0, ge=0)
    false_positive_cost: float = Field(default=5.0, ge=0)
    true_positive_cost: float = Field(default=1.0, ge=0)
    true_negative_cost: float = Field(default=0.0, ge=0)

    def expected_cost(self, tp: int, fp: int, tn: int, fn: int) -> float:
        return (
            fn * self.false_negative_cost
            + fp * self.false_positive_cost
            + tp * self.true_positive_cost
            + tn * self.true_negative_cost
        )


class ThresholdPolicy(BaseModel):
    """Operational policy for selecting the review threshold (``docs/THRESHOLDING.md``)."""

    model_config = ConfigDict(frozen=True)

    recall_target: float = Field(default=0.80, gt=0, le=1)
    max_review_rate: float = Field(default=0.05, gt=0, le=1)
    priority_precision_target: float = Field(default=0.75, gt=0, le=1)
    low_band_fraction: float = Field(
        default=0.5, gt=0, lt=1, description="low_upper = threshold * low_band_fraction"
    )


class DriftThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    psi_warning: float = Field(default=0.10, gt=0)
    psi_drift: float = Field(default=0.25, gt=0)
    drifted_feature_fraction_for_drift: float = Field(default=0.25, gt=0, le=1)
    bins: int = Field(default=10, ge=2, le=100)

    @model_validator(mode="after")
    def _ordered(self) -> DriftThresholds:
        if self.psi_drift <= self.psi_warning:
            raise ValueError("psi_drift must exceed psi_warning")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FRAUD_ML_",
        extra="ignore",
        frozen=True,
    )

    model_dir: Path = Field(default=Path("./models"))
    data_dir: Path = Field(default=Path("./data"))
    reports_dir: Path = Field(default=Path("./reports"))

    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment: str = "fraud-risk-scoring"

    false_negative_cost: float = Field(default=100.0, ge=0)
    false_positive_cost: float = Field(default=5.0, ge=0)
    true_positive_cost: float = Field(default=1.0, ge=0)

    recall_target: float = Field(default=0.80, gt=0, le=1)
    max_review_rate: float = Field(default=0.05, gt=0, le=1)
    priority_precision_target: float = Field(default=0.75, gt=0, le=1)

    api_max_batch_size: int = Field(default=1000, ge=1, le=100_000)
    api_max_request_bytes: int = Field(default=2_000_000, ge=1024)

    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def cost_model(self) -> CostModel:
        return CostModel(
            false_negative_cost=self.false_negative_cost,
            false_positive_cost=self.false_positive_cost,
            true_positive_cost=self.true_positive_cost,
        )

    @property
    def threshold_policy(self) -> ThresholdPolicy:
        return ThresholdPolicy(
            recall_target=self.recall_target,
            max_review_rate=self.max_review_rate,
            priority_precision_target=self.priority_precision_target,
        )


def get_settings() -> Settings:
    return Settings()
