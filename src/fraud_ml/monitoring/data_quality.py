"""Input data-quality snapshot for a scoring batch (missing rates, invalid rows, ranges)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from fraud_ml.contracts.transaction import REQUIRED_COLUMNS
from fraud_ml.data.validation import ValidationResult, validate_frame
from fraud_ml.monitoring.drift import NUMERIC_DRIFT_COLUMNS


class DataQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: int
    valid_rows: int
    invalid_rows: int
    invalid_rate: float
    error_counts: dict[str, int]
    missing_rates: dict[str, float]
    numeric_ranges: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def data_quality_report(
    frame: pd.DataFrame, validation: ValidationResult | None = None
) -> DataQualityReport:
    validation = validation or validate_frame(frame, strict=False)
    missing = {c: float(frame[c].isna().mean()) for c in REQUIRED_COLUMNS if c in frame.columns}
    ranges: dict[str, dict[str, float]] = {}
    for column in NUMERIC_DRIFT_COLUMNS:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values):
                ranges[column] = {
                    "min": float(values.min()),
                    "median": float(values.median()),
                    "max": float(values.max()),
                }
    return DataQualityReport(
        rows=validation.n_rows,
        valid_rows=validation.n_valid,
        invalid_rows=validation.n_invalid,
        invalid_rate=validation.n_invalid / validation.n_rows if validation.n_rows else 0.0,
        error_counts=dict(sorted(validation.error_counts.items())),
        missing_rates=missing,
        numeric_ranges=ranges,
    )
