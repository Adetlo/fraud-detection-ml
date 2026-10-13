"""Frame-level data quality validation against the transaction contract.

The frame validator mirrors the Pydantic contract for bulk data: the same enums, bounds and
cross-field rules, applied as vectorised checks. Rows that fail are *reported*, never silently
coerced. ``strict=True`` (training) raises on the first invalid row; batch scoring uses
``strict=False`` and scores only the valid rows while writing the rejects to a separate file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from fraud_ml.contracts.enums import (
    SUPPORTED_SCHEMA_VERSIONS,
    Channel,
    Currency,
    CustomerSegment,
    FraudType,
    IpRiskBand,
    TransactionType,
)
from fraud_ml.contracts.transaction import (
    LABEL_COLUMNS,
    MAX_ACCOUNT_AGE_DAYS,
    MAX_AMOUNT,
    MAX_FUTURE_DAYS,
    MAX_WINDOW_TRANSACTIONS,
    MIN_TIMESTAMP,
    REQUIRED_COLUMNS,
)
from fraud_ml.exceptions import DataValidationError

CLOSED_VOCABULARIES: dict[str, frozenset[str]] = {
    "currency": frozenset(Currency),
    "transaction_type": frozenset(TransactionType),
    "channel": frozenset(Channel),
    "ip_risk_band": frozenset(IpRiskBand),
}
OPTIONAL_VOCABULARIES: dict[str, frozenset[str]] = {
    "customer_segment": frozenset(CustomerSegment),
    "fraud_type": frozenset(FraudType),
}
_COUNTRY_RE = r"^[A-Z]{2}$"


@dataclass
class ValidationResult:
    valid_mask: pd.Series
    row_errors: pd.Series
    error_counts: dict[str, int] = field(default_factory=dict)
    n_rows: int = 0

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def n_invalid(self) -> int:
        return self.n_rows - self.n_valid

    def valid_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[self.valid_mask.to_numpy()]

    def invalid_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        rejected = frame.loc[~self.valid_mask.to_numpy()].copy()
        rejected["validation_errors"] = self.row_errors.loc[~self.valid_mask.to_numpy()].map(
            ";".join
        )
        return rejected

    def summary(self) -> dict[str, object]:
        return {
            "rows": self.n_rows,
            "valid": self.n_valid,
            "invalid": self.n_invalid,
            "error_counts": dict(sorted(self.error_counts.items())),
        }


def validate_frame(
    frame: pd.DataFrame,
    *,
    require_labels: bool = False,
    strict: bool = True,
    now: datetime | None = None,
) -> ValidationResult:
    """Validate a frame; see module docstring for the strict/non-strict semantics."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if require_labels:
        missing += [c for c in LABEL_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(
            f"missing required columns: {missing}", errors=[f"MISSING_COLUMN:{c}" for c in missing]
        )
    if not pd.api.types.is_datetime64_any_dtype(frame["transaction_timestamp"]):
        raise DataValidationError("transaction_timestamp must be a timezone-aware datetime column")
    if getattr(frame["transaction_timestamp"].dt, "tz", None) is None:
        raise DataValidationError("transaction_timestamp must be timezone-aware")

    n = len(frame)
    codes: list[list[str]] = [[] for _ in range(n)]

    def flag(mask: pd.Series | np.ndarray, code: str) -> None:
        array = np.asarray(mask, dtype=bool)
        for position in np.flatnonzero(array):
            codes[position].append(code)

    for column in ("transaction_id", "account_id"):
        text = frame[column].astype("string")
        flag(text.isna() | (text.str.strip() == ""), f"MISSING_REQUIRED:{column}")
    if "schema_version" in frame.columns:
        versions = frame["schema_version"].astype("string")
        flag(
            versions.notna() & ~versions.isin(SUPPORTED_SCHEMA_VERSIONS), "SCHEMA_VERSION_MISMATCH"
        )

    amount = pd.to_numeric(frame["amount"], errors="coerce")
    flag(amount.isna() | (amount <= 0) | (amount > float(MAX_AMOUNT)), "INVALID_AMOUNT")

    now = now or datetime.now(tz=UTC)
    ts = frame["transaction_timestamp"]
    flag(ts.isna(), "INVALID_TIMESTAMP")
    flag(ts.notna() & (ts < MIN_TIMESTAMP), "INVALID_TIMESTAMP")
    flag(ts.notna() & (ts > now + timedelta(days=MAX_FUTURE_DAYS)), "FUTURE_TIMESTAMP")

    for column, vocabulary in CLOSED_VOCABULARIES.items():
        values = frame[column].astype("string")
        flag(values.isna(), f"MISSING_REQUIRED:{column}")
        flag(values.notna() & ~values.isin(vocabulary), f"UNSUPPORTED_CATEGORY:{column}")
    for column, vocabulary in OPTIONAL_VOCABULARIES.items():
        if column in frame.columns:
            values = frame[column].astype("string")
            flag(values.notna() & ~values.isin(vocabulary), f"UNSUPPORTED_CATEGORY:{column}")

    country = frame["country"].astype("string")
    flag(country.isna() | ~country.str.fullmatch(_COUNTRY_RE).fillna(False), "INVALID_COUNTRY")
    if "beneficiary_country" in frame.columns:
        bene = frame["beneficiary_country"].astype("string")
        flag(bene.notna() & ~bene.str.fullmatch(_COUNTRY_RE).fillna(False), "INVALID_COUNTRY")

    def numeric(column: str) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce").astype("float64")

    for column in ("transactions_1h", "transactions_24h", "unique_counterparties_24h"):
        nums = numeric(column)
        flag(
            nums.isna() | (nums < 0) | (nums > MAX_WINDOW_TRANSACTIONS),
            f"INVALID_FEATURE:{column}",
        )
    age = numeric("account_age_days")
    flag(age.isna() | (age < 0) | (age > MAX_ACCOUNT_AGE_DAYS), "INVALID_FEATURE:account_age_days")
    for column in ("amount_1h", "amount_24h"):
        nums = numeric(column)
        flag(nums.isna() | (nums < 0), f"INVALID_FEATURE:{column}")
    for column in ("inflow_amount_24h", "account_median_amount_30d", "account_mean_amount_30d"):
        if column in frame.columns:
            nums = numeric(column)
            flag(nums.notna() & (nums < 0), f"INVALID_FEATURE:{column}")
    if "days_since_last_transaction" in frame.columns:
        nums = numeric("days_since_last_transaction")
        flag(nums.notna() & (nums < 0), "INVALID_FEATURE:days_since_last_transaction")
        flag(nums.isna() & (numeric("transactions_24h") > 0), "INCONSISTENT_HISTORY")

    flag(numeric("transactions_1h") > numeric("transactions_24h"), "INCONSISTENT_WINDOWS")
    flag(numeric("amount_1h") > numeric("amount_24h") + 1e-9, "INCONSISTENT_WINDOWS")
    flag(numeric("unique_counterparties_24h") > numeric("transactions_24h"), "INCONSISTENT_WINDOWS")

    if require_labels:
        is_fraud = frame["is_fraud"]
        flag(is_fraud.isna(), "MISSING_REQUIRED:is_fraud")
        fraud_type = frame["fraud_type"].astype("string")
        flag((is_fraud == True) & (fraud_type == FraudType.NONE), "LABEL_INCONSISTENT")  # noqa: E712
        flag((is_fraud == False) & (fraud_type != FraudType.NONE), "LABEL_INCONSISTENT")  # noqa: E712

    row_errors = pd.Series(codes, index=frame.index, dtype=object)
    valid_mask = row_errors.map(lambda c: len(c) == 0).astype(bool)
    counts: dict[str, int] = {}
    for row in codes:
        for code in row:
            counts[code] = counts.get(code, 0) + 1
    result = ValidationResult(
        valid_mask=valid_mask, row_errors=row_errors, error_counts=counts, n_rows=n
    )
    if strict and result.n_invalid:
        top = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())[:8])
        raise DataValidationError(
            f"{result.n_invalid} of {n} rows violate the transaction contract ({top})",
            errors=sorted(counts),
        )
    return result
