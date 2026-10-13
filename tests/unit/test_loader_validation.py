from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_ml.data.loader import (
    ensure_optional_columns,
    normalise_frame,
    read_frame,
    sha256_file,
    sha256_frame,
    write_frame,
)
from fraud_ml.data.validation import validate_frame
from fraud_ml.exceptions import DataValidationError
from tests.conftest import NOW, transaction_payload


def _frame(n: int = 5, **overrides: object) -> pd.DataFrame:
    rows = [transaction_payload(transaction_id=f"TXN-SYN-{i:04d}", **overrides) for i in range(n)]
    return normalise_frame(pd.DataFrame(rows))


@pytest.mark.parametrize("suffix", [".parquet", ".jsonl", ".csv"])
def test_round_trip_formats(tmp_path: Path, suffix: str) -> None:
    frame = _frame()
    path = write_frame(frame, tmp_path / f"t{suffix}")
    loaded = read_frame(path)
    assert len(loaded) == len(frame)
    assert loaded["transaction_timestamp"].dt.tz is not None
    assert loaded["amount"].dtype == "float64"
    assert validate_frame(loaded, now=NOW).n_invalid == 0


def test_unsupported_formats(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError):
        read_frame(tmp_path / "x.xlsx")
    with pytest.raises(DataValidationError):
        write_frame(_frame(), tmp_path / "x.xml")


def test_empty_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    _frame(1).iloc[0:0].to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="no rows"):
        read_frame(path)


def test_naive_timestamps_are_rejected_not_assumed_utc(tmp_path: Path) -> None:
    frame = _frame()
    frame["transaction_timestamp"] = frame["transaction_timestamp"].dt.tz_localize(None)
    with pytest.raises(DataValidationError, match="timezone-aware"):
        normalise_frame(frame)
    path = tmp_path / "naive.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="no timezone offset"):
        read_frame(path)


def test_bool_parsing_from_text() -> None:
    frame = pd.DataFrame(
        [transaction_payload(is_weekend="yes"), transaction_payload(is_weekend="0")]
    )
    out = normalise_frame(frame)
    assert out["is_weekend"].tolist() == [True, False]
    with pytest.raises(DataValidationError):
        normalise_frame(pd.DataFrame([transaction_payload(is_weekend="maybe")]))


def test_ensure_optional_columns_adds_defaults() -> None:
    frame = pd.DataFrame([{"transaction_id": "a"}])
    out = ensure_optional_columns(frame)
    assert out["ip_risk_band"].iloc[0] == "UNKNOWN"
    assert out["counterparty_id"].iloc[0] is None


def test_hashes(tmp_path: Path) -> None:
    frame = _frame()
    path = write_frame(frame, tmp_path / "a.parquet")
    assert len(sha256_file(path)) == 64
    assert sha256_frame(frame) == sha256_frame(frame.iloc[::-1])  # order independent
    assert sha256_frame(frame) != sha256_frame(frame.assign(amount=frame["amount"] + 1))


def test_validation_flags_each_rule() -> None:
    frame = _frame(9)
    frame.loc[0, "amount"] = -1.0
    frame.loc[1, "transaction_timestamp"] = pd.Timestamp(NOW + timedelta(days=3))
    frame.loc[2, "channel"] = "PIGEON"
    frame.loc[3, "transactions_1h"] = 9
    frame.loc[4, "country"] = "GBR"
    frame.loc[5, "schema_version"] = "fraud-transaction/7"
    frame.loc[6, "account_age_days"] = -4
    frame.loc[7, "transaction_id"] = " "
    frame.loc[8, "amount_1h"] = 1e9
    result = validate_frame(frame, strict=False, now=NOW)
    assert result.n_rows == 9 and result.n_valid == 0
    codes = result.error_counts
    assert codes["INVALID_AMOUNT"] == 1
    assert codes["FUTURE_TIMESTAMP"] == 1
    assert codes["UNSUPPORTED_CATEGORY:channel"] == 1
    assert codes["INCONSISTENT_WINDOWS"] == 2
    assert codes["INVALID_COUNTRY"] == 1
    assert codes["SCHEMA_VERSION_MISMATCH"] == 1
    assert codes["INVALID_FEATURE:account_age_days"] == 1
    assert codes["MISSING_REQUIRED:transaction_id"] == 1
    rejected = result.invalid_rows(frame)
    assert "validation_errors" in rejected.columns
    assert rejected["validation_errors"].str.len().gt(0).all()
    assert result.summary()["invalid"] == 9


def test_validation_strict_raises_with_codes() -> None:
    frame = _frame(3)
    frame.loc[1, "amount"] = np.nan
    with pytest.raises(DataValidationError) as excinfo:
        validate_frame(frame, now=NOW)
    assert "INVALID_AMOUNT" in excinfo.value.errors


def test_validation_requires_columns_and_tz() -> None:
    frame = _frame().drop(columns=["amount"])
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_frame(frame)
    frame = _frame()
    frame["transaction_timestamp"] = frame["transaction_timestamp"].astype(str)
    with pytest.raises(DataValidationError, match="datetime"):
        validate_frame(frame)
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_frame(_frame(), require_labels=True)


def test_label_consistency_rules() -> None:
    frame = _frame(2)
    frame["is_fraud"] = [True, False]
    frame["fraud_type"] = ["NONE", "STRUCTURING"]
    result = validate_frame(frame, require_labels=True, strict=False, now=NOW)
    assert result.error_counts["LABEL_INCONSISTENT"] == 2


def test_history_consistency_rule() -> None:
    frame = _frame(1, days_since_last_transaction=None, transactions_24h=2)
    result = validate_frame(frame, strict=False, now=NOW)
    assert "INCONSISTENT_HISTORY" in result.error_counts


def test_timestamp_before_minimum() -> None:
    frame = _frame(1)
    frame.loc[0, "transaction_timestamp"] = pd.Timestamp(datetime(1990, 1, 1, tzinfo=UTC))
    assert "INVALID_TIMESTAMP" in validate_frame(frame, strict=False, now=NOW).error_counts
