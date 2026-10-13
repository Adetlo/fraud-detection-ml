"""Read and write transaction frames (Parquet, JSON Lines, CSV) with stable dtypes.

A loaded frame always has timezone-aware UTC timestamps and ``float64`` money columns.
Naive timestamps are rejected rather than assumed to be UTC.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_ml.contracts.transaction import OPTIONAL_COLUMNS, TransactionFeatures
from fraud_ml.exceptions import DataValidationError

MONEY_COLUMNS: tuple[str, ...] = (
    "amount",
    "amount_1h",
    "amount_24h",
    "inflow_amount_24h",
    "account_median_amount_30d",
    "account_mean_amount_30d",
)
TIMESTAMP_COLUMN = "transaction_timestamp"
_OFFSET_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    """Content hash of a frame, independent of row order and index."""
    hashed = pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False)
    total = hashed.sort_values().to_numpy(dtype="uint64").tobytes()
    return hashlib.sha256(total).hexdigest()


def _coerce_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    if TIMESTAMP_COLUMN not in frame.columns:
        return frame
    column = frame[TIMESTAMP_COLUMN]
    if pd.api.types.is_datetime64_any_dtype(column):
        if getattr(column.dt, "tz", None) is None:
            raise DataValidationError(
                f"{TIMESTAMP_COLUMN} must be timezone-aware; naive timestamps are not assumed UTC"
            )
        frame[TIMESTAMP_COLUMN] = column.dt.tz_convert("UTC")
        return frame
    text = column.astype("string")
    missing_offset = text.notna() & ~text.str.strip().str.contains(_OFFSET_RE, regex=True)
    if bool(missing_offset.any()):
        raise DataValidationError(
            f"{int(missing_offset.sum())} {TIMESTAMP_COLUMN} values have no timezone offset"
        )
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(text, utc=True, format="ISO8601", errors="coerce")
    return frame


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the dtype conventions every loader path shares."""
    frame = frame.copy()
    frame = _coerce_timestamps(frame)
    for column in MONEY_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ("is_weekend", "is_new_beneficiary", "is_new_device", "is_fraud"):
        if column in frame.columns and not pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].map(_parse_bool).astype("boolean")
    return ensure_optional_columns(frame)


def ensure_optional_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add optional contract columns that are absent, using the contract defaults."""
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            default = TransactionFeatures.model_fields[column].get_default()
            frame[column] = default if default is not None else None
    return frame


def _parse_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    raise DataValidationError(f"cannot interpret {value!r} as a boolean")


def read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype={"transaction_id": "string", "account_id": "string"})
    else:
        raise DataValidationError(f"unsupported input format {suffix!r} (use .parquet/.jsonl/.csv)")
    if frame.empty:
        raise DataValidationError(f"{path} contains no rows")
    return normalise_frame(frame)


def write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open("w", encoding="utf-8") as handle:
            for record in frame.to_dict(orient="records"):
                handle.write(json.dumps(record, default=_json_default) + "\n")
    elif suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise DataValidationError(f"unsupported output format {suffix!r}")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)
