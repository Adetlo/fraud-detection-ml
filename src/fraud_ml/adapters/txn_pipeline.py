"""Adapter for the curated Parquet output of ``txn-data-pipeline`` (conceptual integration).

The curated layer carries the raw transaction attributes but not the point-in-time account
context, so this adapter (1) maps column names and vocabularies onto ``fraud-transaction/1``,
(2) derives the history fields with the same routine the generator uses, and (3) returns a
frame that passes ``validate_frame``. It is optional: the repository never requires the
pipeline at runtime (``docs/DATA_CONTRACT.md``).

Account age is approximated from the first transaction observed per account unless an
``account_open_date`` column is supplied, and this approximation is documented as such.
"""

from __future__ import annotations

import pandas as pd

from fraud_ml.contracts.enums import SCHEMA_VERSION, TransactionType
from fraud_ml.contracts.transaction import OPTIONAL_COLUMNS
from fraud_ml.data.loader import normalise_frame
from fraud_ml.exceptions import DataValidationError
from fraud_ml.features.velocity import compute_history_features

#: Curated columns the adapter needs (names as produced by txn-data-pipeline).
CURATED_REQUIRED: tuple[str, ...] = (
    "transaction_id",
    "account_id",
    "transaction_timestamp",
    "amount",
    "currency",
    "transaction_type",
    "channel",
    "country",
)
#: Curated transaction types without a counterpart in this contract are mapped explicitly.
TYPE_MAPPING: dict[str, str] = {
    "FEE": TransactionType.PAYMENT.value,
    "INTEREST": TransactionType.DEPOSIT.value,
}
#: Only posted transactions are scored; pending/reversed/cancelled rows are out of scope.
SCORABLE_STATUSES: frozenset[str] = frozenset({"POSTED"})


def adapt_curated_frame(curated: pd.DataFrame, *, only_posted: bool = True) -> pd.DataFrame:
    missing = [c for c in CURATED_REQUIRED if c not in curated.columns]
    if missing:
        raise DataValidationError(f"curated frame is missing columns: {missing}")
    frame = curated.copy()
    if only_posted and "status" in frame.columns:
        frame = frame[frame["status"].astype("string").isin(SCORABLE_STATUSES)]
    if frame.empty:
        raise DataValidationError("no scorable rows after filtering by status")
    frame["transaction_type"] = (
        frame["transaction_type"].astype("string").map(lambda t: TYPE_MAPPING.get(str(t), str(t)))
    )
    frame = normalise_frame(frame)
    for column in ("counterparty_id", "beneficiary_country", "merchant_category", "device_id_hash"):
        if column not in frame.columns:
            frame[column] = None
    frame = compute_history_features(frame)
    frame["schema_version"] = SCHEMA_VERSION
    keep = [
        "schema_version", "transaction_id", "account_id", "transaction_timestamp", "amount",
        "currency", "transaction_type", "channel", "country", "counterparty_id",
        "beneficiary_country",
    ]  # fmt: skip
    keep += [c for c in OPTIONAL_COLUMNS if c in frame.columns and c not in keep]
    keep += [c for c in compute_history_features.__globals__["HISTORY_COLUMNS"] if c not in keep]
    return frame.loc[:, keep].reset_index(drop=True)
