"""Feature catalogue: what enters the model, what never may, and the feature version.

``FEATURE_VERSION`` is persisted in every model artifact; a scorer refuses an artifact whose
feature version differs from the running code (``fraud_ml.models.registry``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fraud_ml.contracts.transaction import LABEL_COLUMNS

FEATURE_VERSION = "features/1"

FeatureGroup = Literal["transaction", "temporal", "velocity", "behavioural"]
FeatureKind = Literal["numeric", "binary", "categorical"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: FeatureGroup
    kind: FeatureKind
    description: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # --- transaction -----------------------------------------------------------------------
    FeatureSpec("log_amount", "transaction", "numeric", "log1p of the amount in reference units"),
    FeatureSpec("amount_band", "transaction", "categorical", "Reference-unit amount bucket"),
    FeatureSpec(
        "threshold_proximity",
        "transaction",
        "numeric",
        "amount / next round threshold (0 above the largest)",
    ),
    FeatureSpec(
        "near_threshold_flag", "transaction", "binary", "Amount within 15% below a round threshold"
    ),
    FeatureSpec("currency", "transaction", "categorical", "Transaction currency"),
    FeatureSpec("transaction_type", "transaction", "categorical", "Transaction type"),
    FeatureSpec("channel", "transaction", "categorical", "Initiation channel"),
    FeatureSpec(
        "country", "transaction", "categorical", "Country of initiation (rare values pooled)"
    ),
    FeatureSpec(
        "beneficiary_country",
        "transaction",
        "categorical",
        "Beneficiary country (rare values pooled)",
    ),
    FeatureSpec(
        "merchant_category", "transaction", "categorical", "Merchant category (rare values pooled)"
    ),
    FeatureSpec("ip_risk_band", "transaction", "categorical", "Upstream IP reputation band"),
    FeatureSpec("customer_segment", "transaction", "categorical", "Customer segment"),
    # --- temporal --------------------------------------------------------------------------
    FeatureSpec("hour_of_day", "temporal", "numeric", "UTC hour 0-23"),
    FeatureSpec("day_of_week", "temporal", "numeric", "Monday=0 .. Sunday=6"),
    FeatureSpec("hour_sin", "temporal", "numeric", "Cyclical encoding of the hour"),
    FeatureSpec("hour_cos", "temporal", "numeric", "Cyclical encoding of the hour"),
    FeatureSpec("unusual_hour_score", "temporal", "numeric", "0 (office hours) .. 1 (deep night)"),
    FeatureSpec("is_weekend", "temporal", "binary", "Saturday or Sunday"),
    FeatureSpec("is_night", "temporal", "binary", "Hour between 00:00 and 05:59"),
    # --- velocity --------------------------------------------------------------------------
    FeatureSpec("transactions_1h", "velocity", "numeric", "Prior transactions in the last hour"),
    FeatureSpec(
        "transactions_24h", "velocity", "numeric", "Prior transactions in the last 24 hours"
    ),
    FeatureSpec("log1p_amount_1h", "velocity", "numeric", "log1p of prior 1h amount"),
    FeatureSpec("log1p_amount_24h", "velocity", "numeric", "log1p of prior 24h amount"),
    FeatureSpec(
        "unique_counterparties_24h", "velocity", "numeric", "Distinct counterparties in prior 24h"
    ),
    FeatureSpec(
        "velocity_ratio_1h_24h", "velocity", "numeric", "transactions_1h / max(transactions_24h, 1)"
    ),
    FeatureSpec(
        "amount_vs_inflow_24h",
        "velocity",
        "numeric",
        "amount / prior 24h inflow (0 when no inflow)",
    ),
    FeatureSpec("inflow_present_24h", "velocity", "binary", "Any deposit/refund in the prior 24h"),
    # --- behavioural -----------------------------------------------------------------------
    FeatureSpec(
        "amount_vs_account_median",
        "behavioural",
        "numeric",
        "amount / 30-day median (missing without history)",
    ),
    FeatureSpec(
        "amount_vs_account_mean",
        "behavioural",
        "numeric",
        "amount / 30-day mean (missing without history)",
    ),
    FeatureSpec(
        "log1p_days_since_last", "behavioural", "numeric", "log1p days since previous transaction"
    ),
    FeatureSpec("log1p_account_age_days", "behavioural", "numeric", "log1p account age"),
    FeatureSpec(
        "has_history", "behavioural", "binary", "Account has at least one prior transaction"
    ),
    FeatureSpec(
        "new_beneficiary_flag",
        "behavioural",
        "binary",
        "Counterparty not seen before on the account",
    ),
    FeatureSpec(
        "new_device_flag", "behavioural", "binary", "Device not seen before on the account"
    ),
    FeatureSpec(
        "cross_border_flag", "behavioural", "binary", "Beneficiary country differs from country"
    ),
    FeatureSpec("dormant_flag", "behavioural", "binary", "No transaction in the previous 30 days"),
    FeatureSpec("new_account_flag", "behavioural", "binary", "Account younger than 30 days"),
)

NUMERIC_FEATURES: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS if s.kind == "numeric")
BINARY_FEATURES: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS if s.kind == "binary")
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    s.name for s in FEATURE_SPECS if s.kind == "categorical"
)
ALL_FEATURES: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)

#: Open-vocabulary categoricals: rare and unseen values are pooled into an explicit bucket.
HIGH_CARDINALITY_CATEGORICAL: tuple[str, ...] = (
    "country",
    "beneficiary_country",
    "merchant_category",
)

#: Identifier columns: carried through for traceability, never used as features.
ID_COLUMNS: tuple[str, ...] = ("transaction_id", "account_id", "counterparty_id", "device_id_hash")

#: Columns that may never enter the feature matrix: labels, identifiers, raw timestamps and
#: anything that would only be known *after* the transaction was investigated.
POST_OUTCOME_COLUMNS: tuple[str, ...] = (
    "investigation_outcome",
    "confirmed_fraud",
    "chargeback_flag",
    "case_id",
    "alert_id",
    "analyst_decision",
    "reported_at",
    "risk_probability",
    "fraud_score",
    "decision_support",
)
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    LABEL_COLUMNS + ID_COLUMNS + POST_OUTCOME_COLUMNS + ("transaction_timestamp", "schema_version")
)

assert not FORBIDDEN_COLUMNS.intersection(ALL_FEATURES)  # noqa: S101 - import-time invariant
assert len(set(ALL_FEATURES)) == len(ALL_FEATURES)  # noqa: S101
