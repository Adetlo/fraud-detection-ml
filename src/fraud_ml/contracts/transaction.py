"""Versioned transaction feature input contract (``fraud-transaction/1``).

A record describes one transaction **plus the point-in-time context** an upstream feature
process knew *before* the transaction was processed. Every history field (velocity counts,
account statistics, days since last transaction) must be computed from strictly earlier
transactions; ``fraud_ml.features.velocity.compute_history_features`` is the reference
implementation and the generator uses it.

Identifiers are synthetic (``TXN-SYN-…``, ``ACC-SYN-…``, ``CP-SYN-…``) and are never used as
predictive features.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from fraud_ml.contracts.enums import (
    CURRENCY_EXPONENT,
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    Channel,
    Currency,
    CustomerSegment,
    FraudType,
    IpRiskBand,
    TransactionType,
)

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


def _country(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z]{2}", value):
        raise ValueError("country codes must be ISO 3166-1 alpha-2")
    return value


CountryCode = Annotated[str, AfterValidator(_country)]
Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

#: Upper bound on a single transaction amount (major units). Larger values are rejected as
#: implausible for this synthetic domain rather than silently clipped.
MAX_AMOUNT = Decimal("100000000")
#: Timestamps more than this many days in the future are rejected.
MAX_FUTURE_DAYS = 1
#: Earliest accepted transaction timestamp.
MIN_TIMESTAMP = datetime(2000, 1, 1, tzinfo=UTC)
#: Velocity counts above this are treated as impossible for one account.
MAX_WINDOW_TRANSACTIONS = 10_000
MAX_ACCOUNT_AGE_DAYS = 36_500


def _normalise_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("transaction_timestamp must carry a timezone offset")
    value = value.astimezone(UTC)
    if value < MIN_TIMESTAMP:
        raise ValueError(f"transaction_timestamp before {MIN_TIMESTAMP.date()}")
    return value


class TransactionFeatures(BaseModel):
    """One scoring request: transaction attributes and point-in-time account context."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION)
    transaction_id: Identifier
    account_id: Identifier
    transaction_timestamp: datetime
    amount: Decimal = Field(gt=0, le=MAX_AMOUNT, description="Major units of ``currency``")
    currency: Currency
    transaction_type: TransactionType
    channel: Channel
    country: CountryCode
    counterparty_id: Identifier | None = None
    beneficiary_country: CountryCode | None = None

    # --- point-in-time context (computed from strictly earlier transactions) ---------------
    is_weekend: bool
    account_age_days: int = Field(ge=0, le=MAX_ACCOUNT_AGE_DAYS)
    transactions_1h: int = Field(ge=0, le=MAX_WINDOW_TRANSACTIONS)
    transactions_24h: int = Field(ge=0, le=MAX_WINDOW_TRANSACTIONS)
    amount_1h: Decimal = Field(ge=0)
    amount_24h: Decimal = Field(ge=0)
    unique_counterparties_24h: int = Field(ge=0, le=MAX_WINDOW_TRANSACTIONS)
    days_since_last_transaction: float | None = Field(
        default=None, ge=0, description="None for the first transaction on an account"
    )
    account_median_amount_30d: Decimal | None = Field(default=None, ge=0)
    account_mean_amount_30d: Decimal | None = Field(default=None, ge=0)
    inflow_amount_24h: Decimal = Field(
        default=Decimal(0), ge=0, description="Sum of deposits/refunds in the prior 24 hours"
    )

    # --- optional attributes ---------------------------------------------------------------
    merchant_category: Category | None = None
    device_id_hash: Identifier | None = None
    ip_risk_band: IpRiskBand = IpRiskBand.UNKNOWN
    customer_segment: CustomerSegment | None = None
    is_new_beneficiary: bool | None = None
    is_new_device: bool | None = None

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {value!r}; supported: "
                f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return value

    @field_validator("transaction_timestamp")
    @classmethod
    def _timestamp(cls, value: datetime) -> datetime:
        return _normalise_timestamp(value)

    @field_validator("amount", "amount_1h", "amount_24h", "inflow_amount_24h", mode="before")
    @classmethod
    def _decimal_from_text(cls, value: Any) -> Any:
        # Floats are converted through ``str`` so 0.1 stays 0.1, not 0.1000000000000000055.
        if isinstance(value, float):
            return Decimal(str(value))
        return value

    @model_validator(mode="after")
    def _consistency(self) -> TransactionFeatures:
        exponent = CURRENCY_EXPONENT[self.currency]
        places = max(0, -int(self.amount.normalize().as_tuple().exponent))  # ignores trailing zeros
        if places > exponent:
            raise ValueError(f"amount has more than {exponent} decimal places for {self.currency}")
        if self.transactions_1h > self.transactions_24h:
            raise ValueError("transactions_1h cannot exceed transactions_24h")
        if self.amount_1h > self.amount_24h:
            raise ValueError("amount_1h cannot exceed amount_24h")
        if self.unique_counterparties_24h > self.transactions_24h:
            raise ValueError("unique_counterparties_24h cannot exceed transactions_24h")
        if self.days_since_last_transaction is None and self.transactions_24h > 0:
            raise ValueError("days_since_last_transaction is required when transactions_24h > 0")
        return self


class LabelledTransaction(TransactionFeatures):
    """Training/evaluation row: the input contract plus synthetic ground-truth labels.

    ``is_fraud`` and ``fraud_type`` exist only in generated or evaluation datasets and are
    guarded out of the feature matrix by ``fraud_ml.features.definitions.FORBIDDEN_COLUMNS``.
    """

    is_fraud: bool
    fraud_type: FraudType = FraudType.NONE

    @model_validator(mode="after")
    def _label_consistency(self) -> LabelledTransaction:
        if self.is_fraud and self.fraud_type == FraudType.NONE:
            raise ValueError("fraud rows must carry a fraud_type other than NONE")
        if not self.is_fraud and self.fraud_type != FraudType.NONE:
            raise ValueError("non-fraud rows must have fraud_type NONE")
        return self


#: Columns every input frame must contain (contract fields without defaults).
REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    name for name, field in TransactionFeatures.model_fields.items() if field.is_required()
)
#: Optional columns the loader creates when missing.
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(
    name for name, field in TransactionFeatures.model_fields.items() if not field.is_required()
)
LABEL_COLUMNS: tuple[str, ...] = ("is_fraud", "fraud_type")
