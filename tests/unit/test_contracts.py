from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fraud_ml.contracts.enums import SCHEMA_VERSION, DecisionSupport, RiskBand
from fraud_ml.contracts.money import to_decimal, to_float
from fraud_ml.contracts.score import RiskPolicy, ScoreResponse
from fraud_ml.contracts.transaction import (
    LABEL_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    LabelledTransaction,
    TransactionFeatures,
)
from tests.conftest import transaction_payload


def test_valid_payload_parses_to_utc_and_decimal(payload: dict) -> None:
    txn = TransactionFeatures.model_validate(payload)
    assert txn.transaction_timestamp.tzinfo is UTC
    assert txn.amount == Decimal("125.50")
    assert txn.schema_version == SCHEMA_VERSION


def test_offset_timestamp_is_normalised_to_utc() -> None:
    txn = TransactionFeatures.model_validate(
        transaction_payload(transaction_timestamp="2025-06-30T16:05:00+02:00")
    )
    assert txn.transaction_timestamp == datetime(2025, 6, 30, 14, 5, tzinfo=UTC)


@pytest.mark.parametrize(
    "overrides",
    [
        {"transaction_timestamp": "2025-06-30T14:05:00"},  # naive
        {"transaction_timestamp": "1999-12-31T00:00:00Z"},  # before MIN_TIMESTAMP
        {"amount": "0"},
        {"amount": "-5"},
        {"amount": "1.005"},  # three decimals for GBP
        {"amount": "999999999999"},
        {"currency": "XXX"},
        {"channel": "CARRIER_PIGEON"},
        {"transaction_type": "GIFT"},
        {"country": "GBR"},
        {"schema_version": "fraud-transaction/9"},
        {"transactions_1h": 5, "transactions_24h": 2},
        {"amount_1h": "500", "amount_24h": "100"},
        {"unique_counterparties_24h": 9, "transactions_24h": 2},
        {"days_since_last_transaction": None, "transactions_24h": 3},
        {"account_age_days": -1},
        {"transaction_id": "  "},
        {"ip_risk_band": "SEVERE"},
    ],
)
def test_invalid_payloads_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        TransactionFeatures.model_validate(transaction_payload(**overrides))


def test_jpy_accepts_whole_units_only() -> None:
    TransactionFeatures.model_validate(transaction_payload(currency="JPY", amount="1500"))
    TransactionFeatures.model_validate(transaction_payload(currency="JPY", amount=10268.0))
    TransactionFeatures.model_validate(transaction_payload(currency="GBP", amount="12.500"))
    with pytest.raises(ValidationError):
        TransactionFeatures.model_validate(transaction_payload(currency="JPY", amount="1500.5"))


def test_float_amount_is_converted_exactly() -> None:
    txn = TransactionFeatures.model_validate(transaction_payload(amount=0.1))
    assert txn.amount == Decimal("0.1")


def test_extra_fields_are_ignored_and_country_upper_cased() -> None:
    txn = TransactionFeatures.model_validate(
        transaction_payload(country="gb", investigation_outcome="x")
    )
    assert txn.country == "GB"
    assert not hasattr(txn, "investigation_outcome")


def test_optional_fields_default() -> None:
    base = transaction_payload(transactions_24h=0, unique_counterparties_24h=0, amount_24h="0")
    minimal = {k: v for k, v in base.items() if k in REQUIRED_COLUMNS}
    txn = TransactionFeatures.model_validate(minimal)
    assert txn.counterparty_id is None
    assert txn.ip_risk_band == "UNKNOWN"
    assert txn.inflow_amount_24h == 0


def test_required_and_optional_columns_partition_fields() -> None:
    assert set(REQUIRED_COLUMNS).isdisjoint(OPTIONAL_COLUMNS)
    assert set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS) == set(TransactionFeatures.model_fields)
    assert LABEL_COLUMNS == ("is_fraud", "fraud_type")


def test_labelled_transaction_consistency() -> None:
    LabelledTransaction.model_validate(transaction_payload(is_fraud=True, fraud_type="STRUCTURING"))
    LabelledTransaction.model_validate(transaction_payload(is_fraud=False, fraud_type="NONE"))
    with pytest.raises(ValidationError):
        LabelledTransaction.model_validate(transaction_payload(is_fraud=True, fraud_type="NONE"))
    with pytest.raises(ValidationError):
        LabelledTransaction.model_validate(
            transaction_payload(is_fraud=False, fraud_type="MULE_ACTIVITY")
        )


def test_risk_policy_bands_and_decisions() -> None:
    policy = RiskPolicy(threshold=0.3, priority_threshold=0.8, low_upper=0.15)
    assert policy.band(0.0) is RiskBand.LOW
    assert policy.band(0.149) is RiskBand.LOW
    assert policy.band(0.15) is RiskBand.MEDIUM
    assert policy.band(0.2999) is RiskBand.MEDIUM
    assert policy.band(0.3) is RiskBand.HIGH
    assert policy.band(0.7999) is RiskBand.HIGH
    assert policy.band(0.8) is RiskBand.CRITICAL
    assert policy.band(1.0) is RiskBand.CRITICAL
    assert RiskPolicy.decision(RiskBand.LOW) is DecisionSupport.PASS
    assert RiskPolicy.decision(RiskBand.MEDIUM) is DecisionSupport.PASS
    assert RiskPolicy.decision(RiskBand.HIGH) is DecisionSupport.REVIEW
    assert RiskPolicy.decision(RiskBand.CRITICAL) is DecisionSupport.PRIORITY_REVIEW
    assert "BLOCK" not in {d.value for d in DecisionSupport}


def test_score_response_bounds() -> None:
    with pytest.raises(ValidationError):
        ScoreResponse(
            transaction_id="x", model_version="m", risk_probability=1.2, risk_band="LOW",
            decision_support="PASS", reason_codes=[], threshold=0.5, scored_at=datetime.now(tz=UTC),
        )  # fmt: skip


def test_money_round_trip() -> None:
    assert to_decimal(to_float(Decimal("1234.56")), "GBP") == Decimal("1234.56")
    assert to_decimal(0.1 + 0.2, "EUR") == Decimal("0.30")
    assert to_decimal("1500", "JPY") == Decimal("1500")
    assert to_decimal(Decimal("99.995"), "USD") == Decimal("100.00")
