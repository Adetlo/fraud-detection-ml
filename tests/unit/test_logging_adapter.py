from __future__ import annotations

import json

import pandas as pd
import pytest

from fraud_ml.adapters.txn_pipeline import adapt_curated_frame
from fraud_ml.data.validation import validate_frame
from fraud_ml.exceptions import DataValidationError
from fraud_ml.logging import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    unbind_context,
)
from tests.conftest import NOW


def test_json_logging_includes_context(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    bind_context(correlation_id="abc", model_version="m/1", ignored=None)
    get_logger("test").info("event.happened", duration_ms=3)
    unbind_context("model_version")
    get_logger("test").info("second")
    clear_context()
    lines = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    first, second = lines[-2], lines[-1]
    assert first["event"] == "event.happened" and first["correlation_id"] == "abc"
    assert first["model_version"] == "m/1" and first["duration_ms"] == 3 and "timestamp" in first
    assert "model_version" not in second and second["level"] == "info"
    configure_logging("INFO", "console")


def _curated(n: int = 6) -> pd.DataFrame:
    base = pd.Timestamp("2025-05-01T09:00:00Z")
    return pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i}" for i in range(n)],
            "schema_version": "financial-transaction/1",
            "source_system": "SYNTHETIC_CORE",
            "account_id": ["ACC-1", "ACC-1", "ACC-2", "ACC-1", "ACC-2", "ACC-1"][:n],
            "counterparty_id": ["CP-1", "CP-2", None, "CP-1", "CP-3", "CP-9"][:n],
            "transaction_type": [
                "TRANSFER",
                "FEE",
                "DEPOSIT",
                "INTEREST",
                "PAYMENT",
                "CARD_PURCHASE",
            ][:n],
            "amount": [100.0, 2.5, 50.0, 1.0, 20.0, 15.0][:n],
            "currency": ["GBP"] * n,
            "transaction_timestamp": [base + pd.Timedelta(hours=i) for i in range(n)],
            "channel": ["ONLINE", "BATCH", "BRANCH", "BATCH", "MOBILE", "CARD"][:n],
            "country": ["GB"] * n,
            "status": ["POSTED", "POSTED", "POSTED", "CANCELLED", "POSTED", "POSTED"][:n],
            "merchant_category": [None, None, None, None, None, "GROCERY"][:n],
            "beneficiary_country": ["GB", "GB", None, None, "FR", "GB"][:n],
        }
    )


def test_adapter_maps_curated_output_to_contract() -> None:
    adapted = adapt_curated_frame(_curated())
    assert len(adapted) == 5  # cancelled row dropped
    assert set(adapted["transaction_type"]) <= {"TRANSFER", "PAYMENT", "DEPOSIT", "CARD_PURCHASE"}
    assert (adapted["schema_version"] == "fraud-transaction/1").all()
    assert validate_frame(adapted, now=NOW).n_invalid == 0
    acc1 = adapted[adapted["account_id"] == "ACC-1"].sort_values("transaction_timestamp")
    assert acc1["transactions_24h"].tolist() == [0, 1, 2]
    assert adapted["account_age_days"].max() == 0  # approximated from first observation


def test_adapter_errors() -> None:
    with pytest.raises(DataValidationError, match="missing columns"):
        adapt_curated_frame(_curated().drop(columns=["amount"]))
    with pytest.raises(DataValidationError, match="no scorable rows"):
        adapt_curated_frame(_curated().assign(status="PENDING"))
    assert len(adapt_curated_frame(_curated().assign(status="PENDING"), only_posted=False)) == 6
