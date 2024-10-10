"""Point-in-time history features: correctness on a hand-built log, no future access."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from fraud_ml.features.velocity import HISTORY_COLUMNS, compute_history_features

T0 = datetime(2025, 3, 1, 10, 0, tzinfo=UTC)


def _log(rows: list[tuple[str, timedelta, float, str, str | None, str | None]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "transaction_id": f"TXN-{i:03d}",
                "account_id": account,
                "transaction_timestamp": T0 + offset,
                "amount": amount,
                "transaction_type": txn_type,
                "counterparty_id": counterparty,
                "device_id_hash": device,
            }
            for i, (account, offset, amount, txn_type, counterparty, device) in enumerate(rows)
        ]
    )
    frame["transaction_timestamp"] = pd.to_datetime(frame["transaction_timestamp"], utc=True)
    return frame


def test_history_values_on_small_log() -> None:
    log = _log(
        [
            ("A", timedelta(0), 100.0, "TRANSFER", "CP1", "D1"),
            ("A", timedelta(minutes=30), 50.0, "DEPOSIT", None, "D1"),
            ("A", timedelta(minutes=50), 20.0, "PAYMENT", "CP2", "D2"),
            ("A", timedelta(hours=5), 10.0, "TRANSFER", "CP1", "D1"),
            ("A", timedelta(days=40), 999.0, "TRANSFER", "CP3", "D1"),
            ("B", timedelta(hours=1), 5.0, "TRANSFER", "CP1", None),
        ]
    )
    out = compute_history_features(log)
    a = out[out["account_id"] == "A"].reset_index(drop=True)
    assert a["transactions_1h"].tolist() == [0, 1, 2, 0, 0]
    assert a["transactions_24h"].tolist() == [0, 1, 2, 3, 0]
    assert a["amount_1h"].tolist() == pytest.approx([0.0, 100.0, 150.0, 0.0, 0.0])
    assert a["amount_24h"].tolist() == pytest.approx([0.0, 100.0, 150.0, 170.0, 0.0])
    assert a["inflow_amount_24h"].tolist() == pytest.approx([0.0, 0.0, 50.0, 50.0, 0.0])
    assert a["unique_counterparties_24h"].tolist() == [0, 1, 1, 2, 0]
    assert a["is_new_beneficiary"].tolist() == [True, False, True, False, True]
    assert a["is_new_device"].tolist() == [True, False, True, False, False]
    assert np.isnan(a.loc[0, "days_since_last_transaction"])
    assert a.loc[1, "days_since_last_transaction"] == pytest.approx(30 / 1440)
    assert a.loc[4, "days_since_last_transaction"] == pytest.approx(40 - 5 / 24)
    assert np.isnan(a.loc[0, "account_median_amount_30d"])
    assert a.loc[3, "account_median_amount_30d"] == pytest.approx(50.0)
    assert a.loc[3, "account_mean_amount_30d"] == pytest.approx(170 / 3)
    assert np.isnan(a.loc[4, "account_median_amount_30d"])  # 30-day window has expired
    assert a["account_age_days"].tolist() == [0, 0, 0, 0, 40]
    b = out[out["account_id"] == "B"].iloc[0]
    assert b["transactions_24h"] == 0 and b["is_new_beneficiary"]
    assert not b["is_new_device"]


def test_history_uses_account_open_date_when_present() -> None:
    log = _log([("A", timedelta(0), 1.0, "TRANSFER", "CP1", None)])
    log["account_open_date"] = pd.Timestamp(T0 - timedelta(days=100))
    assert compute_history_features(log)["account_age_days"].iloc[0] == 100


def test_history_preserves_input_order_and_columns() -> None:
    log = _log(
        [
            ("B", timedelta(hours=2), 1.0, "TRANSFER", "CP1", None),
            ("A", timedelta(0), 1.0, "TRANSFER", "CP1", None),
            ("A", timedelta(hours=1), 1.0, "TRANSFER", "CP1", None),
        ]
    )
    out = compute_history_features(log)
    assert out["transaction_id"].tolist() == log["transaction_id"].tolist()
    assert set(HISTORY_COLUMNS).issubset(out.columns)


def test_history_never_uses_future_rows() -> None:
    log = _log(
        [
            ("A", timedelta(0), 100.0, "TRANSFER", "CP1", "D1"),
            ("A", timedelta(minutes=10), 10.0, "TRANSFER", "CP2", "D1"),
            ("A", timedelta(minutes=20), 10.0, "DEPOSIT", None, "D1"),
            ("A", timedelta(hours=3), 10.0, "TRANSFER", "CP9", "D9"),
        ]
    )
    base = compute_history_features(log)
    mutated = log.copy()
    mutated.loc[3, "amount"] = 1e6
    mutated.loc[3, "counterparty_id"] = "CP1"
    mutated.loc[3, "transaction_type"] = "DEPOSIT"
    mutated.loc[2, "amount"] = 5e5
    changed = compute_history_features(mutated)
    earlier = list(HISTORY_COLUMNS)
    pd.testing.assert_frame_equal(base.loc[:1, earlier], changed.loc[:1, earlier])
    # The mutated row itself only sees its own past, never its own amount.
    assert changed.loc[3, "amount_24h"] == pytest.approx(100.0 + 10.0 + 5e5)
