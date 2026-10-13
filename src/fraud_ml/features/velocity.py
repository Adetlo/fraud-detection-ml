"""Velocity features and the reference point-in-time history computation.

``compute_history_features`` turns an ordered transaction log into the contract's context
fields. For the transaction at position *i* within an account, every value is computed from
rows at positions ``< i`` only; the row itself and everything after it are excluded. This is
the single most important leakage control in the project and it is tested by mutating future
rows and asserting earlier features do not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_ml.contracts.enums import TransactionType

HOUR = 3_600.0
DAY = 86_400.0
WINDOW_1H = HOUR
WINDOW_24H = DAY
WINDOW_30D = 30 * DAY
INFLOW_TYPES: frozenset[str] = frozenset({TransactionType.DEPOSIT, TransactionType.REFUND})

HISTORY_COLUMNS: tuple[str, ...] = (
    "is_weekend",
    "account_age_days",
    "transactions_1h",
    "transactions_24h",
    "amount_1h",
    "amount_24h",
    "unique_counterparties_24h",
    "days_since_last_transaction",
    "account_median_amount_30d",
    "account_mean_amount_30d",
    "inflow_amount_24h",
    "is_new_beneficiary",
    "is_new_device",
)


def _account_history(group: pd.DataFrame, open_ts: float | None) -> pd.DataFrame:
    ts = group["transaction_timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
    amount = group["amount"].to_numpy(dtype="float64")
    inflow = np.where(group["transaction_type"].astype("string").isin(INFLOW_TYPES), amount, 0.0)
    n = len(group)
    cum_amount = np.concatenate([[0.0], np.cumsum(amount)])
    cum_inflow = np.concatenate([[0.0], np.cumsum(inflow)])
    positions = np.arange(n)

    start_1h = np.searchsorted(ts, ts - WINDOW_1H, side="right")
    start_24h = np.searchsorted(ts, ts - WINDOW_24H, side="right")
    start_30d = np.searchsorted(ts, ts - WINDOW_30D, side="right")
    # Window rows are those at positions [start, i): strictly before the current row.
    transactions_1h = positions - start_1h
    transactions_24h = positions - start_24h
    amount_1h = cum_amount[positions] - cum_amount[start_1h]
    amount_24h = cum_amount[positions] - cum_amount[start_24h]
    inflow_24h = cum_inflow[positions] - cum_inflow[start_24h]

    median_30d = np.full(n, np.nan)
    mean_30d = np.full(n, np.nan)
    for i in range(n):
        window = amount[start_30d[i] : i]
        if window.size:
            median_30d[i] = float(np.median(window))
            mean_30d[i] = float(window.mean())

    counterparties = group["counterparty_id"].astype(object).to_numpy()
    devices = (
        group["device_id_hash"].astype(object).to_numpy()
        if "device_id_hash" in group
        else np.full(n, None)
    )
    unique_24h = np.zeros(n, dtype="int64")
    new_beneficiary = np.zeros(n, dtype=bool)
    new_device = np.zeros(n, dtype=bool)
    window_counts: dict[object, int] = {}
    seen_counterparties: set[object] = set()
    seen_devices: set[object] = set()
    left = 0
    for i in range(n):
        while left < start_24h[i]:
            cp = counterparties[left]
            if cp is not None and not (isinstance(cp, float) and np.isnan(cp)):
                window_counts[cp] -= 1
                if window_counts[cp] == 0:
                    del window_counts[cp]
            left += 1
        unique_24h[i] = len(window_counts)
        cp = counterparties[i]
        cp_present = cp is not None and not (isinstance(cp, float) and np.isnan(cp))
        new_beneficiary[i] = cp_present and cp not in seen_counterparties
        dev = devices[i]
        dev_present = dev is not None and not (isinstance(dev, float) and np.isnan(dev))
        new_device[i] = dev_present and dev not in seen_devices
        if cp_present:
            window_counts[cp] = window_counts.get(cp, 0) + 1
            seen_counterparties.add(cp)
        if dev_present:
            seen_devices.add(dev)

    days_since_last = np.full(n, np.nan)
    if n > 1:
        days_since_last[1:] = (ts[1:] - ts[:-1]) / DAY
    first_seen = ts[0] if open_ts is None else open_ts
    account_age_days = np.floor((ts - first_seen) / DAY).clip(min=0).astype("int64")
    weekday = group["transaction_timestamp"].dt.tz_convert("UTC").dt.dayofweek.to_numpy()

    return pd.DataFrame(
        {
            "is_weekend": weekday >= 5,
            "account_age_days": account_age_days,
            "transactions_1h": transactions_1h,
            "transactions_24h": transactions_24h,
            "amount_1h": amount_1h,
            "amount_24h": amount_24h,
            "unique_counterparties_24h": unique_24h,
            "days_since_last_transaction": days_since_last,
            "account_median_amount_30d": median_30d,
            "account_mean_amount_30d": mean_30d,
            "inflow_amount_24h": inflow_24h,
            "is_new_beneficiary": new_beneficiary,
            "is_new_device": new_device,
        },
        index=group.index,
    )


def compute_history_features(log: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time context columns to a transaction log.

    Required columns: ``account_id``, ``transaction_id``, ``transaction_timestamp`` (UTC),
    ``amount``, ``transaction_type``, ``counterparty_id``. Optional: ``device_id_hash``,
    ``account_open_date`` (else account age counts from the first observed transaction).
    Returns a copy in the original row order with ``HISTORY_COLUMNS`` added or replaced.
    """
    ordered = log.sort_values(
        ["account_id", "transaction_timestamp", "transaction_id"], kind="mergesort"
    )
    pieces: list[pd.DataFrame] = []
    for _, group in ordered.groupby("account_id", sort=False):
        open_ts: float | None = None
        if "account_open_date" in group.columns:
            opened = pd.Timestamp(group["account_open_date"].iloc[0])
            if opened.tzinfo is None:
                opened = opened.tz_localize("UTC")
            open_ts = opened.value / 1e9
        pieces.append(_account_history(group, open_ts))
    history = pd.concat(pieces).loc[log.index]
    result = log.copy()
    for column in HISTORY_COLUMNS:
        result[column] = history[column].to_numpy()
    return result


def velocity_features(frame: pd.DataFrame) -> pd.DataFrame:
    txn_1h = frame["transactions_1h"].astype("float64")
    txn_24h = frame["transactions_24h"].astype("float64")
    amount = frame["amount"].astype("float64")
    inflow = frame["inflow_amount_24h"].astype("float64").fillna(0.0)
    return pd.DataFrame(
        {
            "transactions_1h": txn_1h,
            "transactions_24h": txn_24h,
            "log1p_amount_1h": np.log1p(frame["amount_1h"].astype("float64").clip(lower=0.0)),
            "log1p_amount_24h": np.log1p(frame["amount_24h"].astype("float64").clip(lower=0.0)),
            "unique_counterparties_24h": frame["unique_counterparties_24h"].astype("float64"),
            "velocity_ratio_1h_24h": txn_1h / txn_24h.clip(lower=1.0),
            "amount_vs_inflow_24h": np.where(inflow > 0, amount / inflow.replace(0.0, np.nan), 0.0),
            "inflow_present_24h": (inflow > 0).astype("int8"),
        },
        index=frame.index,
    )
