"""Transaction-level features (amount, structuring proximity)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_ml.contracts.money import CURRENCY_TO_REFERENCE

#: Reference-unit edges for ``amount_band``.
AMOUNT_BAND_EDGES: tuple[float, ...] = (0.0, 25.0, 100.0, 500.0, 2_000.0, 10_000.0, float("inf"))
AMOUNT_BAND_LABELS: tuple[str, ...] = ("LT_25", "25_100", "100_500", "500_2K", "2K_10K", "GTE_10K")

#: Round reporting-style thresholds (reference units) used for the structuring proximity signal.
ROUND_THRESHOLDS: tuple[float, ...] = (1_000.0, 5_000.0, 10_000.0)
NEAR_THRESHOLD_FRACTION = 0.85


def amount_in_reference_units(amount: pd.Series, currency: pd.Series) -> pd.Series:
    rates = currency.astype("string").map(CURRENCY_TO_REFERENCE).astype("float64")
    return amount.astype("float64") * rates


def amount_band(amount_ref: pd.Series) -> pd.Series:
    bands = pd.cut(
        amount_ref, bins=list(AMOUNT_BAND_EDGES), labels=list(AMOUNT_BAND_LABELS), right=False
    )
    return bands.astype("string").fillna("UNKNOWN")


def threshold_proximity(amount_ref: pd.Series) -> pd.Series:
    """``amount / next round threshold`` for amounts below the largest threshold, else 0."""
    values = amount_ref.to_numpy(dtype="float64")
    out = np.zeros_like(values)
    thresholds = np.asarray(ROUND_THRESHOLDS)
    idx = np.searchsorted(thresholds, values, side="right")
    within = idx < len(thresholds)
    out[within] = values[within] / thresholds[idx[within]]
    return pd.Series(out, index=amount_ref.index)


def near_threshold_flag(proximity: pd.Series) -> pd.Series:
    return ((proximity >= NEAR_THRESHOLD_FRACTION) & (proximity < 1.0)).astype("int8")


def transaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    amount_ref = amount_in_reference_units(frame["amount"], frame["currency"])
    proximity = threshold_proximity(amount_ref)
    return pd.DataFrame(
        {
            "log_amount": np.log1p(amount_ref.clip(lower=0.0)),
            "amount_band": amount_band(amount_ref),
            "threshold_proximity": proximity,
            "near_threshold_flag": near_threshold_flag(proximity),
            "currency": frame["currency"].astype("string"),
            "transaction_type": frame["transaction_type"].astype("string"),
            "channel": frame["channel"].astype("string"),
            "country": frame["country"].astype("string"),
            "beneficiary_country": frame["beneficiary_country"].astype("string").fillna("NONE"),
            "merchant_category": frame["merchant_category"].astype("string").fillna("NONE"),
            "ip_risk_band": frame["ip_risk_band"].astype("string").fillna("UNKNOWN"),
            "customer_segment": frame["customer_segment"].astype("string").fillna("UNKNOWN"),
        },
        index=frame.index,
    )
