"""Behavioural features: the transaction relative to the account's own history."""

from __future__ import annotations

import numpy as np
import pandas as pd

DORMANT_DAYS = 30.0
NEW_ACCOUNT_DAYS = 30


def behavioural_features(frame: pd.DataFrame) -> pd.DataFrame:
    amount = frame["amount"].astype("float64")
    median = frame["account_median_amount_30d"].astype("float64")
    mean = frame["account_mean_amount_30d"].astype("float64")
    days_since = frame["days_since_last_transaction"].astype("float64")
    age = frame["account_age_days"].astype("float64")
    country = frame["country"].astype("string")
    beneficiary = frame["beneficiary_country"].astype("string")
    new_beneficiary = frame["is_new_beneficiary"].astype("boolean").fillna(False)
    new_device = frame["is_new_device"].astype("boolean").fillna(False)
    return pd.DataFrame(
        {
            "amount_vs_account_median": amount / median.where(median > 0),
            "amount_vs_account_mean": amount / mean.where(mean > 0),
            "log1p_days_since_last": np.log1p(days_since.clip(lower=0.0)),
            "log1p_account_age_days": np.log1p(age.clip(lower=0.0)),
            "has_history": days_since.notna().astype("int8"),
            "new_beneficiary_flag": new_beneficiary.astype("int8"),
            "new_device_flag": new_device.astype("int8"),
            "cross_border_flag": (beneficiary.notna() & (beneficiary != country)).astype("int8"),
            "dormant_flag": (days_since >= DORMANT_DAYS).fillna(False).astype("int8"),
            "new_account_flag": (age < NEW_ACCOUNT_DAYS).astype("int8"),
        },
        index=frame.index,
    )
