"""Reason codes: rule-based risk indicators attached to every score.

Reason codes answer "which known risk signals are present in this transaction". They are
computed from the input features with the documented thresholds below, independently of the
model, so they are stable, auditable and understandable by a reviewer. They are **not** causal
explanations of the model output: a transaction can score HIGH with no reason code (the model
found an interaction the rules do not cover) or LOW with several (the model judged the signals
benign in context). Model-side attributions are reported separately as feature importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_ml.contracts.enums import IpRiskBand, ReasonCode
from fraud_ml.features.behavioural import DORMANT_DAYS, NEW_ACCOUNT_DAYS
from fraud_ml.features.temporal import NIGHT_HOURS
from fraud_ml.features.transaction import (
    NEAR_THRESHOLD_FRACTION,
    amount_in_reference_units,
    threshold_proximity,
)

#: Documented indicator thresholds (``docs/EXPLAINABILITY.md``).
HIGH_AMOUNT_VS_HISTORY_RATIO = 4.0
HIGH_VELOCITY_1H_COUNT = 3
HIGH_VELOCITY_24H_COUNT = 10
MANY_COUNTERPARTIES_24H_COUNT = 5
RAPID_OUTFLOW_MIN_SHARE = 0.5
MAX_REASON_CODES = 5

#: Emission order: the first codes are the ones a reviewer should see first.
PRIORITY: tuple[ReasonCode, ...] = (
    ReasonCode.HIGH_AMOUNT_VS_HISTORY,
    ReasonCode.RAPID_OUTFLOW_AFTER_INFLOW,
    ReasonCode.HIGH_VELOCITY_1H,
    ReasonCode.MANY_COUNTERPARTIES_24H,
    ReasonCode.NEW_BENEFICIARY,
    ReasonCode.DORMANT_ACCOUNT_REACTIVATED,
    ReasonCode.CROSS_BORDER,
    ReasonCode.NEAR_THRESHOLD_AMOUNT,
    ReasonCode.HIGH_RISK_IP,
    ReasonCode.NEW_DEVICE,
    ReasonCode.UNUSUAL_HOUR,
    ReasonCode.HIGH_VELOCITY_24H,
    ReasonCode.NEW_ACCOUNT,
)


def indicator_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Boolean column per reason code, computed from contract columns."""
    amount = frame["amount"].astype("float64")
    median = frame["account_median_amount_30d"].astype("float64")
    inflow = frame["inflow_amount_24h"].astype("float64").fillna(0.0)
    days_since = frame["days_since_last_transaction"].astype("float64")
    country = frame["country"].astype("string")
    beneficiary = frame["beneficiary_country"].astype("string")
    hour = frame["transaction_timestamp"].dt.tz_convert("UTC").dt.hour
    proximity = threshold_proximity(amount_in_reference_units(amount, frame["currency"]))
    new_beneficiary = frame["is_new_beneficiary"].astype("boolean").fillna(False).astype(bool)
    new_device = frame["is_new_device"].astype("boolean").fillna(False).astype(bool)
    indicators = {
        ReasonCode.HIGH_AMOUNT_VS_HISTORY: (
            (amount / median.where(median > 0)) >= HIGH_AMOUNT_VS_HISTORY_RATIO
        ).fillna(False),
        ReasonCode.RAPID_OUTFLOW_AFTER_INFLOW: (inflow > 0)
        & (amount >= RAPID_OUTFLOW_MIN_SHARE * inflow),
        ReasonCode.HIGH_VELOCITY_1H: frame["transactions_1h"].astype("int64")
        >= HIGH_VELOCITY_1H_COUNT,
        ReasonCode.MANY_COUNTERPARTIES_24H: frame["unique_counterparties_24h"].astype("int64")
        >= MANY_COUNTERPARTIES_24H_COUNT,
        ReasonCode.NEW_BENEFICIARY: new_beneficiary,
        ReasonCode.DORMANT_ACCOUNT_REACTIVATED: (days_since >= DORMANT_DAYS).fillna(False),
        ReasonCode.CROSS_BORDER: (beneficiary.notna() & (beneficiary != country)).fillna(False),
        ReasonCode.NEAR_THRESHOLD_AMOUNT: (proximity >= NEAR_THRESHOLD_FRACTION)
        & (proximity < 1.0),
        ReasonCode.HIGH_RISK_IP: frame["ip_risk_band"].astype("string") == IpRiskBand.HIGH,
        ReasonCode.NEW_DEVICE: new_device,
        ReasonCode.UNUSUAL_HOUR: hour.isin(NIGHT_HOURS),
        ReasonCode.HIGH_VELOCITY_24H: frame["transactions_24h"].astype("int64")
        >= HIGH_VELOCITY_24H_COUNT,
        ReasonCode.NEW_ACCOUNT: frame["account_age_days"].astype("int64") < NEW_ACCOUNT_DAYS,
    }
    matrix = pd.DataFrame(
        {code.value: np.asarray(v, dtype=bool) for code, v in indicators.items()}, index=frame.index
    )
    return matrix.loc[:, [code.value for code in PRIORITY]]


def reason_codes(frame: pd.DataFrame, limit: int = MAX_REASON_CODES) -> list[list[ReasonCode]]:
    matrix = indicator_matrix(frame).to_numpy()
    out: list[list[ReasonCode]] = []
    for row in matrix:
        codes = [PRIORITY[i] for i in np.flatnonzero(row)]
        out.append(codes[:limit])
    return out
