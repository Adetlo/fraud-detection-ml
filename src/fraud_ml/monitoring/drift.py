"""Basic drift checks: PSI for numeric and categorical inputs, plus score distribution shift.

Population Stability Index (PSI) compares the binned distribution of a *current* sample with a
*reference* sample (typically the training data)::

    PSI = sum_i (current_i - reference_i) * ln(current_i / reference_i)

Numeric columns are binned by reference quantiles; categorical columns by category (rare
categories pooled). Conventional thresholds: < 0.10 stable, 0.10-0.25 warning, > 0.25 drift.
This is a deliberately simple, label-free method (``docs/DRIFT.md``); it says *that* an input
moved, not whether model performance changed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fraud_ml.config import DriftThresholds
from fraud_ml.contracts.enums import DriftStatus

EPSILON = 1e-4
MAX_CATEGORIES = 20

NUMERIC_DRIFT_COLUMNS: tuple[str, ...] = (
    "amount", "account_age_days", "transactions_1h", "transactions_24h", "amount_1h", "amount_24h",
    "unique_counterparties_24h", "days_since_last_transaction", "account_median_amount_30d",
    "account_mean_amount_30d", "inflow_amount_24h",
)  # fmt: skip
CATEGORICAL_DRIFT_COLUMNS: tuple[str, ...] = (
    "currency", "transaction_type", "channel", "country", "beneficiary_country",
    "merchant_category", "ip_risk_band", "customer_segment", "is_new_beneficiary", "is_new_device",
)  # fmt: skip


class FeatureDrift(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    kind: str
    psi: float
    status: DriftStatus
    reference_rows: int
    current_rows: int
    detail: dict[str, Any] = Field(default_factory=dict)


class DriftReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: DriftStatus
    features_checked: int
    features_warning: int
    features_drifted: int
    drifted_features: list[str]
    warning_features: list[str]
    score_drift: FeatureDrift | None
    thresholds: DriftThresholds
    features: list[FeatureDrift]

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "features_checked": self.features_checked,
            "features_warning": self.features_warning,
            "features_drifted": self.features_drifted,
            "drifted_features": self.drifted_features,
            "warning_features": self.warning_features,
            "score_drift": self.score_drift.model_dump(mode="json") if self.score_drift else None,
        }


def _psi(reference_share: np.ndarray, current_share: np.ndarray) -> float:
    r = np.clip(reference_share, EPSILON, None)
    c = np.clip(current_share, EPSILON, None)
    return float(np.sum((c - r) * np.log(c / r)))


def numeric_psi(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[float, dict[str, Any]]:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype="float64")
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(ref) == 0 or len(cur) == 0:
        return 0.0, {"note": "empty sample"}
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference: one bin around the constant, one on each side
        value = float(edges[0])
        eps = 1e-9 * max(1.0, abs(value))
        edges = np.asarray([-np.inf, value - eps, value + eps, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]
    psi = _psi(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())
    detail = {
        "bins": int(len(edges) - 1),
        "reference_mean": float(ref.mean()),
        "current_mean": float(cur.mean()),
        "reference_missing_rate": float(reference.isna().mean()),
        "current_missing_rate": float(current.isna().mean()),
    }
    return psi, detail


def categorical_psi(reference: pd.Series, current: pd.Series) -> tuple[float, dict[str, Any]]:
    ref = reference.astype("string").fillna("<missing>")
    cur = current.astype("string").fillna("<missing>")
    ref_freq = ref.value_counts(normalize=True)
    top = list(ref_freq.index[:MAX_CATEGORIES])
    categories = [*top, "<other>"]

    def pooled(freq: pd.Series) -> np.ndarray:
        shares = [float(freq.get(c, 0.0)) for c in top]
        shares.append(max(0.0, 1.0 - sum(shares)))
        return np.asarray(shares)

    cur_freq = cur.value_counts(normalize=True)
    ref_shares, cur_shares = pooled(ref_freq), pooled(cur_freq)
    psi = _psi(ref_shares, cur_shares)
    shifts = {
        c: round(float(cs - rs), 4)
        for c, rs, cs in zip(categories, ref_shares, cur_shares, strict=True)
    }
    largest = sorted(shifts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    new_categories = sorted(set(cur_freq.index) - set(ref_freq.index))[:10]
    return psi, {"largest_frequency_shifts": dict(largest), "new_categories": new_categories}


def _status(psi: float, thresholds: DriftThresholds) -> DriftStatus:
    if psi >= thresholds.psi_drift:
        return DriftStatus.DRIFT
    if psi >= thresholds.psi_warning:
        return DriftStatus.WARNING
    return DriftStatus.OK


def drift_check(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    thresholds: DriftThresholds | None = None,
    numeric_columns: tuple[str, ...] = NUMERIC_DRIFT_COLUMNS,
    categorical_columns: tuple[str, ...] = CATEGORICAL_DRIFT_COLUMNS,
    score_column: str | None = None,
) -> DriftReport:
    thresholds = thresholds or DriftThresholds()
    results: list[FeatureDrift] = []
    for column in numeric_columns:
        if column in reference.columns and column in current.columns:
            psi, detail = numeric_psi(reference[column], current[column], thresholds.bins)
            results.append(
                FeatureDrift(
                    feature=column,
                    kind="numeric",
                    psi=psi,
                    status=_status(psi, thresholds),
                    reference_rows=int(reference[column].notna().sum()),
                    current_rows=int(current[column].notna().sum()),
                    detail=detail,
                )
            )
    for column in categorical_columns:
        if column in reference.columns and column in current.columns:
            psi, detail = categorical_psi(reference[column], current[column])
            results.append(
                FeatureDrift(
                    feature=column,
                    kind="categorical",
                    psi=psi,
                    status=_status(psi, thresholds),
                    reference_rows=len(reference),
                    current_rows=len(current),
                    detail=detail,
                )
            )

    score_drift: FeatureDrift | None = None
    if score_column and score_column in reference.columns and score_column in current.columns:
        psi, detail = numeric_psi(reference[score_column], current[score_column], thresholds.bins)
        score_drift = FeatureDrift(
            feature=score_column,
            kind="score",
            psi=psi,
            status=_status(psi, thresholds),
            reference_rows=len(reference),
            current_rows=len(current),
            detail=detail,
        )

    drifted = [r.feature for r in results if r.status is DriftStatus.DRIFT]
    warning = [r.feature for r in results if r.status is DriftStatus.WARNING]
    checked = len(results)
    status = DriftStatus.OK
    if warning or drifted:
        status = DriftStatus.WARNING
    if checked and len(drifted) / checked >= thresholds.drifted_feature_fraction_for_drift:
        status = DriftStatus.DRIFT
    if score_drift is not None and score_drift.status is DriftStatus.DRIFT:
        status = DriftStatus.DRIFT
    elif (
        score_drift is not None
        and score_drift.status is DriftStatus.WARNING
        and status is DriftStatus.OK
    ):
        status = DriftStatus.WARNING
    return DriftReport(
        status=status,
        features_checked=checked,
        features_warning=len(warning),
        features_drifted=len(drifted),
        drifted_features=drifted,
        warning_features=warning,
        score_drift=score_drift,
        thresholds=thresholds,
        features=results,
    )
