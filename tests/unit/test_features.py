from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_ml.data.loader import normalise_frame
from fraud_ml.exceptions import LeakageError
from fraud_ml.features.behavioural import behavioural_features
from fraud_ml.features.definitions import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_SPECS,
    FORBIDDEN_COLUMNS,
    NUMERIC_FEATURES,
)
from fraud_ml.features.pipeline import build_feature_frame, build_preprocessor, guard_no_leakage
from fraud_ml.features.temporal import temporal_features, unusual_hour_score
from fraud_ml.features.transaction import (
    amount_band,
    amount_in_reference_units,
    near_threshold_flag,
    threshold_proximity,
    transaction_features,
)
from fraud_ml.features.velocity import velocity_features
from tests.conftest import transaction_payload


def frame_of(*rows: dict) -> pd.DataFrame:
    return normalise_frame(pd.DataFrame(list(rows)))


def test_feature_catalogue_is_consistent() -> None:
    assert len(FEATURE_SPECS) == len(ALL_FEATURES)
    assert set(NUMERIC_FEATURES) | set(BINARY_FEATURES) | set(CATEGORICAL_FEATURES) == set(
        ALL_FEATURES
    )
    assert not FORBIDDEN_COLUMNS & set(ALL_FEATURES)
    assert "transaction_id" in FORBIDDEN_COLUMNS and "is_fraud" in FORBIDDEN_COLUMNS


def test_amount_features() -> None:
    amount = pd.Series([10.0, 1500.0, 4_600.0, 20_000.0])
    currency = pd.Series(["GBP", "USD", "GBP", "JPY"])
    ref = amount_in_reference_units(amount, currency)
    assert ref.tolist() == pytest.approx([10.0, 1200.0, 4600.0, 110.0])
    assert amount_band(ref).tolist() == ["LT_25", "500_2K", "2K_10K", "100_500"]
    proximity = threshold_proximity(ref)
    assert proximity.tolist() == pytest.approx([0.01, 1200 / 5000, 0.92, 110 / 1000])
    assert near_threshold_flag(proximity).tolist() == [0, 0, 1, 0]
    assert threshold_proximity(pd.Series([50_000.0])).iloc[0] == 0.0


def test_transaction_features_fill_missing_categories() -> None:
    frame = frame_of(
        transaction_payload(beneficiary_country=None, merchant_category=None, customer_segment=None)
    )
    out = transaction_features(frame)
    assert out.loc[0, "beneficiary_country"] == "NONE"
    assert out.loc[0, "merchant_category"] == "NONE"
    assert out.loc[0, "customer_segment"] == "UNKNOWN"
    assert out.loc[0, "log_amount"] == pytest.approx(np.log1p(125.5))


def test_temporal_features() -> None:
    frame = frame_of(
        transaction_payload(transaction_timestamp="2025-06-28T03:30:00Z"),  # Saturday night
        transaction_payload(transaction_timestamp="2025-06-30T14:00:00+02:00"),  # Monday 12:00 UTC
    )
    out = temporal_features(frame)
    assert out["hour_of_day"].tolist() == [3, 12]
    assert out["day_of_week"].tolist() == [5, 0]
    assert out["is_weekend"].tolist() == [1, 0]
    assert out["is_night"].tolist() == [1, 0]
    assert out["unusual_hour_score"].tolist() == [1.0, 0.0]
    assert out.loc[1, "hour_cos"] == pytest.approx(-1.0)
    assert unusual_hour_score(pd.Series([5, 22, 6, 13])).tolist() == [0.75, 0.5, 0.25, 0.0]


def test_velocity_features() -> None:
    frame = frame_of(
        transaction_payload(transactions_1h=2, transactions_24h=4, amount_1h="50", amount_24h="300", inflow_amount_24h="1000", amount="600"),
        transaction_payload(transactions_1h=0, transactions_24h=0, amount_1h="0", amount_24h="0", inflow_amount_24h="0", unique_counterparties_24h=0, days_since_last_transaction=None),
    )  # fmt: skip
    out = velocity_features(frame)
    assert out.loc[0, "velocity_ratio_1h_24h"] == 0.5
    assert out.loc[1, "velocity_ratio_1h_24h"] == 0.0
    assert out.loc[0, "amount_vs_inflow_24h"] == pytest.approx(0.6)
    assert out.loc[1, "amount_vs_inflow_24h"] == 0.0
    assert out["inflow_present_24h"].tolist() == [1, 0]
    assert out.loc[0, "log1p_amount_24h"] == pytest.approx(np.log1p(300))


def test_behavioural_features() -> None:
    frame = frame_of(
        transaction_payload(amount="400", account_median_amount_30d="80", account_mean_amount_30d="100", beneficiary_country="FR", days_since_last_transaction=45.0, account_age_days=10, is_new_beneficiary=True, is_new_device=None),
        transaction_payload(account_median_amount_30d=None, account_mean_amount_30d=None, days_since_last_transaction=None, transactions_24h=0, unique_counterparties_24h=0, amount_24h="0"),
    )  # fmt: skip
    out = behavioural_features(frame)
    assert out.loc[0, "amount_vs_account_median"] == 5.0
    assert out.loc[0, "amount_vs_account_mean"] == 4.0
    assert out.loc[0, "cross_border_flag"] == 1 and out.loc[1, "cross_border_flag"] == 0
    assert out.loc[0, "dormant_flag"] == 1 and out.loc[1, "dormant_flag"] == 0
    assert out.loc[0, "new_account_flag"] == 1 and out.loc[1, "new_account_flag"] == 0
    assert out.loc[0, "new_beneficiary_flag"] == 1 and out.loc[0, "new_device_flag"] == 0
    assert np.isnan(out.loc[1, "amount_vs_account_median"])
    assert out.loc[1, "has_history"] == 0 and out.loc[0, "has_history"] == 1


def test_build_feature_frame_shape_and_guard(small_dataset: pd.DataFrame) -> None:
    x = small_dataset.drop(columns=["is_fraud", "fraud_type"]).head(200)
    features = build_feature_frame(x)
    assert list(features.columns) == list(ALL_FEATURES)
    assert features.index.equals(x.index)
    assert not FORBIDDEN_COLUMNS & set(features.columns)
    with pytest.raises(LeakageError):
        build_feature_frame(small_dataset.head(5))  # labels present
    with pytest.raises(LeakageError):
        build_feature_frame(x.head(5).assign(investigation_outcome="CONFIRMED"))
    with pytest.raises(LeakageError):
        guard_no_leakage(["log_amount", "is_fraud"])


def test_build_feature_frame_is_deterministic(small_dataset: pd.DataFrame) -> None:
    x = small_dataset.drop(columns=["is_fraud", "fraud_type"]).head(300)
    pd.testing.assert_frame_equal(build_feature_frame(x), build_feature_frame(x.copy()))


def test_preprocessor_handles_unknown_open_categories(small_dataset: pd.DataFrame) -> None:
    x = small_dataset.drop(columns=["is_fraud", "fraud_type"])
    pre = build_preprocessor(scale_numeric=True)
    train_features = build_feature_frame(x.head(2000))
    pre.fit(train_features)
    unseen = x.tail(3).copy()
    unseen["country"] = "ZZ"
    unseen["merchant_category"] = "NEVER_SEEN_BEFORE"
    transformed = pre.transform(build_feature_frame(unseen))
    assert transformed.shape[0] == 3
    assert not transformed.isna().any().any()
    assert "country_infrequent_sklearn" in transformed.columns


def test_preprocessor_output_has_no_identifier_columns(small_dataset: pd.DataFrame) -> None:
    x = small_dataset.drop(columns=["is_fraud", "fraud_type"]).head(1000)
    pre = build_preprocessor(scale_numeric=False)
    out = pre.fit_transform(build_feature_frame(x))
    assert not any(
        c.startswith(("transaction_id", "account_id", "counterparty_id")) for c in out.columns
    )
