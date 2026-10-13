"""Hypothesis property tests for the invariants the system must never violate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fraud_ml.config import CostModel, ThresholdPolicy
from fraud_ml.contracts.enums import Channel, Currency, IpRiskBand, TransactionType
from fraud_ml.contracts.money import to_decimal, to_float
from fraud_ml.contracts.score import RiskPolicy
from fraud_ml.contracts.transaction import TransactionFeatures
from fraud_ml.data.loader import normalise_frame
from fraud_ml.features.velocity import HISTORY_COLUMNS, compute_history_features
from fraud_ml.scoring.batch import score_frame_batch
from fraud_ml.scoring.scorer import Scorer
from fraud_ml.training.threshold import select_threshold, threshold_analysis
from tests.conftest import transaction_payload

COUNTRIES = ("GB", "US", "DE", "FR", "SG", "JP", "NG", "ZZ")
MONEY = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("5000000"), places=2)


@st.composite
def transactions(draw: st.DrawFn) -> dict:
    txn_24h = draw(st.integers(0, 40))
    txn_1h = draw(st.integers(0, txn_24h))
    amount_24h = draw(MONEY) if txn_24h else Decimal("0")
    amount_1h = (
        draw(st.decimals(min_value=Decimal("0"), max_value=amount_24h, places=2))
        if txn_1h
        else Decimal("0")
    )
    has_history = txn_24h > 0 or draw(st.booleans())
    currency = draw(st.sampled_from(list(Currency)))
    amount = draw(MONEY)
    if currency is Currency.JPY:
        amount = amount.quantize(Decimal("1"))
        amount = max(amount, Decimal("1"))
    return transaction_payload(
        transaction_id=f"TXN-SYN-{draw(st.integers(0, 10**8)):09d}",
        transaction_timestamp=(
            datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=draw(st.integers(0, 400_000)))
        ).isoformat(),
        amount=str(amount),
        currency=currency.value,
        transaction_type=draw(st.sampled_from(list(TransactionType))).value,
        channel=draw(st.sampled_from(list(Channel))).value,
        country=draw(st.sampled_from(COUNTRIES)),
        beneficiary_country=draw(st.sampled_from([*COUNTRIES, None])),
        counterparty_id=draw(st.sampled_from(["CP-SYN-1", "CP-SYN-NEW", None])),
        is_weekend=draw(st.booleans()),
        account_age_days=draw(st.integers(0, 5000)),
        transactions_1h=txn_1h,
        transactions_24h=txn_24h,
        amount_1h=str(amount_1h),
        amount_24h=str(amount_24h),
        unique_counterparties_24h=draw(st.integers(0, txn_24h)),
        days_since_last_transaction=draw(st.floats(0, 400)) if has_history else None,
        account_median_amount_30d=str(draw(MONEY)) if has_history and draw(st.booleans()) else None,
        account_mean_amount_30d=str(draw(MONEY)) if has_history and draw(st.booleans()) else None,
        inflow_amount_24h=str(
            draw(st.decimals(min_value=Decimal("0"), max_value=Decimal("100000"), places=2))
        ),
        merchant_category=draw(st.sampled_from(["GROCERY", "UNSEEN_CATEGORY", None])),
        ip_risk_band=draw(st.sampled_from(list(IpRiskBand))).value,
        customer_segment=draw(st.sampled_from(["RETAIL", "PREMIUM", "BUSINESS", None])),
        is_new_beneficiary=draw(st.sampled_from([True, False, None])),
        is_new_device=draw(st.sampled_from([True, False, None])),
    )


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(payload=transactions())
def test_risk_probability_always_in_unit_interval_and_deterministic(
    scorer: Scorer, payload: dict
) -> None:
    txn = TransactionFeatures.model_validate(payload)
    first = scorer.score_one(txn)
    second = scorer.score_one(txn)
    assert 0.0 <= first.risk_probability <= 1.0
    assert first.risk_probability == second.risk_probability
    assert first.risk_band == second.risk_band and first.reason_codes == second.reason_codes
    assert first.risk_band == scorer.policy.band(first.risk_probability)


@settings(max_examples=60, deadline=None)
@given(
    y=st.lists(st.integers(0, 1), min_size=5, max_size=200),
    seed=st.integers(0, 10_000),
)
def test_increasing_threshold_never_increases_review_count(y: list[int], seed: int) -> None:
    p = np.random.default_rng(seed).random(len(y))
    analysis = threshold_analysis(np.asarray(y), p, CostModel())
    flagged = (analysis["true_positives"] + analysis["false_positives"]).to_numpy()
    assert np.all(np.diff(flagged) <= 0)
    assert np.all(np.diff(analysis["threshold"].to_numpy()) > 0)


@settings(max_examples=60, deadline=None)
@given(
    y=st.lists(st.integers(0, 1), min_size=5, max_size=200),
    seed=st.integers(0, 10_000),
    recall=st.floats(0.05, 1.0),
    review=st.floats(0.01, 1.0),
)
def test_selected_threshold_lies_in_unit_interval(
    y: list[int], seed: int, recall: float, review: float
) -> None:
    p = np.random.default_rng(seed).random(len(y))
    analysis = threshold_analysis(np.asarray(y), p, CostModel())
    decision = select_threshold(
        analysis, ThresholdPolicy(recall_target=recall, max_review_rate=review)
    )
    assert 0.0 <= decision.threshold <= 1.0
    assert decision.threshold <= decision.priority_threshold <= 1.0
    assert 0.0 <= decision.low_upper <= decision.threshold
    policy = RiskPolicy(
        threshold=decision.threshold,
        priority_threshold=decision.priority_threshold,
        low_upper=decision.low_upper,
    )
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    bands = [order.index(policy.band(x).value) for x in np.linspace(0, 1, 21)]
    assert bands == sorted(bands)


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    rows=st.lists(transactions(), min_size=1, max_size=12),
    corrupt=st.lists(st.integers(0, 11), max_size=4),
)
def test_batch_scoring_row_count_equals_valid_rows(
    scorer: Scorer, rows: list[dict], corrupt: list[int]
) -> None:
    frame = normalise_frame(pd.DataFrame(rows))
    corrupted = {i for i in corrupt if i < len(frame)}
    for i in corrupted:
        frame.loc[i, "amount"] = -1.0
    scores, rejects = score_frame_batch(scorer, frame)
    assert len(scores) == len(frame) - len(corrupted)
    assert len(rejects) == len(corrupted)
    assert len(scores) + len(rejects) == len(frame)


@settings(max_examples=40, deadline=None)
@given(
    offsets=st.lists(st.integers(0, 60 * 24 * 10), min_size=2, max_size=25),
    amounts=st.lists(st.floats(0.01, 10_000), min_size=25, max_size=25),
    seed=st.integers(0, 1000),
)
def test_history_features_never_use_future_timestamps(
    offsets: list[int], amounts: list[float], seed: int
) -> None:
    rng = np.random.default_rng(seed)
    n = len(offsets)
    log = pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(n)],
            "account_id": rng.choice(["A", "B"], size=n),
            "transaction_timestamp": pd.to_datetime(
                [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=o) for o in offsets], utc=True
            ),
            "amount": amounts[:n],
            "transaction_type": rng.choice(["TRANSFER", "DEPOSIT"], size=n),
            "counterparty_id": rng.choice(["CP1", "CP2", None], size=n),
        }
    )
    base = compute_history_features(log)
    ordered = log.sort_values(["transaction_timestamp", "transaction_id"], kind="mergesort")
    cutoff = ordered.index[n // 2]
    cutoff_ts = log.loc[cutoff, "transaction_timestamp"]
    future = log["transaction_timestamp"] > cutoff_ts
    mutated = log.copy()
    mutated.loc[future, "amount"] = 1e9
    mutated.loc[future, "counterparty_id"] = "CP-FUTURE"
    mutated.loc[future, "transaction_type"] = "DEPOSIT"
    again = compute_history_features(mutated)
    past = ~future
    pd.testing.assert_frame_equal(
        base.loc[past, list(HISTORY_COLUMNS)], again.loc[past, list(HISTORY_COLUMNS)]
    )


@settings(max_examples=200)
@given(
    amount=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("99999999"), places=2),
    currency=st.sampled_from([Currency.GBP, Currency.USD, Currency.EUR]),
)
def test_decimal_amount_round_trip_preserves_value(amount: Decimal, currency: Currency) -> None:
    assert to_decimal(to_float(amount), currency) == amount


@settings(max_examples=100)
@given(amount=st.decimals(min_value=Decimal("1"), max_value=Decimal("99999999"), places=0))
def test_jpy_round_trip(amount: Decimal) -> None:
    assert to_decimal(to_float(amount), Currency.JPY) == amount
