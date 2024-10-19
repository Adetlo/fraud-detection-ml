from __future__ import annotations

import pandas as pd
import pytest

from fraud_ml.contracts.enums import FraudType
from fraud_ml.data.split import temporal_split
from fraud_ml.data.synthetic import PATTERNS, GeneratorConfig, generate_dataset
from fraud_ml.data.validation import validate_frame
from tests.conftest import NOW


def test_temporal_split_preserves_order_and_fractions(small_dataset: pd.DataFrame) -> None:
    split = temporal_split(small_dataset, train_fraction=0.7, validation_fraction=0.15)
    n = len(small_dataset)
    assert len(split.train) == int(n * 0.7)
    assert len(split.validation) == int(n * 0.15)
    assert len(split.train) + len(split.validation) + len(split.test) == n
    assert (
        split.train["transaction_timestamp"].max()
        <= split.validation["transaction_timestamp"].min()
    )
    assert (
        split.validation["transaction_timestamp"].max() <= split.test["transaction_timestamp"].min()
    )
    assert split.train["transaction_timestamp"].is_monotonic_increasing
    summary = split.to_dict()
    assert set(summary) == {"train", "validation", "test"}
    assert summary["train"]["positives"] > 0


def test_temporal_split_rejects_bad_fractions(small_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        temporal_split(small_dataset, train_fraction=0.9, validation_fraction=0.2)
    with pytest.raises(ValueError):
        temporal_split(small_dataset, train_fraction=0.0, validation_fraction=0.2)
    with pytest.raises(ValueError):
        temporal_split(small_dataset.head(2))


def test_split_ignores_input_row_order(small_dataset: pd.DataFrame) -> None:
    shuffled = small_dataset.sample(frac=1.0, random_state=3)
    a = temporal_split(small_dataset)
    b = temporal_split(shuffled)
    assert a.train["transaction_id"].tolist() == b.train["transaction_id"].tolist()


def test_generator_is_reproducible_and_valid() -> None:
    cfg = GeneratorConfig(transactions=1500, fraud_rate=0.03, seed=5, days=60)
    a, summary_a = generate_dataset(cfg)
    b, _ = generate_dataset(cfg)
    pd.testing.assert_frame_equal(a, b)
    assert summary_a.transactions == 1500
    assert abs(summary_a.fraud_rate - 0.03) < 0.005
    assert validate_frame(a, require_labels=True, now=NOW).n_invalid == 0
    assert a["transaction_timestamp"].is_monotonic_increasing
    assert a["transaction_id"].str.startswith("TXN-SYN-").all()
    assert a["account_id"].str.startswith("ACC-SYN-").all()
    assert a["counterparty_id"].dropna().str.startswith("CP-SYN-").all()
    assert a["transaction_id"].is_unique
    assert set(a.loc[a["is_fraud"], "fraud_type"]).issubset({t.value for t in FraudType} - {"NONE"})
    assert (a.loc[~a["is_fraud"], "fraud_type"] == "NONE").all()


def test_generator_seed_changes_output() -> None:
    a, _ = generate_dataset(GeneratorConfig(transactions=800, seed=1, days=30))
    b, _ = generate_dataset(GeneratorConfig(transactions=800, seed=2, days=30))
    assert not a["amount"].equals(b["amount"])


def test_generator_covers_every_pattern_type() -> None:
    frame, summary = generate_dataset(
        GeneratorConfig(transactions=20_000, fraud_rate=0.02, seed=3, days=120)
    )
    expected = {fraud_type.value for _, fraud_type, _ in PATTERNS}
    assert expected.issubset(summary.fraud_type_counts)
    assert frame["currency"].nunique() >= 5
    assert frame["channel"].nunique() >= 5
    assert frame["transaction_type"].nunique() >= 5


def test_generator_config_validation() -> None:
    with pytest.raises(ValueError):
        GeneratorConfig(transactions=10)
    with pytest.raises(ValueError):
        GeneratorConfig(fraud_rate=0.9)
    with pytest.raises(ValueError):
        GeneratorConfig(days=2)
    assert GeneratorConfig(transactions=1000, accounts=5).n_accounts == 10


def test_zero_fraud_rate() -> None:
    frame, summary = generate_dataset(
        GeneratorConfig(transactions=500, fraud_rate=0.0, seed=9, days=20)
    )
    assert summary.fraud_rows == 0
    assert len(frame) == 500
