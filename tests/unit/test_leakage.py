"""Explicit leakage controls (docs/FEATURE_ENGINEERING.md, section Leakage Controls)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_ml.contracts.transaction import LABEL_COLUMNS
from fraud_ml.data.split import temporal_split
from fraud_ml.exceptions import LeakageError
from fraud_ml.features.definitions import ALL_FEATURES, FORBIDDEN_COLUMNS, POST_OUTCOME_COLUMNS
from fraud_ml.features.pipeline import build_feature_frame, build_pipeline
from fraud_ml.models.baseline import BaselineParams, build_baseline
from fraud_ml.models.calibration import unwrap_pipeline
from fraud_ml.training.train import TrainingResult


def test_target_column_cannot_enter_feature_matrix(small_dataset: pd.DataFrame) -> None:
    for column in LABEL_COLUMNS:
        assert column in FORBIDDEN_COLUMNS
        assert column not in ALL_FEATURES
    with pytest.raises(LeakageError):
        build_feature_frame(small_dataset.head(10))


@pytest.mark.parametrize("column", POST_OUTCOME_COLUMNS)
def test_post_outcome_fields_are_rejected(small_dataset: pd.DataFrame, column: str) -> None:
    x = small_dataset.drop(columns=list(LABEL_COLUMNS)).head(10).assign(**{column: 1})
    with pytest.raises(LeakageError):
        build_feature_frame(x)


def test_identifiers_and_raw_timestamp_are_not_features() -> None:
    for column in (
        "transaction_id",
        "account_id",
        "counterparty_id",
        "device_id_hash",
        "transaction_timestamp",
    ):
        assert column not in ALL_FEATURES


def test_pipeline_rejects_label_in_training_frame(small_dataset: pd.DataFrame) -> None:
    pipeline = build_baseline(BaselineParams(max_iter=200), seed=0)
    with pytest.raises(LeakageError):
        pipeline.fit(small_dataset.head(300), small_dataset["is_fraud"].head(300).astype(int))


def test_encoders_are_fitted_on_training_split_only(
    trained: TrainingResult, small_dataset: pd.DataFrame
) -> None:
    from fraud_ml.models.registry import load_artifact

    pipeline = unwrap_pipeline(load_artifact(trained.artifact_path).model)
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat_open"]
    split = temporal_split(small_dataset, train_fraction=0.7, validation_fraction=0.15)
    train_countries = set(split.train["country"].astype(str))
    idx = list(encoder.feature_names_in_).index("country")
    learned = set(encoder.categories_[idx])
    assert learned.issubset(train_countries)
    only_later = (
        set(split.test["country"].astype(str)) | set(split.validation["country"].astype(str))
    ) - train_countries
    assert learned.isdisjoint(only_later)


def test_training_frames_do_not_overlap_in_time(small_dataset: pd.DataFrame) -> None:
    split = temporal_split(small_dataset)
    assert split.train["transaction_timestamp"].max() <= split.test["transaction_timestamp"].min()
    assert set(split.train["transaction_id"]).isdisjoint(split.test["transaction_id"])


def test_future_rows_do_not_change_earlier_feature_rows(small_dataset: pd.DataFrame) -> None:
    """Feature derivation is row-wise: appending or mutating later rows leaves earlier rows unchanged."""
    x = small_dataset.drop(columns=list(LABEL_COLUMNS)).head(400).reset_index(drop=True)
    base = build_feature_frame(x)
    mutated = x.copy()
    mutated.loc[200:, "amount"] = mutated.loc[200:, "amount"] * 1000
    mutated.loc[200:, "channel"] = "API"
    again = build_feature_frame(mutated)
    pd.testing.assert_frame_equal(base.iloc[:200], again.iloc[:200])


def test_scaler_statistics_come_from_training_split(small_dataset: pd.DataFrame) -> None:
    x = small_dataset.drop(columns=list(LABEL_COLUMNS))
    split = temporal_split(small_dataset)
    pipeline = build_pipeline(_Dummy(), scale_numeric=True)
    pipeline[:-1].fit(split.train.drop(columns=list(LABEL_COLUMNS)))
    scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scale"]
    train_features = build_feature_frame(split.train.drop(columns=list(LABEL_COLUMNS)))
    imputed = train_features[list(scaler.feature_names_in_)].fillna(
        train_features[list(scaler.feature_names_in_)].median()
    )
    np.testing.assert_allclose(scaler.mean_, imputed.mean().to_numpy(), rtol=1e-6)
    assert len(x) > len(split.train)


class _Dummy:
    def fit(self, *_: object) -> _Dummy:
        return self

    def get_params(self, deep: bool = True) -> dict:
        return {}
