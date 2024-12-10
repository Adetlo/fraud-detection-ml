"""The single feature pipeline shared by training and inference.

``build_feature_frame`` is a pure function from a contract frame to the model's feature
columns. It is embedded in the persisted scikit-learn ``Pipeline`` as a ``FunctionTransformer``
so that batch scoring, the API and training all run exactly the same code. The only *fitted*
state lives in the ``ColumnTransformer`` (imputation medians, scaler statistics, one-hot
vocabularies), and that state is learned from the training split alone.

Unknown categories: closed vocabularies (currency, channel, transaction type, IP band) are
enforced by the contract, so an unknown value is rejected with ``UNSUPPORTED_CATEGORY``
before it reaches the model. Open vocabularies (country, beneficiary country, merchant
category) are encoded with ``handle_unknown="infrequent_if_exist"``: values unseen in
training, or seen fewer than ``MIN_CATEGORY_FREQUENCY`` times, share one explicit
``infrequent`` bucket instead of crashing or being silently dropped.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fraud_ml.contracts.transaction import LABEL_COLUMNS
from fraud_ml.exceptions import LeakageError
from fraud_ml.features.behavioural import behavioural_features
from fraud_ml.features.definitions import (
    ALL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_COLUMNS,
    HIGH_CARDINALITY_CATEGORICAL,
    NUMERIC_FEATURES,
    POST_OUTCOME_COLUMNS,
)
from fraud_ml.features.temporal import temporal_features
from fraud_ml.features.transaction import transaction_features
from fraud_ml.features.velocity import velocity_features

MIN_CATEGORY_FREQUENCY = 20


def guard_no_leakage(columns: list[str] | tuple[str, ...] | pd.Index) -> None:
    """Raise if any forbidden column is about to be used as a model input."""
    leaked = sorted(FORBIDDEN_COLUMNS.intersection(columns))
    if leaked:
        raise LeakageError(f"forbidden columns in feature matrix: {leaked}")


#: Input columns whose mere presence in a scoring/training frame is treated as leakage.
GUARDED_INPUT_COLUMNS: frozenset[str] = frozenset(LABEL_COLUMNS + POST_OUTCOME_COLUMNS)


def guard_input_frame(columns: list[str] | tuple[str, ...] | pd.Index) -> None:
    """Raise if label or post-outcome columns are handed to the feature step."""
    leaked = sorted(GUARDED_INPUT_COLUMNS.intersection(columns))
    if leaked:
        raise LeakageError(
            f"label/post-outcome columns must be dropped before feature building: {leaked}"
        )


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Contract frame -> feature frame with exactly ``ALL_FEATURES`` columns, same index."""
    guard_input_frame(frame.columns)
    parts = [
        transaction_features(frame),
        temporal_features(frame),
        velocity_features(frame),
        behavioural_features(frame),
    ]
    features = pd.concat(parts, axis=1)
    features = features.loc[:, list(ALL_FEATURES)]
    guard_no_leakage(features.columns)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(object)
    return features


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    binary = SimpleImputer(strategy="most_frequent")
    closed = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype="float64")
    open_vocab = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_CATEGORY_FREQUENCY,
        sparse_output=False,
        dtype="float64",
    )
    closed_columns = [c for c in CATEGORICAL_FEATURES if c not in HIGH_CARDINALITY_CATEGORICAL]
    transformer = ColumnTransformer(
        [
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("bin", binary, list(BINARY_FEATURES)),
            ("cat", closed, closed_columns),
            ("cat_open", open_vocab, list(HIGH_CARDINALITY_CATEGORICAL)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer


def build_pipeline(estimator: Any, *, scale_numeric: bool) -> Pipeline:
    """Feature derivation -> fitted preprocessing -> estimator, as one persisted object."""
    return Pipeline(
        [
            ("features", FunctionTransformer(build_feature_frame, validate=False)),
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
