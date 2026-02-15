"""Shared fixtures: a small deterministic dataset and one trained artifact per session."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

os.environ.setdefault("FRAUD_ML_MLFLOW_ENABLED", "false")
os.environ.setdefault("FRAUD_ML_LOG_FORMAT", "json")
os.environ.setdefault("FRAUD_ML_LOG_LEVEL", "WARNING")

from fraud_ml.config import Settings
from fraud_ml.data.loader import sha256_frame, write_frame
from fraud_ml.data.synthetic import GeneratorConfig, generate_dataset
from fraud_ml.models.gradient_boosting import GradientBoostingParams
from fraud_ml.scoring.scorer import Scorer
from fraud_ml.training.train import TrainingConfig, TrainingResult, train_model

SMALL_ROWS = 6_000
SMALL_FRAUD_RATE = 0.02
SMALL_SEED = 11


@pytest.fixture(scope="session")
def small_dataset() -> pd.DataFrame:
    frame, _ = generate_dataset(
        GeneratorConfig(
            transactions=SMALL_ROWS, fraud_rate=SMALL_FRAUD_RATE, seed=SMALL_SEED, days=120
        )
    )
    return frame


@pytest.fixture(scope="session")
def fast_config() -> TrainingConfig:
    return TrainingConfig(
        seed=SMALL_SEED,
        gradient_boosting=GradientBoostingParams(
            n_estimators=80, early_stopping_rounds=10, n_jobs=1
        ),
    )


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("workspace")


@pytest.fixture(scope="session")
def dataset_path(workspace: Path, small_dataset: pd.DataFrame) -> Path:
    return write_frame(small_dataset, workspace / "data" / "transactions.parquet")


@pytest.fixture(scope="session")
def trained(
    workspace: Path, small_dataset: pd.DataFrame, fast_config: TrainingConfig
) -> TrainingResult:
    return train_model(
        small_dataset,
        fast_config,
        model_root=workspace / "models",
        reports_dir=workspace / "reports",
        data_name="transactions.parquet",
        data_version=sha256_frame(small_dataset),
    )


@pytest.fixture(scope="session")
def model_root(trained: TrainingResult) -> Path:
    return trained.artifact_path.parent


@pytest.fixture(scope="session")
def scorer(model_root: Path) -> Scorer:
    return Scorer.from_path(model_root / "latest")


@pytest.fixture(scope="session")
def settings(model_root: Path, workspace: Path) -> Settings:
    return Settings(
        model_dir=model_root,
        data_dir=workspace / "data",
        reports_dir=workspace / "reports",
        mlflow_enabled=False,
        log_format="json",
        log_level="WARNING",
        api_max_batch_size=50,
        api_max_request_bytes=200_000,
    )


@pytest.fixture
def unlabelled(small_dataset: pd.DataFrame) -> pd.DataFrame:
    return small_dataset.drop(columns=["is_fraud", "fraud_type"]).tail(500).reset_index(drop=True)


def transaction_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "fraud-transaction/1",
        "transaction_id": "TXN-SYN-TEST-0001",
        "account_id": "ACC-SYN-000123",
        "transaction_timestamp": "2025-06-30T14:05:00Z",
        "amount": "125.50",
        "currency": "GBP",
        "transaction_type": "TRANSFER",
        "channel": "MOBILE",
        "country": "GB",
        "counterparty_id": "CP-SYN-000123-01",
        "beneficiary_country": "GB",
        "is_weekend": False,
        "account_age_days": 820,
        "transactions_1h": 0,
        "transactions_24h": 2,
        "amount_1h": "0",
        "amount_24h": "240.10",
        "unique_counterparties_24h": 2,
        "days_since_last_transaction": 0.4,
        "account_median_amount_30d": "80.00",
        "account_mean_amount_30d": "95.00",
        "inflow_amount_24h": "0",
        "merchant_category": None,
        "device_id_hash": "DEV-SYN-000123-0",
        "ip_risk_band": "LOW",
        "customer_segment": "RETAIL",
        "is_new_beneficiary": False,
        "is_new_device": False,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def payload() -> dict[str, Any]:
    return transaction_payload()


NOW = datetime(2025, 7, 1, tzinfo=UTC)
