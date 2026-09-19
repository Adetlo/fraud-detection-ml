# Fraud Detection ML

An educational fraud risk scoring and decision-support system for synthetic financial
transactions.

The project demonstrates temporal validation, leakage-safe feature engineering, severe
class-imbalance handling, calibrated risk scoring, threshold optimisation, operational
review-rate analysis, model explainability and monitored API inference.

> **Disclaimer.** This project is an independent educational implementation using synthetic
> financial data. It is designed to demonstrate fraud risk scoring and ML engineering
> concepts and is not intended for production fraud decisioning, regulatory compliance, or
> autonomous blocking of transactions.

## Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [ML problem formulation](#3-ml-problem-formulation)
4. [Synthetic dataset and leakage prevention](#4-synthetic-dataset-and-leakage-prevention)
5. [Models, imbalance and evaluation](#5-models-imbalance-and-evaluation)
6. [Threshold policy and explainability](#6-threshold-policy-and-explainability)
7. [Model versioning and drift monitoring](#7-model-versioning-and-drift-monitoring)
8. [API](#8-api)
9. [CLI](#9-cli)
10. [Docker](#10-docker)
11. [Testing and quality](#11-testing-and-quality)
12. [Limitations](#12-limitations)

## 1. Overview

Fraud is rare, costly and adversarial. A model that ranks transactions by risk is only useful
when its evaluation respects time, its features cannot see the future, its operating point is
an explicit policy and its artifacts can be verified and monitored. This repository builds
that end to end on synthetic data: generate -> validate -> features -> temporal split ->
baseline and XGBoost -> calibration check -> threshold selection -> versioned artifact ->
API / batch scoring with reason codes -> drift reports.

## 2. Architecture

```
src/fraud_ml/
├── contracts/    transaction (input), score (output), enums, money
├── data/         loader, validation, temporal split, synthetic generator
├── features/     definitions, transaction, temporal, velocity (history), behavioural, pipeline
├── models/       baseline, gradient_boosting, calibration, serialization, registry
├── training/     train, evaluation, threshold, experiment (MLflow)
├── scoring/      scorer, explanation (reason codes), batch
├── monitoring/   drift, score_distribution, data_quality
├── api/          routes, dependencies, errors
├── adapters/     txn_pipeline
├── cli/          main (Typer)
├── config.py · logging.py · exceptions.py · main.py (ASGI)
tests/            unit · property · integration · api · e2e
docs/             FEATURE_ENGINEERING · THRESHOLDING · EXPLAINABILITY · DRIFT · SECURITY
```

Stack: Python 3.12, pandas, NumPy, scikit-learn, XGBoost, Pydantic v2, FastAPI, Typer,
MLflow (local file store), joblib, PyArrow, matplotlib, structlog; pytest, Hypothesis,
ruff, mypy (strict), bandit, pip-audit; uv; Docker; GitHub Actions.

## 3. ML problem formulation

* **Unit:** one transaction with point-in-time account context.
* **Target:** synthetic `is_fraud` (about 1% positive).
* **Output:** a risk probability ranked for review prioritisation; decisions are
  recommendations for a human or downstream control, never autonomous blocks.
* **Primary metric:** PR-AUC; operating point chosen by recall target, review-rate ceiling
  and illustrative cost.
* **Validation:** temporal; everything that tunes sees the validation period only.
* **Serving:** online (single and small batch via HTTP) and offline (Parquet/JSONL/CSV).

## 4. Synthetic dataset and leakage prevention

`fraud-ml generate-data` produces a seeded dataset of synthetic accounts and transactions
with injected fraud-like patterns (account takeover, card-not-present bursts, mule activity,
rapid movement, structuring and others). All identifiers are synthetic.

History features (`features/velocity.py`) are computed point-in-time from strictly earlier
transactions, and the feature catalogue (`features/definitions.py`) lists the columns that
may never enter the model (labels, identifiers, post-outcome fields). The feature pipeline
raises `LeakageError` if any of them is present. See
[docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md).

## 5. Models, imbalance and evaluation

* **Baseline:** class-weighted logistic regression on the shared feature pipeline.
* **Advanced model:** XGBoost with `scale_pos_weight` (square-root mode by default) and
  early stopping on the validation period.
* **Calibration:** measured every run (Brier, ECE, reliability table); Platt or isotonic
  calibration is applied only when it improves Brier without hurting ranking.
* **Evaluation:** PR-AUC, ROC-AUC, precision/recall at the selected threshold, confusion
  counts, segment metrics (channel, currency, amount band), feature and permutation
  importance, and plots written to `reports/`.

## 6. Threshold policy and explainability

The threshold is an operational policy selected on validation data: hit the recall target
without exceeding the review-rate ceiling, then minimise the illustrative expected cost.
Every score carries rule-based reason codes (for example `HIGH_AMOUNT_VS_HISTORY`,
`NEW_BENEFICIARY`, `CROSS_BORDER`, `HIGH_RISK_IP`, `NEW_DEVICE`). See
[docs/THRESHOLDING.md](docs/THRESHOLDING.md) and
[docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md).

## 7. Model versioning and drift monitoring

Each training run writes a versioned artifact directory with metadata (model, schema,
feature and score versions, data hash, git commit, seed, library versions, policy, metrics)
and the SHA-256 of the serialized pipeline; loading fails safely on any mismatch.
`fraud-ml drift-check` computes PSI for numeric and categorical inputs and for the score
distribution. See [docs/DRIFT.md](docs/DRIFT.md).

## 8. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | process is up |
| GET | `/health/ready` | a verified model artifact is loaded (503 otherwise) |
| GET | `/api/v1/model` | version, contract versions, policy, metrics |
| POST | `/api/v1/score` | score one contract transaction |
| POST | `/api/v1/score/batch` | score up to `FRAUD_ML_API_MAX_BATCH_SIZE` transactions |

```bash
curl -s -X POST localhost:8000/api/v1/score -H 'Content-Type: application/json' \
  --data @examples/transaction.json
```

Errors use one envelope, `{"error": {"code", "message", "correlation_id", "details"}}`, and
every response carries `X-Correlation-ID`. Feature payloads are never logged.

## 9. CLI

```bash
fraud-ml generate-data --transactions 100000 --fraud-rate 0.01 --seed 42 --output data/generated/
fraud-ml train --input data/generated/transactions.parquet [--fast] [--permutation-importance]
fraud-ml evaluate --model models/latest --input data/generated/transactions.parquet
fraud-ml score-file --input <transactions.parquet> --output reports/scores.parquet
fraud-ml score --input examples/transaction.json
fraud-ml drift-check --reference <reference.parquet> --current <current.parquet> --model models/latest
fraud-ml model-info --model models/latest
fraud-ml serve --port 8000
fraud-ml demo [--transactions 50000 --fraud-rate 0.01 --seed 42] [--fast] [--no-mlflow]
```

## 10. Docker

`docker compose up --build` serves the API on `localhost:8000`, training a small bootstrap
model from synthetic data on first start; `docker compose run --rm fraud-ml fraud-ml demo`
runs the full demo in the same image. The image runs as a non-root user.

## 11. Testing and quality

```bash
uv sync
uv run pytest --cov
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src
uv run bandit -c pyproject.toml -r src -q
```

Unit, property-based (Hypothesis), integration, API and end-to-end suites; CI runs them with
lint, type and security checks and a Docker smoke test. See
[docs/SECURITY.md](docs/SECURITY.md).

## 12. Limitations

Synthetic data only; the fraud patterns are hand-designed and far simpler than real fraud.
Metrics on the small validation and test periods carry large uncertainty. The cost model is
illustrative. None of this should inform decisions about real customers or payments.

## License

MIT.
