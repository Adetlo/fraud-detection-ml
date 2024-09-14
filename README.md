# Fraud Detection ML

Educational fraud risk scoring for synthetic financial transactions.

> **Disclaimer.** This project is an independent educational implementation using synthetic
> financial data. It is not intended for production fraud decisioning, regulatory compliance,
> or autonomous blocking of transactions.

## Status

Early development. The goal is a small but complete fraud risk scoring system that gets the
unglamorous parts right: versioned input contracts, a reproducible synthetic generator with
injected fraud-like patterns, point-in-time account history (no future information), and a
strictly temporal train/validation/test split before any model is trained.

## Planned scope

* Versioned transaction contract (`fraud-transaction/1`) with closed vocabularies
* Seeded synthetic transaction generator with labelled fraud patterns
* Leakage-safe feature engineering shared by training and inference
* Logistic regression baseline and a gradient-boosted model
* Temporal validation, threshold selection and calibrated risk scores
* Scoring API, batch scoring and basic drift monitoring

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src
```

Python 3.12. Settings are read from environment variables prefixed `FRAUD_ML_` (see
`fraud_ml.config`).

## License

MIT.
