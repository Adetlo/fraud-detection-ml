# Contributing

## Setup

```bash
git clone https://github.com/Adetlo/fraud-detection-ml.git
cd fraud-detection-ml
uv sync                      # Python 3.12 + locked dependencies (+ dev tools)
cp .env.example .env         # optional local overrides
uv run fraud-ml demo --fast  # smoke test in ~10 s
```

macOS note: XGBoost needs the OpenMP runtime (`libomp`). With Homebrew: `brew install
libomp`. CI (Ubuntu) and the Docker image install `libgomp` from the system.

## Quality checks (all must pass before a pull request)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest --cov --cov-fail-under=85
uv run bandit -c pyproject.toml -r src -q
uv export --frozen --no-emit-project --no-hashes --format requirements-txt -o requirements-audit.txt
uv run pip-audit -r requirements-audit.txt --strict
```

## Training and evaluation

```bash
uv run fraud-ml generate-data --transactions 50000 --fraud-rate 0.01 --seed 42
uv run fraud-ml train --input data/generated/transactions.parquet
uv run fraud-ml evaluate --model models/latest --input data/generated/transactions.parquet
```

Training reports go to `reports/`; MLflow runs to `./mlruns` (`uv run mlflow ui` to browse).
Committed demo artifacts in `reports/demo/` are regenerated with `fraud-ml demo` and copied
by hand when the numbers in the README change. Never edit metric numbers in docs without
re-running the demo that produced them.

## Tests

```
tests/unit          contracts, loader/validation, split, generator, features, history,
                    leakage controls, threshold/evaluation, models/registry, scoring,
                    monitoring, adapter, logging
tests/property      Hypothesis invariants (probability bounds, determinism, monotone
                    review counts, batch row counts, no-future history, Decimal round trip,
                    threshold bounds)
tests/integration   training artifacts/reports, reproducibility, MLflow file store, CLI
tests/api           FastAPI TestClient: health, model, score, batch, error envelope, limits
tests/e2e           generate -> split -> train -> save -> load -> API -> batch -> drift
```

Run a subset with `uv run pytest tests/unit -q`. Keep the session-scoped `trained` fixture
small; CI must stay well under five minutes.

## Adding a feature

1. Add a `FeatureSpec` to `fraud_ml/features/definitions.py` (name, group, kind,
   description) and implement it in the matching group module. Features must be computable
   from one contract row (plus the point-in-time context fields) and must never read other
   rows, labels, identifiers or post-outcome fields.
2. If the feature needs a new contract field, add it to `TransactionFeatures`,
   `validate_frame`, the generator, `compute_history_features` (if it is history) and the
   data contract doc.
3. Bump `FEATURE_VERSION` when the feature matrix changes: existing artifacts will then
   refuse to load, which is intended.
4. Add unit tests for the computation and, if relevant, a leakage test.
5. Re-run the demo and update `docs/MODEL_EVALUATION.md` / README numbers.

## Adding a model

1. Implement a builder in `fraud_ml/models/` returning `build_pipeline(estimator, ...)`.
2. Wire it into `training/train.py` as a candidate evaluated with
   `_evaluate_candidate`; it must go through the same calibration and threshold policy.
3. Extend `select_winner` only with documented, deterministic rules.
4. Keep the logistic baseline; the comparison is the point.

## Evaluation expectations

* Report PR-AUC, precision, recall, F1, Brier, review rate, alerts per 1,000 and expected
  cost at the selected threshold on both validation and test.
* Never select, calibrate or tune on the test split.
* Never commit metric numbers that did not come from a run of the committed code.
* Call segment reporting "segment performance", not fairness.

## Commit style

Conventional prefixes: `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`, `refactor:`.
One logical change per commit; CI must be green on `main`.
