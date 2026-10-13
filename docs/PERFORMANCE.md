# Performance

Local measurements from `scripts/benchmark.py` (`uv run python scripts/benchmark.py`),
run on 2026-10-13 on an Apple-silicon laptop (arm64, macOS 25.6, 8 CPU cores, Python
3.12.13, XGBoost with 2 threads). These describe this machine and this synthetic data;
they do not imply enterprise-scale throughput.

## Training

| Rows | Generate | Train total (both models, calibration, thresholds, reports) | LR fit | XGBoost fit | Winner |
|---:|---:|---:|---:|---:|---|
| 10,000 | 0.6 s | 0.9 s | 0.06 s | 0.27 s | logistic regression |
| 50,000 | 3.0 s | 2.3 s | 0.48 s | 1.07 s | xgboost |

The whole `fraud-ml demo` (50,000 rows, generate -> train -> score -> batch -> drift) runs
in about 8 seconds including Python start-up and MLflow logging.

## Batch scoring

50,000 unlabelled rows (validation + features + model + Parquet write):
0.32 s, about 156,000 rows per second. The 7,500-row demo batch scored at 73,000 rows per
second, dominated by fixed overhead.

## API latency (in-process `TestClient`, no network, no uvicorn workers)

| | ms |
|---|---:|
| single `/api/v1/score`, p50 | 14.9 |
| single `/api/v1/score`, p95 | 16.8 |
| single `/api/v1/score`, max (300 requests) | 49.0 |
| `/api/v1/score/batch` with 500 transactions | 30.5 |

Single-transaction latency is dominated by building a one-row pandas frame through the
feature pipeline; it is the price of having exactly one feature implementation. A
latency-critical deployment would vectorise through batching at the gateway or compile the
feature step, which is listed in the roadmap.

## Docker

Image size 1.35 GB (Python 3.12 slim + scientific stack + mlflow-skinny and its
dependencies; the virtual environment is 814 MB, dominated by pyarrow, scipy, XGBoost,
pandas and scikit-learn). XGBoost's Linux wheel would also pull in the NVIDIA NCCL library
(about 350 MB) for multi-GPU training; a `uv` dependency override in `pyproject.toml` drops
it because this project is CPU-only. First start trains a 20,000-row bootstrap model in
about 9 seconds before `/health/ready` reports ready.

## Test suite

170 tests (unit, property, integration, API, e2e) in about 12 seconds; coverage 97%.
