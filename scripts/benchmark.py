"""Local benchmark: training time, batch scoring throughput and in-process API latency.

Run with ``uv run python scripts/benchmark.py``; results are printed as JSON and written to
``benchmarks/results.json`` (git-ignored). Numbers describe this machine only.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from fraud_ml.config import Settings
from fraud_ml.contracts.transaction import LABEL_COLUMNS
from fraud_ml.data.loader import sha256_frame, write_frame
from fraud_ml.data.synthetic import GeneratorConfig, generate_dataset
from fraud_ml.main import create_app
from fraud_ml.scoring.batch import score_file
from fraud_ml.scoring.scorer import Scorer
from fraud_ml.training.train import TrainingConfig, train_model


def main() -> None:
    root = Path("benchmarks/work")
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "machine": f"{platform.machine()} / {platform.system()} {platform.release()} / Python {sys.version.split()[0]}",
        "cpu_count": __import__("os").cpu_count(),
    }
    for rows in (10_000, 50_000):
        t = time.perf_counter()
        frame, _ = generate_dataset(GeneratorConfig(transactions=rows, fraud_rate=0.01, seed=42))
        gen = time.perf_counter() - t
        t = time.perf_counter()
        result = train_model(
            frame,
            TrainingConfig(),
            model_root=root / f"models-{rows}",
            reports_dir=root / f"reports-{rows}",
            data_name="bench",
            data_version=sha256_frame(frame),
        )
        results[f"train_{rows}"] = {
            "generate_seconds": round(gen, 2),
            "train_total_seconds": round(result.total_seconds, 2),
            "fit_seconds": {k: round(v.fit_seconds, 2) for k, v in result.candidates.items()},
            "winner": result.winner,
        }
        print(
            f"rows={rows}: generate {gen:.1f}s, train {result.total_seconds:.1f}s", file=sys.stderr
        )

    scorer = Scorer.from_path(root / "models-50000" / "latest")
    frame, _ = generate_dataset(GeneratorConfig(transactions=50_000, fraud_rate=0.01, seed=43))
    unlabelled = frame.drop(columns=list(LABEL_COLUMNS))
    path = write_frame(unlabelled, root / "scoring.parquet")
    summary = score_file(scorer, path, root / "scores.parquet")
    results["batch_scoring_50000"] = summary.to_dict()

    settings = Settings(
        model_dir=root / "models-50000",
        mlflow_enabled=False,
        log_format="json",
        log_level="WARNING",
    )
    with TestClient(create_app(settings)) as client:
        payloads = []
        for _, row in unlabelled.head(300).iterrows():
            record = {
                k: v
                for k, v in row.to_dict().items()
                if not (isinstance(v, float) and math.isnan(v))
            }
            record["transaction_timestamp"] = row["transaction_timestamp"].isoformat()
            payloads.append(record)
        for payload in payloads[:20]:  # warm-up
            client.post("/api/v1/score", json=payload)
        latencies = []
        for payload in payloads:
            t = time.perf_counter()
            response = client.post("/api/v1/score", json=payload)
            latencies.append((time.perf_counter() - t) * 1000)
            assert response.status_code == 200
        latencies.sort()
        results["api_single_score_ms"] = {
            "requests": len(latencies),
            "p50": round(statistics.median(latencies), 2),
            "p95": round(latencies[int(0.95 * len(latencies)) - 1], 2),
            "max": round(latencies[-1], 2),
            "note": "in-process TestClient (no network); includes validation, features and model",
        }
        batch = [payloads[i % len(payloads)] for i in range(500)]
        t = time.perf_counter()
        response = client.post("/api/v1/score/batch", json={"transactions": batch})
        assert response.status_code == 200
        results["api_batch_500_ms"] = round((time.perf_counter() - t) * 1000, 1)

    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
