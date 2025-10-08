"""Score distribution summaries for monitoring dashboards and score-drift baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from fraud_ml.contracts.score import RiskPolicy

QUANTILES: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99, 0.999)


class ScoreDistribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: int
    mean: float
    quantiles: dict[str, float]
    band_rates: dict[str, float]
    review_rate: float
    alerts_per_1000: float
    histogram: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def summarise_scores(
    probabilities: np.ndarray, policy: RiskPolicy, bins: int = 20
) -> ScoreDistribution:
    p = np.asarray(probabilities, dtype="float64")
    if len(p) == 0:
        return ScoreDistribution(
            rows=0,
            mean=0.0,
            quantiles={},
            band_rates={},
            review_rate=0.0,
            alerts_per_1000=0.0,
            histogram={},
        )
    bands = pd.Series([policy.band(float(x)).value for x in p])
    band_rates = {k: float(v) for k, v in bands.value_counts(normalize=True).items()}
    review_rate = float(np.mean(p >= policy.threshold))
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = np.histogram(p, bins=edges)[0]
    histogram = {f"{edges[i]:.2f}-{edges[i + 1]:.2f}": int(counts[i]) for i in range(bins)}
    return ScoreDistribution(
        rows=len(p),
        mean=float(p.mean()),
        quantiles={f"p{int(q * 1000) / 10:g}": float(np.quantile(p, q)) for q in QUANTILES},
        band_rates=band_rates,
        review_rate=review_rate,
        alerts_per_1000=1000.0 * review_rate,
        histogram=histogram,
    )
