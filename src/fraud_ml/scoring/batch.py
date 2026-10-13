"""Batch scoring of Parquet / JSON Lines / CSV files.

Invalid rows are never scored and never silently dropped: they are written to a rejects file
next to the output with their validation codes, and the summary reports both counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_ml.data.loader import read_frame, write_frame
from fraud_ml.data.validation import validate_frame
from fraud_ml.logging import get_logger
from fraud_ml.scoring.scorer import SCORE_COLUMNS, Scorer

log = get_logger(__name__)


@dataclass(frozen=True)
class BatchSummary:
    input_rows: int
    scored_rows: int
    rejected_rows: int
    output_path: Path
    rejects_path: Path | None
    model_version: str
    decision_counts: dict[str, int] = field(default_factory=dict)
    band_counts: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "scored_rows": self.scored_rows,
            "rejected_rows": self.rejected_rows,
            "output_path": str(self.output_path),
            "rejects_path": str(self.rejects_path) if self.rejects_path else None,
            "model_version": self.model_version,
            "decision_counts": self.decision_counts,
            "band_counts": self.band_counts,
            "seconds": round(self.seconds, 3),
            "rows_per_second": round(self.scored_rows / self.seconds, 1) if self.seconds else None,
        }


def score_frame_batch(
    scorer: Scorer, frame: pd.DataFrame, *, scored_at: datetime | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate then score; returns ``(scores, rejects)`` with ``len(scores) == valid rows``."""
    result = validate_frame(frame, strict=False)
    valid = result.valid_rows(frame)
    rejects = result.invalid_rows(frame)
    if valid.empty:
        scores = pd.DataFrame({c: pd.Series(dtype=object) for c in SCORE_COLUMNS})
    else:
        scores = scorer.score_frame(valid, scored_at=scored_at)
    return scores.reset_index(drop=True), rejects.reset_index(drop=True)


def score_file(scorer: Scorer, input_path: Path, output_path: Path) -> BatchSummary:
    import time

    started = time.perf_counter()
    frame = read_frame(input_path)
    scores, rejects = score_frame_batch(scorer, frame, scored_at=datetime.now(tz=UTC))
    write_frame(scores, output_path)
    rejects_path: Path | None = None
    if not rejects.empty:
        rejects_path = output_path.with_name(output_path.stem + "_rejects" + output_path.suffix)
        write_frame(rejects, rejects_path)
    seconds = time.perf_counter() - started
    summary = BatchSummary(
        input_rows=len(frame),
        scored_rows=len(scores),
        rejected_rows=len(rejects),
        output_path=output_path,
        rejects_path=rejects_path,
        model_version=scorer.model_version,
        decision_counts={
            str(k): int(v) for k, v in scores["decision_support"].value_counts().items()
        }
        if len(scores)
        else {},
        band_counts={str(k): int(v) for k, v in scores["risk_band"].value_counts().items()}
        if len(scores)
        else {},
        seconds=seconds,
    )
    log.info(
        "batch.scored",
        model_version=scorer.model_version,
        rows=len(frame),
        scored=len(scores),
        rejected=len(rejects),
        duration_ms=round(seconds * 1000, 1),
    )
    return summary
