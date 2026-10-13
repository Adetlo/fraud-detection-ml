"""Temporal train/validation/test split (ADR 0001).

Rows are ordered by ``transaction_timestamp`` and cut at row-count fractions, so the
validation period follows training and the test period follows validation. Random shuffling
of financial history is never used for final evaluation: it would let the model see the
*future* of each account while being evaluated on its past.

Account-level overlap is deliberate: the same accounts appear in every split, exactly as a
deployed model scores future transactions of known accounts. Because every history feature
is computed from strictly earlier rows, nothing from the validation or test period can reach
the training rows (``tests/unit/test_leakage.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

TIMESTAMP_COLUMN = "transaction_timestamp"
LABEL_COLUMN = "is_fraud"


@dataclass(frozen=True)
class SplitSummary:
    name: str
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp
    positives: int

    @property
    def positive_rate(self) -> float:
        return self.positives / self.rows if self.rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "positives": self.positives,
            "positive_rate": round(self.positive_rate, 6),
        }


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    summaries: tuple[SplitSummary, SplitSummary, SplitSummary]

    def to_dict(self) -> dict[str, Any]:
        return {s.name: s.to_dict() for s in self.summaries}


def _summarise(name: str, frame: pd.DataFrame) -> SplitSummary:
    positives = int(frame[LABEL_COLUMN].sum()) if LABEL_COLUMN in frame.columns else 0
    return SplitSummary(
        name=name,
        rows=len(frame),
        start=frame[TIMESTAMP_COLUMN].iloc[0],
        end=frame[TIMESTAMP_COLUMN].iloc[-1],
        positives=positives,
    )


def temporal_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must lie strictly between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must leave room for a test period")
    if len(frame) < 3:
        raise ValueError("need at least three rows for a three-way temporal split")

    ordered = frame.sort_values([TIMESTAMP_COLUMN, "transaction_id"], kind="mergesort").reset_index(
        drop=True
    )
    n = len(ordered)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    train = ordered.iloc[:train_end]
    validation = ordered.iloc[train_end:validation_end]
    test = ordered.iloc[validation_end:]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("every split must contain at least one row")

    assert train[TIMESTAMP_COLUMN].max() <= validation[TIMESTAMP_COLUMN].min()  # noqa: S101
    assert validation[TIMESTAMP_COLUMN].max() <= test[TIMESTAMP_COLUMN].min()  # noqa: S101
    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        summaries=(
            _summarise("train", train),
            _summarise("validation", validation),
            _summarise("test", test),
        ),
    )
