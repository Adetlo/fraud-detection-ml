"""Temporal features derived from the transaction timestamp only.

Nothing here looks at other rows, so these features can never read the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Hour -> unusualness score. Deep night is 1.0, shoulders decay, daytime is 0.
UNUSUAL_HOUR_SCORE: dict[int, float] = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 0.75,
    6: 0.25,
    21: 0.25,
    22: 0.5,
    23: 0.5,
}
NIGHT_HOURS: frozenset[int] = frozenset(range(0, 6))


def unusual_hour_score(hour: pd.Series) -> pd.Series:
    return hour.map(lambda h: UNUSUAL_HOUR_SCORE.get(int(h), 0.0)).astype("float64")


def temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    ts = frame["transaction_timestamp"].dt.tz_convert("UTC")
    hour = ts.dt.hour.astype("int64")
    dow = ts.dt.dayofweek.astype("int64")
    radians = 2.0 * np.pi * hour / 24.0
    return pd.DataFrame(
        {
            "hour_of_day": hour,
            "day_of_week": dow,
            "hour_sin": np.sin(radians),
            "hour_cos": np.cos(radians),
            "unusual_hour_score": unusual_hour_score(hour),
            "is_weekend": (dow >= 5).astype("int8"),
            "is_night": hour.isin(NIGHT_HOURS).astype("int8"),
        },
        index=frame.index,
    )
