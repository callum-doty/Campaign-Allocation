"""One FRED series as downloaded by fetch.py (data/raw/fred/<id>.csv)."""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "fred"


def load_fred(series_id):
    """Series indexed by observation date (the first day of the month or
    quarter it covers); missing observations dropped."""
    raw = pd.read_csv(RAW / f"{series_id}.csv")
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy()
    return pd.Series(values, index=pd.to_datetime(raw.iloc[:, 0]), name=series_id).dropna()
