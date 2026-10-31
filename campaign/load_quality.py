"""
Challenger quality for each nominee who isn't the incumbent: 1 if they had
a Wikipedia article on Jan 1 of the election year (lookup_quality.py), by
FEC candidate ID. Incumbents aren't looked up; build_races counts them as 0,
since incumbency has its own terms.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "wikipedia"


def load_quality(cycle):
    """Series: cand_id -> quality (0 or 1)."""
    path = RAW / f"quality_{cycle}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}: run fetch() to look up {cycle}'s nominees on Wikipedia")
    found = pd.read_csv(path, dtype={"cand_id": str})
    return found.drop_duplicates("cand_id").set_index("cand_id").quality.astype(float)
