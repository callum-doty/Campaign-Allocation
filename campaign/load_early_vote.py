"""
Each state's share of ballots cast before Election Day, from the U.S. Election
Assistance Commission's Election Administration and Voting Survey (EAVS),
summed over jurisdictions (fetch.py downloads one file per election).

    early = UOCAVA (F1c) + by mail (F1d) + early in person (F1f)
            + mail in all-mail jurisdictions (F1g)
    Election Day = in person at a polling place (F1b) + provisional (F1e)
    early share = early / (early + Election Day)

Ballots reported as "other" (F1h on) are left out, since when they were cast
isn't known. Negative codes (not applicable, missing) count as missing.

Where a state's methods account for under 90% or over 110% of its total
ballots (F1a), the breakdown is incomplete (Alabama and Connecticut in 2024,
New Jersey in 2022 and 2024, Montana in 2022 and 2024, a few others) and
the state's share from the nearest complete year is used instead, which is
closer than a national average for states that vote mostly by mail. 2020,
when the pandemic sent most states to mail voting, stands in only for
itself.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "eavs"
YEARS = (2016, 2018, 2020, 2022, 2024)
EARLY, ELECTION_DAY = ["F1c", "F1d", "F1f", "F1g"], ["F1b", "F1e"]


@lru_cache(maxsize=None)
def early_vote_table():
    """DataFrame (state x year): early share and coverage (the methods'
    ballots over total ballots), straight from the survey."""
    rows = []
    for year in YEARS:
        head = pd.read_csv(RAW / f"eavs_{year}.csv", nrows=0, encoding="latin-1").columns
        state = "State" if "State" in head else "State_Abbr"
        raw = pd.read_csv(RAW / f"eavs_{year}.csv", usecols=[state, "F1a"] + EARLY + ELECTION_DAY,
                          encoding="latin-1", low_memory=False)
        num = raw[["F1a"] + EARLY + ELECTION_DAY].apply(pd.to_numeric, errors="coerce")
        num = num.where(num >= 0)
        g = num.groupby(raw[state].str.strip().str.upper()).sum(min_count=1)
        known = g[EARLY + ELECTION_DAY].sum(axis=1)
        rows.append(pd.DataFrame({"year": year, "early_share": g[EARLY].sum(axis=1) / known,
                                  "coverage": known / g.F1a}))
    return pd.concat(rows).rename_axis("state").reset_index()


def early_vote_share(cycle):
    """Series by state: the share of ballots cast before Election Day in
    `cycle`, from the nearest complete year (other than 2020) where that one
    isn't."""
    table = early_vote_table()
    ok = table[table.coverage.between(0.9, 1.1) & table.early_share.notna() & ((table.year != 2020) | (cycle == 2020))]
    out = {}
    for state, g in ok.groupby("state"):
        nearest = g.iloc[np.argsort(np.abs(g.year.to_numpy() - cycle), kind="stable")[0]]
        out[state] = float(nearest.early_share)
    return pd.Series(out, name="early_share")
