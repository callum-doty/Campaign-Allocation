"""
Early Google political ad spending by House nominees, from Google's public ad
transparency data in BigQuery (bigquery-public-data.google_political_ads), exported by
hand to data/Google ads query.csv: spend by advertiser and year, before Sept 1 and in all.

google_odds compares the two sides' spending before Sept 1, as odds, so it's something a
Sept 1 forecast could have read. It can't reach the bar in spending_signal.py: Google's data
keeps seven years, so 2018 is nearly gone, which leaves two forward tests (2022 and 2024)
rather than five. Before 2020 it's 0, as if no one spent, so only 2020 on informs the fit.

Only advertisers listing an FEC ID can be matched to a nominee, about 40-50% of them. An
unmatched campaign counts as spending nothing, so this understates spending, and Google is
one platform: no Meta, streaming or other digital.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "Google ads query.csv"
GOOGLE_CYCLES = (2020, 2022, 2024, 2026)
ODDS_DOLLARS = 10_000
FEC_ID = r"\b([CH]\d[0-9A-Z]{7})\b"


def fec_spend(export):
    """Spending by (year, FEC ID): spend_before_sep1, spend_all_year. An advertiser listing several IDs counts toward each."""
    ids = export.public_ids.astype(str).str.findall(FEC_ID)
    rows = export.assign(fec=ids).explode("fec").dropna(subset=["fec"])
    return rows.groupby(["year", "fec"])[["spend_before_sep1", "spend_all_year"]].sum()


def add_google(races, export=None):
    """`races` (cycle -> frames with cm_d and cm_r) with google_odds, google_d and google_r: dollars before Sept 1."""
    spend = fec_spend(pd.read_csv(EXPORT, dtype={"public_ids": str}) if export is None else export)
    out = {}
    for cycle, x in races.items():
        before = spend.spend_before_sep1.xs(cycle, level="year") if cycle in spend.index.get_level_values("year") and cycle in GOOGLE_CYCLES else pd.Series(dtype=float)
        d, r = (x[f"cm_{side}"].map(before).fillna(0.0) for side in "dr")
        out[cycle] = x.assign(google_d=d, google_r=r, google_odds=np.log((d + ODDS_DOLLARS) / (r + ODDS_DOLLARS)))
    return out
