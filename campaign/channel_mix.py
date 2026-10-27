"""
How each kind of spender splits its money across channels (TV, digital,
mail, ...) in cheap versus expensive TV markets, from the purpose filers
give on each independent expenditure. If spenders weighed TV's wasted reach,
mail and digital (which don't pay for the whole market) would take a bigger
share where TV is most expensive.
"""

import numpy as np
import pandas as pd

from .load_expenditures import load_expenditures

HOUSE_MAJORITY_PAC = "C00495028"
BANDS = [0, 3, 10, np.inf]
BAND_LABELS = ["Cheap (reach under 3)", "Mid (3-10)", "Expensive (10+)"]


def channel_mix(cycle, markets):
    """Dollars and share of each group's dollars by channel, per market-cost
    band (reach from load_media_markets). Groups: DCCC, House Majority PAC,
    and all outside groups (both sides)."""
    ie = load_expenditures(cycle).merge(markets[["district_id", "reach"]], on="district_id", how="inner")
    ie["band"] = pd.cut(ie.reach, BANDS, labels=BAND_LABELS)
    groups = {"DCCC": ie[ie.bucket == "dccc"],
              "House Majority PAC": ie[ie.spe_id == HOUSE_MAJORITY_PAC],
              "All outside groups": ie[ie.bucket.str.startswith("outside")]}
    rows = []
    for group, g in groups.items():
        totals = g.groupby(["band", "channel"], observed=False).amount.sum().rename("amount").reset_index()
        totals["share"] = totals.amount / totals.groupby("band", observed=False).amount.transform("sum")
        rows.append(totals.assign(group=group, cycle=cycle))
    return pd.concat(rows, ignore_index=True).fillna({"share": 0.0})
