"""
The forecast as it stood on a given date of the campaign (a checkpoint),
fit on every earlier cycle:

- National environment: the generic ballot that day (the day before the
  checkpoint, in data/generic_ballot_dated.csv), plus the average drift from
  that date to the final average in earlier elections, with the drift's
  spread added to the national error (fall_drift.py).
- Money: floor_share counts the outside money that had reached voters by the
  day before the checkpoint, in the training years too, so the model learns
  what money visible by then says. Republican groups attacking a seat show
  up here. As in the baseline, only money neither national committee
  controls counts, and candidates' money is still their July report.
- Everything else is the baseline's.

At Sept 1 this is the Sept 1 forecast: outside money through Aug 31, as in
build_races.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm

from .build_races import FLOOR_PSEUDO_DOLLARS, GENERIC_BALLOT_DATED
from .fall_drift import fall_drift
from .fit_baseline import fit_baseline
from .load_expenditures import load_expenditures
from .predict_margin import predict_margin

CHECKPOINTS = ("09-01", "09-15", "10-01", "10-15", "10-25")


@lru_cache(maxsize=None)
def outside_seen(cycle):
    """checkpoint -> DataFrame by district_id: outside_d, outside_r that had
    reached voters by the day before."""
    ie = load_expenditures(cycle)
    ie = ie[ie.bucket.isin(["outside_d", "outside_r"])]
    return {date: ie[ie.date < pd.Timestamp(f"{cycle}-{date}")]
            .pivot_table(index="district_id", columns="bucket", values="amount", aggfunc="sum")
            .reindex(columns=["outside_d", "outside_r"]).fillna(0.0)
            for date in CHECKPOINTS}


def floor_share_on(races, cycle, date):
    """floor_share with the outside money seen by the day before `date`."""
    seen = outside_seen(cycle)[date]
    k = FLOOR_PSEUDO_DOLLARS
    floor = {side: (races[f"raised_jun_{side}"] - races[f"own_jun_{side}"]).clip(lower=0)
             + races.district_id.map(seen[f"outside_{side}"]).fillna(0.0) for side in ("d", "r")}
    return np.log((floor["d"] + k) / (floor["d"] + floor["r"] + 2 * k))


def dated_forecast(cycle, races, date):
    """`races`: cycle -> build_races(cycle). {'mu', 'p' (arrays in the order
    of races[cycle]), 'sigma', 'sigma_national', 'model', 'drift', 'gb'}."""
    train = pd.concat([r.assign(floor_share=floor_share_on(r, c, date)) for c, r in races.items() if c < cycle])
    model = fit_baseline(train)
    drift = fall_drift(cycle, date)
    dated = pd.read_csv(GENERIC_BALLOT_DATED, dtype={"date": str}).set_index(["cycle", "date"]).generic_ballot
    gb = float(dated[(cycle, date)])
    test = races[cycle].assign(floor_share=floor_share_on(races[cycle], cycle, date), gb=gb + drift["mean"])
    mu = predict_margin(test, model)
    sigma = float(np.hypot(model["sigma"], drift["sd"]))
    return {"mu": mu, "p": norm.cdf(mu / sigma), "sigma": sigma, "model": model, "drift": drift, "gb": gb,
            "sigma_national": float(np.hypot(model["sigma_national"], drift["sd"]))}
