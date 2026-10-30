"""
Forecast an upcoming cycle from a baseline fit on every completed cycle:
each race's chance, the distribution of the seat total (a polling miss moves
every race at once; also with the size of that miss treated as known, for
comparison), and how expected seats move with the generic ballot.

The same forecast with incumbency drifting from cycle to cycle and a term
for incumbents on redrawn lines (fit_drift.py) is alongside: p_drift, and
p_drift_old_lines with that term off, as if every incumbent kept their
lines, plus its seat total. The backtest scores it worse than the baseline,
so it is a comparison, not the forecast. So is the baseline's fit of how
each race's error size depends on its features (fit_spread.py), for the
report.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fit_baseline import fit_baseline
from .fit_drift import fit_drift
from .fit_spread import fit_spread
from .predict_margin import predict_margin
from .seat_distribution import seat_distribution
from .seats_needed import seats_needed


def forecast(races, cycle=2026):
    """`races` maps cycle -> build_races(cycle)."""
    train = pd.concat([r for c, r in races.items() if c < cycle])
    model, drift = fit_baseline(train), fit_drift(train)
    fc = races[cycle].copy()
    fc["mu"] = predict_margin(fc, model)
    fc["p"] = norm.cdf(fc.mu / model["sigma"])
    fc["mu_drift"] = predict_margin(fc, drift)
    fc["p_drift"] = norm.cdf(fc.mu_drift / drift["sigma"])
    fc["p_drift_old_lines"] = norm.cdf(predict_margin(fc.assign(inc_redrawn=0.0), drift) / drift["sigma"])
    gb = float(fc.gb.iloc[0])
    swings = np.round(np.arange(-6, 6.01, 0.5), 1)
    by_gb = {round(gb + float(s), 1): float(norm.cdf((fc.mu + s) / model["sigma"]).sum()) for s in swings}
    need, states = seats_needed(cycle, fc.district_id), fc.district_id.str[:2]
    seats = seat_distribution(fc.mu, model, need, states)
    seats_known = seat_distribution(fc.mu, {**model, "national_df": None}, need, states)
    return {"cycle": cycle, "model": model, "drift": drift, "spread": fit_spread(train, model), "races": fc,
            "gb": gb, "seats_by_gb": by_gb, "seats": seats, "seats_known_sigma": seats_known,
            "seats_drift": seat_distribution(fc.mu_drift, drift, need, states)}
