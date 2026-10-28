"""
Forecast an upcoming cycle from a baseline fit on every completed cycle:
each race's chance, the distribution of the seat total (a polling miss moves
every race at once; also with the size of that miss treated as known, for
comparison), and how expected seats move with the generic ballot.

p_as_open is each race's chance with its incumbent treated as an open seat,
a bound for incumbents on redrawn lines: the incumbency effect is learned
from members running in their own districts, so on new lines the truth lies
between p and p_as_open.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fit_baseline import fit_baseline
from .predict_margin import predict_margin
from .seat_distribution import seat_distribution
from .seats_needed import seats_needed


def forecast(races, cycle=2026):
    """`races` maps cycle -> build_races(cycle)."""
    model = fit_baseline(pd.concat([r for c, r in races.items() if c < cycle]))
    fc = races[cycle].copy()
    fc["mu"] = predict_margin(fc, model)
    fc["p"] = norm.cdf(fc.mu / model["sigma"])
    as_open = fc.assign(d_incumbent=0.0, open_seat=1.0, fresh_d=0.0, fresh_r=0.0, over_inc=0.0,
                        over_open=fc.over_inc + fc.over_open)
    fc["p_as_open"] = norm.cdf(predict_margin(as_open, model) / model["sigma"])
    gb = float(fc.gb.iloc[0])
    swings = np.round(np.arange(-6, 6.01, 0.5), 1)
    by_gb = {round(gb + float(s), 1): float(norm.cdf((fc.mu + s) / model["sigma"]).sum()) for s in swings}
    need, states = seats_needed(cycle, fc.district_id), fc.district_id.str[:2]
    seats = seat_distribution(fc.mu, model, need, states)
    seats_known = seat_distribution(fc.mu, {**model, "national_df": None}, need, states)
    return {"cycle": cycle, "model": model, "races": fc, "gb": gb, "seats_by_gb": by_gb, "seats": seats,
            "seats_known_sigma": seats_known}
