"""
Forecast an upcoming cycle from a baseline fit on every completed cycle, and
show how expected seats move with the national environment. A polling miss
shifts every race at once, so the generic ballot is varied rather than
treated as known.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fit_baseline import fit_baseline
from .predict_margin import predict_margin


def forecast(races, cycle=2026):
    """`races` maps cycle -> build_races(cycle)."""
    model = fit_baseline(pd.concat([r for c, r in races.items() if c < cycle]))
    fc = races[cycle].copy()
    fc["mu"] = predict_margin(fc, model)
    fc["p"] = norm.cdf(fc.mu / model["sigma"])
    gb = float(fc.gb.iloc[0])
    swings = np.round(np.arange(-6, 6.01, 0.5), 1)
    seats = {round(gb + float(s), 1): float(norm.cdf((fc.mu + s) / model["sigma"]).sum()) for s in swings}
    return {"cycle": cycle, "model": model, "races": fc, "gb": gb, "seats_by_gb": seats}
