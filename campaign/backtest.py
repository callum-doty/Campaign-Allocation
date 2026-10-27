"""
Score one completed cycle strictly forward in time: fit the baseline on
every EARLIER cycle, forecast this one (each race, and the seat total with
its shared national error), then ask how DCCC's own spending compares with
the best allocation of the same budget under the spending assumption,
across a grid of assumptions.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .allocate import allocate
from .fit_baseline import fit_baseline
from .predict_margin import predict_margin
from .seat_distribution import seat_distribution
from .seats_needed import seats_needed
from .spending_effect import HALF_SATURATION, SCALE_PTS, spending_effect

SCALES = (0.5, 1.5, 4.5)
HALF_SATURATIONS = (1_650_000, 5_500_000, 16_500_000)
CAP_SHARE = 0.15          # no race gets more than 15% of the budget


def backtest(cycle, races):
    """`races` maps cycle -> build_races(cycle)."""
    model = fit_baseline(pd.concat([r for c, r in races.items() if c < cycle]))
    test = races[cycle].copy()
    test["mu"] = predict_margin(test, model)
    test["p"] = norm.cdf(test.mu / model["sigma"])
    known = test[test.margin.notna()].assign(won=lambda t: (t.margin > 0).astype(float))
    calibration = known.groupby("tier").agg(n=("p", "size"), predicted=("p", "mean"), actual=("won", "mean"))

    budget = float(test.dccc.sum())
    grid = {}
    for scale in SCALES:
        for half in HALF_SATURATIONS:
            x = allocate(test.mu, model["sigma"], test.dccc, budget, CAP_SHARE * budget,
                         scale_pts=scale, half_saturation=half)
            shift = spending_effect(x, scale, half) - spending_effect(test.dccc, scale, half)
            grid[(scale, half)] = (float(norm.cdf((test.mu + shift) / model["sigma"]).sum() - test.p.sum()), x)
    gain, test["model_dccc"] = grid[(SCALE_PTS, HALF_SATURATION)]
    seats = seat_distribution(test.mu, model, seats_needed(cycle, test.district_id))
    seats["actual"] = int((test.margin > 0).sum())
    return {"cycle": cycle, "model": model, "races": test, "calibration": calibration,
            "brier": float(np.mean((known.p - known.won) ** 2)), "budget": budget, "gain": gain,
            "grid": {k: v[0] for k, v in grid.items()}, "seats": seats}
