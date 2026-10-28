"""
Score one completed cycle strictly forward in time: fit the baseline on
every EARLIER cycle, forecast this one (each race, and the seat total with
its shared national error), then ask how DCCC's own spending compares with
the best allocation of the same budget under the spending assumption,
across a grid of assumptions. The same forecast without the incumbent
history features (first-term flags, last cycle's overperformance) is scored
alongside, to show what they add, and so is one using each side's money
over the whole election instead of what was visible by Sept 1, to show
what that hindsight was worth. Two readings need less of that assumption:
how strong party money would have to be for reallocating to gain a full
seat (break_even.py), and how much more effective money would have to be
in each band of races for DCCC's own split to be optimal (implied_belief.py).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .break_even import break_even_scale
from .build_races import FLOOR_PSEUDO_DOLLARS
from .fit_baseline import FEATURES, HISTORY, fit_baseline
from .implied_belief import implied_belief
from .predict_margin import predict_margin
from .reallocation_gain import reallocation_gain
from .seat_distribution import seat_distribution
from .seats_needed import seats_needed
from .spending_effect import HALF_SATURATION, SCALE_PTS

SCALES = (0.5, 1.5, 4.5)
HALF_SATURATIONS = (1_650_000, 5_500_000, 16_500_000)
CAP_SHARE = 0.15          # no race gets more than 15% of the budget


def hindsight_money(races):
    """floor_share from each side's candidate and outside spending over the
    whole election, most of it after Sept 1."""
    d, r, k = races.disbursements_d + races.outside_d, races.disbursements_r + races.outside_r, FLOOR_PSEUDO_DOLLARS
    return races.assign(floor_share=np.log((d + k) / (d + r + 2 * k)))


def backtest(cycle, races):
    """`races` maps cycle -> build_races(cycle)."""
    train = pd.concat([r for c, r in races.items() if c < cycle])
    model = fit_baseline(train)
    without = fit_baseline(train, [f for f in FEATURES if f not in HISTORY])
    test = races[cycle].copy()
    test["mu"] = predict_margin(test, model)
    test["p"] = norm.cdf(test.mu / model["sigma"])
    test["mu_without"] = predict_margin(test, without)
    test["p_without"] = norm.cdf(test.mu_without / without["sigma"])
    hindsight = fit_baseline(hindsight_money(train))
    test["p_hindsight"] = norm.cdf(predict_margin(hindsight_money(test), hindsight) / hindsight["sigma"])
    known = test[test.margin.notna()].assign(won=lambda t: (t.margin > 0).astype(float))
    calibration = known.groupby("tier").agg(n=("p", "size"), predicted=("p", "mean"), actual=("won", "mean"))
    scores = {}
    for name, col in (("with", "p"), ("without", "p_without"), ("hindsight", "p_hindsight")):
        p = known[col].clip(1e-6, 1 - 1e-6)
        scores[name] = {"brier": float(np.mean((p - known.won) ** 2)),
                        "log_loss": float(-np.mean(known.won * np.log(p) + (1 - known.won) * np.log(1 - p)))}

    budget, cap = float(test.dccc.sum()), CAP_SHARE * float(test.dccc.sum())
    grid = {(scale, half): reallocation_gain(test.mu, model["sigma"], test.dccc, cap, scale_pts=scale, half_saturation=half)
            for scale in SCALES for half in HALF_SATURATIONS}
    gain, test["model_dccc"] = grid[(SCALE_PTS, HALF_SATURATION)]
    break_even = {half: break_even_scale(test.mu, model["sigma"], test.dccc, cap, half) for half in HALF_SATURATIONS}
    belief = {half: implied_belief(test.mu, model["sigma"], test.dccc, half) for half in HALF_SATURATIONS}
    seats = seat_distribution(test.mu, model, seats_needed(cycle, test.district_id), test.district_id.str[:2])
    seats["actual"] = int((test.margin > 0).sum())
    return {"cycle": cycle, "model": model, "races": test, "calibration": calibration,
            "brier": scores["with"]["brier"], "scores": scores, "budget": budget, "gain": gain,
            "grid": {k: v[0] for k, v in grid.items()}, "break_even": break_even, "belief": belief, "seats": seats}
