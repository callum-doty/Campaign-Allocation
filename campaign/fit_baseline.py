"""
Job 1: the baseline forecast, fit by ordinary least squares.

    margin_i - GB = a + b PVI_i + c D_incumbent_i + d Open_i + e floor_share_i + error_i
    error_i ~ Normal(0, sigma^2)

The generic ballot enters with a fixed 1:1 pass-through (uniform national
swing) rather than an estimated slope: with one national value per cycle and
seven cycles, a fitted slope would mostly chase polling misses. sigma is the
residual standard deviation, so it includes those misses; that is the right
uncertainty for a forecast made before the votes are counted.

sigma splits into a national part shared by every race in a cycle (the
spread of each cycle's average miss, i.e. the polling error) and a district
part, sigma^2 = sigma_national^2 + sigma_district^2. Each race's own win
probability only needs the total; the split matters for the seat total,
because a polling miss moves every race at once.

Uncontested races (|margin| >= 80) carry no information about competition
and are left out of the fit.
"""

import numpy as np
import pandas as pd

FEATURES = ["pvi", "d_incumbent", "open_seat", "floor_share"]
UNCONTESTED = 80


def fit_baseline(races):
    """{'coef': {name: value}, 'sigma', 'sigma_national', 'sigma_district',
    'n', 'cycles'}"""
    train = races[races.margin.abs() < UNCONTESTED]
    X = np.column_stack([np.ones(len(train))] + [train[f].to_numpy(float) for f in FEATURES])
    y = (train.margin - train.gb).to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = float(np.sqrt(resid @ resid / (len(y) - X.shape[1])))
    national = float(pd.Series(resid).groupby(train.cycle.to_numpy()).mean().std())
    return {"coef": dict(zip(["intercept"] + FEATURES, coef)), "sigma": sigma,
            "sigma_national": national, "sigma_district": float(np.sqrt(sigma ** 2 - national ** 2)),
            "n": len(y), "cycles": sorted(int(c) for c in train.cycle.unique())}
