"""
Job 1: the baseline forecast, fit by ordinary least squares.

    margin_i - GB = a + b PVI_i + c D_incumbent_i + d Open_i + e floor_share_i + error_i
    error_i ~ Normal(0, sigma^2)

The generic ballot enters with a fixed 1:1 pass-through (uniform national
swing) rather than an estimated slope: with one national value per cycle and
seven cycles, a fitted slope would mostly chase polling misses. sigma is the
residual standard deviation, so it includes those misses; that is the right
uncertainty for a forecast made before the votes are counted.

Uncontested races (|margin| >= 80) carry no information about competition
and are left out of the fit.
"""

import numpy as np

FEATURES = ["pvi", "d_incumbent", "open_seat", "floor_share"]
UNCONTESTED = 80


def fit_baseline(races):
    """{'coef': {name: value}, 'sigma': float, 'n': int, 'cycles': [...]}"""
    train = races[races.margin.abs() < UNCONTESTED]
    X = np.column_stack([np.ones(len(train))] + [train[f].to_numpy(float) for f in FEATURES])
    y = (train.margin - train.gb).to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return {"coef": dict(zip(["intercept"] + FEATURES, coef)),
            "sigma": float(np.sqrt(resid @ resid / (len(y) - X.shape[1]))),
            "n": len(y), "cycles": sorted(int(c) for c in train.cycle.unique())}
