"""
Job 1: the baseline forecast, fit by ordinary least squares.

    margin_i - GB = a + b PVI_i + c D_incumbent_i + d Open_i + e floor_share_i
                    + f fresh_d_i + g fresh_r_i + h over_inc_i + k over_open_i + error_i
    error_i ~ Normal(0, sigma^2)

The generic ballot enters with a fixed 1:1 pass-through (uniform national
swing) rather than an estimated slope: with one national value per cycle and
seven cycles, a fitted slope would mostly chase polling misses. sigma is the
residual standard deviation, so it includes those misses; that is the right
uncertainty for a forecast made before the votes are counted.

The incumbency effect is pooled over every cycle although it has shrunk
(by more than half after 2016). Weighting recent cycles more was tested and
forecast worse forward in time: the fits get noisy and overconfident. The
first-term flags (fresh_d, fresh_r) capture the part of the decline that
can be predicted, the wave-year freshmen who run well behind it.

sigma splits into three parts, sigma^2 = sigma_national^2 + sigma_state^2 +
sigma_district^2: a national miss shared by every race in a cycle (the
spread of each cycle's average residual), a state miss shared by the races
in one state that cycle, and the rest. The state part is estimated from how
much each state's average residual varies beyond what its number of races
explains: E[n_g * mean_g^2] = n_g * sigma_state^2 + sigma_district^2 over
state-cycle groups g. Each race's own win probability only needs the total;
the split matters for the seat total, because a miss moves many races at
once.

Uncontested races (|margin| >= 80) carry no information about competition
and are left out of the fit.
"""

import numpy as np
import pandas as pd

HISTORY = ["fresh_d", "fresh_r", "over_inc", "over_open"]
FEATURES = ["pvi", "d_incumbent", "open_seat", "floor_share"] + HISTORY
UNCONTESTED = 80


def fit_baseline(races, features=FEATURES):
    """{'coef': {name: value}, 'sigma', 'sigma_national', 'sigma_state',
    'sigma_district', 'n', 'cycles'}"""
    train = races[races.margin.abs() < UNCONTESTED]
    X = np.column_stack([np.ones(len(train))] + [train[f].to_numpy(float) for f in features])
    y = (train.margin - train.gb).to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = pd.Series(y - X @ coef)
    sigma = float(np.sqrt(resid @ resid / (len(y) - X.shape[1])))
    cycle = train.cycle.to_numpy()
    national = float(resid.groupby(cycle).mean().std())
    within = resid - resid.groupby(cycle).transform("mean")
    groups = within.groupby([cycle, train.district_id.str[:2].to_numpy()])
    n, means = groups.size(), groups.mean()
    district_var = float(((within - groups.transform("mean")) ** 2).sum() / (len(within) - len(n)))
    state = float(np.sqrt(max(((n * means ** 2).sum() - len(n) * district_var) / n.sum(), 0.0)))
    return {"coef": dict(zip(["intercept"] + features, coef)), "sigma": sigma,
            "sigma_national": national, "sigma_state": state,
            "sigma_district": float(np.sqrt(sigma ** 2 - national ** 2 - state ** 2)),
            "n": len(y), "cycles": sorted(int(c) for c in train.cycle.unique())}
