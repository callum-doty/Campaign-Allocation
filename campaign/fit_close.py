"""
Job 1 with an incumbency edge of its own where the race is expected to be
close, to ask whether incumbents there deserve less credit than the pooled
effect gives them:

    margin_i - GB = (the baseline's terms) + q * inc_i * close_i + error_i

inc is +1 for a Democratic incumbent, -1 for a Republican one and 0 for an
open seat; close_i is 1 where the baseline forecast is within CLOSE points
(in sample for the training years, as a forecaster would see it for the year
ahead). q is how much more (positive) or less (negative) the edge is worth
each way in those races than in the rest. It is fit on the same cycles as
the baseline, so like the baseline it can't see incumbency shrinking.
"""

import numpy as np

from .fit_baseline import FEATURES, fit_baseline
from .predict_margin import predict_margin

CLOSE = 10          # points of forecast margin


def close_inc(races, model, close=CLOSE):
    """inc x close for each row of `races`, with closeness judged by the baseline `model`."""
    inc = races.d_incumbent - (races.status == "R incumbent")
    return (inc * (np.abs(predict_margin(races, model)) < close)).to_numpy(float)


def fit_close(train, model, close=CLOSE):
    """The baseline refit on `train` with an inc_close term ('coef'['inc_close'] is q). Predict with
    predict_margin(races.assign(inc_close=close_inc(races, model)), fitted)."""
    return fit_baseline(train.assign(inc_close=close_inc(train, model, close)), FEATURES + ["inc_close"])
