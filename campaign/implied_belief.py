"""
What DCCC's own allocation implies, read backwards from the condition for
an optimum: at a seat-maximizing allocation, the last dollar in every funded
race buys the same expected seats. At the observed allocation the forecast
already reflects the money spent, so one more dollar in race i buys

    m_i = phi(mu_i / sigma) / sigma * f'(observed_i),   f'(x) = scale / (half_saturation + x)

expected seats. Averaging m over DCCC's dollars in each band of races (by
forecast chance of a Democratic win) and dividing the 20-80% band's value by
each band's gives the multiplier: how many times as far a dollar there would
have to move the margin for DCCC's split between that band and the 20-80%
races to be optimal. The same number reads as how many times as much those
seats must be worth to DCCC, or as a sign the forecast understates their
risk; this data can't tell those apart.

The scale cancels in the ratio, so the multiplier doesn't depend on how
strong party money is assumed to be, only on the curve's shape
(half_saturation) and the forecast.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .spending_effect import HALF_SATURATION, SCALE_PTS

BANDS = ("under 20%", "20-80%", "80%+")


def implied_belief(mu, sigma, observed, half_saturation=HALF_SATURATION, scale_pts=SCALE_PTS):
    """One row per band: DCCC's dollars there, the expected seats one more
    $1M buys there at the margin (averaged over those dollars), and the
    multiplier relative to the 20-80% band."""
    mu, observed = np.asarray(mu, float), np.asarray(observed, float)
    p = norm.cdf(mu / sigma)
    band = np.where(p < 0.2, BANDS[0], np.where(p > 0.8, BANDS[2], BANDS[1]))
    marginal = norm.pdf(mu / sigma) / sigma * scale_pts / (half_saturation + observed)
    out = (pd.DataFrame({"band": band, "dollars": observed, "weighted": observed * marginal * 1e6})
           .groupby("band").sum().reindex(list(BANDS)))
    out["per_million"] = out.weighted / out.dollars
    out["multiplier"] = out.per_million["20-80%"] / out.per_million
    return out[["dollars", "per_million", "multiplier"]]
