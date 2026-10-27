"""
The distribution of the Democratic seat total, with the national polling
error shared by every race:

    eta ~ Normal(0, sigma_national^2)                      one draw per election
    P(D wins i | eta) = Phi((mu_i + eta) / sigma_district)
    seats | eta = sum of independent wins ~ approx. Normal(sum p, sum p(1-p))

Integrating over eta (Gauss-Hermite quadrature) gives a mixture that is
wider than the independent-races version: in a polling miss, many close
races go the same way at once. Expected seats are identical either way.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

NODES = 41


def seat_distribution(mu, model, need):
    """`mu`: expected D margins for the modeled races; `need`: seats the
    party needs among them for a majority. Returns p_majority, mean, the
    10th/50th/90th percentiles, and pmf (Series indexed by seat count)."""
    mu = np.asarray(mu, float)
    eta, weights = np.polynomial.hermite_e.hermegauss(NODES)
    weights = weights / weights.sum()
    p = norm.cdf((mu[:, None] + model["sigma_national"] * eta[None]) / model["sigma_district"])
    mean, sd = p.sum(0), np.sqrt((p * (1 - p)).sum(0))
    seats = np.arange(len(mu) + 1)
    pmf = (norm.cdf((seats[:, None] + 0.5 - mean) / sd) - norm.cdf((seats[:, None] - 0.5 - mean) / sd)) @ weights
    pmf = pd.Series(pmf / pmf.sum(), index=seats)
    cdf = pmf.cumsum()
    p10, p50, p90 = (int(cdf.index[cdf.searchsorted(q)]) for q in (0.1, 0.5, 0.9))
    return {"p_majority": float(pmf[pmf.index >= need].sum()), "mean": float(mean @ weights),
            "p10": p10, "p50": p50, "p90": p90, "need": int(need), "pmf": pmf}
