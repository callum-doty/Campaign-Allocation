"""
The distribution of the Democratic seat total, with the national polling
error shared by every race and a state error shared by the races in each
state:

    eta ~ sigma_national * t_df                            one draw per election
    u_s ~ Normal(0, sigma_state^2)                         one draw per state
    P(D wins i | eta, u) = Phi((mu_i + eta + u_s(i)) / sigma_district)

sigma_national is the spread of only a handful of cycle averages, so it
isn't treated as known. A normal miss averaged over how uncertain
sigma_national is (df = cycles - 1) is a Student t with df degrees of
freedom: a big miss is more likely than a normal with the point estimate
says, and the fewer cycles behind it, the more so. This covers how big
misses are, not the smaller uncertainty in where they center. Without a
df, eta ~ Normal(0, sigma_national^2).

Given eta, states are independent. Each state's seat count has a mean and
variance found by integrating over its own draw u_s (law of total
variance), and the seat total given eta is approximately Normal with the
summed mean and variance. Integrating over eta (Gauss-Hermite quadrature,
mapped onto the t through its quantiles) gives a mixture that is wider
than the independent-races version: in a polling miss, many close races go
the same way at once. With a normal eta, expected seats are identical
either way. The t's fatter tails pull each race a little toward even, so
the expected total can differ slightly from summing Phi(mu_i / sigma).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t

NODES = 121            # enough for a t with 1 df (a backtest fit on two cycles)
STATE_NODES = 21


def seat_distribution(mu, model, need, states=None):
    """`mu`: expected D margins for the modeled races; `need`: seats the
    party needs among them for a majority; `states`: each race's state
    (without it, or without a sigma_state in `model`, races share only the
    national error); `model['national_df']`, if set, makes the national
    miss a t with that many degrees of freedom. Returns p_majority, mean,
    the 10th/50th/90th percentiles, and pmf (Series indexed by seat count)."""
    mu = np.asarray(mu, float)
    group = pd.factorize(np.asarray(states))[0] if states is not None else np.arange(len(mu))
    z, weights = np.polynomial.hermite_e.hermegauss(NODES)
    weights = weights / weights.sum()
    df = model.get("national_df")
    eta = z if df is None else -np.sign(z) * t.ppf(norm.sf(np.abs(z)), df)    # the t quantile at Phi(z)
    u, u_weights = np.polynomial.hermite_e.hermegauss(STATE_NODES)
    u_weights = u_weights / u_weights.sum()
    shift = model["sigma_national"] * eta[None, :, None] + model.get("sigma_state", 0.0) * u[None, None, :]
    p = norm.cdf((mu[:, None, None] + shift) / model["sigma_district"])     # race x national x state draw
    seats_by_state, var_by_state = (np.zeros((group.max() + 1,) + p.shape[1:]) for _ in range(2))
    np.add.at(seats_by_state, group, p)
    np.add.at(var_by_state, group, p * (1 - p))
    state_mean = seats_by_state @ u_weights
    state_var = (var_by_state + seats_by_state ** 2) @ u_weights - state_mean ** 2
    mean = state_mean.sum(0)
    sd = np.sqrt(np.maximum(state_var.sum(0), 1e-12))     # a miss far in the tail settles every race
    seats = np.arange(len(mu) + 1)
    pmf = (norm.cdf((seats[:, None] + 0.5 - mean) / sd) - norm.cdf((seats[:, None] - 0.5 - mean) / sd)) @ weights
    pmf = pd.Series(pmf / pmf.sum(), index=seats)
    cdf = pmf.cumsum()
    p10, p50, p90 = (int(cdf.index[cdf.searchsorted(q)]) for q in (0.1, 0.5, 0.9))
    return {"p_majority": float(pmf[pmf.index >= need].sum()), "mean": float(mean @ weights),
            "p10": p10, "p50": p50, "p90": p90, "need": int(need), "pmf": pmf}
