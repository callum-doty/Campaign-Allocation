"""
Job 1 with an incumbency effect that drifts from cycle to cycle, and a term
for incumbents on redrawn lines. Everything else stays pooled, as in the
baseline:

    margin_i - GB = alpha_t + u_t*inc_i + o*Open_i + r*inc_redrawn_i + (the baseline's other terms) + error_i
    u_t = u_(t-1) + eta_t,   eta_t ~ Normal(0, tau^2)

inc is +1 for a Democratic incumbent, -1 for a Republican one and 0 for an
open seat, so u_t is the edge each way and 2*u_t is the baseline's c (the
worth of having the Democrat rather than the Republican as the incumbent);
o is what's left of the baseline's open-seat term. alpha_t takes each
cycle's national miss out of the fit, or a cycle with more Democratic
incumbents would read its miss as a change in incumbency. The forecast uses
the average cycle, as the baseline's intercept does.

inc_redrawn is inc where the state's lines changed since the last election.
No history carries over there (over_inc is zero), so r is how far an
incumbent on new lines runs ahead of one on old lines once last cycle's
result is counted. With each cycle's u_t free, r is learned within cycles:
from years when some states redrew and others didn't, and from the few
unchanged states in 2022. Before any such year (training on 2012-2014
only) it is left out.

Each cycle's u_t is estimated together with everything else, with standard
errors clustered by state and cycle; a Kalman filter then runs the random
walk over them. tau is the value that makes
those estimates most likely, from the training cycles alone. The forecast
takes the filter's last value, and its uncertainty (the filter's own plus
one more step of drift) joins the district-level error, since a wrong u
moves incumbents of the two parties in opposite directions.
"""

import numpy as np
import pandas as pd

from .fit_baseline import FEATURES, UNCONTESTED, split_error

POOLED = [f for f in FEATURES if f != "d_incumbent"] + ["inc_redrawn"]
TAUS = np.arange(0, 8.001, 0.05)       # drift per cycle tried for u_t, points of margin each way


def random_walk(estimates, variances, tau):
    """Local-level Kalman filter over noisy estimates of a random walk:
    (log likelihood, last filtered mean, its variance)."""
    m, P, loglik = estimates[0], variances[0], 0.0
    for y, v in zip(estimates[1:], variances[1:]):
        P += tau ** 2
        F = P + v
        loglik -= 0.5 * (np.log(F) + (y - m) ** 2 / F)
        m, P = m + P / F * (y - m), P * v / F
    return loglik, m, P


def fit_drift(races):
    """fit_baseline's output for the next cycle, its coef in the baseline's
    terms plus inc_redrawn, and 'incumbency': each training cycle's c = 2*u_t
    with its standard error, tau and the forecast's c on the same scale, and
    the redrawn-lines term r (None if no training cycle can tell)."""
    train = races[races.margin.abs() < UNCONTESTED]
    inc = (train.d_incumbent - (train.status == "R incumbent")).to_numpy(float)
    cycles = sorted(train.cycle.unique())
    on_new = train.redrawn.to_numpy() == 1
    identified = any((inc[on_new & here] != 0).any() and (inc[~on_new & here] != 0).any()
                     for here in (train.cycle.to_numpy() == c for c in cycles))
    pooled = POOLED if identified else POOLED[:-1]
    fe = (train.cycle.to_numpy()[:, None] == np.array(cycles)).astype(float)
    X = np.column_stack([fe, fe * inc[:, None]] + [train[f].to_numpy(float) for f in pooled])
    y = (train.margin - train.gb).to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # races in one state miss together (fit_baseline's sigma_state), so standard errors cluster by state and cycle
    bread = np.linalg.inv(X.T @ X)
    group = pd.factorize(train.district_id.str[:2] + train.cycle.astype(str))[0]
    score = np.zeros((group.max() + 1, X.shape[1]))
    np.add.at(score, group, X * resid[:, None])
    cov = bread @ score.T @ score @ bread
    k = len(cycles)
    alpha, u, var = beta[:k], beta[k:2 * k], np.diag(cov)[k:2 * k]
    tau = max(TAUS, key=lambda t: random_walk(u, var, t)[0])
    _, u_next, p_next = random_walk(u, var, tau)
    var_next = p_next + tau ** 2

    a = float(fe.mean(axis=0) @ alpha)                     # the average cycle, weighted by races
    model = split_error(train, resid + fe @ alpha - a, X.shape[1])
    extra = var_next * float((inc != 0).mean())
    model["sigma"] = float(np.sqrt(model["sigma"] ** 2 + extra))
    model["sigma_district"] = float(np.sqrt(model["sigma_district"] ** 2 + extra))
    b = dict(zip(pooled, beta[2 * k:]))
    model["coef"] = {"intercept": a - u_next, "d_incumbent": 2 * u_next, "open_seat": u_next + b.pop("open_seat"), **b}
    i = 2 * k + pooled.index("inc_redrawn") if identified else None
    model["incumbency"] = {"cycles": [int(c) for c in cycles], "c": 2 * u, "c_se": 2 * np.sqrt(var),
                           "tau": 2 * tau, "c_next": 2 * u_next, "c_next_se": 2 * np.sqrt(var_next),
                           "redrawn": float(beta[i]) if identified else None,
                           "redrawn_se": float(np.sqrt(cov[i, i])) if identified else None}
    return model
