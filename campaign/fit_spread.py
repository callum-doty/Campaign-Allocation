"""
Job 1 with each race's error size set by its features. The expected margin
is the baseline's; only how far the result can land from it changes:

    sigma_i^2 = sigma_national^2 + (sigma^2 - sigma_national^2) * m_i
    log m_i = g1*Open_i + g2*redrawn_inc_i + g3*|PVI_i|/10 + constant

redrawn_inc is 1 for an incumbent on lines changed since the last election.
The national miss is shared by every race in a cycle, so the features scale
only the rest of the error, the state and district parts. The constant makes
m average 1 over the training races: the features move uncertainty from one
race to another without changing the total, so any gain in the backtest
comes from which races get it.

The g's are fit by maximum likelihood on the baseline's residuals less each
cycle's average (the national miss), with each cycle's own error size free.
Otherwise a term that fell mostly in one cycle (every incumbent was on new
lines in 2012) would read that cycle's error size as its own. The
redrawn-lines term follows fit_drift.py's rule: it is left out until a
training cycle has incumbents on both new and old lines. Standard errors
are clustered by state and cycle.
"""

import numpy as np
import pandas as pd

from .fit_baseline import UNCONTESTED
from .predict_margin import predict_margin

TERMS = ["open_seat", "redrawn_inc", "lean"]


def spread_terms(races, terms):
    cols = {"open_seat": races.open_seat, "redrawn_inc": races.inc_redrawn != 0, "lean": races.pvi.abs() / 10}
    return np.column_stack([cols[t].to_numpy(float) for t in terms])


def fit_spread(races, model):
    """For the baseline `model` fit on `races`: {'g', 'g_se'} (each term's
    log multiplier on the non-national error variance, per unit), and what
    race_sigma needs."""
    train = races[races.margin.abs() < UNCONTESTED]
    resid = pd.Series(train.margin.to_numpy() - predict_margin(train, model))
    cycle = train.cycle.to_numpy()
    d2 = ((resid - resid.groupby(cycle).transform("mean")) ** 2).to_numpy()
    inc, on_new = train.open_seat.to_numpy() == 0, train.redrawn.to_numpy() == 1
    identified = any((inc & on_new & here).any() and (inc & ~on_new & here).any()
                     for here in (cycle == c for c in np.unique(cycle)))
    terms = TERMS if identified else [t for t in TERMS if t != "redrawn_inc"]
    fe = (cycle[:, None] == np.unique(cycle)).astype(float)
    W = np.column_stack([fe, spread_terms(train, terms)])
    theta = np.r_[np.full(fe.shape[1], np.log(d2.mean())), np.zeros(len(terms))]
    for _ in range(100):                                   # Fisher scoring for log-variance
        step = np.linalg.solve(W.T @ W, W.T @ (d2 / np.exp(W @ theta) - 1))
        theta += step
        if np.abs(step).max() < 1e-10:
            break
    group = pd.factorize(train.district_id.str[:2] + train.cycle.astype(str))[0]
    score = np.zeros((group.max() + 1, W.shape[1]))
    np.add.at(score, group, 0.5 * W * (d2 / np.exp(W @ theta) - 1)[:, None])
    bread = 2 * np.linalg.inv(W.T @ W)
    se = np.sqrt(np.diag(bread @ score.T @ score @ bread))[fe.shape[1]:]
    g = theta[fe.shape[1]:]
    return {"terms": terms, "g": dict(zip(terms, g)), "g_se": dict(zip(terms, se)),
            "level": float(np.exp(spread_terms(train, terms) @ g).mean()),
            "sigma": model["sigma"], "sigma_national": model["sigma_national"]}


def race_sigma(races, spread):
    """Each race's sigma under a fit_spread fit."""
    m = np.exp(spread_terms(races, spread["terms"]) @ np.array(list(spread["g"].values()))) / spread["level"]
    return np.sqrt(spread["sigma_national"] ** 2 + (spread["sigma"] ** 2 - spread["sigma_national"] ** 2) * m)
