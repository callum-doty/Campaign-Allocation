"""
Spend a budget where it buys the most expected seats.

    P(D wins i | x) = Phi((mu_i + shift(x_i) - shift(observed_i)) / sigma)

mu_i is the baseline forecast, which already reflects the money actually
spent, so only DEVIATIONS from the observed allocation move a race; at the
observed allocation every race is exactly at its baseline forecast.

Greedy: each step funds the race whose next chunk buys the most expected
seats per dollar. Each race's win probability is concave in its spending
except where z = (mu + shift) / sigma < -sigma / scale, i.e. races more
than ~40 points out at the default scale (the second derivative is
proportional to -(scale * z / sigma) - 1). So the objective is a sum of
concave functions under a budget constraint, and small greedy steps are
discrete KKT water-filling: they reach the global optimum, within 0.01
seats of a numerical solver on the real backtests. The larger chunk sizes
only matter in that convex tail under steep spending assumptions.

Objectives that weigh outcomes across national environments (a majority
probability, tail risk) need the shared errors (seat_distribution.py);
expected seats don't, because averaging Phi((mu + eta + u) / sigma_district)
over the national and state swings eta and u gives back Phi(mu / sigma).
"""

import numpy as np
from scipy.stats import norm

from .spending_effect import spending_effect

CHUNKS = (1, 2, 5, 10, 20, 50)


def allocate(mu, sigma, observed, budget, cap, step=100_000, **effect):
    """Dollars per race (at most `cap` each, in multiples of `step`).
    `effect` is passed to spending_effect (scale_pts, half_saturation), and
    either may be one value per race (e.g. media-market cost)."""
    mu, observed = np.asarray(mu, float), np.asarray(observed, float)
    base = spending_effect(observed, **effect)
    x, remaining = np.zeros_like(mu), float(budget)
    p = norm.cdf((mu + spending_effect(x, **effect) - base) / sigma)
    while remaining >= step:
        best = (0.0, None, 0.0)                         # (seats per dollar, race, amount)
        for k in CHUNKS:
            amount = k * step
            if amount > remaining:
                break
            fits = x + amount <= cap
            gain = norm.cdf((mu + spending_effect(x + amount, **effect) - base) / sigma) - p
            rate = np.where(fits, gain / amount, -np.inf)
            i = int(np.argmax(rate))
            if rate[i] > best[0]:
                best = (rate[i], i, amount)
        if best[1] is None:
            break
        i, amount = best[1], best[2]
        x[i] += amount
        remaining -= amount
        p = norm.cdf((mu + spending_effect(x, **effect) - base) / sigma)
    return x
