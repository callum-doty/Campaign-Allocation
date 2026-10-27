"""
Expected seats gained by re-spending the observed budget the best way:

    gain = sum_i Phi((mu_i + f(x_i) - f(observed_i)) / sigma) - sum_i Phi(mu_i / sigma)

where x is allocate()'s optimum for the same total under the same curve f.
"""

import numpy as np
from scipy.stats import norm

from .allocate import allocate
from .spending_effect import spending_effect


def reallocation_gain(mu, sigma, observed, cap, **effect):
    """(gain in expected seats, best allocation). `effect` is passed to
    spending_effect (scale_pts, half_saturation)."""
    mu, observed = np.asarray(mu, float), np.asarray(observed, float)
    x = allocate(mu, sigma, observed, observed.sum(), cap, **effect)
    shift = spending_effect(x, **effect) - spending_effect(observed, **effect)
    return float(norm.cdf((mu + shift) / sigma).sum() - norm.cdf(mu / sigma).sum()), x
