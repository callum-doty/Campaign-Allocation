"""
How strong party money would have to be for reallocating DCCC's budget to
gain a full expected seat: the smallest scale_pts at which
reallocation_gain() reaches `target`, at a given half-saturation.

This turns the spending assumption around. Instead of "at 1.5 points,
reallocating gains a tenth of a seat", it reads "reallocating is worth a
seat only if party money is at least this strong", which can be held up
against measured effects.

The gain never falls as the scale rises. Take the best allocation x at one
scale and a scale k times as strong: moving only 1/k of the way from the
observed allocation toward x spends the same budget, and because the curve
is concave it moves every race at least as far as x did at the weaker
scale. So the target is bracketed by doubling and found by root-finding.

Far above the default scale more races sit in allocate()'s convex tail,
where greedy steps can fall short of the optimum, which would make the
break-even an upper bound. On the real backtests it is a tight one: at the
break-even scale, a local solver started from the greedy allocation gains
under 0.01 seats more.
"""

import numpy as np
from scipy.optimize import brentq

from .reallocation_gain import reallocation_gain

MAX_SCALE = 64.0          # about 40x the default; beyond this, report inf


def break_even_scale(mu, sigma, observed, cap, half_saturation, target=1.0):
    """Smallest scale_pts (to within 1%) at which reallocating `observed`
    gains `target` expected seats, or inf if it takes more than MAX_SCALE."""
    def short(scale):
        gain, _ = reallocation_gain(mu, sigma, observed, cap, scale_pts=scale, half_saturation=half_saturation)
        return gain - target

    lo, hi = 0.0, 1.0
    while short(hi) < 0:
        if hi >= MAX_SCALE:
            return np.inf
        lo, hi = hi, 2 * hi
    return float(brentq(short, lo, hi, rtol=0.01))
