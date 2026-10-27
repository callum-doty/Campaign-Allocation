"""
Job 2: how many margin points party money buys in a race.

    shift(x) = scale_pts * log(1 + x / half_saturation)

Diminishing returns in absolute dollars: the first dollar buys
scale_pts / half_saturation points, and that marginal effect halves once
half_saturation dollars are in. Districts have equal populations by law, so
dollars stand in for dollars per voter.

This is an ASSUMPTION, not an estimate. Public data can't separate the
effect of money from the reasons money goes where it goes (strong candidates
and close races attract it). An earlier model used each side's SHARE of
spending instead, which made a dollar's effect scale with 1 / (money already
in the race): $100K in a quiet long-shot race bought ten times the margin of
$100K in a saturated toss-up, and the optimizer poured money into long shots.
Results are reported across a range of both parameters; measured effects
(e.g. randomized holdouts on mail programs) should replace the defaults.

Defaults: the first $1M moves the margin about 0.25 points, and $15M (the
most any race takes) about 2 points.
"""

import numpy as np

SCALE_PTS = 1.5
HALF_SATURATION = 5_500_000


def spending_effect(dollars, scale_pts=SCALE_PTS, half_saturation=HALF_SATURATION):
    return scale_pts * np.log1p(np.maximum(np.asarray(dollars, float), 0) / half_saturation)
