"""
When a dollar reaches voters, relative to when they vote: Job 2's clock.

Persuasion from campaign advertising fades within days. In a randomized TV
experiment, a buy that moved a governor's race about 5 points while it aired
had no detectable effect the week after (Gerber, Gimpel, Green and Shaw 2011,
APSR). Across the 2000 presidential and 2006 statewide races, effects had
half-lives of about 3-4 days (presidential) and 1-2 days (state and local)
(Hill, Lo, Vavreck and Zaller 2013, Political Communication). So a dollar
counts by how much of its effect is left when each voter casts a ballot:

    w_s(D) = (1 - e_s) * 0.5^(D / h)                              Election Day voters
           + e_s * mean over v = 1..W of 0.5^((D - v) / h) [v <= D]  early voters

D = days before Election Day the dollar reached voters (nothing after
Election Day counts), e_s = the state's early-vote share (load_early_vote.py),
early voters spread evenly over the W = 21 days before Election Day, and h the
half-life: 4 days by default, a little more durable than the experiments'
state and local estimates, with 2 and 8 days as bounds. Any lasting effect
(name recognition, turnout) is left out.

A race's effective dollars are its dollars times w on the day each reached
voters, divided by DCCC's average w that cycle. So DCCC's whole budget keeps
the strength the spending curve assumes; only how well each dollar was timed,
relative to the rest, changes. Per race:

- e_obs: effective dollars per dollar, as DCCC timed them.
- e_best: what a dollar counts there spread evenly over the best two weeks
  for that state's ballots. Not the best single day: a race can absorb only
  so much airtime and mail at once.

An efficiency e is the same curve with half_saturation / e, so the
allocation, gain and implied-belief code take it unchanged.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .allocate import allocate
from .load_early_vote import early_vote_share
from .load_expenditures import load_expenditures
from .spending_effect import HALF_SATURATION, spending_effect

HALF_LIFE, WINDOW = 4.0, 21
HALF_LIVES = (2.0, 4.0, 8.0)
LATEST = 60                       # days before Election Day searched for the best timing
SPREAD = 14                       # the best timing spreads a race's money over this many days


def election_day(cycle):
    """The Tuesday after the first Monday in November."""
    first = pd.Timestamp(f"{cycle}-11-01")
    return first + pd.Timedelta(days=(7 - first.weekday()) % 7 + 1)


def persuasion_weight(days_before, early_share, half_life=HALF_LIFE, window=WINDOW):
    """w for dollars reaching voters `days_before` Election Day, in a state
    casting `early_share` of its ballots early (both broadcast)."""
    days = np.asarray(days_before, float)
    decay = lambda t: 0.5 ** (np.maximum(t, 0) / half_life)
    voted_after = np.arange(1, window + 1)                   # early voters, days before Election Day
    early = np.where(voted_after <= days[..., None], decay(days[..., None] - voted_after), 0.0).mean(axis=-1)
    return np.where(days >= 0, (1 - np.asarray(early_share)) * decay(days) + np.asarray(early_share) * early, 0.0)


def ballot_timing(cycle, races, half_life=HALF_LIFE, window=WINDOW):
    """DataFrame in the order of `races`: dollars (DCCC's), e_obs, e_best."""
    shares = early_vote_share(cycle)
    spent = load_expenditures(cycle).query("bucket == 'dccc'")
    spent = spent[spent.district_id.isin(races.district_id)]
    days = (election_day(cycle) - spent.date).dt.days.to_numpy()
    w = persuasion_weight(days, spent.district_id.str[:2].map(shares).to_numpy(), half_life, window)
    mean_w = np.average(w, weights=spent.amount)
    effective = (spent.amount * w).groupby(spent.district_id).sum() / mean_w
    dollars = spent.amount.groupby(spent.district_id).sum()
    days_out = np.arange(LATEST + SPREAD)
    best = {s: np.convolve(persuasion_weight(days_out, e, half_life, window), np.ones(SPREAD) / SPREAD, "valid").max()
            / mean_w for s, e in shares.items()}
    out = pd.DataFrame({"dollars": races.district_id.map(dollars).fillna(0.0).to_numpy(),
                        "e_best": races.district_id.str[:2].map(best).to_numpy()})
    e_obs = races.district_id.map(effective / dollars).to_numpy()
    out["e_obs"] = np.where(out.dollars > 0, e_obs, out.e_best)
    return out


def timed_gain(mu, sigma, observed, e_obs, e_best, cap, move=True, **effect):
    """(expected seats gained, allocation) with every dollar timed to its
    state's ballots: re-spending the same budget across races (move=True),
    or keeping DCCC's race totals and only re-timing them (move=False).
    Gains are against DCCC's money as it actually timed it."""
    observed, e_obs, e_best = (np.asarray(a, float) for a in (observed, e_obs, e_best))
    effect = {**effect, "half_saturation": effect.get("half_saturation", HALF_SATURATION) / e_best}
    equiv = observed * e_obs / e_best              # DCCC's money as it landed, in best-timed dollars
    x = allocate(mu, sigma, equiv, observed.sum(), cap, **effect) if move else observed
    shift = spending_effect(x, **effect) - spending_effect(equiv, **effect)
    return float(norm.cdf((np.asarray(mu) + shift) / sigma).sum() - norm.cdf(np.asarray(mu) / sigma).sum()), x
