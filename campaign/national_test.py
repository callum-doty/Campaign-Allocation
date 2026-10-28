"""
Does household stress explain where the House vote lands relative to the
generic ballot? A national test over 2002-2024, where the generic ballot is
the final RealClearPolitics average every year (data/generic_ballot.csv).
RealClearPolitics averages start in 2002. Earlier years have only single
polls from different pollsters, so they are left out rather than mixed in.

    miss_t = house_vote_t - gb_t                     (Democratic margin, points)
    model 0: miss_t = a
    model 1: miss_t = a + b * s_t                    s_t = +1 Democratic president, -1 Republican
    model 2: miss_t = a + b * s_t + c * s_t * Z_t    Z_t = a household-stress composite

a is how far the polls lean on average, b how the president's party runs
against its polls, and c whether it runs further behind them when household
stress is high (the hypothesis: c < 0). Fixed before the comparison was run:

- Z is each composite in household_index.py, June over June of the election year.
- Each model is scored two ways: leaving one election out at a time, and
  forward in time (each year from 2012 predicted from 2002 up to the year
  before).
- A composite earns a place in the forecast only if model 2 beats model 1
  both ways, and shuffling Z across elections gives a c at least as
  negative as the fitted one less than 10% of the time.
- A benchmark with no economic data: model 1 plus m * s_t * midterm_t, the
  president's party losing ground in midterm years.

The same test runs against the generic ballot on Sept 1 (in
data/generic_ballot_dated.csv), the polls a September decision actually sees: RealClearPolitics'
own average that day, or for 2002-2006, which have no average for that date,
the mean spread of each nonpartisan pollster's latest poll ending in the 30
days before (extended a week at a time until there are three pollsters).
Where both exist (2008-2024, but 2022's list doesn't reach back to August),
that rule lands within about a point of RealClearPolitics' own average.

The September version also asks the question directly (economics_only): does
a ballot built from economics alone, the president's party's margin from a
midterm penalty and a composite, predict the House vote better than the
September polls (model 0 on the September miss)?
"""

import numpy as np
import pandas as pd

from .build_races import GENERIC_BALLOT, GENERIC_BALLOT_DATED
from .household_index import household_index
from .national_vote import national_vote

CYCLES = tuple(range(2002, 2025, 2))
FORWARD_FROM = 2012
PRESIDENT = {2002: "R", 2004: "R", 2006: "R", 2008: "R", 2010: "D", 2012: "D", 2014: "D", 2016: "D",
             2018: "R", 2020: "R", 2022: "D", 2024: "D", 2026: "R"}
COMPOSITES = ("basket", "stress")
PASS_P = 0.10


def national_table(cycles=CYCLES):
    """One row per election: gb and gb_sept (final and Sept 1 generic
    ballot), house_vote, miss and miss_sept (vote minus each), president,
    s (+1 D, -1 R), midterm (1 or 0), and each composite (household_index)."""
    ballot = pd.read_csv(GENERIC_BALLOT).set_index("cycle")
    dated = pd.read_csv(GENERIC_BALLOT_DATED, dtype={"date": str})
    sept1 = dated[dated.date == "09-01"].set_index("cycle").generic_ballot
    table = pd.DataFrame({"gb": ballot.generic_ballot.reindex(cycles), "gb_sept": sept1.reindex(cycles),
                          "house_vote": national_vote(cycles)})
    table["miss"] = table.house_vote - table.gb
    table["miss_sept"] = table.house_vote - table.gb_sept
    table["midterm"] = (table.index % 4 == 2).astype(float)
    table["president"] = table.index.map(PRESIDENT)
    table["s"] = np.where(table.president == "D", 1.0, -1.0)
    return table.join(household_index(cycles))


def national_test(table, composites=COMPOSITES, n_perm=20_000, seed=0):
    """Root-mean-square error of each model's predicted miss, left-one-out
    ('loo') and forward in time ('forward'): models 0 and 1 under
    'constant' and 'president', model 2 under 'composites'[name], which also
    holds its full-sample coefficients ('coef': a, b, c), the permutation
    p-value for c ('p'), whether it passes, and how the composite moves with
    the president's party's own generic-ballot margin ('gb_corr')."""
    y, s, cycles = table.miss.to_numpy(float), table.s.to_numpy(float), table.index.to_numpy()

    def design(*columns):
        return np.column_stack([np.ones(len(y)), *columns])

    def fit(X, rows):
        return np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]

    def scores(X):
        loo = [y[i] - X[i] @ fit(X, np.arange(len(y)) != i) for i in range(len(y))]
        forward = [y[i] - X[i] @ fit(X, cycles < c) for i, c in enumerate(cycles) if c >= FORWARD_FROM]
        return {"loo": float(np.sqrt(np.mean(np.square(loo)))), "forward": float(np.sqrt(np.mean(np.square(forward))))}

    out = {"constant": scores(design()), "president": scores(design(s)),
           "midterm": scores(design(s, s * table.midterm.to_numpy(float))), "composites": {}}
    rng = np.random.default_rng(seed)
    everything = np.ones(len(y), bool)
    for name in composites:
        z = table[name].to_numpy(float)
        coef = fit(design(s, s * z), everything)
        shuffled = np.array([fit(design(s, s * rng.permutation(z)), everything)[2] for _ in range(n_perm)])
        result = {**scores(design(s, s * z)), "coef": dict(zip("abc", map(float, coef))),
                  "p": float(np.mean(shuffled <= coef[2])),
                  "gb_corr": float(np.corrcoef(z, s * table.gb.to_numpy(float))[0, 1])}
        result["passes"] = bool(result["loo"] < out["president"]["loo"] and
                                result["forward"] < out["president"]["forward"] and result["p"] < PASS_P)
        out["composites"][name] = result
    return out


def economics_only(table, composites=COMPOSITES):
    """Root-mean-square error predicting the House vote from economics
    alone, left-one-out ('loo') and forward in time ('forward'): the
    president's party's margin from a midterm penalty ('midterm'), then
    also each composite (by name)."""
    y = (table.s * table.house_vote).to_numpy(float)    # the president's party's margin; errors are the same either way
    cycles = table.index.to_numpy()

    def scores(*columns):
        X = np.column_stack([np.ones(len(y)), *columns])
        fit = lambda rows: np.linalg.lstsq(X[rows], y[rows], rcond=None)[0]
        loo = [y[i] - X[i] @ fit(np.arange(len(y)) != i) for i in range(len(y))]
        forward = [y[i] - X[i] @ fit(cycles < c) for i, c in enumerate(cycles) if c >= FORWARD_FROM]
        return {"loo": float(np.sqrt(np.mean(np.square(loo)))), "forward": float(np.sqrt(np.mean(np.square(forward))))}

    midterm = table.midterm.to_numpy(float)
    return {"midterm": scores(midterm), **{name: scores(midterm, table[name].to_numpy(float)) for name in composites}}
