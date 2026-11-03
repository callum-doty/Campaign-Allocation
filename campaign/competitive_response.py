"""
Do DCCC and NRCC respond to each other, or to the same information? The two-player
game in solve_game.py assumes how each side would respond. This measures what they did.

A week-by-week panel of each race's committee spending from Aug 1 to the election:
this week's money (in inverse hyperbolic sines, so zero is allowed and differences
read roughly as percentages) on the other side's money last week and next week, its
own last week, with race and cycle-week effects, standard errors clustered by race.

Next week's money is the check. If the other side's spending last week predicts this
week but next week's doesn't, that looks like a response. If next week's predicts as
well, both are following the same news or plans, since ads are booked ahead and dates
here are when they ran. It can't separate those any further.

First movers: in races both committees entered, how far apart their first dates fall,
against the same dates shuffled among races in the cycle.
"""

import numpy as np
import pandas as pd

from .load_expenditures import load_expenditures

START = "08-01"
SHUFFLES = 500


def weekly_panel(cycle, race_ids=None):
    """race x week: dccc, nrcc, outside_d, outside_r dollars, for races with any committee money in the window."""
    ie = load_expenditures(cycle).dropna(subset=["date"])
    ie = ie[(ie.date >= pd.Timestamp(f"{cycle}-{START}")) & (ie.date <= pd.Timestamp(f"{cycle}-11-30"))]
    if race_ids is not None:
        ie = ie[ie.district_id.isin(race_ids)]
    ie = ie.assign(week=((ie.date - pd.Timestamp(f"{cycle}-{START}")).dt.days // 7).astype(int))
    wide = ie.pivot_table(index=["district_id", "week"], columns="bucket", values="amount", aggfunc="sum", fill_value=0.0).reindex(
        columns=["dccc", "nrcc", "outside_d", "outside_r"], fill_value=0.0)
    active = wide.groupby(level=0)[["dccc", "nrcc"]].sum().sum(axis=1) > 0
    races, weeks = active[active].index, range(int(ie.week.max()) + 1)
    return wide.reindex(pd.MultiIndex.from_product([races, weeks], names=["district_id", "week"]), fill_value=0.0)


def two_way_demean(x, race, week, steps=30):
    x = x.copy()
    for _ in range(steps):
        x = x - x.groupby(race).transform("mean")
        x = x - x.groupby(week).transform("mean")
    return x


def response(panels, own, other):
    """{'lag','lead','own_lag': (coefficient, se)}: `own` this week on `other` last and next week and `own` last week."""
    rows = []
    for cycle, p in panels.items():
        p = np.arcsinh(p[[own, other]]).reset_index()
        by = p.groupby("district_id")
        rows.append(p.assign(other_lag=by[other].shift(1), other_lead=by[other].shift(-1), own_lag=by[own].shift(1), race=p.district_id + str(cycle), cw=str(cycle) + "w" + p.week.astype(str)).dropna())
    d = pd.concat(rows, ignore_index=True)
    names = ["other_lag", "other_lead", "own_lag"]
    data = two_way_demean(d[[own] + names], d.race, d.cw)
    X, y = data[names].to_numpy(float), data[own].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    group = pd.factorize(d.race)[0]
    score = np.zeros((group.max() + 1, X.shape[1]))
    np.add.at(score, group, X * (y - X @ beta)[:, None])
    bread = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(bread @ score.T @ score @ bread))
    return {"lag": (beta[0], se[0]), "lead": (beta[1], se[1]), "own_lag": (beta[2], se[2]), "n": len(d), "races": d.race.nunique()}


def first_movers(cycles, seed=0):
    """Races both committees entered: n, DCCC's share of moving first, median gap in days, share within 14 days, and that
    share with one committee's first dates shuffled among the cycle's races."""
    rng, frames = np.random.default_rng(seed), []
    for cycle in cycles:
        ie = load_expenditures(cycle).dropna(subset=["date"])
        first = ie[ie.bucket.isin(["dccc", "nrcc"])].groupby(["district_id", "bucket"]).date.min().unstack().dropna()
        frames.append(first.assign(cycle=cycle))
    both = pd.concat(frames)
    gap = (both.dccc - both.nrcc).dt.days
    shuffled = []
    for _ in range(SHUFFLES):
        moved = pd.concat([g.nrcc.sample(frac=1, random_state=int(rng.integers(1 << 31))).set_axis(g.index) for _, g in both.groupby("cycle")])
        shuffled.append(((both.dccc - moved).dt.days.abs() <= 14).mean())
    return {"n": len(both), "dccc_first": float((gap < 0).mean()), "tie": float((gap == 0).mean()), "median_gap": float(gap.abs().median()),
            "within_14": float((gap.abs() <= 14).mean()), "shuffled_within_14": float(np.mean(shuffled)), "shuffled_high": float(np.quantile(shuffled, 0.975))}
