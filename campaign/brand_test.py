"""
Is the generic ballot a survey of party brand more than of informed choice? If
most voters follow the label, two things should show in the results.

- Pass-through. The forecast moves every race 1:1 with the generic ballot. If polls
  overstate how far habit-driven voters will move, less than 1:1 should forecast better,
  perhaps only in presidential years, when more low-information voters turn out. Nationally
  that's the House vote on the polls, 2002-2024. By race it's fixed pass-throughs in the
  forward backtest, which shrink toward the training years' average ballot through the
  intercept. These are scored raw: taking out the year's miss would erase what's being tested.
- PVI by money. If voters with no candidate information default to the label, the district's
  lean should count for more, and the error be tighter, where little is spent. Under LOW_MONEY
  is measured on all money, which is hindsight and only describes, and on what a Sept 1
  forecaster sees, which is what's held to the bar in spending_signal.py.

Results don't say why a slope is steeper: low-money races also tend to have an incumbent and a
token challenger. Who answers polls and how much voters know take survey data, not this.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fit_baseline import FEATURES, UNCONTESTED, fit_baseline
from .fit_close import CLOSE
from .predict_margin import predict_margin
from .seat_distribution import seat_distribution
from .seats_needed import seats_needed
from .spending_signal import BACKTEST, forecast_year, log_loss

# name -> pass-through in (midterms, presidential years)
GRID = {"1:1": (1.0, 1.0), "0.90": (0.9, 0.9), "0.75": (0.75, 0.75), "0.50": (0.5, 0.5),
        "0.75, presidential years only": (1.0, 0.75), "0.50, presidential years only": (1.0, 0.5), "0.75, midterms only": (0.75, 1.0)}
LOW_MONEY = 2_000_000
MONEY_BANDS = [-1, 500_000, 2_000_000, 5_000_000, 10_000_000, 1e12]
MONEY_LABELS = ["Under $0.5M", "$0.5-2M", "$2-5M", "$5-10M", "Over $10M"]
NEAR_LEAN = 15


def is_presidential(cycle):
    return cycle % 4 == 0


def scaled(races, midterm, presidential):
    """`races` with each cycle's generic ballot multiplied by its pass-through."""
    return {c: x.assign(gb=x.gb * (presidential if is_presidential(c) else midterm)) for c, x in races.items()}


def national_passthrough(table, polls=("gb", "gb_sept")):
    """The House vote on the polls, by poll and kind of election: k, se, n. `table`: national_table()."""
    rows = []
    groups = {"All": np.ones(len(table), bool), "Presidential": np.array([is_presidential(c) for c in table.index]),
              "Midterm": np.array([not is_presidential(c) for c in table.index])}
    for poll in polls:
        for name, keep in groups.items():
            x, y = table[poll].to_numpy(float)[keep], table.house_vote.to_numpy(float)[keep]
            X = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            e = y - X @ beta
            se = np.sqrt((e @ e) / (len(y) - 2) * np.linalg.inv(X.T @ X)[1, 1])
            rows.append({"poll": poll, "group": name, "k": float(beta[1]), "se": float(se), "n": len(y)})
    return pd.DataFrame(rows)


def fixed_passthrough(races, grid=GRID, years=BACKTEST, leave_out=None):
    """One row per pass-through and year, forward: log_loss, brier, close (races the 1:1 forecast put within
    CLOSE), rmse, seat_error (expected minus actual Democratic seats), all raw. `leave_out`: a cycle
    neither trained on nor scored."""
    base = {y: forecast_year(races, y, FEATURES) for y in years}
    rows = []
    for name, (midterm, presidential) in grid.items():
        scaled_races = scaled(races, midterm, presidential)
        for year in years:
            if year == leave_out:
                continue
            train = {c: x for c, x in scaled_races.items() if c != leave_out}
            f, near = forecast_year(train, year, FEATURES), np.abs(base[year]["mu"]) < CLOSE
            rows.append({"spec": name, "year": year, "log_loss": log_loss(f["p"], f["won"]).mean(), "brier": ((f["p"] - f["won"]) ** 2).mean(),
                         "close": log_loss(f["p"][near], f["won"][near]).mean(), "rmse": np.sqrt(((f["mu"] - f["margin"])[f["contested"]] ** 2).mean()),
                         "seat_error": f["p"].sum() - f["won"].sum()})
    return pd.DataFrame(rows)


def seat_sensitivity(races, cycle=2026, grid=GRID):
    """Expected Democratic seats under each pass-through: mean, p10, p90, p_majority. Fit on every earlier cycle."""
    fc = races[cycle]
    need, states = seats_needed(cycle, fc.district_id), fc.district_id.str[:2]
    rows = {}
    for name, (midterm, presidential) in grid.items():
        r = scaled(races, midterm, presidential)
        model = fit_baseline(pd.concat([x for c, x in r.items() if c < cycle]))
        seats = seat_distribution(predict_margin(r[cycle], model), model, need, states)
        rows[name] = {k: seats[k] for k in ("mean", "p10", "p90", "p_majority")}
    return pd.DataFrame(rows).T


def add_money_bands(races):
    """`races` with sept_money (June receipts plus August outside money), all_money (everything spent), and the terms below."""
    out = {}
    for cycle, x in races.items():
        sept = x.raised_jun_d + x.raised_jun_r + x.outside_aug_d + x.outside_aug_r
        everything = x.disbursements_d + x.disbursements_r + x.outside_d + x.outside_r
        low, low_all = (sept < LOW_MONEY).astype(float), (everything < LOW_MONEY).astype(float)
        log_money = np.log1p(sept) - np.log1p(LOW_MONEY)
        out[cycle] = x.assign(sept_money=sept, all_money=everything, low_money=low, low_all=low_all, log_money=log_money, pvi_low=x.pvi * low,
                              pvi_low_all=x.pvi * low_all, pvi_log=x.pvi * log_money, pvi_curve=x.pvi * x.pvi.abs() / 10)
    return out


def money_slopes(done):
    """How much more a point of PVI is worth under LOW_MONEY: one row per version, with the slope above LOW_MONEY, the
    difference, its standard error and n. `done`: finished cycles' races from add_money_bands."""
    near = done[done.pvi.abs() <= NEAR_LEAN]
    open_terms = [f for f in FEATURES if f not in ("open_seat", "d_incumbent", "fresh_d", "fresh_r", "over_inc")]
    versions = [("All money", done, FEATURES + ["low_all", "pvi_low_all"], "pvi_low_all"), ("Money seen by Sept 1", done, FEATURES + ["low_money", "pvi_low"], "pvi_low"),
                ("Money seen by Sept 1, with a curve in PVI", done, FEATURES + ["low_money", "pvi_low", "pvi_curve"], "pvi_low"),
                (f"All money, lean within {NEAR_LEAN}", near, FEATURES + ["low_all", "pvi_low_all"], "pvi_low_all"),
                (f"Money seen by Sept 1, lean within {NEAR_LEAN}", near, FEATURES + ["low_money", "pvi_low"], "pvi_low"),
                ("Money seen by Sept 1, open seats", done[done.open_seat == 1], open_terms + ["low_money", "pvi_low"], "pvi_low")]
    rows = []
    for name, data, features, term in versions:
        fit = fit_baseline(data, features)
        rows.append({"version": name, "above": fit["coef"]["pvi"], "difference": fit["coef"][term], "se": fit["se"][term], "n": fit["n"]})
    return pd.DataFrame(rows)


def error_by_money(races, money="all_money", years=BACKTEST):
    """Forward errors with the year's miss removed, in contested races, by money band: n, sd."""
    parts = []
    for year in years:
        f = forecast_year(races, year, FEATURES)
        known = races[year][races[year].margin.notna()]
        parts.append(pd.DataFrame({"error": f["margin"] - f["mu"] - f["miss"], "money": known[money].to_numpy(float)})[f["contested"]])
    errors = pd.concat(parts)
    bands = pd.cut(errors.money, MONEY_BANDS, labels=MONEY_LABELS)
    return errors.groupby(bands, observed=True).error.agg(n="size", sd="std")


def error_size_test(races, years=BACKTEST):
    """By year: raw log loss with one sigma, and with the part of the error that isn't national set by whether the race is
    under LOW_MONEY by Sept 1, keeping the training average. Columns one, by_money, change."""
    rows = {}
    for year in years:
        train = pd.concat([x for c, x in races.items() if c < year])
        model = fit_baseline(train)
        fit_on = train[train.margin.abs() < UNCONTESTED]
        resid = pd.Series(fit_on.margin.to_numpy() - predict_margin(fit_on, model))
        resid = resid - resid.groupby(fit_on.cycle.to_numpy()).transform("mean")
        low = fit_on.low_money.to_numpy() == 1
        var_low, var_high = resid[low].var(), resid[~low].var()
        rest = model["sigma"] ** 2 - model["sigma_national"] ** 2
        scale = rest / ((var_low * low.sum() + var_high * (~low).sum()) / len(resid))
        test = races[year][races[year].margin.notna()]
        mu, won = predict_margin(test, model), (test.margin > 0).to_numpy(float)
        sigma = np.sqrt(model["sigma_national"] ** 2 + scale * np.where(test.low_money == 1, var_low, var_high))
        one, by_money = log_loss(norm.cdf(mu / model["sigma"]), won).mean(), log_loss(norm.cdf(mu / sigma), won).mean()
        rows[year] = {"one": one, "by_money": by_money, "change": by_money - one}
    return pd.DataFrame(rows).T
