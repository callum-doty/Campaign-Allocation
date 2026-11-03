"""
Does operating-expenditure data improve the forecast beyond the June 30 surplus?
Each candidate feature joins baseline + surplus in the same forward backtest. The
bar, fixed before looking at results: better than baseline + surplus on log loss
with the year's miss taken out in at least 4 of 5 years, better on raw log loss on
average, and not worse in the races the baseline put within 10 points.

placebo() asks how often the bar is cleared by chance, by shuffling a feature among
that year's races. With eight tries, one pass means little on its own.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fit_baseline import FEATURES, fit_baseline
from .fit_close import CLOSE
from .load_donors import DONOR_FEATURES
from .mine_spending import clr, kmeans, shares, MIN_BUDGET
from .predict_margin import predict_margin
from .redrawn_retention import RETENTION_FEATURES
from .spending_features import SECOND, SINGLE, SPLIT

BACKTEST = (2016, 2018, 2020, 2022, 2024)
SURPLUS = list(FEATURES) + ["surplus"]
SPECS = {"baseline": list(FEATURES), "surplus": SURPLUS, **{f: SURPLUS + [f] for f in SINGLE + SECOND + DONOR_FEATURES + RETENTION_FEATURES}, "split": SURPLUS + SPLIT,
         "pvi_low": SURPLUS + ["low_money", "pvi_low"], "pvi_log": SURPLUS + ["log_money", "pvi_log"]}


def log_loss(p, won):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(won * np.log(p) + (1 - won) * np.log(1 - p))


def forecast_year(races, year, features):
    """{'mu','p','p_within','won','margin','state'} for the year's decided races, fit on earlier cycles."""
    model = fit_baseline(pd.concat([r for c, r in races.items() if c < year]), features)
    test = races[year][races[year].margin.notna()]
    mu, margin = predict_margin(test, model), test.margin.to_numpy(float)
    contested = np.abs(margin) < 80
    miss = (margin - mu)[contested].mean()
    return {"mu": mu, "margin": margin, "contested": contested, "miss": miss, "won": (margin > 0).astype(float),
            "p": norm.cdf(mu / model["sigma"]), "state": test.district_id.str[:2].to_numpy(),
            "p_within": norm.cdf((mu + miss) / np.sqrt(model["sigma"] ** 2 - model["sigma_national"] ** 2))}


def score_specs(races, specs=SPECS, years=BACKTEST):
    """One row per spec and year: log_loss, within (year's miss removed), brier, close_within (races the baseline put
    within CLOSE points), margin_within."""
    runs = {(n, y): forecast_year(races, y, f) for n, f in specs.items() for y in years}
    rows = []
    for (name, year), r in runs.items():
        near = np.abs(runs[("baseline", year)]["mu"]) < CLOSE
        rows.append({"spec": name, "year": year, "log_loss": log_loss(r["p"], r["won"]).mean(),
                     "within": log_loss(r["p_within"], r["won"]).mean(), "brier": ((r["p"] - r["won"]) ** 2).mean(),
                     "close_within": log_loss(r["p_within"][near], r["won"][near]).mean(),
                     "margin_within": np.sqrt((((r["mu"] + r["miss"]) - r["margin"])[r["contested"]] ** 2).mean())})
    return pd.DataFrame(rows), runs


def change(scores, column, reference="surplus"):
    """Each spec's change against `reference` by year, its mean, and years better."""
    wide = scores.pivot(index="spec", columns="year", values=column)
    delta = wide.sub(wide.loc[reference])
    years = list(wide.columns)
    return delta.assign(mean=delta[years].mean(axis=1), better=(delta[years] < -1e-9).sum(axis=1))


def bootstrap(runs, name, reference="surplus", draws=4000, seed=3):
    """Mean change in within log loss against `reference`, with a 95% interval resampling state-years."""
    parts = []
    for year in BACKTEST:
        a, b = runs[(name, year)], runs[(reference, year)]
        parts.append(pd.DataFrame({"group": a["state"] + str(year), "d": log_loss(a["p_within"], a["won"]) - log_loss(b["p_within"], b["won"])}))
    both = pd.concat(parts)
    totals = both.groupby("group").d.agg(["sum", "size"])
    picks = np.random.default_rng(seed).integers(0, len(totals), (draws, len(totals)))
    means = totals["sum"].to_numpy()[picks].sum(axis=1) / totals["size"].to_numpy()[picks].sum(axis=1)
    return both.d.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975)


def placebo(races, feature, draws=400, seed=7):
    """{'pass_rate', 'as_good_within', 'as_good_raw'}: `feature` shuffled among each year's races, `draws` times, against its real scores."""
    columns = SURPLUS + ["margin", "gb", "district_id", "cycle", feature]
    slim = {c: r[r.margin.notna() | (c > max(BACKTEST))][columns].copy() for c, r in races.items() if c <= max(BACKTEST)}
    base = {y: forecast_year(slim, y, SURPLUS) for y in BACKTEST}

    def gate():
        raw, within = [], []
        for y in BACKTEST:
            f = forecast_year(slim, y, SURPLUS + [feature])
            raw.append((log_loss(f["p"], f["won"]) - log_loss(base[y]["p"], f["won"])).mean())
            within.append((log_loss(f["p_within"], f["won"]) - log_loss(base[y]["p_within"], f["won"])).mean())
        return np.mean(raw), np.mean(within), sum(w < -1e-9 for w in within)
    real = gate()
    rng = np.random.default_rng(seed)
    original = {c: r[feature].to_numpy().copy() for c, r in slim.items()}
    shuffled = []
    for _ in range(draws):
        for c, r in slim.items():
            r[feature] = rng.permutation(original[c])
        shuffled.append(gate())
    shuffled = np.array(shuffled)
    return {"pass_rate": float(((shuffled[:, 2] >= 4) & (shuffled[:, 0] < 0)).mean()),
            "as_good_within": float((shuffled[:, 1] <= real[1]).mean()), "as_good_raw": float((shuffled[:, 0] <= real[0]).mean())}


def cluster_signal(races, pre, k):
    """The clusters as forecast features, fit only on earlier cycles: change against baseline + surplus by year."""
    table = {c: shares(pre[c]).query("total >= @MIN_BUDGET") for c in races if c <= max(BACKTEST)}
    rows = []
    for year in BACKTEST:
        fitted = kmeans(clr(pd.concat([t for c, t in table.items() if c < year])).to_numpy(), k)
        withdummies = {}
        for c in [c for c in table if c <= year]:
            label = pd.Series(fitted.predict(clr(table[c]).to_numpy()), index=table[c].index)
            withdummies[c] = races[c].assign(**{f"cluster_{j}": (races[c].cm_d.map(label) == j).astype(float) - (races[c].cm_r.map(label) == j).astype(float)
                                                for j in range(k - 1)})
        a = forecast_year(withdummies, year, SURPLUS + [f"cluster_{j}" for j in range(k - 1)])
        b = forecast_year(withdummies, year, SURPLUS)
        rows.append({"year": year, "log_loss": (log_loss(a["p"], a["won"]) - log_loss(b["p"], b["won"])).mean(),
                     "within": (log_loss(a["p_within"], a["won"]) - log_loss(b["p_within"], b["won"])).mean()})
    return pd.DataFrame(rows).set_index("year")
