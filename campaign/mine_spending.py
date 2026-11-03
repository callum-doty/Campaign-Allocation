"""
What campaigns buy, as data mining rather than forecasting, with the fixes each
naive version needs.

Clustering. Spending shares sum to 1, so k-means runs on centered log-ratios of
the shares, which respect that, and leaves total spending out, or it would mostly
find that big campaigns are big. stability() reruns on resamples, since a
silhouette score alone will pick a clean-looking split that doesn't repeat.

Association rules. Nearly every campaign buys a little of everything over a
cycle, so "bought X" gives lift near 1. An item here means a category took at least
ITEM_SHARE of the campaign's budget in that phase, and rules run forward from an
earlier phase to a later one. Well-funded and competitive campaigns buy more of
everything, so conditional lift compares each rule with what's expected among
campaigns of the same budget tercile and race type.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

GROUPS = {"Fundraising": "Fundraising", "Processing fees": "Fundraising", "Broadcast": "Broadcast", "Production": "Production",
          "Digital": "Digital", "Mail/print": "Mail/print", "Field/phones": "Field/phones", "Polling/research": "Polling/research",
          "Payroll/admin": "Payroll/admin", "Events": "Events", "Consulting/unspecified": "Consulting/other", "Pass-through": "Consulting/other",
          "Other": "Consulting/other", "Contributions": "Contributions"}
PSEUDO_SHARE = 0.005
MIN_BUDGET = 100_000
ITEM_SHARE = 0.10
PHASE_BUDGET = 25_000
PHASES = ("Pre", "Q1", "Q2", "Q3", "Q4")


def shares(rows):
    """One row per committee: each group's share of its positive spending, and total."""
    grouped = rows[rows.amount > 0].assign(group=lambda r: r.category.map(GROUPS))
    money = grouped.pivot_table(index="committee_id", columns="group", values="amount", aggfunc="sum", fill_value=0.0)
    return money.div(money.sum(axis=1), axis=0).assign(total=money.sum(axis=1))


def clr(share_table):
    """Centered log-ratios of the share columns (everything but total)."""
    logged = np.log(share_table.drop(columns="total") + PSEUDO_SHARE)
    return logged.sub(logged.mean(axis=1), axis=0)


def kmeans(x, k, seed=0):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)


def stability(x, k, draws=20, seed=0):
    """Mean adjusted Rand index between the full clustering and clusterings on 80% resamples, scored on the full data."""
    rng, full = np.random.default_rng(seed), kmeans(x, k, seed).predict(x)
    scores = []
    for i in range(draws):
        part = x[rng.choice(len(x), int(0.8 * len(x)), replace=False)]
        scores.append(adjusted_rand_score(full, kmeans(part, k, seed + i + 1).predict(x)))
    return float(np.mean(scores))


def choose_k(x, ks=range(3, 9)):
    """k -> silhouette and stability."""
    return pd.DataFrame({k: {"silhouette": silhouette_score(x, kmeans(x, k).labels_, sample_size=min(3000, len(x)), random_state=0),
                             "stability": stability(x, k)} for k in ks}).T


def tv_substitution(fall):
    """Do candidates leave TV where it's expensive? `fall`: one row per candidate, with spending shares after June 30,
    total, reach, party, ici, district_id, cycle. Returns each band's mean shares, and each channel's slope on log reach
    (controlling for budget, party, incumbency and year, standard errors clustered by state and year)."""
    fall = fall.assign(band=pd.cut(fall.reach, [0, 3, 10, np.inf], labels=["Cheap (reach under 3)", "Mid (3-10)", "Expensive (10+)"]))
    by_band = fall.groupby("band", observed=True)
    channels = ["Broadcast", "Digital", "Mail/print", "Consulting/other", "Fundraising"]
    table = by_band[channels].mean().assign(candidates=by_band.size())
    X = np.column_stack([np.ones(len(fall)), np.log(fall.reach), np.log(fall.total), fall.party == "D", fall.ici == "I",
                         fall.cycle == fall.cycle.max()]).astype(float)
    group = pd.factorize(fall.district_id.str[:2] + fall.cycle.astype(str))[0]
    slopes = {}
    for channel in channels[:4]:
        y = fall[channel].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        score = np.zeros((group.max() + 1, X.shape[1]))
        np.add.at(score, group, X * (y - X @ beta)[:, None])
        bread = np.linalg.inv(X.T @ X)
        slopes[channel] = {"tenfold": beta[1] * np.log(10), "se": np.sqrt(np.diag(bread @ score.T @ score @ bread))[1] * np.log(10)}
    return table, pd.DataFrame(slopes).T


def phase_of(dates, cycle):
    """Pre (before the election year), Q1-Q4 of the election year; later dates fall in Q4."""
    return pd.Series(np.where(dates.dt.year < cycle, "Pre", "Q" + dates.dt.quarter.astype(str)), index=dates.index)


def baskets(rows, cycle):
    """One row per committee and item, 'Q2 Polling/research' meaning that category took ITEM_SHARE of a Q2 budget of
    at least PHASE_BUDGET."""
    r = rows[rows.amount > 0].assign(group=lambda d: d.category.map(GROUPS), phase=lambda d: phase_of(d.date, cycle))
    by = r.groupby(["committee_id", "phase", "group"]).amount.sum().rename("amount").reset_index()
    by["budget"] = by.groupby(["committee_id", "phase"]).amount.transform("sum")
    keep = by[(by.budget >= PHASE_BUDGET) & (by.amount >= ITEM_SHARE * by.budget)]
    return keep.assign(item=keep.phase + " " + keep.group)[["committee_id", "item"]]


def spent_at_all(rows, cycle):
    """The naive basket: any money on a category, over the whole cycle."""
    r = rows[rows.amount > 0].assign(item=lambda d: d.category.map(GROUPS))
    return r[["committee_id", "item"]].drop_duplicates()


def rules(table, strata, min_support=0.05, min_confidence=0.5, forward=True, max_antecedent=2):
    """Rules {A} => {B}. `table`: committees by items, True or False; `strata`: each committee's group for conditional
    lift. forward: every item in A comes from an earlier phase than B. Returns support, confidence, lift, conditional_lift."""
    order = {p: i for i, p in enumerate(PHASES)}
    phase = lambda item: order[item.split(" ")[0]]
    common = [c for c in table.columns if table[c].mean() >= min_support]
    out, n = [], len(table)
    antecedents = [(a,) for a in common] + [c for c in combinations(common, 2)] if max_antecedent == 2 else [(a,) for a in common]
    for a in antecedents:
        has = table[list(a)].all(axis=1)
        if has.mean() < min_support:
            continue
        for b in common:
            if b in a or (forward and any(phase(i) >= phase(b) for i in a)):
                continue
            both = has & table[b]
            if both.mean() < min_support or both.sum() / has.sum() < min_confidence:
                continue
            frame = pd.DataFrame({"a": has, "b": table[b], "s": strata.reindex(table.index)})
            sizes = frame.groupby("s").agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"))
            expected = (sizes.a * sizes.b / sizes.n).sum()
            out.append({"if": " + ".join(a), "then": b, "support": both.mean(), "confidence": both.sum() / has.sum(),
                        "lift": both.mean() / (has.mean() * table[b].mean()), "conditional_lift": both.sum() / expected})
    return pd.DataFrame(out)
