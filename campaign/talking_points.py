"""Talking-points pilot: are right-leaning outlets' topics more concentrated and slower to
change than left-leaning outlets', 2016-2026, and does the gap come before generic-ballot moves?

Design (fixed before looking at results):
- Articles: data/talking_points_articles.csv.gz, one full GDELT day per sampled Tuesday, outlets
  in data/partisan_outlets.csv (AllSides-rated, partisan audience score |s| >= 0.5, news and
  opinion sites only). Duplicate URLs dropped.
- Two vocabularies. (a) GDELT themes, minus outside taxonomies (TAX_, WB_, CRISISLEX_, UNGP_)
  and minus themes on more than 20% of all articles. (b) Headline words: adjacent-word pairs
  from the URL slug after dropping stop words, numbers and short tokens (closer to a talking
  point than a theme tag; only for URLs whose slug has 3+ words).
- Each article weighs 1, split evenly over its terms. Every side-month is a random draw of
  N_DRAW articles (averaged over REPS draws), so neither side, and no month, looks more
  scattered just from having fewer articles. Months with fewer than N_DRAW on either side
  are dropped.
- Concentration: share of a side's month on its own 10 most common terms.
- Turnover: share of a side's month on terms outside its top K over months m-5..m-3
  (K = 25 themes, 100 headline pairs). Higher = more new material.
- Gap = left minus right. Summary by period, with Newey-West standard errors (6 lags).
- Checks: sides shuffled within month (gap should be ~0); biggest outlet per side dropped.
- Added after the first run showed Daily Kos supplying 47% of left articles in 2024-2026 and
  headline pairs full of column titles ("cheers jeers", "kagro morning", "daniel greenfield"):
  (1) outlet-balanced draws, each outlet in a side-month equally likely to supply an article;
  (2) series titles dropped: a headline pair is a series title if one outlet makes 90%+ of its
  side's uses and it shows up on 8+ sampled days for that outlet; (3) political articles only
  (a GDELT theme in POLITICAL), in case general-news outlets look broad just from covering
  crime, weather and business; (4) each headline slug kept once per side (the first time it
  appears), since 5-11% of articles repeat an earlier headline on their side: the same story
  at several URLs, AlterNet and Raw Story sharing stories, and recurring posts. All four are
  post hoc, and the corrected runs apply (1), (2) and (4) together, with (3) on top.
- Lead test (monthly): change in generic-ballot margin over the next 3 months on the turnover
  gap, controlling for the current margin and the president's party; and the reverse
  (gap on the past 3 months' change). Newey-West, 6 lags.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm

from .plots.plot_talking_points import plot_talking_points

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = DATA / "talking_points_results.json"
FIGURE = ROOT / "figures" / "17_talking_points.png"
N_DRAW = 80
REPS = 20
TOP_CONC = 10
K = {"themes": 25, "headline": 100}
DROP_PREFIX = ("TAX_", "WB_", "CRISISLEX_", "UNGP_")
STOP = set("""a an the and or but of to in on at for from by with without about as into over after before
under between through during against among is are was were be been being has have had do does did
not no nor so than too very can will just would could should may might must shall this that these those
it its it's he she they them his her their our we you your i me my mine us who whom whose which what when
where why how all any both each few more most other some such only own same here there then once
up down out off again further also new says said say video watch read report update live photos photo
index html htm php aspx story article articles news blog blogs post amp id via vs get gets s t
""".split())
POLITICAL = {"ELECTION", "LEGISLATION", "USPEC_POLITICS_GENERAL1", "EPU_POLICY_POLITICAL"}
PRESIDENT = [("2017-01-20", "R"), ("2021-01-20", "D"), ("2025-01-20", "R")]


def load_articles():
    d = pd.read_csv(DATA / "talking_points_articles.csv.gz", dtype={"date": str})
    d = d.drop_duplicates("url")
    d["month"] = pd.to_datetime(d.date, format="%Y%m%d").dt.to_period("M")
    return d.reset_index(drop=True)


def slug_of(url):
    path = re.sub(r"^https?://[^/]+", "", str(url)).split("?")[0]
    segs = [s for s in path.split("/") if s]
    if not segs:
        return ""
    slug = max(segs, key=lambda s: len(re.findall(r"[a-z]+", s.lower())))
    slug = re.sub(r"\.(s?html?|php|aspx?)$", "", slug.lower())
    words = [w for w in re.split(r"[-_+.]+", slug) if w.isalpha()]
    return "-".join(words) if len(words) >= 3 else ""


def dedupe_slugs(d):
    """Keep the first article for each headline slug on each side; keep slugless articles."""
    d = d.sort_values(["date", "url"]).copy()
    d["slug"] = d.url.map(slug_of)
    dup = (d.slug != "") & d.duplicated(["side", "slug"])
    return d[~dup].drop(columns="slug").reset_index(drop=True)


def slug_pairs(url):
    path = re.sub(r"^https?://[^/]+", "", str(url)).split("?")[0]
    segs = [s for s in path.split("/") if s]
    if not segs:
        return []
    slug = max(segs, key=lambda s: len(re.findall(r"[a-z]+", s.lower())))
    slug = re.sub(r"\.(s?html?|php|aspx?)$", "", slug.lower())
    words = [w for w in re.split(r"[-_+.]+", slug) if w]
    if len(words) < 3:
        return []
    keep = [w for w in words if len(w) > 2 and w.isalpha() and w not in STOP]
    return [f"{a} {b}" for a, b in zip(keep, keep[1:])]


def term_lists(d, vocab):
    if vocab == "themes":
        lists = d.themes.fillna("").str.split(";").map(
            lambda ts: [t for t in ts if t and not t.startswith(DROP_PREFIX)])
        df = pd.Series([t for ts in lists for t in set(ts)]).value_counts() / len(d)
        common = set(df[df > 0.20].index)
        return lists.map(lambda ts: [t for t in ts if t not in common])
    return d.url.map(slug_pairs)


def drop_series_titles(d, lists, share=0.9, min_days=8):
    """Remove headline pairs that are one outlet's recurring column or show title."""
    e = pd.DataFrame({"domain": d.domain.values, "side": d.side.values, "date": d.date.values,
                      "t": lists.values}).explode("t").dropna(subset=["t"]).drop_duplicates()
    per = e.groupby(["side", "t", "domain"]).agg(k=("date", "size"), days=("date", "nunique")).reset_index()
    tot = per.groupby(["side", "t"]).k.transform("sum")
    flag = per[(per.k / tot >= share) & (per.days >= min_days)]
    bad = {s: set(g.t) for s, g in flag.groupby("side")}
    out = pd.Series([[t for t in ts if t not in bad.get(side, ())] for ts, side in zip(lists, d.side)], index=lists.index)
    return out, flag


def weight_matrix(lists):
    vocab = {}
    rows, cols, vals = [], [], []
    for i, ts in enumerate(lists):
        ts = list(dict.fromkeys(ts))
        for t in ts:
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
            vals.append(1.0 / len(ts))
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(lists), len(vocab)))
    return m, np.array(sorted(vocab, key=vocab.get))


def measures(d, lists, vocab, rng, reps=REPS, shuffle=False, balance=False):
    """Monthly concentration and turnover per side, equal sample sizes, averaged over draws."""
    has = lists.map(len) > 0
    d = d[has].reset_index(drop=True)
    m, _ = weight_matrix(lists[has].reset_index(drop=True))
    months = sorted(d.month.unique())
    idx = {(mo, s): np.flatnonzero((d.month == mo).values & (d.side == s).values)
           for mo in months for s in "LR"}
    dom = d.domain.values
    out = []
    for r in range(reps):
        sums = {}
        for mo in months:
            a, b = idx[(mo, "L")], idx[(mo, "R")]
            n = N_DRAW
            if min(len(a), len(b)) < n:
                continue
            if shuffle:
                pool = np.concatenate([a, b])
                rng.shuffle(pool)
                a, b = pool[: len(a)], pool[len(a):]
            for s, ix in (("L", a), ("R", b)):
                p = None
                if balance:  # each outlet equally likely to supply an article
                    _, inv, cnt = np.unique(dom[ix], return_inverse=True, return_counts=True)
                    p = 1.0 / cnt[inv]
                    p /= p.sum()
                pick = rng.choice(ix, n, replace=False, p=p)
                sums[(mo, s)] = np.asarray(m[pick].sum(axis=0)).ravel()
        for (mo, s), v in sums.items():
            tot = v.sum()
            conc = np.sort(v)[::-1][:TOP_CONC].sum() / tot
            prior = [sums.get((mo - k, s)) for k in (3, 4, 5)]
            turn = np.nan
            if all(p is not None for p in prior):
                ref = np.argsort(sum(prior))[::-1][: K[vocab]]
                turn = 1 - v[ref].sum() / tot
            out.append((r, mo, s, conc, turn))
    o = pd.DataFrame(out, columns=["rep", "month", "side", "concentration", "turnover"])
    return o.groupby(["month", "side"])[["concentration", "turnover"]].mean().unstack("side")


def nw_mean(x, lags=6):
    x = pd.Series(x).dropna()
    if len(x) < 8:
        return np.nan, np.nan, len(x)
    f = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return f.params[0], f.bse[0], len(x)


def period(mo):
    y = mo.year
    return "2016-2019" if y < 2020 else ("2020-2023" if y < 2024 else "2024-2026")


def president(mo):
    t = mo.to_timestamp()
    party = "D"  # Obama before 2017-01-20
    for start, p in PRESIDENT:
        if t >= pd.Timestamp(start):
            party = p
    return party


def summarize(res, label):
    rows = []
    for meas in ("concentration", "turnover"):
        gap = res[(meas, "L")] - res[(meas, "R")]
        for per, g in gap.groupby(gap.index.map(period)):
            est, se, n = nw_mean(g)
            rows.append((label, meas, per, res[(meas, "L")][g.index].mean(), res[(meas, "R")][g.index].mean(), est, se, n))
        est, se, n = nw_mean(gap)
        rows.append((label, meas, "all", res[(meas, "L")].mean(), res[(meas, "R")].mean(), est, se, n))
    return pd.DataFrame(rows, columns=["run", "measure", "period", "left", "right", "gap_L_minus_R", "se", "months"])


def ballot_monthly():
    p = pd.read_csv(DATA / "generic_ballot_polls_2017_2026.csv", parse_dates=["end_date"])
    return p.groupby(p.end_date.dt.to_period("M")).margin.mean()


def lead_test(res, gb):
    gap = (res[("turnover", "L")] - res[("turnover", "R")]).rename("gap")
    df = pd.concat([gap, gb.rename("gb")], axis=1).sort_index()
    df.index = pd.PeriodIndex(df.index, freq="M")
    full = pd.period_range(df.index.min(), df.index.max(), freq="M")
    df = df.reindex(full)
    df["gb_fwd"] = df.gb.shift(-3) - df.gb
    df["gb_back"] = df.gb - df.gb.shift(3)
    df["dem_pres"] = [1.0 if president(m) == "D" else 0.0 for m in df.index]
    out = {}
    a = df.dropna(subset=["gap", "gb", "gb_fwd"])
    X = sm.add_constant(a[["gap", "gb", "dem_pres"]])
    f = sm.OLS(a.gb_fwd, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    out["forward"] = (f.params["gap"], f.bse["gap"], int(f.nobs))
    b = df.dropna(subset=["gap", "gb_back"])
    X = sm.add_constant(b[["gb_back", "dem_pres"]])
    f = sm.OLS(b.gap, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    out["reverse"] = (f.params["gb_back"], f.bse["gb_back"], int(f.nobs))
    c = df.dropna(subset=["gap"])
    X = sm.add_constant(c[["dem_pres"]])
    f = sm.OLS(c.gap, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    out["gap_when_dem_president"] = (f.params["dem_pres"], f.bse["dem_pres"], int(f.nobs))
    return out, df


def flat(res):
    f = res.copy()
    f.columns = [f"{m}_{s}" for m, s in f.columns]
    f.index = f.index.astype(str)
    return f.reset_index().rename(columns={"index": "month"}).round(5).to_dict(orient="list")


def main():
    rng = np.random.default_rng(20261109)
    d = load_articles()
    counts = d.groupby([d.month.map(lambda m: m.year), "side"]).size().unstack()
    print("articles", len(d), "days", d.date.nunique())
    print(counts)
    gb = ballot_monthly()
    tables, monthly, leads = [], {}, {}
    for vocab in ("themes", "headline"):
        lists = term_lists(d, vocab)
        print(vocab, "share with terms", (lists.map(len) > 0).groupby(d.side).mean().round(3).to_dict())
        res = measures(d, lists, vocab, rng)
        monthly[vocab] = flat(res)
        tables.append(summarize(res, vocab))
        tables.append(summarize(measures(d, lists, vocab, rng, reps=5, shuffle=True), vocab + " shuffled"))
        top = d.groupby("side").domain.agg(lambda s: s.value_counts().index[0])
        keep = ~d.domain.isin(top.values)
        sub = d[keep].reset_index(drop=True)
        tables.append(summarize(measures(sub, lists[keep].reset_index(drop=True), vocab, rng, reps=5),
                                vocab + f" without {'/'.join(top.values)}"))
        lt, _ = lead_test(res, gb)
        leads[vocab] = {k: dict(zip(("coef", "se", "n"), v)) for k, v in lt.items()}
        # post hoc corrections, on slug-deduplicated articles
        dc = dedupe_slugs(d)
        lc = term_lists(dc, vocab)
        name = vocab + " corrected"
        if vocab == "headline":
            lc, flag = drop_series_titles(dc, lc)
            print("articles after dedupe", len(dc), "series titles dropped:", flag.groupby("side").size().to_dict())
        res = measures(dc, lc, vocab, rng, balance=True)
        monthly[name] = flat(res)
        tables.append(summarize(res, name))
        tables.append(summarize(measures(dc, lc, vocab, rng, reps=5, shuffle=True, balance=True), name + " shuffled"))
        lt, _ = lead_test(res, gb)
        leads[name] = {k: dict(zip(("coef", "se", "n"), v)) for k, v in lt.items()}
        pol = dc.themes.fillna("").str.split(";").map(lambda ts: bool(POLITICAL & set(ts))).values
        res = measures(dc[pol].reset_index(drop=True), lc[pol].reset_index(drop=True), vocab, rng, reps=10, balance=True)
        monthly[name + ", political only"] = flat(res)
        tables.append(summarize(res, name + ", political only"))
    for k, v in leads.items():
        for kk, r in v.items():
            print(f"{k} {kk}: {r['coef']:+.3f} (se {r['se']:.3f}, n {r['n']})")
    t = pd.concat(tables)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    out = {
        "days": int(d.date.nunique()), "articles": int(len(d)),
        "articles_by_year": {str(k): {s: int(v) for s, v in r.items()} for k, r in counts.iterrows()},
        "n_draw": N_DRAW, "reps": REPS,
        "summary": t.round(5).to_dict(orient="records"),
        "lead": leads,
        "monthly": monthly,
        "ballot_monthly": {str(k): round(float(v), 3) for k, v in gb.items()},
    }
    RESULTS.write_text(json.dumps(out, indent=1))
    plot_talking_points(out, FIGURE)
    return t


if __name__ == "__main__":
    main()
