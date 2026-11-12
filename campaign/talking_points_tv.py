"""Talking points on cable news: does Fox News repeat its usual phrases more than MSNBC, did it
get more repetitive from 2024 to the 2026 midterm, and does the gap come before generic-ballot
moves? The TV follow-up to campaign/talking_points.py, which measured news websites.

Design (fixed before any results; written while the caption fetch was still running):
- Captions: Internet Archive TV News Archive, west-coast feeds FOXNEWSW (right), MSNBCW, which
  became MSNOW in November 2025 (left), and CNNW (a reference, not a side). Fetched by
  talking_points_tv_fetch.py into data/raw/tv_captions/: one caption snippet per program minute.
- Days: the website pilot's sampled Tuesdays, every 4th Tuesday from 2016-01-05 to 2026-11-03,
  programs whose recording starts that UTC day. Paid programming dropped at fetch time. Reruns
  found by content: a program is dropped if half or more of its distinct 8-word runs already
  aired in an earlier program that day (a title rule would drop live hours that share a generic
  title such as "MSNBC Live").
- Cleaning: lowercase; bracketed sound notes, music marks and speaker tags (">> name:") removed;
  "'s" stripped. Ads, promos and replayed clips: any 8-word run that appears in two or more
  different programs on the same channel-day is masked in all of them, and masked spans break
  word pairs.
- Words: stop words (the website list, a spoken-filler list and network names) and words of
  two letters or fewer are skipped; pairs bridge over them. Words seen on fewer than 3
  channel-days in the whole sample are treated as caption garble and skipped too.
- Terms: (a) word pairs, adjacent kept words within a minute; (b) word triples, as a check.
- Program catchphrases: a term is dropped from a channel if one program title makes 90%+ of
  that channel's uses and uses it on 8+ sampled days (the website pilot's column-title rule).
- Minutes with fewer than 10 pair uses (mostly ads, music and masked clips) are dropped.
- Weighting: each use of a term counts once. Every channel-month is a draw of 400 minutes, each
  program that day equally likely to supply a minute, averaged over 20 draws. Months where any
  channel has fewer than 400 minutes are dropped.
- Concentration: share of a channel-month's term uses on its own 10 most used terms.
- Repeat share (1 - turnover): share of a channel-month's term uses on terms among its own 100
  most used over months m-5..m-3. K = 500 as a check.
- Hypotheses: H1, Fox's repeat share is higher than MSNBC's in 2024-2026. H2, Fox's repeat share
  rose from 2016-2023 to 2024-2026 more than MSNBC's (gap regressed on a 2024+ indicator).
  H3, the gap comes before generic-ballot moves (the website pilot's lead test, unchanged).
  Newey-West standard errors, 6 lags. CNN moving with both channels would point to caption
  format or the news cycle rather than partisan style.
- Checks: minutes shuffled between Fox and MSNBC within month (gap should be ~0); each
  channel's biggest program (most minutes) dropped; triples; K = 500.
- Settled before results but after first looking at the captions: the rerun rule went from
  title-based to content-based, and "him having whatever stuff sort" joined the filler list.
- Added after the first run (post hoc): pairs naming Trump or Biden dropped, since MSNBC says
  "donald trump" more than Fox does; and each channel's share of pair uses naming Biden, by year.
"""
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm

from .plots.plot_talking_points_tv import plot_talking_points_tv
from .talking_points import STOP, ballot_monthly, flat, lead_test, nw_mean, period, summarize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CAPS = DATA / "raw" / "tv_captions"
CACHE = DATA / "raw" / "tv_content.parquet"
RESULTS = DATA / "talking_points_tv_results.json"
FIGURE = ROOT / "figures" / "18_talking_points_tv.png"
SIDE = {"MSNBC": "L", "FOX": "R", "CNN": "C"}
N_DRAW = 400
REPS = 20
TOP_CONC = 10
K_MAIN = 100
MIN_TERMS = 10
MASK_N = 8
RERUN_SHARE = 0.5
MIN_DAYS_WORD = 3
SHOW_SHARE, SHOW_DAYS = 0.9, 8
FILLER = set("""
going gonna wanna gotta know knew knows think thought thinking people like right well yeah yes okay ok
really got get getting gets lot lots one two three want wanted wants look looking looks thing things
let see seen seeing come comes coming came make makes making made way back time times today tonight
now actually little say saying tell telling told talk talked talking take taking took put sure kind
mean means good great big many much even still every first last next year years day days week weeks
morning night evening afternoon thank thanks joining appreciate break stay tuned ahead minute minutes
hour hours live breaking show shows guys sir maybe something anything everything nothing someone
anybody everybody somebody anyone everyone also already always never ever around else whether another
able used use need needs needed went goes doing done happen happened happens happening point question
absolutely exactly certainly obviously course fact part whole keep keeps long old uh um oh hey wow
gonna lot probably pretty quite though although because since until while yet enough ago away
together different important best better bad really talk give gave given find found try trying tried
start started end ended called call calling feel felt believe believed understand understood
commercial sponsor brought welcome hello hi bye folks everybody ladies gentlemen joins joined
him having whatever stuff sort
""".split())
NETWORK = {"fox", "foxnews", "msnbc", "msnow", "cnn"}
SKIP = STOP | FILLER | NETWORK
SOUND = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪")
SPEAKER = re.compile(r">>\s*[a-z][a-z .'\-]{0,40}?:")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*")
BREAK = "|"


# ---------- captions to content words ----------

def tokens(text):
    t = SOUND.sub(" ", str(text).lower().replace("’", "'"))
    t = SPEAKER.sub(" ", t).replace(">>", " ")
    return [re.sub(r"'s$", "", w) for w in WORD.findall(t)]


def clean_day(path):
    """One channel-day: drop reruns, mask 8-word runs shared by 2+ programs, keep content words.
    A program is a rerun if RERUN_SHARE or more of its distinct 8-word runs already appeared in
    an earlier program that day (programs taken in start-time order). Returns rows (identifier,
    show, minute, content string with BREAK at masked spans), the masked share of words in kept
    programs, the number of reruns dropped, and the day's set of content words."""
    d = pd.read_csv(path, sep="\t", dtype={"text": str}, keep_default_na=False)
    toks = [tokens(t) for t in d.text]
    progs = d.identifier.values
    streams = defaultdict(list)  # program -> list of (row, position)
    for r, (p, ts) in enumerate(zip(progs, toks)):
        streams[p].extend((r, i) for i in range(len(ts)))
    words = {p: [toks[r][i] for r, i in s] for p, s in streams.items()}
    grams = {p: [hash(tuple(w[i:i + MASK_N])) for i in range(len(w) - MASK_N + 1)] for p, w in words.items()}
    earlier, reruns = set(), set()
    for p in sorted(words, key=lambda i: i.split("_")[2]):
        g = set(grams[p])
        if g and len(g & earlier) / len(g) >= RERUN_SHARE:
            reruns.add(p)
            continue
        earlier |= g
    owners = defaultdict(set)
    for p, g in grams.items():
        if p not in reruns:
            for h in g:
                owners[h].add(p)
    masked = [np.zeros(len(ts), bool) for ts in toks]
    for p, g in grams.items():
        if p in reruns:
            continue
        pos = streams[p]
        for i, h in enumerate(g):
            if len(owners[h]) > 1:
                for r, j in pos[i:i + MASK_N]:
                    masked[r][j] = True
    rows, vocab, n_all, n_mask = [], set(), 0, 0
    for r, ts in enumerate(toks):
        if progs[r] in reruns:
            continue
        out, prev_break = [], False
        for w, m in zip(ts, masked[r]):
            if m:
                if not prev_break:
                    out.append(BREAK)
                prev_break = True
                continue
            if len(w) <= 2 or "'" in w or w in SKIP:
                continue
            out.append(w)
            vocab.add(w)
            prev_break = False
        n_all += len(ts)
        n_mask += int(masked[r].sum())
        rows.append((d.identifier.iat[r], d["show"].iat[r], int(d.minute.iat[r]), " ".join(out)))
    return rows, (n_mask / n_all if n_all else np.nan), len(reruns), vocab


def build_content():
    files = sorted(CAPS.glob("*.tsv.gz"))
    frames, masked, day_vocab = [], {}, Counter()
    with ProcessPoolExecutor(12) as pool:
        for f, (rows, share, n_rerun, vocab) in zip(files, pool.map(clean_day, files, chunksize=2)):
            day, chan = f.name.split(".")[0].split("_")
            g = pd.DataFrame(rows, columns=["identifier", "show", "minute", "content"])
            g["date"], g["channel"] = day, chan
            frames.append(g)
            masked[(day, chan)] = (share, n_rerun)
            day_vocab.update(vocab)
    d = pd.concat(frames, ignore_index=True)
    rare = {w for w, n in day_vocab.items() if n < MIN_DAYS_WORD}
    d["content"] = [" ".join(w for w in c.split() if w not in rare) for c in d.content]
    d.to_parquet(CACHE, index=False)
    m = pd.DataFrame([(k[0], k[1], v[0], v[1]) for k, v in masked.items()],
                     columns=["date", "channel", "masked_share", "reruns"])
    m.to_csv(CACHE.with_suffix(".masked.csv"), index=False)
    return d


def load_content(rebuild=False):
    d = build_content() if rebuild or not CACHE.exists() else pd.read_parquet(CACHE)
    d["side"] = d.channel.map(SIDE)
    d["month"] = pd.to_datetime(d.date, format="%Y%m%d").dt.to_period("M")
    return d


def ngrams(content, n):
    out = []
    for seg in content.split(BREAK):
        w = seg.split()
        out += [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return out


def count_matrix(d, n):
    vocab, rows, cols = {}, [], []
    for i, c in enumerate(d.content):
        for t in ngrams(c, n):
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
    m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(d), len(vocab)))
    m.sum_duplicates()
    return m, np.array(sorted(vocab, key=vocab.get))


def drop_catchphrases(d, m):
    """Zero terms one program title owns on its channel (90%+ of uses, 8+ sampled days)."""
    flagged = {}
    for chan, ix in d.groupby("channel").indices.items():
        sub = m[ix]
        shows = d["show"].values[ix]
        codes, show_idx = np.unique(shows, return_inverse=True)
        g = sp.csr_matrix((np.ones(len(ix)), (show_idx, np.arange(len(ix)))), shape=(len(codes), len(ix)))
        uses = (g @ sub).tocsc()
        days_key = pd.factorize(pd.Series(shows) + "|" + d.date.values[ix])[0]
        show_of_day = pd.Series(show_idx).groupby(days_key).first().values
        gd = sp.csr_matrix((np.ones(len(ix)), (days_key, np.arange(len(ix)))), shape=(days_key.max() + 1, len(ix)))
        present = ((gd @ sub) > 0).astype(float)
        gs = sp.csr_matrix((np.ones(len(show_of_day)), (show_of_day, np.arange(len(show_of_day)))),
                           shape=(len(codes), len(show_of_day)))
        days = (gs @ present).tocsc()
        total = np.asarray(uses.sum(axis=0)).ravel()
        top = np.asarray(uses.argmax(axis=0)).ravel()
        top_uses = np.asarray(uses.max(axis=0).todense()).ravel()
        top_days = np.asarray(days[top, np.arange(m.shape[1])]).ravel()
        bad = (total > 0) & (top_uses / np.maximum(total, 1) >= SHOW_SHARE) & (top_days >= SHOW_DAYS)
        flagged[chan] = np.flatnonzero(bad)
    m = m.tocoo()
    chan_of_row = d.channel.values[m.row]
    keep = np.ones(m.nnz, bool)
    for chan, cols in flagged.items():
        keep &= ~((chan_of_row == chan) & np.isin(m.col, cols))
    m = sp.csr_matrix((m.data[keep], (m.row[keep], m.col[keep])), shape=m.shape)
    return m, flagged


# ---------- measures ----------

def measures(meta, m, rng, k=K_MAIN, reps=REPS, shuffle=False, sides=("L", "R", "C")):
    """Monthly concentration and turnover per channel, count-weighted, equal minute draws."""
    tot = np.asarray(m.sum(axis=1)).ravel()
    keepcols = np.flatnonzero(np.asarray(m.sum(axis=0)).ravel() >= 5)  # rarer terms can't make a top list
    m = m[:, keepcols].tocsr()
    months = sorted(meta.month.unique())
    prog = meta.identifier.values
    groups = meta.groupby(["month", "side"]).indices
    empty = np.array([], int)
    out = []
    for r in range(reps):
        sums, tots = {}, {}
        for mo in months:
            ixs = {s: groups.get((mo, s), empty) for s in sides}
            if min(len(v) for v in ixs.values()) < N_DRAW:
                continue
            if shuffle:
                a, b = ixs["L"], ixs["R"]
                pool = np.concatenate([a, b])
                rng.shuffle(pool)
                ixs["L"], ixs["R"] = pool[: len(a)], pool[len(a):]
            for s, ix in ixs.items():
                _, inv, cnt = np.unique(prog[ix], return_inverse=True, return_counts=True)
                p = 1.0 / cnt[inv]
                pick = rng.choice(ix, N_DRAW, replace=False, p=p / p.sum())
                sums[(mo, s)] = np.asarray(m[pick].sum(axis=0)).ravel()
                tots[(mo, s)] = tot[pick].sum()
        for (mo, s), v in sums.items():
            t = tots[(mo, s)]
            conc = np.partition(v, -TOP_CONC)[-TOP_CONC:].sum() / t
            prior = [sums.get((mo - j, s)) for j in (3, 4, 5)]
            turn = np.nan
            if all(p is not None for p in prior):
                ref = np.argpartition(sum(prior), -k)[-k:]
                turn = 1 - v[ref].sum() / t
            out.append((r, mo, s, conc, turn))
    o = pd.DataFrame(out, columns=["rep", "month", "side", "concentration", "turnover"])
    return o.groupby(["month", "side"])[["concentration", "turnover"]].mean().unstack("side")


def shift_test(res):
    """H2: repeat-share gap (Fox - MSNBC) and each channel, 2024+ against 2016-2023."""
    out = {}
    rep = 1 - res["turnover"]
    series = {"gap_fox_minus_msnbc": rep["R"] - rep["L"], "fox": rep["R"], "msnbc": rep["L"]}
    if "C" in rep:
        series["cnn"] = rep["C"]
    for name, s in series.items():
        s = s.dropna()
        post = np.array([mo.year >= 2024 for mo in s.index], float)
        f = sm.OLS(s.values, sm.add_constant(post)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        out[name] = {"before": float(f.params[0]), "change_2024_on": float(f.params[1]), "se": float(f.bse[1]), "n": int(f.nobs)}
    return out


def by_year(res):
    rep = (1 - res["turnover"]) * 100
    conc = res["concentration"] * 100
    g = lambda x: x.groupby(x.index.map(lambda m: m.year)).mean().round(3)
    return {"repeat": g(rep).to_dict(orient="index"), "concentration": g(conc).to_dict(orient="index")}


def top_terms(d, m, terms, start="2025-01", n=15):
    sel = (d.month >= pd.Period(start, "M")).values
    out = {}
    for chan in SIDE:
        rows = np.flatnonzero(sel & (d.channel == chan).values)
        v = np.asarray(m[rows].sum(axis=0)).ravel()
        top = np.argsort(v)[::-1][:n]
        out[chan] = [(terms[i], int(v[i])) for i in top]
    return out


def main(rebuild=False):
    rng = np.random.default_rng(20261110)
    d = load_content(rebuild)
    print("minutes", len(d), "channel-days", d.groupby(["date", "channel"]).ngroups)
    m2, t2 = count_matrix(d, 2)
    m2, flag2 = drop_catchphrases(d, m2)
    print("catchphrase pairs dropped", {c: len(v) for c, v in flag2.items()})
    ok = np.asarray(m2.sum(axis=1)).ravel() >= MIN_TERMS
    d, m2 = d[ok].reset_index(drop=True), m2[ok]
    minutes = d.groupby([d.month.map(lambda x: x.year), "channel"]).size().unstack()
    print(minutes)
    gb = ballot_monthly()
    tables, monthly, leads, shifts, years = [], {}, {}, {}, {}

    def run(name, meta, m, reps=REPS, **kw):
        res = measures(meta, m, rng, reps=reps, **kw)
        tables.append(summarize(res, name))
        if not kw.get("shuffle"):
            monthly[name] = flat(res)
            shifts[name] = shift_test(res)
            years[name] = by_year(res)
            lt, _ = lead_test(res, gb)
            leads[name] = {k: dict(zip(("coef", "se", "n"), v)) for k, v in lt.items()}
        return res

    run("pairs", d, m2)
    run("pairs shuffled", d, m2, reps=5, shuffle=True)
    run("pairs K=500", d, m2, reps=10, k=500)
    big = d.groupby("channel")["show"].agg(lambda s: s.value_counts().index[0])
    keep = ~d["show"].isin(big.values).values
    run("pairs without " + "/".join(big.values), d[keep].reset_index(drop=True), m2[keep], reps=10)
    m3, t3 = count_matrix(d, 3)
    m3, flag3 = drop_catchphrases(d, m3)
    run("triples", d, m3, reps=10)
    # post hoc: without pairs naming the president or his predecessor
    names = np.array(["trump" in x or "biden" in x for x in t2])
    m_nt = (m2 @ sp.diags((~names).astype(float))).tocsr()
    m_nt.eliminate_zeros()
    run("pairs without trump/biden", d, m_nt, reps=10)
    tot = np.asarray(m2.sum(axis=1)).ravel()
    bid = np.asarray(m2[:, np.flatnonzero(np.array(["biden" in x for x in t2]))].sum(axis=1)).ravel()
    yr = d.month.map(lambda x: x.year).values
    bs = pd.DataFrame({"channel": d.channel.values, "year": yr, "b": bid, "t": tot}).groupby(["year", "channel"]).sum()
    biden_share = (bs.b / bs.t * 100).round(3).unstack()
    print("share of pair uses naming Biden (%)\n", biden_share)

    t = pd.concat(tables)
    # turnover gap L - R equals Fox minus MSNBC repeat share; flip concentration to Fox minus MSNBC
    t["fox_minus_msnbc"] = t.gap_L_minus_R.where(t.measure == "turnover", -t.gap_L_minus_R)
    pd.set_option("display.width", 220)
    print(t.round(4).to_string(index=False))
    for k, v in shifts.items():
        print(k, json.dumps({a: {b: round(c, 4) if isinstance(c, float) else c for b, c in r.items()} for a, r in v.items()}))
    for k, v in leads.items():
        for kk, r in v.items():
            print(f"{k} {kk}: {r['coef']:+.4f} (se {r['se']:.4f}, n {r['n']})")
    masked = pd.read_csv(CACHE.with_suffix(".masked.csv"), dtype={"date": str})
    masked_year = masked.groupby([masked.date.str[:4], "channel"]).masked_share.mean().unstack().round(3)
    reruns_year = masked.groupby([masked.date.str[:4], "channel"]).reruns.mean().unstack().round(2)
    print("share of words masked as shared runs\n", masked_year, "\nreruns dropped per day\n", reruns_year)
    tops = {"pairs": top_terms(d, m2, t2), "triples": top_terms(d, m3, t3)}
    for k, v in tops["pairs"].items():
        print(k, v[:15])
    out = {
        "channel_days": int(d.groupby(["date", "channel"]).ngroups),
        "minutes_by_year": {str(k): {c: int(x) for c, x in r.items()} for k, r in minutes.iterrows()},
        "masked_share_by_year": {str(k): r.to_dict() for k, r in masked_year.iterrows()},
        "reruns_per_day_by_year": {str(k): r.to_dict() for k, r in reruns_year.iterrows()},
        "catchphrases_dropped": {"pairs": {c: len(v) for c, v in flag2.items()},
                                 "triples": {c: len(v) for c, v in flag3.items()}},
        "dropped_programs": big.to_dict(),
        "n_draw": N_DRAW, "reps": REPS,
        "summary": t.round(5).to_dict(orient="records"),
        "shift_2024": shifts, "by_year": {k: {a: {str(y): r for y, r in b.items()} for a, b in v.items()} for k, v in years.items()},
        "lead": leads, "top_terms_2025_2026": tops,
        "biden_share_by_year": {str(k): r.to_dict() for k, r in biden_share.iterrows()},
        "monthly": monthly,
    }
    RESULTS.write_text(json.dumps(out, indent=1, default=float))
    plot_talking_points_tv(out, FIGURE)
    return t


if __name__ == "__main__":
    import sys
    main(rebuild="--rebuild" in sys.argv)
