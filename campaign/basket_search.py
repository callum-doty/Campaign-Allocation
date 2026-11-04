"""
A combinatoric search over what a Sept 1 forecaster could see, and a test of what
committees' choices of seats say. BASKET_DESIGN.md is the design, committed before
any real combination was scored, and this is that design as code.

    python -m campaign.basket_search --blind    only flipped, shuffled or planted data
    python -m campaign.basket_search            the real run, once

Items belong to a side, and residuals are signed toward Democrats. A combination
counts +1 where only the Democrat has it, -1 where only the Republican does and 0
where both or neither do, so party-blind effects don't cancel and no race counts
twice.

Three tests:

- trees: depth-3 trees on the items, trained on earlier years' misses, added to the
  forecast and held to the bar in spending_signal.py. A depth-3 tree on these items
  is a sum of combinations of up to three, so this asks whether any of them help
  forecasts, pooling weak ones. Fit twice per race, once from each side, and read as
  the difference, so the shift is party-blind too.
- baskets: every pair and triple with enough support, scored on what it adds beyond
  its single items, in units of its own state-clustered standard error, against the
  largest value across all baskets when whole state-years have their signs flipped at
  random. A triple only counts as new if it beats its pairs. Taking pairs out of the
  score itself would double a triple's standard error and leave it no chance.
- committee: a small model of DCCC entering seats the forecast calls safe, fit without
  any result, then one test of whether seats it scores high that no committee entered
  ran behind the forecast. No committee money went into those seats, so a miss there
  can't be committee dollars.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .build_races import build_races
from .dated_forecast import dated_forecast
from .fit_baseline import FEATURES, UNCONTESTED, fit_baseline
from .fit_close import CLOSE
from .freeze_forecast import money_terms
from .load_donors import committee_donors, gifts_through_june, load_donors, race_measures
from .load_oppexp import categorize, load_oppexp
from .mine_spending import MIN_BUDGET, shares
from .predict_margin import predict_margin
from .redrawn_retention import incumbent_retention
from .spending_features import add_spending, committee_measures
from .spending_signal import BACKTEST, SURPLUS, forecast_year, log_loss

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "basket_results.json"
CANDIDATE = ROOT / "data" / "basket_candidate_2026.csv"
ENTRY = ROOT / "data" / "basket_entry_2026.csv"
FROZEN = ROOT / "data" / "forecast_freeze_2026_v2.csv"
CYCLES = (2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026)
LIVE = 2026

SHARES = {"tv": "Broadcast", "digital": "Digital", "mail": "Mail/print", "field": "Field/phones", "polling": "Polling/research",
          "fundraising": "Fundraising", "payroll": "Payroll/admin"}
MEASURES = ["raised", "surplus", "burn", "runway", "debt", "fund_cost", *SHARES, "accel", "vendor_hhi", "vendor_retention",
            "donors", "repeat", "in_state", "top10"]
LOW_RETENTION = 0.6
QUARTILES = (0.25, 0.75)
MIN_RACES, MIN_CLOSE, MIN_CYCLES = 50, 15, 4
MIN_LEFT = 0.05
TREES = {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.02, "subsample": 0.8, "min_samples_leaf": 2 * MIN_RACES, "random_state": 0}
BAR_DRAWS = 2000
PLACEBO_DRAWS = 200
MISS_DRAWS = 10000
ALPHA = 0.05
SAFE = 0.8
FLAGS = ["first_term", "incumbent", "opp_quality", "opp_money", "own_surplus", "own_polling", "redrawn", "burn", "repeat"]
COMMITTEE = {"d": "dccc", "r": "nrcc"}
PARTY = {"d": "D", "r": "R"}


def side_measures(x, cycle, rows, vendors, early_donors):
    """{'d', 'r'}: frames by race of each nominee's June 30 measures. Missing stays missing, so a campaign with nothing to measure isn't read as low."""
    share = shares(rows)
    share = share[share.total >= MIN_BUDGET]
    states, out = x.district_id.str[:2], {}
    for side in "dr":
        cm, raised, spent = x[f"cm_{side}"], x[f"raised_jun_{side}"].where(lambda v: v > 0), x[f"spent_jun_{side}"]
        spend = x[f"spend_{side}"].where(lambda v: v >= MIN_BUDGET)
        people = race_measures(early_donors, cm, states)
        given = people.donors > 0
        out[side] = pd.DataFrame({
            "raised": raised, "surplus": (raised - spent).clip(lower=0), "burn": (spent / raised).clip(upper=3),
            "runway": x[f"runway_{side}"].where(cm.notna()), "debt": (x[f"debt_{side}"] / raised).clip(upper=3), "fund_cost": (x[f"fund_{side}"] / raised).clip(upper=3),
            **{name: cm.map(share[group]) if group in share else np.nan for name, group in SHARES.items()},
            "accel": x[f"late_{side}"] / spend, "vendor_hhi": cm.map(vendors.vendor_hhi.reindex(share.index)),
            "vendor_retention": cm.map(vendors.vendor_retention.reindex(share.index)) if cycle > CYCLES[0] else np.nan,
            "donors": people.donors.where(cm.notna()), "repeat": people.repeat.where(given) if cycle > CYCLES[0] else np.nan,
            "in_state": people.in_state.where(given), "top10": people.top10.where(given)}, index=x.index)
    return out


def side_items(x, measures, kept):
    """{'d', 'r'}: boolean frames by race, one column per item. `kept`: district_id -> share of an incumbent's district kept in a redraw.
    High and low are the top and bottom quartile among the cycle's nominees, so an item means the same share of campaigns in every year."""
    pooled = pd.concat(measures.values())
    low, high = pooled.quantile(QUARTILES[0]), pooled.quantile(QUARTILES[1])
    varies = pooled.max() > pooled.min()
    out = {}
    for side, other in ("dr", "rd"):
        m, opp, mine, theirs = measures[side], measures[other], x.status == f"{PARTY[side]} incumbent", x.status == f"{PARTY[other]} incumbent"
        table = {}
        for name in MEASURES:
            table[f"{name}_hi"], table[f"{name}_lo"] = (m[name] > high[name]) & varies[name], (m[name] <= low[name]) & varies[name]
        table.update(incumbent=mine, first_term=x[f"fresh_{side}"] != 0, challenger=theirs, quality=x[f"q_{side}"] == 1,
                     opp_quality=x[f"q_{other}"] == 1, opp_first_term=x[f"fresh_{other}"] != 0,
                     opp_raised_hi=opp.raised > high.raised, opp_raised_lo=opp.raised <= low.raised,
                     low_retention=mine & (x.district_id.map(kept) < LOW_RETENTION), open=x.status == "Open", redrawn=x.redrawn == 1)
        out[side] = pd.DataFrame(table, index=x.index)
    return out


def load():
    """(races, items, measures), each by cycle: add_spending's frames, side_items and side_measures."""
    races = {c: money_terms(build_races(c)) for c in CYCLES}
    rows = {c: load_oppexp(c) for c in CYCLES}
    pre, history = {}, []
    for c in CYCLES:
        through_june = rows[c][rows[c].date <= pd.Timestamp(f"{c}-06-30")]
        history.append(through_june)
        pre[c] = categorize(through_june, pd.concat(history, ignore_index=True))
    races = add_spending(races, pre)
    gifts, last_vendors, items, measures = load_donors(), None, {}, {}
    for c in CYCLES:
        vendors, last_vendors = committee_measures(c, pre[c], last_vendors)
        early = committee_donors(gifts_through_june(gifts[c], c), gifts.get(c - 2))
        measures[c] = side_measures(races[c], c, pre[c], vendors, early)
        redraw = incumbent_retention(c).drop_duplicates("district_id").set_index("district_id")
        items[c] = side_items(races[c], measures[c], redraw.retained.astype(float))
    return races, items, measures


def within(values, cell):
    """`values` with each cell's mean taken out."""
    return values - pd.Series(values).groupby(cell).transform("mean").to_numpy()


def search_frame(races, items):
    """{'frame', 'D', 'R', 'names'}: the backtest's contested races, with e, the surplus forecast's miss with the year's miss removed, and near, the
    races the baseline put within CLOSE; D and R are the two sides' items as boolean arrays in the same order."""
    parts, d, r = [], [], []
    for y in BACKTEST:
        t = races[y][races[y].margin.notna()]
        f, base = forecast_year(races, y, SURPLUS), forecast_year(races, y, list(FEATURES))
        keep = f["contested"]
        parts.append(pd.DataFrame({"cycle": y, "district_id": t.district_id, "status": t.status, "e": f["margin"] - (f["mu"] + f["miss"]),
                                   "near": np.abs(base["mu"]) < CLOSE})[keep])
        d.append(items[y]["d"].loc[t.index][keep].to_numpy(bool))
        r.append(items[y]["r"].loc[t.index][keep].to_numpy(bool))
    frame = pd.concat(parts, ignore_index=True)
    frame["cell"] = frame.cycle.astype(str) + frame.status
    frame["cluster"] = pd.factorize(frame.district_id.str[:2] + frame.cycle.astype(str))[0]
    return {"frame": frame, "D": np.vstack(d), "R": np.vstack(r), "names": list(items[BACKTEST[0]]["d"].columns)}


def signed(d, r, combo):
    return d[:, combo].all(axis=1).astype(float) - r[:, combo].all(axis=1).astype(float)


def what_is_left(d, r, combo, cell, depth=1):
    """(left, share): the combination with its cell means and its sub-combinations of up to `depth` items taken out, and the share of its variance that leaves."""
    z = within(signed(d, r, combo), cell)
    parts = [within(signed(d, r, list(sub)), cell) for size in range(1, min(depth, len(combo) - 1) + 1) for sub in combinations(combo, size)]
    parts = np.column_stack([p for p in parts if p.any()] or [np.zeros(len(z))])
    left = z - parts @ np.linalg.lstsq(parts, z, rcond=None)[0]
    return left, float(left @ left / (z @ z)) if z.any() else 0.0


def baskets(search):
    """The pairs and triples with enough support that are more than copies of their parts: {'combos', 'left' (races x baskets), 'n', 'close', 'cycles'}."""
    frame, d, r = search["frame"], search["D"], search["R"]
    near, cycle, cell = frame.near.to_numpy(), frame.cycle.to_numpy(), frame.cell.to_numpy()
    found = {"combos": [], "left": [], "n": [], "close": [], "cycles": []}
    for size in (2, 3):
        for combo in combinations(range(d.shape[1]), size):
            on = signed(d, r, list(combo)) != 0
            if on.sum() < MIN_RACES or on[near].sum() < MIN_CLOSE or len(set(cycle[on])) < MIN_CYCLES:
                continue
            left, share = what_is_left(d, r, list(combo), cell)
            if share < MIN_LEFT:
                continue
            found["combos"].append(combo)
            found["left"].append(left)
            found["n"].append(int(on.sum()))
            found["close"].append(int(on[near].sum()))
            found["cycles"].append(len(set(cycle[on])))
    found["left"] = np.column_stack(found["left"])
    return found


def cluster_sums(found, search):
    """Baskets x clusters: each state-year's contribution to each basket's statistic."""
    frame = search["frame"]
    e = within(frame.e.to_numpy(), frame.cell.to_numpy())
    clusters = frame.cluster.to_numpy()
    return np.column_stack([np.bincount(clusters, weights=found["left"][:, j] * e, minlength=clusters.max() + 1) for j in range(found["left"].shape[1])]).T


def flip_bar(sums, draws=BAR_DRAWS, seed=11, batch=200):
    """The largest |statistic| across baskets in each draw, flipping the sign of whole state-years at random. Never includes the real signs."""
    rng, scale, out = np.random.default_rng(seed), np.sqrt((sums ** 2).sum(axis=1)), []
    for start in range(0, draws, batch):
        signs = rng.choice([-1.0, 1.0], size=(sums.shape[1], min(batch, draws - start)))
        out.append((np.abs(sums @ signs).T / scale).max(axis=1))
    return np.concatenate(out)


def score_baskets(found, search, bar):
    """The real statistic for every basket, largest first. passes: clears the bar, with the same sign in at least MIN_CYCLES years. named: passes and, for a triple, none of
    its pairs passes and it beats them all, so a pair's coattails aren't a new combination. beyond_pairs: a passing triple's statistic with its pairs out too, for reading only."""
    frame, sums, combos = search["frame"], cluster_sums(found, search), found["combos"]
    cell, years = frame.cell.to_numpy(), frame.cycle.to_numpy()
    e, power = within(frame.e.to_numpy(), cell), (found["left"] ** 2).sum(axis=0)
    t = sums.sum(axis=1) / np.sqrt((sums ** 2).sum(axis=1))
    same = np.array([sum(np.sign(found["left"][years == y, j] @ e[years == y]) == np.sign(t[j]) for y in BACKTEST) for j in range(len(t))])
    passes = (np.abs(t) > bar) & (same >= MIN_CYCLES)
    by_combo = {c: j for j, c in enumerate(combos)}

    def beats_pairs(j):
        pairs = [by_combo[pair] for pair in combinations(combos[j], 2) if pair in by_combo]
        return all(not passes[i] and abs(t[j]) > abs(t[i]) for i in pairs)

    def beyond(j):
        if len(combos[j]) < 3 or not passes[j]:
            return np.nan
        left, _ = what_is_left(search["D"], search["R"], list(combos[j]), cell, depth=2)
        part = np.bincount(frame.cluster.to_numpy(), weights=left * e)
        return part.sum() / np.sqrt((part ** 2).sum())
    table = pd.DataFrame({"basket": [" + ".join(search["names"][i] for i in c) for c in combos], "n": found["n"], "close": found["close"], "cycles": found["cycles"],
                          "points": sums.sum(axis=1) / power, "se": np.sqrt((sums ** 2).sum(axis=1)) / power, "t": t, "same_sign_years": same, "passes": passes,
                          "named": [bool(passes[j] and (len(combos[j]) == 2 or beats_pairs(j))) for j in range(len(t))], "beyond_pairs": [beyond(j) for j in range(len(t))]})
    return table.reindex(table.t.abs().sort_values(ascending=False).index)


def fit_trees(d, r, miss):
    """Trees on each race seen from both sides, the miss flipped for the Republican's view. `d`, `r`: the sides' items as arrays; `miss`: signed toward Democrats."""
    return GradientBoostingRegressor(**TREES).fit(np.vstack([d, r]), np.concatenate([miss, -miss]))


def party_blind_shift(trees, d, r):
    """Half the difference between the two sides' readings, so swapping the sides flips the shift."""
    return (trees.predict(d) - trees.predict(r)) / 2


def tree_shift(races, items, year):
    """(shift, model): what depth-3 trees trained on earlier years' misses add to the surplus forecast of `year`'s scored races."""
    model = fit_baseline(pd.concat([r for c, r in races.items() if c < year]), SURPLUS)
    d, r, y = [], [], []
    for c, t in races.items():
        if c >= year:
            continue
        keep = t.margin.abs() < UNCONTESTED
        miss = (t.margin - predict_margin(t, model))[keep]
        y.append((miss - miss.groupby(t.status[keep]).transform("mean")).to_numpy(float))
        d.append(items[c]["d"][keep].to_numpy(float))
        r.append(items[c]["r"][keep].to_numpy(float))
    trees = fit_trees(np.vstack(d), np.vstack(r), np.concatenate(y))
    scored = races[year].margin.notna() if year != LIVE else pd.Series(True, index=races[year].index)
    return party_blind_shift(trees, items[year]["d"][scored].to_numpy(float), items[year]["r"][scored].to_numpy(float)), model


def reference(races):
    """year -> the surplus forecast and the baseline's close races, the comparison every gate() needs."""
    return {y: {**forecast_year(races, y, SURPLUS), "near": np.abs(forecast_year(races, y, list(FEATURES))["mu"]) < CLOSE} for y in BACKTEST}


def gate(races, items, ref):
    """The bar in spending_signal.py against the surplus forecast: {'within', 'raw', 'close', 'better', 'passes'}."""
    within_change, raw, close = [], [], []
    for y in BACKTEST:
        base = ref[y]
        shift, model = tree_shift(races, items, y)
        mu = base["mu"] + shift
        miss = (base["margin"] - mu)[base["contested"]].mean()
        p = norm.cdf(mu / model["sigma"])
        p_within = norm.cdf((mu + miss) / np.sqrt(model["sigma"] ** 2 - model["sigma_national"] ** 2))
        change = log_loss(p_within, base["won"]) - log_loss(base["p_within"], base["won"])
        within_change.append(change.mean())
        raw.append((log_loss(p, base["won"]) - log_loss(base["p"], base["won"])).mean())
        close.append(change[base["near"]].mean())
    better = sum(w < -1e-9 for w in within_change)
    return {"within": float(np.mean(within_change)), "raw": float(np.mean(raw)), "close": float(np.mean(close)), "better": better, "by_year": dict(zip(BACKTEST, within_change)),
            "passes": bool(better >= 4 and np.mean(raw) < 0 and np.mean(close) <= 0)}


def shuffle_items(items, rng):
    """`items` with whole rows shuffled among each year's races, the same way for both sides."""
    out = {}
    for c, sides in items.items():
        order = rng.permutation(len(sides["d"]))
        out[c] = {s: pd.DataFrame(t.to_numpy()[order], index=t.index, columns=t.columns) for s, t in sides.items()}
    return out


def placebo(races, items, ref, draws=PLACEBO_DRAWS, seed=7):
    """gate() on shuffled items, `draws` times: arrays of within, raw and passes."""
    rng = np.random.default_rng(seed)
    runs = [gate(races, shuffle_items(items, rng), ref) for _ in range(draws)]
    return {key: np.array([r[key] for r in runs]) for key in ("within", "raw", "passes")}


def own_view(t, measures, side):
    """One row per race from `side`'s view: the entry model's FLAGS, using only what a Sept 1 forecaster could see."""
    other = "r" if side == "d" else "d"
    money = (t[f"raised_jun_{other}"] - t[f"own_jun_{other}"]).clip(lower=0) + t[f"outside_aug_{other}"]
    return pd.DataFrame({"first_term": (t[f"fresh_{side}"] != 0).astype(float), "incumbent": (t.status == f"{PARTY[side]} incumbent").astype(float),
                         "opp_quality": (t[f"q_{other}"] == 1).astype(float), "opp_money": np.log1p(money / 1000), "own_surplus": np.log1p(measures.surplus.fillna(0.0) / 1000),
                         "own_polling": measures.polling.fillna(0.0), "redrawn": (t.redrawn == 1).astype(float), "burn": measures.burn.fillna(0.0), "repeat": measures.repeat.fillna(0.0)})


def safe_seats(races, measures, side="d", forecast="baseline"):
    """Contested races the forecast gave `side` at least SAFE, 2016-2024, from that side's view: FLAGS, mu_own, e_own (ahead of the forecast, the year's miss removed), entered."""
    parts, sign = [], 1.0 if side == "d" else -1.0
    for y in BACKTEST:
        decided = races[y].margin.notna()
        t = races[y][decided]
        if forecast == "sept1":
            f = dated_forecast(y, races, "09-01")
            mu, p = f["mu"][decided.to_numpy()], f["p"][decided.to_numpy()]
        else:
            f = forecast_year(races, y, SURPLUS if forecast == "surplus" else list(FEATURES))
            mu, p = f["mu"], f["p"]
        margin = t.margin.to_numpy(float)
        contested = np.abs(margin) < UNCONTESTED
        view = own_view(t, measures[y][side].loc[t.index], side).assign(
            cycle=y, district_id=t.district_id, status=t.status, mu_own=sign * mu, p_own=p if side == "d" else 1 - p, e_own=sign * (margin - mu - (margin - mu)[contested].mean()),
            entered=t[COMMITTEE[side]] > 0, committee_money=(t.dccc > 0) | (t.nrcc > 0), outside_money=(t.outside_d > 0) | (t.outside_r > 0), contested=contested)
        parts.append(view[(view.p_own >= SAFE) & view.contested])
    return pd.concat(parts, ignore_index=True)


def fit_entry(train, columns=None):
    columns = columns or FLAGS + ["mu_own"]
    scale = StandardScaler().fit(train[columns])
    return scale, LogisticRegression(C=1.0, max_iter=2000).fit(scale.transform(train[columns]), train.entered), columns


def flag_score(fit, frame):
    """The entry model's linear score with how safe the seat is left out, so a high score means flags, not closeness."""
    scale, model, columns = fit
    return scale.transform(frame[columns])[:, :len(FLAGS)] @ model.coef_[0][:len(FLAGS)]


def left_out_scores(seats):
    """Each year's flag_score from a model fit on the other years."""
    out = pd.Series(0.0, index=seats.index)
    for y in BACKTEST:
        out[seats.cycle == y] = flag_score(fit_entry(seats[seats.cycle != y]), seats[seats.cycle == y])
    return out


def entry_check(seats):
    """Fit without any result. Pooled AUC with each year fit on the others: how safe the seat is, that plus first term, and all FLAGS."""
    out = {}
    for name, columns in (("safety", ["mu_own"]), ("safety_first_term", ["first_term", "mu_own"]), ("all_flags", FLAGS + ["mu_own"])):
        guess = pd.Series(0.0, index=seats.index)
        for y in BACKTEST:
            train, test = seats[seats.cycle != y], seats[seats.cycle == y]
            scale, model, _ = fit_entry(train, columns)
            guess[test.index] = model.decision_function(scale.transform(test[columns]))
        out[name] = float(roc_auc_score(seats.entered, guess))
    return out


def miss_parts(seats, score):
    """Seats no committee entered: x, e and s (the miss and the score, with cell means taken out, and how safe the seat is out of the score), and each state-year's sum."""
    x = seats.assign(score=score)[~seats.committee_money]
    cell = (x.cycle.astype(str) + x.status).to_numpy()
    e, s, mu = within(x.e_own.to_numpy(), cell), within(x.score.to_numpy(), cell), within(x.mu_own.to_numpy(), cell)
    s = s - mu * (mu @ s) / (mu @ mu)
    clusters = pd.factorize(x.district_id.str[:2] + x.cycle.astype(str))[0]
    return x, e, s, clusters, np.bincount(clusters, weights=s * e)


def miss_test(seats, score, draws=MISS_DRAWS, seed=5):
    """Among seats no committee entered, do higher entry scores go with running further behind the forecast? Whole state-years are flipped for
    the p-value. Passes with a negative sign, p under ALPHA and a negative sign in at least 4 years."""
    x, e, s, _, sums = miss_parts(seats, score)
    t = sums.sum() / np.sqrt((sums ** 2).sum())
    flips = np.random.default_rng(seed).choice([-1.0, 1.0], size=(len(sums), draws))
    p = float((np.abs(sums @ flips) / np.sqrt((sums ** 2).sum()) >= abs(t)).mean())
    negative = sum((s[x.cycle.to_numpy() == y] @ e[x.cycle.to_numpy() == y]) < 0 for y in x.cycle.unique())
    top = x.score >= x.score.quantile(0.9)
    return {"seats": len(x), "points_per_sd": float((s @ e) / (s @ s) * s.std()), "t": float(t), "p": p, "years_negative": int(negative),
            "passes": bool(t < 0 and p < ALPHA and negative >= 4), "top_tenth": float(x.e_own[top].mean()), "top_tenth_seats": int(top.sum()), "rest": float(x.e_own[~top].mean()),
            "entered": float(seats.e_own[seats.committee_money].mean()), "entered_seats": int(seats.committee_money.sum())}


def live_files(races, items, measures, seats):
    """The two 2026 records, written without any 2026 result: the trees' shift on the surplus forecast, and the entry score on the frozen baseline."""
    now = races[LIVE]
    shift, model = tree_shift(races, items, LIVE)
    mu = predict_margin(now, model)
    pd.DataFrame({"district_id": now.district_id, "mu_surplus": mu, "shift": shift, "mu_with_shift": mu + shift, "p_surplus": norm.cdf(mu / model["sigma"]),
                  "p_with_shift": norm.cdf((mu + shift) / model["sigma"])}).round(6).to_csv(CANDIDATE, index=False)
    frozen = pd.read_csv(FROZEN).set_index("district_id")
    view = own_view(now, measures[LIVE]["d"], "d").assign(district_id=now.district_id, mu_own=now.district_id.map(frozen.mu_baseline), p_own=now.district_id.map(frozen.p_baseline),
                                                          dollars=now.district_id.map(frozen.dccc_coordinated).fillna(0.0), both_nominees=now.cm_d.notna() & now.cm_r.notna())
    view = view[(view.p_own >= SAFE) & view.both_nominees]
    fit = fit_entry(seats)
    view = view.assign(entered=view.dollars > 0, score=flag_score(fit, view))
    view[["district_id", "mu_own", "p_own", "dollars", "entered", "score"]].round(6).to_csv(ENTRY, index=False)
    return {"seats": len(view), "entered": int(view.entered.sum()), "auc": float(roc_auc_score(view.entered, fit[1].decision_function(fit[0].transform(view[fit[2]]))))}


def run():
    """The real run. Writes data/basket_results.json and the two 2026 records, and refuses to run twice."""
    if RESULTS.exists():
        raise SystemExit(f"{RESULTS.name} already exists. Move it aside on purpose if you mean to run this again.")
    races, items, measures = load()
    if races[LIVE].margin.notna().any():
        raise SystemExit("2026 results are in the data, so the 2026 records can't be written blind.")
    seats = safe_seats(races, measures)
    entry = {"check": entry_check(seats), "seats": len(seats), "entered": int(seats.entered.sum()), "live": live_files(races, items, measures, seats)}

    ref = reference(races)
    real, shuffled = gate(races, items, ref), placebo(races, items, ref)
    trees = {**real, "placebo_pass_rate": float(shuffled["passes"].mean()), "as_good_within": float((shuffled["within"] <= real["within"]).mean()), "as_good_raw": float((shuffled["raw"] <= real["raw"]).mean())}
    trees["clears"] = bool(real["passes"] and trees["as_good_within"] <= ALPHA)

    search = search_frame(races, items)
    found = baskets(search)
    draws = flip_bar(cluster_sums(found, search))
    bar = float(np.quantile(draws, 0.95))
    table = score_baskets(found, search, bar)

    score = left_out_scores(seats)
    variants = {"main": miss_test(seats, score), "no_outside_money": miss_test(seats.assign(committee_money=seats.committee_money | seats.outside_money), score),
                "without_2020": miss_test(seats[seats.cycle != 2020], score[seats.cycle != 2020])}
    for name, frame in {"sept1": safe_seats(races, measures, forecast="sept1"), "surplus": safe_seats(races, measures, forecast="surplus"), "nrcc": safe_seats(races, measures, side="r")}.items():
        variants[name] = miss_test(frame, left_out_scores(frame))
    results = {"entry": entry, "trees": trees, "baskets": {"pool": len(search["names"]), "scored": len(table), "bar": bar, "passing": int(table.passes.sum()), "named": int(table.named.sum()), "table": table.to_dict("records")},
               "miss_test": variants}
    RESULTS.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: v for k, v in results.items() if k != "baskets"}, indent=2, default=float))
    print(table.head(20).to_string())


def plant(races, items, combos, deltas):
    """`races` with each combination moving contested margins by its delta, toward the side that has it."""
    out = {}
    for c, t in races.items():
        margin = t.margin
        for combo, delta in zip(combos, deltas):
            z = items[c]["d"].iloc[:, combo].all(axis=1).astype(float) - items[c]["r"].iloc[:, combo].all(axis=1).astype(float)
            margin = margin + delta * z * (t.margin.abs() < UNCONTESTED)
        out[c] = t.assign(margin=margin)
    return out


def tree_power(races, items, combos, k, delta, cutoff, draws, seed=19):
    """On shuffled items, so nothing real is left: plant `k` combinations of `delta` points with random signs. Share clearing the bar, and clearing it and the placebo's `cutoff`."""
    rng, runs = np.random.default_rng(seed), []
    for _ in range(draws):
        mixed = shuffle_items(items, rng)
        picks = rng.choice(len(combos), k, replace=False)
        world = plant(races, mixed, [list(combos[i]) for i in picks], delta * rng.choice([-1.0, 1.0], k))
        runs.append(gate(world, mixed, reference(world)))
    return {"bar": float(np.mean([r["passes"] for r in runs])), "bar_and_placebo": float(np.mean([r["passes"] and r["within"] <= cutoff for r in runs])), "within": float(np.mean([r["within"] for r in runs]))}


def basket_power(found, search, sums, bar, delta, draws=500, picks=100, seed=13):
    """Per sampled basket, the share of flips where `delta` points planted on it clear the bar: (n, power) arrays."""
    frame, d, r = search["frame"], search["D"], search["R"]
    rng, clusters, cell = np.random.default_rng(seed), frame.cluster.to_numpy(), frame.cell.to_numpy()
    chosen = rng.choice(len(found["combos"]), min(picks, len(found["combos"])), replace=False)
    power = []
    for j in chosen:
        z = within(signed(d, r, list(found["combos"][j])), cell)
        planted = delta * np.bincount(clusters, weights=found["left"][:, j] * z, minlength=sums.shape[1])
        total = sums[j][:, None] * rng.choice([-1.0, 1.0], size=(sums.shape[1], draws)) + planted[:, None]
        power.append(float((np.abs(total.sum(axis=0)) / np.sqrt((total ** 2).sum(axis=0)) > bar).mean()))
    return np.array(found["n"])[chosen], np.array(power)


def miss_power(seats, score, delta, draws=2000, seed=17):
    """Share of flips where `delta` points per standard deviation of the score, planted on the miss, reach p under ALPHA."""
    _, _, s, clusters, sums = miss_parts(seats, score)
    rng = np.random.default_rng(seed)
    critical = np.quantile(np.abs(sums @ rng.choice([-1.0, 1.0], size=(len(sums), draws))) / np.sqrt((sums ** 2).sum()), 1 - ALPHA)
    planted = delta * np.bincount(clusters, weights=s * s) / s.std()
    total = sums[:, None] * rng.choice([-1.0, 1.0], size=(len(sums), draws)) + planted[:, None]
    return float((np.abs(total.sum(axis=0)) / np.sqrt((total ** 2).sum(axis=0)) > critical).mean())


def blind(races, items, measures):
    """The numbers in BASKET_DESIGN.md, from flipped, shuffled or planted data. Nothing here computes a real basket's statistic, the trees' real gate or the real miss test."""
    search = search_frame(races, items)
    frame, found = search["frame"], baskets(search)
    sums = cluster_sums(found, search)
    draws = flip_bar(sums)
    bar = float(np.quantile(draws, 0.95))
    se = np.sqrt((sums ** 2).sum(axis=1)) / (found["left"] ** 2).sum(axis=0)
    n = np.array(found["n"])
    bands = {"50-99": n < 100, "100-199": (n >= 100) & (n < 200), "200+": n >= 200}
    print(f"contested {len(frame)}, close {int(frame.near.sum())}, state-years {frame.cluster.nunique()}, residual sd {frame.e.std():.2f}")
    print(f"items {len(search['names'])}: {', '.join(search['names'])}")
    print(f"item shares by side, 2024: {items[2024]['d'].mean().round(2).to_dict()}")
    print(f"baskets scored {len(n)}, pairs {sum(len(c) == 2 for c in found['combos'])}, triples {sum(len(c) == 3 for c in found['combos'])}, races median {np.median(n):.0f}")
    print(f"bar: median {np.median(draws):.2f}, 95th {bar:.2f}")
    for name, mask in bands.items():
        print(f"  {name} races: {mask.sum()} baskets, smallest clearing effect, median {np.median(bar * se[mask]):.1f} points" if mask.any() else f"  {name} races: none")
    for delta in (2, 3, 4, 6):
        sizes, power = basket_power(found, search, sums, bar, delta)
        print(f"  {delta} points planted on one basket clears the bar: " + ", ".join(f"{name} {np.median(power[(sizes >= lo) & (sizes < hi)]):.0%}" for name, lo, hi in (("50-99", 50, 100), ("100-199", 100, 200), ("200+", 200, 10 ** 6)) if ((sizes >= lo) & (sizes < hi)).any()))
    ref = reference(races)
    shuffled = placebo(races, items, ref)
    cutoff = float(np.quantile(shuffled["within"], ALPHA))
    print(f"trees on shuffled items: bar passes {shuffled['passes'].mean():.0%}, within {np.median(shuffled['within']):+.5f} median, 5th percentile {cutoff:+.5f}; raw {np.median(shuffled['raw']):+.5f}")
    for k, delta in ((1, 6), (10, 3), (25, 1.5), (25, 3)):
        print(f"  trees, {k} planted at {delta} points:", tree_power(races, items, found["combos"], k, delta, cutoff, draws=100), flush=True)
    seats = safe_seats(races, measures)
    score = left_out_scores(seats)
    print(f"DCCC safe seats {len(seats)}, entered {int(seats.entered.sum())}, no committee money {int((~seats.committee_money).sum())}, by year {seats.groupby('cycle').entered.sum().to_dict()}")
    for delta in (0.5, 1, 1.5, 2):
        print(f"  miss test, {delta} points per sd of score reaches p < {ALPHA}: {miss_power(seats, score, delta):.0%}")
    frozen = pd.read_csv(FROZEN)
    now = races[LIVE].assign(p=races[LIVE].district_id.map(frozen.set_index("district_id").p_baseline), dollars=races[LIVE].district_id.map(frozen.set_index("district_id").dccc_coordinated))
    safe = now[(now.p >= SAFE) & now.cm_d.notna() & now.cm_r.notna()]
    print(f"2026 frozen baseline: safe with both nominees {len(safe)}, DCCC coordinated {int((safe.dollars > 0).sum())}; all safe {int((now.p >= SAFE).sum())}")


if __name__ == "__main__":
    blind(*load()) if "--blind" in sys.argv else run()
