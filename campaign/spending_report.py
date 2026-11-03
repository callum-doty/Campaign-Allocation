"""
Write SPENDING.md: what House campaigns buy, from FEC operating expenditures, and
whether it improves the forecast. Not part of run.py: it needs about 450MB of
extra downloads, plus about 23GB placed by hand (load_donors.py, redrawn_retention.py),
and its answer so far is that the forecast doesn't need it.

    python -m campaign.spending_report
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import backtest
from .brand_test import (GRID, MONEY_LABELS, add_money_bands, error_by_money, error_size_test, fixed_passthrough, money_slopes, national_passthrough,
                         seat_sensitivity)
from .build_races import build_races
from .channel_mix import BANDS, BAND_LABELS, HOUSE_MAJORITY_PAC
from .classify_purpose import VAGUE, classify_text, vendor_key
from .competitive_response import first_movers, response, weekly_panel
from .fit_baseline import fit_baseline
from .freeze_forecast import money_terms
from .google_ads import add_google
from .load_candidates import load_candidates
from .load_donors import DONOR_FEATURES, add_donors, coverage, gifts_through_june, load_donors
from .load_expenditures import load_expenditures
from .load_media_markets import load_media_markets
from .load_oppexp import categorize, load_oppexp, nominee_committees
from .load_results import load_results
from .market_spillover import shared_markets, spillover_fit
from .mine_spending import (MIN_BUDGET, baskets, choose_k, clr, kmeans, rules, shares, spent_at_all, tv_substitution)
from .national_test import national_table
from .predict_margin import predict_margin
from .redrawn_retention import (add_new_voters, carried_over, forecast_margin, incumbent_retention, retention_bands, retention_slope,
                                retention_test)
from .spending_features import SECOND, add_spending
from .spending_signal import BACKTEST, SPECS, SURPLUS, bootstrap, change, cluster_signal, placebo, score_specs

ROOT = Path(__file__).resolve().parents[1]
CYCLES = (2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026)
DONE = CYCLES[:-1]
LIVE = CYCLES[-1]
LABELS = {"persuasion_odds": "Money raised less fundraising costs", "contact_odds": "Early voter-contact money", "fund_share": "Share spent on fundraising",
          "admin_share": "Share spent on payroll and overhead", "inc_poll": "Incumbent's early polling", "pro_vendor": "Share at vendors with a competitive-race record",
          "accel": "Share of spending in April-June", "split": "Fundraising, contact and overhead together",
          "burn": "Spent as a share of raised", "runway": "Months of cash left", "fund_cost": "Fundraising cost as a share of raised",
          "digital_mail": "Digital and mail together", "polling_digital": "Polling and digital together", "broadcast_field": "TV and field together",
          "poll_first": "Polled before first voter contact", "contact_lead": "How early voter contact began", "vendor_hhi": "Vendor concentration",
          "vendor_retention": "Kept last cycle's vendors", "debt_burden": "Debts as a share of raised",
          "donor_odds": "Itemized donors", "local_odds": "Itemized donors in the state", "top10_share": "Share from the top 10 donors",
          "repeat_share": "Share from last cycle's donors", "in_state_share": "Share from the state",
          "new_voters": "Share of the district new to its incumbent", "new_voters_within": "The same, against that redraw's average",
          "pvi_low": "Lean counts differently under $2M seen by Sept 1", "pvi_log": "Lean counts differently by money seen by Sept 1"}
CHOSEN_K = 3
LIVE_TESTS = {"accel": ROOT / "data" / "spending_candidate_2026.csv", "new_voters_within": ROOT / "data" / "retention_candidate_2026.csv"}
MIN_CLUSTERS = 5
TRAIN_THROUGH = 2018


def gold_accuracy(history):
    """Accuracy on the blind fresh sample, by how it was sampled: {sample: (accuracy, n)}."""
    gold = pd.read_csv(ROOT / "data" / "purpose_gold.csv", dtype={"fec_category": str})
    gold = gold[gold["round"] == "fresh"].assign(stage1=lambda g: classify_text(g.purpose, g.fec_category), vendor=lambda g: vendor_key(g.vendor))
    out = {}
    for cycle, part in gold.groupby("cycle"):
        out[cycle] = categorize(part, history[history.cycle <= cycle])
    gold = pd.concat(out.values())
    return {s: (float((g.category == g.label).mean()), len(g)) for s, g in gold.groupby("sample")}


def who(cycle, races):
    candidates = load_candidates(cycle, load_results(cycle)).drop_duplicates("cand_id").set_index("cand_id")
    cm = nominee_committees(cycle)
    t = pd.DataFrame({"committee_id": cm.to_numpy(), "cycle": cycle, "district_id": candidates.district_id.reindex(cm.index).to_numpy(),
                      "party": candidates.party.reindex(cm.index).to_numpy(), "ici": candidates.ici.reindex(cm.index).to_numpy()})
    r = races[cycle].set_index("district_id")
    return t.assign(margin=t.district_id.map(r.margin), tier=t.district_id.map(r.tier)).drop_duplicates("committee_id").set_index(["cycle", "committee_id"])


def number_table(frame, formats):
    head = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    return head + "\n".join("| " + " | ".join(formats.get(c, str)(v) for c, v in zip(frame.columns, row)) + " |" for row in frame.itertuples(index=False))


def spending_report():
    print("loading races and expenditures")
    built = {c: build_races(c) for c in CYCLES}
    races = {c: money_terms(r) for c, r in built.items()}
    rows = {c: load_oppexp(c) for c in CYCLES}
    pre, full, history_pre, history_all = {}, {}, [], []
    for c in CYCLES:
        through_june = rows[c][rows[c].date <= pd.Timestamp(f"{c}-06-30")]
        history_pre.append(through_june)
        history_all.append(rows[c])
        pre[c] = categorize(through_june, pd.concat(history_pre, ignore_index=True))
        full[c] = categorize(rows[c], pd.concat(history_all, ignore_index=True))
    every_row = pd.concat(history_pre, ignore_index=True)
    races = add_spending(races, pre)
    print("reading donors")
    donor_rows = load_donors()
    races = add_money_bands(add_new_voters(add_google(add_donors(races, donor_rows))))
    people = pd.concat([who(c, races) for c in DONE])

    # how much there is, and what it bought
    all_full = pd.concat(full[c] for c in DONE)
    total = all_full.amount.sum()
    bought = (all_full.groupby("category").amount.sum() / total).sort_values(ascending=False)
    bought_pre = (pd.concat(pre[c] for c in DONE).groupby("category").amount.sum() / sum(pre[c].amount.sum() for c in DONE)).reindex(bought.index)
    vague_share = float(all_full.amount[all_full.category.isin(VAGUE)].sum() / total)
    gold = gold_accuracy(every_row)

    # the forecast test
    print("scoring the forecast")
    scores, runs = score_specs(races)
    within, raw = change(scores, "within"), change(scores, "log_loss")
    close, brier, margin_err = change(scores, "close_within"), change(scores, "brier"), change(scores, "margin_within")
    all_done = pd.concat([races[c] for c in DONE])
    fits = {n: fit_baseline(all_done, f) for n, f in SPECS.items() if n not in ("baseline", "surplus")}
    passes = [n for n in LABELS if within.better[n] >= 4 and raw["mean"][n] < 0 and close["mean"][n] <= 0]
    signal_rows, intervals = [], {}
    for n in LABELS:
        term = "fund_odds" if n == "split" else n
        mean, low, high = bootstrap(runs, n)
        intervals[n] = (low, high)
        signal_rows.append({"Feature": LABELS[n], "Fit 2012-2024: coefficient (t)": f"{fits[n]['coef'][term]:+.2f} ({fits[n]['coef'][term] / fits[n]['se'][term]:+.1f})" if n != "split" else "see text",
                            "Log loss, miss removed": f"{within['mean'][n]:+.5f}", "Years better": f"{within.better[n]:.0f} of 5", "95% interval": f"{low:+.4f} to {high:+.4f}",
                            "Raw log loss": f"{raw['mean'][n]:+.5f}", "Close races": f"{close['mean'][n]:+.4f}", "Margin error, points": f"{margin_err['mean'][n]:+.3f}", "Clears the bar": "yes" if n in passes else "no"})
    print("running the placebo")
    check = {n: placebo(races, n) for n in passes if n != "split"}
    chance = {n: 1 - (1 - c["pass_rate"]) ** len(LABELS) for n, c in check.items()}
    surplus_gain = change(scores, "within", "baseline").loc["surplus"]

    # donors
    nominee_rows = {c: donor_rows[c][donor_rows[c].committee_id.isin(set(nominee_committees(c)))] for c in DONE}
    gifts = sum(len(gifts_through_june(nominee_rows[c], c)) for c in DONE)
    donor_cover = [coverage(nominee_rows[c], c) for c in DONE]
    donor_passes = sum(n in passes for n in DONOR_FEATURES)
    all_cycles_done = pd.concat([races[c] for c in DONE])
    donor_corr = all_cycles_done.donor_odds.corr(all_cycles_done.floor_share)

    # Google
    google_years = (2022, 2024)
    google_scores, _ = score_specs(races, {"baseline": SPECS["baseline"], "surplus": SURPLUS, "google": SURPLUS + ["google_odds"]}, years=google_years)
    google_within, google_raw = change(google_scores, "within").loc["google"], change(google_scores, "log_loss").loc["google"]
    google_money = {c: races[c].google_d.sum() + races[c].google_r.sum() for c in (2020, 2022, 2024)}
    google_either = {c: ((races[c].google_d > 0) | (races[c].google_r > 0)).sum() for c in (2020, 2022, 2024)}
    google_fit = fit_baseline(pd.concat([races[c] for c in (2020, 2022, 2024)]), SURPLUS + ["google_odds"])

    # the generic ballot as a brand survey
    print("testing pass-through")
    national = national_table()
    pass_national = national_passthrough(national).set_index(["poll", "group"])
    pass_scores = fixed_passthrough(races)
    pass_change = pass_scores.pivot(index="spec", columns="year", values="log_loss").reindex(list(GRID))
    pass_change = pass_change.sub(pass_change.loc["1:1"])
    pass_years = list(pass_change.columns)
    pass_without = fixed_passthrough(races, leave_out=2020).pivot(index="spec", columns="year", values="log_loss").reindex(list(GRID))
    pass_without = pass_without.sub(pass_without.loc["1:1"])
    pass_seats = seat_sensitivity(races)
    big_presidential = national[[c % 4 == 0 and abs(g) >= 5 for c, g in zip(national.index, national.gb)]]
    slopes = money_slopes(pd.concat([races[c] for c in DONE]))
    error_all, error_sept = error_by_money(races, "all_money"), error_by_money(races, "sept_money")
    error_size = error_size_test(races)
    low_counts = {c: int(races[c].low_all.sum()) for c in DONE}

    # clusters
    print("clustering")
    table = pd.concat({c: shares(pre[c]) for c in DONE}, names=["cycle"]).query("total >= @MIN_BUDGET")
    x = clr(table).to_numpy()
    ks = choose_k(x)
    fitted = kmeans(x, CHOSEN_K)
    clustered = table.assign(cluster=fitted.labels_).join(people)
    clustered["won"] = np.where(clustered.margin.isna(), np.nan, np.where(clustered.party == "D", clustered.margin > 0, clustered.margin < 0))
    profile = clustered.drop(columns=["total"]).groupby("cluster")[list(table.columns.drop("total"))].mean()
    order = profile.Broadcast.sort_values(ascending=False).index
    names = {order[0]: "Early TV", }
    rest = [c for c in order[1:]]
    incumbent_share = clustered.groupby("cluster").ici.apply(lambda s: (s == "I").mean())
    maintenance = incumbent_share[rest].idxmax()
    names[maintenance] = "Incumbent maintenance"
    names[[c for c in rest if c != maintenance][0]] = "Broad build-up"
    wins = clustered.pivot_table(index="cluster", columns="ici", values="won", aggfunc="mean")
    forecast_clusters = cluster_signal(races, pre, CHOSEN_K)

    # TV
    fall_parts = []
    print("backtesting for the TV test")
    bt = {c: backtest(c, built) for c in (2022, 2024)}
    for cycle, b in bt.items():
        markets = load_media_markets(cycle)[["district_id", "reach"]]
        close_races = b["races"][["district_id", "mu"]].merge(markets, on="district_id")
        after = full[cycle][(full[cycle].date > pd.Timestamp(f"{cycle}-06-30"))]
        s = shares(after).join(people.xs(cycle, level=0)[["district_id", "party", "ici"]], how="inner").reset_index()
        fall_parts.append(s.merge(close_races, on="district_id").assign(cycle=cycle))
    fall = pd.concat(fall_parts)
    fall = fall[(fall.mu.abs() < 15) & (fall.total >= MIN_BUDGET)]
    tv_table, tv_slopes = tv_substitution(fall)
    independent = []
    for cycle, b in bt.items():
        close_ids = b["races"].district_id[b["races"].mu.abs() < 15]
        ie = load_expenditures(cycle).merge(load_media_markets(cycle)[["district_id", "reach"]], on="district_id")
        ie = ie[ie.district_id.isin(close_ids)].assign(band=lambda d: pd.cut(d.reach, BANDS, labels=BAND_LABELS))
        independent += [ie[ie.bucket == "dccc"].assign(group="DCCC"), ie[ie.spe_id == HOUSE_MAJORITY_PAC].assign(group="House Majority PAC"),
                        ie[ie.bucket.str.startswith("outside")].assign(group="All outside groups")]
    mix = pd.concat(independent).groupby(["group", "band", "channel"], observed=True).amount.sum().unstack(fill_value=0.0)
    mix = mix.div(mix.sum(axis=1), axis=0)
    independent_rows = pd.DataFrame([{"Spender": group, "TV market": band, "TV": mix.TV[(group, band)], "Digital": mix.Digital[(group, band)], "Mail": mix.Mail[(group, band)],
                                      "Media, unspecified": mix["Media, unspecified"][(group, band)]}
                                     for group in ("DCCC", "House Majority PAC", "All outside groups") for band in mix.loc[group].index])
    def tv_change(group):
        return mix.TV[(group, mix.loc[group].index[-1])] - mix.TV[(group, mix.loc[group].index[0])]

    # the other ideas
    print("measuring committee response")
    panels = {c: weekly_panel(c) for c in DONE}
    pairs = [("DCCC", "dccc", "NRCC", "nrcc"), ("NRCC", "nrcc", "DCCC", "dccc"), ("DCCC", "dccc", "its outside allies", "outside_d"), ("NRCC", "nrcc", "its outside allies", "outside_r")]
    reactions = {(a, c): response(panels, b, d) for a, b, c, d in pairs}
    reaction_view = number_table(pd.DataFrame([{"This week's money": a, "Against": c, "Their money last week (se)": f"{r['lag'][0]:+.3f} ({r['lag'][1]:.3f})", "Their money next week (se)": f"{r['lead'][0]:+.3f} ({r['lead'][1]:.3f})",
                                                "Own money last week": f"{r['own_lag'][0]:+.3f}"} for (a, c), r in reactions.items()]), {})
    reaction_n = next(iter(reactions.values()))
    movers = first_movers(DONE)
    close_markets = shared_markets({c: b["races"][["district_id", "mu", "dccc", "nrcc", "outside_d", "outside_r"]].assign(committee=lambda d: d.dccc + d.nrcc, outside=lambda d: d.outside_d + d.outside_r) for c, b in bt.items()})
    spill_committee, spill_outside = spillover_fit(close_markets, "committee"), spillover_fit(close_markets, "outside")
    redraws = (2012, 2016, 2018, 2022, 2024)
    retained = {c: retention_test(forecast_margin(built, c), c) for c in redraws}
    pooled = pd.concat(retained.values(), ignore_index=True)
    slope_all, slope_state = retention_slope(pooled), retention_slope(pooled, within_state=True)
    slope_close, slope_forward = retention_slope(pooled[pooled.mu.abs() < 15]), retention_slope(pooled[pooled.cycle > redraws[0]])
    without_state = [retention_slope(pooled[pooled.district_id.str[:2] != s])["slope"] for s in pooled.district_id.str[:2].unique()]
    carry = {y: carried_over(pooled[pooled.cycle < y], retained[y]) for y in redraws[1:]}
    carry_better = [y for y, (before, after) in carry.items() if after < before]
    redraw_rows = []
    for c in redraws:
        fit = retention_slope(retained[c])
        redraw_rows.append({"Redraw": c, "States": fit["states"], "Incumbents": fit["n"], "Median share kept": f"{retained[c].retained.median():.0%}", "Points per 100% kept": f"{fit['slope']:+.1f}",
                            "Standard error": f"{fit['se']:.1f}" if fit["states"] >= MIN_CLUSTERS else "too few states"})
    redraw_view = number_table(pd.DataFrame(redraw_rows), {})
    retained_view = number_table(retention_bands(pooled).rename(columns={"band": "Share of the new district they kept", "size": "Incumbents", "mean": "Points against the redraw's average"}),
                                 {"Points against the redraw's average": "{:+.1f}".format})
    incumbency_2022 = bt[2022]["model"]["coef"]["d_incumbent"] / 2
    kept_now = incumbent_retention(LIVE)
    live_mu = forecast_margin(built, LIVE).set_index("district_id").mu
    low_now = kept_now[(kept_now.district_id.map(live_mu).abs() < 15) & (kept_now.retained < 0.6)].sort_values("district_id")
    redrawn_now = sorted(set(kept_now.district_id.str[:2]))
    model_redrawn = sorted(set(built[LIVE][built[LIVE].redrawn == 1].district_id.str[:2]))

    # rules
    print("mining rules")
    budgets = pd.concat({c: shares(full[c]) for c in DONE}, names=["cycle"]).query("total >= @MIN_BUDGET")
    strata = budgets[["total"]].join(people[["ici", "tier"]])
    strata["tercile"] = strata.groupby(level=0).total.transform(lambda t: pd.qcut(t, 3, labels=False))
    strata = (strata.tercile.astype(str) + "|" + strata.ici.fillna("?") + "|" + np.where(strata.tier.isin(["Safe D", "Safe R"]), "safe", "contested"))

    def crosstab(parts):
        long = pd.concat([p.assign(cycle=c) for c, p in parts.items()]).set_index(["cycle", "committee_id"])
        long = long[long.index.isin(strata.index)].reset_index()
        return pd.crosstab([long.cycle, long.committee_id], long["item"]).astype(bool).reindex(strata.index, fill_value=False)
    naive_table = crosstab({c: spent_at_all(full[c], c) for c in DONE})
    naive_rules = rules(naive_table, strata, forward=False, max_antecedent=1)
    item_table = crosstab({c: baskets(full[c], c) for c in DONE})
    train = item_table[item_table.index.get_level_values(0) <= TRAIN_THROUGH]
    test = item_table[item_table.index.get_level_values(0) > TRAIN_THROUGH]
    found = rules(train, strata)

    def holds(row):
        a, b = row["if"].split(" + "), test[row["then"]]
        has = test[a].all(axis=1)
        frame = pd.DataFrame({"a": has, "b": b, "s": strata.reindex(test.index)})
        sizes = frame.groupby("s").agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum"))
        return pd.Series({"test_confidence": (has & b).sum() / max(has.sum(), 1), "test_conditional": (has & b).sum() / max((sizes.a * sizes.b / sizes.n).sum(), 1e-9)})
    found = found.join(found.apply(holds, axis=1))
    category = lambda s: {i.split(" ", 1)[1] for i in s.split(" + ")}
    found["cross"] = [not (category(a) & category(b)) for a, b in zip(found["if"], found["then"])]
    strong = found[found.conditional_lift >= 1.2]
    cross = found[found.cross & (found.conditional_lift >= 1.2) & (found.test_conditional >= 1.1)]
    single_cross = cross[~cross["if"].str.contains(r" \+ ")].sort_values("test_conditional", ascending=False)

    # vendors
    latest = full[DONE[-1]]
    latest = latest[(latest.amount > 0) & (latest.vendor != "")].merge(people.xs(DONE[-1], level=0)[["party"]], left_on="committee_id", right_index=True)
    by_party = latest.groupby(["vendor", "party"]).amount.sum().unstack(fill_value=0.0)
    both_parties = (by_party.D > 0) & (by_party.R > 0)
    vendor_rows = []
    for c in ("Broadcast", "Digital", "Fundraising", "Mail/print", "Polling/research", "Field/phones"):
        q = latest[latest.category == c].groupby("vendor").amount.sum().sort_values(ascending=False)
        vendor_rows.append({"Category": c, "Dollars": f"${q.sum() / 1e6:.0f}M", "Vendors": f"{len(q):,}", "Top 5 share": f"{q.head(5).sum() / q.sum():.0%}", "Largest": ", ".join(q.head(3).index)})

    # the live tests, written down before results
    for candidate, live_path in LIVE_TESTS.items():
        if candidate in passes and not live_path.exists():
            model = fit_baseline(all_done, SURPLUS + [candidate])
            base_model = fit_baseline(all_done, SURPLUS)
            now = races[LIVE]
            pd.DataFrame({"district_id": now.district_id, candidate: now[candidate], "mu_surplus": predict_margin(now, base_model), "mu_with_candidate": predict_margin(now, model),
                          "p_surplus": norm.cdf(predict_margin(now, base_model) / base_model["sigma"]), "p_with_candidate": norm.cdf(predict_margin(now, model) / model["sigma"])}).round(6).to_csv(live_path, index=False)

    pct = lambda v: f"{v:.0%}"
    bought_table = number_table(pd.DataFrame({"Category": bought.index, "Whole cycle": bought.to_numpy(), "Through June 30": bought_pre.to_numpy()}), {"Whole cycle": pct, "Through June 30": pct})
    signal_table = number_table(pd.DataFrame(signal_rows), {})
    profile_rows = pd.DataFrame({"Share of pre-June 30 spending": profile.columns, **{names[c]: profile.loc[c].to_numpy() for c in order}})
    profile_table = number_table(profile_rows, {names[c]: pct for c in order})
    sizes = clustered.groupby("cluster").agg(n=("total", "size"), median=("total", "median"))
    cluster_rows = pd.DataFrame({"Cluster": [names[c] for c in order], "Campaigns": [f"{sizes.n[c]:,}" for c in order], "Median spent by June 30": [f"${sizes['median'][c] / 1e3:,.0f}K" for c in order],
                                 "Incumbents": [pct(incumbent_share[c]) for c in order], "Incumbents won": [pct(wins["I"][c]) for c in order],
                                 "Challengers won": [pct(wins["C"][c]) for c in order], "Open-seat nominees won": [pct(wins["O"][c]) for c in order]})
    cluster_table = number_table(cluster_rows, {})
    tv_rows = tv_table.reset_index().rename(columns={"band": "TV market"})[["TV market", "candidates", "Broadcast", "Digital", "Mail/print", "Consulting/other"]]
    tv_view = number_table(tv_rows.rename(columns={"candidates": "Candidates"}), {"Broadcast": pct, "Digital": pct, "Mail/print": pct, "Consulting/other": pct})
    rule_rows = pd.DataFrame({"If a campaign puts 10%+ of a phase's budget in": single_cross["if"], "Then 10%+ of the later phase in": single_cross["then"], "Share of campaigns": single_cross.support.map(pct),
                              "How often, 2012-2018": single_cross.confidence.map(pct), "Lift": single_cross.lift.map("{:.2f}".format), "Conditional lift": single_cross.conditional_lift.map("{:.2f}".format),
                              "How often, 2020-2024": single_cross.test_confidence.map(pct), "Conditional lift, 2020-2024": single_cross.test_conditional.map("{:.2f}".format)})
    rule_table = number_table(rule_rows, {})
    pass_view = number_table(pd.DataFrame([{"Pass-through": n, **{str(y): f"{pass_change.loc[n, y]:+.4f}" for y in pass_years}, "Average": f"{pass_change.loc[n, pass_years].mean():+.4f}",
                                            "Years better": f"{(pass_change.loc[n, pass_years] < -1e-9).sum()} of {len(pass_years)}"} for n in GRID if n != "1:1"]), {})
    pass_seat_view = number_table(pd.DataFrame([{"Pass-through": n, "Expected Democratic seats": f"{r['mean']:.1f}", "80% range": f"{r['p10']:.0f}-{r['p90']:.0f}", "Chance of a majority": f"{r['p_majority']:.1%}"} for n, r in pass_seats.iterrows()]), {})
    slope_view = number_table(pd.DataFrame([{"Money under $2M": r.version, "Points per point of lean above $2M": f"{r.above:.2f}", "More under $2M": f"{r.difference:+.2f}", "Standard error": f"{r.se:.2f}", "Races": f"{r.n:,}"} for r in slopes.itertuples()]), {})
    error_view = number_table(pd.DataFrame([{"Money": label, "All money: error, points": f"{error_all.sd[label]:.1f}", "Races": f"{error_all.n[label]:,}", "Seen by Sept 1: error, points": f"{error_sept.sd[label]:.1f}", "Races ": f"{error_sept.n[label]:,}"} for label in MONEY_LABELS]), {})
    independent_view = number_table(independent_rows, {"TV": pct, "Digital": pct, "Mail": pct, "Media, unspecified": pct})
    challenger_spent = clustered[clustered.ici == "C"].groupby("cluster").total.median()
    vendor_table = number_table(pd.DataFrame(vendor_rows), {})
    c, kept_check = check["accel"], check["new_voters_within"]
    pass_text = (f"**{LABELS['accel']}** clears it. A placebo that shuffles it among each year's races clears the bar {c['pass_rate']:.0%} of the time, so among "
                 f"{len(LABELS)} tries at least one pass was about {chance['accel']:.0%} likely by luck. Only {c['as_good_within']:.1%} of 400 shuffles match it on log loss with the miss removed and {c['as_good_raw']:.1%} on raw log loss, "
                 f"but with {len(LABELS)} tries a result like the first isn't unusual either. So it's a candidate, not a finding. It isn't obvious what it measures, and its 95% interval still covers zero. "
                 f"Its 2026 forecasts are in [`data/spending_candidate_2026.csv`](data/spending_candidate_2026.csv), written before results, so the election can test it. The only other pass, a second form of retention, is weaker (section 6).\n\n")
    harmful = [LABELS[n] for n in LABELS if intervals[n][0] > 0]
    second_passes = sum(n in passes for n in SECOND)
    first_set = [n for n in LABELS if n not in SECOND]
    gold_d, gold_u = gold["dollars"], gold["uniform"]
    text = f"""# What House campaigns buy, and whether it predicts anything

Generated by `python -m campaign.spending_report`. Every number is computed from the data unless marked otherwise. Covers the nominees' campaigns in {DONE[0]}-{DONE[-1]}: {sum(len(full[c]) for c in DONE):,} payments, ${total / 1e9:.1f}B, from FEC operating expenditures (Schedule B). {LIVE} is used only for the forecast candidate below.

## Summary

- **Spending categories don't improve the forecast beyond the June 30 surplus.** The direct test, adding fundraising, voter-contact and overhead money to a forecast that already knows raised minus spent, makes log loss slightly {'worse' if within['mean']['split'] > 0 else 'better'} ({within['mean']['split']:+.5f}, better in {within.better['split']:.0f} of 5 years). The surplus itself remains the gain ({surplus_gain['mean']:+.5f} against the baseline, better in {surplus_gain['better']:.0f} of 5).
- **The signals people expect mostly fail out of sample.** Discounting receipts by fundraising costs fits well but calls winners worse. Early incumbent polling does signal trouble in the fit, but doesn't help forecasts. Spending more on fundraising goes with doing better, not worse.
- **Candidates do leave TV where it's expensive.** In competitive races in {min(bt)} and {max(bt)}, TV falls from {tv_table.Broadcast.iloc[0]:.0%} of fall budgets in the cheapest markets to {tv_table.Broadcast.iloc[-1]:.0%} in the costliest, while digital rises from {tv_table.Digital.iloc[0]:.0%} to {tv_table.Digital.iloc[-1]:.0%}. Independent spenders' TV share moves {tv_change('All outside groups') * 100:+.0f} points on the same comparison, and DCCC's {tv_change('DCCC') * 100:+.0f} (section 4).
- **The one durable cross-category pattern is polling, then TV.** With size and race type held fixed, a campaign that puts at least 10% of one quarter's budget in polling is about {single_cross.test_conditional.min():.2f}-{single_cross.test_conditional.max():.2f} times as likely to put at least 10% of the next quarter's in TV, in later elections as well as the ones the rule was found in.
- **Campaign types mostly re-measure incumbency.** Three spending profiles are stable, but they're weakly separated and don't help the forecast.
- **A second round of {len(SECOND)} ideas found {second_passes} that clear the bar.** Cash runway, burn rate, debt, channel combinations, purchase order and vendor loyalty all fall short.
- **Donor data doesn't clear the bar either.** {donor_passes} of {len(DONOR_FEATURES)} features built from {gifts / 1e6:.1f} million itemized gifts pass. Donor counts fit strongly, but say little the money doesn't (section 2).
- **Incumbents run better where they kept more of their old voters, but it barely helps a forecast.** Across {slope_all['n']} incumbents in five redraws, one who kept half their district ran about {slope_all['slope'] / 2:.1f} points behind one who kept all of it, and about {slope_close['slope'] / 2:.1f} in races the forecast rated close. The redraws disagree with each other, and the forecast gain is too small to separate from chance (section 6).
- **Treating the generic ballot as a brand survey doesn't improve the forecast.** Shrinking it helps presidential years only through 2020 ({pass_change[pass_years].mean(axis=1)['0.75, presidential years only']:+.4f} on raw log loss, {pass_without.mean(axis=1)['0.75, presidential years only']:+.4f} without 2020). The district's lean does count for more in low-money races, but the error isn't tighter there and forecasts don't improve (section 6).
- **The committees move together, but not visibly in reaction to each other.** The other side's spending next week predicts this week's better than last week's does ({reaction_n['lead'][0]:+.2f} against {reaction_n['lag'][0]:+.2f}), which fits shared plans or news better than a response (section 6).

## 1. The data and its traps

| Trap | What happens | What this does |
|---|---|---|
| Memo rows | A memo row restates part of the payment above it. Counted too, campaigns' totals run 10-18% over what they reported | A payment with memo rows is replaced by them, so a credit card bill's detail is kept and its total counted once |
| Vague purposes | "MEDIA" or "CONSULTING" says little about what was bought | Rules on the text, then each vendor's clearly labeled record, using only payments made so far |
| Hindsight | Fall spending reflects how the race is going | Forecast features use only payments dated through June 30, which the July report disclosed |
| Spending shares sum to 1 | Distances between raw shares mislead, and total spending swamps them | Clusters use centered log-ratios and leave total spending out |
| "Bought X" | Nearly everyone buys a little of everything | Items require 10% of a phase's budget, and rules are compared within budget tercile and race type |

**How well the categories work.** {len(every_row):,} payments through June 30 carry {every_row.purpose.nunique():,} distinct purposes. Claude labeled samples of payments from their purpose and vendor without seeing the classifier's answer. On a first sample, before any tuning, it got 89% right (a count from that check, not recomputed here). After fixing what that showed, a fresh sample gave {gold_d[0]:.0%} of a dollar-weighted sample (n = {gold_d[1]}) and {gold_u[0]:.0%} of a row-weighted one (n = {gold_u[1]}), each within about 8 points. So about {1 - gold_d[0]:.0%}-{1 - gold_u[0]:.0%} of payments are misfiled. Claude also wrote the rules, so a person checking `data/purpose_gold.csv` would be a stronger test. {vague_share:.0%} of dollars stay vague.

{bought_table}

Mail and digital can't be fully split between fundraising and persuasion. "POSTAGE" or "DIGITAL CONSULTING" stays with its channel unless the text or vendor says otherwise.

## 2. Does it improve the forecast?

Each feature joins the forecast with the June 30 surplus already in it, in the same forward backtest as the main report, {BACKTEST[0]}-{BACKTEST[-1]}. Changes below are against that forecast, so negative is better. The bar was fixed before looking: better with each year's polling miss removed in at least 4 of 5 years, better on raw log loss on average, and not worse in the races the baseline put within 10 points.

{signal_table}

The interval resamples state-years for the log loss with the miss removed. "Close races" also has the miss removed.

**The direct test fails.** Split into categories, the terms add little: fundraising {fits['split']['coef']['fund_odds']:+.2f} (t = {fits['split']['coef']['fund_odds'] / fits['split']['se']['fund_odds']:+.1f}), voter contact {fits['split']['coef']['contact_odds']:+.2f} (t = {fits['split']['coef']['contact_odds'] / fits['split']['se']['contact_odds']:+.1f}), overhead {fits['split']['coef']['admin_odds']:+.2f} (t = {fits['split']['coef']['admin_odds'] / fits['split']['se']['admin_odds']:+.1f}). What a campaign spent matters. What it spent it on adds little that a forecaster can use.

{pass_text}**The first round.**
- **Money raised less fundraising costs** fits strongly and gets margins closer, but calls winners worse, the same pattern as symmetric money.
- **Early incumbent polling** runs the expected way: incumbents who poll early run behind. It's visible in the fit but doesn't help out of sample.
- **Fundraising share** runs opposite to the "churn" story: campaigns that spend more of their budget on fundraising do slightly better than the money alone says. There's no sign that heavy fundraising is a warning sign.
- **Vendors with a competitive-race record** go with doing worse, and this feature is the one clearly harmful out of sample. Sharing a vendor doesn't mean much, and the sign isn't one to interpret.

**The second round**, tried after the first with the same bar: {', '.join(LABELS[n].lower().replace('tv ', 'TV ') for n in SECOND)}. {second_passes} of {len(SECOND)} clear it. Some fit well: spent as a share of raised has a t of {fits['burn']['coef']['burn'] / fits['burn']['se']['burn']:+.1f}, and keeping last cycle's vendors {fits['vendor_retention']['coef']['vendor_retention'] / fits['vendor_retention']['se']['vendor_retention']:+.1f}. Neither helps out of sample.

**The third round: who gave.** These use FEC's individual contribution files: {gifts:,} itemized gifts through June 30, from donors FEC's rules generally itemize, past $200 in a cycle. They account for a median of {min(donor_cover):.0%}-{max(donor_cover):.0%} of what each campaign's July report calls itemized, by cycle. A donor is a name and ZIP code, since FEC has no donor ID. The five features were fixed before scoring: {', '.join(LABELS[n].lower() for n in DONOR_FEATURES)}. {donor_passes} of {len(DONOR_FEATURES)} clear the bar.
- **Donor counts** fit strongly (t = {fits['donor_odds']['coef']['donor_odds'] / fits['donor_odds']['se']['donor_odds']:+.1f}), and in-state donors more so ({fits['local_odds']['coef']['local_odds'] / fits['local_odds']['se']['local_odds']:+.1f}), but make forecasts worse out of sample. They're {donor_corr:.2f} correlated with the money term, so they mostly restate it. In-state means the state, since these files have no district and span several maps.
- **The share from last cycle's donors** came closest: better in {within.better['repeat_share']:.0f} of 5 years with the miss removed, on raw log loss ({raw['mean']['repeat_share']:+.5f}) and in close races ({close['mean']['repeat_share']:+.4f}). But most of its gain is 2022 and 2024 ({within.loc['repeat_share', 2022]:+.4f} and {within.loc['repeat_share', 2024]:+.4f}), its interval covers zero, and it runs negative: the more of a campaign's money comes from last cycle's donors, the worse it does than the rest of the forecast says. It misses the bar by a year, so it isn't a finding.
- **Concentration among the top donors** does nothing.

Across all three rounds, those whose interval is entirely on the harmful side: {', '.join(h.lower() for h in harmful)}.

## 3. Campaign types

K-means on pre-June 30 spending shares of {len(table):,} campaigns with at least ${MIN_BUDGET / 1e3:.0f}K spent.

| k | {' | '.join(str(k) for k in ks.index)} |
|---|{'---|' * len(ks)}
| Silhouette | {' | '.join(f'{v:.2f}' for v in ks.silhouette)} |
| Stability on resamples | {' | '.join(f'{v:.2f}' for v in ks.stability)} |

Three clusters repeat on resamples, but silhouettes this low mean they're regions of a continuum rather than distinct kinds of campaign.

{profile_table}

{cluster_table}

Most of this is incumbency: the maintenance type is nearly all incumbents. Challengers already on TV by June win more often than other challengers, but they've also spent far more by then: a median of ${challenger_spent[order[0]] / 1e3:,.0f}K against ${challenger_spent[order[1]] / 1e3:,.0f}K and ${challenger_spent[order[2]] / 1e3:,.0f}K. As forecast features, with clusters fit only on earlier years, they make log loss with the miss removed {forecast_clusters.within.mean():+.5f} (better in {int((forecast_clusters.within < 0).sum())} of 5 years).

## 4. Do candidates respond to TV prices?

Candidates, in races the forecast had within 15 points in {min(bt)} and {max(bt)}, spending after June 30:

{tv_view}

Going from a market where a TV buy reaches twice the district to one that reaches twenty times, with budget, party, incumbency and year held fixed: TV moves {tv_slopes.tenfold['Broadcast']:+.1%} of the budget (standard error {tv_slopes.se['Broadcast']:.1%}), digital {tv_slopes.tenfold['Digital']:+.1%} ({tv_slopes.se['Digital']:.1%}), mail {tv_slopes.tenfold['Mail/print']:+.1%} ({tv_slopes.se['Mail/print']:.1%}). Vague spending moves {tv_slopes.tenfold['Consulting/other']:+.1%} ({tv_slopes.se['Consulting/other']:.1%}), so {'it could account for part of the drop in TV' if tv_slopes.tenfold['Consulting/other'] > 2 * tv_slopes.se['Consulting/other'] else 'mislabeled TV buys in expensive markets do not explain the drop'}. This covers two elections.

Independent spenders in the same races and the same markets:

{independent_view}

So DCCC's TV share moves {tv_change('DCCC') * 100:+.0f} points from the cheapest markets to the costliest, House Majority PAC's {tv_change('House Majority PAC') * 100:+.0f} and all outside groups' {tv_change('All outside groups') * 100:+.0f}, against {(tv_table.Broadcast.iloc[-1] - tv_table.Broadcast.iloc[0]) * 100:+.0f} for candidates. These two tables have no controls, and independent spenders buy fewer channels, so read the contrast as suggestive. Candidates may also be reacting to more than price. They generally pay lower ad rates than outside groups (from general knowledge, not this data), which should make them less sensitive to price, and big metro markets could differ in their electorates and vendors, which this can't separate.

## 5. What goes with what

"Spent anything on X" is close to useless, as expected: the {len(naive_rules)} rules found that way average a lift of {naive_rules.lift.mean():.2f}. With an item meaning 10% of a phase's budget, and rules running only forward in time, {TRAIN_THROUGH} and earlier give {len(found)} rules with at least 5% support and 50% confidence. Their raw lift averages {found.lift.mean():.2f}, and {found.conditional_lift.mean():.2f} once compared within budget tercile, incumbency and safe or contested race, so about {(found.lift.mean() - found.conditional_lift.mean()) / (found.lift.mean() - 1):.0%} of the apparent association was campaign size and type.

Most rules that survive say a campaign keeps doing what it was doing: {int((~strong.cross).sum())} of {len(strong)} with a conditional lift of at least 1.2 stay in the same category. Across categories, only polling leading to TV holds up:

{rule_table}

These are rules without a second condition. Adding one (payroll, fundraising) gives near-copies of them. Polling doesn't have to cause the TV buy: a campaign planning TV polls first to decide what to say.

**The vendor market.** In {DONE[-1]}, {both_parties.mean():.1%} of vendors served both parties, with {by_party[both_parties].sum().sum() / by_party.sum().sum():.1%} of the dollars, so the market is almost entirely split by party. Vendors here include individuals paid directly, and names are matched as filed, so a firm with two spellings counts twice.

{vendor_table}

No category is concentrated, so anyone selling to campaigns is competing with many small firms, and on one side of the aisle.

## 6. The other ideas

**Do the committees respond to each other?** Week by week from Aug 1 through the election, {reaction_n['races']} races in {DONE[0]}-{DONE[-1]}, each committee's money against the other's, with race and week effects (inverse hyperbolic sines, so these read roughly as elasticities):

{reaction_view}

They do move together, but the links are small, and next week's money predicts this week's better than last week's does, so there's no sign of one side reacting to the other within weeks. That fits both following the same news or plans, though ads are booked ahead, and dates here are when they ran, so weekly data can't fully settle it. Among the {movers['n']} races both entered, DCCC went first in {movers['dccc_first']:.0%} and NRCC in {1 - movers['dccc_first'] - movers['tie']:.0%}, so neither leads. Their first dates fall within 14 days of each other in {movers['within_14']:.0%}, against {movers['shuffled_within_14']:.0%} with dates shuffled among the cycle's races (95% of shuffles at or below {movers['shuffled_high']:.0%}). They enter around the same time, more than chance but not by much. None of this tests the main report's two-player game, which is about where each side places a whole budget.

**Do committees use shared TV markets?** A buy reaches every race in its market, and the main report's TV-cost index already counts that. The question here is whether committees act on it. Among {spill_committee['n']} races the forecast had within 10 points in {min(bt)} and {max(bt)}, {spill_committee['with_other']:.0%} shared a main market with another. Each additional close race in the market changes log committee money by {spill_committee['n_other']:+.2f} (standard error {spill_committee['se']:.2f}) and outside money by {spill_outside['n_other']:+.2f} ({spill_outside['se']:.2f}), and reach shared among those races explains committee money no better than plain reach (R-squared {spill_committee['r2_shared']:.2f} against {spill_committee['r2_reach']:.2f}). There's no visible discount, though this is a small sample with wide errors.

**Does keeping more of an old district help?** Measured block by block in people from the 2010 and 2020 Censuses, for every redraw since 2012 but North Carolina's for 2020, which Census has no file for, so its members are left out of 2022 as well. Florida's 2016 races aren't in this model at all, because its data has no 2012 presidential vote on the new lines, so 2016 here is North Carolina and Virginia.

{redraw_view}

Comparing incumbents only within a redraw, each additional 100% kept goes with {slope_all['slope']:+.1f} points against the forecast (standard error {slope_all['se']:.1f}, clustered by state, n = {slope_all['n']}), {slope_state['slope']:+.1f} ({slope_state['se']:.1f}) within the same state and year, and {slope_close['slope']:+.1f} ({slope_close['se']:.1f}, n = {slope_close['n']}) in races the forecast put within 15 points. Leaving out any one state, the first runs {min(without_state):+.1f} to {max(without_state):+.1f}. 2012 has no earlier year to fit on, so its forecast comes from later years; without it the estimate is {slope_forward['slope']:+.1f} ({slope_forward['se']:.1f}).

{retained_view}

That's the direction you'd expect if voters who never had this member give less of the usual edge. If they gave none, the slope would be about the forecast's incumbency edge, {incumbency_2022:.1f} points in 2022 (half the gap it puts between seats held by each party). The estimate across all redraws is under half of that. Members whose new seats were drawn to be hard could also run behind for other reasons, which comparing within a state only partly handles.

It doesn't do much for a forecast. The redraws disagree: 2012's estimate is near zero until compared within a state, 2024's has the other sign, and 2016's and 2018's rest on a few members in two states and one. Carrying the slope forward from earlier redraws lowers the error in {len(carry_better)} of {len(carry)} ({', '.join(str(y) for y in carry_better)}), by at most {max(before - after for before, after in carry.values()):.2f} points. Added to the forecast, the share of a district new to its incumbent fails the bar with the wrong sign (t = {fits['new_voters']['coef']['new_voters'] / fits['new_voters']['se']['new_voters']:+.1f}): 2012's strong incumbents all have new voters, so it also picks up the era. A second form, tried after that, measures the share against that redraw's average. It technically clears the bar, at {within['mean']['new_voters_within']:+.5f} with the miss removed, but 2020 had no redraw to test and 2016's and 2018's changes round to zero, its fit is weak (t = {fits['new_voters_within']['coef']['new_voters_within'] / fits['new_voters_within']['se']['new_voters_within']:+.1f}), and {kept_check['as_good_within']:.0%} and {kept_check['as_good_raw']:.0%} of shuffles match it on log loss with the miss removed and raw. So it's a weaker candidate than April-June spending. Its 2026 forecasts are in [`data/retention_candidate_2026.csv`](data/retention_candidate_2026.csv), written before results.

**2026's redraw can be measured too.** Census published the lines in `cd120.zip` (dated Oct 7, checked). Its redrawn states {'match' if redrawn_now == model_redrawn else 'differ from'} the ones the forecast treats as redrawn: {', '.join(redrawn_now)}. {len(kept_now)} incumbents run on them, with a median of {kept_now.retained.median():.0%} of the district already theirs. {len(low_now)} in races the forecast puts within 15 points kept under 60%: {', '.join(low_now.district_id)}. Neither frozen forecast adjusts for this.

**Google ads.** Google's public ad data, matched through the FEC IDs advertisers list, covers nominees in {google_either[2020]}, {google_either[2022]} and {google_either[2024]} of 434 races before Sept 1 in 2020, 2022 and 2024, for ${google_money[2020] / 1e6:.1f}M, ${google_money[2022] / 1e6:.1f}M and ${google_money[2024] / 1e6:.1f}M. 2018 has almost nothing left. Comparing the two sides as odds fits weakly (t = {google_fit['coef']['google_odds'] / google_fit['se']['google_odds']:+.1f}) and makes forecasts worse in both years it can be tested: {google_within[2022]:+.5f} and {google_within[2024]:+.5f} with the miss removed, {google_raw[2022]:+.5f} and {google_raw[2024]:+.5f} raw. Only advertisers listing an FEC ID match, about 40-50% of nominees, so this understates what campaigns spent, and it's one platform.

**Is the generic ballot a survey of party brand?** If most voters follow the label, polls that swing far should overstate how far voters move, so the forecast's 1:1 pass-through should be too much. Nationally, here's how much the House vote moved per point of polling, 2002-2024:

| Polls | All {pass_national.n['gb', 'All']} elections | {pass_national.n['gb', 'Presidential']} presidential | {pass_national.n['gb', 'Midterm']} midterm |
|---|---|---|---|
| Final | {pass_national.k['gb', 'All']:.2f} ({pass_national.se['gb', 'All']:.2f}) | {pass_national.k['gb', 'Presidential']:.2f} ({pass_national.se['gb', 'Presidential']:.2f}) | {pass_national.k['gb', 'Midterm']:.2f} ({pass_national.se['gb', 'Midterm']:.2f}) |
| Sept 1 | {pass_national.k['gb_sept', 'All']:.2f} ({pass_national.se['gb_sept', 'All']:.2f}) | {pass_national.k['gb_sept', 'Presidential']:.2f} ({pass_national.se['gb_sept', 'Presidential']:.2f}) | {pass_national.k['gb_sept', 'Midterm']:.2f} ({pass_national.se['gb_sept', 'Midterm']:.2f}) |

None is distinguishable from 1, and September polls show no overreaction. Their problem is a fairly steady Democratic lean, which the Sept 1 forecast's fall drift already corrects. By race, in the forward backtest, against 1:1:

{pass_view}

These are raw, because taking out the year's miss would erase the effect being tested. The presidential versions look like wins, but they're one election. {', '.join(f"{c} polled D+{r.gb:.1f} and voted D+{r.house_vote:.1f}" for c, r in big_presidential.iterrows())}: the only two large presidential polls disagree, and polls in the other years were within about a point of even, so shrinking them changes little. Leaving 2020 out of both training and scoring, the 0.75 version's change falls to {pass_without.mean(axis=1)['0.75, presidential years only']:+.4f}. The idea was also prompted by 2020, so 2020 can't confirm it. 2026 is a midterm, where the evidence is {pass_national.k['gb', 'Midterm']:.2f} ({pass_national.se['gb', 'Midterm']:.2f}) on six elections. For scale only, here's what each pass-through does to the D+{races[LIVE].gb.iloc[0]:.1f} forecast, written before results. Neither frozen forecast uses any of these but 1:1:

{pass_seat_view}

**Does a district's lean count for more where little is spent?** It does:

{slope_view}

A slope of 2 would be pure brand voting, because PVI is in vote-share points and a margin moves twice as fast, so low-money open seats come closest. But low-money races also tend to have an incumbent and a token challenger, so this can't separate voters defaulting to the label from races with no real opposition. There are {min(low_counts.values())}-{max(low_counts.values())} races a cycle under $2M of all money. The error isn't tighter in them:

{error_view}

Setting each race's error size by money is better in {int((error_size.change < 0).sum())} of {len(error_size)} years ({error_size.change.mean():+.5f} raw). As forecast features, the lean under $2M and the lean by log money both miss the bar: {within['mean']['pvi_low']:+.5f} and {within['mean']['pvi_log']:+.5f} with the miss removed, better in {within.better['pvi_low']:.0f} and {within.better['pvi_log']:.0f} of 5, and {raw['mean']['pvi_low']:+.5f} and {raw['mean']['pvi_log']:+.5f} raw. Margins get closer ({margin_err['mean']['pvi_low']:+.2f} and {margin_err['mean']['pvi_log']:+.2f} points), but winners aren't called better, because these seats are rarely close. How much voters know and who answers polls take survey data, which this doesn't have.

**Not run**, because of what they need (access and format details are from general knowledge, not checked here unless noted):

| Idea | What it needs |
|---|---|
| Meta ad spending | An identity-verified Ad Library account and a developer token. Meta's public script repository (facebookresearch/Ad-Library-API-Script-Repository, checked) wraps the API and still needs the token. It reports ranges, by state |
| FCC political files | The API responds (checked), but filings are mostly scanned PDFs |
| TV spot data | Wesleyan Media Project's data, open to academics for a fee per dataset, or a commercial AdImpact license. Requested on 2026-11-03, not yet delivered |

## 7. Limits

- **Payment dates, not delivery.** A buy paid for in June can run in July, and a vendor's invoice can lag. FCC files or ad archives would show what ran and when.
- **Vendors obscure sub-vendors.** A payment to a media firm says nothing about whether it became TV, digital or fees, and FEC doesn't require more.
- **Purposes are the filer's words.** About {1 - gold_d[0]:.0%}-{1 - gold_u[0]:.0%} of payments are misfiled on the sample above, and the labels are Claude's.
- **{len(LABELS)} forecast features is a lot for five elections.** The placebo addresses this for those that passed. A sixth year is the real test.
- **Candidates only.** Party and outside spending are in the main report. A campaign's money isn't all the money in its race.
- **Later amendments** are in the bulk file, a small leak for a Sept 1 forecast that the July totals don't have.
"""
    (ROOT / "SPENDING.md").write_text(text)
    print("wrote SPENDING.md")
    print(within[["mean", "better"]].round(5))
    print("passes", passes, check, chance)


if __name__ == "__main__":
    spending_report()
