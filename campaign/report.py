"""Write REPORT.md and its figures from the pipeline's results."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .backtest import HALF_SATURATIONS, SCALES
from .ballot_timing import HALF_LIFE, HALF_LIVES, SPREAD, WINDOW
from .break_even import MAX_SCALE
from .channel_mix import BAND_LABELS
from .compare_maps import compare_maps
from .fit_baseline import fit_baseline
from .household_index import STRESS
from .dated_forecast import CHECKPOINTS
from .load_expenditures import load_expenditures
from .media_cost import GAMMAS
from .national_test import PASS_P
from .plots.plot_allocation import plot_allocation
from .plots.plot_battleground import plot_battleground
from .plots.plot_calibration import plot_calibration
from .plots.plot_channel_mix import plot_channel_mix
from .plots.plot_equilibrium import plot_equilibrium
from .plots.plot_household_stress import plot_household_stress
from .plots.plot_map_changes import plot_map_changes
from .plots.plot_poll_lean import plot_poll_lean
from .plots.plot_pvi_vs_margin import plot_pvi_vs_margin
from .plots.plot_safe_seat_share import plot_safe_seat_share
from .plots.plot_seat_distribution import plot_seat_distribution
from .plots.plot_seats_by_gb import plot_seats_by_gb
from .plots.plot_spending_effect import plot_spending_effect
from .plots.plot_through_fall import plot_through_fall
from .spending_effect import HALF_SATURATION, SCALE_PTS, spending_effect

ROOT = Path(__file__).resolve().parents[1]


def report(races, backtests, fc, games, media, mixes, national):
    """`races`: cycle -> build_races(cycle); `backtests`: backtest() results
    in cycle order; `fc`: forecast(); `games`: cycle -> solve_game();
    `media`: media_cost() results; `mixes`: channel_mix() results;
    `national`: national_test() results plus the national_table() as 'table'."""
    fig = ROOT / "figures"
    fig.mkdir(exist_ok=True)
    last = max(c for c in races if c < fc["cycle"])
    plot_pvi_vs_margin(races[last], fc["model"], fig / "1_pvi_vs_margin.png")
    plot_spending_effect(fig / "2_spending_effect.png")
    plot_calibration(backtests, fig / "3_calibration.png")
    plot_allocation(backtests, fig / "4_allocation.png")
    plot_equilibrium(backtests, games, fig / "5_equilibrium.png")
    plot_channel_mix(mixes, fig / "6_channel_mix.png")
    plot_map_changes(fig / "7_map_changes.png")
    plot_seats_by_gb(fc, fig / "8_seats_by_gb.png")
    plot_seat_distribution(fc, fig / "9_seat_distribution.png")
    plot_battleground(fc, fig / "10_battleground.png")
    plot_through_fall(backtests, fig / "11_through_the_fall.png")
    plot_safe_seat_share(backtests, fig / "12_safe_seat_share.png")
    plot_household_stress(national["table"], fig / "13_household_stress.png")
    plot_poll_lean(national["table"], fig / "14_poll_lean.png")

    m, c = fc["model"], fc["model"]["coef"]
    data_rows = "\n".join(
        f"| {cy} | {len(r)} | {int((r.margin.abs() < 80).sum()) or '-'} | {r.gb.iloc[0]:+.1f} | "
        f"${r.dccc.sum() / 1e6:.0f}M | ${r.nrcc.sum() / 1e6:.0f}M | ${r.outside_d.sum() / 1e6:.0f}M | ${r.outside_r.sum() / 1e6:.0f}M |"
        for cy, r in races.items())
    bt_rows = "\n".join(
        f"| {b['cycle']} | {b['brier']:.3f} | {b['races'].p[b['races'].margin.notna()].sum():.0f} | "
        f"{int((b['races'].margin > 0).sum())} | ${b['budget'] / 1e6:.0f}M | {b['gain']:+.2f} | "
        f"{min(b['grid'].values()):+.2f} to {max(b['grid'].values()):+.2f} |" for b in backtests)
    grid = "\n".join(
        f"| {s:g} pts | " + " | ".join(f"{np.mean([b['grid'][(s, h)] for b in backtests]):+.2f}" for h in HALF_SATURATIONS) + " |"
        for s in SCALES)
    pooled = pd.concat([b["races"] for b in backtests])
    in_play = pooled.p.between(0.2, 0.8)
    share_dccc, share_model = pooled.dccc[in_play].sum() / pooled.dccc.sum(), pooled.model_dccc[in_play].sum() / pooled.model_dccc.sum()
    safe_dccc = pooled.dccc[pooled.p > 0.8].sum() / pooled.dccc.sum()
    seats = fc["seats_by_gb"]
    gb = fc["gb"]
    maps = compare_maps(new_lines=fc["cycle"])
    redrawn = maps[maps.changed].district_id.str[:2]
    harris_old, harris_new = (maps[f"{k}_share"][maps.district_id.str[:2].isin(redrawn)] > 0.5 for k in ("old", "new"))
    close = fc["races"][fc["races"].p.between(0.2, 0.8)]
    years = f"{backtests[0]['cycle']}-{backtests[-1]['cycle']}"
    mean_gain = np.mean([b["gain"] for b in backtests])

    seat_rows, covered = [], 0
    for b in backtests:
        s = b["seats"]
        inside = s["p10"] <= s["actual"] <= s["p90"]
        covered += inside
        p_maj = ">99%" if s["p_majority"] > 0.995 else "<1%" if s["p_majority"] < 0.005 else f"{s['p_majority']:.0%}"
        seat_rows.append(f"| {b['cycle']} | {s['mean']:.0f} | {s['p10']}-{s['p90']} | {s['actual']}{'' if inside else ' (outside)'} | "
                         f"{s['need']} | {p_maj} | {'Democrats' if s['actual'] >= s['need'] else 'Republicans'} |")
    protect_rows = []
    for b in backtests:
        r = b["races"]
        safe = r[(r.p > 0.8) & (r.dccc > 0)]
        lost = safe[safe.margin < 0].district_id.tolist()
        protect_rows.append(f"| {b['cycle']} | {safe.dccc.sum() / r.dccc.sum():.0%} | "
                            f"{r.model_dccc[r.p > 0.8].sum() / r.model_dccc.sum():.0%} | ${safe.dccc.sum() / 1e6:.0f}M | "
                            f"{safe.dccc[safe.status == 'D incumbent'].sum() / max(safe.dccc.sum(), 1):.0%} | "
                            f"{len(lost)}{': ' + ', '.join(lost) if lost else ''} |")
    s26 = fc["seats"]
    nat_lo, nat_hi = (m["sigma_national"] * np.sqrt(m["national_df"] / chi2.ppf(q, m["national_df"])) for q in (0.95, 0.05))
    known = [b["seats_known_sigma"] for b in backtests]
    covered_known = sum(k["p10"] <= k["actual"] <= k["p90"] for k in known)
    width, width_known = (np.mean([s["p90"] - s["p10"] for s in group]) for group in ([b["seats"] for b in backtests], known))
    changed = set(maps[maps.changed].district_id)
    moved = fc["races"][fc["races"].district_id.isin(changed) & (fc["races"].status != "Open")]
    in_play_moved = moved[moved.p.between(0.2, 0.8) | moved.p_as_open.between(0.2, 0.8)].sort_values("p")
    bound_rows = "\n".join(f"| {r.district_id} | {r.status} | {r.pvi:+.1f} | {r.p:.0%} | {r.p_as_open:.0%} |"
                           for r in in_play_moved.itertuples())
    bound_swing = (in_play_moved.p_as_open - in_play_moved.p).abs()
    bound_net = (moved.p_as_open - moved.p).sum()
    calls_right = sum((b["seats"]["p_majority"] > 0.5) == (b["seats"]["actual"] >= b["seats"]["need"]) for b in backtests)
    missed = [b for b in backtests if not b["seats"]["p10"] <= b["seats"]["actual"] <= b["seats"]["p90"]]
    missed_text = "; ".join(f"{b['cycle']}, when Democrats won {b['seats']['actual']} against a range of "
                            f"{b['seats']['p10']}-{b['seats']['p90']}" for b in missed) or "none"
    r20 = next(b["races"] for b in backtests if b["cycle"] == 2020)
    lost_2020 = int(((r20.p > 0.8) & (r20.dccc > 0) & (r20.margin < 0)).sum())
    safe_inc = pooled.dccc[(pooled.p > 0.8) & (pooled.status == "D incumbent")].sum() / pooled.dccc[pooled.p > 0.8].sum()

    # the forecast a Sept 1 decision could make (backtest's "sept")
    sept_rows = "\n".join(
        f"| {b['cycle']} | {b['sept']['gb']:+.1f} | {d['mean']:+.1f} (spread {d['sd']:.1f}, {d['n']} elections) | "
        f"{b['sept']['gb'] + d['mean']:+.1f} | {r.gb.iloc[0]:+.1f} | "
        f"{b['scores']['with']['log_loss']:.3f} / {b['scores']['sept']['log_loss']:.3f} | "
        f"{b['seats']['p10']}-{b['seats']['p90']} / {b['sept']['seats']['p10']}-{b['sept']['seats']['p90']} | {b['seats']['actual']} |"
        for b in backtests for r, d in [(b["races"], b["sept"]["drift"])])
    ll_final, ll_sept = (np.mean([b["scores"][k]["log_loss"] for b in backtests]) for k in ("with", "sept"))
    sept_better = [str(b["cycle"]) for b in backtests if b["scores"]["sept"]["log_loss"] < b["scores"]["with"]["log_loss"]]
    covered_sept = sum(b["sept"]["seats"]["p10"] <= b["seats"]["actual"] <= b["sept"]["seats"]["p90"] for b in backtests)
    drift_last = backtests[-1]["sept"]["drift"]
    safe_sept = pooled.dccc[pooled.p_sept > 0.8].sum() / pooled.dccc.sum()
    multiplier = lambda k: f"{k:.1f}x" if np.isfinite(k) else "no DCCC money"
    def safe_when_spent(b):      # share of DCCC's dollars that went to races 80%+ when they were spent
        t = b["when_spent"]["tranches"]
        return t.amount[t.p > 0.8].sum() / t.amount.sum()
    judged_rows = "\n".join(
        f"| {b['cycle']} | {r.dccc[r.p > 0.8].sum() / r.dccc.sum():.0%} / {r.dccc[r.p_sept > 0.8].sum() / r.dccc.sum():.0%} / "
        f"{safe_when_spent(b):.0%} | {r.model_dccc_sept[r.p_sept > 0.8].sum() / max(r.model_dccc_sept.sum(), 1):.0%} | "
        f"{b['gain']:+.2f} / {b['sept']['gain']:+.2f} | {multiplier(b['belief'][HALF_SATURATION].multiplier['80%+'])} / "
        f"{multiplier(b['sept']['belief'][HALF_SATURATION].multiplier['80%+'])} / "
        f"{multiplier(b['when_spent']['belief'][HALF_SATURATION].multiplier['80%+'])} |"
        for b in backtests for r in [b["races"]])
    sept_k = [b["sept"]["belief"][HALF_SATURATION].multiplier["80%+"] for b in backtests]
    spent_k = [b["when_spent"]["belief"][HALF_SATURATION].multiplier["80%+"] for b in backtests]
    budgets = np.array([b["races"].dccc.sum() for b in backtests])
    safe_spent = float(np.dot([safe_when_spent(b) for b in backtests], budgets) / budgets.sum())

    # how the forecast changes through the fall, checkpoint by checkpoint
    tranche_dollars = pd.concat([b["when_spent"]["tranches"] for b in backtests]).groupby("checkpoint").amount.sum()
    label = lambda md: f"{pd.Timestamp(f'2000-{md}'):%b} {int(md[3:])}"
    ll_at = {d: np.mean([b["scores"][d]["log_loss"] for b in backtests]) for d in CHECKPOINTS}
    term_rows = "\n".join(
        f"| {label(d)} | {backtests[-1]['when_spent']['drift'][d]['mean']:+.1f} | {backtests[-1]['when_spent']['drift'][d]['sd']:.1f} | "
        f"{ll_at[d]:.3f} | {tranche_dollars.get(d, 0) / tranche_dollars.sum():.0%} |" for d in CHECKPOINTS)
    worst = max(((b["scores"][d]["log_loss"] - b["scores"]["with"]["log_loss"], b["cycle"], d) for b in backtests for d in CHECKPOINTS))
    worst_b = next(b for b in backtests if b["cycle"] == worst[1])
    dated_ballot = pd.read_csv(ROOT / "data" / "generic_ballot_dated.csv", dtype={"date": str}).set_index(["cycle", "date"]).generic_ballot
    shifted = [b for b in backtests if (b["races"].dccc[b["races"].p > 0.8].sum() - b["races"].dccc[b["races"].p_sept > 0.8].sum())
               / b["races"].dccc.sum() > 0.05]
    years_shifted = [str(b["cycle"]) for b in shifted]
    shifted_text = (", ".join(years_shifted[:-1]) + " and " + years_shifted[-1]) if len(years_shifted) > 1 else "".join(years_shifted)
    shifted_why = all(b["races"].gb.iloc[0] > b["sept"]["gb"] + b["sept"]["drift"]["mean"] for b in shifted)
    timing = pd.concat([load_expenditures(b["cycle"]).query("bucket == 'dccc'").assign(cycle=b["cycle"]) for b in backtests])
    by_date = lambda md: pd.to_datetime(timing.cycle.astype(str) + md)
    early_share = timing.amount[timing.date <= by_date("-09-30")].sum() / timing.amount.sum()
    late_share = timing.amount[timing.date > by_date("-10-15")].sum() / timing.amount.sum()

    # 2020 with and without the incumbent history (first-term flags, last cycle's overperformance)
    defended_lost = {col: r20[(r20[col] > 0.8) & (r20.dccc > 0) & (r20.margin < 0)] for col in ("p", "p_without")}
    lost_with, lost_without = (set(defended_lost[c].district_id) for c in ("p", "p_without"))
    cleared, still, added = sorted(lost_without - lost_with), sorted(lost_with & lost_without), sorted(lost_with - lost_without)
    fresh_lost = int(defended_lost["p_without"].fresh_d.sum())
    safe_20 = {col: r20.dccc[r20[col] > 0.8].sum() / r20.dccc.sum() for col in ("p", "p_without")}
    c20 = r20[r20.margin.abs() < 80]
    behind = {col: c20.margin - c20[col] for col in ("mu", "mu_without")}
    fresh20, vet20 = c20.fresh_d == 1, (c20.status == "D incumbent") & (c20.fresh_d == 0)

    def upsets(b, col):  # an 80%+ favorite that lost, either side
        r = b["races"][b["races"].margin.notna()]
        return int((((r[col] > 0.8) & (r.margin < 0)) | ((r[col] < 0.2) & (r.margin > 0))).sum())
    history_rows = "\n".join(
        f"| {b['cycle']} | {b['scores']['without']['brier']:.3f} / {b['scores']['with']['brier']:.3f} | "
        f"{b['scores']['without']['log_loss']:.3f} / {b['scores']['with']['log_loss']:.3f} | "
        f"{upsets(b, 'p_without')} / {upsets(b, 'p')} |" for b in backtests)
    ll_change = {b["cycle"]: b["scores"]["with"]["log_loss"] - b["scores"]["without"]["log_loss"] for b in backtests}
    helped = sorted(ll_change, key=ll_change.get)[:2]
    hurt = [c for c in sorted(ll_change) if ll_change[c] > 0.001]
    ll_mean = {k: np.mean([b["scores"][k]["log_loss"] for b in backtests]) for k in ("without", "with", "hindsight")}

    # the same forecast with each side's money over the whole election (hindsight) instead of what Sept 1 could see
    money_rows = "\n".join(
        f"| {b['cycle']} | {b['scores']['hindsight']['brier']:.3f} / {b['scores']['with']['brier']:.3f} | "
        f"{b['scores']['hindsight']['log_loss']:.3f} / {b['scores']['with']['log_loss']:.3f} | "
        f"{upsets(b, 'p_hindsight')} / {upsets(b, 'p')} |" for b in backtests)
    done = pd.concat([r for c, r in races.items() if c <= last])
    seen_outside = done[["outside_aug_d", "outside_aug_r"]].to_numpy().sum() / done[["outside_d", "outside_r"]].to_numpy().sum()
    raised_june = done[["raised_jun_d", "raised_jun_r"]].to_numpy().sum() / done[["disbursements_d", "disbursements_r"]].to_numpy().sum()
    r26 = fc["races"]
    listed = ROOT / "data" / f"nominees_{fc['cycle']}.csv"
    hand_checked = sorted(pd.read_csv(listed).district_id.unique()) if listed.exists() else []
    no_july = r26[((r26.raised_jun_d == 0) & (r26.disbursements_d > 250_000))
                  | ((r26.raised_jun_r == 0) & (r26.disbursements_r > 250_000))]
    no_july_close = no_july[no_july.p.between(0.1, 0.9)]
    named = [f"{d} ({p:.0%} Democratic)" for d, p in zip(no_july_close.district_id, no_july_close.p)]
    misjudged = ", ".join(named[:-1]) + (" and " if len(named) > 1 else "") + (named[-1] if named else "")
    close_sept = pooled[pooled.margin.notna() & pooled.p.between(0.05, 0.95)]
    won_close = (close_sept.margin > 0).astype(float)
    ll_close = {col: float(-np.mean(won_close * np.log(close_sept[col].clip(1e-6, 1 - 1e-6))
                                    + (1 - won_close) * np.log(1 - close_sept[col].clip(1e-6, 1 - 1e-6))))
                for col in ("p_hindsight", "p")}
    first = min(races)
    early, late = (fit_baseline(pd.concat([r for cy, r in races.items() if lo <= cy <= hi]))["coef"]
                   for lo, hi in ((first, 2016), (2018, last)))

    def first_million(b):  # what DCCC's first $1M in a race would have to move it, at each break-even
        return [float(spending_effect(1e6, s, h)) for h, s in b["break_even"].items() if np.isfinite(s)]
    be_rows = "\n".join(
        f"| {b['cycle']} | " + " / ".join(f"{s:.1f}" if np.isfinite(s) else f"over {MAX_SCALE:g}" for s in b["break_even"].values())
        + f" | {b['break_even'][HALF_SATURATION] / SCALE_PTS:.0f}x | {min(first_million(b)):.1f}-{max(first_million(b)):.1f} pts |"
        for b in backtests)
    be_default = [b["break_even"][HALF_SATURATION] for b in backtests]
    first_needed = [v for b in backtests for v in first_million(b)]
    first_default = float(spending_effect(1e6))
    belief_rows = []
    for b in backtests:
        d, safe_k = b["belief"][HALF_SATURATION], [b["belief"][h].multiplier["80%+"] for h in HALF_SATURATIONS]
        belief_rows.append(  # a band where DCCC spent nothing has no marginal cost to read back
            f"| {b['cycle']} | " + " / ".join(f"${1 / v:.0f}M" if d.dollars[band] > 0 else "none spent"
                                             for band, v in d.per_million.items()) + " | "
            + " | ".join(f"{d.multiplier[band]:.1f}x on ${d.dollars[band] / 1e6:.1f}M" if d.dollars[band] > 0
                         else "no DCCC money" for band in ("under 20%", "80%+"))
            + f" | {min(safe_k):.1f}-{max(safe_k):.1f}x |")
    safe_k = {b["cycle"]: b["belief"][HALF_SATURATION].multiplier["80%+"] for b in backtests}
    k_lo, k_hi = min(safe_k, key=safe_k.get), max(safe_k, key=safe_k.get)
    k_spread = max(np.ptp([b["belief"][h].multiplier["80%+"] for h in HALF_SATURATIONS]) for b in backtests)

    # Job 2's clock: each DCCC dollar counted by how much persuasion is left when voters vote
    def timing_reading(b, half_life):
        r, tm = b["races"], b["timing"][half_life]
        t, d = tm["races"], r.dccc.to_numpy()
        landed = lambda m: (d[m] * t.e_obs.to_numpy()[m]).sum() / (d[m] * t.e_best.to_numpy()[m]).sum()
        safe, close_band = (r.p > 0.8).to_numpy(), r.p.between(0.2, 0.8).to_numpy()
        effective = d * t.e_obs.to_numpy()
        return {"all": landed(d > 0), "close": landed(close_band & (d > 0)), "safe": landed(safe & (d > 0)),
                "dollars_safe": d[safe].sum() / d.sum(), "effective_safe": effective[safe].sum() / effective.sum(),
                "retime": tm["gain_retime"], "both": tm["gain_both"], "k": tm["belief"].multiplier["80%+"]}
    clock = {h: {b["cycle"]: timing_reading(b, h) for b in backtests} for h in HALF_LIVES}
    clock_rows = "\n".join(
        f"| {c} | {v['all']:.0%} / {v['close']:.0%} / {v['safe']:.0%} | {v['dollars_safe']:.0%} / {v['effective_safe']:.0%} | "
        f"{b['gain']:+.2f} / {v['retime']:+.2f} / {v['both']:+.2f} | {multiplier(b['belief'][HALF_SATURATION].multiplier['80%+'])} / {multiplier(v['k'])} |"
        for b in backtests for c, v in [(b["cycle"], clock[HALF_LIFE][b["cycle"]])])
    budget_w = {b["cycle"]: b["races"].dccc.sum() for b in backtests}
    clock_grid = "\n".join(
        f"| {h:g} days | {min(v['all'] for v in clock[h].values()):.0%}-{max(v['all'] for v in clock[h].values()):.0%} | "
        f"{min(v['retime'] for v in clock[h].values()):+.2f} to {max(v['retime'] for v in clock[h].values()):+.2f} | "
        f"{min(v['both'] for v in clock[h].values()):+.2f} to {max(v['both'] for v in clock[h].values()):+.2f} | "
        f"{sum(v['effective_safe'] * budget_w[c] for c, v in clock[h].items()) / sum(budget_w.values()):.0%} |" for h in HALF_LIVES)
    c0 = clock[HALF_LIFE]
    gains_place = [b["gain"] for b in backtests]

    game_rows = []
    for b in backtests:
        g, safe = games[b["cycle"]], b["races"].p.to_numpy() > 0.8
        game_rows.append(f"| {b['cycle']} | {b['races'].dccc[safe].sum() / b['races'].dccc.sum():.0%} | "
                         f"{g['x_d'][safe].sum() / g['x_d'].sum():.0%} | {g['seats_observed']:.1f} vs {g['seats_equilibrium']:.1f} | "
                         f"{g['exploit_observed']:.2f} | {g['exploit_equilibrium']:.2f} |")
    both = pd.concat([pd.DataFrame({"p": b["races"].p.to_numpy(), "dccc": b["races"].dccc.to_numpy(),
                                    "nrcc": b["races"].nrcc.to_numpy(), "eq_d": games[b["cycle"]]["x_d"],
                                    "eq_r": games[b["cycle"]]["x_r"]}) for b in backtests])
    close_band = both.p.between(0.2, 0.8)
    wide = {c: 1 - both[c][close_band].sum() / both[c].sum() for c in ("dccc", "nrcc", "eq_d", "eq_r")}
    hi = {c: both[c][both.p > 0.8].sum() / both[c].sum() for c in ("dccc", "nrcc")}
    lo = {c: both[c][both.p < 0.2].sum() / both[c].sum() for c in ("dccc", "nrcc")}
    max_eq_safe = max(games[b["cycle"]]["x_d"][b["races"].p.to_numpy() > 0.8].sum() / games[b["cycle"]]["x_d"].sum()
                      for b in backtests)
    media_rows = []
    for mc in media:
        res = {r["gamma"]: r for r in mc["results"]}
        media_rows.append(
            f"| {mc['cycle']} | {mc['tv_share']:.0%} | ${mc['dccc_per_race_expensive'] / 1e6:.1f}M vs ${mc['dccc_per_race_other'] / 1e6:.1f}M | "
            f"{mc['dccc_share_expensive']:.0%} | " + " / ".join(f"{res[gm]['model_share_expensive']:.0%}" for gm in GAMMAS)
            + " | " + " / ".join(f"{res[gm]['gain']:+.2f}" for gm in GAMMAS) + " |")
    latest = media[-1]["races"].set_index("district_id")
    ny = latest.loc[["NJ-07", "NY-04", "NY-17"]]
    mix = pd.concat(mixes).groupby(["group", "band", "channel"], observed=False).amount.sum()
    mix_share = mix / mix.groupby(level=["group", "band"], observed=False).transform("sum")
    cheap, pricey = BAND_LABELS[0], BAND_LABELS[-1]
    mix_years = f"{min(m.cycle.iloc[0] for m in mixes)}-{max(m.cycle.iloc[0] for m in mixes)}"
    lr20 = next(b["races"] for b in backtests if b["cycle"] == 2020).query("tier == 'Likely R' and margin.notna()")

    # section 8: household stress against the national polling miss
    nt, comp = national["table"], national["composites"]
    party = {"D": "Democrats", "R": "Republicans"}
    nat_span, nat_n = f"{nt.index.min()}-{nt.index.max()}", len(nt)
    nat_rows = "\n".join(f"| {y} | {party[r.president]} | {r.gb_sept:+.1f} | {r.gb:+.1f} | {r.house_vote:+.1f} | "
                         f"{r.miss_sept:+.1f} | {r.miss:+.1f} | {r.basket:+.1f}% | {r.stress:+.2f} |" for y, r in nt.iterrows())
    labels = {"basket": "cost basket", "stress": "credit and labor stress"}
    sept, econ = national["sept"], national["economics"]

    def both(key):               # final, then September, scores for one model
        get = (lambda res: res[key]) if key in national else (lambda res: res["composites"][key])
        return " | ".join(f"{get(res)['loo']:.2f} | {get(res)['forward']:.2f}" for res in (national, sept))
    model_rows = "\n".join([f"| Polls' average lean | {both('constant')} |", f"| + president's party | {both('president')} |",
                            f"| + president's party in midterms (no economic data) | {both('midterm')} |"]
                           + [f"| + {labels[k]} x president's party | {both(k)} |" for k in comp])
    sept_passed = [labels[k] for k, v in sept["composites"].items() if v["passes"]]
    sb, ss = sept["composites"]["basket"], sept["composites"]["stress"]
    econ_rows = "\n".join(
        [f"| Final polls, plus their average lean | {national['constant']['loo']:.2f} | {national['constant']['forward']:.2f} |",
         f"| Sept 1 polls, plus their average lean | {sept['constant']['loo']:.2f} | {sept['constant']['forward']:.2f} |",
         f"| Economics only: president's party and a midterm penalty, no economic data | {econ['midterm']['loo']:.2f} | {econ['midterm']['forward']:.2f} |"]
        + [f"| Economics only: + {labels[k]} | {econ[k]['loo']:.2f} | {econ[k]['forward']:.2f} |" for k in comp])
    best_econ = min(v["loo"] for v in econ.values())
    tie = abs(econ["midterm"]["forward"] - sept["constant"]["forward"])
    vs_sept = ("roughly ties" if tie < 0.25 else "beats" if econ["midterm"]["forward"] < sept["constant"]["forward"] else "trails")
    sept_over = int((nt.miss_sept < 0).sum())
    passed = [labels[k] for k, v in comp.items() if v["passes"]]
    verdict = ("Neither composite earns a place in the forecast" if not passed else
               f"The {' and '.join(passed)} earn{'s' if len(passed) == 1 else ''} a place in the forecast")
    worse = [labels[k] for k, v in comp.items() if v["loo"] > national["president"]["loo"] and v["forward"] > national["president"]["forward"]]
    yb, ys = nt.basket.idxmax(), nt.stress.idxmax()

    def against_polls(year):     # how the president's party ran against its final polls
        gap = nt.s[year] * nt.miss[year]
        return f"{party[nt.president[year]]}, the president's party, ran {abs(gap):.1f} points {'ahead of' if gap > 0 else 'behind'} their final polls"

    def direction(c):
        return "points the way the hypothesis says" if c < 0 else "runs the opposite way"
    cb, cs = comp["basket"], comp["stress"]
    lean_best = national["constant"]["loo"] < national["president"]["loo"] and national["constant"]["forward"] < national["president"]["forward"]
    lean_text = (f" Knowing which party holds the White House doesn't help either: the polls' average lean alone "
                 f"(Democrats running {abs(nt.miss.mean()):.1f} points {'behind' if nt.miss.mean() < 0 else 'ahead of'} "
                 f"their final polls) predicts best." if lean_best else "")

    text = f"""# Campaign allocation report

Generated by `python run.py` on {dt.date.today().isoformat()}. Every number below is computed by the pipeline unless marked otherwise.

## Summary

- **The forecast holds up out of sample.** Each year from {years} was predicted using only earlier elections. Brier scores run from {min(b['brier'] for b in backtests):.3f} to {max(b['brier'] for b in backtests):.3f}, and predicted and actual win rates match by tier from Safe D through Lean R (figure 3).
- **DCCC's own spending was close to the best use of the same money, if you trust the polls.** At the default spending assumption, reallocating DCCC's budget would have gained {min(b['gain'] for b in backtests):+.2f} to {max(b['gain'] for b in backtests):+.2f} expected seats per cycle (average {mean_gain:+.2f}). Even the most generous assumption tested gains at most {max(max(b['grid'].values()) for b in backtests):+.2f}. The main difference: DCCC put {safe_dccc:.0%} of its money into races already 80%+ likely to go Democratic, {safe_inc:.0%} of it protecting its own incumbents. In 2020 that caution was justified: {lost_2020} of those "safe" seats were lost anyway. And part of what looks like over-caution was the forecast's error, not DCCC's. First-term members, many elected in a wave, run well behind the usual incumbency edge. Flagging them, and carrying over each seat's last result, cuts the share of DCCC's 2020 money the forecast calls safe from {safe_20['p_without']:.0%} to {safe_20['p']:.0%} (figure 4, sections 4-5).
- **Two readings that don't assume how strong party money is.** Reallocating DCCC's budget gains a full seat only if its first $1M in a race moves the margin {min(first_needed):.1f}-{max(first_needed):.1f} points, against {first_default:.2f} assumed. And DCCC's own split is optimal only if a dollar in its 80%+ races works {min(safe_k.values()):.1f}-{max(safe_k.values()):.1f} times as hard as one in its 20-80% races, or those seats are worth that much more to it (section 5).
- **Judged with what a Sept 1 decision could see, less of that money looks over-cautious.** A forecast built from the Sept 1 polls plus the usual fall drift (about {abs(drift_last['mean']):.0f} points toward {'Republicans' if drift_last['mean'] < 0 else 'Democrats'}) is about as accurate on average as the final-poll one (log loss {ll_sept:.3f} against {ll_final:.3f}, better in {len(sept_better)} of {len(backtests)} years), and its seat ranges hold the result in {covered_sept} of {len(backtests)} years. Under it, DCCC's share in 80%+ seats falls from {safe_dccc:.0%} to {safe_sept:.0%}. Judging each dollar by the forecast on the date it reached voters instead (checkpoints from Sept 1 to Oct 25), the share is {safe_spent:.0%}, and what's left still implies a dollar in a safe seat works {np.nanmin(spent_k):.1f}-{np.nanmax(spent_k):.1f} times as hard. Through the fall the national uncertainty shrinks, but the race forecasts don't get sharper (sections 4-5).
- **If persuasion fades in days, timing matters more than placement.** Counting each DCCC dollar by how much of its effect is left when voters cast their ballots (a {HALF_LIFE:g}-day half-life, from TV-ad experiments, and each state's early-vote share from the federal EAVS), DCCC's money landed at {min(v['all'] for v in c0.values()):.0%}-{max(v['all'] for v in c0.values()):.0%} of what the same dollars could count at the best time. Re-timing within the same races gains {min(v['retime'] for v in c0.values()):+.1f} to {max(v['retime'] for v in c0.values()):+.1f} expected seats per cycle, against {min(gains_place):+.2f} to {max(gains_place):+.2f} for moving money between races, before the higher price of late airtime. Timing doesn't explain the safe-seat spending (section 5).
- **Neither strategy nor TV prices explain that defensive spending.** In a two-player game against NRCC, DCCC's best play puts at most {max_eq_safe:.0%} of its money in 80%+ seats (figure 5). DCCC already spends about half as much per race in the costliest TV markets, and pricing in market cost makes the model leave them almost entirely. No spender shifts toward mail where TV is most expensive (figure 6).
- **{'Household stress does not explain where the polls miss.' if not passed else 'Household stress helps explain where the polls miss.'}** Over the {nat_n} elections with a final RealClearPolitics generic-ballot average ({nat_span}), {'neither the household cost basket from the HOUSING AFFORDABILITY TRACKER (its own items and weights, pulled from FRED) nor a credit-and-labor stress index predicts' if not passed else 'the ' + ' and '.join(passed) + ' predict' + ('s' if len(passed) == 1 else '')} where the House vote lands relative to the polls{'; both make the prediction worse' if len(worse) == len(comp) else ''}. {'The same holds against the polls on Sept 1, when the allocation decision is made' if not sept_passed else 'Against the Sept 1 polls, the ' + ' and '.join(sept_passed) + ' help'}, and a ballot built from economics alone misses the House vote by more than the September polls do leaving one election out ({best_econ:.1f} points against {sept['constant']['loo']:.1f}) (section 8).
- **{fc['cycle']}: the national environment matters more than the new maps.** Redrawn maps in {redrawn.nunique()} states move Harris-won districts from {int(harris_old.sum())} to {int(harris_new.sum())} in those states (figure 7). At the current D{gb:+.1f} generic ballot, the model expects {s26['mean']:.0f} Democratic seats among {len(fc['races'])} modeled races (80% range {s26['p10']}-{s26['p90']}), a {s26['p_majority']:.0%} chance of a majority. A 3-point polling miss moves the expected total by about {(seats[round(gb + 3, 1)] - seats[round(gb - 3, 1)]) / 2:.0f} seats (figures 8-9).

## 1. Data

| Source | What | Notes |
|---|---|---|
| [The Downballot](https://www.the-downballot.com/p/the-downballots-calculations-of-presidential) | 2008-2024 presidential results by congressional district, one sheet per set of district lines | Calculations by The Downballot (formerly Daily Kos Elections). Fetched at run time, not redistributed |
| [MIT Election Lab](https://doi.org/10.7910/DVN/IG0UN2) | U.S. House general-election returns, 1976-2024 | CC0. Downloaded by hand (Dataverse asks for a short form) |
| [Jonathan Cervas](https://jonathancervas.com/maps/media-markets/) | Each district's population by Nielsen TV market, 2022-2026 lines | Derived by Cervas from Nielsen market boundaries and Census 2024 county estimates. Parsed from his page at run time, not redistributed |
| [FEC bulk data](https://www.fec.gov/data/browse-data/?tab=bulk-data) | All-candidate summaries (weball), each candidate's principal campaign committee (cn), and every reported independent expenditure, 2012-2026 | Public, no API key |
| [FEC API](https://api.open.fec.gov/developers/) | Each House campaign's July report: money raised and spent through June 30 of the election year, 2012-2026 | Public; needs a free api.data.gov key in `FEC_API_KEY` |
| [`data/generic_ballot.csv`](data/generic_ballot.csv) | Final RealClearPolitics generic-ballot average before each election, 2002-2026, and the average on Sept 1, 2002-2024 | Compiled by hand, sources in the file |
| [FRED](https://fred.stlouisfed.org) | The HOUSING AFFORDABILITY TRACKER's household price basket ({nt.basket_items.max()} series) and {len(STRESS)} credit and labor series, monthly or quarterly | Section 8 only. Public, no key. Today's revised values |

**What the raw data gets wrong, and what this repo does about it** (each checked by hand while building this repo, October 2026):

1. **Partisan lean on the wrong map.** A House race has to be compared with the presidential vote *on its own district lines*, which change with every redraw: 2012; FL, NC, PA and VA between 2016 and 2020; 2022; 2024; and {redrawn.nunique()} states for {fc['cycle']}. Each cycle here uses the sheet for the lines it was run on. PVI uses only the most recent presidential election before the race. In forward tests that gave better-calibrated forecasts (lower Brier scores) than a two-election average in every year from 2016 to 2024.
2. **Court rulings after the data was published.** The Downballot's July 2026 sheet has Missouri on its new map; the U.S. Supreme Court ruled on 2026-09-25 that Missouri votes on its 2022 lines. The 2026 forecast uses the 2022 lines there.
3. **Fusion voting.** In New York and Connecticut one candidate can appear on several party lines. Counting only the Democratic and Republican lines puts NY-01 (2022) at R+6.6; summing each candidate's Working Families and Conservative votes gives the true R+11.0.
4. **State party labels.** Minnesota's Democratic-Farmer-Labor and North Dakota's Democratic-NPL candidates are Democrats. Missing them turns both states into Republican landslides.
5. **Amended reports counted twice.** The FEC's independent-expenditure file keeps every version of a report. An amendment replaces the whole report it amends, so rows from superseded reports are dropped. Raw, the file puts DCCC's 2024 independent spending at $103.0M; with the rule it is $74.2M, against FEC's published $74.26M (2022: $96.4M, matching FEC exactly). House Majority PAC's 2024 total drops from $241M to $196M. Deduplicating by transaction ID instead looks right for DCCC but collapses Congressional Leadership Fund's 2022 spending from $227M to $1M, because its transaction IDs repeat.
6. **Spending with no party attached.** On 10-15% of rows the target candidate's party is blank, including $213M of CLF's $217M in 2024. Dropping those rows undercounts Republican outside money by nearly three quarters in 2024 ($87M instead of $314M). Here the party comes from FEC's candidate list instead: the candidate in that district with that surname. 98-99.5% of every cycle's House general-election spending ends up assigned to a side.
7. **The wrong candidate's money.** FEC's candidate list includes primary losers and House members running for another office, whose committees still sit under their old House district (Gallego, AZ-03, and Schiff, CA-30, in 2024; a stale Sinema committee in AZ-09 spending $4.4M in 2026). Candidates are matched to the names on the November ballot by surname, after stripping accents ("SÁNCHEZ") and searching the whole FEC name for married names ("MATTELIN, CAITLIN RAE DRAPER" is the ballot's Caitlin Draper). Members filed under their pre-redistricting district are matched within the state on surname plus first initial (Schweikert, AZ-01, 2022). For 2026, with no ballot yet, an incumbent flag counts only for a current member from that state who isn't running for Senate this cycle (Hinson in IA-02 and Barr in KY-06 are, so those seats are open). Members leaving for a governor's race or resigning can still be mislabeled, because FEC doesn't track those; IA-04 and SD-00, both safe Republican seats, are the known cases.
8. **Gaps in FEC's summary file.** A few nominees are missing from it altogether (DeLauro in 2016, Cisneros in 2024), and their spending counts as $0. That leaves 1-6 races per cycle decided by under 20 points with an unmatched nominee.
9. **Fraudulent filings.** Some files contain single "expenditures" of $1 billion or more: a "COMMITTEE 300" reports $10B twice "for cash in hand" in 2022, and one person files billions under names like "Warren Buffet" in 2026. No real single House expenditure from 2012 to 2026 tops $5M, so any row over $25M is dropped.
10. **Polls, not results, for the national environment.** A forecast can only use the polling average before the election. (An earlier version of this project used the actual House popular vote for some years, which leaks the answer.)
11. **2026 committee money is invisible here.** *NRSC v. FEC* (June 30, 2026) struck down limits on party coordinated spending. DCCC reports $0 in independent expenditures and $60.9M in coordinated spending for 2026 through Oct 14, and coordinated spending isn't in this file. So the allocation analysis covers {years}, not 2026. Through 2024 coordinated spending was small (DCCC: $5.4M in 2024).

| Cycle | Races | Contested | Generic ballot | DCCC | NRCC | Outside D | Outside R |
|---|---|---|---|---|---|---|---|
{data_rows}

Races with under $100K of total spending and Alaska (ranked-choice) are excluded. "Contested" means decided by less than 80 points. Dollar figures are House general-election independent expenditures in modeled races, deduplicated.

## 2. Features

| Feature | Definition | Why |
|---|---|---|
| `pvi` | 100 x (district D two-party presidential share - national share), last presidential election before the race, on the race's own lines | The single strongest predictor |
| `d_incumbent`, `open_seat` | FEC incumbent flag on the matched nominees; a Republican incumbent is the baseline | Incumbents run ahead of their district's lean |
| `floor_share` | log((F_D + $10K) / (F_D + F_R + $20K)), F = what that side's nominee had raised from others by June 30 + outside money that had reached voters by Aug 31 | Candidate strength and how hard each side contests the race, using only money neither national committee controls, and only as much of it as a forecaster deciding on Sept 1 could see |
| `fresh_d`, `fresh_r` | The party's nominee is a first-term incumbent: not among the state's House winners four years earlier | Members first elected in a wave run well behind the usual incumbency edge |
| `over_inc`, `over_open` | Last cycle's margin - that year's generic ballot - 1.6 x its PVI, if the state's lines haven't changed. Set when that party's incumbent runs again (`over_inc`) or the seat is open (`over_open`), zero otherwise | How far the seat's party ran ahead of the district's lean, some of which carries over |
| `gb` | Generic-ballot average before the election | The national environment, applied as a uniform swing |

## 3. The model

The model does two separate jobs.

**Job 1: the baseline forecast** (fit by least squares on contested races, {m['cycles'][0]}-{m['cycles'][-1]}, n = {m['n']}):

```
margin_i - GB = a + b*PVI_i + c*D_incumbent_i + d*Open_i + e*floor_share_i
                + f*fresh_d_i + g*fresh_r_i + h*over_inc_i + k*over_open_i + error_i,   error_i ~ Normal(0, sigma^2)
P(Democrat wins race i) = Phi(expected margin_i / sigma)
```

| a (intercept) | b (PVI) | c (D incumbent) | d (open seat) | e (floor share) | f (first-term D) | g (first-term R) | h (overperformance, incumbent) | k (overperformance, open) | sigma |
|---|---|---|---|---|---|---|---|---|---|
| {c['intercept']:+.2f} | {c['pvi']:.2f} | {c['d_incumbent']:+.2f} | {c['open_seat']:+.2f} | {c['floor_share']:.2f} | {c['fresh_d']:+.2f} | {c['fresh_r']:+.2f} | {c['over_inc']:.2f} | {c['over_open']:.2f} | {m['sigma']:.2f} |

Each point of PVI is worth {c['pvi']:.1f} points of margin. Having the Democrat rather than the Republican as the incumbent is worth {c['d_incumbent']:.0f} points, {-c['fresh_d']:.0f} less for a first-term Democrat; first-term Republicans show no such gap ({c['fresh_r']:+.1f}). About a fifth of how far a seat's party ran ahead of its lean last time carries into the next election ({c['over_inc']:.2f} with the incumbent running again, {c['over_open']:.2f} for an open seat). The generic ballot is fixed at a 1:1 pass-through rather than estimated: with one national value per cycle, a fitted slope would mostly chase polling misses. sigma is the residual spread, so it includes those misses; that is the right uncertainty for a forecast made before the votes are counted.

**National, state and district error.** Part of every race's error is shared: in a year the polls miss, they miss everywhere at once, and races in the same state also miss together. Splitting it out, sigma^2 = sigma_national^2 + sigma_state^2 + sigma_district^2. sigma_national = {m['sigma_national']:.1f} points is the spread of each cycle's average miss. sigma_state = {m['sigma_state']:.1f} is how much each state's average miss varies beyond what its number of races explains, and sigma_district = {m['sigma_district']:.1f} is the rest. A single race's probability needs only the total. The seat total needs the split: conditional on the national and state misses, races are independent, and averaging over both gives the seat distribution in section 7.

sigma_national rests on only {len(m['cycles'])} cycle averages, so it is itself uncertain: anywhere from {nat_lo:.1f} to {nat_hi:.1f} points is plausible (90% range). The seat total doesn't treat it as known. Averaging a normal national miss over that uncertainty gives a Student t with {m['national_df']} degrees of freedom, so a big miss is more likely than the point estimate alone says, and the fewer cycles behind the estimate, the more so. Each race's own probability, and so the allocation, still uses the point estimate.

![PVI vs margin](figures/1_pvi_vs_margin.png)

**Job 2: what party money does.** This is an assumption, not an estimate:

```
shift(x) = scale * log(1 + x / half_saturation),   default scale = 1.5 points, half_saturation = $5.5M
P(Democrat wins i | allocation x) = Phi((expected margin_i + shift(x_i) - shift(observed_i)) / sigma)
```

Public data can't separate the effect of money from the reasons money goes where it goes. Close races and strong candidates attract it. Returns diminish in absolute dollars, not in spending share; a share-based curve makes the first dollar in a quiet long-shot race look ten times as powerful as a dollar in a toss-up. The baseline already reflects the money actually spent, so only deviations from DCCC's observed allocation move a race. Results are shown across a range of both parameters; measured effects (for example randomized holdouts on mail programs) should replace the defaults.

![Spending effect](figures/2_spending_effect.png)

**Allocation.** DCCC's budget is spent greedily: each step funds the race where the next chunk buys the most expected seats, with no race taking more than 15% of the budget. This is exact, not a shortcut. Each race's win probability is concave in its spending unless the race is more than about 40 points out (z < -sigma/scale), so the objective is a sum of concave functions under one budget, and small greedy steps reach the global optimum. A numerical solver lands within 0.01 seats of it. Splitting the error doesn't change this allocation either, because averaging over the national and state misses gives back exactly the pooled sigma.

## 4. Does the baseline forecast well?

Each year is forecast from a model fit only on earlier years.

| Year | Brier score | Expected D seats | Actual D seats | DCCC budget | Gain from reallocating (default) | Gain range (9 assumptions) |
|---|---|---|---|---|---|---|
{bt_rows}

Seats are counted among modeled races, so Alaska and races with under $100K of spending are left out. The biggest miss is 2020, and three errors stacked up there. The polls overstated Democrats nationally, and the average contested race ran {-behind['mu'].mean():.1f} points behind the forecast. Democratic incumbents ran {-behind['mu'][c20.status == 'D incumbent'].mean():.1f} points behind and Republican incumbents only {-behind['mu'][c20.status == 'R incumbent'].mean():.1f}: the 2020 forecast learned incumbency from earlier years, and the edge shrank sharply after 2016 (section 9). And many of those Democrats were first-term members elected in the 2018 wave. Without the incumbent history below they ran {-behind['mu_without'][fresh20].mean():.1f} points behind the forecast; with it, {-behind['mu'][fresh20].mean():.1f}, in line with the {-behind['mu'][vet20].mean():.1f} of other Democratic incumbents.

**What the incumbent history adds.** The same forecast without the first-term flags and last cycle's overperformance, fit and scored the same way:

| Year | Brier: without / with | Log loss: without / with | Upsets of an 80%+ favorite: without / with |
|---|---|---|---|
{history_rows}

Averaged over {years}, log loss falls from {ll_mean['without']:.3f} to {ll_mean['with']:.3f}. The history helps most in {helped[0]} and {helped[1]}{', and costs a little in ' + ' and '.join(map(str, hurt)) if hurt else ''}.

**What hindsight money was worth.** The money feature counts only what a forecaster deciding on Sept 1 could see: what each nominee had raised by June 30 (the FEC's July report; {raised_june:.0%} as much as nominees went on to spend over the whole election), less the candidate's own contributions and loans, and outside money that had reached voters by Aug 31 ({seen_outside:.0%} of all outside money, {first}-{last}). An earlier version of this repo counted all of it, though most is spent in the fall and some follows how the race is going. That hindsight version, fit and scored the same way:

| Year | Brier: hindsight / Sept 1 | Log loss: hindsight / Sept 1 | Upsets of an 80%+ favorite: hindsight / Sept 1 |
|---|---|---|---|
{money_rows}

Averaged over {years}, log loss is {ll_mean['hindsight']:.3f} with hindsight and {ll_mean['with']:.3f} without. On the {len(close_sept)} races the September forecast put between 5% and 95%, it is {ll_close['p_hindsight']:.3f} with hindsight and {ll_close['p']:.3f} without. So hindsight made the forecast look somewhat better than a September decision could be, mostly in close races. Dropping the money feature altogether forecasts worse than either (tested while building this repo): the money visible by September keeps most of its value.

![Calibration](figures/3_calibration.png)

**The seat total, with the national and state errors shared across races and the size of the national error uncertain:**

| Year | Expected D seats | 80% range | Actual | Needed for majority | P(D majority) | Majority won by |
|---|---|---|---|---|---|---|
{chr(10).join(seat_rows)}

The 80% range held the actual result in {covered} of {len(backtests)} years, and the favorite for the majority won in {calls_right} of {len(backtests)}. Outside the range: {missed_text}. ("Needed" counts only modeled races; seats outside the model are credited by their partisan lean.) Each backtest's national error rests only on the cycles before it: {len(backtests[0]['model']['cycles'])} for {backtests[0]['cycle']}, {len(backtests[-1]['model']['cycles'])} for {backtests[-1]['cycle']}. Treating its size as known instead, the 80% ranges average {width_known:.0f} seats wide rather than {width:.0f} and hold the result in {covered_known} of {len(backtests)} years. The fatter tails also pull each race a little toward even, so expected seats here can differ by a seat or two from the race-by-race total in the first table.

**The forecast a Sept 1 decision could make.** Everything above uses the final pre-election generic ballot, which a decision made in September didn't have. The Sept 1 version (`fall_drift.py`) feeds the same fitted model the generic ballot on Sept 1, plus the average fall drift of earlier elections: how far the polls moved between Sept 1 and the final average. It adds the drift's spread to the national error, since every race moves with it. The polls have moved toward {'Republicans' if drift_last['mean'] < 0 else 'Democrats'} through the fall, by {abs(drift_last['mean']):.1f} points on average over the {drift_last['n']} elections before {backtests[-1]['cycle']}, give or take {drift_last['sd']:.1f}.

| Year | Generic ballot, Sept 1 | Expected fall drift (from earlier elections) | Best guess of the final | Final generic ballot | Log loss: final polls / Sept 1 | 80% seat range: final polls / Sept 1 | Actual seats |
|---|---|---|---|---|---|---|---|
{sept_rows}

Averaged over {years}, log loss is {ll_sept:.3f} at Sept 1 against {ll_final:.3f} with the final polls. The Sept 1 forecast does better in {', '.join(sept_better) if sept_better else 'no year'} and worse in the other {len(backtests) - len(sept_better)}, as older information should{', so its edge on average comes from that one year' if len(sept_better) == 1 else ''}. Its wider uncertainty makes the seat ranges more honest: they hold the result in {covered_sept} of {len(backtests)} years, against {covered} with the final polls.

**Through the fall.** The same forecast can be made at any date (`dated_forecast.py`): the generic ballot that day plus the average drift from that date to the final average, and outside money counted as far as it had reached voters by then, in the training years too. Five checkpoints, with each date's generic ballot and its source in [`data/generic_ballot_dated.csv`](data/generic_ballot_dated.csv):

| Checkpoint | Expected drift to the final average (from the {backtests[-1]['when_spent']['drift'][CHECKPOINTS[0]]['n']} elections before {backtests[-1]['cycle']}) | Its spread | Log loss, {years} | DCCC's money that reached voters from then until the next checkpoint |
|---|---|---|---|---|
{term_rows}
| Final polls (hindsight) | 0 | 0 | {ll_final:.3f} | |

The national uncertainty shrinks through the fall, from a spread of {backtests[-1]['when_spent']['drift'][CHECKPOINTS[0]]['sd']:.1f} points at {label(CHECKPOINTS[0])} to {backtests[-1]['when_spent']['drift'][CHECKPOINTS[-1]]['sd']:.1f} by {label(CHECKPOINTS[-1])}. The race forecasts don't get sharper on average, though: log loss stays between {min(ll_at.values()):.3f} and {max(ll_at.values()):.3f} at every checkpoint, against {ll_final:.3f} with the final polls. Polls can swing and come back: the checkpoint furthest behind its own year's final-poll forecast is {label(worst[2])}, {worst[1]} (log loss {worst_b['scores'][worst[2]]['log_loss']:.3f} against {worst_b['scores']['with']['log_loss']:.3f}), when the generic ballot stood at {dated_ballot[(worst[1], worst[2])]:+.1f}, against a final average of {worst_b['races'].gb.iloc[0]:+.1f}. (Dollars out before Sept 1 are counted with the Sept 1 checkpoint.)

![Through the fall](figures/11_through_the_fall.png)

## 5. Was DCCC's money well placed?

Average gain from reallocating DCCC's budget over {years}, in expected seats per cycle, by spending assumption:

| Scale \\ half-saturation | $1.65M | $5.5M | $16.5M |
|---|---|---|---|
{grid}

Under every assumption tested the answer is "close to optimal": a fraction of a seat at the default, and under two seats even if party money is three times as powerful and saturates three times as fast. The model would put {share_model:.0%} of the money in races between 20% and 80%, against DCCC's {share_dccc:.0%}.

**How strong would party money have to be?** The grid fixes a strength and asks what reallocating gains. Turned around: how strong would party money have to be for reallocating DCCC's budget to gain one full expected seat? The gain never falls as the scale rises, so each cycle has one break-even scale for each curve shape:

| Year | Scale needed, half-saturation $1.65M / $5.5M / $16.5M | Times the default scale (at $5.5M) | DCCC's first $1M in a race would have to move it |
|---|---|---|---|
{be_rows}

At the default half-saturation the scale would have to be {min(be_default) / SCALE_PTS:.0f} to {max(be_default) / SCALE_PTS:.0f} times the default. Put in dollars, the curve's shape barely matters: under all three half-saturations, a full seat needs DCCC's first $1M in a race to move its margin {min(first_needed):.1f} to {max(first_needed):.1f} points, against {first_default:.2f} assumed. That is a number a measured effect can be checked against. A randomized test that found less than {min(first_needed):.2f} points per first $1M would put the gain from reallocating under a seat in every cycle tested, under any of the three curve shapes.

![Allocation](figures/4_allocation.png)

**Where the difference goes: protecting incumbents.** Maximizing expected seats puts almost nothing in races the forecast already has at 80%+; DCCC puts real money there every year:

| Year | DCCC's budget in 80%+ races | Model, same budget | DCCC dollars there | Share to Democratic incumbents | Of those seats, lost anyway |
|---|---|---|---|---|---|
{chr(10).join(protect_rows)}

Outside 2018, most of it protects DCCC's own incumbents, consistent with its "Frontline" program for vulnerable members. In 2018 the 80%+ money went to Republican seats the forecast already had as likely pickups. In 2020 the caution was justified, and part of what looks like over-caution was the forecast's own error. Without the incumbent history, the forecast puts {safe_20['p_without']:.0%} of DCCC's 2020 money in 80%+ seats, and {len(lost_without)} of the seats DCCC defended there were lost, {fresh_lost} of them held by first-term members. With it, that share is {safe_20['p']:.0%}, and {', '.join(cleared) if cleared else 'none'} drop below 80%. {', '.join(still)} stay above 80% and were lost anyway{(', and ' + ', '.join(added) + (' is' if len(added) == 1 else ' are') + ' newly rated above 80% and lost') if added else ''}. The 2020 forecast still overrated Democratic incumbents as a group (section 4). Whether DCCC's money saved any others can't be seen in this data. So "close to optimal" is conditional on the forecast being right. DCCC's main deviation from it is insurance against it being wrong, concentrated on its own members. Two other objectives were tested while building this repo and are not in the pipeline. Maximizing the probability of a majority reproduces the defense in 2018 and 2020 but goes on offense in years Democrats trail (2016, 2022, 2024). Maximizing expected seats in the worst quarter of national environments adds defense every year, but most in the years DCCC defended least. Neither matches DCCC by itself.

**Judged with what DCCC could see.** DCCC's money reaches voters from September to November: {early_share:.0%} of it by Sept 30 and {late_share:.0%} after Oct 15 ({years}). So the readings are redone two ways (section 4): every dollar judged by the Sept 1 forecast, and each dollar judged by the forecast at the last checkpoint before it reached voters ("when spent"). The best allocation and the gain need one forecast for the whole budget, so they use Sept 1.

| Year | DCCC's budget in 80%+ races: final polls / Sept 1 / when spent | Model, same budget, in 80%+ races (Sept 1) | Gain from reallocating: final polls / Sept 1 | 80%+ multiplier: final polls / Sept 1 / when spent |
|---|---|---|---|---|
{judged_rows}

Pooled, the share of DCCC's money in races the forecast calls 80%+ falls from {safe_dccc:.0%} to {safe_sept:.0%}.{(' The change is in ' + shifted_text + (', years when' if len(shifted) > 1 else ', a year when') + " the final polls ended up more Democratic than a September forecaster, expecting the usual fall drift, would have guessed. Seats that look safe in hindsight looked less safe then.") if shifted and shifted_why else ''} The gain from reallocating barely changes, and DCCC's remaining 80%+ money still implies a dollar there works {np.nanmin(sept_k):.1f}-{np.nanmax(sept_k):.1f} times as hard as one in a 20-80% race. Judged dollar by dollar, the share is {safe_spent:.0%}, between the two: later in the fall the polls and the outside money had mostly moved toward what the final-poll forecast says, and the multiplier is {np.nanmin(spent_k):.1f}-{np.nanmax(spent_k):.1f}x. So judging DCCC by what it could see when it spent explains about {1 - safe_spent / safe_dccc:.0%} of its safe-seat spending pooled ({1 - safe_sept / safe_dccc:.0%} if every dollar is judged at Sept 1), and not the rest (figure 12).

![Safe-seat share](figures/12_safe_seat_share.png)

**What would DCCC have to believe?** The model's allocation stops where the last dollar in every funded race buys the same expected seats (apart from races at the 15% cap). DCCC's own allocation can be read backwards against that condition: at DCCC's actual spending, what does one more expected seat cost at the margin in each band, and how much harder would a dollar have to work in the costlier bands for DCCC's split to be optimal?

| Year | Cost of one more expected seat at the margin (default assumption): under 20% / 20-80% / 80%+ | Multiplier needed: under-20% races | Multiplier needed: 80%+ races | 80%+ multiplier across half-saturations |
|---|---|---|---|---|
{chr(10).join(belief_rows)}

The multiplier is how many times as far a dollar would have to move the margin in that band, compared with DCCC's 20-80% races, for its split to be optimal. The same number also reads as how many times as much those seats would have to be worth to DCCC, or as a sign that the forecast understates their risk. This data can't tell the three apart. Each band's cost is DCCC's dollars there divided by the expected seats that spending a little more in proportion would buy, a dollar-weighted average of the marginal effect rather than an average of each race's cost, so one small buy in a hopeless race can't dominate a band (a band holding nothing else can). In 80%+ races it runs from {safe_k[k_lo]:.1f}x ({k_lo}) to {safe_k[k_hi]:.1f}x ({k_hi}). Unlike the gain, it doesn't depend on how strong party money is assumed to be: the strength cancels in the ratio. Moving the half-saturation from $1.65M to $16.5M changes no year's 80%+ multiplier by more than {k_spread:.1f}. The costs in the first column do scale with the assumed strength; only their ratios are free of it.

**When the money lands.** Everything above treats a dollar the same whether it reaches voters on Sept 10 or Nov 1. But persuasion fades within days. In a randomized TV experiment, a buy that moved a governor's race about 5 points while it aired had no detectable effect the week after (Gerber, Gimpel, Green and Shaw 2011). Across the 2000 presidential and 2006 statewide races, ad effects had half-lives of about 3-4 days for president and 1-2 days for state and local races (Hill, Lo, Vavreck and Zaller 2013). And many voters cast their ballots weeks before Election Day. So `ballot_timing.py` counts each DCCC dollar by how much of its effect is left when each voter votes:

- **Decay:** a half-life of {HALF_LIFE:g} days by default, a little more durable than the experiments' state and local estimates, with {' and '.join(f'{h:g}' for h in HALF_LIVES if h != HALF_LIFE)} days as bounds. Lasting effects (name recognition, turnout) are left out.
- **Ballots:** each state's share cast before Election Day, from the Election Assistance Commission's Election Administration and Voting Survey for that year (mail, early in person and overseas ballots against Election Day and provisional ones). Early voters are spread evenly over the {WINDOW} days before Election Day.
- **Best timing:** the same money spread over the best {SPREAD} days for that state's ballots, not the best single day, since a race can absorb only so much airtime and mail at once.

Effective dollars are scaled so DCCC's whole budget keeps the strength the spending curve assumes; only how well each dollar was timed changes. At a {HALF_LIFE:g}-day half-life:

| Year | DCCC's money as it landed, share of its best timing: all / 20-80% races / 80%+ races | DCCC's budget in 80%+ races: dollars / effective | Expected seats gained: moving money between races / re-timing it in the same races / both | 80%+ multiplier: as above / with timing |
|---|---|---|---|---|
{clock_rows}

| Half-life | DCCC's money as it landed, share of its best timing | Gain from re-timing only | Gain from re-timing and moving | DCCC's effective budget in 80%+ races, pooled |
|---|---|---|---|---|
{clock_grid}

If persuasion fades this fast, when DCCC spends matters more than where. Its money landed at {min(v['all'] for v in c0.values()):.0%}-{max(v['all'] for v in c0.values()):.0%} of what the same dollars could count at the best time: {early_share:.0%} of it reaches voters by Sept 30, five weeks or more before nearly all of them vote. Keeping each race's budget but timing it to the state's ballots gains {min(v['retime'] for v in c0.values()):+.1f} to {max(v['retime'] for v in c0.values()):+.1f} expected seats per cycle, against {min(gains_place):+.2f} to {max(gains_place):+.2f} for moving money between races at the same assumption. That doesn't explain the safe-seat spending, though. DCCC times its 80%+ money about as well as its 20-80% money, its effective share in those races is close to its dollar share, and the multiplier barely moves. Three things the gain leaves out cut it down: airtime bought late costs more (outside groups and party committees don't get candidates' lowest rates), early mail and field work also reach the voters who vote early, and some effects of money (name recognition, turnout) outlast a TV ad's. So treat the re-timing gain as an upper bound under the decay assumption, not a measured effect.

**Would a committee facing a strategic opponent spend differently?** Another explanation for defending "safe" seats is deterrence: a seat DCCC leaves at $0 is where NRCC's first dollars go furthest. That can be tested by solving the game. DCCC maximizes expected Democratic seats and NRCC minimizes them, each re-spending its real budget under the same spending curve. The payoff is concave for DCCC and convex for NRCC, so a stable equilibrium exists. Fictitious play finds it (each side best-responds to the average of the other's past choices).

| Year | DCCC's budget in 80%+ races: actual | At equilibrium | Expected D seats: actual vs equilibrium | Seats the two sides could still gain: actual | At equilibrium |
|---|---|---|---|---|---|
{chr(10).join(game_rows)}

The equilibrium puts almost nothing in 80%+ seats. A raid on a seat that's 85% Democratic buys margin where it barely changes who wins, so it isn't worth deterring. Both committees' actual allocations are within about a tenth of a seat of the equilibrium value. Deterrence doesn't explain the defense either.

![Equilibrium](figures/5_equilibrium.png)

Figure 5 shows the whole picture. At equilibrium each side's best reply is to meet the other in the same close races: {1 - wide['eq_d']:.0%} of DCCC's money and {1 - wide['eq_r']:.0%} of NRCC's goes to races between 20% and 80%. In reality both spread much wider, DCCC with {wide['dccc']:.0%} of its money outside that band and NRCC with {wide['nrcc']:.0%}. Most of that extra money lands in the same place, seats Democrats are 80%+ likely to win: {hi['dccc']:.0%} of DCCC's budget defends them and {hi['nrcc']:.0%} of NRCC's attacks them. Seats Democrats are under 20% likely to win get far less, {lo['dccc']:.0%} of DCCC's budget on offense and {lo['nrcc']:.0%} of NRCC's on defense.

## 6. TV markets and channels

TV time is bought by market, not by district, so a TV buy pays to reach everyone in the market. [Jonathan Cervas's analysis](https://jonathancervas.com/maps/media-markets/) splits each district's population across Nielsen TV markets for the 2022, 2024 and 2026 lines. His `reach`, the people in a district's main market per district resident, measures how much of a TV buy is wasted. It is {ny.market_pop.iloc[2] / ny['pop'].iloc[2]:.0f} for NY-17 (the New York market), against a median of about {latest.market_pop.div(latest['pop']).median():.0f}. That overstates the waste where one market holds several battlegrounds: a New York buy reaches NJ-07, NY-04 and NY-17 at once, which cuts the figure to {ny.shared_reach.iloc[2]:.1f}. So the cost index here counts every battleground district (20-80%) sharing the market. It raises the saturation point only for the TV share of spending, with strength gamma (0 = no market cost, 1 = TV dollars buy proportionally less in costlier markets).

| Year | TV share of DCCC's spending | DCCC per battleground race: costliest quarter of markets vs rest | DCCC's budget in costliest quarter | Model's share there (gamma 0 / 0.5 / 1) | Gain vs DCCC (gamma 0 / 0.5 / 1) |
|---|---|---|---|---|---|
{chr(10).join(media_rows)}

DCCC already spends about half as much per race where TV is costliest. Pricing in market cost pushes the model further the same way: with any real cost penalty it leaves those districts almost entirely, and DCCC's gap to the model widens a little. That runs against a common intuition. Expensive districts don't need more money from a seat-maximizer; the same dollar buys less there, so it goes where it buys more.

![Channel mix](figures/6_channel_mix.png)

Mail and digital don't pay for a whole market, so they should take a bigger share where TV is most expensive. Across {mix_years} they don't:

- **DCCC** shifts only slightly toward digital ({mix_share.get(('DCCC', cheap, 'Digital'), 0):.0%} in the cheapest markets, {mix_share.get(('DCCC', pricey, 'Digital'), 0):.0%} in the costliest) and reports no mail.
- **House Majority PAC** goes the other way: TV is {mix_share.get(('House Majority PAC', cheap, 'TV'), 0):.0%} of its spending in the cheapest markets and {mix_share.get(('House Majority PAC', pricey, 'TV'), 0):.0%} in the costliest.
- **All outside groups'** mail share is {mix_share.get(('All outside groups', cheap, 'Mail'), 0):.0%} in the cheapest markets and {mix_share.get(('All outside groups', pricey, 'Mail'), 0):.0%} in the costliest.

Whether that is a missed opportunity depends on how many votes a dollar of mail moves compared with a dollar of TV, which this data can't tell. Randomized holdouts on mail programs can.

## 7. {fc['cycle']}

Forecast from a baseline fit on all {m['cycles'][0]}-{m['cycles'][-1]} races, at the current generic ballot (D{gb:+.1f}).

![Map changes](figures/7_map_changes.png)

![Seats by generic ballot](figures/8_seats_by_gb.png)

| Generic ballot | D{gb - 3:+.1f} | D{gb:+.1f} (now) | D{gb + 3:+.1f} |
|---|---|---|---|
| Expected Democratic seats | {seats[round(gb - 3, 1)]:.0f} | {seats[round(gb, 1)]:.0f} | {seats[round(gb + 3, 1)]:.0f} |

Averaging over the national and state errors instead of fixing them gives the full distribution. The model puts Democrats at {s26['mean']:.0f} seats among modeled races, an 80% range of {s26['p10']}-{s26['p90']}, and a {s26['p_majority']:.0%} chance of reaching the {s26['need']} they need for a majority. That already allows for the size of a national miss being estimated from only {len(m['cycles'])} elections (section 3); treating it as known would give {fc['seats_known_sigma']['p_majority']:.0%}. It is still likely optimistic: the generic ballot here is the final average, not one a September decision could see, and the state and district errors are still assumed normal. A 4-point polling miss like 2020's would leave Democrats near {seats[round(gb - 4, 1)]:.0f} seats.

![Seat distribution](figures/9_seat_distribution.png)

{len(close)} races sit between 20% and 80%. {sum(close.district_id.isin(maps[maps.changed].district_id))} of them are on redrawn lines; most of the rest are Republican-held seats that a D{gb:+.1f} environment puts in play.

![Battleground](figures/10_battleground.png)

**Incumbents on redrawn lines.** The incumbency effect is learned from members running in their own districts, and is probably too large for members whose districts were redrawn. Each redrawn-incumbent race with either chance between 20% and 80% is shown below, with full incumbency credit and with the seat treated as open. The truth likely lies in between, depending on how much of the new district the member already represented.

| Race | Incumbent | PVI | With incumbency | As open seat |
|---|---|---|---|---|
{bound_rows}

Individual races move {bound_swing.min() * 100:.0f}-{bound_swing.max() * 100:.0f} points, but the shifts run in both directions and largely cancel: treating all {len(moved)} redrawn incumbents as open moves the expected Democratic total by {bound_net:+.1f} seats.

## 8. Does household stress explain where the polls miss?

The forecast applies the final generic ballot 1:1 as the national environment. If household costs or credit stress move voters in ways the polls miss, a measure of them should predict where the House vote lands relative to the polls. That is a national question, so it isn't limited to the race-level model's seven cycles: it can use every election with a final RealClearPolitics average, {nat_span} ({nat_n} elections). Earlier years have only single polls from different pollsters, which would mix a much noisier measurement in.

Two composites were fixed before either was compared with an election result (`household_index.py`), both measured June over June of the election year, which a September decision can see:

- **Cost basket:** the household basket from the HOUSING AFFORDABILITY TRACKER sheet, each item's price change weighted 1-3 as the sheet weights it, pulled from FRED. It covers {nt.basket_items.min()} items in {nt.basket_items.idxmin()} and {nt.basket_items.max()} by {nt.basket_items.idxmax()}, as series start.
- **Credit and labor stress:** the average change, in points, of the unemployment rate, U-6, and banks' credit card and mortgage delinquency rates.

The test (`national_test.py`) regresses the miss (House vote minus generic ballot, Democratic margin) on the polls' average lean, the president's party (+1 Democratic, -1 Republican), and the composite times the president's party. The hypothesis is that the president's party runs further behind its polls when household stress is high. A composite passes only if it beats the model without it both leaving one election out and forward in time (each year from 2012 predicted from the years before), and shuffling it across elections gives an effect at least as strong less than {PASS_P:.0%} of the time.

The same test also runs against the generic ballot on Sept 1, the polls a September decision actually sees: RealClearPolitics' own average that day, read from the daily series on its pages (2008-{nt.index.max()}). For 2002-2006, which have no average for that date, it uses the mean spread of each nonpartisan pollster's latest poll in the 30 days before. Where both exist, that rule lands within about a point of RealClearPolitics' own average. Every value and its source is in [`data/generic_ballot_dated.csv`](data/generic_ballot_dated.csv).

| Year | President's party | Generic ballot, Sept 1 | Generic ballot, final | House vote | Miss vs Sept 1 | Miss vs final | Cost basket | Credit and labor stress (pts) |
|---|---|---|---|---|---|---|---|---|
{nat_rows}

Prediction error of the miss (root mean square, points):

| Model | Final polls: leaving one out | Final polls: forward, 2012-{nt.index.max()} | Sept 1 polls: leaving one out | Sept 1 polls: forward |
|---|---|---|---|---|
{model_rows}

{verdict}. The basket's effect {direction(cb['coef']['c'])} but is small: {cb['coef']['c']:+.2f} points per point of basket inflation, about {cb['coef']['c'] * nt.basket[yb]:+.1f} in {yb}, and shuffling the basket across elections gives one at least as negative {cb['p']:.0%} of the time. Credit and labor stress {direction(cs['coef']['c'])} ({cs['coef']['c']:+.2f} per point; {cs['p']:.0%} of shuffles come out at least as negative). The two biggest shocks show why. In {yb} the basket rose {nt.basket[yb]:.1f}%, the most of any year here, and {against_polls(yb)}. In {ys} credit and labor stress rose the most ({nt.stress[ys]:+.1f} points), and {against_polls(ys)}.{lean_text}

![Household stress](figures/13_household_stress.png)

**Against the September polls the answer is the same.** {('Neither composite earns a place there either' if not sept_passed else 'The ' + ' and '.join(sept_passed) + (' earns' if len(sept_passed) == 1 else ' earn') + ' a place there')}: the basket's effect is {sb['coef']['c']:+.2f} per point ({sb['p']:.0%} of shuffles at least as negative) and stress's {ss['coef']['c']:+.2f} ({ss['p']:.0%}). The September polls miss by more than the final ones, {sept['constant']['loo']:.1f} points against {national['constant']['loo']:.1f} leaving one out, and they lean: Democrats ran behind their Sept 1 polls in {sept_over} of {nat_n} years, by {abs(nt.miss_sept.mean()):.1f} points on average. Most of that is gone by the final polls ({abs(nt.miss.mean()):.1f} points), so the polls tend to drift toward Republicans through the fall. Knowing which party holds the White House or whether it's a midterm doesn't predict how far, and neither do the composites (figure 14).

![Poll lean](figures/14_poll_lean.png)

**Can a ballot built from economics replace the polls?** Predicting the House vote itself (root mean square error, points):

| Prediction | Leaving one election out | Forward, 2012-{nt.index.max()} |
|---|---|---|
{econ_rows}

Leaving one election out, the best economics-only ballot misses by {best_econ:.1f} points, against {sept['constant']['loo']:.1f} for the September polls and {national['constant']['loo']:.1f} for the final ones. Forward in time, the version with no economic data at all, only which party holds the White House and whether it's a midterm, {vs_sept} the September polls ({econ['midterm']['forward']:.2f} against {sept['constant']['forward']:.2f}). Adding the cost basket or credit and labor stress makes it worse ({econ['basket']['forward']:.2f} and {econ['stress']['forward']:.2f}).

The final polls already move a little with both: the president's party's generic-ballot margin correlates {cb['gb_corr']:+.2f} with the basket and {cs['gb_corr']:+.2f} with stress across these years, so by November some of it is priced in.

The same {nat_n} elections give a second reading on the size of a national miss: a standard deviation of {nt.miss.std():.1f} points, against sigma_national = {m['sigma_national']:.1f} from the race-level model's {len(m['cycles'])} cycles. They measure slightly different things (the national vote against the poll, and the average miss in modeled races), but both put a typical national miss at two to three points.

What this doesn't settle: whether the same stress moves some districts more than others (district exposure from Census data times these composites), and whether the values as first published, rather than today's revised ones, would do better.

## 9. Limitations

- **The spending effect is assumed.** The allocation results depend on it, and public data can't pin it down. Section 5 adds two readings that need less of it: how strong it would have to be for reallocating to matter, and what DCCC's own allocation implies. Both still assume the same curve in every race, and that each party's money moves the margin independently of the other's.
- **Likely R seats are overrated**, at about {pooled[pooled.tier == 'Likely R'].p.mean():.0%} predicted against {(pooled[pooled.tier == 'Likely R'].margin > 0).mean():.0%} actual. Most of the gap is 2020 ({lr20.p.mean():.0%} predicted, {(lr20.margin > 0).mean():.0%} actual), the national polling miss again, rather than a problem in deep-red seats specifically.
- **The incumbency effect has shrunk.** Fit on {first}-2016 alone, having the Democrat rather than the Republican as the incumbent is worth {early['d_incumbent']:.0f} points; on 2018-{last}, {late['d_incumbent']:.0f}. The model pools every cycle and gives {fc['cycle']} incumbents {c['d_incumbent']:.0f} points, more than recent cycles alone would. Weighting recent cycles more was tested while building this repo and forecast worse forward in time: the fits got noisy and overconfident. The first-term flag captures the part of the decline that can be predicted.
- **The incumbency effect is learned from incumbents running in their own districts**, so it is probably too large for members on redrawn {fc['cycle']} maps (bounds in section 7).
- **{fc['cycle']} nominees are the top spenders reported so far** until official results name them, and their money counts through June 30, as in past cycles. Where the top spender had no House report through June 30 (a Senate run, a campaign that ended, a primary loser), the actual nominees were looked up by hand, with sources, in [`data/nominees_{fc['cycle']}.csv`](data/nominees_{fc['cycle']}.csv): {len(hand_checked)} districts, {', '.join(hand_checked)}. Other districts' top spenders were not checked one by one.{(' Still without a House report through June 30: ' + ', '.join(no_july.district_id) + '; that side counts as $0' + ('. ' + misjudged + (' are' if len(no_july_close) > 1 else ' is') + ' competitive and may be misjudged as a result.' if len(no_july_close) else '.')) if len(no_july) else ''}
- **The main forecast uses the final pre-election generic ballot.** Section 4 adds the forecast a Sept 1 decision could make, and section 5 redoes the safe-seat readings with it. DCCC spends from September to November, so neither is exactly what it saw for any one dollar. The 2026 forecast uses the latest average.
- **TV-market data starts with the 2022 maps**, so market costs cover the 2022 and 2024 backtests only. Channels come from filers' own purpose descriptions; some say only "MEDIA" and stay unspecified.
- **The national error is estimated from {len(m['cycles'])} elections.** The seat totals allow for that (a Student t rather than a normal), but each race's own probability, and so the allocation and the backtest scores, still treats its size as known.
"""
    (ROOT / "REPORT.md").write_text(text)
