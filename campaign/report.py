"""Write REPORT.md and its figures from the pipeline's results."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import HALF_SATURATIONS, SCALES
from .break_even import MAX_SCALE
from .channel_mix import BAND_LABELS
from .compare_maps import compare_maps
from .media_cost import GAMMAS
from .plots.plot_allocation import plot_allocation
from .plots.plot_battleground import plot_battleground
from .plots.plot_calibration import plot_calibration
from .plots.plot_channel_mix import plot_channel_mix
from .plots.plot_equilibrium import plot_equilibrium
from .plots.plot_map_changes import plot_map_changes
from .plots.plot_pvi_vs_margin import plot_pvi_vs_margin
from .plots.plot_seat_distribution import plot_seat_distribution
from .plots.plot_seats_by_gb import plot_seats_by_gb
from .plots.plot_spending_effect import plot_spending_effect
from .spending_effect import HALF_SATURATION, SCALE_PTS, spending_effect

ROOT = Path(__file__).resolve().parents[1]


def report(races, backtests, fc, games, media, mixes):
    """`races`: cycle -> build_races(cycle); `backtests`: backtest() results
    in cycle order; `fc`: forecast(); `games`: cycle -> solve_game();
    `media`: media_cost() results; `mixes`: channel_mix() results."""
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
        belief_rows.append(
            f"| {b['cycle']} | " + " / ".join(f"${1 / v:.0f}M" for v in d.per_million) + " | "
            + " | ".join(f"{d.multiplier[band]:.1f}x on ${d.dollars[band] / 1e6:.1f}M" for band in ("under 20%", "80%+"))
            + f" | {min(safe_k):.1f}-{max(safe_k):.1f}x |")
    safe_k = {b["cycle"]: b["belief"][HALF_SATURATION].multiplier["80%+"] for b in backtests}
    k_lo, k_hi = min(safe_k, key=safe_k.get), max(safe_k, key=safe_k.get)
    k_spread = max(np.ptp([b["belief"][h].multiplier["80%+"] for h in HALF_SATURATIONS]) for b in backtests)

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

    text = f"""# Campaign allocation report

Generated by `python run.py` on {dt.date.today().isoformat()}. Every number below is computed by the pipeline unless marked otherwise.

## Summary

- **The forecast holds up out of sample.** Each year from {years} was predicted using only earlier elections. Brier scores run from {min(b['brier'] for b in backtests):.3f} to {max(b['brier'] for b in backtests):.3f}, and predicted and actual win rates match by tier from Safe D through Lean R (figure 3).
- **DCCC's own spending was close to the best use of the same money, if you trust the polls.** At the default spending assumption, reallocating DCCC's budget would have gained {min(b['gain'] for b in backtests):+.2f} to {max(b['gain'] for b in backtests):+.2f} expected seats per cycle (average {mean_gain:+.2f}). Even the most generous assumption tested gains at most {max(max(b['grid'].values()) for b in backtests):+.2f}. The main difference: DCCC put {safe_dccc:.0%} of its money into races already 80%+ likely to go Democratic, nearly all of it protecting its own incumbents. In 2020, when the polls overstated Democrats, {lost_2020} of those "safe" seats were lost anyway, so the caution was justified (figure 4, section 5).
- **Two readings that don't assume how strong party money is.** Reallocating DCCC's budget gains a full seat only if its first $1M in a race moves the margin {min(first_needed):.1f}-{max(first_needed):.1f} points, against {first_default:.2f} assumed. And DCCC's own split is optimal only if a dollar in its 80%+ races works {min(safe_k.values()):.1f}-{max(safe_k.values()):.1f} times as hard as one in its 20-80% races, or those seats are worth that much more to it (section 5).
- **Neither strategy nor TV prices explain that defensive spending.** In a two-player game against NRCC, DCCC's best play puts at most {max_eq_safe:.0%} of its money in 80%+ seats (figure 5). DCCC already spends about half as much per race in the costliest TV markets, and pricing in market cost makes the model leave them almost entirely. No spender shifts toward mail where TV is most expensive (figure 6).
- **{fc['cycle']}: the national environment matters more than the new maps.** Redrawn maps in {redrawn.nunique()} states move Harris-won districts from {int(harris_old.sum())} to {int(harris_new.sum())} in those states (figure 7). At the current D{gb:+.1f} generic ballot, the model expects {s26['mean']:.0f} Democratic seats among {len(fc['races'])} modeled races (80% range {s26['p10']}-{s26['p90']}), a {s26['p_majority']:.0%} chance of a majority. A 3-point polling miss moves the expected total by about {(seats[round(gb + 3, 1)] - seats[round(gb - 3, 1)]) / 2:.0f} seats (figures 8-9).

## 1. Data

| Source | What | Notes |
|---|---|---|
| [The Downballot](https://www.the-downballot.com/p/the-downballots-calculations-of-presidential) | 2008-2024 presidential results by congressional district, one sheet per set of district lines | Calculations by The Downballot (formerly Daily Kos Elections). Fetched at run time, not redistributed |
| [MIT Election Lab](https://doi.org/10.7910/DVN/IG0UN2) | U.S. House general-election returns, 1976-2024 | CC0. Downloaded by hand (Dataverse asks for a short form) |
| [Jonathan Cervas](https://jonathancervas.com/maps/media-markets/) | Each district's population by Nielsen TV market, 2022-2026 lines | Derived by Cervas from Nielsen market boundaries and Census 2024 county estimates. Parsed from his page at run time, not redistributed |
| [FEC bulk data](https://www.fec.gov/data/browse-data/?tab=bulk-data) | All-candidate summaries (weball) and every reported independent expenditure, 2012-2026 | Public, no API key |
| [`data/generic_ballot.csv`](data/generic_ballot.csv) | Final RealClearPolitics generic-ballot average before each election | Compiled by hand, sources in the file |

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
| `floor_share` | log((F_D + $10K) / (F_D + F_R + $20K)), F = candidate + outside spending that side | Candidate strength and how hard each side contests the race, using only money neither national committee controls |
| `gb` | Generic-ballot average before the election | The national environment, applied as a uniform swing |

## 3. The model

The model does two separate jobs.

**Job 1: the baseline forecast** (fit by least squares on contested races, {m['cycles'][0]}-{m['cycles'][-1]}, n = {m['n']}):

```
margin_i - GB = a + b*PVI_i + c*D_incumbent_i + d*Open_i + e*floor_share_i + error_i,   error_i ~ Normal(0, sigma^2)
P(Democrat wins race i) = Phi(expected margin_i / sigma)
```

| a (intercept) | b (PVI) | c (D incumbent) | d (open seat) | e (floor share) | sigma |
|---|---|---|---|---|---|
| {c['intercept']:+.2f} | {c['pvi']:.2f} | {c['d_incumbent']:+.2f} | {c['open_seat']:+.2f} | {c['floor_share']:.2f} | {m['sigma']:.2f} |

Each point of PVI is worth {c['pvi']:.1f} points of margin. Having the Democrat rather than the Republican as the incumbent is worth {c['d_incumbent']:.0f} points. The generic ballot is fixed at a 1:1 pass-through rather than estimated: with one national value per cycle, a fitted slope would mostly chase polling misses. sigma is the residual spread, so it includes those misses; that is the right uncertainty for a forecast made before the votes are counted.

**National and district error.** Part of every race's error is shared: in a year the polls miss, they miss everywhere at once. Splitting it out, sigma^2 = sigma_national^2 + sigma_district^2, with sigma_national = {m['sigma_national']:.1f} points (the spread of each cycle's average miss) and sigma_district = {m['sigma_district']:.1f}. A single race's probability needs only the total. The seat total needs the split: conditional on the national miss, races are independent, and averaging over the miss gives the seat distribution in section 6.

![PVI vs margin](figures/1_pvi_vs_margin.png)

**Job 2: what party money does.** This is an assumption, not an estimate:

```
shift(x) = scale * log(1 + x / half_saturation),   default scale = 1.5 points, half_saturation = $5.5M
P(Democrat wins i | allocation x) = Phi((expected margin_i + shift(x_i) - shift(observed_i)) / sigma)
```

Public data can't separate the effect of money from the reasons money goes where it goes. Close races and strong candidates attract it. Returns diminish in absolute dollars, not in spending share; a share-based curve makes the first dollar in a quiet long-shot race look ten times as powerful as a dollar in a toss-up. The baseline already reflects the money actually spent, so only deviations from DCCC's observed allocation move a race. Results are shown across a range of both parameters; measured effects (for example randomized holdouts on mail programs) should replace the defaults.

![Spending effect](figures/2_spending_effect.png)

**Allocation.** DCCC's budget is spent greedily: each step funds the race where the next chunk buys the most expected seats, with no race taking more than 15% of the budget. This is exact, not a shortcut. Each race's win probability is concave in its spending unless the race is more than about 40 points out (z < -sigma/scale), so the objective is a sum of concave functions under one budget, and small greedy steps reach the global optimum. A numerical solver lands within 0.01 seats of it. Splitting the error doesn't change this allocation either, because averaging over the national miss gives back exactly the pooled sigma.

## 4. Does the baseline forecast well?

Each year is forecast from a model fit only on earlier years.

| Year | Brier score | Expected D seats | Actual D seats | DCCC budget | Gain from reallocating (default) | Gain range (9 assumptions) |
|---|---|---|---|---|---|---|
{bt_rows}

Seats are counted among modeled races, so Alaska and races with under $100K of spending are left out. The biggest miss is 2020, when polls overstated Democrats by about 4 points nationally; a polling miss shifts every race at once.

![Calibration](figures/3_calibration.png)

**The seat total, with the national error shared across races:**

| Year | Expected D seats | 80% range | Actual | Needed for majority | P(D majority) | Majority won by |
|---|---|---|---|---|---|---|
{chr(10).join(seat_rows)}

The 80% range held the actual result in {covered} of {len(backtests)} years, and the favorite for the majority won in {calls_right} of {len(backtests)}. Outside the range: {missed_text}. ("Needed" counts only modeled races; seats outside the model are credited by their partisan lean.)

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

Outside 2018, nearly all of it protects DCCC's own incumbents, consistent with its "Frontline" program for vulnerable members. In 2018 the 80%+ money went to Republican seats the forecast already had as likely pickups. In 2020 that caution was justified. The forecast trusted polls that overstated Democrats by about 4 points, and {lost_2020} of the "safe" seats DCCC defended were lost anyway: the model was wrong to call them safe. Whether DCCC's money saved any others can't be seen in this data. So "close to optimal" is conditional on the polls being right. DCCC's main deviation from the model is insurance against them being wrong, concentrated on its own members. Two other objectives were tested while building this repo and are not in the pipeline. Maximizing the probability of a majority reproduces the defense in 2018 and 2020 but goes on offense in years Democrats trail (2016, 2022, 2024). Maximizing expected seats in the worst quarter of national environments adds defense every year, but most in the years DCCC defended least. Neither matches DCCC by itself.

**What would DCCC have to believe?** The model's allocation stops where the last dollar in every funded race buys the same expected seats (apart from races at the 15% cap). DCCC's own allocation can be read backwards against that condition: at DCCC's actual spending, what does one more expected seat cost at the margin in each band, and how much harder would a dollar have to work in the costlier bands for DCCC's split to be optimal?

| Year | Cost of one more expected seat at the margin (default assumption): under 20% / 20-80% / 80%+ | Multiplier needed: under-20% races | Multiplier needed: 80%+ races | 80%+ multiplier across half-saturations |
|---|---|---|---|---|
{chr(10).join(belief_rows)}

The multiplier is how many times as far a dollar would have to move the margin in that band, compared with DCCC's 20-80% races, for its split to be optimal. The same number also reads as how many times as much those seats would have to be worth to DCCC, or as a sign that the forecast understates their risk. This data can't tell the three apart. In 80%+ races it runs from {safe_k[k_lo]:.1f}x ({k_lo}) to {safe_k[k_hi]:.1f}x ({k_hi}). Unlike the gain, it doesn't depend on how strong party money is assumed to be: the strength cancels in the ratio. Moving the half-saturation from $1.65M to $16.5M changes no year's 80%+ multiplier by more than {k_spread:.1f}. The costs in the first column do scale with the assumed strength; only their ratios are free of it.

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

Averaging over the national error instead of fixing it gives the full distribution. The model puts Democrats at {s26['mean']:.0f} seats among modeled races, an 80% range of {s26['p10']}-{s26['p90']}, and a {s26['p_majority']:.0%} chance of reaching the {s26['need']} they need for a majority. Treat that as optimistic. The national error is estimated from only seven elections, and normal tails understate big misses (section 4). A 4-point polling miss like 2020's would leave Democrats near {seats[round(gb - 4, 1)]:.0f} seats.

![Seat distribution](figures/9_seat_distribution.png)

{len(close)} races sit between 20% and 80%. {sum(close.district_id.isin(maps[maps.changed].district_id))} of them are on redrawn lines; most of the rest are Republican-held seats that a D{gb:+.1f} environment puts in play.

![Battleground](figures/10_battleground.png)

**Incumbents on redrawn lines.** The incumbency effect is learned from members running in their own districts, and is probably too large for members whose districts were redrawn. Each redrawn-incumbent race with either chance between 20% and 80% is shown below, with full incumbency credit and with the seat treated as open. The truth likely lies in between, depending on how much of the new district the member already represented.

| Race | Incumbent | PVI | With incumbency | As open seat |
|---|---|---|---|---|
{bound_rows}

Individual races move {bound_swing.min() * 100:.0f}-{bound_swing.max() * 100:.0f} points, but the shifts run in both directions and largely cancel: treating all {len(moved)} redrawn incumbents as open moves the expected Democratic total by {bound_net:+.1f} seats.

## 8. Limitations

- **The spending effect is assumed.** The allocation results depend on it, and public data can't pin it down. Section 5 adds two readings that need less of it: how strong it would have to be for reallocating to matter, and what DCCC's own allocation implies. Both still assume the same curve in every race, and that each party's money moves the margin independently of the other's.
- **Likely R seats are overrated**, at about {pooled[pooled.tier == 'Likely R'].p.mean():.0%} predicted against {(pooled[pooled.tier == 'Likely R'].margin > 0).mean():.0%} actual. Most of the gap is 2020 ({lr20.p.mean():.0%} predicted, {(lr20.margin > 0).mean():.0%} actual), the national polling miss again, rather than a problem in deep-red seats specifically.
- **The incumbency effect is learned from incumbents running in their own districts**, so it is probably too large for members on redrawn {fc['cycle']} maps (bounds in section 6).
- **{fc['cycle']} spending is as reported to the FEC so far.** The nominee for each party is the top current spender until official results name them.
- **TV-market data starts with the 2022 maps**, so market costs cover the 2022 and 2024 backtests only. Channels come from filers' own purpose descriptions; some say only "MEDIA" and stay unspecified.
- **The national error is estimated from seven elections** and assumed normal, so big polling misses are likely more common than the seat distribution implies.
"""
    (ROOT / "REPORT.md").write_text(text)
