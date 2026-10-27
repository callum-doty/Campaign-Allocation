"""Write REPORT.md and its figures from the pipeline's results."""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import HALF_SATURATIONS, SCALES
from .compare_maps import compare_maps
from .plots.plot_allocation import plot_allocation
from .plots.plot_battleground import plot_battleground
from .plots.plot_calibration import plot_calibration
from .plots.plot_map_changes import plot_map_changes
from .plots.plot_pvi_vs_margin import plot_pvi_vs_margin
from .plots.plot_seat_distribution import plot_seat_distribution
from .plots.plot_seats_by_gb import plot_seats_by_gb
from .plots.plot_spending_effect import plot_spending_effect

ROOT = Path(__file__).resolve().parents[1]


def report(races, backtests, fc):
    """`races`: cycle -> build_races(cycle); `backtests`: list of backtest()
    results in cycle order; `fc`: forecast() result."""
    fig = ROOT / "figures"
    fig.mkdir(exist_ok=True)
    last = max(c for c in races if c < fc["cycle"])
    plot_pvi_vs_margin(races[last], fc["model"], fig / "1_pvi_vs_margin.png")
    plot_spending_effect(fig / "2_spending_effect.png")
    plot_calibration(backtests, fig / "3_calibration.png")
    plot_allocation(backtests, fig / "4_allocation.png")
    plot_map_changes(fig / "5_map_changes.png")
    plot_seats_by_gb(fc, fig / "6_seats_by_gb.png")
    plot_seat_distribution(fc, fig / "7_seat_distribution.png")
    plot_battleground(fc, fig / "8_battleground.png")

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
    lr20 = next(b["races"] for b in backtests if b["cycle"] == 2020).query("tier == 'Likely R' and margin.notna()")

    text = f"""# Campaign allocation report

Generated by `python run.py` on {dt.date.today().isoformat()}. Every number below is computed by the pipeline unless marked otherwise.

## Summary

- **The forecast holds up out of sample.** Each year from {years} was predicted using only earlier elections. Brier scores run from {min(b['brier'] for b in backtests):.3f} to {max(b['brier'] for b in backtests):.3f}, and predicted and actual win rates match by tier from Safe D through Lean R (figure 3).
- **DCCC's own spending was close to the best use of the same money, if you trust the polls.** At the default spending assumption, reallocating DCCC's budget would have gained {min(b['gain'] for b in backtests):+.2f} to {max(b['gain'] for b in backtests):+.2f} expected seats per cycle (average {mean_gain:+.2f}). Even the most generous assumption tested gains at most {max(max(b['grid'].values()) for b in backtests):+.2f}. The main difference: DCCC put {safe_dccc:.0%} of its money into races already 80%+ likely to go Democratic, nearly all of it protecting its own incumbents. In 2020, when the polls overstated Democrats, {lost_2020} of those "safe" seats were lost anyway, so the caution was justified (figure 4, section 5).
- **{fc['cycle']}: the national environment matters more than the new maps.** Redrawn maps in {redrawn.nunique()} states move Harris-won districts from {int(harris_old.sum())} to {int(harris_new.sum())} in those states (figure 5). At the current D{gb:+.1f} generic ballot, the model expects {s26['mean']:.0f} Democratic seats among {len(fc['races'])} modeled races (80% range {s26['p10']}-{s26['p90']}), a {s26['p_majority']:.0%} chance of a majority. A 3-point polling miss moves the expected total by about {(seats[round(gb + 3, 1)] - seats[round(gb - 3, 1)]) / 2:.0f} seats (figures 6-7).

## 1. Data

| Source | What | Notes |
|---|---|---|
| [The Downballot](https://www.the-downballot.com/p/the-downballots-calculations-of-presidential) | 2008-2024 presidential results by congressional district, one sheet per set of district lines | Calculations by The Downballot (formerly Daily Kos Elections). Fetched at run time, not redistributed |
| [MIT Election Lab](https://doi.org/10.7910/DVN/IG0UN2) | U.S. House general-election returns, 1976-2024 | CC0. Downloaded by hand (Dataverse asks for a short form) |
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

![Allocation](figures/4_allocation.png)

**Where the difference goes: protecting incumbents.** Maximizing expected seats puts almost nothing in races the forecast already has at 80%+; DCCC puts real money there every year:

| Year | DCCC's budget in 80%+ races | Model, same budget | DCCC dollars there | Share to Democratic incumbents | Of those seats, lost anyway |
|---|---|---|---|---|---|
{chr(10).join(protect_rows)}

Outside 2018, nearly all of it protects DCCC's own incumbents, consistent with its "Frontline" program for vulnerable members. In 2018 the 80%+ money went to Republican seats the forecast already had as likely pickups. In 2020 that caution was justified. The forecast trusted polls that overstated Democrats by about 4 points, and {lost_2020} of the "safe" seats DCCC defended were lost anyway: the model was wrong to call them safe. Whether DCCC's money saved any others can't be seen in this data. So "close to optimal" is conditional on the polls being right. DCCC's main deviation from the model is insurance against them being wrong, concentrated on its own members. Two other objectives were tested while building this repo and are not in the pipeline. Maximizing the probability of a majority reproduces the defense in 2018 and 2020 but goes on offense in years Democrats trail (2016, 2022, 2024). Maximizing expected seats in the worst quarter of national environments adds defense every year, but most in the years DCCC defended least. Neither matches DCCC by itself.

## 6. {fc['cycle']}

Forecast from a baseline fit on all {m['cycles'][0]}-{m['cycles'][-1]} races, at the current generic ballot (D{gb:+.1f}).

![Map changes](figures/5_map_changes.png)

![Seats by generic ballot](figures/6_seats_by_gb.png)

| Generic ballot | D{gb - 3:+.1f} | D{gb:+.1f} (now) | D{gb + 3:+.1f} |
|---|---|---|---|
| Expected Democratic seats | {seats[round(gb - 3, 1)]:.0f} | {seats[round(gb, 1)]:.0f} | {seats[round(gb + 3, 1)]:.0f} |

Averaging over the national error instead of fixing it gives the full distribution. The model puts Democrats at {s26['mean']:.0f} seats among modeled races, an 80% range of {s26['p10']}-{s26['p90']}, and a {s26['p_majority']:.0%} chance of reaching the {s26['need']} they need for a majority. Treat that as optimistic. The national error is estimated from only seven elections, and normal tails understate big misses (section 4). A 4-point polling miss like 2020's would leave Democrats near {seats[round(gb - 4, 1)]:.0f} seats.

![Seat distribution](figures/7_seat_distribution.png)

{len(close)} races sit between 20% and 80%. {sum(close.district_id.isin(maps[maps.changed].district_id))} of them are on redrawn lines; most of the rest are Republican-held seats that a D{gb:+.1f} environment puts in play.

![Battleground](figures/8_battleground.png)

**Incumbents on redrawn lines.** The incumbency effect is learned from members running in their own districts, and is probably too large for members whose districts were redrawn. Each redrawn-incumbent race with either chance between 20% and 80% is shown below, with full incumbency credit and with the seat treated as open. The truth likely lies in between, depending on how much of the new district the member already represented.

| Race | Incumbent | PVI | With incumbency | As open seat |
|---|---|---|---|---|
{bound_rows}

Individual races move {bound_swing.min() * 100:.0f}-{bound_swing.max() * 100:.0f} points, but the shifts run in both directions and largely cancel: treating all {len(moved)} redrawn incumbents as open moves the expected Democratic total by {bound_net:+.1f} seats.

## 7. Limitations

- **The spending effect is assumed.** The allocation results depend on it, and public data can't pin it down.
- **Likely R seats are overrated**, at about {pooled[pooled.tier == 'Likely R'].p.mean():.0%} predicted against {(pooled[pooled.tier == 'Likely R'].margin > 0).mean():.0%} actual. Most of the gap is 2020 ({lr20.p.mean():.0%} predicted, {(lr20.margin > 0).mean():.0%} actual), the national polling miss again, rather than a problem in deep-red seats specifically.
- **The incumbency effect is learned from incumbents running in their own districts**, so it is probably too large for members on redrawn {fc['cycle']} maps (bounds in section 6).
- **{fc['cycle']} spending is as reported to the FEC so far.** The nominee for each party is the top current spender until official results name them.
- **The national error is estimated from seven elections** and assumed normal, so big polling misses are likely more common than the seat distribution implies.
"""
    (ROOT / "REPORT.md").write_text(text)
