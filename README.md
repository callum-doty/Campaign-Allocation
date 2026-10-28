# Campaign allocation

Where should a party committee spend its money across House races? This repo answers that in two separate steps and checks both against real elections:

1. **Forecast each race** from public data: partisan lean on the district's actual lines, incumbency (with first-term members flagged), how the seat's party ran last time, how much each side's candidate had raised from others by June 30 and outside groups had spent by the end of August (what a September decision can see), and the national environment.
2. **Assume what party money does** (an explicit, adjustable curve with diminishing returns), then spend a budget where it buys the most expected seats and compare with what DCCC actually did.

**[Read the report](REPORT.md)** for the data, features, math, results and charts.

## Headline results

- **The forecast is calibrated out of sample.** Each year from 2016 to 2024 was predicted only from earlier elections, and predicted and actual win rates match by tier.
- **It only uses money a September decision could see.** Candidate money counts through June 30 and outside money through August 31. An earlier version counted each side's spending over the whole election, most of it in the fall; that hindsight made close races look easier to call than they were (log loss 0.407 against 0.416 on races the September forecast put between 5% and 95%).
- **DCCC's own spending was within a few tenths of a seat** of the best allocation of the same budget at the default assumption, and under two seats even under generous ones. The main difference is money protecting its own incumbents in seats the forecast called safe, a caution that was justified in 2020.
- **Part of 2020's "safe" money wasn't over-caution.** Incumbency is worth less than half what it was before 2018, and first-term members elected in a wave run further behind still. Flagging them and carrying over each seat's last result improves the forecast out of sample (log loss 0.095 → 0.088), and cuts the share of DCCC's 2020 money the forecast calls safe from 37% to 32%.
- **Turned around, the answer needs less of the spending assumption.** Reallocating would gain a full seat only if party money were 5 to 10 times as strong as assumed (DCCC's first $1M in a race moving it 1.2-2.5 points, not 0.25). DCCC's own split is optimal only if a dollar in races the forecast already gave Democrats at 80%+ works 1.5 to 4.2 times as hard as one in close races, a ratio that doesn't depend on how strong money is.
- **Neither strategy nor TV prices explain that.** In a two-player game against NRCC, the equilibrium puts almost nothing in safe seats. DCCC already spends about half as much per race in the costliest TV markets. No spender shifts toward mail where TV is most expensive.
- **Judged with what a Sept 1 decision could see, part of DCCC's safe-seat spending stops looking over-cautious.** A forecast built from the Sept 1 polls plus the usual fall drift (about 2 points toward Republicans, with its spread as extra uncertainty) is about as accurate on average as the final-poll forecast (log loss 0.087 against 0.088, though only 2020 is better) and its 80% seat ranges hold the result in all 5 backtest years, 2020 included. Under it, DCCC's share of money in 80%+ seats falls from 22% to 17% (2020: 32% to 18%). Judging each dollar instead by the forecast on the date it reached voters (checkpoints from Sept 1 to Oct 25, each with that day's polls and the outside money out by then), the share is 20%: later in the fall the polls and outside money had mostly moved toward what the final polls say. What's left still implies a dollar in a safe seat works 1.4-3.0 times as hard as one in a close race. Through the fall the national uncertainty shrinks, but the race forecasts don't get sharper.
- **If persuasion fades in days, when DCCC spends matters more than where.** TV-ad experiments find persuasion has a half-life of a few days, and many voters cast ballots weeks before Election Day (each state's share from the federal Election Administration and Voting Survey). Counting each dollar by how much of its effect is left when voters vote (4-day half-life), DCCC's money landed at 23-43% of what the same dollars could count over each state's best two weeks. Re-timing within the same races would gain 0.4-1.4 expected seats per cycle, against 0.06-0.19 for moving money between races (0.3-0.9 at an 8-day half-life), before the higher price of late airtime. Timing doesn't explain the safe-seat spending.
- **Household stress doesn't explain where the polls miss.** Over the 12 elections with a final RealClearPolitics generic-ballot average (2002-2024), neither the household cost basket from the HOUSING AFFORDABILITY TRACKER (its own items and weights, pulled from FRED) nor a credit-and-labor stress index predicts where the House vote lands relative to the polls. Both were fixed before being tested, and both make the prediction worse, leaving one election out and forward in time. In 2022, with the basket up 14.5%, the president's party ran only 0.3 points behind its polls. The same holds against the polls on Sept 1, when the allocation decision is made. A ballot built from economics alone misses the House vote by 5.3-6.7 points leaving one election out, against 3.4 for the September polls and 2.4 for the final ones. The polls' own lean matters more: Democrats ran behind their Sept 1 polls in 9 of 12 years, by 3.0 points on average, under presidents of both parties.
- **2026:** the new maps cost Democrats about four Harris-won districts, but a 3-point move in the national environment swings about a dozen expected seats. With polling error shared nationally and within each state, and the size of a national miss treated as uncertain (it rests on seven elections), the model gives Democrats a 97% chance of a majority (98% if that size were known), which is still likely optimistic.

## Run it

```bash
pip install -r requirements.txt
python run.py
```

`run.py` downloads what's missing into `data/raw/` (about 170 MB, not committed), builds every cycle, runs the backtests and 2026 forecast, and rewrites `REPORT.md` and `figures/`. It takes about two and a half minutes once the data is downloaded, most of it solving the two-player game and finding the break-even spending effect. `python run.py --refresh` re-downloads the 2026 FEC files, which are still being filed.

**A free FEC API key:** money raised through June 30 comes from the FEC's API, which needs a key from [api.data.gov/signup](https://api.data.gov/signup/). Set it as `FEC_API_KEY` before the first run.

**One manual step:** MIT's Dataverse asks for a short guestbook form before the House results download. Get `1976-2024-house.tab` from [doi.org/10.7910/DVN/IG0UN2](https://doi.org/10.7910/DVN/IG0UN2) and save it as `data/raw/house_1976_2024.csv`. `run.py` stops with these instructions if the file is missing.

Tests (synthetic data, about a second): `pytest`

## Layout

One function per file, in pipeline order:

```
run.py                      the whole pipeline
campaign/
  fetch.py                  download raw data
  load_presidential.py      presidential results by district, one map
  load_pvi.py               partisan lean on each election's own lines
  load_results.py           House results (fusion lines, state party labels)
  load_candidates.py        FEC nominees matched to the ballot
  load_early_money.py       each campaign's money through June 30 (FEC July report)
  load_expenditures.py      every independent expenditure, cleaned, with its channel
  load_spending.py          independent expenditures totalled by district
  load_media_markets.py     each district's TV market (Cervas)
  build_races.py            one row per race with every feature
  incumbent_history.py      first-term incumbents, last cycle's overperformance
  fit_baseline.py           job 1: the forecast (least squares)
  predict_margin.py         expected margin under a fitted model
  spending_effect.py        job 2: assumed effect of party money
  allocate.py               greedy budget allocation
  reallocation_gain.py      seats gained by re-spending the observed budget
  break_even.py             how strong money must be for reallocating to gain a seat
  implied_belief.py         what DCCC's own allocation implies, band by band
  seat_distribution.py      seat total with national error shared across races
  seats_needed.py           seats needed among modeled races for a majority
  backtest.py               score a past cycle, forward in time
  forecast.py               forecast an upcoming cycle
  solve_game.py             DCCC vs NRCC as a two-player game
  media_cost.py             allocation with TV-market cost priced in
  channel_mix.py            TV / digital / mail shares by market cost
  compare_maps.py           which districts were redrawn
  fall_drift.py             how the generic ballot moves from a date in the fall to Election Day
  dated_forecast.py         the forecast as it stood on a date of the campaign
  load_early_vote.py        each state's share of ballots cast before Election Day (EAVS)
  ballot_timing.py          each dollar counted by how much persuasion is left when voters vote
  load_fred.py              one FRED series
  household_index.py        household cost basket and credit/labor stress, June over June
  national_vote.py          national House popular vote
  national_test.py          does household stress explain the national polling miss?
  report.py                 write REPORT.md
  plots/                    one chart per file
data/generic_ballot.csv     final polling averages, 2002-2026, with sources
data/generic_ballot_dated.csv  polling averages on five dates each fall, 2002-2024, with sources
data/nominees_2026.csv      2026 nominees checked by hand where the top-spender guess fails, with sources
tests/test_model.py
```

## Data and credit

- Presidential results by congressional district: **[The Downballot](https://www.the-downballot.com/p/the-downballots-calculations-of-presidential)** (formerly Daily Kos Elections). Their calculations are fetched at run time and not redistributed here. Please credit them if you use this work.
- House election returns: **MIT Election Data and Science Lab**, *U.S. House 1976-2024*, Harvard Dataverse, [doi:10.7910/DVN/IG0UN2](https://doi.org/10.7910/DVN/IG0UN2) (CC0).
- Campaign finance: **Federal Election Commission** [bulk data](https://www.fec.gov/data/browse-data/?tab=bulk-data) and [API](https://api.open.fec.gov/developers/).
- TV markets by congressional district: **[Jonathan Cervas](https://jonathancervas.com/maps/media-markets/)**, derived from Nielsen market boundaries and Census county estimates. Parsed from his page at run time and not redistributed here.
- 2026 nominees in districts where FEC data alone picks the wrong candidate: checked by hand against Wikipedia's 2026 House election pages and news reports, one source per row in [`data/nominees_2026.csv`](data/nominees_2026.csv).
- Generic-ballot averages: **RealClearPolitics** / RealClearPolling, compiled in [`data/generic_ballot.csv`](data/generic_ballot.csv) (final averages, from 2002) and [`data/generic_ballot_dated.csv`](data/generic_ballot_dated.csv) (the average on five dates each fall, read from RealClearPolitics' daily series for 2008-2024; for 2002-2006, which have none, each nonpartisan pollster's latest poll as RealClearPolitics lists them).
- How each state's ballots were cast, 2016-2024: **U.S. Election Assistance Commission**, Election Administration and Voting Survey ([datasets](https://www.eac.gov/research-and-data/datasets-codebooks-and-surveys)), downloaded at run time.
- Household prices and credit/labor series: **FRED**, Federal Reserve Bank of St. Louis ([fred.stlouisfed.org](https://fred.stlouisfed.org)), no key needed. The basket's items and 1-3 weights come from the HOUSING AFFORDABILITY TRACKER sheet; each item's FRED series is the one matching that sheet's history (`household_index.py`).

## Caveats

- **The effect of party money is assumed, not measured.** Public data can't separate it from the reasons money goes where it goes. Treat the allocation results as "under this assumption", and replace it with measured effects (for example randomized holdouts on mail or ad programs) when you have them. The break-even effect in the report is the number to compare a measured effect against. Both it and the implied-belief ratios still assume the same curve in every race.
- **How fast persuasion fades is assumed**, from TV-ad experiments (half-life 4 days, with 2 and 8 shown), and early ballots are spread evenly over the last three weeks. The re-timing gain ignores the higher price of late airtime and any lasting effect of money, so it is an upper bound under that assumption.
- **The main forecast uses the final pre-election generic ballot.** The report adds the forecast as it stood on five dates from Sept 1 to Oct 25 (that day's polls plus the usual drift to Election Day, and the outside money out by then) and redoes the safe-seat readings with it, every dollar at Sept 1 and each dollar at the date it was spent. Candidates' own money still counts only through June 30, since later FEC reports aren't downloaded.
- **No 2026 allocation analysis.** After *NRSC v. FEC* (June 2026), DCCC spends through coordinated expenditures, which aren't in the bulk independent-expenditure file.
