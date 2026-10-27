# Campaign allocation

Where should a party committee spend its money across House races? This repo answers that in two separate steps and checks both against real elections:

1. **Forecast each race** from public data: partisan lean on the district's actual lines, incumbency, how much each side's candidates and outside groups are spending, and the national environment.
2. **Assume what party money does** (an explicit, adjustable curve with diminishing returns), then spend a budget where it buys the most expected seats and compare with what DCCC actually did.

**[Read the report](REPORT.md)** for the data, features, math, results and charts.

## Headline results

- **The forecast is calibrated out of sample.** Each year from 2016 to 2024 was predicted only from earlier elections, and predicted and actual win rates match by tier.
- **DCCC's own spending was within a few tenths of a seat** of the best allocation of the same budget at the default assumption, and under two seats even under generous ones. The main difference is money protecting its own incumbents in seats the polls called safe, a caution that was justified in 2020.
- **2026:** the new maps cost Democrats about four Harris-won districts, but a 3-point move in the national environment swings about a dozen expected seats. With polling error shared across races, the model gives Democrats a 98% chance of a majority, which is likely optimistic.

## Run it

```bash
pip install -r requirements.txt
python run.py
```

`run.py` downloads what's missing into `data/raw/` (about 170 MB, not committed), builds every cycle, runs the backtests and 2026 forecast, and rewrites `REPORT.md` and `figures/`. It takes under a minute once the data is downloaded. `python run.py --refresh` re-downloads the 2026 FEC files, which are still being filed.

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
  load_spending.py          independent expenditures (amendments deduplicated)
  build_races.py            one row per race with every feature
  fit_baseline.py           job 1: the forecast (least squares)
  predict_margin.py         expected margin under a fitted model
  spending_effect.py        job 2: assumed effect of party money
  allocate.py               greedy budget allocation
  seat_distribution.py      seat total with national error shared across races
  seats_needed.py           seats needed among modeled races for a majority
  backtest.py               score a past cycle, forward in time
  forecast.py               forecast an upcoming cycle
  compare_maps.py           which districts were redrawn
  report.py                 write REPORT.md
  plots/                    one chart per file
data/generic_ballot.csv     polling averages, with sources
tests/test_model.py
```

## Data and credit

- Presidential results by congressional district: **[The Downballot](https://www.the-downballot.com/p/the-downballots-calculations-of-presidential)** (formerly Daily Kos Elections). Their calculations are fetched at run time and not redistributed here. Please credit them if you use this work.
- House election returns: **MIT Election Data and Science Lab**, *U.S. House 1976-2024*, Harvard Dataverse, [doi:10.7910/DVN/IG0UN2](https://doi.org/10.7910/DVN/IG0UN2) (CC0).
- Campaign finance: **Federal Election Commission** [bulk data](https://www.fec.gov/data/browse-data/?tab=bulk-data).
- Generic-ballot averages: **RealClearPolitics** / RealClearPolling, compiled in [`data/generic_ballot.csv`](data/generic_ballot.csv).

## Caveats

- **The effect of party money is assumed, not measured.** Public data can't separate it from the reasons money goes where it goes. Treat the allocation results as "under this assumption", and replace it with measured effects (for example randomized holdouts on mail or ad programs) when you have them.
- **No 2026 allocation analysis.** After *NRSC v. FEC* (June 2026), DCCC spends through coordinated expenditures, which aren't in the bulk independent-expenditure file.
