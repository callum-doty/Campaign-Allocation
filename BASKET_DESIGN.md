# Basket search: the design, before scoring

Written November 4, 2026, the day after the election. This file and [`campaign/basket_search.py`](campaign/basket_search.py) fix a combinatoric search over what a Sept 1 forecaster could see, and a test of what committees' choices of seats say. The commit that adds them comes before any real combination is scored.

[SPENDING.md](SPENDING.md) tested 28 features one at a time, and only three of them were hand-picked pairs of channels. This asks the wider question once, with the rules written down first.

## What has been seen so far

- **The earlier results.** Every result in REPORT.md and SPENDING.md, on the same residuals. The item pool below can't be blind to those, so it takes every measure instead of the ones that looked good.
- **Blind checks.** Everything in "What these tests can detect" comes from whole state-years with flipped signs, shuffled items or planted effects. `run()` was dry-run once on made-up margins and shuffled items. No real combination's statistic, real run of the trees or real miss test has been computed.
- **Committee counts.** How many seats each committee entered, which uses no result.
- **The election.** Polls have closed, and whoever runs this may know headline results. There are no 2026 margins in the repo, and `run()` stops if it finds any. This is not a pre-election freeze like `data/forecast_freeze_2026_v2.csv`, and shouldn't be described as one.

## The three tests

| Test | Question | What it's good at finding |
|---|---|---|
| 1. Trees | Does any flexible combination of the items improve forecasts? | Many weak effects together |
| 2. Baskets | Which single combination adds the most beyond its items? | One large effect |
| 3. Committee | Do public signs of committee interest predict misses in seats no committee entered? | Whether committees' signal is public, apart from their dollars |

## Common to tests 1 and 2

The forward backtest from SPENDING.md section 2. Each year from 2016 to 2024 is forecast from earlier cycles, by the baseline plus the June 30 surplus. A race's miss is its actual margin minus the forecast, with the year's average miss removed, signed toward Democrats. Only contested races count, those decided by under 80 points. That's 1,927 races, 267 of which the baseline put within 10 points, in 240 state-years, with a miss SD of 6.75 points.

**Items.** 51, all known by Sept 1. Each campaign has 20 measures through June 30, and each gives two items: high is the cycle's top quartile among nominees and low is its bottom quartile.

| Group | Measures |
|---|---|
| Money | raised, raised minus spent, spent as a share of raised, months of cash left, debts as a share of raised, fundraising cost as a share of raised |
| Share of spending | TV, digital, mail, field, polling, fundraising, payroll and overhead, April-June |
| Vendors | concentration, kept from last cycle |
| Donors | itemized donors, share from last cycle's donors, share from the state, share from the top 10 |

Then nine flags for one side (incumbent, first term, challenger, experienced nominee, experienced opponent, first-term opponent, opponent's money high, opponent's money low, an incumbent who kept under 60% of the district in a redraw) and two for the race (open seat, redrawn state).

- **Quartiles, not fixed cutoffs.** A fixed cutoff can mostly mark the era: digital over 10% of early spending goes from 3% of campaigns in 2012 to 41% in 2022, while the incumbency edge fell by more than half.
- **Missing stays missing.** A measure that can't be computed makes a campaign neither high nor low. Spending shares and vendor measures need $100K spent through June 30, the cutoff in SPENDING.md section 3.
- **Some lows mean none.** Low TV means no TV by June, for 49% of 2024 campaigns. Low debt means no debt, for 57%.
- **Items belong to a side.** A combination counts +1 where only the Democrat has it, -1 where only the Republican does, and 0 where both or neither do. `{incumbent}` can't cancel itself, and no race counts twice.

## Test 1: trees

Gradient-boosted trees (`TREES` in the code: depth 3, 100 trees, learning rate 0.02, 80% subsample, no leaf under 100 rows) on the items. A depth-3 tree on yes-or-no items is a sum of combinations of up to three, the same family as test 2.

- **Trained on:** each earlier cycle's contested races, on what a surplus forecast fit to those same cycles missed, with cycle-by-seat-type means removed. Each race is seen from both sides, with the miss flipped for the Republican's view.
- **Read as:** half the difference between the two sides' predictions, added to the forecast. Swapping the sides flips the shift, so it's party-blind.
- **The bar:** the one in `spending_signal.py`, against the surplus forecast. Better with the year's miss removed in at least 4 of 5 years, better on raw log loss on average, and not worse in races the baseline put within 10.
- **The placebo:** 200 runs with whole rows of items shuffled among each year's races. Shuffled data clears the bar alone 11% of the time, so **clearing also requires at most 5% of shuffles to match it on log loss with the miss removed**.
- **Tally:** this is the 29th forecast feature tested, and should be reported as one.

## Test 2: baskets

Every pair and triple of items, with these rules:

- **Support:** at least 50 races, 15 of them ones the baseline put within 10 points, across at least 4 of 5 cycles. A race counts when exactly one side has the combination.
- **Not a copy:** at least 5% of its variance has to remain after its single items are taken out.
- **Statistic:** what the combination adds to the miss beyond its single items and cycle-by-seat-type means, divided by its own standard error, clustered by state-year.
- **Bar:** the 95th percentile of the largest absolute statistic across all baskets, over 2,000 draws flipping the sign of whole state-years at random. The bar is **4.04**, or 5% family-wise. Flips keep whatever error races in a state share, and center every basket on zero.
- **Passes:** clears the bar, with the same sign in at least 4 years.
- **Named:** a passing pair, or a passing triple when none of its pairs passes and it beats them all. A triple can't get credit for what its pair already does.

That leaves 3,961 baskets scored: 735 pairs and 3,226 triples, typically covering 101 races. **Four items can't appear in any of them, because they rarely reach close races: low raised, low surplus, opponent's money low and low retention.** The trees still see them.

A named basket is a candidate, not a finding. All five years went into choosing it, so its backtest isn't independent, and only 2026 can test it.

## Test 3: committee

**Step 1 uses no results, so it can be read freely.** Fit a logistic model (ridge, `C = 1`, standardized) of DCCC entering a seat, across contested seats the baseline gave Democrats at least 80%, 2016-2024. That's 861 seats, of which DCCC entered 47 and 809 had no DCCC or NRCC money.

- **Flags, fixed here:** first term, incumbent, experienced opponent, opponent's money, own June 30 surplus, own share of spending on polling, redrawn state, spent as a share of raised, share from last cycle's donors.
- **Control:** how safe the seat is, the forecast margin. It stays in the fit and out of the score, so a high score means the flags, not closeness.
- **Scores:** each year's come from a model fit on the other four.
- **Reported:** pooled area under the ROC curve for how safe the seat is alone, that plus first term, and all flags.

**Step 2 is one test, run once.** Among the 809 seats with no committee money: does a higher score go with running further behind the forecast?

- **Estimate:** the slope of the miss on the score, taking out cycle-by-seat-type means and how safe the seat is.
- **Passes:** a negative sign, p under 0.05 from 10,000 flips of whole state-years, and negative in at least 4 of 5 years.
- **Also shown:** the average miss in the top tenth of scores (about 81 seats), the rest, and seats with committee money.

**Robustness, reported but not part of passing:** the Sept 1 forecast to define safe seats, the surplus forecast instead of the baseline, no outside money either, without 2020, and the NRCC mirror (11 entries, so descriptive).

## How the outcomes read

| Outcome | Reading |
|---|---|
| Trees clear, no basket named | Combinations help, but no single one is large. A candidate for the forecast only after 2026. |
| Basket named, trees don't clear | One real-looking archetype, too rare to help forecasts overall. Worth describing and testing in 2026. |
| Neither | Nothing as large as the table below shows. SPENDING.md can then say combinations were searched, not only single features. It can't say interactions don't exist. |
| Miss test passes | Public data flags risk the forecast misses, even where no committee spent. Part of the safe-seat spending is then the forecast's error, and the score is a candidate feature. |
| Miss test fails, flags predict entry | The public pattern in committee choices doesn't carry a signal about misses. Committee interest is then habit or private information, which this can't tell apart. |
| Flags don't predict entry | The score is noise, so the miss test says little either way. |

No outcome here changes the forecast or either freeze. Adopting anything would need the 2026 test first.

## The 2026 test

`run()` writes two records with no 2026 result in the data:

- [`data/basket_candidate_2026.csv`](data/basket_candidate_2026.csv): the trees' shift on the surplus forecast, written whether or not they clear.
- [`data/basket_entry_2026.csv`](data/basket_entry_2026.csv): the entry score from all of 2016-2024, on the frozen baseline's safe seats with both nominees. That's 150 seats, 22 with DCCC coordinated money through Oct 14.

The entry score can be checked right away, since entry isn't a result. Once results are in:

- **Trees:** change in log loss against the surplus forecast, with the miss removed and raw.
- **Named baskets:** sign and size in 2026.
- **Miss test:** the same test on 2026's safe seats with no committee money, using the frozen score.

One year is a weak test. A basket with 100 races across five years has about 20 in 2026, for a standard error near 1.5 points.

## What these tests can detect

From blind runs only. Baskets are planted on flipped data and trees on shuffled items.

**Baskets:** share of the time one basket with a planted effect clears the bar.

| Races in basket | Smallest effect that can clear | 2 points | 3 points | 4 points | 6 points |
|---|---|---|---|---|---|
| 50-99 | 3.4 points | 1% | 8% | 33% | 94% |
| 100-199 | 2.7 points | 5% | 38% | 86% | 100% |
| 200+ | 2.2 points | 15% | 71% | 99% | 100% |

For scale, challenger quality, the best feature added so far, is 1.8 points (se 0.5). Most real effects of that size would be missed, so a null here is not evidence that none exist.

**Trees:** share of the time they clear both the bar and the placebo. With shuffled items, the 5th percentile of the log loss change is -0.00040.

| Planted | Clears | Log loss change, miss removed |
|---|---|---|
| 1 basket at 6 points | 31% | -0.00076 |
| 10 baskets at 3 points | 72% | -0.00146 |
| 25 baskets at 1.5 points | 57% | -0.00084 |
| 25 baskets at 3 points | 91% | -0.00331 |

The two complement each other. Trees often catch 25 baskets of 1.5 points each, smaller than the search can usually name alone. The basket search catches one large basket that the trees usually miss.

**Miss test:** 27% at 0.5 points per SD of the score, 81% at 1, 100% at 1.5.

## What changed during the blind checks

All of these were decided on flipped, shuffled or planted data, before commit:

- **Scoring against single items, not all parts.** Taking a triple's pairs out too doubled its standard error (1.3 against 0.7 points at 100-199 races) and made it need over 5 points. The rule against coattails replaced it.
- **A clustered statistic.** It's the honest form, though it changes little here. With an unclustered standard error and the same flips the bar is 4.22, against 4.04, and the two errors match at the median (the clustered one is 0.89-1.16 times the other, 5th to 95th percentile). An earlier version scored raw means, without taking out single items, and its bar was 4.8.
- **Tree settings.** Four were tried on planted data. They scored within noise of each other, so the choice rests on three points: a learning rate of 0.02 costs less on shuffled data than 0.05, 0.01 shrinks planted gains further, and leaves can't be smaller than test 2's 50 races.
- **More baskets than first sized.** An early sizing on Democrat-minus-Republican quartiles found 211 baskets. Counting items by side gives 3,961, because each race has two chances to hold a combination.

## Limits

- **Selection before the search.** Max-T corrects for choosing among these baskets, not for the choice of measures, which came from earlier work on the same elections.
- **After the election.** The rules are fixed before any scoring, but 2026 isn't as clean as the earlier freezes. 2028 would be.
- **Seats committees skipped.** They may have polled them and found them safe, which pulls a real signal toward zero. A null in the miss test is weak evidence.
- **Scores use later years.** Each year's entry score is fit on the other four, including later ones. Only 2026 is scored forward.
- **2026's entries differ.** They're coordinated spending through Oct 14, and the model is fit on independent spending over whole cycles.
- **Few entries.** 47 entries for ten terms, 31 of them in 2020 and 2022. The penalty helps, but individual flags can't be read closely.
- **2016's trees** train on two cycles.
- **Purpose labels** are wrong on about 11-16% of payments, which blurs the spending items.

## To run it

Blind checks, which reproduce the numbers above:

```bash
python -m campaign.basket_search --blind
```

The real run, which writes `data/basket_results.json` and the two 2026 records, and refuses to run twice:

```bash
python -m campaign.basket_search
```

Both need the data `spending_report` uses, including the donor files.
