# Everything tested

Every idea tested in this repo, whether it was kept or not: the data it used, what it found, and where it's written up. It covers work from October 26 to November 4, 2026, and not the earlier version of this project.

With five backtest years, enough tries will turn up something that looks real by luck. Listing every try, failures included, is how a reader can judge the few that passed.

**Status:**

- **Adopted:** in the forecast or the main analysis.
- **Alternative:** frozen as a separate 2026 forecast, not used in the main one.
- **Candidate:** passed narrowly. Written down before 2026 results so the election can test it, but not a finding.
- **Lead:** a pattern seen only after looking at these same years. 2026 predictions are written down.
- **Rejected:** failed its test, or doesn't explain what it was meant to.
- **Can't tell:** the test couldn't separate the answer from noise.
- **Reading:** describes what happened, with no pass or fail.

**The usual test for a forecast feature.** Each year from 2016 to 2024 is forecast from earlier cycles only. A feature has to improve log loss with each year's national polling miss removed in at least 4 of 5 years, improve raw log loss on average, and not make races the baseline put within 10 points worse. This bar was fixed in `spending_signal.py` before the spending features were scored. Earlier tests in REPORT.md compare log loss and Brier scores in the same forward backtest.

## The forecast

| Idea | Data | Result | Status | Write-up |
|---|---|---|---|---|
| Partisan lean from the last presidential election, on each race's own lines | The Downballot | Better calibrated than a two-election average in every year | Adopted | [REPORT](REPORT.md) §1 |
| One incumbency effect, pooled over every cycle | FEC, MIT | +10 points for a Democratic rather than a Republican incumbent | Adopted | [REPORT](REPORT.md) §3 |
| Incumbency that drifts from cycle to cycle | FEC, MIT | Margins closer (7.1 points off against 7.4), but winners called worse in 5 of 5 years (log loss 0.092 against 0.086) | Rejected | [REPORT](REPORT.md) §4 |
| Weighting recent cycles more | FEC, MIT | Forecasts worse. Tested while building, not in the pipeline | Rejected | [REPORT](REPORT.md) §9 |
| A smaller incumbent edge in close races | FEC, MIT | The drifting edge calls close races worse in 4 of 5 years. An edge fit from earlier years comes out larger, not smaller, and scores about the same | Rejected | [REPORT](REPORT.md) §4 |
| First-term incumbents, and how far the seat's party ran ahead of its lean last time | MIT | Log loss 0.092 → 0.086 | Adopted | [REPORT](REPORT.md) §4 |
| Challenger known before the campaign: a Wikipedia article by Jan 1 | Wikipedia revision histories | +1.8 points (standard error 0.5). Log loss 0.087 → 0.086, better in 3 of 5 years | Adopted | [REPORT](REPORT.md) §4 |
| Less credit for incumbents on redrawn lines | The Downballot | They ran no worse: -0.2 to +0.1 points out of sample. Full credit kept | Rejected | [REPORT](REPORT.md) §4 |
| Each race's error size set by open seat, redrawn lines and lean | FEC, MIT | Log loss 0.087 against 0.086 with one error size. No term more than two standard errors from zero | Rejected | [REPORT](REPORT.md) §4 |
| Error split into national, state and district parts | MIT | Needed for the seat total. Doesn't change any single race | Adopted | [REPORT](REPORT.md) §3 |
| The size of a national miss treated as uncertain (a Student t) | MIT, RealClearPolitics | 80% seat ranges 33 seats wide on average instead of 24, holding the result in 4 of 5 years either way | Adopted | [REPORT](REPORT.md) §4 |
| Only money a Sept 1 decision could see: receipts through June 30, outside money through Aug 31 | FEC API and independent expenditures | Replaces whole-election money, which looked better (log loss 0.410 against 0.421 in races between 5% and 95%) because fall money follows how the race is going | Adopted | [REPORT](REPORT.md) §4 |
| No money term at all | — | Worse than either | Rejected | [REPORT](REPORT.md) §4 |
| Symmetric money: the log odds of each side's money | FEC | Scores about as well in the backtest | Alternative | [REPORT](REPORT.md) §7, [freeze](data/forecast_freeze_2026_v2.json) |
| June 30 raised minus spent | FEC July reports | Log loss -0.00117 against the baseline, better in 4 of 5 years | Alternative | [SPENDING](SPENDING.md) summary, [freeze](data/forecast_freeze_2026_v2.json) |
| The generic ballot passed through at less than 1:1 | RealClearPolitics, 2002-2024 | National slopes aren't distinguishable from 1. Shrinking presidential years looks like a win but rests on 2020 alone (-0.0041, or -0.0002 without 2020) | Rejected | [SPENDING](SPENDING.md) §6 |
| A Sept 1 forecast: that day's generic ballot plus the usual fall drift | RealClearPolitics dated averages | Log loss 0.085 against 0.086 with the final polls. 80% seat ranges hold the result in 5 of 5 years | Adopted, for judging spending | [REPORT](REPORT.md) §4 |
| The forecast at five dates from Sept 1 to Oct 25 | Same | Log loss 0.085-0.089 at every date. National uncertainty shrinks, race forecasts don't get sharper | Reading | [REPORT](REPORT.md) §4 |

## The national environment

Tested over the 12 elections with a final RealClearPolitics average, 2002-2024. Both composites were fixed before either was compared with a result.

| Idea | Data | Result | Status | Write-up |
|---|---|---|---|---|
| Household cost basket, times the president's party | FRED, 41 series weighted as the HOUSING AFFORDABILITY TRACKER weights them | Predicts the polling miss worse: 2.97 points off leaving one election out, against 2.44 for the polls' lean alone. 36% of shuffles as strong | Rejected | [REPORT](REPORT.md) §8 |
| Credit and labor stress, times the president's party | FRED, 4 series | Worse (2.74), and runs the opposite way | Rejected | [REPORT](REPORT.md) §8 |
| The president's party, or a midterm penalty, with no economic data | — | Worse than the polls' lean alone | Rejected | [REPORT](REPORT.md) §8 |
| All of the above against the Sept 1 polls | RealClearPolitics daily series | Same answer | Rejected | [REPORT](REPORT.md) §8 |
| A House ballot built from economics alone | FRED | 5.3 points off leaving one election out, against 3.4 for the Sept 1 polls | Rejected | [REPORT](REPORT.md) §8 |

## Where committee money goes

Most of these try to explain one thing: why DCCC puts money in seats the forecast already gives it at 80%+.

| Idea | Data | Result | Status | Write-up |
|---|---|---|---|---|
| Reallocating DCCC's budget, under 9 assumptions about what money does | FEC independent expenditures, 2016-2024 | +0.05 to +0.19 seats per cycle at the default, +0.01 to +1.67 across all nine | Reading | [REPORT](REPORT.md) §5 |
| How strong money would have to be for a one-seat gain | Same | DCCC's first $1M in a race would have to move it 1.2-2.7 points, against 0.25 assumed | Reading | [REPORT](REPORT.md) §5 |
| What DCCC's own split implies | Same | A dollar in an 80%+ seat would have to work 1.4-3.3 times as hard as one in a 20-80% seat | Reading | [REPORT](REPORT.md) §5 |
| Hindsight: judging each dollar by what was known when it was spent | RealClearPolitics dated averages, expenditure dates | DCCC's 80%+ share falls from 22% to 18-19%. Explains about 14% of it | Reading | [REPORT](REPORT.md) §5 |
| The forecast's error on first-term members | MIT | Moves 2020's 80%+ share only from 30.5% to 29.9%, though 7 of those seats were lost anyway | Reading | [REPORT](REPORT.md) §5 |
| Maximizing the chance of a majority instead of expected seats | — | Defends in 2018 and 2020, attacks in the other years. Tested while building, not in the pipeline | Rejected | [REPORT](REPORT.md) §5 |
| Maximizing seats in the worst quarter of national environments | — | Adds defense every year, most in the years DCCC defended least. Not in the pipeline | Rejected | [REPORT](REPORT.md) §5 |
| Deterrence: DCCC and NRCC as a two-player game | FEC | The equilibrium puts at most 2% in 80%+ seats | Rejected | [REPORT](REPORT.md) §5 |
| TV-market cost | Cervas's market maps | DCCC already spends about half as much per race in the costliest markets. Pricing cost in moves the model further from DCCC | Rejected | [REPORT](REPORT.md) §6 |
| Mail and digital where TV is expensive | Expenditure purposes | No spender shifts toward mail | Reading | [REPORT](REPORT.md) §6 |
| Timing: persuasion that fades within days, against when ballots are cast | Election Administration and Voting Survey, published ad experiments | DCCC's money landed at 23-43% of its best timing. Re-timing gains +0.4 to +1.4 seats, an upper bound. Doesn't explain the safe-seat money | Reading | [REPORT](REPORT.md) §5 |
| Seats with committee money run closer than the forecast | FEC, 2016-2024 | 80-99% Democratic seats with money ran 3.8-5.9 points behind, against 2.3 without | Reading | [REPORT](REPORT.md) §7 |
| Committees reacting to each other week by week | Expenditure dates, 336 races | Next week's money predicts this week's better than last week's does. No sign of reaction | Rejected | [SPENDING](SPENDING.md) §6 |
| Committees spending more where close races share a TV market | Cervas's market maps | -0.15 (standard error 0.50). No visible effect, wide errors | Rejected | [SPENDING](SPENDING.md) §6 |
| What the same airtime costs each kind of buyer | 14,356 station political files from the FCC, 2024 and 2026 | Coordinated party buys pay the candidate's rate (1.00x), a party's independent buys 2.40x, outside groups 2.3-2.5x. Reallocation gains double but stay under a seat | Reading | [SPENDING](SPENDING.md) §7 |
| TV prices by market: rate, price gap, cost per rating point | FCC files, including ratings printed on orders | Design fixed first. Rates fail as a cost. On prices neither plan saw, population alone does as well as cost per point | Rejected; the population proxy stands | Unmerged branch `claude/gifted-taussig-875c78`, SPENDING §7 |

## Campaign spending, vendors and donors as forecast features

28 features, each against the usual test. A placebo that shuffles a feature among each year's races clears that test about 10% of the time, so at least one pass in 28 was about 95% likely by luck. Each feature's numbers are in the table in [SPENDING](SPENDING.md) §2.

| Round | Data | Features | Passed | Write-up |
|---|---|---|---|---|
| First | FEC operating expenditures, 3.7 million payments, classified by purpose and vendor | Money raised less fundraising costs; early voter-contact money; share spent on fundraising; share on payroll and overhead; incumbent's early polling; share at vendors with a competitive-race record; share of spending in April-June; fundraising, contact and overhead together | 1 of 8: share in April-June, a Candidate | [SPENDING](SPENDING.md) §2 |
| Second | Same | Spent as a share of raised; months of cash left; fundraising cost as a share of raised; digital and mail; polling and digital; TV and field; polled before first voter contact; how early voter contact began; vendor concentration; kept last cycle's vendors; debts as a share of raised | 0 of 11 | [SPENDING](SPENDING.md) §2 |
| Donors | FEC individual files, 7.0 million itemized gifts | Itemized donors; donors in the state; share from the top 10; share from last cycle's donors; share from the state | 0 of 5 | [SPENDING](SPENDING.md) §2 |
| Redistricting | Census 2010 and 2020 blocks | Share of the district new to its incumbent (fails, wrong sign); the same against that redraw's average | 1 of 2: the second, a weaker Candidate | [SPENDING](SPENDING.md) §6 |
| Lean by money | FEC | Lean counts differently under $2M; lean counts differently by log money. Lean is steeper where little is spent, but error isn't tighter | 0 of 2 | [SPENDING](SPENDING.md) §6 |

## Other spending analyses

| Idea | Data | Result | Status | Write-up |
|---|---|---|---|---|
| Campaign types by k-means | Pre-June 30 spending shares, 3,874 campaigns | Weak structure (best silhouette 0.25), mostly whether a campaign was on TV by June. Worse as forecast features | Rejected | [SPENDING](SPENDING.md) §3 |
| Candidates leaving TV where it's expensive | FEC operating expenditures, Cervas | Going from a market twice a district's size to twenty times, TV falls 10.0% of the budget (standard error 1.9%) and digital rises 6.1% | Reading | [SPENDING](SPENDING.md) §4 |
| What goes with what: association rules | Same | Only polling followed by TV the next quarter holds, in later elections too (conditional lift 1.19-1.24) | Reading | [SPENDING](SPENDING.md) §5 |
| The vendor market | Same | 5.2% of 2024 vendors served both parties | Reading | [SPENDING](SPENDING.md) §5 |
| Early Google ad spending | Google's political ad data | Forecasts worse in both years it can be tested | Rejected | [SPENDING](SPENDING.md) §6 |

## Combinations and committee signals

The search and the committee test were designed and committed (33049ef) before any real combination was scored.

| Idea | Data | Result | Status | Write-up |
|---|---|---|---|---|
| Gradient-boosted trees on 51 yes-or-no items | Everything a Sept 1 forecaster could see | Log loss +0.00008 with the miss removed; 48% of shuffles as good | Rejected | [BASKET_DESIGN](BASKET_DESIGN.md) |
| Every pair and triple of those items, 3,961 in all | Same | None clears the family-wise bar of 4.04. The largest is t = -3.90 | Rejected | [BASKET_DESIGN](BASKET_DESIGN.md) |
| Seats that look like ones committees enter run behind the forecast, even where none did | Same | Passes (t = -5.40), but mostly a known misfit in the forecast's money term, and it makes forecasts worse | Rejected after follow-up | [BASKET_DESIGN](BASKET_DESIGN.md), "Added after the run" |
| Near-line seats no committee entered rarely flip | FEC, 2016-2024 | 3 of 134 lost, against 13.8 expected. Found by looking at these same years | Lead | [BASKET_DESIGN](BASKET_DESIGN.md), "A lead" |
| The same, at the same Sept 1 Cook rating | Wikipedia's ratings pages as of Sept 1 | Design fixed first. The gap stays about 9 points (p 0.17 and 0.08 under two money forms), on 7 losses at most | Can't tell | Unmerged branch `claude/sleepy-kowalevski-97a039`, COMMITTEE_RATINGS.md |
| Unexpected committee spending predicts forecast errors | Same | The defender's: no sign. The attacker's: -0.63 points per standard deviation (p under 0.001), which fits both information and dollars | Reading | Same |

## Tested outside the pipeline

Checked in working sessions while building the repo. The code wasn't kept, so the numbers can't be reproduced here, and only the verdicts are given.

| Idea | Data | Result | Status |
|---|---|---|---|
| Nominees' Wikipedia page views, Jan 1-Aug 31 | [`data/wikipedia_views.csv`](data/wikipedia_views.csv), competitive races in 2018, 2022 and 2024 | Mostly re-measures incumbency and money | Rejected |
| Press tone, Democrat against Republican | GDELT through BigQuery, [`data/news_bq*.csv`](data/) | No gain on held-out years, and close to zero with incumbency held fixed: incumbents get worse press | Rejected |
| A separate term for the challenger's money | FEC | Worse on log loss and Brier, scored on a fixed set of close races | Rejected |
| Small-dollar share of receipts | FEC July reports | No effect | Rejected |
| Real cash on hand instead of raised minus spent | FEC July reports | About the same as raised minus spent with the miss removed, so not adopted | Rejected |

## Written down for 2026

| Record | Committed | Before |
|---|---|---|
| [`forecast_freeze_2026.csv`](data/forecast_freeze_2026.csv): every race under four money specs | Nov 2, 13:34 CST | Election Day |
| [`forecast_freeze_2026_v2.csv`](data/forecast_freeze_2026_v2.csv): the same with corrected nominees and each race's committee money | Nov 2, 15:19 CST | Election Day |
| [`spending_candidate_2026.csv`](data/spending_candidate_2026.csv), [`retention_candidate_2026.csv`](data/retention_candidate_2026.csv): the two candidate features | Nov 3, 14:19 CST | Polls closed |
| [`basket_candidate_2026.csv`](data/basket_candidate_2026.csv), [`basket_entry_2026.csv`](data/basket_entry_2026.csv), [`basket_nearline_2026.csv`](data/basket_nearline_2026.csv): the trees' shift, the entry score, and the near-line seats with three predictions | Nov 4, 15:06 CST | Any result entered the repo, but after polls closed |
| `committee_ratings_2026.csv`: those seats' Sept 1 ratings and a fourth prediction | Nov 4, 15:37 CST, unmerged branch `claude/sleepy-kowalevski-97a039` | Same |

## Data considered but not used

| Source | Why not | Write-up |
|---|---|---|
| Meta's ad library | Needs an identity-verified account and a developer token, and reports spending in ranges by state | [SPENDING](SPENDING.md) §6 |
| Wesleyan Media Project TV airings | Requested Nov 3, 2026, not yet delivered | [SPENDING](SPENDING.md) §6 |
| AdImpact | Commercial license | [SPENDING](SPENDING.md) §6 |
