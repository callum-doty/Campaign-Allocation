"""
Score one completed cycle strictly forward in time: fit the baseline on
every EARLIER cycle, forecast this one (each race, and the seat total with
its shared national error, also with that error's size treated as known
for comparison), then ask how DCCC's own spending compares with
the best allocation of the same budget under the spending assumption,
across a grid of assumptions. The same forecast without the incumbent
history features (first-term flags, last cycle's overperformance) is scored
alongside, to show what they add, and so is one using each side's money
over the whole election instead of what was visible by Sept 1, to show
what that hindsight was worth. Two readings need less of that assumption:
how strong party money would have to be for reallocating to gain a full
seat (break_even.py), and how much more effective money would have to be
in each band of races for DCCC's own split to be optimal (implied_belief.py).

All of that judges a fall of spending with the final pre-election polls,
which most of DCCC's decisions didn't have. So the cycle is also forecast as
it stood at each checkpoint of the fall (dated_forecast.py), and the
safe-seat readings are redone two ways: with the Sept 1 forecast for every
dollar (DCCC's money in 80%+ races, the best allocation, the implied 80%+
multiplier), and dollar by dollar, each judged by the forecast at the last
checkpoint before it reached voters (dollars out before Sept 1 by the Sept 1
forecast).

Last, Job 2's clock (ballot_timing.py): each DCCC dollar is discounted by how
much of its persuasion is left when voters cast their ballots, and the
allocation is redone with money timed to each state's ballots, across the
persuasion half-lives tested.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .ballot_timing import HALF_LIVES, ballot_timing, timed_gain
from .break_even import break_even_scale
from .build_races import FLOOR_PSEUDO_DOLLARS
from .dated_forecast import CHECKPOINTS, dated_forecast
from .fit_baseline import FEATURES, HISTORY, fit_baseline
from .implied_belief import implied_belief
from .load_expenditures import load_expenditures
from .predict_margin import predict_margin
from .reallocation_gain import reallocation_gain
from .seat_distribution import seat_distribution
from .seats_needed import seats_needed
from .spending_effect import HALF_SATURATION, SCALE_PTS

SCALES = (0.5, 1.5, 4.5)
HALF_SATURATIONS = (1_650_000, 5_500_000, 16_500_000)
CAP_SHARE = 0.15          # no race gets more than 15% of the budget


def hindsight_money(races):
    """floor_share from each side's candidate and outside spending over the
    whole election, most of it after Sept 1."""
    d, r, k = races.disbursements_d + races.outside_d, races.disbursements_r + races.outside_r, FLOOR_PSEUDO_DOLLARS
    return races.assign(floor_share=np.log((d + k) / (d + r + 2 * k)))


def backtest(cycle, races):
    """`races` maps cycle -> build_races(cycle)."""
    train = pd.concat([r for c, r in races.items() if c < cycle])
    model = fit_baseline(train)
    without = fit_baseline(train, [f for f in FEATURES if f not in HISTORY])
    test = races[cycle].copy()
    test["mu"] = predict_margin(test, model)
    test["p"] = norm.cdf(test.mu / model["sigma"])
    test["mu_without"] = predict_margin(test, without)
    test["p_without"] = norm.cdf(test.mu_without / without["sigma"])
    hindsight = fit_baseline(hindsight_money(train))
    test["p_hindsight"] = norm.cdf(predict_margin(hindsight_money(test), hindsight) / hindsight["sigma"])
    dated = {date: dated_forecast(cycle, races, date) for date in CHECKPOINTS}
    first = dated[CHECKPOINTS[0]]
    sept = {**first["model"], "sigma": first["sigma"], "sigma_national": first["sigma_national"], "gb": first["gb"]}
    test["mu_sept"], test["p_sept"] = first["mu"], first["p"]
    for date, f in dated.items():
        test[f"p_{date}"] = f["p"]
    known = test[test.margin.notna()].assign(won=lambda t: (t.margin > 0).astype(float))
    calibration = known.groupby("tier").agg(n=("p", "size"), predicted=("p", "mean"), actual=("won", "mean"))
    scores = {}
    for name, col in [("with", "p"), ("without", "p_without"), ("hindsight", "p_hindsight"), ("sept", "p_sept")] + \
                     [(date, f"p_{date}") for date in CHECKPOINTS]:
        p = known[col].clip(1e-6, 1 - 1e-6)
        scores[name] = {"brier": float(np.mean((p - known.won) ** 2)),
                        "log_loss": float(-np.mean(known.won * np.log(p) + (1 - known.won) * np.log(1 - p)))}

    budget, cap = float(test.dccc.sum()), CAP_SHARE * float(test.dccc.sum())
    grid = {(scale, half): reallocation_gain(test.mu, model["sigma"], test.dccc, cap, scale_pts=scale, half_saturation=half)
            for scale in SCALES for half in HALF_SATURATIONS}
    gain, test["model_dccc"] = grid[(SCALE_PTS, HALF_SATURATION)]
    break_even = {half: break_even_scale(test.mu, model["sigma"], test.dccc, cap, half) for half in HALF_SATURATIONS}
    belief = {half: implied_belief(test.mu, model["sigma"], test.dccc, half) for half in HALF_SATURATIONS}
    need, states = seats_needed(cycle, test.district_id), test.district_id.str[:2]
    seats = seat_distribution(test.mu, model, need, states)
    seats_known = seat_distribution(test.mu, {**model, "national_df": None}, need, states)
    seats_sept = seat_distribution(test.mu_sept, sept, need, states)
    seats["actual"] = seats_known["actual"] = seats_sept["actual"] = int((test.margin > 0).sum())
    gain_sept, test["model_dccc_sept"] = reallocation_gain(test.mu_sept, sept["sigma"], test.dccc, cap)
    sept.update(drift=first["drift"], seats=seats_sept, gain=gain_sept,
                belief={half: implied_belief(test.mu_sept, sept["sigma"], test.dccc, half) for half in HALF_SATURATIONS})

    # each DCCC dollar judged by the forecast at the last checkpoint before it reached voters
    spent = load_expenditures(cycle).query("bucket == 'dccc'")
    spent = spent[spent.district_id.isin(test.district_id)]
    stamps = pd.to_datetime([f"{cycle}-{d}" for d in CHECKPOINTS])
    spent = spent.assign(checkpoint=[CHECKPOINTS[max(stamps.searchsorted(d, side="right") - 1, 0)] for d in spent.date])
    tranches = spent.groupby(["district_id", "checkpoint"]).amount.sum().reset_index()
    row = tranches.district_id.map(pd.Series(np.arange(len(test)), index=test.district_id))
    mu_t = np.array([dated[c]["mu"][i] for c, i in zip(tranches.checkpoint, row)])
    sigma_t = np.array([dated[c]["sigma"] for c in tranches.checkpoint])
    observed_t = test.dccc.to_numpy()[row]
    tranches = tranches.assign(p=norm.cdf(mu_t / sigma_t))
    when_spent = {"tranches": tranches, "drift": {date: f["drift"] for date, f in dated.items()},
                  "belief": {half: implied_belief(mu_t, sigma_t, observed_t, half, weights=tranches.amount)
                             for half in HALF_SATURATIONS}}
    timing = {}
    for half_life in HALF_LIVES:
        t = ballot_timing(cycle, test, half_life=half_life)
        retime, _ = timed_gain(test.mu, model["sigma"], test.dccc, t.e_obs, t.e_best, cap, move=False)
        both, x = timed_gain(test.mu, model["sigma"], test.dccc, t.e_obs, t.e_best, cap)
        landed = test.dccc.to_numpy() * t.e_obs / t.e_best          # DCCC's money as it landed, in best-timed dollars
        timing[half_life] = {"races": t, "gain_retime": retime, "gain_both": both, "model_dccc": x,
                             "belief": implied_belief(test.mu, model["sigma"], landed, HALF_SATURATION / t.e_best,
                                                      weights=test.dccc)}
    return {"cycle": cycle, "model": model, "races": test, "calibration": calibration,
            "brier": scores["with"]["brier"], "scores": scores, "budget": budget, "gain": gain,
            "grid": {k: v[0] for k, v in grid.items()}, "break_even": break_even, "belief": belief, "seats": seats,
            "seats_known_sigma": seats_known, "sept": sept, "when_spent": when_spent, "timing": timing}
