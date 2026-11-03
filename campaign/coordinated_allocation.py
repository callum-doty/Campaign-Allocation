"""
Where DCCC's and NRCC's coordinated money went in an unfinished cycle, read
the way the backtest reads their independent expenditures, as of the last
report filed (Oct 14). Earlier years are cut at the same date, so the money is
comparable, though their dates are when ads reached voters and these are when
the committee paid.

Each committee is read from its own side: its "safe" seats are the ones its
own party is 80%+ to win and its "long shots" under 20%, and the multiplier and
reallocation gain are implied_belief's and reallocation_gain's with the margin's
sign flipped for NRCC. Both are repeated with symmetric money (freeze_forecast.py),
since how much counts as safe depends on that choice.

Last, whether money flags races the forecast rates too lopsided: how far each
race ran from the forecast in 2016-2024, by which committees had spent there by
Oct 14, among seats the forecast rated 80-99% or 1-20% Democratic. It can't
separate committees knowing something from their money working.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .fit_baseline import fit_baseline
from .freeze_forecast import SPECS, money_rows, money_terms
from .implied_belief import implied_belief
from .load_coordinated import coordinated_by_race, load_coordinated
from .load_expenditures import load_expenditures
from .load_media_markets import load_media_markets
from .predict_margin import predict_margin
from .reallocation_gain import reallocation_gain

CUTOFF = "10-14"
CAP_SHARE = 0.15
COMMITTEES = (("dccc", 1), ("nrcc", -1))
BANDS = [0, 3, 10, np.inf]
BAND_LABELS = ["Cheap (reach under 3)", "Mid (3-10)", "Expensive (10+)"]


def reading(mu, sigma, money, year, spec):
    """One row per committee. `money`: dccc and nrcc dollars aligned with `mu`."""
    rows = []
    for who, sign in COMMITTEES:
        own, x = np.asarray(mu, float) * sign, money[who].to_numpy(float)
        p, budget = norm.cdf(own / sigma), x.sum()
        gain, _ = reallocation_gain(own, sigma, x, CAP_SHARE * budget)
        rows.append({"year": year, "spec": spec, "who": who, "budget": budget, "races": int((x > 0).sum()),
                     "long_shot": x[p < 0.2].sum() / budget, "safe": x[p > 0.8].sum() / budget,
                     "multiplier": implied_belief(own, sigma, x).multiplier["80%+"], "gain": gain})
    return rows


def coordinated_allocation(races, backtests, fc):
    """{'table', 'landed', 'overlap', 'lopsided', 'signal', 'channel', 'reported'}: `races`: cycle -> build_races(cycle);
    `backtests`: backtest() results; `fc`: forecast() of the unfinished cycle."""
    cycle, rows, history, landed = fc["cycle"], [], [], {}
    symmetric_money = {c: money_terms(r) for c, r in races.items()}
    for b in backtests:
        year, test = b["cycle"], b["races"]
        ie = load_expenditures(year)
        ie = ie[ie.bucket.isin(["dccc", "nrcc"])]
        seen = ie[ie.date <= pd.Timestamp(f"{year}-{CUTOFF}")]
        landed[year] = seen.groupby("bucket").amount.sum() / ie.groupby("bucket").amount.sum()
        money = (seen.pivot_table(index="district_id", columns="bucket", values="amount", aggfunc="sum", fill_value=0)
                 .reindex(index=test.district_id, columns=["dccc", "nrcc"]).fillna(0.0))
        rows += reading(test.mu, b["model"]["sigma"], money, year, "baseline")
        model = fit_baseline(pd.concat([r for c, r in symmetric_money.items() if c < year]), SPECS["symmetric"])
        rows += reading(predict_margin(symmetric_money[year], model), model["sigma"], money, year, "symmetric")
        history.append(money_rows(pd.DataFrame({
            "district_id": test.district_id, "mu_baseline": test.mu, "p_baseline": test.p,
            "dccc_coordinated": money.dccc.to_numpy(), "nrcc_coordinated": money.nrcc.to_numpy()}),
            test.set_index("district_id").margin).assign(state_year=lambda t: t.district_id.str[:2] + str(year)))

    now = fc["races"]
    spent = coordinated_by_race(cycle).reindex(now.district_id).fillna(0.0)
    rows += reading(now.mu, fc["model"]["sigma"], spent, cycle, "baseline")
    model = fit_baseline(pd.concat([r for c, r in symmetric_money.items() if c < cycle]), SPECS["symmetric"])
    rows += reading(predict_margin(symmetric_money[cycle], model), model["sigma"], spent, cycle, "symmetric")

    both = (spent.dccc > 0).to_numpy() & (spent.nrcc > 0).to_numpy()
    lopsided = both & (now.p.to_numpy() > 0.8) & (now.p.to_numpy() < 0.99)

    def signal(g):
        by_cluster = (g.resid - g.resid.mean()).groupby(g.state_year).sum()
        return pd.Series({"n": len(g), "p": g.p_baseline.mean(), "won": g.won.mean(), "resid": g.resid.mean(),
                          "se": np.sqrt((by_cluster ** 2).sum()) / len(g)})
    pooled = pd.concat(history)
    pooled = pooled[pooled.band != ""]

    rows_now = load_coordinated(cycle).merge(load_media_markets(cycle)[["district_id", "reach"]], on="district_id")
    rows_now["band"] = pd.cut(rows_now.reach, BANDS, labels=BAND_LABELS)
    by_band = rows_now.groupby(["bucket", "band", "channel"], observed=True).amount.sum().unstack(fill_value=0.0)
    return {"table": pd.DataFrame(rows), "landed": pd.DataFrame(landed).T,
            "overlap": {"both": int(both.sum()), "correlation": float(spent.dccc.corr(spent.nrcc))},
            "lopsided": {"n": int(lopsided.sum()), "dccc": float(spent.dccc[lopsided].sum()), "nrcc": float(spent.nrcc[lopsided].sum()),
                         "races": now.district_id[lopsided].tolist()},
            "signal": pooled.groupby(["band", "who"])[["resid", "p_baseline", "won", "state_year"]].apply(signal),
            "channel": by_band.div(by_band.sum(axis=1), axis=0),
            "reported": load_coordinated(cycle).groupby("bucket").amount.sum()}
