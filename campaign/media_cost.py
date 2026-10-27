"""
Does pricing in TV-market cost change the allocation verdict?

A TV buy pays for the whole market, so the cost of reaching a district's
voters scales with the market's population per voter it's meant to reach.
That overstates the waste where one market holds several battlegrounds (a
New York buy reaches NJ-07, NY-04 and NY-17 at once), so the cost index
counts every battleground district (20-80% forecast) in the market:

    shared_reach_i = market population / battleground population in that market
                     (the district itself always counts)

The penalty applies only to the TV share of spending (mail and digital
don't pay for the whole market), taken from DCCC's own purpose codes:

    half_saturation_i = HALF_SATURATION * (tv_share * (shared_reach_i / median)^gamma + 1 - tv_share)

gamma = 0 is the shipped model; gamma = 1 means TV dollars buy proportionally
less where the market is bigger relative to the battlegrounds in it.
"""

import numpy as np
from scipy.stats import norm

from .allocate import allocate
from .spending_effect import HALF_SATURATION, spending_effect

GAMMAS = (0.0, 0.5, 1.0)
CAP_SHARE = 0.15


def media_cost(bt, markets, tv_share):
    """`bt`: a backtest() result; `markets`: load_media_markets(cycle);
    `tv_share`: TV's share of DCCC's spending that cycle."""
    r = bt["races"].merge(markets[["district_id", "market", "market_pop", "pop", "main_market_share"]],
                          on="district_id", how="left")
    battleground = r.p.between(0.2, 0.8)
    in_market = r["pop"] * r.main_market_share
    bg_pop = in_market[battleground].groupby(r.market[battleground]).sum()
    shared = r.market_pop / (r.market.map(bg_pop).fillna(0) + in_market.where(~battleground, 0))
    r["shared_reach"] = shared.fillna(shared[battleground].median())          # Alaska, Hawaii
    relative = (r.shared_reach / r.shared_reach[battleground].median()).to_numpy(float)
    r["expensive"] = battleground & (r.shared_reach >= r.shared_reach[battleground].quantile(0.75))

    mu, sigma, obs = r.mu.to_numpy(float), bt["model"]["sigma"], r.dccc.to_numpy(float)
    rows = []
    for gamma in GAMMAS:
        h = HALF_SATURATION * (tv_share * relative ** gamma + 1 - tv_share)
        x = allocate(mu, sigma, obs, obs.sum(), CAP_SHARE * obs.sum(), half_saturation=h)
        shift = spending_effect(x, half_saturation=h) - spending_effect(obs, half_saturation=h)
        rows.append({"gamma": gamma, "gain": float(norm.cdf((mu + shift) / sigma).sum() - r.p.sum()),
                     "model_share_expensive": float(x[r.expensive].sum() / x.sum())})
    bg = r[battleground]
    return {"cycle": bt["cycle"], "races": r, "tv_share": tv_share, "results": rows,
            "dccc_share_expensive": float(obs[r.expensive].sum() / obs.sum()),
            "dccc_per_race_expensive": float(bg.dccc[bg.expensive].mean()),
            "dccc_per_race_other": float(bg.dccc[~bg.expensive].mean()),
            "median_reach": float(np.nanmedian(r.shared_reach[battleground]))}
