"""
A TV buy reaches everyone in a market, so a committee with several close races in one
market gets more for the same ad. The allocation treats races separately and charges
each the whole market. Do committees spend as if they saw that discount?

For each race the forecast has within CLOSE points: committee money on how many other
close races share its main market, with its own reach and closeness held fixed. And
the same question put differently: does money track reach, or reach shared among the
market's close races, better?
"""

import numpy as np
import pandas as pd

from .load_media_markets import load_media_markets

CLOSE = 10


def shared_markets(frames):
    """`frames`: cycle -> DataFrame of district_id, mu and money. One row per close race, with n_other and reach."""
    rows = []
    for cycle, f in frames.items():
        m = f.merge(load_media_markets(cycle)[["district_id", "market", "market_name", "reach"]], on="district_id")
        close = m[m.mu.abs() < CLOSE].copy()
        close["n_other"] = close.groupby("market").district_id.transform("size") - 1
        rows.append(close.assign(cycle=cycle))
    return pd.concat(rows, ignore_index=True)


def spillover_fit(close, money="committee"):
    """Slope of log money on n_other, with log reach, |mu| and year held fixed (se clustered by market), and the fit (R^2) of
    log money on log reach against log(reach shared among the market's close races)."""
    y = np.log1p(close[money])
    base = [np.ones(len(close)), np.abs(close.mu), *(close.cycle == c for c in sorted(close.cycle.unique())[1:])]
    X = np.column_stack(base + [np.log(close.reach), close.n_other]).astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    group = pd.factorize(close.cycle.astype(str) + close.market.astype(str))[0]
    score = np.zeros((group.max() + 1, X.shape[1]))
    np.add.at(score, group, X * (y - X @ beta).to_numpy()[:, None])
    bread = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(bread @ score.T @ score @ bread))

    def r2(price):
        Z = np.column_stack(base + [price]).astype(float)
        fitted = Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"n_other": beta[-1], "se": se[-1], "reach": beta[-2], "reach_se": se[-2], "r2_reach": r2(np.log(close.reach)),
            "r2_shared": r2(np.log(close.reach / (1 + close.n_other))), "n": len(close), "with_other": float((close.n_other > 0).mean())}
