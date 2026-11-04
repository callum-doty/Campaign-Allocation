"""
Which TV markets' political files to read: the main market of every race the forecast
put within CLOSE_MARKET points. Those are where sponsors compete for the same airtime, and
where price matters to where money goes.
"""

import pandas as pd

from .build_races import build_races
from .fetch_fcc import CLOSE_MARKET
from .load_media_markets import load_media_markets
from .redrawn_retention import forecast_margin

CYCLES = (2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026)


def markets_for(cycle):
    """Sorted market names, for the forecast made before `cycle`."""
    built = {c: build_races(c) for c in CYCLES if c <= cycle}
    races = forecast_margin(built, cycle)
    close = races[races.mu.abs() < CLOSE_MARKET].district_id
    markets = load_media_markets(cycle).set_index("district_id").market_name
    return sorted(markets.reindex(close).dropna().unique())
