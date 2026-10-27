"""
Each district's TV market, from Jonathan Cervas's analysis of Nielsen
markets and congressional districts:
https://jonathancervas.com/maps/media-markets/

The page embeds its data; fetch.py saves the page and this parses it. It
covers the 2022, 2024 and 2026 lines (not Alaska or Hawaii). Method, per
the page: Nielsen market polygons, each county assigned to a market by
point-in-polygon, Census 2024 county populations, and district populations
split across markets by iterative proportional fitting over district x
county overlaps. One known gap: Nielsen splits Riverside County between Los
Angeles and Palm Springs; the county method puts it all in Los Angeles.

reach = people in the district's main market per district resident: what a
TV buy pays for per voter it is meant to reach (NY-17: 28; median ~4.5).
His 2026 set puts Missouri on its 2025 map, so Missouri uses the 2024 rows
(load_pvi.MAP_OVERRIDES).
"""

import json
from pathlib import Path

import pandas as pd

from .load_pvi import MAP_OVERRIDES

PAGE = Path(__file__).resolve().parents[1] / "data" / "raw" / "media_markets.html"
COLUMNS = ["district_id", "market", "market_name", "market_pop", "pop", "main_market_share", "n_markets", "reach"]


def load_media_markets(cycle):
    html = PAGE.read_text(encoding="utf-8")
    decode = json.JSONDecoder().raw_decode
    lines = decode(html[html.index("const CDL=") + len("const CDL="):])[0]["cycles"]
    markets = {m["i"]: m for m in decode(html[html.index("const DATA=") + len("const DATA="):])[0]["dmas"]}
    if str(cycle) not in lines:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(lines[str(cycle)]["cds"])
    for (c, state), from_cycle in MAP_OVERRIDES.items():
        if c == cycle:
            old = pd.DataFrame(lines[str(from_cycle)]["cds"])
            df = pd.concat([df[df.st != state], old[old.st == state]])
    df = df.rename(columns={"sd": "district_id", "prim": "market", "primsh": "main_market_share", "n": "n_markets"})
    df["market_name"] = df.market.map(lambda i: markets[i]["name"])
    df["market_pop"] = df.market.map(lambda i: markets[i]["pop"])
    return df[COLUMNS].reset_index(drop=True)
