"""
Which districts were redrawn between two maps: the same presidential
election, computed on both maps, differs. The sheets round to 0.1 or whole
points, so a lone district differing by under ~0.75 points of share is
rounding; a real redraw moves voters between at least two districts.
"""

import pandas as pd

from .load_presidential import load_presidential
from .load_pvi import MAP_OVERRIDES

TOLERANCE = 0.0075


def compare_maps(old_lines=2024, new_lines=2026, year=2024):
    """district_id, old_share, new_share, changed (D two-party share of
    `year`'s presidential vote on each map)."""
    old = load_presidential(old_lines).query("pres_year == @year").set_index("district_id").d_share
    new = load_presidential(new_lines).query("pres_year == @year").set_index("district_id").d_share
    for (cycle, state), lines in MAP_OVERRIDES.items():
        if cycle == new_lines and lines == old_lines:
            new[new.index.str.startswith(state + "-")] = old[old.index.str.startswith(state + "-")]
    out = pd.DataFrame({"old_share": old, "new_share": new}).dropna()
    differs = (out.new_share - out.old_share).abs() > TOLERANCE
    redrawn_states = differs.groupby(out.index.str[:2]).sum().loc[lambda n: n >= 2].index
    out["changed"] = differs & out.index.str[:2].isin(redrawn_states)
    return out.rename_axis("district_id").reset_index()
