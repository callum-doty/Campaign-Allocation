"""
Partisan lean of each district, on the lines that House election used.

    PVI_i = 100 * (district D two-party share - national D two-party share)

using the most recent presidential election held BEFORE the House race, so
nothing from the same November leaks in. One election rather than Cook's
75/25 blend of two: older elections are often missing on newer maps, and in
forward backtests the single most recent election predicted House margins
better than a two-election average in every year tested (2016-2024).
"""

import pandas as pd

from .load_presidential import load_presidential

# Certified national two-party D share of the presidential vote.
NATIONAL_D2PARTY = {2008: 0.5369, 2012: 0.5196, 2016: 0.5111, 2020: 0.5227, 2024: 0.4925}
# House cycle -> the sheet describing its lines (2014 used the 2012 map).
LINES = {2012: 2012, 2014: 2012, 2016: 2016, 2018: 2018, 2020: 2020, 2022: 2022, 2024: 2024, 2026: 2026}
# (cycle, state) -> lines actually used, where courts overrode the sheet:
# the U.S. Supreme Court ruled on 2026-09-25 that Missouri votes on its 2022
# map in 2026, after The Downballot's July 2026 sheet put it on the 2025 map.
MAP_OVERRIDES = {(2026, "MO"): 2024}


def load_pvi(cycle):
    """Series: district_id -> PVI (percentage points, D-positive)."""
    year = max(y for y in NATIONAL_D2PARTY if y < cycle)
    pres = load_presidential(LINES[cycle])
    for (c, state), lines in MAP_OVERRIDES.items():
        if c == cycle:
            other = load_presidential(lines)
            pres = pd.concat([pres[~pres.district_id.str.startswith(state + "-")],
                              other[other.district_id.str.startswith(state + "-")]])
    pres = pres[pres.pres_year == year].set_index("district_id")["d_share"]
    return ((pres - NATIONAL_D2PARTY[year]) * 100).rename("pvi")
