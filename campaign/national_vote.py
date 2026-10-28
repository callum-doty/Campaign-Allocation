"""
The national House popular vote: Democratic minus Republican votes as a
share of every vote cast for the House, in points, summed over each
district's general election. D and R votes come from load_results (fusion
lines and state party labels handled), the total from MIT's total votes per
district. Uncontested races where a state prints no votes (Florida) count as
zero, as in the official tally. It matches the official national margins to
about a tenth of a point (2018 D+8.6, 2022 R+2.8).
"""

import pandas as pd

from .load_results import RAW, load_results


def national_vote(cycles):
    """Series: cycle -> national House margin (D minus R, points)."""
    raw = pd.read_csv(RAW, dtype={"district": str}, low_memory=False)
    out = {}
    for cycle in cycles:
        keep = (raw.year == cycle) & (raw.stage == "GEN") & (raw["mode"] == "TOTAL")
        for col in ("runoff", "special"):
            keep &= ~raw[col].astype(str).str.upper().eq("TRUE")
        g = raw[keep]
        total = g.groupby([g.state_po, g.district]).totalvotes.first().sum()
        results = load_results(cycle)
        out[cycle] = 100 * (results.d_votes.sum() - results.r_votes.sum()) / total
    return pd.Series(out, name="house_vote")
