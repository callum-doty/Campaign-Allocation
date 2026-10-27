"""
One row per House race, with every feature the model uses.

Features:
- pvi: partisan lean on the race's own lines (load_pvi).
- d_incumbent / open_seat: from FEC's incumbent flag on the matched
  nominees; a Republican incumbent is the baseline.
- floor_share = log((F_D + k) / (F_D + F_R + 2k)), F = candidate spending
  plus outside (non-national-committee) spending for that side, k = $10K.
  The share of money neither national committee controls, standing in for
  candidate strength and how hard each side is contesting the race. It is a
  forecasting feature, not a causal effect.
- gb: the generic-ballot polling average heading into the election.
- dccc / nrcc: the national committees' own spending (the allocation question).
- tier: a label from PVI and incumbency (incumbent worth +/-2 points), for
  reporting only; the model doesn't use it.

Races with under $100K of total spending are dropped (mostly uncontested
seats), and so is Alaska (top-four primary and ranked-choice general).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .load_candidates import load_candidates
from .load_pvi import load_pvi
from .load_results import load_results
from .load_spending import load_spending

GENERIC_BALLOT = Path(__file__).resolve().parents[1] / "data" / "generic_ballot.csv"
MIN_TOTAL_SPEND = 100_000
EXCLUDED_STATES = {"AK"}
FLOOR_PSEUDO_DOLLARS = 10_000
TIERS = [(10, "Safe D"), (5, "Likely D"), (1, "Lean D"), (-3, "Toss-Up"), (-5, "Lean R"), (-10, "Likely R")]


def build_races(cycle):
    results = load_results(cycle)
    candidates = load_candidates(cycle, results).pivot(index="district_id", columns="party")
    candidates.columns = [f"{field}_{party.lower()}" for field, party in candidates.columns]
    races = (load_pvi(cycle).rename_axis("district_id").reset_index()
             .merge(results[["district_id", "margin"]], on="district_id", how="left")
             .merge(load_spending(cycle), on="district_id", how="left")
             .merge(candidates, left_on="district_id", right_index=True, how="left"))
    for col in ("outside_d", "outside_r", "dccc", "nrcc", "disbursements_d", "disbursements_r"):
        races[col] = races.get(col, pd.Series(0.0, index=races.index)).fillna(0.0)

    races["cycle"] = cycle
    races["gb"] = pd.read_csv(GENERIC_BALLOT).set_index("cycle").generic_ballot[cycle]
    races["d_incumbent"] = (races.get("ici_d") == "I").astype(float)
    races["open_seat"] = ((races.d_incumbent == 0) & (races.get("ici_r") != "I")).astype(float)
    races["status"] = np.select([races.d_incumbent == 1, races.open_seat == 1], ["D incumbent", "Open"], "R incumbent")
    races["floor_d"] = races.disbursements_d + races.outside_d
    races["floor_r"] = races.disbursements_r + races.outside_r
    k = FLOOR_PSEUDO_DOLLARS
    races["floor_share"] = np.log((races.floor_d + k) / (races.floor_d + races.floor_r + 2 * k))
    effective = races.pvi + 2 * races.d_incumbent - 2 * ((races.d_incumbent == 0) & (races.open_seat == 0))
    races["tier"] = np.select([effective >= t for t, _ in TIERS], [name for _, name in TIERS], "Safe R")

    total = races.floor_d + races.floor_r + races.dccc + races.nrcc
    keep = (total >= MIN_TOTAL_SPEND) & ~races.district_id.str[:2].isin(EXCLUDED_STATES)
    return races[keep].reset_index(drop=True)[[
        "cycle", "district_id", "tier", "status", "pvi", "d_incumbent", "open_seat", "floor_share", "gb",
        "disbursements_d", "disbursements_r", "outside_d", "outside_r", "dccc", "nrcc", "margin"]]
