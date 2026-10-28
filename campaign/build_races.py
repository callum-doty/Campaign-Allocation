"""
One row per House race, with every feature the model uses.

Features:
- pvi: partisan lean on the race's own lines (load_pvi).
- d_incumbent / open_seat: from FEC's incumbent flag on the matched
  nominees; a Republican incumbent is the baseline.
- fresh_d / fresh_r, over_inc / over_open: first-term incumbents, and how
  far the seat's party ran ahead of its lean two years earlier
  (incumbent_history.py).
- floor_share = log((F_D + k) / (F_D + F_R + 2k)), F = what that side's
  nominee had raised from others by June 30 (the July FEC report, less the
  candidate's own contributions and loans) plus outside
  (non-national-committee) money that had reached voters by Aug 31, k = $10K.
  The share of money neither national committee controls, standing in for
  candidate strength and how hard each side is contesting the race, counted
  only as far as a forecaster deciding on Sept 1 could see it: most
  candidate and outside spending comes in the fall, partly following how
  the race is going. It is a forecasting feature, not a causal effect.
- gb: the generic-ballot polling average heading into the election. Its
  values on earlier dates of the campaign are in GENERIC_BALLOT_DATED
  (dated_forecast.py).
- dccc / nrcc: the national committees' own spending (the allocation question).
- tier: a label from PVI and incumbency (incumbent worth +/-2 points), for
  reporting only; the model doesn't use it.

Races with under $100K of total spending over the whole election are
dropped (mostly uncontested seats; this only picks which races are
modeled), and so is Alaska (top-four primary and ranked-choice general).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .incumbent_history import incumbent_history
from .load_candidates import load_candidates
from .load_early_money import load_early_money
from .load_pvi import load_pvi
from .load_results import load_results
from .load_spending import load_spending

GENERIC_BALLOT = Path(__file__).resolve().parents[1] / "data" / "generic_ballot.csv"
GENERIC_BALLOT_DATED = Path(__file__).resolve().parents[1] / "data" / "generic_ballot_dated.csv"
MIN_TOTAL_SPEND = 100_000
EXCLUDED_STATES = {"AK"}
FLOOR_PSEUDO_DOLLARS = 10_000
DECISION = "08-31"                 # money visible by the end of August counts for a Sept 1 decision
TIERS = [(10, "Safe D"), (5, "Likely D"), (1, "Lean D"), (-3, "Toss-Up"), (-5, "Lean R"), (-10, "Likely R")]


def build_races(cycle):
    results = load_results(cycle)
    candidates = load_candidates(cycle, results).pivot(index="district_id", columns="party")
    candidates.columns = [f"{field}_{party.lower()}" for field, party in candidates.columns]
    races = (load_pvi(cycle).rename_axis("district_id").reset_index()
             .merge(results[["district_id", "margin"]], on="district_id", how="left")
             .merge(load_spending(cycle, pd.Timestamp(f"{cycle}-{DECISION}")), on="district_id", how="left")
             .merge(candidates, left_on="district_id", right_index=True, how="left"))
    early = load_early_money(cycle)
    for side in ("d", "r"):
        ids = races.get(f"cand_id_{side}", pd.Series(None, index=races.index, dtype=object))
        for field in ("raised", "spent", "own"):
            races[f"{field}_jun_{side}"] = ids.map(early[field])
    for col in ("outside_d", "outside_r", "dccc", "nrcc", "disbursements_d", "disbursements_r", "outside_aug_d",
                "outside_aug_r", "raised_jun_d", "raised_jun_r", "spent_jun_d", "spent_jun_r", "own_jun_d", "own_jun_r"):
        races[col] = races.get(col, pd.Series(0.0, index=races.index)).fillna(0.0)

    races["cycle"] = cycle
    ballot = pd.read_csv(GENERIC_BALLOT).set_index("cycle").generic_ballot
    races["gb"] = ballot[cycle]
    races["d_incumbent"] = (races.get("ici_d") == "I").astype(float)
    races["open_seat"] = ((races.d_incumbent == 0) & (races.get("ici_r") != "I")).astype(float)
    races["status"] = np.select([races.d_incumbent == 1, races.open_seat == 1], ["D incumbent", "Open"], "R incumbent")
    races = races.join(incumbent_history(cycle, races, ballot.get(cycle - 2)))
    for side in ("d", "r"):
        raised = (races[f"raised_jun_{side}"] - races[f"own_jun_{side}"]).clip(lower=0)
        races[f"floor_{side}"] = raised + races[f"outside_aug_{side}"]
    k = FLOOR_PSEUDO_DOLLARS
    races["floor_share"] = np.log((races.floor_d + k) / (races.floor_d + races.floor_r + 2 * k))
    effective = races.pvi + 2 * races.d_incumbent - 2 * ((races.d_incumbent == 0) & (races.open_seat == 0))
    races["tier"] = np.select([effective >= t for t, _ in TIERS], [name for _, name in TIERS], "Safe R")

    total = races[["disbursements_d", "disbursements_r", "outside_d", "outside_r", "dccc", "nrcc"]].sum(axis=1)
    keep = (total >= MIN_TOTAL_SPEND) & ~races.district_id.str[:2].isin(EXCLUDED_STATES)
    return races[keep].reset_index(drop=True)[[
        "cycle", "district_id", "tier", "status", "pvi", "d_incumbent", "open_seat", "floor_share",
        "fresh_d", "fresh_r", "over_inc", "over_open", "gb",
        "disbursements_d", "disbursements_r", "outside_d", "outside_r", "dccc", "nrcc", "margin",
        "raised_jun_d", "raised_jun_r", "spent_jun_d", "spent_jun_r", "own_jun_d", "own_jun_r",
        "outside_aug_d", "outside_aug_r"]]
