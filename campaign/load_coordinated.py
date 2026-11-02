"""
DCCC's and NRCC's coordinated spending in House races (FEC Schedule F, by
fetch.py), one row per payment. After NRSC v. FEC (June 2026) struck down the
limits, it is how both committees spend: each reports $0 in independent
expenditures for 2026.

- Race: the district of the nominee it's for, by FEC candidate ID, since a
  member can file under an old number (Moskowitz, FL-23, running in FL-25).
  Otherwise the district on the filing: a primary loser's money still belongs
  to that race.
- Channel: only media and mail can be told apart. DCCC's 2026 filings say
  "MEDIA BUY" with no separate digital label, and NRCC's say "MEDIA", so TV
  and digital can't be split, unlike in the independent expenditures.
- Date: when the committee paid, not when the ad reached voters, which Schedule
  E reports and Schedule F doesn't.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .load_candidates import load_candidates
from .load_results import load_results

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec"
PARTY = {"dccc": "D", "nrcc": "R"}


def load_coordinated(cycle):
    """district_id, bucket (dccc / nrcc), channel (Media / Mail / Other), amount, date, cand_id, purpose."""
    rows = pd.read_csv(RAW / f"coordinated_{cycle}.csv", dtype={"cand_id": str, "district": str})
    candidates = load_candidates(cycle, load_results(cycle))
    nominee = candidates.dropna(subset=["cand_id"]).drop_duplicates(["cand_id", "party"]).set_index(["cand_id", "party"]).district_id
    key = pd.MultiIndex.from_arrays([rows.cand_id, rows.bucket.map(PARTY)])
    filed = rows.state + "-" + rows.district.fillna("0").str.zfill(2)
    purpose = rows.purpose.fillna("").str.upper()
    media = purpose.str.contains("MEDIA|ADVERTIS") & ~(purpose.str.contains("PRODUCTION") & ~purpose.str.contains("BUY"))
    return rows.assign(district_id=pd.Series(nominee.reindex(key).to_numpy(), index=rows.index).fillna(filed),
                       channel=np.select([purpose.str.contains("MAIL|POSTAGE|PRINTING"), media], ["Mail", "Media"], "Other"),
                       date=pd.to_datetime(rows.date.astype(str), format="%Y%m%d"))[
        ["district_id", "bucket", "channel", "amount", "date", "cand_id", "purpose"]]


def coordinated_by_race(cycle):
    """Indexed by district_id: dccc, nrcc (dollars)."""
    rows = load_coordinated(cycle)
    return rows.pivot_table(index="district_id", columns="bucket", values="amount", aggfunc="sum",
                            fill_value=0).reindex(columns=["dccc", "nrcc"], fill_value=0)
