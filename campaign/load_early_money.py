"""
Each House campaign's money as of its July report: cycle-to-date receipts
and disbursements through June 30 of the election year, the last report
every campaign files before a September decision, and how much of the
receipts was the candidate's own money (contributions plus loans).

From the FEC API's report summaries (fetch.py), keeping each version the
FEC had received by Aug 31. An amendment replaces the report it amends, so
the latest version of each report counts.

- Runoffs and special elections: a pre-election report due near July 15
  replaces the July quarterly (a July primary runoff, say), so where a
  committee has no July report, a pre-election report ending between
  June 1 and July 15 stands in. With that, 2012's matched nominees with
  June 30 money account for 99% of nominee spending.
- Which committee: the candidate's principal or other authorized committee
  in FEC's candidate-committee links for that cycle or the next, whichever
  raised the most by June 30. The next cycle's file catches committees FEC
  links late (Gillen's 2024 campaign in NY-04 is linked only for 2026).
  Not the candidate master file's principal committee: that is the latest
  one, sometimes created after the election (Coffman's 2014 campaign filed
  under a committee later re-designated). Joint fundraising committees are
  left out; they pass their money on to the campaign.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec"
JULY = {"Q2", "M7"}
MONEY = {"total_receipts_ytd": "raised", "total_disbursements_ytd": "spent"}
OWN = ["candidate_contribution_ytd", "loans_made_by_candidate_ytd"]


def load_early_money(cycle):
    """Indexed by FEC candidate ID: raised, spent, own (dollars through June 30)."""
    reports = pd.read_csv(RAW / f"reports_{cycle}.csv", dtype={"committee_id": str})
    reports["end"] = pd.to_datetime(reports.coverage_end_date.str[:10])
    july = reports[reports.report_type.isin(JULY)]
    stand_in = reports[~reports.report_type.isin(JULY)
                       & reports.end.between(pd.Timestamp(f"{cycle}-06-01"), pd.Timestamp(f"{cycle}-07-15"))]
    stand_in = stand_in[~stand_in.committee_id.isin(july.committee_id)]
    latest = (pd.concat([july, stand_in]).sort_values(["end", "receipt_date", "file_number"])
              .drop_duplicates("committee_id", keep="last"))

    files = [RAW / f"ccl{str(c)[2:]}.txt" for c in (cycle, cycle + 2)]
    links = pd.concat(pd.read_csv(f, sep="|", header=None, dtype=str, usecols=[0, 3, 5],
                                  names=["cand_id", "committee_id", "designation"]) for f in files if f.exists())
    links = links[links.designation.isin(["P", "A"])].drop_duplicates(["cand_id", "committee_id"])
    money = latest.merge(links[["cand_id", "committee_id"]], on="committee_id").rename(columns=MONEY)
    money["own"] = money[OWN].fillna(0).sum(axis=1)
    money = money[["cand_id", "raised", "spent", "own"]].fillna(0)
    return money.sort_values("raised").drop_duplicates("cand_id", keep="last").set_index("cand_id")
