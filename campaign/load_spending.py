"""
Independent expenditures in House general elections, by district and side.

- Amended reports: an amendment replaces the whole report it amends (it
  names it in prev_file_num), so rows from any report later superseded are
  dropped. Without this DCCC's 2024 total reads $103.0M; with it, $74.2M,
  matching FEC's published $74.26M (2022: $96.4M, matching exactly).
- Side: supporting a Democrat or opposing a Republican is Democratic-side
  money. The target candidate's party is missing on 10-15% of rows, and
  missing it silently drops most of Congressional Leadership Fund's 2024
  spending ($213M of $217M). When it's missing, the party comes from FEC's
  candidate list: the candidate in that district with that surname.
- DCCC's and NRCC's own spending is split out by spender ID: it is the money
  the national committees control (the allocation question). Everything
  else, mostly super PACs, is "outside" money neither committee controls.
- Fraudulent filings: the files contain a few single "expenditures" of
  $1 billion or more (a "COMMITTEE 300" reporting $10B twice "for cash in
  hand" in 2022; one person filing billions under names like "Warren
  Buffet" in 2026). No real single House expenditure in 2012-2026 tops $5M,
  so any row over $25M is dropped.
- Coordinated party spending is not in this file. It was small through 2024
  (DCCC: $5.4M in 2024), but after NRSC v. FEC (June 2026) struck down the
  limits it is how DCCC spends in 2026, so 2026 committee money is missing.
"""

import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec"
NATIONAL_COMMITTEES = {"C00000935": "dccc", "C00075820": "nrcc"}
MAX_SINGLE_EXPENDITURE = 25_000_000
PARTY = {"DEM": "D", "DFL": "D", "REP": "R"}
WORD = re.compile(r"[A-Z]{2,}")


def load_spending(cycle):
    """district_id, outside_d, outside_r, dccc, nrcc (dollars)."""
    ie = pd.read_csv(RAW / f"independent_expenditure_{cycle}.csv", dtype=str, low_memory=False)
    superseded = set(ie.prev_file_num.dropna()) & set(ie.file_num.dropna())
    ie = ie[(ie.can_office == "H") & (ie.ele_type == "G") & ~ie.file_num.isin(superseded)].copy()
    ie["amount"] = pd.to_numeric(ie.exp_amo, errors="coerce").fillna(0).abs()
    ie = ie[ie.amount <= MAX_SINGLE_EXPENDITURE]
    ie["district_id"] = ie.can_office_state.str.strip() + "-" + ie.can_office_dis.str.strip().str.zfill(2)

    stated = ie.cand_pty_aff.fillna("").str.upper()
    ie["party"] = stated.str.contains("DEMOCRAT").map({True: "D"}).fillna(stated.str.contains("REPUBLICAN").map({True: "R"}))
    candidates = pd.read_csv(RAW / f"weball{str(cycle)[2:]}.txt", sep="|", header=None, dtype=str,
                             usecols=[1, 4, 18, 19]).set_axis(["name", "party", "state", "district"], axis=1)
    candidates["party"] = candidates.party.map(PARTY)
    candidates["key"] = (candidates.state + "-" + candidates.district.str.zfill(2) + " "
                         + candidates.name.str.split(",").str[0].str.upper().str.findall(WORD).str[-1])
    by_surname = candidates.dropna().groupby("key").party.agg(lambda p: p.iloc[0] if p.nunique() == 1 else None)
    surname = ie.cand_name.fillna("").str.split(",").str[0].str.upper().str.findall(WORD).str[-1]
    ie["party"] = ie.party.fillna((ie.district_id + " " + surname).map(by_surname))

    support = ie.sup_opp.str.upper() == "S"
    side = pd.Series(None, index=ie.index, dtype=object)
    side[((ie.party == "D") & support) | ((ie.party == "R") & ~support)] = "outside_d"
    side[((ie.party == "R") & support) | ((ie.party == "D") & ~support)] = "outside_r"
    ie["bucket"] = ie.spe_id.map(NATIONAL_COMMITTEES).fillna(side)
    ie = ie[ie.bucket.notna()]

    out = ie.pivot_table(index="district_id", columns="bucket", values="amount", aggfunc="sum", fill_value=0)
    return out.reindex(columns=["outside_d", "outside_r", "dccc", "nrcc"], fill_value=0).reset_index()
