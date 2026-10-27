"""
Every independent expenditure in House general elections, cleaned, one row
per transaction.

- Amended reports: an amendment replaces the whole report it amends (it
  names it in prev_file_num), so rows from any report later superseded are
  dropped. Without this DCCC's 2024 total reads $103.0M; with it, $74.2M,
  matching FEC's published $74.26M (2022: $96.4M, matching exactly).
- Side: supporting a Democrat or opposing a Republican is Democratic-side
  money. The target candidate's party is missing on 10-15% of rows, and
  missing it silently drops most of Congressional Leadership Fund's 2024
  spending ($213M of $217M). When it's missing, the party comes from FEC's
  candidate list: the candidate in that district with that surname.
- Bucket: DCCC's and NRCC's own spending is split out by spender ID; it is
  the money the national committees control. Everything else, mostly super
  PACs, is "outside" money neither committee controls.
- Fraudulent filings: the files contain a few single "expenditures" of
  $1 billion or more (a "COMMITTEE 300" reporting $10B twice "for cash in
  hand" in 2022; one person filing billions under names like "Warren
  Buffet" in 2026). No real single House expenditure in 2012-2026 tops $5M,
  so any row over $25M is dropped.
- Channel: from the filer's stated purpose ("Media Buy", "Digital Media
  Buy Estimate", "Direct Mail - Estimate", ...). The first matching rule
  wins, so "Digital Media Buy" is digital, not TV, and "Media Production"
  is production. DCCC's plain "Media Buy" is read as TV: it labels its
  digital buys separately. A bare "MEDIA" or "Advertising" (most of it
  Congressional Leadership Fund's) could be either, and stays unspecified.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec"
NATIONAL_COMMITTEES = {"C00000935": "dccc", "C00075820": "nrcc"}
MAX_SINGLE_EXPENDITURE = 25_000_000
PARTY = {"DEM": "D", "DFL": "D", "REP": "R"}
WORD = re.compile(r"[A-Z]{2,}")
CHANNELS = [  # (channel, pattern on the lower-cased purpose), first match wins
    ("Production", r"production|creative|design"),
    ("Digital", r"digital|online|internet|\bweb|facebook|google|youtube|social media|stream|\bott\b|\bctv\b|e-?mail|\btext|\bsms"),
    ("Mail", r"mail|postage|printing|literature"),
    ("Radio", r"radio"),
    ("Field/phones", r"canvass|field|phone|\bcall|door"),
    ("TV", r"\btv\b|televis|broadcast|cable|media buy|media placement|airtime|ad buy"),
    ("Media, unspecified", r"\bmedia\b|advertis|video"),   # e.g. a bare "MEDIA": TV or digital, can't tell
]


def load_expenditures(cycle):
    """district_id, spe_id, spe_nam, bucket (dccc / nrcc / outside_d /
    outside_r), channel, amount."""
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

    purpose = ie.pur.fillna("").str.lower()
    ie["channel"] = np.select([purpose.str.contains(p, regex=True) for _, p in CHANNELS],
                              [c for c, _ in CHANNELS], "Other")
    return ie[ie.bucket.notna()][["district_id", "spe_id", "spe_nam", "bucket", "channel", "amount"]].reset_index(drop=True)
