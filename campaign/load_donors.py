"""
Who gave to each House nominee by June 30, from FEC's individual contribution
files (bulk indiv, about 21GB zipped for 2012-2026, placed by hand in
data/raw/fec/indiv/ and not part of run.py), for spending_signal.py to test.

Fixed before scoring, each comparing the two sides so it's positive when it favors
the Democrat:

- donor_odds: unique itemized donors, as odds.
- local_odds: unique itemized donors in the candidate's state, as odds. FEC gives no
  district, and these files span maps, so a state is as local as this gets.
- top10_share: the difference in the share of itemized dollars from the top 10 donors.
- repeat_share: the difference in the share from donors who gave the same committee
  last cycle. 2012 has no earlier file, so it's 0 there, as vendor_retention is.
- in_state_share: the difference in the share from the candidate's state.

The traps:
- Only itemized gifts appear, generally from donors past $200 in a cycle. Small donors
  below that aren't here, and the July report's unitemized total already tested as nothing.
- FEC doesn't identify donors across filings, so a donor is a name and ZIP: last name,
  first word of the first name, 5-digit ZIP. Moving or a misspelling makes two donors.
- Each zip holds itcont.txt and the same rows again under by_date/. Only itcont.txt is read.
- Names and street-level detail aren't kept. The cache holds a hash of the donor key,
  which is enough to count donors. Don't report individual donors from these files:
  FEC restricts using them to solicit, and they aren't needed here.

    python -m campaign.load_donors
"""

import hashlib
import io
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .load_early_money import latest_july
from .load_oppexp import nominee_committees

ROOT_FEC = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec"
RAW = ROOT_FEC / "indiv"
CYCLES = (2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026)
DONOR_FEATURES = ["donor_odds", "local_odds", "top10_share", "repeat_share", "in_state_share"]
ODDS_DONORS = 10
TOP = 10
GIFTS = {"15", "15E", "22Y"}
NOT_LETTER = re.compile(r"[^A-Z ]")


def donor_key(name, zip_code):
    """A 16-character hash of last name, first word of the first name and 5-digit ZIP."""
    last, _, rest = name.upper().partition(",")
    first = NOT_LETTER.sub("", rest).split()
    key = f"{NOT_LETTER.sub('', last).strip()}|{first[0] if first else ''}|{zip_code[:5]}"
    return hashlib.blake2b(key.encode(), digest_size=8).hexdigest()


def parse_line(line):
    """One row of itcont.txt, as (committee_id, tp, state, date, amount, donor), or None if it isn't a gift from a person."""
    f = line.decode("latin-1").rstrip("\n").split("|")
    if len(f) < 21 or f[5] not in GIFTS or f[6] != "IND" or f[18] == "X":
        return None
    try:
        return f[0], f[5], f[9], f[13], float(f[14]), donor_key(f[7], f[10])
    except ValueError:
        return None


def extract(cycle, wanted):
    """Cache and return the cycle's gifts to committees in `wanted`."""
    cached = RAW / f"rows_{cycle}.csv.gz"
    if cached.exists():
        return pd.read_csv(cached, dtype={"committee_id": str, "tp": str, "state": str, "date": str, "donor": str})
    keep = {w.encode() for w in wanted}
    out = []
    with zipfile.ZipFile(RAW / f"indiv{str(cycle)[2:]}.zip") as z, z.open("itcont.txt") as f:
        for line in io.BufferedReader(f, 1 << 22):
            if line[:9] in keep:
                row = parse_line(line)
                if row is not None:
                    out.append(row)
    rows = pd.DataFrame(out, columns=["committee_id", "tp", "state", "date", "amount", "donor"])
    rows.to_csv(cached, index=False)
    return rows


def gifts_through_june(rows, cycle):
    """Rows dated from the start of the two-year cycle through June 30 of the election year. Dates are MMDDYYYY."""
    dates = pd.to_datetime(rows.date, format="%m%d%Y", errors="coerce")
    return rows[(dates >= pd.Timestamp(f"{cycle - 1}-01-01")) & (dates <= pd.Timestamp(f"{cycle}-06-30"))]


def committee_donors(rows, earlier):
    """By (committee_id, donor): amount (net of refunds, positive only), state, and repeat: gave this committee in `earlier`, the last cycle's rows."""
    by = rows.groupby(["committee_id", "donor"]).agg(amount=("amount", "sum"), state=("state", "first")).reset_index()
    by = by[by.amount > 0]
    last = set(zip(earlier.committee_id, earlier.donor)) if earlier is not None else set()
    return by.assign(repeat=[pair in last for pair in zip(by.committee_id, by.donor)])


def race_measures(donors, committee, states):
    """By race: donors, local, top10, repeat and in_state for one side. `committee`: race -> committee_id; `states`: race -> two-letter state."""
    frame = pd.DataFrame({"committee_id": committee, "home": states}).reset_index(names="race").dropna(subset=["committee_id"])
    d = donors.merge(frame, on="committee_id")
    d["local"] = d.state == d.home
    total = d.groupby("race").amount.sum()
    top = d.sort_values("amount", ascending=False).groupby("race").head(TOP).groupby("race").amount.sum()
    out = pd.DataFrame({"donors": d.groupby("race").donor.nunique(), "local": d[d.local].groupby("race").donor.nunique(),
                        "top10": top / total, "repeat": d[d.repeat].groupby("race").amount.sum() / total,
                        "in_state": d[d.local].groupby("race").amount.sum() / total})
    return out.reindex(committee.index).fillna(0.0)


def add_donors(races, rows):
    """`races` (cycle -> add_spending's frames, with cm_d and cm_r) with DONOR_FEATURES. `rows`: cycle -> extract()."""
    out = {}
    for cycle, x in races.items():
        early = gifts_through_june(rows[cycle], cycle)
        donors = committee_donors(early, rows.get(cycle - 2))
        states = x.district_id.str[:2]
        d, r = (race_measures(donors, x[f"cm_{side}"], states) for side in "dr")
        out[cycle] = x.assign(donor_odds=np.log((d.donors + ODDS_DONORS) / (r.donors + ODDS_DONORS)),
                              local_odds=np.log((d.local + ODDS_DONORS) / (r.local + ODDS_DONORS)),
                              top10_share=d.top10 - r.top10, repeat_share=d.repeat - r.repeat, in_state_share=d.in_state - r.in_state)
    return out


def coverage(rows, cycle, reports=None):
    """Median, among nominees' committees with over $5,000 itemized on the July report, of these rows' dollars through
    June 30 over that report's itemized total. The report's "ytd" fields run from the start of the two-year cycle for House committees."""
    reports = pd.read_csv(ROOT_FEC / f"reports_{cycle}.csv", low_memory=False) if reports is None else reports
    july = latest_july(reports, cycle).set_index("committee_id")
    itemized = (july.total_individual_contributions_ytd - july.individual_unitemized_contributions_ytd)
    itemized = itemized[itemized.index.isin(set(nominee_committees(cycle))) & (itemized > 5000)]
    found = gifts_through_june(rows, cycle).groupby("committee_id").amount.sum()
    return float((found.reindex(itemized.index).fillna(0.0) / itemized).median())


def load_donors():
    """cycle -> rows for every nominee's committee in any cycle, so a committee's earlier donors are found even if it wasn't a nominee's then."""
    wanted = set().union(*(set(nominee_committees(c)) for c in CYCLES))
    with ProcessPoolExecutor(4) as pool:
        return dict(zip(CYCLES, pool.map(extract, CYCLES, [wanted] * len(CYCLES))))


if __name__ == "__main__":
    for cycle, found in load_donors().items():
        print(cycle, len(found), f"{found.amount.sum() / 1e9:.2f}B")
