"""
Download every raw input into data/raw/, skipping files already there.

All sources are public. One needs a free key (FEC's API, below):

- The Downballot: 2008-2024 presidential results by congressional district,
  one Google Sheet per set of district lines.
  https://www.the-downballot.com/p/the-downballots-calculations-of-presidential
- MIT Election Data and Science Lab: U.S. House returns, 1976-2024 (CC0).
  https://doi.org/10.7910/DVN/IG0UN2 . The one manual step: Dataverse asks
  for a short guestbook form before this file downloads, so it can't be
  scripted. Download 1976-2024-house.tab and save it as
  data/raw/house_1976_2024.csv.
- FEC bulk data: the all-candidate summary file (weball), the
  candidate-committee links (ccl), and every reported independent
  expenditure, per two-year cycle.
  https://www.fec.gov/data/browse-data/?tab=bulk-data
- FEC API: House campaigns' July reports (money raised and spent through
  June 30 of the election year) and the pre-election reports that replace
  them in runoff and special-election states, which the bulk files don't
  have. Needs a free key from
  https://api.data.gov/signup/ in the FEC_API_KEY environment variable.
- Jonathan Cervas: TV markets and congressional districts (2022-2026
  lines), embedded in his interactive page, which is saved as-is.
  https://jonathancervas.com/maps/media-markets/
- FRED (Federal Reserve Bank of St. Louis): the household cost basket and
  credit and labor series behind household_index.py, one CSV per series.
  https://fred.stlouisfed.org
"""

import io
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from .household_index import FRED_IDS

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# House election whose district lines the sheet describes -> (sheet id, tab gid)
DOWNBALLOT_SHEETS = {
    2012: ("1xn6nCNM97oFDZ4M-HQgoUT3X4paOiSDsRMSuxbaOBdg", "0"),           # 2012 and 2014 lines
    2016: ("1VfkHtzBTP5gf4jAu8tcVQgsBJ1IDvXEHjuMqYlOgYbA", "0"),
    2018: ("1zLNAuRqPauss00HDz4XbTH2HqsCzMe0pR8QmD1K8jk8", "0"),
    2020: ("1XbUXnI9OyfAuhP5P3vWtMuGc5UJlrhXbzZo3AwMuHtk", "0"),
    2022: ("1CKngqOp8fzU22JOlypoxNsxL6KSAH920Whc-rd7ebuM", "1871835782"),
    2024: ("1ng1i_Dm_RMDnEvauH44pgE6JCUsapcuu8F2pCfeLWFo", "620838163"),
    2026: ("1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0", "620838163"),
}
MIT_URL = "https://doi.org/10.7910/DVN/IG0UN2"
MEDIA_MARKETS_URL = "https://jonathancervas.com/maps/media-markets/"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FEC_BULK = "https://www.fec.gov/files/bulk-downloads"
FEC_REPORTS = "https://api.open.fec.gov/v1/reports/house-senate/"
JULY_REPORTS = ("Q2", "M7")    # quarterly and monthly filers' reports through June 30
STAND_INS = ("12P", "12R", "12S", "12C")   # pre-election reports, which replace the July report when due near it
REPORT_FIELDS = ["committee_id", "report_type", "coverage_end_date", "receipt_date", "file_number",
                 "total_receipts_ytd", "total_disbursements_ytd", "cash_on_hand_end_period",
                 "candidate_contribution_ytd", "loans_made_by_candidate_ytd",
                 "total_individual_contributions_ytd", "individual_unitemized_contributions_ytd"]
HEADERS = {"User-Agent": "Campaign-Allocation data fetch (github.com/callum-doty/Campaign-Allocation)"}


def fetch(cycles=(2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026), refresh=()):
    """Download what's missing. Cycles in `refresh` (e.g. a live cycle whose
    FEC filings are still coming in) are re-downloaded even if present."""
    jobs = []  # (url, destination, member of a zip to extract or None)
    for lines, (sheet, gid) in DOWNBALLOT_SHEETS.items():
        url = f"https://docs.google.com/spreadsheets/d/{sheet}/export?format=csv&gid={gid}"
        jobs.append((url, RAW / "presidential" / f"lines_{lines}.csv", None, lines))
    for cycle in cycles:
        yy = str(cycle)[2:]
        jobs.append((f"{FEC_BULK}/{cycle}/weball{yy}.zip", RAW / "fec" / f"weball{yy}.txt", f"weball{yy}.txt", cycle))
        jobs.append((f"{FEC_BULK}/{cycle}/ccl{yy}.zip", RAW / "fec" / f"ccl{yy}.txt", "ccl.txt", cycle))
        jobs.append((f"{FEC_BULK}/{cycle}/independent_expenditure_{cycle}.csv",
                     RAW / "fec" / f"independent_expenditure_{cycle}.csv", None, cycle))
    jobs.append((MEDIA_MARKETS_URL, RAW / "media_markets.html", None, None))
    jobs += [(f"{FRED_CSV}?id={sid}", RAW / "fred" / f"{sid}.csv", None, None) for sid in FRED_IDS]

    for url, dest, member, cycle in jobs:
        if dest.exists() and cycle not in refresh:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {dest.relative_to(RAW.parent)}")
        resp = requests.get(url, headers=HEADERS, timeout=600)
        resp.raise_for_status()
        if member:
            dest.write_bytes(zipfile.ZipFile(io.BytesIO(resp.content)).read(member))
        else:
            dest.write_bytes(resp.content)

    for cycle in cycles:
        dest = RAW / "fec" / f"reports_{cycle}.csv"
        if not dest.exists() or cycle in refresh:
            fetch_reports(cycle, dest)

    if not (RAW / "house_1976_2024.csv").exists():
        raise SystemExit(f"Missing House results. Download 1976-2024-house.tab from {MIT_URL} "
                         f"(Dataverse asks for a short guestbook form) and save it as "
                         f"{RAW / 'house_1976_2024.csv'}")


def fetch_reports(cycle, dest):
    """Every House campaign's July report (through June 30) and
    pre-election reports for the election year, each version the FEC
    received by Aug 31 (amendments included), with cycle-to-date totals."""
    key = os.environ.get("FEC_API_KEY")
    if not key:
        raise SystemExit("Set FEC_API_KEY to a free key from https://api.data.gov/signup/ "
                         "to download FEC report summaries.")
    rows, page, pages = [], 1, 1
    while page <= pages:
        params = {"api_key": key, "year": cycle, "committee_type": "H", "report_type": JULY_REPORTS + STAND_INS,
                  "max_receipt_date": f"{cycle}-08-31", "per_page": 100, "page": page,
                  "sort": "beginning_image_number"}   # unique per filing: paging on a tied key skips and repeats rows
        resp = requests.get(FEC_REPORTS, params=params, headers=HEADERS, timeout=120)
        if resp.status_code == 429:                  # rate limit: wait and retry the page
            time.sleep(60)
            continue
        if int(resp.headers.get("X-Ratelimit-Remaining", 10)) < 2:
            time.sleep(60)
        if not resp.ok:                              # not raise_for_status: its message would print the key
            raise SystemExit(f"FEC API error {resp.status_code} for {cycle}: {resp.text[:200]}")
        data = resp.json()
        rows += [{f: r.get(f) for f in REPORT_FIELDS} for r in data["results"]]
        pages = data["pagination"]["pages"]
        page += 1
    reports = pd.DataFrame(rows, columns=REPORT_FIELDS)
    repeats = int(reports.duplicated().sum())          # the API lists a few filings twice, identically
    if len(reports) != data["pagination"]["count"] or repeats > 0.005 * len(reports):
        raise SystemExit(f"FEC reports for {cycle}: {len(reports)} rows ({repeats} repeated), "
                         f"expected {data['pagination']['count']}; not saved")
    reports = reports.drop_duplicates()
    print(f"downloaded {len(reports)} FEC reports for {cycle}")
    reports.to_csv(dest, index=False)
