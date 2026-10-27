"""
Download every raw input into data/raw/, skipping files already there.

All sources are public and need no API key:

- The Downballot: 2008-2024 presidential results by congressional district,
  one Google Sheet per set of district lines.
  https://www.the-downballot.com/p/the-downballots-calculations-of-presidential
- MIT Election Data and Science Lab: U.S. House returns, 1976-2024 (CC0).
  https://doi.org/10.7910/DVN/IG0UN2 . The one manual step: Dataverse asks
  for a short guestbook form before this file downloads, so it can't be
  scripted. Download 1976-2024-house.tab and save it as
  data/raw/house_1976_2024.csv.
- FEC bulk data: the all-candidate summary file (weball) and every reported
  independent expenditure, per two-year cycle.
  https://www.fec.gov/data/browse-data/?tab=bulk-data
- Jonathan Cervas: TV markets and congressional districts (2022-2026
  lines), embedded in his interactive page, which is saved as-is.
  https://jonathancervas.com/maps/media-markets/
"""

import io
import zipfile
from pathlib import Path

import requests

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
FEC_BULK = "https://www.fec.gov/files/bulk-downloads"
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
        jobs.append((f"{FEC_BULK}/{cycle}/independent_expenditure_{cycle}.csv",
                     RAW / "fec" / f"independent_expenditure_{cycle}.csv", None, cycle))
    jobs.append((MEDIA_MARKETS_URL, RAW / "media_markets.html", None, None))

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

    if not (RAW / "house_1976_2024.csv").exists():
        raise SystemExit(f"Missing House results. Download 1976-2024-house.tab from {MIT_URL} "
                         f"(Dataverse asks for a short guestbook form) and save it as "
                         f"{RAW / 'house_1976_2024.csv'}")
