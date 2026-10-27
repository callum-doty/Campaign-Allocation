"""
Presidential results by congressional district, on one set of district lines.

The Downballot's sheets differ in layout (some have a row of years above the
candidate names, some don't, some add margin columns), so each election is
identified from its pair of candidate names rather than from the header.
Percentages are rounded to 0.1 point in some sheets and to whole points in
others; D two-party share is computed from whatever precision is given.
"""

import csv
import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "presidential"
ELECTIONS = {("OBAMA", "MCCAIN"): 2008, ("OBAMA", "ROMNEY"): 2012, ("CLINTON", "TRUMP"): 2016,
             ("BIDEN", "TRUMP"): 2020, ("HARRIS", "TRUMP"): 2024}
DISTRICT = re.compile(r"^([A-Z]{2})-(\d{1,2}|AL)$")


def load_presidential(lines_cycle):
    """district_id, pres_year, d_share (D / (D + R)) for the map used in
    the House election of `lines_cycle`."""
    rows = list(csv.reader((RAW / f"lines_{lines_cycle}.csv").open(newline="", encoding="utf-8")))
    header = next(r for r in rows if any(c.strip().upper() in {d for d, _ in ELECTIONS} for c in r))
    names = [c.strip().upper() for c in header]
    columns = {ELECTIONS[(names[j], names[j + 1])]: j
               for j in range(len(names) - 1) if (names[j], names[j + 1]) in ELECTIONS}
    out = []
    for r in rows:
        m = DISTRICT.match(r[0].strip()) if r else None
        if not m:
            continue
        district_id = f"{m.group(1)}-{'00' if m.group(2) == 'AL' else m.group(2).zfill(2)}"
        for year, j in columns.items():
            try:
                d, rep = float(r[j]), float(r[j + 1])
            except (ValueError, IndexError):
                continue      # older elections are only partly back-computed onto later maps
            out.append((district_id, year, d / (d + rep)))
    return pd.DataFrame(out, columns=["district_id", "pres_year", "d_share"])
