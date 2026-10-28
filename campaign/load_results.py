"""
House general-election results by district, from MIT Election Lab.

- General election only: runoffs, special elections and write-ins dropped.
- Minnesota's Democratic-Farmer-Labor and North Dakota's Democratic-NPL
  count as Democratic; "Republican, Libertarian" as Republican.
- Fusion voting: in New York and Connecticut a candidate can appear on
  several party lines. Each candidate's votes are summed across ALL their
  lines (Working Families, Conservative, ...), not just the D or R line;
  counting only the major-party line understates NY-01 2022's Republican
  margin by 4.4 points.
- Every candidate with a D (or R) line counts toward that party, which
  handles Louisiana's multi-candidate November election and California's
  top-two (a D-vs-D race shows up as uncontested).
- Names: about 60 rows, nearly all 2022, are UTF-8 decoded twice ("TOM
  OâHALLERAN" for O'Halleran's curly apostrophe). They are repaired, and
  curly quotes straightened, so names match FEC's spelling. Otherwise
  O'Halleran (AZ-02, 2022) and García (IL-04, 2022) go unmatched and their
  money counts as $0.
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "house_1976_2024.csv"
DEM = {"DEMOCRAT", "DEMOCRATIC", "DEMOCRATIC-FARMER-LABOR", "DEMOCRATIC-NPL", "DEMOCRATIC-NONPARTISAN LEAGUE"}
REP = {"REPUBLICAN", "REPUBLICAN, LIBERTARIAN"}
QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def clean_name(name):
    """Undo UTF-8 decoded twice ("OâHALLERAN") and straighten curly quotes."""
    for wrong in ("latin-1", "cp1252"):
        try:
            name = name.encode(wrong).decode("utf-8")
            break
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name.translate(QUOTES)


def load_results(cycle):
    """district_id, d_votes, r_votes, margin (D minus R, points of the
    two-party vote), d_names, r_names (every candidate on that party's line).
    Empty for cycles MIT hasn't published yet."""
    raw = pd.read_csv(RAW, dtype={"district": str}, low_memory=False)
    keep = (raw.year == cycle) & (raw.stage == "GEN") & (raw["mode"] == "TOTAL")
    for col in ("runoff", "special", "writein"):
        keep &= ~raw[col].astype(str).str.upper().eq("TRUE")
    g = raw[keep].copy()
    g["district_id"] = g.state_po + "-" + g.district.str.zfill(2)
    g["candidate"] = g.candidate.astype(str).map(clean_name)
    g["votes"] = pd.to_numeric(g.candidatevotes, errors="coerce").fillna(0)
    label = g.party.fillna("").str.upper()

    # each candidate belongs to the major party of their biggest D/R line
    major = g[label.isin(DEM | REP)].assign(side=label.map(lambda p: "D" if p in DEM else "R"))
    side = (major.sort_values("votes").drop_duplicates(["district_id", "candidate"], keep="last")
            .set_index(["district_id", "candidate"]).side)
    totals = g.groupby(["district_id", "candidate"]).votes.sum().to_frame().join(side, how="inner").reset_index()

    out = totals.pivot_table(index="district_id", columns="side", values="votes", aggfunc="sum", fill_value=0)
    out = out.reindex(columns=["D", "R"], fill_value=0).rename(columns={"D": "d_votes", "R": "r_votes"})
    names = totals.groupby(["district_id", "side"]).candidate.apply(list).unstack()
    out["d_names"] = names.get("D")
    out["r_names"] = names.get("R")
    two_party = out.d_votes + out.r_votes
    out["margin"] = 100 * (out.d_votes - out.r_votes) / two_party.where(two_party > 0)
    return out.reset_index()
