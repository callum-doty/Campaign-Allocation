"""
Each district's general-election nominees from the FEC all-candidate file
(weball): their total disbursements and incumbent/challenger flag.

FEC lists every registered House candidate, including primary losers and
House members running for another office whose committee still sits under
their old House ID (Gallego in AZ-03 and Schiff in CA-30 in 2024; a stale
Sinema ID in AZ-09). So each FEC candidate is matched to a name actually on
the November ballot (MIT): the ballot surname must appear somewhere in the
FEC name, after stripping accents ("SÁNCHEZ" vs "SANCHEZ"). Searching the
whole FEC name, not just the part before the comma, catches married names
("MATTELIN, CAITLIN RAE DRAPER" is the ballot's Caitlin Draper).
Apostrophes are dropped on both sides, because FEC spells the same kind of
name both ways ("O'HALLERAN, TOM" but "DESPOSITO, ANTHONY P"). FEC's party
code is sometimes unknown ("UNK"); a name match in the district is enough
then. After redistricting FEC often keeps a member's old district number
(Schweikert ran in AZ-01 in 2022 but is filed under AZ-06), so a nominee
with no match in the district is matched elsewhere in the state when the
surname and first initial both agree. A nominee still unmatched spent too
little to file or is missing from FEC's summary file (DeLauro in 2016), and
counts as $0. For a cycle with no published results yet (2026), the top D
and R spender per district stand in for the nominees, after dropping
"incumbents" who are no longer House members (see below), except where
data/nominees_{cycle}.csv names the actual nominees, checked by hand with
sources, along with their incumbency. That list covers districts where the
top spender had no House report through June 30: a Senate run (Weil,
FL-06; Cotton, AR-04), a campaign that ended (Chatzky, NY-17), or a primary
loser. Several of the right nominees were incumbents (Clyde, GA-09;
Foushee, NC-04; Fine, FL-06, whom FEC still lists as a challenger).
"""

import csv
import re
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec"
SHEETS = Path(__file__).resolve().parents[1] / "data" / "raw" / "presidential"
DATA = Path(__file__).resolve().parents[1] / "data"
DISTRICT = re.compile(r"^[A-Z]{2}-(\d{1,2}|AL)$")
COLUMNS = {0: "cand_id", 1: "name", 2: "ici", 4: "party_code", 7: "disbursements", 18: "state", 19: "district"}
PARTY = {"DEM": "D", "DFL": "D", "REP": "R"}
SUFFIXES = {"JR", "SR", "II", "III", "IV", "MR", "MRS", "MS", "DR", "HON", "THE", "MD"}
WORD = re.compile(r"[A-Z]{2,}")
SENATE_RUN_DOLLARS = 100_000


def load_candidates(cycle, results):
    """district_id, party, cand_id (FEC), disbursements, ici ('I'
    incumbent, 'C' challenger, 'O' open seat) and FEC name for each matched
    nominee. `results` is load_results(cycle); pass an empty frame for an
    unfinished cycle."""
    everyone = pd.read_csv(RAW / f"weball{str(cycle)[2:]}.txt", sep="|", header=None, dtype=str,
                           usecols=list(COLUMNS)).rename(columns=COLUMNS)
    fec = everyone[everyone.cand_id.str.startswith("H")].copy()
    fec["fec_party"] = fec.party_code.map(PARTY)
    fec["district_id"] = fec.state + "-" + fec.district.str.zfill(2)
    fec["disbursements"] = pd.to_numeric(fec.disbursements, errors="coerce").fillna(0)
    columns = ["district_id", "party", "cand_id", "disbursements", "ici", "name"]
    fec_ascii = (fec.name.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.upper()
                 .str.replace("'", "", regex=False))         # FEC writes both O'HALLERAN and DESPOSITO
    fec["words"] = fec_ascii.str.findall(WORD).map(set)
    fec["initial"] = fec_ascii.str.split(",").str[1].str.strip().str[:1]

    if results.empty:
        # No ballot yet. An "incumbent" flag only counts for a current House
        # member from that state (The Downballot's sheet lists them) who is
        # not running for Senate this cycle. That drops stale committees
        # (Sinema's in AZ-09; Ted Budd's and Wiley Nickel's in NC-13) and
        # members leaving for the Senate (Hinson, IA-02; Barr, KY-06), and
        # keeps members running on a new district number (Bera in CA-03). A
        # Senate run counts if its FEC ID was issued this cycle ("S6" for
        # 2026) and it has spent $100K+, which skips old or abandoned bids.
        members = {(row[0][:2], WORD.findall(row[1].upper())[-1])
                   for row in csv.reader((SHEETS / f"lines_{cycle}.csv").open(encoding="utf-8"))
                   if len(row) > 1 and DISTRICT.match(row[0].strip()) and WORD.findall(row[1].upper())}
        senate = everyone[everyone.cand_id.str.startswith(f"S{str(cycle)[-1]}")
                          & (pd.to_numeric(everyone.disbursements, errors="coerce") >= SENATE_RUN_DOLLARS)]
        running = {(s, WORD.findall(n.split(",")[0].upper())[-1], n.split(",")[1].strip()[:1])
                   for s, n in zip(senate.state, senate.name) if "," in n and WORD.findall(n.split(",")[0].upper())}
        is_member = [any((s, w) in members for w in words) for s, words in zip(fec.state, fec.words)]
        for_senate = [any((s, w, i) in running for w in words) for s, words, i in zip(fec.state, fec.words, fec.initial)]
        current = pd.Series([ici != "I" or (m and not r) for ici, m, r in zip(fec.ici, is_member, for_senate)],
                            index=fec.index)
        top = fec[fec.fec_party.notna() & current].sort_values("disbursements", ascending=False)
        guess = top.drop_duplicates(["district_id", "fec_party"]).rename(columns={"fec_party": "party"})[columns]
        listed = DATA / f"nominees_{cycle}.csv"
        if not listed.exists():
            return guess
        known = pd.read_csv(listed, dtype=str)
        actual = known[["district_id", "party", "cand_id", "ici"]].merge(
            fec.drop_duplicates("cand_id")[["cand_id", "disbursements", "name"]], on="cand_id")
        keep = ~guess.set_index(["district_id", "party"]).index.isin(known.set_index(["district_id", "party"]).index)
        return pd.concat([guess[keep], actual[columns]], ignore_index=True)

    ballot = pd.concat([results[["district_id", col]].explode(col).rename(columns={col: "ballot_name"}).assign(party=party)
                        for party, col in (("D", "d_names"), ("R", "r_names"))]).dropna()
    ballot_words = (ballot.ballot_name.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
                    .str.upper().str.replace("'", "", regex=False).str.findall(WORD).map(lambda words: [w for w in words if w not in SUFFIXES] or [""]))
    ballot["surname"], ballot["initial"] = ballot_words.str[-1], ballot_words.str[0].str[:1]
    ballot["state"] = ballot.district_id.str[:2]

    pairs = ballot.merge(fec.drop(columns="district_id").rename(columns={"party_code": "fec_code"}),
                         on="state", suffixes=("", "_fec"))
    pairs["fec_district"] = pairs.state + "-" + pairs.district.str.zfill(2)
    pairs["in_district"] = pairs.fec_district == pairs.district_id
    same_party = (pairs.fec_party == pairs.party) | pairs.fec_party.isna()
    named = pd.Series([s in w for s, w in zip(pairs.surname, pairs.words)], index=pairs.index)
    found = same_party & named & (pairs.in_district | (pairs.initial == pairs.initial_fec))
    best = pairs[found].sort_values(["in_district", "disbursements"], ascending=False)
    return best.drop_duplicates(["district_id", "party"])[columns]
