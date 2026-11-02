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
then. Either file can misspell a name or shorten it (MIT's JEFF DENHEM and
DOUG LA MALFA, FEC's DELARUO for DeLauro, MADELEINE DEAN CUNNANE filed as
DEAN), so within the district a surname a few letters off counts if a given
name agrees too. After redistricting FEC often keeps a member's old district
number (Schweikert ran in AZ-01 in 2022 but is filed under AZ-06), so a
nominee with no match in the district is matched elsewhere in the state when
the surname and first name or nickname both agree. Where a district holds
several matches, FEC's surname and a first initial agreeing beats the surname
appearing anywhere in the FEC name (the ballot's "RICK" JOHN isn't FLEMING,
JOHN C); after that the record with the most money counts, since FEC can
hold two for one person (AL MCAFFREY). FEC also flags some members running
again as challengers or open-seat candidates, apparently by their later
status (Schneider, Garcia, Gallego and Shea-Porter, who all lost in 2014;
Doggett on his new TX-37 in 2022), so a nominee who won the state's
previous general election for the same party, matched by surname and first
name or nickname, is the incumbent. A party with more than one name on the
ballot (California's top two, Louisiana's November election) is represented
by its incumbent if it has one, then by its biggest spender. FEC's summary
file for a year leaves out some candidates FEC hasn't linked to a committee
(Self and Ivey in 2024). If a party's only names are those, another year's
file gives the ID when that campaign filed reports this year, and its
spending through June 30 stands in for the total. A nominee still unmatched
spent too little to file, and counts as $0. For a cycle with no published results yet (2026), the top D
and R spender per district stand in for the nominees, after dropping
"incumbents" who are no longer House members (see below), except where
data/nominees_{cycle}.csv names the actual nominees, checked by hand with
sources, along with their incumbency. FEC lists some current members as
challengers or open-seat candidates (a dozen in 2026: Self in TX-03,
Menendez in NJ-08, Casar running in the new TX-37), so a nominee who is a
current member from the state, with the same party, surname and first name
or nickname, is counted as the incumbent. That list covers districts where the
top spender had no House report through June 30: a Senate run (Weil,
FL-06; Cotton, AR-04), a campaign that ended (Chatzky, NY-17), or a primary
loser. Several of the right nominees were incumbents (Clyde, GA-09;
Foushee, NC-04; Fine, FL-06, whom FEC still lists as a challenger).
"""

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .load_early_money import load_early_money
from .load_results import load_results
from .lookup_quality import SUFFIXES as FOLDED_SUFFIXES, fold, name_parts, same_first, split_name

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec"
SHEETS = Path(__file__).resolve().parents[1] / "data" / "raw" / "presidential"
DATA = Path(__file__).resolve().parents[1] / "data"
DISTRICT = re.compile(r"^[A-Z]{2}-(\d{1,2}|AL)$")
COLUMNS = {0: "cand_id", 1: "name", 2: "ici", 4: "party_code", 7: "disbursements", 18: "state", 19: "district"}
PARTY = {"DEM": "D", "DFL": "D", "REP": "R"}
SUFFIXES = {"JR", "SR", "II", "III", "IV", "MR", "MRS", "MS", "DR", "HON", "THE", "MD"}
WORD = re.compile(r"[A-Z]{2,}")
SENATE_RUN_DOLLARS = 100_000
CLOSE_SURNAME = 0.75


def same_member(name, member):
    """Whether a nominee's name (FEC's or the ballot's) is the member's: 'ONDER JR, ROBERT FRANK' is Bob Onder,
    but 'HOOPER, JAMES D' is not John James."""
    firsts, _, surname = name_parts(name)
    member_firsts, _, member_surname = name_parts(member)
    return bool(member_surname) and member_surname[-1] in surname and \
        any(same_first(a, b) for a in firsts for b in member_firsts)


def close_name(fec_name, ballot):
    """For a name one file misspells or shortens: a surname a few letters off ('DENHAM, JEFF' and 'JEFF DENHEM', 'DEAN,
    MADELEINE' and 'MADELEINE DEAN CUNNANE', 'LAMALFA, DOUG' and 'DOUG LA MALFA') and a given name in common
    ('HIGGINS, CAPTAIN CLAY')."""
    surname, given = split_name(fec_name)
    surname = fold(surname).replace(" ", "")
    words = [w for w in fold(ballot.replace("\\", "")).split() if w not in FOLDED_SUFFIXES]
    return any(SequenceMatcher(None, surname, piece).ratio() >= CLOSE_SURNAME for piece in words + ["".join(words[-2:])]) and \
        any(same_first(a, b) for a in fold(given).split() for b in words)


def other_year_records(cycle):
    """cand_id, name, state, party and spent (through June 30) of House candidates in another year's FEC summary file
    whose campaign filed reports in `cycle`, nearest year first."""
    early = load_early_money(cycle)
    files = sorted((f for f in RAW.glob("weball*.txt") if f.stem != f"weball{str(cycle)[2:]}"),
                   key=lambda f: abs(2000 + int(f.stem[6:]) - cycle))
    found = pd.concat([pd.read_csv(f, sep="|", header=None, dtype=str, usecols=[0, 1, 4, 18]) for f in files])
    found.columns = ["cand_id", "name", "party_code", "state"]
    found = found[found.cand_id.str.startswith("H") & found.cand_id.isin(early.index)].drop_duplicates("cand_id")
    return found.assign(party=found.party_code.map(PARTY), spent=found.cand_id.map(early.spent))


def load_candidates(cycle, results, every_name=False):
    """district_id, party, cand_id (FEC), disbursements, ici ('I'
    incumbent, 'C' challenger, 'O' open seat), FEC name and the name on the
    ballot (ballot_name; the hand-checked name, or missing, before results)
    for each matched nominee. `results` is load_results(cycle); pass an empty frame for an
    unfinished cycle. With `every_name`, every matched name on the ballot, not
    only the one representing its party."""
    everyone = pd.read_csv(RAW / f"weball{str(cycle)[2:]}.txt", sep="|", header=None, dtype=str,
                           usecols=list(COLUMNS)).rename(columns=COLUMNS)
    fec = everyone[everyone.cand_id.str.startswith("H")].copy()
    fec["fec_party"] = fec.party_code.map(PARTY)
    fec["district_id"] = fec.state + "-" + fec.district.str.zfill(2)
    fec["disbursements"] = pd.to_numeric(fec.disbursements, errors="coerce").fillna(0)
    columns = ["district_id", "party", "cand_id", "disbursements", "ici", "name", "ballot_name"]
    fec_ascii = (fec.name.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.upper()
                 .str.replace("'", "", regex=False))         # FEC writes both O'HALLERAN and DESPOSITO
    fec["words"] = fec_ascii.str.findall(WORD).map(set)
    fec["initial"] = fec_ascii.str.split(",").str[1].str.strip().str[:1]
    fec["surname_words"] = fec_ascii.str.split(",").str[0].str.findall(WORD).map(set)

    if results.empty:
        # No ballot yet. An "incumbent" flag only counts for a current House
        # member from that state (The Downballot's sheet lists them) who is
        # not running for Senate this cycle. That drops stale committees
        # (Sinema's in AZ-09; Ted Budd's and Wiley Nickel's in NC-13) and
        # members leaving for the Senate (Hinson, IA-02; Barr, KY-06), and
        # keeps members running on a new district number (Bera in CA-03). A
        # Senate run counts if its FEC ID was issued this cycle ("S6" for
        # 2026) and it has spent $100K+, which skips old or abandoned bids,
        # and if it isn't flagged as a senator's: "S6" also starts IDs from
        # 2006, and Bob Menendez's still spends, which made his son (NJ-08)
        # look like a Senate candidate.
        sheet = [row for row in csv.reader((SHEETS / f"lines_{cycle}.csv").open(encoding="utf-8"))
                 if len(row) > 2 and DISTRICT.match(row[0].strip()) and WORD.findall(row[1].upper())]
        members = {(row[0][:2], WORD.findall(row[1].upper())[-1]) for row in sheet}
        senate = everyone[everyone.cand_id.str.startswith(f"S{str(cycle)[-1]}")
                          & (pd.to_numeric(everyone.disbursements, errors="coerce") >= SENATE_RUN_DOLLARS)
                          & (everyone.ici != "I")]
        running = {(s, WORD.findall(n.split(",")[0].upper())[-1], n.split(",")[1].strip()[:1])
                   for s, n in zip(senate.state, senate.name) if "," in n and WORD.findall(n.split(",")[0].upper())}
        is_member = [any((s, w) in members for w in words) for s, words in zip(fec.state, fec.words)]
        for_senate = [any((s, w, i) in running for w in words) for s, words, i in zip(fec.state, fec.words, fec.initial)]
        # FEC also lists some current members as challengers or open-seat
        # candidates (Self, TX-03; Menendez, NJ-08; Onder, MO-03; Casar on his
        # new TX-37). A nominee is the incumbent when a member from the state
        # has the same party, surname and first name or nickname.
        sitting = [not r and any(row[0][:2] == s and row[2].strip("() ") == party and same_member(name, row[1])
                                 for row in sheet)
                   for s, party, name, r in zip(fec.state, fec.fec_party, fec.name, for_senate)]
        fec.loc[sitting, "ici"] = "I"
        current = pd.Series([ici != "I" or (m and not r) for ici, m, r in zip(fec.ici, is_member, for_senate)],
                            index=fec.index)
        top = fec[fec.fec_party.notna() & current].sort_values("disbursements", ascending=False, kind="stable")
        guess = top.drop_duplicates(["district_id", "fec_party"]).rename(columns={"fec_party": "party"}).assign(
            ballot_name=None)[columns]
        listed = DATA / f"nominees_{cycle}.csv"
        if not listed.exists():
            return guess
        known = pd.read_csv(listed, dtype=str)
        actual = known[["district_id", "party", "cand_id", "ici", "name"]].rename(columns={"name": "ballot_name"}).merge(
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
    pairs = pairs[(pairs.fec_party == pairs.party) | pairs.fec_party.isna()]
    named = [s in w for s, w in zip(pairs.surname, pairs.words)]
    agrees = [n and same_member(f, b) for n, f, b in zip(named, pairs.name, pairs.ballot_name)]
    leading = [s in w and (i == j or a) for s, w, i, j, a in zip(pairs.surname, pairs.surname_words, pairs.initial, pairs.initial_fec, agrees)]
    # 3: FEC's surname and a first initial or name agree; 2: the surname is somewhere in the FEC name; 1: a close name in the district
    pairs = pairs.assign(fit=[3 if l else 2 if n else 1 if here and close_name(f, b) else 0
                              for l, n, here, f, b in zip(leading, named, pairs.in_district, pairs.name, pairs.ballot_name)])
    found = pairs[(pairs.in_district & (pairs.fit > 0)) | pd.Series(agrees, index=pairs.index)]
    records = found.sort_values(["in_district", "fit", "disbursements"], ascending=False, kind="stable").drop_duplicates(
        ["district_id", "party", "ballot_name"]).copy()

    # members FEC flags as challengers or open-seat candidates (see above)
    prior = load_results(cycle - 2).dropna(subset=["winner"])
    won = {}
    for d, m, w in zip(prior.district_id, prior.margin, prior.winner):
        won.setdefault((d[:2], "D" if m > 0 else "R"), []).append(w)

    def returning(state, party, *names):
        return any(same_member(n, w) for n in names for w in won.get((state, party), []))

    records.loc[[returning(s, p, b, n) for s, p, b, n in zip(records.state, records.party, records.ballot_name, records.name)], "ici"] = "I"
    records["sitting"] = records.ici == "I"
    nominees = records.sort_values(["sitting", "in_district", "disbursements"], ascending=False, kind="stable").drop_duplicates(
        ["district_id", "party"])[columns]
    key = ["district_id", "party", "ballot_name"] if every_name else ["district_id", "party"]
    nominees = records[columns] if every_name else nominees

    left = ballot[~ballot.set_index(key).index.isin(nominees.set_index(key).index)]
    if left.empty:
        return nominees
    elsewhere = other_year_records(cycle)
    filled = []
    for r in left.itertuples():
        pool = elsewhere[(elsewhere.state == r.state) & (elsewhere.party == r.party)]
        pool = pool[[same_member(n, r.ballot_name) for n in pool.name]]
        if not pool.empty:
            filled.append({"district_id": r.district_id, "party": r.party, "cand_id": pool.cand_id.iloc[0], "disbursements": pool.spent.iloc[0],
                           "ici": "I" if returning(r.state, r.party, r.ballot_name) else None, "name": pool.name.iloc[0],
                           "ballot_name": r.ballot_name})
    if not filled:
        return nominees
    filled = pd.DataFrame(filled).sort_values("disbursements", ascending=False, kind="stable").drop_duplicates(key)
    return pd.concat([nominees, filled[columns]], ignore_index=True)
