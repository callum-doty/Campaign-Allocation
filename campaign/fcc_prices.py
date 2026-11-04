"""
Does the same airtime cost different sponsors different amounts? The allocation model counts a
party dollar the same however it's spent. Political files show the rate each sponsor was
charged, so this compares rates for the same station, time slot, days, week and spot length.

Fixed from a six-station pilot, before scaling up:
- Four comparisons: coordinated against candidate, party independent against candidate, party
  independent against coordinated, and outside groups against candidate.
- Within LOWEST_UNIT_DAYS of the general election, when candidates are due a station's lowest
  rate (from general knowledge of FCC rules, not this data), and separately before that.
- Only :30 spots, or lines that don't state a length.
- Intervals resample stations, since one station's slots aren't independent of each other.

A rate says what a spot cost, not what it delivered. The cheapest spots can be bumped for a
higher bidder, so a price gap may overstate the difference in what actually aired.

Added after seeing the first read: `typical`, the median of stations' own medians, since the
pooled median leans on the busiest stations, and the checks in `price_evidence`. Added after the
final read: `busiest_station`, since one station held over a third of a comparison.
"""

import hashlib
import re
import sys
import time

import numpy as np
import pandas as pd

from .ballot_timing import election_day
from .fetch_fcc import DOWNLOAD_WORKERS, RAW, download, file_path
from .implied_belief import implied_belief
from .load_fcc import PARTY, days_mask, load_lines, read_text, sponsor_kind
from .reallocation_gain import reallocation_gain
from .spending_effect import HALF_SATURATION

LOWEST_UNIT_DAYS = 60
COMPARISONS = [("coordinated", "candidate"), ("party_independent", "candidate"), ("party_independent", "coordinated"), ("outside", "candidate")]
DRAWS = 2000
HOUSE_GROUPS = re.compile(r"house majority|congressional leadership|\bclf\b|american action network|united democracy|\budp\b", re.I)
BIG_FOUR = re.compile(r"ABC|CBS|NBC|FOX", re.I)
SAMPLE_DAYS_BEFORE, SAMPLE_DAYS_AFTER = 75, 45
PER_FOLDER = 6
PATIENCE, COOL_OFF = 8, 1800
COMPARATORS = {"coordinated", "party_independent", "outside"}
NO_PAIRS = {"n": 0, "stations": 0, "median": np.nan, "low": np.nan, "high": np.nan, "above": np.nan, "typical": np.nan, "typical_low": np.nan, "typical_high": np.nan}


def price_sample(manifest, stations, cycle, per_folder=PER_FOLDER, seed=0):
    """The files worth reading for prices, in random order: PDFs uploaded around the election at big-four affiliates, from candidates,
    parties and the main House groups, at stations with a candidate and at least one other kind, up to `per_folder` from each sponsor.
    `stations`: stations.csv indexed by id. A price needs only one document per slot, so the full 100GB isn't needed for this."""
    m = manifest[manifest.extension.str.lower() == "pdf"].drop_duplicates("file_id").copy()
    m["kind"] = [sponsor_kind(a, b) for a, b in zip(m.section, m.sponsor)]
    uploaded = pd.to_datetime(m.uploaded, errors="coerce", utc=True).dt.tz_localize(None)
    day = pd.Timestamp(election_day(cycle))
    near = uploaded.between(day - pd.Timedelta(days=SAMPLE_DAYS_BEFORE), day + pd.Timedelta(days=SAMPLE_DAYS_AFTER))
    big = m.entity_id.map(stations.network).fillna("").str.contains(BIG_FOUR)
    wanted = m.kind.isin(["candidate", "coordinated", "party_independent"]) | ((m.kind == "outside") & m.sponsor.fillna("").str.contains(HOUSE_GROUPS))
    use = m[near & big & wanted]
    kinds = use.groupby("station").kind.agg(set)
    fit = kinds[kinds.map(lambda held: "candidate" in held and bool(held & COMPARATORS))].index
    use = use[use.station.isin(fit)].sample(frac=1.0, random_state=seed)
    return use.groupby(["station", "sponsor"]).head(per_folder)


def slot_cells(lines):
    """Median rate by station, slot, days, week and sponsor kind, one column per kind."""
    usable = lines[lines.seconds.isna() | (lines.seconds == 30)]
    usable = usable.assign(mask=usable.days.map(days_mask))
    return usable.groupby(["station", "slot", "mask", "week", "kind"]).rate.median().unstack("kind")


def price_ratio(cells, a, b, draws=DRAWS, seed=0):
    """Rate of `a` over rate of `b` in slots both bought: n, stations, median, low, high (95%, resampling stations), above (share over 1.1),
    and typical, typical_low, typical_high: the same for the median of stations' own medians."""
    if a not in cells or b not in cells:
        return dict(NO_PAIRS)
    pairs = cells[[a, b]].dropna()
    if pairs.empty:
        return dict(NO_PAIRS)
    ratio = (pairs[a] / pairs[b]).rename("ratio").reset_index()
    groups = [g.ratio.to_numpy() for _, g in ratio.groupby("station")]
    by_station = np.array([np.median(g) for g in groups])
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, len(groups), len(groups)) for _ in range(draws)]
    medians = [np.median(np.concatenate([groups[i] for i in p])) for p in picks]
    typical = [np.median(by_station[p]) for p in picks]
    return {"n": len(ratio), "stations": len(groups), "median": float(ratio.ratio.median()), "low": float(np.quantile(medians, 0.025)),
            "high": float(np.quantile(medians, 0.975)), "above": float((ratio.ratio > 1.1).mean()), "typical": float(np.median(by_station)),
            "typical_low": float(np.quantile(typical, 0.025)), "typical_high": float(np.quantile(typical, 0.975))}


def periods(lines, cycle):
    """Slot cells for {"late": within LOWEST_UNIT_DAYS of the election, "early": before that}."""
    start = pd.Timestamp(election_day(cycle)) - pd.Timedelta(days=LOWEST_UNIT_DAYS)
    cells = slot_cells(lines)
    weeks = cells.index.get_level_values("week")
    return {"late": cells[weeks >= start], "early": cells[weeks < start]}


def price_table(lines, cycle):
    """One row per comparison and period: late (within LOWEST_UNIT_DAYS of the election) and early."""
    rows = []
    for period, part in periods(lines, cycle).items():
        for a, b in COMPARISONS:
            rows.append({"cycle": cycle, "period": period, "a": a, "b": b, **price_ratio(part, a, b)})
    return pd.DataFrame(rows)


def party_files(manifest):
    """PDFs in a year's full listing by sponsor kind: files and the stations holding them. party_unclear: unknowns whose folder names a party."""
    m = manifest[manifest.extension.str.lower() == "pdf"].drop_duplicates("file_id")
    kinds = pd.Series([sponsor_kind(a, b) for a, b in zip(m.section, m.sponsor)], index=m.index, name="kind")
    names_party = pd.Series([bool(PARTY.search(str(n))) for n in m.sponsor], index=m.index)
    kinds = kinds.where(~((kinds == "unknown") & names_party), "party_unclear")
    return pd.DataFrame({"files": kinds.value_counts(), "stations": m.entity_id.groupby(kinds).nunique()})


def folder_check(sample, lines, cycle):
    """Do folder names hold up? From each document in `sample` on disk: `repeats`: documents filed byte for byte under two kinds of sponsor at one station; `names_party`:
    share of each kind's documents naming DCCC or NRCC; `without`: the late comparisons with candidate-folder documents that name one left out."""
    rows = []
    for row in sample.itertuples():
        path = file_path(row)
        if path.exists():
            rows.append({"file_id": row.file_id, "station": row.station, "kind": row.kind, "digest": hashlib.md5(path.read_bytes()).hexdigest(), "party": bool(PARTY.search(read_text(path)))})
    docs = pd.DataFrame(rows)
    named = docs.file_id[(docs.kind == "candidate") & docs.party]
    late = periods(lines[~lines.file_id.isin(named)], cycle)["late"]
    return {"repeats": int((docs.groupby(["station", "digest"]).kind.transform("nunique") > 1).sum()), "names_party": docs.groupby("kind").party.mean(),
            "without": {a: price_ratio(late, a, "candidate") for a in ("coordinated", "party_independent", "outside")}}


def stations_in_both(late, a, b="candidate"):
    """`price_ratio` per year, using only stations with a same-slot pair in each year. `late`: cycle -> late slot cells."""
    held = [set(c[[a, b]].dropna().index.get_level_values("station")) if a in c and b in c else set() for c in late.values()]
    both = set.intersection(*held)
    return {y: price_ratio(c[c.index.get_level_values("station").isin(both)], a, b) for y, c in late.items()}


def busiest_station(late, a, b):
    """The station with the most same-slot pairs of `a` and `b`: {"station", "n", "median", "against_candidate": its `a` over a candidate's, "without": `price_ratio`
    without it}. None if there are no pairs."""
    if a not in late or b not in late or late[[a, b]].dropna().empty:
        return None
    ratio = (late[a] / late[b]).dropna()
    where = ratio.index.get_level_values("station")
    name = ratio.groupby(where).size().idxmax()
    here = late.index.get_level_values("station") == name
    return {"station": name, "n": int((where == name).sum()), "median": float(ratio[where == name].median()),
            "against_candidate": float((late[here][a] / late[here]["candidate"]).median()), "without": price_ratio(late[~here], a, b)}


def price_evidence(cycles):
    """Everything the report prints, from the lines `fcc_prices read` last saved: {"table", "unflagged", "flag_rate", "flag_stations", "shared", "busiest", "read", "sample",
    "files", "folders"}. flag_rate: share of lines marked preemptible, by sponsor kind, at stations that mark any."""
    lines = {y: pd.read_csv(RAW / f"fcc_lines_{y}.csv.gz", parse_dates=["week"], dtype={"file_id": str}) for y in cycles}
    read = {y: pd.read_csv(RAW / f"fcc_docs_{y}.csv.gz", dtype={"file_id": str}) for y in cycles}
    sample = {y: pd.read_csv(RAW / f"sample_{y}.csv", dtype=str) for y in cycles}
    flagging = {y: x[x.station.isin(x.station[x.preemptible])] for y, x in lines.items()}
    return {"table": pd.concat(price_table(lines[y], y) for y in cycles),
            "unflagged": pd.concat(price_table(lines[y][~lines[y].preemptible], y) for y in cycles),
            "flag_rate": {y: x.groupby("kind").preemptible.mean() for y, x in flagging.items()},
            "flag_stations": {y: x.station.nunique() for y, x in flagging.items()},
            "shared": stations_in_both({y: periods(lines[y], y)["late"] for y in cycles}, "outside"),
            "busiest": {y: busiest_station(periods(lines[y], y)["late"], "party_independent", "coordinated") for y in cycles},
            "read": read, "sample": {y: len(s) for y, s in sample.items()},
            "files": {y: party_files(pd.read_csv(RAW / f"manifest_{y}.csv", dtype=str)) for y in cycles},
            "folders": {y: folder_check(sample[y][sample[y].file_id.isin(read[y].file_id)], lines[y], y) for y in cycles}}


def airtime_multiple(media_share, premium):
    """How many dollars of independent buying one coordinated dollar matches, if its media share buys `premium` times the airtime and the rest buys the same."""
    return 1 + media_share * (premium - 1)


def reading_at(mu, sigma, spent, multiple, cap_share):
    """Reallocation gain in expected seats and the 80%+ multiplier for `spent`, with each dollar going `multiple` times as far along the spending curve."""
    half_saturation = HALF_SATURATION / multiple
    gain, _ = reallocation_gain(mu, sigma, spent, cap_share * spent.sum(), half_saturation=half_saturation)
    return {"gain": gain, "multiplier": implied_belief(mu, sigma, spent, half_saturation=half_saturation).multiplier["80%+"]}


if __name__ == "__main__":
    command, years = sys.argv[1], [int(y) for y in sys.argv[2:]]
    every_station = pd.read_csv(RAW / "stations.csv", dtype=str).drop_duplicates("id").set_index("id")
    for y in years:
        path = RAW / f"sample_{y}.csv"
        if not path.exists():
            price_sample(pd.read_csv(RAW / f"manifest_{y}.csv", dtype=str), every_station, y).to_csv(path, index=False)
        sample = pd.read_csv(path, dtype=str)
        if command == "download":
            for attempt in range(PATIENCE):
                saved, failed, refused = download(sample, DOWNLOAD_WORKERS)
                print(y, len(sample), "files", saved, "saved", failed, "failed", flush=True)
                if not refused:
                    break
                time.sleep(COOL_OFF)
        elif command == "read":
            docs, lines = load_lines(y, sample)
            print(y, "documents", len(docs), "readable", round(docs.readable.mean(), 3), "with lines", round((docs.lines > 0).mean(), 3), "distinct lines", len(lines), flush=True)
            print(price_table(lines, y).round(2).to_string(index=False), flush=True)
