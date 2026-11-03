"""
How much of an incumbent's new district did they already represent? The forecast
marks incumbents on redrawn lines with a yes or no. A 5% trim and a mostly new
district are different things, so this measures the share and asks whether it explains
how far they ran from the forecast.

Measured block by block, in people, on the census nearest the map: each state's 2010 or
2020 redistricting file (PL 94-171) gives each block's population and its district at the
time (111th and 116th Congress), and Census's block equivalency files give the later ones.
About 2.3GB of downloads placed by hand in data/raw/census/, plus FEC's weball10.txt in
data/raw/fec/ for where members ran in 2010, not part of run.py. That covers every redraw
since 2012 but North Carolina's for 2020, which has no such file. So its members are left out
of 2022 as well, since the 116th Congress's lines aren't what they were elected on. A state
counts as redrawn if over MIN_MOVED of its people changed districts, which leaves out
Census's small corrections.

Fixed before scoring 2012, 2016 and 2018:
- the slope by redraw and pooled, against the forecast the rest of the model uses. 2012 has
  no earlier year to fit on, so it's fit on later years, which makes it the one estimate
  that isn't forward in time.
- carrying the slope forward: each redraw's slope from the earlier ones alone.
- new_voters as a forecast feature, against the same bar as everything in spending_signal.py:
  the share of the district new to its incumbent, negative for a Democrat and positive for a
  Republican, so positive favors the Democrat like the money terms. It's 0 in open seats, on
  unchanged lines, and for members with no earlier district here (elected in a special).

Tried after that, as a second form, because new_voters came out with the wrong sign: 2012's
strong incumbents all have new voters, so it also picks up the era. new_voters_within is the
same share measured against that redraw's average, so it holds only the differences among
incumbents in one redraw. It counts as another try.
"""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .fit_baseline import fit_baseline
from .load_candidates import load_candidates
from .load_results import load_results
from .predict_margin import predict_margin

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "census"
LINES = {2012: (2010, 111, 113), 2014: (2010, 113, 114), 2016: (2010, 114, 115), 2018: (2010, 115, 116),
         2022: (2020, 116, 118), 2024: (2020, 118, 119), 2026: (2020, 119, 120)}
BASE = {2010: 111, 2020: 116}
WRONG_LINES = {2022: {"NC"}}
MIN_MOVED = 0.01
FIRST_CYCLE = 2012
WEBALL = str(RAW.parent / "fec" / "weball{yy}.txt")
GEO_2020 = {2: "level", 7: "record", 9: "block", 62: "base"}
GEO_2010 = [(8, 11), (18, 25), (27, 29), (29, 32), (54, 60), (61, 65), (153, 155)]
BLOCK_LEVEL = "750"
RETAINED_BANDS = [-0.01, 0.4, 0.6, 0.8, 0.9, 1.0]
RETAINED_LABELS = ["Under 40%", "40-60%", "60-80%", "80-90%", "Over 90%"]
RETENTION_FEATURES = ["new_voters", "new_voters_within"]
STATES = dict(zip("01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split(),
                  "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()))


def state_blocks(zipped):
    """One state's populated blocks from a 2020 PL 94-171 file: block, people, base (its district_id on the 116th Congress's lines)."""
    code = zipped.name[:2]
    with zipfile.ZipFile(zipped) as z:
        geo = pd.read_csv(z.open(f"{code}geo2020.pl"), sep="|", header=None, usecols=list(GEO_2020), dtype=str, encoding="latin-1", quoting=3).rename(columns=GEO_2020)
        counts = pd.read_csv(z.open(f"{code}000012020.pl"), sep="|", header=None, usecols=[4, 5], dtype={4: str, 5: float}).rename(columns={4: "record", 5: "people"})
    blocks = geo[geo.level == BLOCK_LEVEL].merge(counts, on="record")
    return pd.DataFrame({"block": blocks.block, "people": blocks.people, "base": code.upper() + "-" + blocks.base})[lambda d: d.people > 0]


def state_blocks_2010(zipped):
    """The same from a 2010 file, whose geography is fixed-width and whose districts are the 111th Congress's."""
    code = zipped.name[:2]
    with zipfile.ZipFile(zipped) as z:
        geo = pd.read_fwf(z.open(f"{code}geo2010.pl"), colspecs=GEO_2010, header=None, dtype=str, encoding="latin-1",
                          names=["level", "record", "state", "county", "tract", "tabblock", "base"])
        counts = pd.read_csv(z.open(f"{code}000012010.pl"), header=None, usecols=[4, 5], dtype={4: str, 5: float}).rename(columns={4: "record", 5: "people"})
    blocks = geo[geo.level == BLOCK_LEVEL].merge(counts, on="record")
    return pd.DataFrame({"block": blocks.state + blocks.county + blocks.tract + blocks.tabblock, "people": blocks.people,
                         "base": code.upper() + "-" + blocks.base})[lambda d: d.people > 0]


def block_people(vintage):
    """Every state's populated blocks in the 2010 or 2020 census, cached since the files take a few minutes to read."""
    cached = RAW / f"block_people_{vintage}.csv.gz"
    if cached.exists():
        return pd.read_csv(cached, dtype={"block": str, "base": str})
    read = state_blocks if vintage == 2020 else state_blocks_2010
    blocks = pd.concat(read(f) for f in sorted((RAW / f"pl{vintage}").glob(f"*{vintage}.pl.zip")))
    blocks.to_csv(cached, index=False)
    return blocks


def block_lines(congress):
    """block, district from a block equivalency file. Each zip's national file covers every state."""
    with zipfile.ZipFile(RAW / "bef" / f"cd{congress}.zip") as z:
        name = next(n for n in z.namelist() if "national" in n.lower())
        lines = pd.read_csv(z.open(name), dtype=str, skipinitialspace=True)
    block, code = lines.iloc[:, 0], lines.iloc[:, -1]
    return pd.DataFrame({"block": block, "district": block.str[:2].map(STATES) + "-" + code})


def overlap(blocks):
    """Share of each new district's people who were in each old one: old, new, retained. `blocks`: old, new, people."""
    people = blocks.groupby(["old", "new"]).people.sum()
    return (people / people.groupby(level="new").transform("sum")).rename("retained").reset_index()


def changed_states(blocks):
    """The states where over MIN_MOVED of the people changed districts. `blocks`: old, new, people."""
    states = blocks.new.str[:2]
    moved = blocks.people.where(blocks.old != blocks.new, 0.0).groupby(states).sum() / blocks.people.groupby(states).sum()
    return set(moved[moved > MIN_MOVED].index)


def blocks_for(cycle):
    """block, people, old (the lines the cycle's incumbents were elected on) and new, in the states redrawn for `cycle`."""
    vintage, before, after = LINES[cycle]
    people = block_people(vintage)
    old = people[["block", "base"]].rename(columns={"base": "old"}) if before == BASE[vintage] else block_lines(before).rename(columns={"district": "old"})
    both = people[["block", "people"]].merge(old, on="block").merge(block_lines(after).rename(columns={"district": "new"}), on="block")
    both = both[~both.new.str[:2].isin(WRONG_LINES.get(cycle, set()))]
    return both[both.new.str[:2].isin(changed_states(both))]


def district_overlap(cycle):
    """overlap(blocks_for(cycle)), cached, since the block files take about a minute to read."""
    cached = RAW / f"overlap_{cycle}.csv"
    if cached.exists():
        return pd.read_csv(cached, dtype={"old": str, "new": str})
    table = overlap(blocks_for(cycle))
    table.to_csv(cached, index=False)
    return table


def earlier_districts(cycle):
    """Series, cand_id -> the district that cycle. 2010 comes before the forecast's data, so it's FEC's all-candidate file as filed."""
    if cycle >= FIRST_CYCLE:
        return load_candidates(cycle, load_results(cycle)).drop_duplicates("cand_id").set_index("cand_id").district_id
    rows = pd.read_csv(WEBALL.format(yy=str(cycle)[2:]), sep="|", header=None, dtype=str, usecols=[0, 18, 19], quoting=3, encoding="latin-1")
    rows = rows[rows[0].str.startswith("H")].drop_duplicates(0).set_index(0)
    return (rows[18] + "-" + rows[19].str.zfill(2)).rename("district_id")


def incumbent_retention(cycle):
    """One row per incumbent running on the cycle's redrawn lines with an earlier district here: district_id, cand_id, party, old, retained."""
    if cycle not in LINES:
        return pd.DataFrame(columns=["cycle", "district_id", "cand_id", "party", "old", "retained"])
    districts = district_overlap(cycle)
    share = districts.set_index(["old", "new"]).retained
    now = load_candidates(cycle, load_results(cycle))
    then = earlier_districts(cycle - 2)
    incumbents = now[(now.ici == "I") & now.district_id.str[:2].isin(set(districts.new.str[:2]))]
    old = incumbents.cand_id.map(then)
    known = old.notna()
    return pd.DataFrame({"cycle": cycle, "district_id": incumbents.district_id[known], "cand_id": incumbents.cand_id[known], "party": incumbents.party[known], "old": old[known],
                         "retained": [float(share.get(pair, 0.0)) for pair in zip(old[known], incumbents.district_id[known])]}).reset_index(drop=True)


def forecast_margin(built, cycle):
    """`built[cycle]` with mu, the baseline forecast: fit on earlier cycles, or on later ones for the first."""
    done = {c: r for c, r in built.items() if r.margin.notna().any()}
    train = [r for c, r in done.items() if c < cycle] or [r for c, r in done.items() if c > cycle]
    return built[cycle].assign(mu=predict_margin(built[cycle], fit_baseline(pd.concat(train))))


def retention_test(races, cycle):
    """incumbent_retention with resid: actual minus forecast margin, the year's average miss taken out, signed so positive
    means the incumbent beat it. `races`: races with mu and margin. Only contested races, and only where the race's status agrees."""
    contested = races[races.margin.abs() < 80]
    miss = (contested.margin - contested.mu).mean()
    known = contested.set_index("district_id")
    rows = incumbent_retention(cycle)
    rows = rows[rows.district_id.isin(known.index)]
    race = known.loc[rows.district_id]
    agrees = (race.status.to_numpy() == "D incumbent") == (rows.party.to_numpy() == "D")
    sign = np.where(rows.party == "D", 1.0, -1.0)
    return rows.assign(resid=sign * (race.margin.to_numpy() - race.mu.to_numpy() - miss), margin=race.margin.to_numpy(), mu=race.mu.to_numpy())[agrees].reset_index(drop=True)


def retention_slope(table, within_state=False):
    """Points of margin per unit of retained share, comparing incumbents only within a redraw, with a standard
    error clustered by state. `within_state` compares each only with others in their state that year."""
    states = table.district_id.str[:2]
    x, y = table.retained.astype(float), table.resid.astype(float)
    cycles = table.cycle.astype(str) if "cycle" in table else pd.Series("", index=table.index)
    group = states + cycles if within_state else cycles
    x, y = x - x.groupby(group).transform("mean"), y - y.groupby(group).transform("mean")
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    e = y.to_numpy() - X @ beta
    bread = np.linalg.inv(X.T @ X)
    scores = [X[states == s].T @ e[states == s] for s in states.unique()]
    return {"slope": float(beta[1]), "se": float(np.sqrt((bread @ sum(np.outer(g, g) for g in scores) @ bread)[1, 1])), "n": len(table), "states": len(scores)}


def retention_bands(table):
    """Incumbents and their average resid against their redraw's average, by how much of the new district they kept."""
    bands = pd.cut(table.retained, RETAINED_BANDS, labels=RETAINED_LABELS)
    centered = table.resid - table.resid.groupby(table.cycle).transform("mean")
    return centered.groupby(bands, observed=True).agg(["size", "mean"]).rename_axis("band").reset_index()


def carried_over(fitted, later):
    """Root mean squared resid in `later`, as it stands and after applying `fitted`'s slope: did the earlier slope help?"""
    shift = retention_slope(fitted)["slope"] * (later.retained - later.retained.mean())
    return float(np.sqrt((later.resid ** 2).mean())), float(np.sqrt(((later.resid - shift) ** 2).mean()))


def add_new_voters(races):
    """`races` (cycle -> frames with district_id and status) with new_voters."""
    out = {}
    for cycle, x in races.items():
        rows = incumbent_retention(cycle)
        measured = 1 - rows.drop_duplicates("district_id").set_index("district_id").retained.astype(float)
        new, centered = x.district_id.map(measured).fillna(0.0), x.district_id.map(measured - measured.mean()).fillna(0.0)
        sign = np.select([x.status == "D incumbent", x.status == "R incumbent"], [-1.0, 1.0], 0.0)
        out[cycle] = x.assign(new_voters=sign * new, new_voters_within=sign * centered)
    return out
