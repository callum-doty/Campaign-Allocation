"""Generic-ballot polls, November 2016 to November 2026, one row per poll.

Sources:
- FiveThirtyEight's generic_ballot_polls_historical.csv and generic_ballot_polls.csv
  (archived by the Wayback Machine on 2025-03-05; 538 closed that month). Covers polls
  ending 2016-11-04 to 2025-03-03.
- Wikipedia's "2026 United States elections polling" page (wikitext saved on 2026-11-09),
  both generic-ballot tables. Used for polls ending after 2025-03-03.

One row per poll: likely voters if asked, else registered voters, else all adults; several
questions at that population are averaged. Output: data/generic_ballot_polls_2017_2026.csv
(end_date, pollster, population, dem, rep, margin = dem - rep, source).

usage: python3 -I campaign/talking_points_ballot.py HIST.csv CURRENT.csv WIKITEXT
"""
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "generic_ballot_polls_2017_2026.csv"
RANK = {"lv": 0, "rv": 1, "v": 1, "a": 2}
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}


def from_538(paths):
    p = pd.concat([pd.read_csv(x) for x in paths], ignore_index=True).drop_duplicates("question_id")
    p["end_date"] = pd.to_datetime(p.end_date, format="%m/%d/%y")
    p["population"] = p.population.str.lower()
    p["rank"] = p.population.map(RANK).fillna(3)
    p = p[p["rank"] == p.groupby("poll_id")["rank"].transform("min")]
    g = p.groupby("poll_id").agg(end_date=("end_date", "first"), pollster=("display_name", "first"),
                                 population=("population", "first"), dem=("dem", "mean"), rep=("rep", "mean"))
    g["source"] = "538"
    return g.reset_index(drop=True)


def strip_templates(s):
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"\{\{[^{}]*\}\}", "", s)
    return s


def end_of(text):
    t = strip_templates(text).replace("&nbsp;", " ").split("|")[-1]
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"\s*[–—-]\s*", t)
    first, last = parts[0], parts[-1]
    years = re.findall(r"(20\d\d)", t)
    year = int(years[-1])
    mlast = re.search(r"([A-Za-z]+)\s+(\d{1,2})", last)
    if mlast:
        month, day = MONTHS[mlast.group(1).lower()], int(mlast.group(2))
    else:
        month = MONTHS[re.search(r"([A-Za-z]+)", first).group(1).lower()]
        day = int(re.search(r"(\d{1,2})", last).group(1))
    return date(year, month, day)


def is_date(cell):
    t = strip_templates(cell).split("|")[-1]
    return bool(re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d", t))


def pct(cell):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", strip_templates(cell).split("|")[-1])
    return float(m.group(1)) if m else None


def from_wiki(path):
    w = open(path, encoding="utf-8").read()
    rows = []
    for table in re.findall(r"\{\|.*?\n\|\}", w, flags=re.S):
        pollster = when = None
        for block in table.split("\n|-")[1:]:
            cells = [c[1:] for c in block.strip("\n").split("\n") if c.startswith("|") and not c.startswith("|}")]
            if not cells:
                continue
            if len(cells) >= 7 and not is_date(cells[0]):
                pollster, when, rest = cells[0], cells[1], cells[2:]
            elif is_date(cells[0]):
                when, rest = cells[0], cells[1:]
            else:
                rest = cells
            if len(rest) < 3 or when is None:
                continue
            pop = re.search(r"\((LV|RV|A|V)\)", rest[0])
            vals = [pct(c) for c in rest[1:] if "±" not in c and pct(c) is not None]
            if len(vals) < 2:
                continue
            dem, rep = vals[0], vals[1]
            name = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", strip_templates(pollster)).split("|")[-1].strip()
            rows.append(dict(end_date=pd.Timestamp(end_of(when)), pollster=name,
                             population=pop.group(1).lower() if pop else "a", dem=dem, rep=rep, key=(name, when)))
    d = pd.DataFrame(rows)
    d["rank"] = d.population.map(RANK).fillna(3)
    d = d[d["rank"] == d.groupby("key")["rank"].transform("min")]
    g = d.groupby("key").agg(end_date=("end_date", "first"), pollster=("pollster", "first"),
                             population=("population", "first"), dem=("dem", "mean"), rep=("rep", "mean"))
    g["source"] = "wikipedia"
    return g.reset_index(drop=True)


def main(hist, cur, wiki):
    a = from_538([hist, cur])
    b = from_wiki(wiki)
    b = b[b.end_date > a.end_date.max()]
    out = pd.concat([a, b], ignore_index=True).sort_values("end_date")
    out["margin"] = out.dem - out.rep
    out.to_csv(OUT, index=False, float_format="%.2f")
    print(out.groupby([out.end_date.dt.year, "source"]).size())
    print(out.tail(5))


if __name__ == "__main__":
    main(*sys.argv[1:4])
