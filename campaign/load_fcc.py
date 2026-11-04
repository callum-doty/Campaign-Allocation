"""
Spot lines from the political files fetch_fcc.py downloads: for each document, which time
slot and week was bought, how many spots, and at what rate.

    python -m campaign.load_fcc 2024 2026

Stations' traffic systems print a line header with the program and time, followed by either a row per
week (start, end, days, spots a week, rate) or a row per aired spot (day, date, length, air time, ad, rate).
`parse_lines` reads both. Other layouts, mostly agencies' own forms, aren't read. Documents are orders, revisions and invoices, and
overlap. A rate is a rate wherever it appears, so comparing prices only needs each distinct line once.
Adding up dollars would need more care, so nothing here does.

- Readable: pdftotext's output, with under MIN_CHARS a page counted as a scan. Scans aren't read.
- Slot: the "7:00 AM-9:00 AM" on the line header where there is one, or else the program
  text, with whether it runs on weekdays or at the weekend.
- Rates are gross, as charged, before any agency commission, for every sponsor alike.
- Sponsor: from the folder a station filed it in (`sponsor_kind`). Stations name folders by hand,
  so this is only as good as those names, and unmatched names count as "unknown".
"""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .fetch_fcc import RAW, file_path

MIN_CHARS = 200
READ_WORKERS = 8
ROW = re.compile(r"(\d\d/\d\d/\d\d)\s+(\d\d/\d\d/\d\d)\s+([MTWFSU\-\d]{7})\s+(\d+)\s+\$([\d,]+\.\d\d)")
AIRED = re.compile(r"^\s*[A-Z0-9\-]{3,8}\s+(?:M|Tu|W|Th|F|Sa|Su)\s+(\d\d/\d\d/\d\d)\s+:(\d\d)\s+\d{1,2}:\d\d\s?[AP]M\b.*?\$([\d,]+\.\d\d)")
SPAN = re.compile(r"(\d\d/\d\d/\d\d)\s+to\s+\d\d/\d\d/\d\d\s+\d+x\s+([MTWFSU\-\d]{7})")
PREEMPTIBLE = re.compile(r"preemptible", re.I)
WINDOW = re.compile(r"\d{1,2}:\d\d\s?[AP]M\s?-\s?\d{1,2}:\d\d\s?[AP]M")
LOOSE_TIME = re.compile(r"\d{1,2}(?::?\d\d)?\s?[AaPp]?[Mm]?\s?-\s?\d{1,2}(?::?\d\d)?\s?[AaPp][Mm]?")
LENGTH = re.compile(r"(?<![\d:]):(\d\d)\b(?!:)")
NOT_A_HEADER = re.compile(r"Start Date|^\s*CS\s|ACTUAL|Spots: #|Weeks:")
PARTY = re.compile(r"\bnrcc\b|\bdccc\b|national republican congressional|democratic (congressional|cogressional|cong\b)|dem cong", re.I)
PARTY_WORDS = re.compile(r"\bnrcc\b|\bdccc\b|national republican congressional( committee| cmte)?|democratic (congressional|cogressional)( campaign)?( committee| cmte| cmpgn)?|"
                         r"\bie\b|\bind(ependent)?( expenditures?)?\b|\bcommittee\b|\bcampaign\b|\bcmpgn\b|[^a-z]", re.I)
INDEPENDENT = re.compile(r"\bie\b|\bind(ependent)?\b|expenditure", re.I)
CANDIDATE = re.compile(r"-[DR]-US House|\bfor (us )?(congress|house)\b|\bcongress\b|committee to elect|citizens for|friends of|\bfor [A-Za-z]+( \d{4})?$|\([DR]\)|\bCD ?\d", re.I)
OUTSIDE = re.compile(r"\bpac\b|super ?pac|house majority|leadership fund|\baction\b|\bfund\b|forward|\bproject\b|alliance|america|united democracy|\bfuture\b|\bleague\b|\bvotes?\b|vets\b|\bunion\b|"
                     r"association|network|growth|prosperity|\bclub\b|coalition|foundation|institute|chamber|council|freedom|defend|protect|progress|women|veterans|\binc\b|\bllc\b", re.I)
UNCLEAR = re.compile(r"victory (fund|committee)|\d|district|^[A-Z]{2}$|house of representatives|republican|democrat\b|campaign committee|network|sports", re.I)
PERSON = re.compile(r"^[A-Za-z.'\-]+( [A-Za-z.'\-]+){1,3}$")
STATES = {"ohio", "kansas", "iowa", "texas", "maine", "nevada", "arizona", "michigan", "florida", "colorado", "wisconsin", "pennsylvania", "virginia", "california", "new york", "nebraska", "montana"}


def sponsor_kind(section, sponsor):
    """candidate, coordinated (a party with a candidate), party_independent, outside, or unknown when a folder's name doesn't settle it:
    a party's name alone in a candidate folder, a party with a candidate's name in a non-candidate one, joint victory funds, and places."""
    name = re.sub(r"\s+", " ", str(sponsor)).strip()
    if PARTY.search(name):
        bare = not PARTY_WORDS.sub("", name)
        if section == "non_candidate":
            return "party_independent" if bare else "unknown"
        if bare:
            return "party_independent" if INDEPENDENT.search(name) else "unknown"
        return "coordinated"
    if section == "non_candidate":
        return "outside"
    if name.lower() in STATES:
        return "unknown"
    if CANDIDATE.search(name) and not re.search(r"victory (fund|committee)", name, re.I):
        return "candidate"
    if UNCLEAR.search(name):
        return "unknown"
    if OUTSIDE.search(name):
        return "outside"
    return "candidate" if PERSON.match(name) else "unknown"


def slot_of(line):
    """The line header's time slot and any spot length: (slot, length in seconds or None)."""
    length = LENGTH.search(line)
    window = WINDOW.search(line)
    if window:
        slot = re.sub(r"\s", "", window[0]).upper()
    else:
        text = re.sub(r"\d\d/\d\d/\d\d", " ", line)
        text = re.sub(r"\b[MTWFSU\-\d]{7}\b.*$", " ", text)
        text = re.sub(r"^\s*[A-Z]?\s*\d+\s+[A-Z]{3,5}(-TV)?\s", " ", LENGTH.sub(" ", text))
        text = re.sub(r"\s+(NM\b|P\d\d\b|\$).*$", "", text)
        slot = re.sub(r"\s+", " ", text).strip().upper()[:40]
    return slot, int(length[1]) if length else None


def parse_lines(text):
    """One row per weekly line or aired spot in a document's text: week, days, spots, rate, slot, seconds, preemptible. A line header holds
    until the next one, since a long flight's rows run on past it."""
    out, slot, seconds, preemptible, span = [], "", None, False, None
    for line in text.splitlines():
        row = ROW.search(line)
        aired = AIRED.search(line)
        weeks = SPAN.search(line)
        if row:
            out.append({"week": row[1], "days": row[3], "spots": int(row[4]), "rate": float(row[5].replace(",", "")), "slot": slot, "seconds": seconds, "preemptible": preemptible})
        elif aired:
            week, days = span if span else (aired[1], "")
            out.append({"week": week, "days": days, "spots": 1, "rate": float(aired[3].replace(",", "")), "slot": slot, "seconds": int(aired[2]), "preemptible": preemptible})
        elif weeks:
            span = (weeks[1], weeks[2])
        elif PREEMPTIBLE.search(line):
            preemptible = True
        elif not NOT_A_HEADER.search(line) and LOOSE_TIME.search(line) and re.search(r"[A-Za-z]{2}", line):
            slot, seconds = slot_of(line)
            preemptible = False
    return out


def days_mask(days):
    """A week row's days with counts and letters alike: "MTWTF--" and "22222--" are both "xxxxx--"."""
    return re.sub(r"[^-]", "x", str(days))


def read_text(path):
    """pdftotext's output for `path`, cached beside it."""
    cached = path.with_suffix(path.suffix + ".txt")
    if not cached.exists():
        subprocess.run(["pdftotext", "-layout", str(path), str(cached)], capture_output=True)
    return cached.read_text(errors="ignore") if cached.exists() else ""


def read_document(row):
    """(document summary, spot lines) for a manifest row's file, or (None, []) if it hasn't been downloaded."""
    path = file_path(row)
    if not path.exists() or row.extension != "pdf":
        return None, []
    text = read_text(path)
    pages = text.count("\f") or 1
    chars = len(re.sub(r"\s", "", text))
    found = parse_lines(text) if chars / pages >= MIN_CHARS else []
    base = {"station": row.station, "dma": row.dma, "section": row.section, "sponsor": row.sponsor, "kind": sponsor_kind(row.section, row.sponsor), "file_id": row.file_id}
    return {**base, "uploaded": row.uploaded, "pages": pages, "readable": chars / pages >= MIN_CHARS, "lines": len(found), "invoice": "invoice" in text[:3000].lower()}, [{**base, **f} for f in found]


def load_lines(year, manifest=None):
    """Write and return documents and distinct priced lines for a year: fcc_docs_{year}.csv.gz and fcc_lines_{year}.csv.gz."""
    manifest = pd.read_csv(RAW / f"manifest_{year}.csv", dtype=str) if manifest is None else manifest
    with ThreadPoolExecutor(READ_WORKERS) as pool:
        results = list(pool.map(read_document, manifest.itertuples()))
    docs = pd.DataFrame([d for d, _ in results if d is not None])
    lines = pd.DataFrame([f for _, found in results for f in found])
    lines["week"] = pd.to_datetime(lines.week, format="%m/%d/%y", errors="coerce")
    lines = lines[(lines.rate > 0) & lines.week.notna() & (lines.slot != "")]
    lines = lines.drop_duplicates(["station", "sponsor", "slot", "week", "days", "rate", "spots"])
    docs.to_csv(RAW / f"fcc_docs_{year}.csv.gz", index=False)
    lines.to_csv(RAW / f"fcc_lines_{year}.csv.gz", index=False)
    return docs, lines


if __name__ == "__main__":
    for y in sys.argv[1:]:
        d, x = load_lines(int(y))
        print(y, "documents", len(d), "readable", round(d.readable.mean(), 3), "with lines", round((d.lines > 0).mean(), 3), "distinct lines", len(x), flush=True)
