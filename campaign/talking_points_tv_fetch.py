"""Fetch cable-news closed captions from the Internet Archive's TV News Archive for the
talking-points TV test.

For each sampled Tuesday (the website pilot's days: every 4th Tuesday from 2016-01-05) and each
channel (west-coast feeds FOXNEWSW, MSNBCW, which became MSNOW in November 2025, and CNNW), list
the programs whose recording starts that UTC day, drop paid programming, and read the
per-minute caption text shown on each program's public details page. The caption files
themselves are access-restricted; the details page shows the same text a minute at a time.
Reruns are kept here and found by content in talking_points_tv.py, since networks reuse one
title ("MSNBC Live", "CNN Newsroom") for different live hours.

Output: data/raw/tv_captions/YYYYMMDD_CHANNEL.tsv.gz (identifier, show, minute, text), written
once every program for that channel-day has been tried. An existing file is topped up with any
listed program it lacks.

usage: python3 -I campaign/talking_points_tv_fetch.py [LAST_DATE] [PASSES]
"""
import csv
import datetime as dt
import gzip
import html
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "tv_captions"
CTX = ssl.create_default_context(cafile=certifi.where())
UA = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}
CHANNELS = {"FOX": ["TV-FOXNEWSW"], "MSNBC": ["TV-MSNBCW", "TV-MSNOW"], "CNN": ["TV-CNNW"]}
PAID = re.compile(r"paid program|infomercial", re.I)
SNIP = re.compile(r'<div class="snipin[^"]*"[^>]*>(.*?)</div>', re.S)


def tuesdays(last, passes=1):
    d, out = dt.date(2016, 1, 5), []
    while d <= last:
        out.append(d)
        d += dt.timedelta(days=7)
    order = []
    for off in (0, 2, 1, 3)[:passes]:
        order += out[off::4]
    return order


def get(url, tries=5):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(10 * (attempt + 1) if e.code in (429, 503) else 3 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def programs(collection, day):
    q = f"collection:{collection} AND date:{day.isoformat()}"
    url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
        {"q": q, "fl[]": "identifier", "rows": 300, "output": "json"})
    docs = json.loads(get(url))["response"]["docs"]
    ids = sorted(d["identifier"] for d in docs)
    stamp = day.strftime("%Y%m%d")
    return [i for i in ids if i.split("_")[1] == stamp]


def show_of(identifier):
    return " ".join(identifier.split("_")[3:]).strip()


def pick(ids):
    """Drop paid programming; order by start time."""
    return [i for i in sorted(ids, key=lambda s: s.split("_")[2]) if not PAID.search(show_of(i))]


def captions(identifier):
    page = get(f"https://archive.org/details/{identifier}")
    if page is None:
        return []
    out = []
    for k, b in enumerate(SNIP.findall(page)):
        t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", b))).strip()
        out.append((k, t))
    return out


def one_day(day, chan, pool):
    dest = OUT / f"{day:%Y%m%d}_{chan}.tsv.gz"
    have = {}
    if dest.exists():
        with gzip.open(dest, "rt", newline="") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for i, _, k, t in r:
                have.setdefault(i, []).append((int(k), t))
    ids = []
    for c in CHANNELS[chan]:
        ids += programs(c, day)
    keep = pick(ids)
    missing = [i for i in keep if i not in have]
    if have and not missing:
        return "skip"
    have.update(zip(missing, pool.map(captions, missing)))
    tmp = dest.with_suffix(".tmp")
    n = 0
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["identifier", "show", "minute", "text"])
        for i in keep:
            for k, t in have.get(i, []):
                w.writerow([i, show_of(i), k, t])
                n += 1
    tmp.rename(dest)
    return f"{len(ids)} listed, {len(keep)} kept, {len(missing)} fetched, {sum(1 for i in keep if have.get(i))} with text, {n} minutes"


def main(last=dt.date(2026, 11, 3), passes=1):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(6) as pool:
        for day in tuesdays(last, passes):
            for chan in CHANNELS:
                try:
                    msg = one_day(day, chan, pool)
                except Exception as e:
                    msg = f"error {e}"
                print(day, chan, msg, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    last = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date(2026, 11, 3)
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(last, passes)
