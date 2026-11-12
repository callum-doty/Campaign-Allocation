"""Download one full day of GDELT GKG files per week (Tuesdays, 2016-01-05 on) and keep
only articles from data/partisan_outlets.csv.

Each day's 96 zipped 15-minute files go into their own empty temp directory, are read by
talking_points_extract.rows_from_zip, then deleted. Output: data/talking_points_articles.csv.gz
(date, domain, side, themes, url). Days already in the output's done-log are skipped, and days
run in an interleaved order (every 4th week first) so a partial run still spans the decade.

usage: python3 -I campaign/talking_points_fetch.py WORKDIR [LAST_DATE] [PASSES]
(PASSES=1 is every 4th week, 2 adds the weeks halfway between, 4 is every week)
"""
import csv
import datetime as dt
import gzip
import shutil
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import certifi
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign"))
from talking_points_extract import load_outlets, rows_from_zip  # noqa: E402

OUT = ROOT / "data" / "talking_points_articles.csv.gz"
DONE = ROOT / "data" / "talking_points_days_done.txt"
CTX = ssl.create_default_context(cafile=certifi.where())
BASE = "https://data.gdeltproject.org/gdeltv2/{}.gkg.csv.zip"


def tuesdays(last, passes=4):
    d = dt.date(2016, 1, 5)
    out = []
    while d <= last:
        out.append(d)
        d += dt.timedelta(days=7)
    order = []
    for off in (0, 2, 1, 3)[:passes]:
        order += out[off::4]
    return order


def fetch(url, dest):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60, context=CTX) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return False


def one_file(args):
    """Download one 15-minute file, keep partisan-outlet rows, delete it. Runs in a worker."""
    stamp, tmp = args
    dest = Path(tmp) / f"{stamp}.gkg.csv.zip"
    if not fetch(BASE.format(stamp), dest):
        return stamp, None
    try:
        rows = list(rows_from_zip(dest, OUTLETS))
    except Exception as e:  # bad or truncated zip
        print(stamp, "skip", e, file=sys.stderr)
        rows = None
    dest.unlink(missing_ok=True)
    return stamp, rows


OUTLETS = load_outlets()


def main(workdir, last, passes=4):
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    work = Path(workdir)
    for day in tuesdays(last, passes):
        tag = day.strftime("%Y%m%d")
        if tag in done:
            continue
        tmp = work / tag
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        stamps = [f"{tag}{h:02d}{m:02d}00" for h in range(24) for m in (0, 15, 30, 45)]
        results = list(POOL.map(one_file, [(st, str(tmp)) for st in stamps]))
        got = [r is not None for _, r in results]
        n = 0
        new = not OUT.exists()
        with gzip.open(OUT, "at", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "domain", "side", "themes", "url"])
            for _, rows in results:
                for row in rows or []:
                    w.writerow(row)
                    n += 1
        shutil.rmtree(tmp, ignore_errors=True)
        if sum(got) < 80:  # mostly missing: leave it undone so a rerun retries it
            print(tag, "only", sum(got), "files; not marked done", flush=True)
            continue
        with open(DONE, "a") as f:
            f.write(f"{tag}\n")
        print(tag, "files", sum(got), "rows", n, flush=True)


if __name__ == "__main__":
    POOL = ProcessPoolExecutor(14)
    last = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date(2026, 11, 3)
    main(sys.argv[1], last, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
