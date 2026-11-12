"""Pull article-level theme tags for partisan outlets out of GDELT GKG 15-minute files.

Reads zipped GKG files (paths on the command line, or a directory), keeps rows whose
SourceCommonName is in data/partisan_outlets.csv, and appends
date, domain, side, themes (V1 Themes, ';'-joined) to the output CSV.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_outlets(path=ROOT / "data" / "partisan_outlets.csv"):
    with open(path) as f:
        return {r["domain"]: r["side"] for r in csv.DictReader(f)}


def rows_from_zip(path, outlets):
    """Yield (date, domain, side, themes, url) for partisan-outlet rows. Lines are filtered on
    the source field before decoding, since a day of GKG is several GB of text."""
    keys = {d.encode(): side for d, side in outlets.items()}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as fh:
                for raw in io.BufferedReader(fh, 1 << 20):
                    head = raw.split(b"\t", 5)
                    if len(head) < 6:
                        continue
                    side = keys.get(head[3])
                    if side is None:
                        continue
                    p = raw.decode("utf-8", "replace").split("\t")
                    if len(p) < 8:
                        continue
                    themes = sorted({t for t in p[7].split(";") if t})
                    yield p[1][:8], p[3], side, ";".join(themes), p[4]


def main(out, paths):
    outlets = load_outlets()
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.gkg.csv.zip")) if p.is_dir() else [p]
    new = not Path(out).exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "domain", "side", "themes", "url"])
        for fp in files:
            try:
                for r in rows_from_zip(fp, outlets):
                    w.writerow(r)
            except zipfile.BadZipFile:
                print("bad zip", fp, file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
