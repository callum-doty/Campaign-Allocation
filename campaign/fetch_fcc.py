"""
TV stations' political files from the FCC's Online Public Inspection File: what each
sponsor ordered and was billed, by program, with the rate. Free, no account, read
through the API documented at publicfiles.fcc.gov/developer. Not part of run.py.

    python -m campaign.fetch_fcc stations
    python -m campaign.fetch_fcc manifest 2024 2026
    python -m campaign.fetch_fcc download 2024 2026

- Stations: every TV facility with its Nielsen market, network and owner.
- Markets: those holding a race the forecast put within CLOSE_MARKET points, in `markets_for`.
  A buy reaches the whole market, so every station there is listed, and stations
  with no House folder drop out on their own.
- Folders: Political Files/{year}/Federal/US House and Non-Candidate Issue Ads hold one folder per
  sponsor. Outside groups and committees' independent buys mostly sit in the second.
- Listing costs about 0.2 seconds a request. Downloads go two at a time with a pause, skip what's on
  disk, and can be stopped and restarted at any point. On 2026-11-03, after about 2,000 files at
  1.9 a second, FCC's network began answering every download with a 403 and a notice of "unusually
  high demand" while its listing API kept working. So downloads now run at about 0.5 a second, and
  `download` stops at the first 403 or 429 rather than retrying.

Files land in data/raw/fcc/, which isn't committed.
"""

import csv
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fcc"
API = "https://publicfiles.fcc.gov/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; Campaign-Allocation project)"}
CLOSE_MARKET = 15
LIST_WORKERS = 6
DOWNLOAD_WORKERS = 2
DOWNLOAD_PAUSE = 1.0
RETRIES = 4
REFUSED = {403, 429}
SKIP_WORDS = re.compile(r"\b(of|the)\b")
MANIFEST_FIELDS = ["year", "station", "entity_id", "dma", "section", "sponsor", "file_id", "file_manager_id", "folder_id", "file_name", "extension", "bytes", "uploaded"]


def get_json(url, params=None):
    """The response as JSON, or None after RETRIES tries."""
    for attempt in range(RETRIES):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if response.status_code == 200:
                return response.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1 + 2 * attempt)
    return None


def fetch_stations():
    """Write stations.csv: id, station, dma, network, owner, service, status, city, state."""
    RAW.mkdir(parents=True, exist_ok=True)
    listing = get_json(f"{API}/service/tv/facility/getall")["results"]["facilityList"]

    def detail(item):
        found = get_json(f"{API}/service/tv/facility/id/{item['id']}")
        f = (found or {}).get("results", {}).get("facility") or {}
        return {"id": item["id"], "station": item["callSign"], "dma": f.get("nielsenDma"), "network": f.get("networkAfil"), "owner": f.get("partyName"),
                "service": f.get("service"), "status": f.get("status"), "city": f.get("communityCity"), "state": f.get("communityState")}
    with ThreadPoolExecutor(LIST_WORKERS) as pool:
        rows = list(pool.map(detail, listing))
    pd.DataFrame(rows).to_csv(RAW / "stations.csv", index=False)
    return len(rows)


def market_key(name):
    """A market name reduced so FCC's "Wilkes Barre-Scranton-Hztn" and Nielsen's "Wilkes Barre-Scranton, PA" meet: its places, lower-case
    without spaces or punctuation, parentheses and state codes dropped."""
    name = re.sub(r"\(.*?\)|\(.*$|,.*$", "", str(name)).lower().replace("ft.", "fort").replace("st.", "saint")
    name = re.sub(r"\b(dc|d\.c\.)\b", "", name)
    places = [re.sub(r"[^a-z]", "", p) for p in re.split(r"[-&/]", name)]
    return tuple(p for p in places if p)


FCC_SPELLINGS = {"Sacramento-Stockton-Modesto, CA": "SACRAMNTO-STKTON-MODESTO", "Portland, OR": "PORTLAND OR", "Columbus, GA": "COLUMBUS GA (OPELIKA AL)", "Columbus, OH": "COLUMBUS OH",
                 "Columbia, SC": "COLUMBIA SC", "Charleston, SC": "CHARLESTON SC", "Rochester, MN-Mason City, IA-Austin, MN": "ROCHESTR-MASON CITY-AUSTIN"}


def market_states(name):
    """The state codes after a Nielsen market's comma."""
    return set(re.findall(r"\b[A-Z]{2}\b", str(name).split(",")[-1])) if "," in str(name) else set()


def station_markets(stations, wanted):
    """`stations` limited to the markets named in `wanted`. An FCC market matches if its first two places do, or else its first
    place does and one of its stations is in that market's state, since several markets start with the same city. Returns
    (rows with a market column, the wanted names that matched nothing)."""
    keys = stations.dma.map(market_key)
    chosen, missed = [], []
    for name in wanted:
        key = market_key(name)
        here = stations.dma.str.upper().eq(FCC_SPELLINGS[name]) if name in FCC_SPELLINGS else keys.map(lambda k: k[:2] == key[:2])
        if not here.any():
            first = keys.map(lambda k: bool(k) and k[0] == key[0])
            states = stations[first].groupby("dma").state.agg(set)
            fits = {dma for dma, held in states.items() if held & market_states(name)}
            here = stations.dma.isin(fits)
        if here.any():
            chosen.append(stations[here].assign(market=name))
        else:
            missed.append(name)
    return (pd.concat(chosen).drop_duplicates("id") if chosen else stations.iloc[0:0]), missed


def folder_by_path(entity_id, path):
    """The folder at `path`, with its id, or None."""
    found = get_json(f"{API}/manager/folder/path.json", {"folderPath": path, "entityId": entity_id, "sourceService": "tv"})
    folders = (found or {}).get("folder") or []
    return folders[0] if folders else None


def folder_contents(entity_id, folder_id):
    """The folder with its subfolders and files, or {}."""
    found = get_json(f"{API}/manager/folder/id/{folder_id}.json", {"entityId": entity_id})
    return (found or {}).get("folder") or {}


def station_files(station, year):
    """Rows for MANIFEST_FIELDS: every file in the station's US House and non-candidate sponsor folders that year."""
    rows = []
    for section, path in (("house", f"Political Files/{year}/Federal/US House"), ("non_candidate", f"Political Files/{year}/Non-Candidate Issue Ads")):
        top = folder_by_path(station.id, path)
        if not top:
            continue
        queue = [(top["entity_folder_id"], "")]
        while queue:
            folder_id, sponsor = queue.pop()
            here = folder_contents(station.id, folder_id)
            for sub in here.get("subfolders") or []:
                queue.append((sub["entity_folder_id"], sponsor or sub["folder_name"]))
            for f in here.get("files") or []:
                rows.append({"year": year, "station": station.station, "entity_id": station.id, "dma": station.dma, "section": section, "sponsor": sponsor, "file_id": f["file_id"],
                             "file_manager_id": f["file_manager_id"], "folder_id": f["folder_id"], "file_name": f["file_name"], "extension": f["file_extension"], "bytes": f["file_size"], "uploaded": f["create_ts"]})
    return rows


def write_manifest(year, stations):
    """Write manifest_{year}.csv, one row per file. Returns its row count."""
    with ThreadPoolExecutor(LIST_WORKERS) as pool, open(RAW / f"manifest_{year}.csv", "w", newline="") as out:
        writer = csv.DictWriter(out, MANIFEST_FIELDS)
        writer.writeheader()
        count = 0
        futures = [pool.submit(station_files, s, year) for s in stations.itertuples()]
        for done, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            writer.writerows(rows)
            count += len(rows)
            if done % 50 == 0:
                print(f"{year}: {done} of {len(futures)} stations, {count} files", flush=True)
                out.flush()
    return count


def file_path(row):
    return RAW / "files" / str(row.year) / row.station / f"{row.file_manager_id}.{row.extension}"


def download_url(row):
    return f"{API}/manager/download/{row.folder_id}/{row.file_manager_id}.{row.extension}"


class Refused(Exception):
    """FCC answered a download with a status in REFUSED."""


def download_one(row, pause=DOWNLOAD_PAUSE):
    """Save one file if it isn't on disk. True if it's there afterward. Raises Refused rather than retrying a refusal."""
    path = file_path(row)
    if path.exists():
        return True
    for attempt in range(RETRIES):
        try:
            response = requests.get(download_url(row), headers=HEADERS, timeout=120)
            if response.status_code in REFUSED:
                raise Refused(f"{response.status_code} on {row.file_manager_id}")
            if response.status_code == 200 and response.content:
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".part")
                partial.write_bytes(response.content)
                partial.rename(path)
                time.sleep(pause)
                return True
        except requests.RequestException:
            pass
        time.sleep(1 + 2 * attempt)
    return False


def download(manifest, workers=DOWNLOAD_WORKERS):
    """Download every file in `manifest`, printing progress, until done or refused. Returns (saved, failed, refused)."""
    rows = list(manifest.itertuples())
    refused = threading.Event()
    counts = {"saved": 0, "failed": 0}
    started = time.time()

    def work(row):
        if refused.is_set():
            return
        try:
            ok = download_one(row)
        except Refused as why:
            refused.set()
            print(f"stopped: FCC refused a download ({why})", flush=True)
            return
        counts["saved" if ok else "failed"] += 1
        done = counts["saved"] + counts["failed"]
        if done % 250 == 0:
            print(f"{done} of {len(rows)}, {counts['failed']} failed, {done / (time.time() - started):.1f} a second", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, rows))
    return counts["saved"], counts["failed"], refused.is_set()


if __name__ == "__main__":
    command, years = sys.argv[1], [int(y) for y in sys.argv[2:]]
    if command == "stations":
        print(fetch_stations(), "stations")
    elif command == "manifest":
        from .fcc_markets import markets_for
        all_stations = pd.read_csv(RAW / "stations.csv", dtype=str)
        all_stations = all_stations[all_stations.status.eq("LICENSED") & all_stations.dma.notna()]
        for y in years:
            chosen, missing = station_markets(all_stations, markets_for(y))
            print(f"{y}: {len(chosen)} stations; markets with no match: {missing}", flush=True)
            print(y, write_manifest(y, chosen), "files", flush=True)
    elif command == "download":
        for y in years:
            print(y, download(pd.read_csv(RAW / f"manifest_{y}.csv", dtype=str)), flush=True)
