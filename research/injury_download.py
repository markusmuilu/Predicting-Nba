"""
Downloads the official injury report PDFs for the regular seasons 2018-19 to 2025-26.

History of this file: the first version probed every hour of every game day at about 12 requests a
second. After roughly 5,000 requests the archive's CDN started answering 403 to *everything*, including
files that exist, and that version recorded those refusals as "missing" and poisoned its own cache. The
block lifted after about a quarter of an hour. This version is built so that cannot recur:

- 2 requests a second in total.
- A canary: before any batch of results is recorded, a file known to exist is fetched. If the canary fails
  the batch is thrown away, the run waits (5 minutes, repeating) until the canary succeeds, and redoes it.
  "missing" is only ever written when the canary passed straight after the batch.
- Cadence discovery: reports are not published at the same times in every era (3 a day in 2018-19,
  hourly later, every 15 minutes from late December 2025). For two sample days a month, every possible
  hour is probed; the hours that exist become the probe list for the rest of that month.
- Newest seasons first, so a partial run is already useful.

Everything is restartable: found files stay on disk, probes are recorded in data/injury_pdf/_index.csv,
and the discovered cadence in data/injury_pdf/_cadence.json.

Run: python -m research.injury_download
"""

import csv
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import requests

from research.config import ROOT
from research.player_features import load_games

PDF_DIR = ROOT / "data" / "injury_pdf"
INDEX = PDF_DIR / "_index.csv"
CADENCE = PDF_DIR / "_cadence.json"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{name}.pdf"
SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20", "2018-19"]   # newest first
FIRST_REPORT_DAY = "2018-12-17"
QUARTER_HOUR_FROM = "2025-12-19"
CANARY = "2025-01-15_05PM"       # a report known to exist; if it stops answering we are blocked
GAP = 0.5                        # seconds between request starts, all threads together (2 requests a second)
BATCH = 120
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
_lock, _last = threading.Lock(), [0.0]


def log(msg):
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def all_names(day):
    """Every candidate file name for one date, 07:00 to 23:45, in both naming styles."""
    out = []
    for hour in range(7, 24):
        h12, ap = hour % 12 or 12, "AM" if hour < 12 else "PM"
        out.append(f"{day}_{h12:02d}{ap}")
        if day >= QUARTER_HOUR_FROM:
            out += [f"{day}_{h12:02d}_{m:02d}{ap}" for m in (0, 15, 30, 45)]
    return out


def suffix(name):
    """The time part of a file name, e.g. '2025-01-15_05PM' -> '05PM'."""
    return name.split("_", 1)[1]


def game_days():
    games = load_games(sorted(SEASONS))
    by_season = defaultdict(set)
    for d, s in zip(games["date"], games["season"]):
        d = datetime.strptime(d, "%Y-%m-%d") if isinstance(d, str) else d.to_pydatetime()
        for back in (0, 1):
            by_season[s].add((d - timedelta(days=back)).strftime("%Y-%m-%d"))
    return {s: sorted(x for x in days if x >= FIRST_REPORT_DAY) for s, days in by_season.items()}


# ── Requests, rate limit, canary ──────────────────────────────────────────────

def request(name, session):
    with _lock:
        wait = _last[0] + GAP - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
    for attempt in range(3):
        try:
            r = session.get(URL.format(name=name), timeout=25, headers=HEADERS)
            if r.status_code == 200 and len(r.content) > 1000:
                return "found", r.content
            if r.status_code in (403, 404):
                return "missing", None
            return f"http{r.status_code}", None
        except Exception:
            time.sleep(2 + attempt)
    return "error", None


def canary_ok(session):
    status, _ = request(CANARY, session)
    return status == "found"


def wait_for_canary(session):
    waited = 0
    while not canary_ok(session):
        log(f"canary failed: the archive is refusing known-good files; waiting 5 minutes (waited {waited} so far)")
        time.sleep(300)
        waited += 5


def load_index():
    known = {}
    if INDEX.exists():
        with open(INDEX, newline="") as f:
            for row in csv.reader(f):
                known[row[0]] = row[1]
    return known


def probe_batch(names, session, known, index_file):
    """Fetches names; records them only if a canary passes right afterwards; otherwise waits and redoes the batch."""
    while True:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda n: (n, *request(n, session)), names))
        if canary_ok(session):
            break
        log("batch discarded: the canary failed straight after it")
        wait_for_canary(session)
    for name, status, content in results:
        if status == "found":
            (PDF_DIR / f"{name}.pdf").write_bytes(content)
        if status in ("found", "missing"):
            index_file.write(f"{name},{status},{len(content) if content else 0}\n")
            known[name] = status
    index_file.flush()
    return sum(1 for _, s, _ in results if s == "found")


# ── Cadence discovery and the full plan ───────────────────────────────────────

def discover_cadence(days_by_season, session, known, index_file):
    """For two sample game days a month, probe every candidate time; the hours found are that month's probe list."""
    cadence = json.loads(CADENCE.read_text()) if CADENCE.exists() else {}
    months = defaultdict(list)
    for days in days_by_season.values():
        for d in days:
            months[d[:7]].append(d)
    for month in sorted(months, reverse=True):
        if month in cadence:
            continue
        days = months[month]
        samples = [days[len(days) // 3], days[2 * len(days) // 3]]
        found = set()
        for d in samples:
            names = [n for n in all_names(d) if n not in known]
            if names:
                probe_batch(names, session, known, index_file)
            found |= {suffix(n) for n in all_names(d) if known.get(n) == "found"}
        cadence[month] = sorted(found)
        CADENCE.write_text(json.dumps(cadence, indent=0))
        log(f"cadence {month}: {len(found)} report times a day")
    return cadence


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    session, known = requests.Session(), load_index()
    wait_for_canary(session)
    days_by_season = game_days()
    with open(INDEX, "a", newline="") as index_file:
        cadence = discover_cadence(days_by_season, session, known, index_file)
        plan = [f"{d}_{t}" for s in SEASONS for d in sorted(days_by_season.get(s, []), reverse=True)
                for t in cadence.get(d[:7], []) if f"{d}_{t}" not in known]
        log(f"{len(known)} probes known; {len(plan)} to do (about {len(plan) * GAP / 60:.0f} minutes)")
        found = 0
        for i in range(0, len(plan), BATCH):
            found += probe_batch(plan[i:i + BATCH], session, known, index_file)
            log(f"{min(i + BATCH, len(plan))}/{len(plan)} probed, {found} found in this run")
    log("done")


if __name__ == "__main__":
    main()
