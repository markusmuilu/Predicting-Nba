"""
Live snapshot logger: records what was knowable before each game, as it happens.

Why this exists: the injury and odds situation just before tip-off cannot be recovered later
from ESPN (its injury list is a current snapshot with no history) or from a cheap odds
history. The official NBA injury report PDFs are archived, but ESPN's view, and the prices on
the odds API, are not. Every day this does not run is a day of data lost.

What it records (all raw, all timestamped, under data/live/, git-ignored):
- every ~hour: ESPN's injury list and the newest official injury report PDF;
- for every game on the ESPN scoreboard, at T-60, T-30 and T-15 minutes before tip-off:
  the ESPN injury list, the newest official PDF, and, at the cutoffs in ODDS_CUTOFFS, one
  odds snapshot (games that share a due moment share one call, because one call returns all
  games);
- data/live/manifest.jsonl, one line per snapshot, so later code can find the right files.

Odds budget: the odds API allows 500 requests a month. January 2026 had 165 distinct tip-off
times, so three odds cutoffs would need about 495 calls, with no room for anything else.
Odds are therefore taken at T-60 and T-15 only (about 330 a month); the injury snapshots are
free and taken at all three cutoffs. `--estimate` recomputes the numbers.

Designed to be called every few minutes by something outside this file (a loop, cron, Task
Scheduler). It keeps its own state in data/live/state.json, so calling it twice, or after a
restart, never duplicates a snapshot. A snapshot that is due may be taken up to WINDOW_MIN
minutes late (a missed tick should not lose a game), never early.

The odds API key is read from .env with dotenv_values; it is never printed or written out.

Run:
    python -m research.live_logger --once        # one tick
    python -m research.live_logger --loop        # tick every 5 minutes until stopped
    python -m research.live_logger --status      # what has been recorded so far
    python -m research.live_logger --estimate    # monthly odds-call estimate
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import pytz
import requests
from dotenv import dotenv_values

from research.config import ROOT

LIVE_DIR = ROOT / "data" / "live"
ET = pytz.timezone("America/New_York")

ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
INJURY_PDF = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

CUTOFFS = (60, 30, 15)          # minutes before tip-off at which a snapshot is taken
ODDS_CUTOFFS = (60, 15)         # the subset that also takes an odds snapshot (quota, see above)
WINDOW_MIN = 10                 # a due snapshot may be up to this many minutes late
HOURLY_MIN = 55                 # minimum gap between the routine hourly snapshots
ODDS_RESERVE = 60               # stop taking odds snapshots when fewer than this many calls remain
PDF_LOOKBACK_HOURS = 4          # how far back to search for the newest report
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"


# ── Small helpers ─────────────────────────────────────────────────────────────

def log(msg):
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}", flush=True)


def stamp(now):
    return now.strftime("%Y%m%dT%H%M%SZ")


def get(url, params=None, retries=3, timeout=20, binary=False):
    """GET with a short retry. Returns the parsed JSON (or bytes if binary), or None on failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r.content if binary else r.json()
            if r.status_code in (403, 404):
                return None                       # the injury archive answers 403 for a missing file
        except Exception as e:
            log(f"request failed ({type(e).__name__}) {url.split('?')[0]}")
        time.sleep(1 + attempt)
    return None


def load_state():
    path = LIVE_DIR / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"done": {}, "last_hourly": None, "last_pdf_et": None}


def save_state(state):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_DIR / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(LIVE_DIR / "state.json")          # atomic on the same filesystem


def rel(path):
    """Path as stored in the manifest: relative to the repo root when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def save_raw(subdir, prefix, now, obj=None, content=None, suffix=".json.gz"):
    """JSON is stored gzip-compressed (ESPN's injury list is about 1 MB of mostly repeated text)."""
    d = LIVE_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{prefix}_{stamp(now)}{suffix}"
    if content is not None:
        path.write_bytes(content)
    else:
        path.write_bytes(gzip.compress(json.dumps(obj).encode("utf-8")))
    return rel(path)


def save_injuries(now, injuries, state):
    """
    Saves ESPN's injury list unless it is identical to the last one saved (the list changes a few
    times a day, but is fetched much more often). The manifest then points at the existing file,
    so a snapshot still names exactly the data that was current at that moment.
    """
    if not injuries:
        return None
    digest = hashlib.sha1(json.dumps(injuries.get("injuries"), sort_keys=True).encode()).hexdigest()
    if state.get("last_inj_hash") == digest and state.get("last_inj_path"):
        return state["last_inj_path"]
    path = save_raw("espn_injuries", "espn_injuries", now, obj=injuries)
    state["last_inj_hash"], state["last_inj_path"] = digest, path
    return path


def manifest(entry):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_DIR / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── The three data sources ────────────────────────────────────────────────────

def scoreboard_events(now, fetch=get):
    """Games on the ESPN scoreboard for today and tomorrow (Eastern), with their tip-off times in UTC."""
    events, seen = [], set()
    et_today = now.astimezone(ET).date()
    for offset in (0, 1):
        day = (et_today + timedelta(days=offset)).strftime("%Y%m%d")
        data = fetch(ESPN_SCOREBOARD, {"dates": day})
        for ev in (data or {}).get("events", []):
            if ev.get("id") in seen:                  # a game near midnight can be listed under two dates
                continue
            seen.add(ev.get("id"))
            comp = ev.get("competitions", [{}])[0]
            teams = {c.get("homeAway"): c.get("team", {}).get("abbreviation") for c in comp.get("competitors", [])}
            events.append({
                "event_id": ev.get("id"),
                "tip": datetime.fromisoformat(ev["date"].replace("Z", "+00:00")),
                "state": comp.get("status", {}).get("type", {}).get("state"),
                "home": teams.get("home"), "away": teams.get("away"),
            })
    return events


def pdf_candidates(t_et):
    """
    File-name stamps for a report issued at Eastern time t_et. The archive switched format around
    22 December 2025: '2025-01-15_05PM' (on the hour only) became '2026-01-15_05_00PM' (any quarter hour).
    """
    names = [t_et.strftime("%Y-%m-%d_%I_%M%p")]
    if t_et.minute == 0:
        names.append(t_et.strftime("%Y-%m-%d_%I%p"))
    return names


def newest_injury_pdf(now, state, fetch=get):
    """
    Finds and saves the newest official injury report. Reports are filed on the quarter hour, so this
    steps back from now in 15-minute steps and stops at the first file that exists, or at the last
    report already saved (nothing newer exists then).
    """
    floor = now - timedelta(hours=PDF_LOOKBACK_HOURS)
    last = state.get("last_pdf_et")
    t = now.astimezone(ET).replace(second=0, microsecond=0)
    t -= timedelta(minutes=t.minute % 15)
    limit = floor.astimezone(ET)
    while t >= limit:
        key = t.strftime("%Y-%m-%d %H:%M")
        if last and key <= last:
            return {"path": state.get("last_pdf_path"), "report_et": last, "new": False}
        for name in pdf_candidates(t):
            content = fetch(INJURY_PDF.format(stamp=name), retries=1, binary=True)
            if content and len(content) > 1000:
                d = LIVE_DIR / "injury_pdf"
                d.mkdir(parents=True, exist_ok=True)
                path = d / f"Injury-Report_{name}.pdf"
                path.write_bytes(content)
                state["last_pdf_et"], state["last_pdf_path"] = key, rel(path)
                return {"path": rel(path), "report_et": key, "new": True}
        t -= timedelta(minutes=15)
    return {"path": state.get("last_pdf_path"), "report_et": last, "new": False}


def odds_snapshot(now, fetch=get):
    """One odds call (all games). Skipped when the remaining quota is below ODDS_RESERVE."""
    key = dotenv_values(ROOT / ".env").get("ODDS_API_KEY")
    if not key:
        return {"error": "no ODDS_API_KEY"}
    try:
        r = requests.get(ODDS_URL, params={"apiKey": key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal",
                                            "dateFormat": "iso", "bookmakers": "pinnacle"}, timeout=20)
    except Exception as e:
        return {"error": type(e).__name__}
    remaining = r.headers.get("x-requests-remaining")
    if r.status_code != 200:
        return {"error": f"status {r.status_code}", "remaining": remaining}
    path = save_raw("odds", "odds", now, obj=r.json())
    out = {"path": path, "remaining": remaining, "used": r.headers.get("x-requests-used")}
    if remaining is not None and int(remaining) < ODDS_RESERVE:
        out["warning"] = "quota below reserve, odds snapshots will pause"
    return out


# ── One tick ──────────────────────────────────────────────────────────────────

def due_items(events, state, now):
    """(event, cutoff) pairs that are due now and not yet recorded."""
    due = []
    for ev in events:
        if ev["state"] != "pre":
            continue
        for cut in CUTOFFS:
            target = ev["tip"] - timedelta(minutes=cut)
            key = f"{ev['event_id']}:{cut}"
            if target <= now < target + timedelta(minutes=WINDOW_MIN) and key not in state["done"]:
                due.append((ev, cut, key))
    return due


def tick(now=None, do_odds=True, fetch=get, odds_fn=odds_snapshot):
    now = now or datetime.now(timezone.utc)
    state = load_state()
    events = scoreboard_events(now, fetch)
    due = due_items(events, state, now)
    hourly_due = (state["last_hourly"] is None or
                  now - datetime.fromisoformat(state["last_hourly"]) >= timedelta(minutes=HOURLY_MIN))
    summary = {"time": now.isoformat(), "games_seen": len(events), "due": len(due), "hourly": hourly_due, "files": []}
    if not due and not hourly_due:
        save_state(state)
        return summary

    injuries = fetch(ESPN_INJURIES)
    inj_path = save_injuries(now, injuries, state)
    pdf = newest_injury_pdf(now, state, fetch)
    odds = None
    if do_odds and any(cut in ODDS_CUTOFFS for _, cut, _ in due):
        odds = odds_fn(now)
        if odds.get("warning"):
            log(f"odds: {odds['warning']} (remaining {odds.get('remaining')})")
        if odds.get("error"):
            log(f"odds snapshot failed: {odds['error']}")

    if hourly_due:
        state["last_hourly"] = now.isoformat()
        manifest({"kind": "hourly", "time": now.isoformat(), "espn_injuries": inj_path, "pdf": pdf.get("path"),
                  "pdf_report_et": pdf.get("report_et")})
    for ev, cut, key in due:
        entry = {"kind": "cutoff", "time": now.isoformat(), "event_id": ev["event_id"], "cutoff_min": cut,
                 "home": ev["home"], "away": ev["away"], "tip_utc": ev["tip"].isoformat(),
                 "minutes_to_tip": round((ev["tip"] - now).total_seconds() / 60, 1),
                 "espn_injuries": inj_path, "pdf": pdf.get("path"), "pdf_report_et": pdf.get("report_et"),
                 "odds": odds.get("path") if odds and cut in ODDS_CUTOFFS else None}
        manifest(entry)
        state["done"][key] = now.isoformat()
    summary["files"] = [p for p in (inj_path, pdf.get("path"), (odds or {}).get("path")) if p]
    save_state(state)
    return summary


# ── Reporting and the quota estimate ──────────────────────────────────────────

def status():
    path = LIVE_DIR / "manifest.jsonl"
    if not path.exists():
        print("nothing recorded yet")
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    cut = [r for r in rows if r["kind"] == "cutoff"]
    print(f"{len(rows)} manifest rows: {len(rows) - len(cut)} hourly, {len(cut)} cutoff snapshots")
    for c in CUTOFFS:
        sub = [r for r in cut if r["cutoff_min"] == c]
        print(f"  T-{c}: {len(sub)} snapshots, {sum(1 for r in sub if r['odds'])} with odds")
    if cut:
        late = [r["minutes_to_tip"] for r in cut]
        print(f"  minutes to tip actually achieved: min {min(late)}, max {max(late)}")
    print("files:", {d.name: sum(1 for _ in d.iterdir()) for d in LIVE_DIR.iterdir() if d.is_dir()})


def estimate(month="2026-01"):
    """Distinct tip-off times in a month, times the number of odds cutoffs, against the 500-call quota."""
    year, mon = map(int, month.split("-"))
    day, slots, games = datetime(year, mon, 1), 0, 0
    while day.month == mon:
        data = get(ESPN_SCOREBOARD, {"dates": day.strftime("%Y%m%d")}) or {}
        evs = data.get("events", [])
        slots += len({e["date"] for e in evs})
        games += len(evs)
        day += timedelta(days=1)
        time.sleep(0.2)
    print(f"{month}: {games} games, {slots} distinct tip-off times")
    for n in (1, 2, 3):
        print(f"  {n} odds cutoff(s): {slots * n} calls a month (quota 500)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--every", type=int, default=300, help="seconds between ticks in --loop mode")
    ap.add_argument("--no-odds", action="store_true", help="skip the odds API (saves quota while testing)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--estimate", nargs="?", const="2026-01")
    args = ap.parse_args()
    if args.status:
        return status()
    if args.estimate:
        return estimate(args.estimate)
    if not (args.once or args.loop):
        ap.print_help()
        return
    while True:
        try:
            s = tick(do_odds=not args.no_odds)
            if s["due"] or s["hourly"]:
                log(f"tick: {s['games_seen']} games on the board, {s['due']} due, hourly={s['hourly']}, files={len(s['files'])}")
        except Exception as e:                      # a logger must keep going through a bad tick
            log(f"tick failed: {type(e).__name__}: {e}")
        if not args.loop:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
