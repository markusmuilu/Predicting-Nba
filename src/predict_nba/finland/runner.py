"""
The Finnish leagues on Fly.io, a loop of its own next to the NBA loop (start.sh).

Every 15 minutes it checks the time; once a day after 10:30 Finnish time (the rosters are announced at 10:00) it
rebuilds the ratings from the results service and writes portfolio/finland.json, served at /portfolio/finland.json.

Every upcoming game is logged twice in finland/predictions.jsonl, never rewritten:
  early     the first time the game appears in the schedule (up to three weeks ahead)
  gameday   on the day of the game, from the morning rebuild, if the game has not started
The log is also served, as /portfolio/finland_predictions.jsonl, so the page can score the pre-game record.

The older seasons from basket.fi's statistics service (history.csv) take minutes to fetch and do not change, so they
are kept in R2 and copied into the local cache at start.
"""

import json
import time
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from predict_nba.finland import CACHE_DIR
from predict_nba.utils.logger import logger
from predict_nba.utils.s3_client import S3Client

FI = ZoneInfo("Europe/Helsinki")
LOG = "finland/predictions.jsonl"
HISTORY = "finland/history.csv"
LAST_RUN = "finland/last_run.txt"
START_MIN = 10 * 60 + 30
EVERY_MIN = 15


def text(s3, key):
    b = s3.download(key)
    return b.decode("utf-8") if b else ""


def rebuild(s3, now):
    from predict_nba.finland import pipeline

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hist = CACHE_DIR / "history.csv"
    if not hist.exists():
        body = s3.download(HISTORY)
        if body:
            hist.write_bytes(body)
    data = pipeline.build(refresh=True)
    if hist.exists() and not s3.download(HISTORY):
        s3.upload(HISTORY, hist.read_bytes())

    seen, lines = set(), text(s3, LOG)
    for line in lines.splitlines():
        if line.strip():
            r = json.loads(line)
            seen.add((r["when"], r["home"], r["away"], r.get("kind", "early")))
    today = now.astimezone(FI).date().isoformat()
    stamp = now.astimezone(timezone.utc).isoformat(timespec="seconds")
    new = []
    for pool, p in data["pools"].items():
        for u in p["upcoming"]:
            tip = datetime.strptime(u["when"], "%Y-%m-%d %H:%M").replace(tzinfo=FI)
            if tip <= now:
                continue
            key = (u["when"], u["home"], u["away"])
            if key + ("early",) not in seen:
                new.append({**u, "pool": pool, "kind": "early", "predicted_at": stamp})
            if u["when"][:10] == today and key + ("gameday",) not in seen:
                new.append({**u, "pool": pool, "kind": "gameday", "predicted_at": stamp})
    if lines and not lines.endswith("\n"):
        lines += "\n"
    body = lines + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
    s3.upload(LOG, body.encode("utf-8"))
    s3.upload("portfolio/finland_predictions.jsonl", body.encode("utf-8"))
    s3.upload("portfolio/finland.json", json.dumps(data, separators=(",", ":"), default=str).encode("utf-8"))
    s3.upload(LAST_RUN, today.encode("utf-8"))
    logger.info(f"finland: rebuilt, {sum(r['kind'] == 'early' for r in new)} early and "
                f"{sum(r['kind'] == 'gameday' for r in new)} game-day predictions logged")


def step(s3, now=None):
    now = now or datetime.now(FI)
    local = now.astimezone(FI)
    if local.hour * 60 + local.minute < START_MIN:
        return
    if text(s3, LAST_RUN).strip() == local.date().isoformat():
        return
    rebuild(s3, now)


def run():
    s3 = S3Client()
    while True:
        try:
            step(s3)
        except Exception:
            logger.error(f"finland step failed: {traceback.format_exc()}")
        now = datetime.now(timezone.utc)
        time.sleep(max(EVERY_MIN * 60 - (now.minute % EVERY_MIN) * 60 - now.second, 30))


if __name__ == "__main__":
    run()
