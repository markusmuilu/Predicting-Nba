"""
The NBA's official injury reports: parse them, and look up what was known before a game.

The reports are archived as PDFs at ak-static.cms.nba.com/referee/injury/ (first one 17 Dec 2018).
Facts established by probing the archive (see the brief in the vault):
- File names: '2025-01-15_05PM' (one file per hour) until about 21 Dec 2025, then
  '2026-01-15_05_00PM' on a 15-minute grid.
- The name is only a bucket. The report inside carries its own time in the header
  ('Injury Report: 01/15/25 05:30 PM'): the file named 05PM is the 5:30 report (2018 to Nov 2025),
  on 21 Dec 2025 the stamp was HH:45, and from late December 2025 it matches the name. Using the
  file name as the "as of" time would leak up to 30 minutes of information, so everything here
  uses the header time, which is US Eastern wall-clock time.
- No reports exist in the offseason or in the preseason.

Layouts: 2018-19 to Oct 2019 has Category and Previous Status columns; later reports have
Player Name, Current Status, Reason; some years list "Available" players as well. Older files
are stored rotated by 90 degrees, and wrapped cells are centred on their row (the first line of
a two-line reason sits *above* the player's line). So the parser:
  1. maps every word to the displayed orientation;
  2. takes the column positions from each page's own header row;
  3. uses the status word of each row as the row's anchor, and attaches every other word to the
     nearest anchor by vertical position.
A team that has not filed yet appears as one row with status "NOT YET SUBMITTED" and no player.
"""

import re
from datetime import datetime

import pandas as pd
import pymupdf

STATUSES = {"Out", "Questionable", "Doubtful", "Probable", "Available"}
KNOWN_TEAMS = {
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers",
    "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "LA Clippers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks",
    "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic",
    "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs",
    "Toronto Raptors", "Utah Jazz", "Washington Wizards",
}
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
MATCHUP_RE = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")
HEADER_TIME_RE = re.compile(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2})\s*([AP]M)")
LABELS = ["Game Date", "Game Time", "Matchup", "Team", "Player Name", "Category", "Current Status", "Reason", "Previous Status"]


def header_time(text):
    """The report's own time, e.g. 'Injury Report: 01/15/25 05:30 PM' -> naive datetime in US Eastern wall-clock time."""
    m = HEADER_TIME_RE.search(text)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%m/%d/%y %I:%M %p")


def visual_words(page):
    """Words as (x0, y0, text) in the displayed orientation, sorted top to bottom then left to right."""
    out = []
    for w in page.get_text("words"):
        r = pymupdf.Rect(w[:4]) * page.rotation_matrix
        out.append((min(r.x0, r.x1), min(r.y0, r.y1), w[4]))
    return sorted(out, key=lambda t: (round(t[1] / 3), t[0]))


def column_starts(words):
    """x position of each column, from the header row (the line holding both 'Matchup' and 'Player')."""
    lines = {}
    for x, y, t in words:
        lines.setdefault(round(y / 3), []).append((x, y, t))
    for key in sorted(lines):
        line = sorted(lines[key])
        texts = [t for _, _, t in line]
        if "Matchup" in texts and "Player" in texts:
            starts, i = {}, 0
            while i < len(line):
                two = " ".join(texts[i:i + 2])
                if two in LABELS:
                    starts[two] = line[i][0]
                    i += 2
                elif texts[i] in LABELS:
                    starts[texts[i]] = line[i][0]
                    i += 1
                else:
                    i += 1
            return starts, line[0][1]
    return None, None


def column_of(x, starts):
    name = None
    for label, start in sorted(starts.items(), key=lambda kv: kv[1]):
        if x >= start - 4:
            name = label
    return name


def block_events(words, kind, starts):
    """
    (y, value) events for a block-level column. Date, time and matchup are single tokens; a team name can
    wrap over two lines, so team lines are accumulated until they spell a known team (the event is placed
    at the y of the first line, which is level with the first player of that team).
    """
    events = []
    if kind == "team":
        lines = {}
        for x, y, t in words:
            lines.setdefault(round(y / 3), []).append((y, x, t))
        pending, first_y = [], None
        for k in sorted(lines):
            line = sorted(lines[k], key=lambda w: w[1])
            txt = " ".join(t for _, _, t in line)
            if txt == "NOT YET SUBMITTED":
                continue
            txt = txt.replace("NOT YET SUBMITTED", "").strip()
            if not txt:
                continue
            if not pending:
                first_y = line[0][0]
            pending.append(txt)
            joined = " ".join(pending)
            if joined in KNOWN_TEAMS:
                events.append((first_y, joined))
                pending = []
            elif len(pending) > 3:                          # something unrecognisable: keep it visible, then move on
                events.append((first_y, joined))
                pending = []
        if pending:
            events.append((first_y, " ".join(pending)))
        return events
    for x, y, t in words:
        t = t.replace("(ET)", "").strip()
        if (kind == "date" and DATE_RE.match(t)) or (kind == "time" and TIME_RE.match(t)) or (kind == "matchup" and MATCHUP_RE.match(t)):
            events.append((y, t))
    return sorted(events)


def latest(events, y, current):
    """The value of the last event at or above y (within a few points), else the current value."""
    value = current
    for ey, v in events:
        if ey <= y + 4:
            value = v
    return value


def parse_page(page, state, problems, pno, starts_prev):
    """Rows of one page. `state` carries the game and team across pages; so do the column positions."""
    words = visual_words(page)
    starts, header_y = column_starts(words)
    if starts is None:
        if starts_prev is None:
            problems.append(f"page {pno}: no header row and none before it")
            return [], starts_prev
        starts = starts_prev                              # later pages of newer reports have no header row
        title = [i for i, (_, _, t) in enumerate(words) if t == "Report:"]
        title_y = words[title[0]][1] if title else -1
        body = [w for w in words if abs(w[1] - title_y) > 4] if title else list(words)
    else:
        body = [(x, y, t) for x, y, t in words if y > header_y + 6]
    foot = [i for i, (_, _, t) in enumerate(body) if t == "Page"]
    if foot:
        fy = body[foot[-1]][1]
        body = [w for w in body if abs(w[1] - fy) > 4]

    # "NOT YET SUBMITTED" sits in the first data column after the team (Reason, or Category in 2018-19), on
    # the team's own line. Its words are removed so they never leak into a neighbouring row's reason.
    nys_ys = [y for _, y, t in body if t == "SUBMITTED"]
    body = [(x, y, t) for x, y, t in body
            if not (t in ("NOT", "YET", "SUBMITTED") and any(abs(y - ny) <= 2 for ny in nys_ys))]
    cols = {c: [(x, y, t) for x, y, t in body if column_of(x, starts) == c] for c in starts}
    status_col = "Current Status"
    anchors = [(y, "row") for x, y, t in cols.get(status_col, []) if t in STATUSES]
    team_words = cols.get("Team", [])
    anchors += [(y, "nys") for y in nys_ys]
    anchors.sort()
    if not anchors:
        return [], starts
    anchor_y = [a[0] for a in anchors]
    ev_date = block_events(cols.get("Game Date", []), "date", starts)
    ev_time = block_events(cols.get("Game Time", []), "time", starts)
    ev_match = block_events(cols.get("Matchup", []), "matchup", starts)
    ev_team = block_events(team_words, "team", starts)

    def nearest(y):
        return min(range(len(anchor_y)), key=lambda i: abs(anchor_y[i] - y))

    cells = [{c: [] for c in ("Player Name", "Category", "Reason", "Previous Status", status_col) if c in starts} for _ in anchors]
    for c in cells[0]:
        for x, y, t in cols.get(c, []):
            cells[nearest(y)][c].append((y, x, t))

    def text(cell, c):
        return " ".join(t for _, _, t in sorted(cell.get(c, []))).strip()

    rows = []
    for (y, kind), cell in zip(anchors, cells):
        state["date"] = latest(ev_date, y, state["date"])
        state["time"] = latest(ev_time, y, state["time"])
        state["matchup"] = latest(ev_match, y, state["matchup"])
        state["team"] = latest(ev_team, y, state["team"])
        base = {"date": state["date"], "time": state["time"], "matchup": state["matchup"], "team": state["team"]}
        if kind == "nys":
            rows.append({**base, "player": None, "status": "NOT YET SUBMITTED", "reason": None, "category": None,
                         "previous_status": None})
            continue
        player = text(cell, "Player Name")
        if not player:
            problems.append(f"page {pno}: status without a player at y={y:.0f}")
            continue
        rows.append({**base, "player": player, "status": text(cell, status_col), "reason": text(cell, "Reason") or None,
                     "category": text(cell, "Category") if "Category" in starts else None,
                     "previous_status": text(cell, "Previous Status") if "Previous Status" in starts else None})
    return rows, starts


def parse_pdf(path_or_bytes):
    """Returns (rows DataFrame, report_time datetime or None, problems list)."""
    doc = pymupdf.open(stream=path_or_bytes, filetype="pdf") if isinstance(path_or_bytes, bytes) else pymupdf.open(path_or_bytes)
    problems, rows = [], []
    state = {"date": None, "time": None, "matchup": None, "team": None}
    rep_time = header_time(doc[0].get_text())
    if rep_time is None:
        problems.append("no header time")
    starts = None
    for pno, page in enumerate(doc, start=1):
        page_rows, starts = parse_page(page, state, problems, pno, starts)
        rows += page_rows
    df = pd.DataFrame(rows)
    if not df.empty:
        df["report_time"] = rep_time
        bad = df[(df.status != "NOT YET SUBMITTED") & ~df.status.isin(STATUSES)]
        if len(bad):
            problems.append(f"{len(bad)} rows with an unknown status: {sorted(set(bad.status))[:3]}")
        unknown_team = sorted(set(df.team.dropna()) - KNOWN_TEAMS)
        if unknown_team:
            problems.append(f"unrecognised team text: {unknown_team[:3]}")
        if df[["date", "matchup", "team"]].isna().any().any():
            problems.append("rows with a missing date, matchup or team")
    return df, rep_time, problems


# ── Validation against the real schedule ─────────────────────────────────────

def team_abbreviations():
    """Full team name as printed in the reports -> abbreviation used by the schedule."""
    from nba_api.stats.static import teams as nba_teams
    names = {t["full_name"]: t["abbreviation"] for t in nba_teams.get_teams()}
    names["LA Clippers"] = "LAC"
    return names


def validate(df, schedule):
    """
    Checks one parsed report against the schedule (a DataFrame with date, home, away from load_games).
    Returns counts: game blocks found, blocks that match a real game on that date, rows whose team is not
    one of the two teams in their block, and rows with an unrecognised team name.
    """
    abbr = team_abbreviations()
    real = {(pd.Timestamp(r.date).strftime("%m/%d/%Y"), f"{r.away}@{r.home}") for r in schedule.itertuples()}
    blocks = df[["date", "matchup"]].drop_duplicates()
    matched = sum((r.date, r.matchup) in real for r in blocks.itertuples())
    mismatched_blocks = [(r.date, r.matchup) for r in blocks.itertuples() if (r.date, r.matchup) not in real]
    wrong_team = unknown = 0
    for r in df.itertuples():
        a = abbr.get(r.team)
        if a is None:
            unknown += 1
        elif r.matchup is None or a not in r.matchup.split("@"):
            wrong_team += 1
    return {"blocks": len(blocks), "blocks_matched": matched, "blocks_unmatched": mismatched_blocks[:5],
            "rows": len(df), "rows_wrong_team": wrong_team, "rows_unknown_team": unknown}
