"""ESPN: scoreboard, box scores in the training data's column layout, and the injury list."""

import time
import unicodedata

import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_TO_NBA = {"UTAH": "UTA", "GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "WSH": "WAS"}
REGULAR, PLAYOFFS = 2, 3


def abbr(team):
    a = team.get("abbreviation", "")
    return ESPN_TO_NBA.get(a, a)


def norm_name(s):
    """Player key shared by the stats.nba.com history and ESPN: ascii, lower case, suffixes and punctuation dropped."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    for suffix in (" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"):
        s = s.removesuffix(suffix)
    return "".join(ch for ch in s if ch.isalnum())


def get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def scoreboard(day):
    """Games on an ET calendar day: id, tip-off (UTC ISO), state (pre/in/post), season type, teams, scores."""
    data = get(SCOREBOARD, {"dates": day.strftime("%Y%m%d")})
    out = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        side = {c["homeAway"]: c for c in comp["competitors"]}
        out.append({
            "id": ev["id"], "tip": ev["date"], "state": ev["status"]["type"]["state"],
            "season_type": ev.get("season", {}).get("type"),
            "home": abbr(side["home"]["team"]), "away": abbr(side["away"]["team"]),
            "home_name": side["home"]["team"]["displayName"], "away_name": side["away"]["team"]["displayName"],
            "home_pts": int(side["home"].get("score") or 0), "away_pts": int(side["away"].get("score") or 0),
        })
    return out


def _made_att(s):
    m, a = (s or "0-0").split("-")
    return int(m or 0), int(a or 0)


def _int(s):
    try:
        return int(str(s).replace("+", ""))
    except ValueError:
        return 0


def box_rows(event_id, game_date, season):
    """
    One row per player who got on the floor, in the stats.nba.com LeagueGameLog layout the features expect.
    Players marked did-not-play, or with no minutes, are left out, as LeagueGameLog leaves them out.
    """
    data = get(SUMMARY, {"event": event_id})
    rows = []
    for team in data.get("boxscore", {}).get("players", []):
        t = abbr(team["team"])
        for block in team.get("statistics", []):
            labels = block.get("labels", [])
            for a in block.get("athletes", []):
                if a.get("didNotPlay") or not a.get("stats"):
                    continue
                v = dict(zip(labels, a["stats"]))
                minutes = _int(v.get("MIN", 0))
                if minutes <= 0:
                    continue
                fgm, fga = _made_att(v.get("FG"))
                f3m, f3a = _made_att(v.get("3PT"))
                ftm, fta = _made_att(v.get("FT"))
                name = a["athlete"]["displayName"]
                rows.append({
                    "GAME_ID": f"E{event_id}", "GAME_DATE": game_date, "season": season, "TEAM_ABBREVIATION": t,
                    "PLAYER_NAME": name, "key": norm_name(name),
                    "MIN": minutes, "PTS": _int(v.get("PTS")), "REB": _int(v.get("REB")), "OREB": _int(v.get("OREB")),
                    "AST": _int(v.get("AST")), "STL": _int(v.get("STL")), "BLK": _int(v.get("BLK")),
                    "TOV": _int(v.get("TO")), "PF": _int(v.get("PF")), "FGM": fgm, "FGA": fga, "FG3M": f3m,
                    "FG3A": f3a, "FTM": ftm, "FTA": fta, "PLUS_MINUS": _int(v.get("+/-")),
                })
    return rows


def injuries():
    """ESPN's injury list now: [{team, player, key, status}]."""
    data = get(INJURIES)
    out = []
    for team in data.get("injuries", []):
        for item in team.get("injuries", []):
            ath = item.get("athlete", {})
            out.append({"team": abbr(ath.get("team", {})), "player": ath.get("displayName"),
                        "key": norm_name(ath.get("displayName")), "status": item.get("status")})
    return out
