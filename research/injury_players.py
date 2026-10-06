"""
Maps the player names printed in the injury reports to NBA player ids.

The reports print "Robinson III, Glenn" or "McCullar Jr., Kevin"; the box-score logs print
"Glenn Robinson III". Matching is done in three steps, from strictest to loosest, always within
the season of the report, and the method used is recorded for every row:

  exact_team     same normalised name among players who appeared for that team that season
  exact_season   same normalised name among all players that season (covers a player traded to a
                 team he has not yet played for)
  exact_any      same normalised name across every season on record, and the NBA's full player list. This is
                 what finds a star who missed the whole season and so has no box score in it
  fuzzy          closest name on the team (difflib ratio >= 0.88) when exactly one candidate qualifies,
                 or the same last name when it is unique on the team
  override       research/injury_name_overrides.csv (normalised report name -> player id), for the leftovers
  none           no match. Mostly players who never appear in a box score that season (two-way and
                 G League assignment players), which cannot be matched from game logs by construction.

Normalising: "Last, First" -> "first last", accents removed, lower case, punctuation removed, suffixes
(jr, sr, ii, iii, iv) dropped.
"""

import difflib
import re
import unicodedata

import pandas as pd

from research.config import ROOT
from research.fetch import nba_game_log

OVERRIDES = ROOT / "research" / "injury_name_overrides.csv"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalise(name):
    if not isinstance(name, str):
        return ""
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[.'’\-]", " ", name.lower())
    return " ".join(t for t in name.split() if t not in SUFFIXES)


def season_of(date_str):
    """'01/15/2025' -> '2024-25'."""
    d = pd.to_datetime(date_str, format="%m/%d/%Y")
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def build_lookup(seasons):
    """Per season: normalised name -> {player ids}, and (team abbreviation, normalised name) -> {player ids}."""
    logs = pd.concat([nba_game_log(s, "P").assign(season=s)[["season", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]]
                      for s in seasons], ignore_index=True).drop_duplicates()
    logs["key"] = logs["PLAYER_NAME"].map(normalise)
    by_season, by_team = {}, {}
    for r in logs.itertuples():
        by_season.setdefault((r.season, r.key), set()).add(r.PLAYER_ID)
        by_team.setdefault((r.season, r.TEAM_ABBREVIATION, r.key), set()).add(r.PLAYER_ID)
    names = {(s, t): sorted({k for (ss, tt, k) in by_team if ss == s and tt == t}) for (s, t, _) in by_team}
    return by_season, by_team, names


def build_any_season_lookup():
    """normalised name -> {ids}, from every cached box-score log and the NBA's static player list."""
    from nba_api.stats.static import players as nba_players
    from research.config import EXTRA_SEASONS, RAW_DIR, SEASONS
    by_any = {}
    for p in nba_players.get_players():
        by_any.setdefault(normalise(p["full_name"]), set()).add(p["id"])
    for s in EXTRA_SEASONS + SEASONS:
        path = RAW_DIR / f"nba_gamelog_P_{s}.csv"
        if path.exists():
            for pid, name in pd.read_csv(path, usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates().itertuples(index=False):
                by_any.setdefault(normalise(name), set()).add(pid)
    return by_any


def load_overrides():
    if not OVERRIDES.exists():
        return {}
    df = pd.read_csv(OVERRIDES)
    return {r.normalised_name: int(r.player_id) for r in df.itertuples()}


def map_players(reports, team_abbr):
    """
    reports: DataFrame from injury_parse_all with date, team (full name), player. team_abbr: full name -> abbreviation.
    Adds player_id (or <NA>) and match_method. Rows without a player (NOT YET SUBMITTED) are left unmapped.
    """
    df = reports.copy()
    df["season"] = df["date"].map(season_of)
    df["abbr"] = df["team"].map(team_abbr)
    df["key"] = df["player"].map(normalise)
    seasons = sorted(df.season.unique())
    by_season, by_team, names = build_lookup([s for s in seasons if s >= "2018-19"])
    overrides = load_overrides()
    by_any = build_any_season_lookup()
    ids, methods = [], []
    cache = {}
    for r in zip(df.season, df.abbr, df.key, df.player):
        season, abbr, key, raw = r
        ck = (season, abbr, key)
        if ck in cache:
            pid, method = cache[ck]
        elif not isinstance(raw, str):
            pid, method = None, "no_player"
        else:
            found = by_team.get((season, abbr, key))
            if found and len(found) == 1:
                pid, method = next(iter(found)), "exact_team"
            else:
                found = by_season.get((season, key))
                if found and len(found) == 1:
                    pid, method = next(iter(found)), "exact_season"
                elif key in overrides:
                    pid, method = overrides[key], "override"
                elif len(by_any.get(key, ())) == 1:
                    pid, method = next(iter(by_any[key])), "exact_any"
                else:
                    pid, method = None, "none"
                    cand = names.get((season, abbr), [])
                    close = difflib.get_close_matches(key, cand, n=2, cutoff=0.88)
                    if len(close) == 1:
                        ids_ = by_team[(season, abbr, close[0])]
                        if len(ids_) == 1:
                            pid, method = next(iter(ids_)), "fuzzy"
                    if pid is None:
                        last = key.split()[-1] if key else ""
                        same_last = [c for c in cand if c.split()[-1] == last]
                        if len(same_last) == 1:
                            ids_ = by_team[(season, abbr, same_last[0])]
                            if len(ids_) == 1:
                                pid, method = next(iter(ids_)), "fuzzy"
            cache[ck] = (pid, method)
        ids.append(pid)
        methods.append(method)
    df["player_id"] = pd.array(ids, dtype="Int64")
    df["match_method"] = methods
    return df
