"""Tests for the two production fixes on this branch (empty odds table, season from the date)."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from predict_nba.utils.season import current_season                      # noqa: E402


def test_season_label_from_date():
    assert current_season(date(2026, 10, 7)) == "2026-27"
    assert current_season(date(2027, 3, 1)) == "2026-27"
    assert current_season(date(2026, 6, 30)) == "2025-26"
    assert current_season(date(2026, 7, 1)) == "2026-27"


def test_empty_odds_keep_their_columns(monkeypatch):
    import predict_nba.utils.oddsfetcher as of

    class Resp:
        def raise_for_status(self): pass
        def json(self): return []

    monkeypatch.setattr(of.requests, "get", lambda *a, **k: Resp())
    df = of.OddsFetcher.fetch_odds()
    assert df.empty
    assert list(df.columns) == ["home_team", "away_team", "home_odds", "away_odds"]
    # the exact expression that used to raise KeyError('home_team') in daily_generate.py
    assert df[(df["home_team"] == "Boston Celtics") & (df["away_team"] == "New York Knicks")].empty
