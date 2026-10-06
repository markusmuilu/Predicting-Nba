"""Tests for the due-window logic of research/live_logger.py. No network: fetchers are faked."""

from datetime import datetime, timedelta, timezone

import pytest

from research import live_logger as L

TIP = datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)


def event(eid="1", tip=TIP, state="pre"):
    return {"event_id": eid, "tip": tip, "state": state, "home": "BOS", "away": "NYK"}


def fresh_state():
    return {"done": {}, "last_hourly": None, "last_pdf_et": None}


def test_not_due_before_the_cutoff():
    now = TIP - timedelta(minutes=61)
    assert L.due_items([event()], fresh_state(), now) == []


def test_due_exactly_at_each_cutoff():
    for cut in L.CUTOFFS:
        now = TIP - timedelta(minutes=cut)
        due = L.due_items([event()], fresh_state(), now)
        assert [c for _, c, _ in due] == [cut]


def test_late_tick_still_catches_it_but_only_within_the_window():
    assert L.due_items([event()], fresh_state(), TIP - timedelta(minutes=60 - 9))
    assert L.due_items([event()], fresh_state(), TIP - timedelta(minutes=60 - L.WINDOW_MIN - 1)) == []


def test_never_early_and_never_after_tip():
    assert L.due_items([event()], fresh_state(), TIP - timedelta(minutes=75)) == []
    assert L.due_items([event()], fresh_state(), TIP + timedelta(minutes=1)) == []


def test_already_recorded_is_not_due_again():
    state = fresh_state()
    state["done"]["1:60"] = "x"
    assert L.due_items([event()], state, TIP - timedelta(minutes=60)) == []


def test_only_pregame_events_are_due():
    assert L.due_items([event(state="in")], fresh_state(), TIP - timedelta(minutes=60)) == []


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(L, "ROOT", tmp_path)
    return tmp_path


def run_tick(now, events, odds_calls, isolated):
    def fake_fetch(url, params=None, retries=3, timeout=20, binary=False):
        if url == L.ESPN_SCOREBOARD:
            return {"events": [{"id": e["event_id"], "date": e["tip"].strftime("%Y-%m-%dT%H:%MZ"),
                                "competitions": [{"status": {"type": {"state": e["state"]}},
                                                  "competitors": [{"homeAway": "home", "team": {"abbreviation": "BOS"}},
                                                                  {"homeAway": "away", "team": {"abbreviation": "NYK"}}]}]}
                               for e in events]}
        if url == L.ESPN_INJURIES:
            return {"injuries": []}
        return None                                    # no PDF found

    def fake_odds(now):
        odds_calls.append(now)
        return {"path": f"data/live/odds/odds_{L.stamp(now)}.json", "remaining": "400"}

    return L.tick(now=now, fetch=fake_fetch, odds_fn=fake_odds)


def test_games_sharing_a_tip_time_share_one_odds_call(isolated):
    calls = []
    evs = [event("1"), event("2"), event("3")]
    s = run_tick(TIP - timedelta(minutes=60), evs, calls, isolated)
    assert s["due"] == 3 and len(calls) == 1


def test_no_odds_call_at_the_t30_cutoff(isolated):
    calls = []
    run_tick(TIP - timedelta(minutes=30), [event()], calls, isolated)
    assert calls == []


def test_tick_is_idempotent(isolated):
    calls = []
    now = TIP - timedelta(minutes=60)
    run_tick(now, [event()], calls, isolated)
    s2 = run_tick(now + timedelta(minutes=2), [event()], calls, isolated)
    assert s2["due"] == 0 and len(calls) == 1


def test_pdf_candidate_names_cover_both_formats():
    et = L.ET.localize(datetime(2025, 1, 15, 17, 0))
    assert L.pdf_candidates(et) == ["2025-01-15_05_00PM", "2025-01-15_05PM"]
    assert L.pdf_candidates(et + timedelta(minutes=15)) == ["2025-01-15_05_15PM"]
