"""
The player model's loop, run as its own process next to the API and the daily regression job (start.sh).

Every 15 minutes:
1. Records results of yesterday's and today's finished games, and appends each finished regular-season game's ESPN
   box score to the player history, so tomorrow's features include it.
2. Predicts each game twice, both stored and never changed:
   early   as soon as both teams have finished their previous game and its box score is in the history, for games
           up to two days ahead;
   final   for games tipping off within the next LEAD_MIN minutes, with ESPN's injury list taken then. With a
           15-minute cadence that is 60 to 75 minutes before tip-off; a game whose window was missed is still
           predicted if it has not started. The pre-game record is scored on the final prediction.
   Each round: the player model (players listed Out removed from the previous-game roster), the deployed regression
   for the same game in-process, and one odds call for the prices at that moment.
3. Writes portfolio/nba-season.json, served at /portfolio/nba-season.json: every prediction with its result and score,
   and the schedule of games not predicted yet (today and the next two days), so the page can show what is coming.

A failure in one step is logged and the loop carries on; nothing here touches the regression's own daily job.
"""

import time
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz

from predict_nba.player import espn
from predict_nba.player.features import player_rows, standardise, team_inputs
from predict_nba.player.model import PlayerModel
from predict_nba.player.store import BACKTEST, BOX_DONE, META, MODEL, PORTFOLIO, Store
from predict_nba.utils.logger import logger
from predict_nba.utils.season import current_season

ET = pytz.timezone("America/New_York")
LEAD_MIN = 75
EVERY_MIN = 15


def norm_team(name):
    n = (name or "").lower().replace("los angeles ", "la ")
    return "".join(ch for ch in n if ch.isalnum())


def shadow(home, away, predictor):
    """The deployed regression's P(home win), computed the same way as the /predict endpoint."""
    from predict_nba.pipeline.data_cleaner import DataCleaner
    from predict_nba.pipeline.data_collector import DataCollector
    try:
        collector = DataCollector()
        collector.get_current_season(home)
        collector.get_current_season(away)
        DataCleaner().clean_prediction_data(home, away)
        r = predictor.predict_matchup_with_bundle(home, away)
        c = float(r["confidence"]) / 100
        return round(c if str(r["winner"]) == home else 1 - c, 4)
    except Exception:
        logger.error(f"shadow {home}-{away} failed: {traceback.format_exc(limit=2)}")
        return None


def odds():
    from predict_nba.utils.oddsfetcher import OddsFetcher
    try:
        df = OddsFetcher.fetch_odds()
        return {(norm_team(r.home_team), norm_team(r.away_team)): (float(r.home_odds), float(r.away_odds))
                for r in df.itertuples()}
    except Exception:
        logger.error(f"odds failed: {traceback.format_exc(limit=2)}")
        return {}


class PlayerRunner:
    def __init__(self):
        self.store = Store()
        self.model = PlayerModel(self.store.get(MODEL), self.store.get(META))
        self.predictor = None

    def regression(self):
        if self.predictor is None:
            from predict_nba.pipeline.model_predictor import ModelPredictor
            self.predictor = ModelPredictor()
            self.predictor.load_bundle()
        return self.predictor

    def step(self, now=None):
        now = now or datetime.now(timezone.utc)
        today = now.astimezone(ET).date()
        season = current_season(today)
        pred_key, res_key = f"player_live/predictions_{season}.jsonl", f"player_live/results_{season}.json"
        results = self.store.get_json(res_key, {})
        box_done = set(self.store.get_json(BOX_DONE, []))
        logged = self.store.jsonl(pred_key)
        done_final = {p["id"] for p in logged if p.get("round", "final") == "final"}
        done_early = {p["id"] for p in logged if p.get("round") == "early"}

        days = [today + timedelta(days=k) for k in (-1, 0, 1, 2)]
        board, new_box = {}, []
        for day in days:
            board[day] = [g for g in espn.scoreboard(day) if g["season_type"] in (espn.REGULAR, espn.PLAYOFFS)]
            for g in board[day]:
                g["day"] = day
                if g["state"] == "post":
                    results[g["id"]] = {"home_pts": g["home_pts"], "away_pts": g["away_pts"],
                                        "hw": int(g["home_pts"] > g["away_pts"])}
                    if g["season_type"] == espn.REGULAR and g["id"] not in box_done:
                        new_box += espn.box_rows(g["id"], str(day), season)
                        box_done.add(g["id"])
        self.store.put_json(res_key, results)

        history = None
        if new_box:
            history = pd.concat([self.store.history(), pd.DataFrame(new_box)], ignore_index=True)
            self.store.save_history(history)
            self.store.put_json(BOX_DONE, sorted(box_done))
            logger.info(f"player model: {len(new_box)} player rows added from finished games")

        tip = lambda g: datetime.fromisoformat(g["tip"].replace("Z", "+00:00"))
        window = [g for day in days for g in board[day]]

        def ready(g):
            """Both teams have played every game before this one, and those games are in the history."""
            for team in (g["home"], g["away"]):
                for o in window:
                    if team in (o["home"], o["away"]) and tip(o) < tip(g):
                        if o["state"] != "post" or (o["season_type"] == espn.REGULAR and o["id"] not in box_done):
                            return False
            return True

        upcoming = [g for g in window if g["state"] == "pre" and now < tip(g)]
        final_due = [g for g in upcoming if g["id"] not in done_final and tip(g) <= now + timedelta(minutes=LEAD_MIN)]
        early_due = [g for g in upcoming if g["id"] not in done_final and g["id"] not in done_early
                     and tip(g) > now + timedelta(minutes=LEAD_MIN) and ready(g)]
        for rnd, due in (("final", final_due), ("early", early_due)):
            if due:
                if history is None:
                    history = self.store.history()
                self.predict(due, season, now, history, pred_key, rnd)
        done_final |= {g["id"] for g in final_due}
        done_any = done_final | done_early | {g["id"] for g in early_due}
        schedule = [{"d": str(g["day"]), "t": g["tip"], "h": g["home"], "a": g["away"]}
                    for g in upcoming if g["day"] >= today and g["id"] not in done_any]
        self.export(season, schedule)

    def predict(self, due, season, now, history, pred_key, rnd):
        injuries = espn.injuries()
        out = {}
        for i in injuries:
            if i["status"] == "Out":
                out.setdefault(i["team"], set()).add(i["key"])
        # A player's features depend only on his own games, so only the players in the teams' last games are needed.
        teams = {t for g in due for t in (g["home"], g["away"])}
        last = history.assign(d=pd.to_datetime(history.GAME_DATE)).sort_values("d")
        last_ids = {last[last.TEAM_ABBREVIATION == t].GAME_ID.iloc[-1] for t in teams if (last.TEAM_ABBREVIATION == t).any()}
        keys = set(history.loc[history.GAME_ID.isin(last_ids), "key"])
        rows = standardise(player_rows(history[history.key.isin(keys)]), self.model.meta)
        X, W, C, infos = [], [], [], []
        for g in due:
            xh, wh, ch, ih = team_inputs(rows, g["home"], g["day"], season, self.model.meta, out.get(g["home"], set()))
            xa, wa, ca, ia = team_inputs(rows, g["away"], g["day"], season, self.model.meta, out.get(g["away"], set()))
            X.append([xh, xa]); W.append([wh, wa]); C.append([ch, ca]); infos.append((ih, ia))
        p = self.model.p_home(np.array(X), np.array(W), np.array(C))
        prices = odds()
        records = []
        for g, ph, (ih, ia) in zip(due, p, infos):
            oh, oa = prices.get((norm_team(g["home_name"]), norm_team(g["away_name"])), (None, None))
            ps = shadow(g["home"], g["away"], self.regression())
            records.append({"id": g["id"], "round": rnd, "date": str(g["day"]), "tip": g["tip"], "home": g["home"], "away": g["away"],
                            "pm": round(float(ph), 4), "ps": ps, "oh": oh, "oa": oa,
                            "predicted_at": now.isoformat(timespec="seconds"),
                            "minutes_before_tip": round((datetime.fromisoformat(g["tip"].replace("Z", "+00:00")) - now).total_seconds() / 60),
                            "out_home": ih["removed_out"], "out_away": ia["removed_out"],
                            "roster_from": [ih["roster_from"], ia["roster_from"]]})
            logger.info(f"player model ({rnd}): {g['away']} at {g['home']} -> {ph:.3f}, regression {ps}, price {oh}/{oa}")
        self.store.append_jsonl(pred_key, records)
        self.store.append_jsonl(f"player_live/injuries_{season}.jsonl",
                                [{"time": now.isoformat(timespec="seconds"), "round": rnd, "espn": injuries}])

    def export(self, season, schedule=()):
        """Last season (fixed) plus this season's predictions, results and schedule, for the portfolio."""
        data = self.store.get_json(BACKTEST, {"backtest": {"season": "2025-26", "games": []}, "live": {}})
        results = self.store.get_json(f"player_live/results_{season}.json", {})
        by_id = {}
        for p in self.store.jsonl(f"player_live/predictions_{season}.jsonl"):
            by_id.setdefault(p["id"], {})[p.get("round", "final")] = p
        games = []
        for gid, rounds in by_id.items():
            p = rounds.get("final") or rounds["early"]
            e = rounds.get("early")
            r = results.get(gid)
            games.append({"d": p["date"], "t": p["tip"], "h": p["home"], "a": p["away"], "hw": r["hw"] if r else None,
                          "hs": r["home_pts"] if r else None, "as": r["away_pts"] if r else None,
                          "r": "final" if "final" in rounds else "early",
                          "pm": p["pm"], "ps": p["ps"], "oh": p["oh"], "oa": p["oa"],
                          "pa": p["predicted_at"], "xo": [p.get("out_home", []), p.get("out_away", [])],
                          "early": {"pm": e["pm"], "ps": e["ps"], "oh": e["oh"], "oa": e["oa"], "pa": e["predicted_at"]}
                          if e and "final" in rounds else None})
        # Games the loop failed to predict (out of memory on 21 Oct 2026), reconstructed afterwards from the history
        # before that day: published as backtests, never counted in the pre-game record.
        for p in self.store.get_json(f"player_live/reconstructed_{season}.json", []):
            if p["id"] in by_id:
                continue
            r = results.get(p["id"])
            games.append({"d": p["date"], "t": p["tip"], "h": p["home"], "a": p["away"], "hw": r["hw"] if r else None,
                          "hs": r["home_pts"] if r else None, "as": r["away_pts"] if r else None, "r": "reconstructed",
                          "pm": p["pm"], "ps": None, "oh": None, "oa": None, "pa": p["computed_at"], "xo": [[], []],
                          "early": None})
        games.sort(key=lambda g: g["t"])
        data["live"] = {"season": season, "opens": "2026-10-20", "games": games, "schedule": list(schedule)}
        data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.store.put_json(PORTFOLIO + "nba-season.json", data)


def run():
    runner = None
    while True:
        try:
            runner = runner or PlayerRunner()
            runner.step()
        except Exception:
            logger.error(f"player model step failed: {traceback.format_exc()}")
        now = datetime.now(timezone.utc)
        wait = EVERY_MIN * 60 - (now.minute % EVERY_MIN) * 60 - now.second
        time.sleep(max(wait, 30))


if __name__ == "__main__":
    run()
