"""
The player model's loop, run as its own process next to the API and the daily regression job (start.sh).

Every 15 minutes:
1. Records results of yesterday's and today's finished games, and appends each finished regular-season game's ESPN
   box score to the player history, so tomorrow's features include it.
2. For games tipping off within the next LEAD_MIN minutes that have no prediction yet: takes ESPN's injury list now,
   predicts with the player model (players listed Out removed from the previous-game roster), asks the deployed
   regression for the same game in-process, and takes one odds call. With a 15-minute cadence each game is predicted
   once, 60 to 75 minutes before tip-off; a game whose window was missed is still predicted if it has not started.
3. Writes portfolio/nba-season.json, served at /portfolio/nba-season.json.

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
        predicted = {p["id"] for p in self.store.jsonl(pred_key)}

        todays, new_box = [], []
        for day in (today - timedelta(days=1), today):
            for g in espn.scoreboard(day):
                if g["season_type"] not in (espn.REGULAR, espn.PLAYOFFS):
                    continue
                if g["state"] == "post":
                    results[g["id"]] = {"home_pts": g["home_pts"], "away_pts": g["away_pts"],
                                        "hw": int(g["home_pts"] > g["away_pts"])}
                    if g["season_type"] == espn.REGULAR and g["id"] not in box_done:
                        new_box += espn.box_rows(g["id"], str(day), season)
                        box_done.add(g["id"])
                if day == today:
                    todays.append(g)
        self.store.put_json(res_key, results)

        history = None
        if new_box:
            history = pd.concat([self.store.history(), pd.DataFrame(new_box)], ignore_index=True)
            self.store.save_history(history)
            self.store.put_json(BOX_DONE, sorted(box_done))
            logger.info(f"player model: {len(new_box)} player rows added from finished games")

        tip = lambda g: datetime.fromisoformat(g["tip"].replace("Z", "+00:00"))
        due = [g for g in todays if g["state"] == "pre" and g["id"] not in predicted
               and now < tip(g) <= now + timedelta(minutes=LEAD_MIN)]
        if due:
            self.predict(due, today, season, now, history if history is not None else self.store.history(), pred_key)
        self.export(season)

    def predict(self, due, today, season, now, history, pred_key):
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
            xh, wh, ch, ih = team_inputs(rows, g["home"], today, season, self.model.meta, out.get(g["home"], set()))
            xa, wa, ca, ia = team_inputs(rows, g["away"], today, season, self.model.meta, out.get(g["away"], set()))
            X.append([xh, xa]); W.append([wh, wa]); C.append([ch, ca]); infos.append((ih, ia))
        p = self.model.p_home(np.array(X), np.array(W), np.array(C))
        prices = odds()
        records = []
        for g, ph, (ih, ia) in zip(due, p, infos):
            oh, oa = prices.get((norm_team(g["home_name"]), norm_team(g["away_name"])), (None, None))
            ps = shadow(g["home"], g["away"], self.regression())
            records.append({"id": g["id"], "date": str(today), "tip": g["tip"], "home": g["home"], "away": g["away"],
                            "pm": round(float(ph), 4), "ps": ps, "oh": oh, "oa": oa,
                            "predicted_at": now.isoformat(timespec="seconds"),
                            "minutes_before_tip": round((datetime.fromisoformat(g["tip"].replace("Z", "+00:00")) - now).total_seconds() / 60),
                            "out_home": ih["removed_out"], "out_away": ia["removed_out"],
                            "roster_from": [ih["roster_from"], ia["roster_from"]]})
            logger.info(f"player model: {g['away']} at {g['home']} -> {ph:.3f}, regression {ps}, price {oh}/{oa}")
        self.store.append_jsonl(pred_key, records)
        self.store.append_jsonl(f"player_live/injuries_{season}.jsonl",
                                [{"time": now.isoformat(timespec="seconds"), "espn": injuries}])

    def export(self, season):
        """Last season (fixed) plus this season's predictions and results, for the portfolio."""
        data = self.store.get_json(BACKTEST, {"backtest": {"season": "2025-26", "games": []}, "live": {}})
        results = self.store.get_json(f"player_live/results_{season}.json", {})
        games = []
        for p in self.store.jsonl(f"player_live/predictions_{season}.jsonl"):
            r = results.get(p["id"])
            games.append({"d": p["date"], "t": p["tip"], "h": p["home"], "a": p["away"], "hw": r["hw"] if r else None,
                          "pm": p["pm"], "ps": p["ps"], "oh": p["oh"], "oa": p["oa"]})
        data["live"] = {"season": season, "opens": "2026-10-20", "games": games}
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
