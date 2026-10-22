"""
The player model for any two teams, now: each team's roster from its latest game, minus the players ESPN lists as
out, the same inputs the loop uses for a real prediction. Serves the portfolio's "Ask the model".

The player history and the injury list are kept in memory for 15 minutes; each request then only builds features
for the two rosters.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from predict_nba.player import espn
from predict_nba.player.features import player_rows, standardise, team_inputs
from predict_nba.player.model import PlayerModel
from predict_nba.player.store import META, MODEL, Store
from predict_nba.utils.season import current_season

ET = pytz.timezone("America/New_York")
TTL = 15 * 60
_state = {"at": 0.0}


def _load():
    if time.time() - _state["at"] > TTL:
        store = Store()
        if "model" not in _state:
            _state["model"] = PlayerModel(store.get(MODEL), store.get(META))
        h = store.history()
        _state["history"] = h.assign(d=pd.to_datetime(h.GAME_DATE))
        out = {}
        for i in espn.injuries():
            if i["status"] == "Out":
                out.setdefault(i["team"], {})[i["key"]] = i["player"]
        _state["out"] = out
        _state["at"] = time.time()
    return _state


def predict(home, away):
    s = _load()
    h, model = s["history"], s["model"]
    today = datetime.now(ET).date()
    season = current_season(today)
    ids = {}
    for t in (home, away):
        games = h[h.TEAM_ABBREVIATION == t]
        if games.empty:
            raise ValueError(f"No games for {t}.")
        ids[t] = games.sort_values("d").GAME_ID.iloc[-1]
    keys = set(h.loc[h.GAME_ID.isin(ids.values()), "key"])
    rows = standardise(player_rows(h[h.key.isin(keys)].drop(columns="d")), model.meta)
    xh, wh, ch, ih = team_inputs(rows, home, today, season, model.meta, set(s["out"].get(home, {})))
    xa, wa, ca, ia = team_inputs(rows, away, today, season, model.meta, set(s["out"].get(away, {})))
    p = float(model.p_home(np.array([[xh, xa]]), np.array([[wh, wa]]), np.array([[ch, ca]]))[0])
    return {"home": home, "away": away, "p_home": round(p, 4),
            "roster_from": {home: ih["roster_from"], away: ia["roster_from"]},
            "out": {home: ih["removed_out"], away: ia["removed_out"]},
            "injuries_at": datetime.fromtimestamp(s["at"]).astimezone(pytz.utc).strftime("%Y-%m-%d %H:%M UTC")}
