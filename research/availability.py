"""
Step 4 of the brief: from an injury report snapshot to an availability-aware team strength.

Pipeline, for every game and each cutoff (T-60, T-30, T-15):

1. Candidate players for each team: the ten players of the team's previous game with the most expected minutes
   (the roster known before tip-off, as in the earlier "previous-game roster" models) plus every player the
   report lists for the team (a star who missed the last game and is "Probable" tonight is not in the previous
   roster but is on the report).
2. P(plays) for each candidate:
   - listed players: the observed share who played, by status and reason group (a lookup table fitted on the
     training seasons only: injury, rest, G League, other), then a one-variable logistic adjustment for
     the player's expected minutes;
   - players who are not listed: a logistic regression on expected minutes (starters almost always play,
     fringe players often do not).
3. Minutes: expected minutes_i = lambda * P(plays)_i * exp_min_i, where lambda >= 1 spreads the minutes of absent
   players over the remaining ones so that a team's minutes stay close to what it had (capped at 1.5).
4. Strength: the sum over candidates of (5 * minutes_i / sum of minutes) * rating_i, using each player's rating as
   it stood before the game (player Elo and ridge plus-minus snapshots from research/rating_snapshots.py).

Everything fitted here (the lookup table, the logistic models, lambda's cap) uses the training seasons
2019-20 to 2023-24 only. Validation (2024-25) and test (2025-26) are scored once.

Outputs: data/processed/avail_features.csv.gz (one row per game and cutoff) and research/results/availability_summary.json.

Run: python -m research.availability
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.config import PROCESSED_DIR, RESULTS_DIR, TEST_SEASON, TRAIN_SEASONS, VAL_SEASON
from research.dl_data import KEY_SCALE, N_SEQ, gather, hand_features, load_all, window_arrays
from research.injury_reports import team_abbreviations
from research.player_ratings import usable_seasons

CUTOFFS = (60, 30, 15)
TRAIN = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]       # the first season with injury reports to the last training season
LAMBDA_CAP = 1.5
PRIOR_MINUTES = 12.0


def reason_group(reason, category=None):
    """injury | rest | gleague | personal | none, from the reason text (and the 2018-19 Category column)."""
    text = f"{category or ''} {reason or ''}".lower()
    if "g league" in text or "two-way" in text or "two way" in text:
        return "gleague"
    if "rest" in text and "restrict" not in text:
        return "rest"
    if any(k in text for k in ("personal", "not with team", "suspension", "trade", "coach", "health and safety", "covid", "bereave")):
        return "personal"
    if text.strip() in ("", "-", "- -", "nan -", "nan nan"):
        return "none"
    return "injury"


def load_listed():
    """Snapshot rows with player ids and a game id; one row per (game, cutoff, player)."""
    snaps = pd.read_csv(PROCESSED_DIR / "injury_snapshots.csv.gz", dtype={"matchup": str})
    from research.injury_reports import split_status
    fx = [split_status(a, b) for a, b in zip(snaps.status, snaps.reason)]
    snaps["status"], snaps["reason"] = [f[0] for f in fx], [f[1] for f in fx]
    snaps = snaps[snaps.player.notna() & snaps.status.isin(["Out", "Questionable", "Doubtful", "Probable", "Available"])].copy()
    pmap = pd.read_csv(PROCESSED_DIR / "injury_player_map.csv.gz")[["date", "team", "player", "player_id", "abbr"]].drop_duplicates(["date", "team", "player"])
    snaps = snaps.merge(pmap, on=["date", "team", "player"], how="left")
    snaps["reason_group"] = [reason_group(r, c) for r, c in zip(snaps.reason, snaps.category)]
    return snaps


def main():
    seasons = usable_seasons()
    games, plog, rosters = load_all(seasons)
    games["date_us"] = pd.to_datetime(games["date"]).dt.strftime("%m/%d/%Y")
    games["matchup"] = games["away"] + "@" + games["home"]
    gidx = {(r.date_us, r.matchup): i for i, r in enumerate(games.itertuples())}
    pid_to_idx = dict(zip(plog["PLAYER_ID"], plog["pidx"]))

    snaps = load_listed()
    snaps["g"] = [gidx.get((d, m), -1) for d, m in zip(snaps.date, snaps.matchup)]
    snaps = snaps[(snaps.g >= 0) & snaps.player_id.notna()].copy()
    snaps["pidx"] = snaps.player_id.astype("int64").map(pid_to_idx)
    snaps = snaps[snaps.pidx.notna()].copy()
    snaps["pidx"] = snaps.pidx.astype(int)
    snaps["side"] = np.where(snaps.abbr == games.home.to_numpy()[snaps.g], 0, np.where(snaps.abbr == games.away.to_numpy()[snaps.g], 1, -1))
    snaps = snaps[snaps.side >= 0].copy()
    print(f"{len(snaps):,} listed (game, cutoff, player) rows with an id", flush=True)

    # Who actually played: (game id, pidx) with minutes
    played = set(zip(plog["GAME_ID"], plog["pidx"]))
    snaps["game_id"] = games.game_id.to_numpy()[snaps.g]
    snaps["played"] = [(gid, p) in played for gid, p in zip(snaps.game_id, snaps.pidx)]

    # Expected minutes for each listed player at the game date (the mean of his last 10 games' minutes)
    arrays = window_arrays(plog)
    day_of = (pd.to_datetime(games["date"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    sstart = games.season.str[:4].astype(int).to_numpy()
    q = snaps.drop_duplicates(["g", "pidx"])[["g", "pidx"]]
    win, valid = gather(q.pidx.to_numpy(), day_of[q.g.to_numpy()], sstart[q.g.to_numpy()], arrays)
    hf = hand_features(win, valid)
    expm = pd.Series(np.where(valid[:, -10:].any(1), hf[:, 0], PRIOR_MINUTES), index=pd.MultiIndex.from_arrays([q.g.to_numpy(), q.pidx.to_numpy()]))
    snaps["exp_min"] = [expm[(g, p)] for g, p in zip(snaps.g, snaps.pidx)]
    snaps["season"] = games.season.to_numpy()[snaps.g]

    # --- P(plays | listed): lookup by (status, reason group), then a minutes adjustment
    tr = snaps[snaps.season.isin(TRAIN)]
    table = tr.groupby(["status", "reason_group"]).played.agg(["mean", "count"]).reset_index()
    print("\nShare of listed players who played, training seasons:\n", table.round(3).to_string(index=False), flush=True)
    lookup = {(r.status, r.reason_group): (r["mean"] if r["count"] >= 30 else None) for _, r in table.iterrows()}
    fallback = tr.groupby("status").played.mean().to_dict()
    snaps["p_base"] = [lookup.get((s, r)) if lookup.get((s, r)) is not None else fallback[s] for s, r in zip(snaps.status, snaps.reason_group)]
    snaps["logit_base"] = np.log(np.clip(snaps.p_base, 0.01, 0.99) / (1 - np.clip(snaps.p_base, 0.01, 0.99)))
    adj = LogisticRegression(C=1e6).fit(np.c_[tr.assign(l=snaps.loc[tr.index, "logit_base"]).l, np.log1p(tr.exp_min)], tr.played)
    X = np.c_[snaps.logit_base, np.log1p(snaps.exp_min)]
    snaps["p_play"] = adj.predict_proba(X)[:, 1]

    summary = {"listed_rows": int(len(snaps)), "training_table": table.round(4).to_dict("records"), "calibration": {}}
    for name, mask in (("train", snaps.season.isin(TRAIN)), ("validation", snaps.season == VAL_SEASON), ("test", snaps.season == TEST_SEASON)):
        sub = snaps[mask]
        by = sub.groupby("status").agg(predicted=("p_play", "mean"), actual=("played", "mean"), n=("played", "size")).round(3)
        ll = -np.mean(sub.played * np.log(np.clip(sub.p_play, 1e-6, 1)) + (1 - sub.played) * np.log(np.clip(1 - sub.p_play, 1e-6, 1)))
        summary["calibration"][name] = {"log_loss": float(ll), "by_status": by.reset_index().to_dict("records")}
        print(f"\nP(plays) {name}: log loss {ll:.4f}\n{by.to_string()}", flush=True)
    snaps.to_pickle(PROCESSED_DIR / "avail_listed.pkl")
    (RESULTS_DIR / "availability_summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
