"""
Dataset for the deep-learning experiments: raw player sequences, and hand-built features
computed from those very same sequences.

For every game, both teams, the ten players with the most expected minutes are described by
their last 20 box-score rows *before this game's date*. That window is the only player
information used. From it we derive two views:

  SEQ  the raw window, one row per past game (16 box-score stats, days since the previous
       game, days before the target game, same-season flag, valid flag), standardised.
       A sequence model can read this directly.
  HF   22 hand-built numbers computed from the same window: last-10 and last-20 means,
       per-36-minute rates, shooting shape, days since last game, how much history exists.
       These play the role of the "feature engineering" of the earlier experiments.

Because both views come from identical raw data, comparing a model that reads SEQ with one
that reads HF isolates one question: does the network need the features to be built for it?

Rosters, as before: "previous" (players in the team's previous game, known before tip-off,
the fair setting) and "actual" (players in this game, optimistic). In both, the window of
each player ends strictly before the game being predicted, so a player in the previous-game
roster has his previous game in his window: no stale-by-one-game features this time.

Output (data/processed/dl_<mode>_*.npy): SEQ [G,2,10,20,20] float16, HF [G,2,10,22], EXP [G,2,10]
(expected minutes, 0 = empty slot), CTX [G,2,3], plus dl_games.csv with the game table and
point-in-time league context (league points per game and home win rate so far this season).
"""

import numpy as np
import pandas as pd

from research.config import EXTRA_SEASONS, PROCESSED_DIR, RAW_DIR, SEASONS, VAL_SEASON
from research.fetch import nba_game_log
from research.player_features import REST_CAP, load_games, team_context

STATS = ["MIN", "PTS", "REB", "OREB", "AST", "STL", "BLK", "TOV", "PF", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS"]
N_SEQ = 20
TOP = 10
CHANNELS = len(STATS) + 4      # + gap since previous game, days before target, same season, valid
PRIOR_MINUTES = 12.0
HF_NAMES = (["r10_MIN", "r10_PTS", "r10_REB", "r10_AST", "r10_STL", "r10_BLK", "r10_TOV", "r10_PM",
             "r20_MIN", "r20_PTS", "r20_PM", "ts_pct", "fg3_rate", "ft_rate", "usage36", "pts36", "reb36", "ast36", "pm36",
             "log_n_games", "days_since_last", "no_history"])
KEY_SCALE = 100000     # pidx * KEY_SCALE + day ordinal sorts rows by player then date


def season_start(s):
    return int(s[:4])


def load_all(seasons):
    games = load_games(seasons).sort_values(["date", "game_id"]).reset_index(drop=True)
    plog = pd.concat([nba_game_log(s, "P").assign(season=s) for s in seasons], ignore_index=True)
    plog[STATS] = plog[STATS].fillna(0)
    plog = plog[plog["MIN"] > 0].copy()
    plog["day"] = (pd.to_datetime(plog["GAME_DATE"]) - pd.Timestamp("1970-01-01")).dt.days
    plog["pidx"] = plog["PLAYER_ID"].map({pid: k for k, pid in enumerate(plog["PLAYER_ID"].unique())})
    plog["sstart"] = plog["season"].map(season_start)
    plog = plog.sort_values(["pidx", "day", "GAME_ID"]).reset_index(drop=True)

    rosters = {(gid, team): grp["pidx"].to_numpy() for (gid, team), grp in plog.groupby(["GAME_ID", "TEAM_ABBREVIATION"])}
    ok = [(g.game_id, g.home) in rosters and (g.game_id, g.away) in rosters for g in games.itertuples()]
    return games[ok].reset_index(drop=True), plog, rosters


def league_context(games):
    """
    Point-in-time league context: league mean points per team-game and home win rate in the
    current season *before* each game, with last season's full-season value as a prior worth
    200 games so the first weeks of a season are not noise.
    """
    g = games.copy()
    g["pts"] = (g["PTS_home"] + g["PTS_away"]) / 2.0
    out_ppg, out_hw = np.zeros(len(g)), np.zeros(len(g))
    prev_ppg, prev_hw = 100.0, 0.58
    for season, idx in g.groupby("season", sort=False).indices.items():
        idx = np.sort(idx)
        cum_pts = np.concatenate([[0.0], np.cumsum(g["pts"].to_numpy()[idx])[:-1]])
        cum_hw = np.concatenate([[0.0], np.cumsum(g["home_win"].to_numpy()[idx])[:-1]])
        k = np.arange(len(idx))
        out_ppg[idx] = (prev_ppg * 200 + cum_pts) / (200 + k)
        out_hw[idx] = (prev_hw * 200 + cum_hw) / (200 + k)
        prev_ppg, prev_hw = g["pts"].to_numpy()[idx].mean(), g["home_win"].to_numpy()[idx].mean()
    return out_ppg, out_hw


def window_arrays(plog):
    keys = plog["pidx"].to_numpy().astype(np.int64) * KEY_SCALE + plog["day"].to_numpy()
    raw = plog[STATS].to_numpy(np.float32)
    day = plog["day"].to_numpy()
    pidx = plog["pidx"].to_numpy()
    sstart = plog["sstart"].to_numpy()
    prev_gap = np.full(len(plog), 60.0, np.float32)
    same = pidx[1:] == pidx[:-1]
    prev_gap[1:][same] = np.minimum(day[1:][same] - day[:-1][same], 60)
    return keys, raw, day, pidx, sstart, prev_gap


def gather(query_pidx, query_day, query_sstart, arrays):
    """
    Raw windows for many (player, day) queries: the last N_SEQ rows strictly before `day`.
    Returns window [Q, N_SEQ, CHANNELS] (unstandardised, zero padded on the left) and the raw stats.
    """
    keys, raw, day, pidx, sstart, prev_gap = arrays
    q = query_pidx.astype(np.int64) * KEY_SCALE + query_day
    pos = np.searchsorted(keys, q, side="left")                       # rows before this index are earlier
    steps = pos[:, None] - N_SEQ + np.arange(N_SEQ)[None, :]           # [Q, N_SEQ]
    inside = steps >= 0
    steps_c = np.clip(steps, 0, len(keys) - 1)
    valid = inside & (pidx[steps_c] == query_pidx[:, None])
    win = np.zeros((len(q), N_SEQ, CHANNELS), np.float32)
    win[:, :, :len(STATS)] = raw[steps_c]
    win[:, :, len(STATS)] = prev_gap[steps_c]
    win[:, :, len(STATS) + 1] = np.minimum(query_day[:, None] - day[steps_c], 120)
    win[:, :, len(STATS) + 2] = (sstart[steps_c] == query_sstart[:, None])
    win[:, :, len(STATS) + 3] = 1.0
    win *= valid[:, :, None]
    return win, valid


def hand_features(win, valid):
    """22 hand-built numbers from the raw window (no standardisation yet)."""
    s = {n: win[:, :, i] for i, n in enumerate(STATS)}
    v10, v20 = valid[:, -10:], valid
    n10 = np.maximum(v10.sum(1), 1)
    n20 = np.maximum(v20.sum(1), 1)
    m10 = lambda x: x[:, -10:].sum(1) / n10
    sum10 = lambda x: x[:, -10:].sum(1)
    shots = sum10(s["FGA"]) + 0.44 * sum10(s["FTA"])
    mins = np.maximum(sum10(s["MIN"]), 1.0)
    last_day_gap = np.where(valid.any(1), win[:, -1, len(STATS) + 1], 120.0)
    feats = [m10(s["MIN"]), m10(s["PTS"]), m10(s["REB"]), m10(s["AST"]), m10(s["STL"]), m10(s["BLK"]), m10(s["TOV"]), m10(s["PLUS_MINUS"]),
             s["MIN"].sum(1) / n20, s["PTS"].sum(1) / n20, s["PLUS_MINUS"].sum(1) / n20,
             np.where(shots > 0, sum10(s["PTS"]) / (2 * np.maximum(shots, 1e-9)), 0.0),
             np.where(sum10(s["FGA"]) > 0, sum10(s["FG3A"]) / np.maximum(sum10(s["FGA"]), 1e-9), 0.0),
             np.where(sum10(s["FGA"]) > 0, sum10(s["FTA"]) / np.maximum(sum10(s["FGA"]), 1e-9), 0.0),
             36 * (shots + sum10(s["TOV"])) / mins, 36 * sum10(s["PTS"]) / mins, 36 * sum10(s["REB"]) / mins,
             36 * sum10(s["AST"]) / mins, 36 * sum10(s["PLUS_MINUS"]) / mins,
             np.log1p(valid.sum(1)), np.minimum(last_day_gap, 60) / 60.0, (~valid.any(1)).astype(np.float32)]
    return np.stack(feats, axis=1).astype(np.float32)


def build(roster_mode, seasons=None, chunk=1500):
    seasons = seasons or [s for s in EXTRA_SEASONS if (RAW_DIR / f"nba_gamelog_P_{s}.csv").exists()
                          and (RAW_DIR / f"nba_gamelog_T_{s}.csv").exists()] + SEASONS
    games, plog, rosters = load_all(seasons)
    arrays = window_arrays(plog)
    G = len(games)
    pre_val = games["season"].map(season_start).to_numpy() < season_start(VAL_SEASON)

    # Previous game of each team: id of the game whose roster is "known before tip-off".
    last, prev_id = {}, {}
    for g in games.itertuples():
        for team in (g.home, g.away):
            prev_id[(g.game_id, team)] = last.get(team)
            last[team] = g.game_id

    day_of = (pd.to_datetime(games["date"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    sstart_of = games["season"].map(season_start).to_numpy()

    SEQ = np.zeros((G, 2, TOP, N_SEQ, CHANNELS), np.float16)
    HF = np.zeros((G, 2, TOP, len(HF_NAMES)), np.float32)
    EXP = np.zeros((G, 2, TOP), np.float32)
    PIDX = np.full((G, 2, TOP), -1, np.int32)                      # player index in each slot, -1 = empty
    raw_keep = []                                                    # (game, side, slot) raw stats to standardise later

    for start in range(0, G, chunk):
        stop = min(G, start + chunk)
        q_p, q_d, q_s, q_where = [], [], [], []
        for i in range(start, stop):
            g = games.iloc[i]
            for side, team in enumerate((g["home"], g["away"])):
                src = g["game_id"]
                if roster_mode == "previous" and prev_id[(g["game_id"], team)] is not None:
                    src = prev_id[(g["game_id"], team)]
                ids = rosters[(src, team)]
                q_p.append(ids)
                q_d.append(np.full(len(ids), day_of[i]))
                q_s.append(np.full(len(ids), sstart_of[i]))
                q_where.append(np.stack([np.full(len(ids), i), np.full(len(ids), side)], 1))
        q_p, q_d, q_s, where = map(np.concatenate, (q_p, q_d, q_s, q_where))
        win, valid = gather(q_p, q_d, q_s, arrays)
        hf = hand_features(win, valid)
        exp_min = np.where(valid[:, -10:].any(1), hf[:, 0], PRIOR_MINUTES)       # r10_MIN is the expected minutes
        for i in range(start, stop):
            for side in (0, 1):
                m = np.where((where[:, 0] == i) & (where[:, 1] == side))[0]
                order = m[np.argsort(-exp_min[m], kind="stable")][:TOP]
                k = len(order)
                SEQ[i, side, :k] = win[order]
                HF[i, side, :k] = hf[order]
                EXP[i, side, :k] = np.maximum(exp_min[order], 0.5)
                PIDX[i, side, :k] = q_p[order]

    # Standardise with pre-validation games only (these are training data in every window setting).
    use = np.broadcast_to(pre_val[:, None, None], EXP.shape) & (EXP > 0)
    hf_mu, hf_sd = HF[use].mean(0), HF[use].std(0) + 1e-6
    HF = np.where((EXP > 0)[..., None], (HF - hf_mu) / hf_sd, 0.0).astype(np.float32)
    seq_use = SEQ[use].astype(np.float32)                                      # [n, N_SEQ, CHANNELS]
    flat = seq_use.reshape(-1, CHANNELS)
    flat = flat[flat[:, -1] > 0]                                               # valid rows only
    s_mu, s_sd = flat.mean(0), flat.std(0) + 1e-6
    s_mu[-1], s_sd[-1] = 0.0, 1.0                                              # leave the valid flag as 0/1
    valid_flag = SEQ[..., -1:] > 0
    SEQ = np.where(valid_flag, (SEQ.astype(np.float32) - s_mu) / s_sd, 0.0).astype(np.float16)
    SEQ[..., -1] = valid_flag[..., 0]

    ctx = team_context(games)
    CTX = np.zeros((G, 2, 3), np.float32)
    for i, g in enumerate(games.itertuples()):
        CTX[i, 0] = ctx.loc[(g.game_id, g.home)].to_numpy()
        CTX[i, 1] = ctx.loc[(g.game_id, g.away)].to_numpy()

    ppg, hw = league_context(games)
    games = games.assign(league_ppg=ppg, league_home_win=hw, season_start=sstart_of)
    games.to_csv(PROCESSED_DIR / "dl_games.csv", index=False)
    for name, arr in (("seq", SEQ), ("hf", HF), ("exp", EXP), ("ctx", CTX), ("pidx", PIDX)):
        np.save(PROCESSED_DIR / f"dl_{roster_mode}_{name}.npy", arr)
    print(f"[{roster_mode}] {G} games, seasons {seasons[0]}..{seasons[-1]}, SEQ {SEQ.shape}, "
          f"players per side avg {(EXP > 0).sum(-1).mean():.1f}")


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "previous")
