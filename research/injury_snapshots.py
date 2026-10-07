"""
Step 3 of the brief: what the injury report said about each game at T-60, T-30 and T-15 minutes before tip-off.

For every game that appears in the parsed reports (data/processed/injury_reports.csv.gz), and for each cutoff:
  1. the tip-off time comes from the reports themselves (date and time in Eastern time, '07:00 (ET)', with
     hours 1 to 11 read as p.m. and 12 as noon; NBA games do not tip off before noon);
  2. the snapshot is the latest report whose own header time is at or before tip-off minus the cutoff and
     that lists the game. Header times, not file names, are used throughout (the file named 05PM holds the 5:30 report);
  3. its rows for that game are the snapshot. If no report lists the game by then, the game has no snapshot at that cutoff.

Output:
  data/processed/injury_snapshots.csv.gz    one row per game, cutoff, team and player listed
  data/processed/injury_snapshot_games.csv  one row per game and cutoff: which report was used, how long before tip
                                            it was filed, and how many players it lists
  research/results/injury_snapshot_summary.json   how often the cutoffs pick different reports or different statuses

Run: python -m research.injury_snapshots
"""

import json
from bisect import bisect_right

import pandas as pd

from research.config import PROCESSED_DIR, RESULTS_DIR
from research.injury_reports import split_status

CUTOFFS = (60, 30, 15)


def tip_times(df):
    """One tip-off time per (date, matchup): the most common time printed across all reports that list the game."""
    games = df[["date", "matchup", "time"]].dropna().copy()
    hh = games["time"].str.slice(0, 2).astype(int)
    games["hour"] = hh.where(hh == 12, hh + 12)
    games["tip"] = pd.to_datetime(games["date"] + " " + games["hour"].astype(str) + ":" + games["time"].str.slice(3, 5), format="%m/%d/%Y %H:%M")
    mode = games.groupby(["date", "matchup"])["tip"].agg(lambda s: s.mode().iloc[0])
    return mode.rename("tip").reset_index()


def main():
    df = pd.read_csv(PROCESSED_DIR / "injury_reports.csv.gz", dtype={"time": str}, parse_dates=["report_time"])
    df = df[df.report_time.notna() & df.matchup.notna() & df.date.notna()]
    fixed = [split_status(st, rs) for st, rs in zip(df.status, df.reason)]       # the merged-cell layout of 16-17 Dec 2019
    df["status"], df["reason"] = [f[0] for f in fixed], [f[1] for f in fixed]
    tips = tip_times(df)
    print(f"{len(df):,} rows, {df.file.nunique()} reports, {len(tips)} distinct games with a tip time", flush=True)

    # For each game, the sorted reports that list it
    listing = df[["date", "matchup", "file", "report_time"]].drop_duplicates().sort_values("report_time")
    by_game = {k: (g.report_time.tolist(), g.file.tolist()) for k, g in listing.groupby(["date", "matchup"])}
    rows_by = {k: g for k, g in df.groupby(["file", "date", "matchup"], sort=False)}

    game_rows, snap_rows = [], []
    for t in tips.itertuples():
        times, files = by_game[(t.date, t.matchup)]
        for cut in CUTOFFS:
            cutoff = t.tip - pd.Timedelta(minutes=cut)
            i = bisect_right(times, cutoff)
            if i == 0:
                game_rows.append({"date": t.date, "matchup": t.matchup, "tip": t.tip, "cutoff": cut, "file": None, "report_time": None,
                                  "minutes_before_tip": None, "n_listed": 0})
                continue
            f, rt = files[i - 1], times[i - 1]
            sub = rows_by[(f, t.date, t.matchup)]
            game_rows.append({"date": t.date, "matchup": t.matchup, "tip": t.tip, "cutoff": cut, "file": f, "report_time": rt,
                              "minutes_before_tip": (t.tip - rt).total_seconds() / 60, "n_listed": int(sub.player.notna().sum())})
            snap = sub.assign(cutoff=cut, tip=t.tip)
            snap_rows.append(snap[["date", "matchup", "tip", "cutoff", "file", "report_time", "team", "player", "status", "reason", "category"]])
    games = pd.DataFrame(game_rows)
    snaps = pd.concat(snap_rows, ignore_index=True)
    games.to_csv(PROCESSED_DIR / "injury_snapshot_games.csv", index=False)
    snaps.to_csv(PROCESSED_DIR / "injury_snapshots.csv.gz", index=False)

    # How much do the cutoffs differ?
    wide = games.pivot(index=["date", "matchup"], columns="cutoff", values="file")
    season = pd.to_datetime(wide.index.get_level_values("date"), format="%m/%d/%Y")
    season = pd.Series([f"{d.year if d.month >= 7 else d.year - 1}-{str(d.year + 1 if d.month >= 7 else d.year)[2:]}" for d in season], index=wide.index)
    status = snaps.assign(key=snaps.team.fillna("") + "|" + snaps.player.fillna("")).pivot_table(
        index=["date", "matchup", "key"], columns="cutoff", values="status", aggfunc="first")
    status["diff_60_15"] = status[60].fillna("-") != status[15].fillna("-")
    status["diff_60_30"] = status[60].fillna("-") != status[30].fillna("-")
    status["diff_30_15"] = status[30].fillna("-") != status[15].fillna("-")
    changed_games = status.groupby(level=["date", "matchup"])[["diff_60_30", "diff_30_15", "diff_60_15"]].any()
    summary = {}
    for s in sorted(season.unique()):
        idx = wide.index[season == s]
        w = wide.loc[idx]
        ch = changed_games.reindex(idx).fillna(False)
        summary[s] = {"games": int(len(idx)),
                      "no_snapshot_at_T60": int(w[60].isna().sum()),
                      "different_report_60_vs_30": float((w[60] != w[30]).mean()), "different_report_30_vs_15": float((w[30] != w[15]).mean()),
                      "different_report_60_vs_15": float((w[60] != w[15]).mean()),
                      "any_status_changed_60_vs_30": float(ch.diff_60_30.mean()), "any_status_changed_30_vs_15": float(ch.diff_30_15.mean()),
                      "any_status_changed_60_vs_15": float(ch.diff_60_15.mean())}
    dist = games[games.minutes_before_tip.notna()].groupby(["cutoff"]).minutes_before_tip.describe().round(1)
    (RESULTS_DIR / "injury_snapshot_summary.json").write_text(json.dumps(summary, indent=1))
    print(pd.DataFrame(summary).T.round(3).to_string())
    print("\nminutes between the chosen report and tip-off, by cutoff:\n", dist.to_string())


if __name__ == "__main__":
    main()
