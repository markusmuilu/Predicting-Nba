"""
Parses every downloaded injury report PDF into one table, and checks each file against the schedule.

Output:
  data/processed/injury_reports.csv.gz        one row per report, game, team and player
  data/processed/injury_parse_summary.csv     one row per file: report time, rows, game blocks, how many match the
                                              real schedule, parse problems

A file counts as clean when it has a header time, no parse problems, and every game block it lists exists on that
date in the schedule. Blocks missing from the schedule are reported, not dropped: a postponed game can legitimately
appear in a report.

Run: python -m research.injury_parse_all     (a few minutes, eight processes)
"""

import csv
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from research.config import PROCESSED_DIR, ROOT
from research.injury_reports import parse_pdf, validate
from research.player_features import load_games

PDF_DIR = ROOT / "data" / "injury_pdf"
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
_SCHEDULE = {}


def _init():
    _SCHEDULE["games"] = load_games(SEASONS)


def parse_one(path):
    try:
        df, rep_time, problems = parse_pdf(str(path))
    except Exception as e:                                    # one bad file must not stop the run
        return path.name, None, {"file": path.name, "error": f"{type(e).__name__}: {e}"}
    summary = {"file": path.name, "report_time": rep_time, "rows": len(df), "problems": "; ".join(problems)[:300], "error": ""}
    if not df.empty:
        v = validate(df, _SCHEDULE["games"])
        summary.update(blocks=v["blocks"], blocks_matched=v["blocks_matched"], rows_wrong_team=v["rows_wrong_team"],
                       rows_unknown_team=v["rows_unknown_team"], unmatched=str(v["blocks_unmatched"])[:200])
        df["file"] = path.name
    return path.name, df, summary


def main():
    files = sorted(p for p in PDF_DIR.glob("*.pdf"))
    print(f"parsing {len(files)} files", flush=True)
    frames, summaries = [], []
    with ProcessPoolExecutor(8, initializer=_init) as pool:
        for i, (name, df, summary) in enumerate(pool.map(parse_one, files, chunksize=20), start=1):
            summaries.append(summary)
            if df is not None and not df.empty:
                frames.append(df)
            if i % 2000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    rows.to_csv(PROCESSED_DIR / "injury_reports.csv.gz", index=False)
    summ = pd.DataFrame(summaries)
    summ.to_csv(PROCESSED_DIR / "injury_parse_summary.csv", index=False)
    clean = summ[(summ.error == "") & (summ.problems == "") & (summ.get("blocks_matched") == summ.get("blocks"))]
    print(f"{len(rows):,} rows from {len(frames)} files; {len(clean)} of {len(summ)} files fully clean")
    print("files with errors:", int((summ.error != "").sum()), " with parse problems:", int((summ.problems != "").sum()))


if __name__ == "__main__":
    main()
