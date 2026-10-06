"""
Shared settings for the player-model research.

Everything that defines the experiment lives here, so the split and the seeds
cannot drift between scripts.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Local cache for downloaded data. data/ is git-ignored.
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Outputs that are committed: metrics, plots, predictions per model.
RESULTS_DIR = ROOT / "research" / "results"

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# Fixed chronological split. Every model is trained, tuned and scored on these.
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
VAL_SEASON = "2024-25"
TEST_SEASON = "2025-26"

# Extra, older seasons for the "does more history help?" experiments. They are only ever
# used for training (they precede the training window), so the validation and test
# seasons and the fixed split are unchanged.
EXTRA_SEASONS = ["2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20"]

SEED = 42

for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
