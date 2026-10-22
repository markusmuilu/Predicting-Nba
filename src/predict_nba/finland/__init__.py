"""
Finnish men's basketball ratings and predictions (Korisliiga, I divisioona, II divisioona).

Runs on Fly.io next to the NBA loop (predict_nba.finland.runner), moved here on 22 Oct 2026 from a GitHub Action in
markusmuilu/nba-dashboard whose scheduled runs started hours late. Data comes from the Finnish Basketball
Association's public results service and basket.fi's statistics service.
"""

import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("FINLAND_CACHE", Path.home() / ".cache" / "finland"))
SEED = 42
