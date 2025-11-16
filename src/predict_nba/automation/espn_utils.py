"""
ESPN-related helpers for daily automation.

Includes:
- ESPN → internal abbreviation normalization
- UTC → EST date conversion for ESPN event dates
"""

from datetime import datetime

import pytz

# ESPN uses some abbreviations that differ from your internal pipeline.
# Map ESPN → internal.
ESPN_ABBR_FIX = {
    "UTAH": "UTA",
    "GS": "GSW",
    "NY": "NYK",
    "SA": "SAS",
    "NO": "NOP",
    "WSH": "WAS", 
}


def normalize_abbr(raw_abbr: str) -> str:
    """Normalize ESPN team abbreviations into the project's abbreviations."""
    if not raw_abbr:
        return None
    return ESPN_ABBR_FIX.get(raw_abbr.upper(), raw_abbr.upper())


def espn_to_est_date(espn_date_str: str) -> str:
    """
    Convert an ESPN UTC datetime string (ISO format) to an EST calendar date.

    Returns:
        str: date as 'YYYY-MM-DD' in America/New_York timezone.
    """
    utc = pytz.utc
    est = pytz.timezone("America/New_York")

    dt_utc = datetime.fromisoformat(espn_date_str.replace("Z", "+00:00"))
    dt_est = dt_utc.astimezone(est)
    return dt_est.strftime("%Y-%m-%d")
