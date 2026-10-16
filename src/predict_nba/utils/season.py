"""NBA season label from a date, so no season string is hardcoded."""

from datetime import date, datetime


def current_season(today=None):
    """'2026-27' for any date from 1 July 2026 to 30 June 2027 (a season starts in the autumn and ends by June)."""
    d = today or datetime.now().date()
    if isinstance(d, datetime):
        d = d.date()
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"
