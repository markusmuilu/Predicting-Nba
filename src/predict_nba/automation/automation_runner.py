"""
Automation runner for Docker.
On first launch:
    - Runs prediction update immediately.
After that:
    - Runs every day at 12:00 Helsinki time.
"""

import time
import traceback
from datetime import datetime, timedelta
import pytz

from predict_nba.automation import DailyPredictor
from predict_nba.utils.logger import logger
from predict_nba.utils.wait_for_model import wait_for_required_files


wait_for_required_files()


HELSINKI_TZ = pytz.timezone("Europe/Helsinki")
RUN_HOUR = 12
RUN_MINUTE = 0


def seconds_until_next_noon():
    """Calculate seconds until next 12:00 Helsinki time."""
    now = datetime.now(HELSINKI_TZ)
    today_noon = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)

    if now >= today_noon:
        today_noon += timedelta(days=1)

    delta = today_noon - now
    return max(1, int(delta.total_seconds()))


def run_daily_automation():
    dp = DailyPredictor()

    # First time run immediatly
    try:
        logger.info("Running FIRST automation job immediately...")
        dp.run_all()
        logger.info("First run complete.")
    except Exception:
        logger.error("First run failed:")
        logger.error(traceback.format_exc())

    # Now schedule daily runs
    while True:
        try:
            sleep_seconds = seconds_until_next_noon()
            logger.info(f"Sleeping {sleep_seconds} seconds until next run at 12:00 Helsinki…")
            time.sleep(sleep_seconds)

            logger.info("Running scheduled daily prediction job...")
            dp.run_all()
            logger.info("Scheduled job completed.")
        except Exception:
            logger.error("Automation failed:")
            logger.error(traceback.format_exc())
            time.sleep(60)


if __name__ == "__main__":
    run_daily_automation()
