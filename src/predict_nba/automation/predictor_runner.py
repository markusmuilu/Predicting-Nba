"""
High-level API for daily prediction automation.

Provides a small wrapper around:
- update_predictions()  (finished games → history)
- generate_new_predictions()  (new games → current_predictions)
"""

from predict_nba.automation.daily_update import update_predictions
from predict_nba.automation.daily_generate import generate_new_predictions
from predict_nba.utils.logger import logger


class DailyPredictor:
    """
    Orchestrates the daily prediction workflow.

    Typical usage:
        dp = DailyPredictor()
        dp.run_all()
    """

    def run_update_only(self):
        """Resolve finished games based on ESPN and update history."""
        logger.info("Running daily update for finished games...")
        return update_predictions()

    def run_generate_only(self):
        """Generate predictions for today's unstarted games."""
        logger.info("Running daily generation for today's games...")
        return generate_new_predictions()

    def run_all(self):
        """
        First resolve any finished games,
        then generate new predictions for today's schedule.
        """
        updated = self.run_update_only()
        created = self.run_generate_only()
        return {"updated": updated, "created": created}
