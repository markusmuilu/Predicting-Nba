"""
History and current prediction management built on S3 JSON files.

Files:
- history/prediction_history.json: all past finished games with results.
- current/current_predictions.json: predictions for games not yet resolved.
"""

import sys
import json
import io

import numpy as np

from predict_nba.utils.s3_client import S3Client
from predict_nba.utils.exception import CustomException


class HistoryManager:
    """Provides high-level access to current and historical prediction data."""

    HISTORY_KEY = "history/prediction_history.json"
    CURRENT_KEY = "current/current_predictions.json"

    def __init__(self):
        try:
            self.s3 = S3Client()
        except Exception as e:
            CustomException(f"Failed to initialize S3 client for HistoryManager: {e}", sys)
            self.s3 = None

    @staticmethod
    def _clean(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    def load_current_predictions(self):
        """Return the list of current predictions from S3 (or [])."""
        if self.s3 is None:
            return []
        return json.loads(self.s3.download(self.CURRENT_KEY).decode("utf-8").strip())

    def save_current_predictions(self, rows):
        """Overwrite current_predictions JSON in S3."""
        if self.s3 is None:
            return
        # Clean numpy types to keep JSON friendly
        cleaned = [{k: self._clean(v) for k, v in row.items()} for row in rows]
        data = json.dumps(cleaned, indent=2).encode("utf-8")
        self.s3.upload(self.CURRENT_KEY, data, "application/json")

    def append_history(self, new_entries):
        """
        Append new entries to prediction_history JSON in S3.

        Skips entries whose gameId already exists in history.
        """
        if self.s3 is None or not new_entries:
            return

        new_entries = [{k: self._clean(v) for k, v in row.items()} for row in new_entries]
        history = json.loads(self.s3.download(self.HISTORY_KEY).decode("utf-8").strip())
        existing_ids = {h.get("gameId") for h in history if "gameId" in h}

        to_add = [e for e in new_entries if e.get("gameId") not in existing_ids]
        history.extend(to_add)

        self.s3.upload(self.HISTORY_KEY, history, "application/json")
