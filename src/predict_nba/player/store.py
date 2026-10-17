"""R2 storage for the player model: weights, the box-score history, predictions, results, and the portfolio data."""

import io
import json

import pandas as pd

from predict_nba.utils.s3_client import S3Client

PREFIX = "player_live/"
MODEL = PREFIX + "player_model.npz"
META = PREFIX + "player_model_meta.json"
HISTORY = PREFIX + "player_box.csv.gz"
BOX_DONE = PREFIX + "box_done.json"
BACKTEST = PREFIX + "backtest.json"
PORTFOLIO = "portfolio/"


class Store:
    def __init__(self):
        self.s3 = S3Client()

    def get(self, key):
        return self.s3.download(key)

    def put(self, key, content: bytes, content_type=None):
        self.s3.upload(key, content, content_type)

    def get_json(self, key, default):
        b = self.get(key)
        return json.loads(b) if b else default

    def put_json(self, key, obj):
        self.put(key, json.dumps(obj, separators=(",", ":")).encode(), "application/json")

    def history(self):
        return pd.read_csv(io.BytesIO(self.get(HISTORY)), dtype={"GAME_ID": str}, compression="gzip")

    def save_history(self, df):
        buf = io.BytesIO()
        df.to_csv(buf, index=False, compression={"method": "gzip"})
        self.put(HISTORY, buf.getvalue(), "application/gzip")

    def jsonl(self, key):
        b = self.get(key)
        return [json.loads(l) for l in b.decode().splitlines() if l.strip()] if b else []

    def append_jsonl(self, key, rows):
        if not rows:
            return
        old = self.get(key) or b""
        new = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        self.put(key, old + new, "application/x-ndjson")
