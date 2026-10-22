"""
Client for the Finnish Basketball Association's results service (tulospalvelu.basket.fi), which is built on
TorneoPal. The site is a single-page app that calls a JSON API; this calls the same API the way the page does.

How it authenticates, and why nothing secret is stored here: the web page ships a client key in its own
JavaScript and sends it as the Accept header ("json/<key>") together with the page's Origin and Referer.
Every visitor's browser receives it. This client reads it from the page's JavaScript at run time rather than
keeping a copy, so no key is written to the repository, and a rotated key is picked up automatically.
The use is personal analysis at a polite rate (one request a second), cached on disk so nothing is fetched twice.

Raw responses are cached in .cache/finland/raw/ (git-ignored).
"""

import hashlib
import json
import re
import time

import requests

from predict_nba.finland import CACHE_DIR

BASE = "https://tulospalvelu.basket.fi"
API = "https://koripallo-api.torneopal.net/taso/rest"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
CACHE = CACHE_DIR / "raw"
DELAY = 1.0                       # seconds between requests


class TorneoPal:
    def __init__(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._key = None
        self._last = 0.0

    def key(self):
        """The client key, read from the page's own JavaScript (not stored)."""
        if self._key is None:
            # Look through every script the page loads: the bundle layout changes when the site is rebuilt
            # (October 2026 it moved from /js/chunk-vendors.*.js to /assets/index-*.js).
            index = self.session.get(BASE + "/", headers=UA, timeout=30).text
            scripts = re.findall(r'(?:src|href)="(/[^"]+\.js)"', index)
            for path in sorted(scripts, key=lambda p: ("index" not in p and "vendor" not in p, p)):
                code = self.session.get(BASE + path, headers=UA, timeout=90).text
                m = re.search(r'json/([a-z0-9]{12,})', code)
                if m:
                    self._key = m.group(1)
                    break
            if self._key is None:
                raise RuntimeError("client key not found in the results page's scripts")
        return self._key

    def call(self, method, refresh=False, **params):
        """One API call, cached by method and parameters."""
        tag = hashlib.sha1(json.dumps([method, sorted(params.items())]).encode()).hexdigest()[:16]
        path = CACHE / f"{method}_{tag}.json"
        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))
        wait = self._last + DELAY - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            self._last = time.monotonic()
            try:
                r = self.session.get(f"{API}/{method}", params=params, timeout=40,
                                     headers={**UA, "Accept": f"json/{self.key()}", "Origin": BASE, "Referer": BASE + "/"})
                if r.status_code == 200:
                    data = r.json()
                    if str(data.get("call", {}).get("status")).lower() == "ok":
                        path.write_text(json.dumps({"params": params, "method": method, "data": data}), encoding="utf-8")
                        return {"params": params, "method": method, "data": data}
                time.sleep(2 + attempt * 3)
            except Exception:
                time.sleep(2 + attempt * 3)
        raise RuntimeError(f"{method} {params} failed after retries")
