"""Shared HTTP session with retry/backoff for rate-limited public APIs."""
from __future__ import annotations

import sys
import threading
import time
from urllib.parse import urlparse

import requests

_session: requests.Session | None = None

# Minimum seconds between requests per host (public read limits are modest).
MIN_INTERVAL = {"api.elections.kalshi.com": 0.12, "gamma-api.polymarket.com": 0.05,
                "clob.polymarket.com": 0.05}
_last: dict[str, float] = {}
_lock = threading.Lock()


def _pace(url: str) -> None:
    host = urlparse(url).hostname or ""
    gap = MIN_INTERVAL.get(host, 0.0)
    if not gap:
        return
    with _lock:
        now = time.monotonic()
        wait = _last.get(host, 0.0) + gap - now
        _last[host] = max(now, _last.get(host, 0.0) + gap)
    if wait > 0:
        time.sleep(wait)


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = "prediction-arb-scanner/0.1"
    return _session


def get_json(url: str, params: dict | None = None, *, method: str = "GET", json_body=None,
             retries: int = 10, timeout: float = 30.0):
    delay = 1.0
    last_err: Exception | None = None
    for _ in range(retries):
        _pace(url)
        try:
            r = session().request(method, url, params=params, json=json_body, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else delay
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err or 'rate limited'}")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
