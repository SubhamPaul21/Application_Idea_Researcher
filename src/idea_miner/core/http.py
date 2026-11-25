"""Polite HTTP client: per-host throttling, retries with Retry-After, generic User-Agent."""
from __future__ import annotations

import threading
import time
from typing import Any

import httpx

from .. import config


class HttpError(Exception):
    def __init__(self, status: int, url: str, detail: str = "") -> None:
        self.status = status
        self.url = url
        self.detail = detail
        super().__init__(f"HTTP {status} from {url}" + (f": {detail}" if detail else ""))


class Http:
    def __init__(self) -> None:
        cfg = config.settings().get("http", {})
        self.client = httpx.Client(
            timeout=cfg.get("timeout_s", 30),
            headers={"User-Agent": cfg.get("user_agent", "idea-miner/0.1")},
            follow_redirects=True,
        )
        self.min_interval: dict[str, float] = cfg.get("min_interval_s", {})
        self._next_slot: dict[str, float] = {}
        self._lock = threading.Lock()

    def _throttle(self, host: str, override: float | None) -> None:
        interval = override if override is not None else self.min_interval.get(host, 0.25)
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, 0.0))
            self._next_slot[host] = slot + interval
        if slot > now:
            time.sleep(slot - now)

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json: Any = None,
        min_interval: float | None = None,
        retries: int = 3,
    ) -> httpx.Response:
        host = httpx.URL(url).host
        for attempt in range(retries + 1):
            self._throttle(host, min_interval)
            try:
                resp = self.client.request(method, url, params=params, headers=headers, json=json)
            except httpx.TransportError as e:
                if attempt == retries:
                    raise HttpError(0, url, type(e).__name__) from e
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 1)
                time.sleep(min(wait, 60))
                continue
            if resp.status_code >= 400:
                raise HttpError(resp.status_code, url, resp.text[:200].replace("\n", " "))
            return resp
        raise HttpError(0, url, "retries exhausted")

    def get_json(self, url: str, params: dict[str, Any] | None = None, **kw: Any) -> Any:
        return self.request("GET", url, params=params, **kw).json()

    def post_json(self, url: str, payload: Any, **kw: Any) -> Any:
        return self.request("POST", url, json=payload, **kw).json()
