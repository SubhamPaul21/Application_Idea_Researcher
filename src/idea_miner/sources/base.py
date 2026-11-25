from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterator

from .. import config
from ..core.http import Http
from ..core.models import RawItem


@dataclass
class FetchContext:
    niche: dict[str, Any]
    since: datetime
    max_results: int          # per query
    max_keywords: int         # queries per source per niche
    countries: list[str]      # lower-case ISO codes, e.g. ["us", "in"]
    http: Http
    log: Callable[[str], None]

    def terms(self, field: str = "keywords") -> list[str]:
        values = self.niche.get(field) or self.niche.get("keywords") or []
        return [v for v in values if v][: self.max_keywords]


class Source(ABC):
    id: str = ""

    def __init__(self) -> None:
        self.meta: dict[str, Any] = config.sources_registry().get(self.id, {})

    @property
    def label(self) -> str:
        return self.meta.get("label", self.id)

    def available(self) -> tuple[bool, str]:
        if not self.meta.get("enabled", False):
            return False, "disabled in config/sources.yaml"
        key_env = self.meta.get("key_env")
        if self.meta.get("key_required") and not config.env(key_env):
            return False, f"needs {key_env} in .env"
        return True, "ready"

    def applies_to(self, niche: dict[str, Any]) -> bool:
        return True

    @abstractmethod
    def fetch(self, ctx: FetchContext) -> Iterator[RawItem]:
        """Yield items for the niche. Handle HTTP errors internally: log via ctx.log and move on."""
