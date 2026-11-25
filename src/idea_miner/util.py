from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: object) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Convert now so an impossible value (e.g. year 9999 with a negative offset) fails here, not at storage.
        return dt.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def chunks(seq: Sequence[T], size: int) -> Iterator[list[T]]:
    size = max(1, size)
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].rstrip("-") or "item"
