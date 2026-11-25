from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RawItem(BaseModel):
    """One post/comment/review/issue as returned by a source adapter (before storage)."""

    source: str
    source_id: str
    url: str
    title: str = ""
    text: str = ""
    created_at: datetime
    engagement: dict[str, float] = Field(default_factory=dict)
    author: str = ""  # hashed before storage, never stored raw
    extra: dict[str, Any] = Field(default_factory=dict)  # "context" is shown to the LLM

    @property
    def id(self) -> str:
        return f"{self.source}:{self.source_id}"
