"""Hacker News via the official Algolia search API (Ask HN posts + comments)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from ..core.compliance import html_to_text
from ..core.http import HttpError
from ..core.models import RawItem
from .base import FetchContext, Source

URL = "https://hn.algolia.com/api/v1/search_by_date"


class HackerNews(Source):
    id = "hackernews"

    def fetch(self, ctx: FetchContext) -> Iterator[RawItem]:
        since_ts = int(ctx.since.timestamp())
        per_page = min(100, ctx.max_results)
        for term in ctx.terms("keywords"):
            got, page = 0, 0
            while got < ctx.max_results:
                params = {
                    "query": term, "tags": "(ask_hn,comment)", "hitsPerPage": per_page, "page": page,
                    "numericFilters": f"created_at_i>{since_ts}",
                }
                try:
                    data = ctx.http.get_json(URL, params)
                except HttpError as e:
                    ctx.log(f"Hacker News '{term}': {e}")
                    break
                hits = data.get("hits", [])
                for h in hits:
                    is_comment = bool(h.get("comment_text"))
                    text = html_to_text(h.get("comment_text") or h.get("story_text") or "")
                    title = h.get("title") or ""
                    if not (text or title):
                        continue
                    context = "Hacker News comment" if is_comment else "Ask HN post"
                    if is_comment and h.get("story_title"):
                        context += f" in the thread '{h['story_title']}'"
                    yield RawItem(
                        source=self.id,
                        source_id=str(h["objectID"]),
                        url=f"https://news.ycombinator.com/item?id={h['objectID']}",
                        title=title,
                        text=text,
                        created_at=datetime.fromtimestamp(h.get("created_at_i", since_ts), tz=timezone.utc),
                        engagement={"points": h.get("points") or 0, "comments": h.get("num_comments") or 0},
                        author=h.get("author") or "",
                        extra={"context": context},
                    )
                got += len(hits)
                page += 1
                if not hits or page >= data.get("nbPages", 0):
                    break
