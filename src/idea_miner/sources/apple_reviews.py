"""Apple App Store: find top apps via the iTunes Search API, then read 1-3 star reviews from the
public customer-review RSS/JSON feed. Review text is only paraphrased downstream, never republished."""
from __future__ import annotations

from typing import Any, Iterator

from ..core.http import HttpError
from ..core.models import RawItem
from ..util import parse_iso, utcnow
from .base import FetchContext, Source

SEARCH_URL = "https://itunes.apple.com/search"
REVIEWS_URL = "https://itunes.apple.com/{cc}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"


def _label(entry: dict[str, Any], *path: str) -> str:
    node: Any = entry
    for p in path:
        node = node.get(p, {}) if isinstance(node, dict) else {}
    return node.get("label", "") if isinstance(node, dict) else ""


class AppleReviews(Source):
    id = "apple_reviews"

    def fetch(self, ctx: FetchContext) -> Iterator[RawItem]:
        apps_per_term = int(self.meta.get("apps_per_term", 3))
        pages = int(self.meta.get("review_pages", 2))
        for cc in ctx.countries:
            seen: set[int] = set()
            for term in ctx.terms("app_store_terms"):
                try:
                    found = ctx.http.get_json(SEARCH_URL, {"term": term, "country": cc, "entity": "software",
                                                           "limit": apps_per_term})
                except HttpError as e:
                    ctx.log(f"App Store search '{term}' ({cc}): {e}")
                    continue
                for app in found.get("results", []):
                    app_id = app.get("trackId")
                    if not app_id or app_id in seen:
                        continue
                    seen.add(app_id)
                    yield from self._reviews(ctx, cc, app_id, app.get("trackName", ""), pages)

    def _reviews(self, ctx: FetchContext, cc: str, app_id: int, name: str, pages: int) -> Iterator[RawItem]:
        for page in range(1, pages + 1):
            try:
                feed = ctx.http.get_json(REVIEWS_URL.format(cc=cc, page=page, app_id=app_id))
            except HttpError:
                return
            entries = feed.get("feed", {}).get("entry", [])
            if isinstance(entries, dict):
                entries = [entries]
            if not entries:
                return
            too_old = False
            for e in entries:
                rating_s = _label(e, "im:rating")
                if not rating_s:
                    continue  # app metadata entry, not a review
                rating = int(rating_s)
                updated = parse_iso(_label(e, "updated")) or utcnow()
                if updated < ctx.since:
                    too_old = True
                    continue
                if rating > 3:
                    continue
                yield RawItem(
                    source=self.id,
                    source_id=f"{cc}-{_label(e, 'id')}",
                    url=f"https://apps.apple.com/{cc}/app/id{app_id}",
                    title=_label(e, "title"),
                    text=_label(e, "content"),
                    created_at=updated,
                    engagement={"votes": int(_label(e, "im:voteSum") or 0), "low_rating": 4 - rating},
                    author=_label(e, "author", "name"),
                    extra={"app": name, "app_id": app_id, "rating": rating, "country": cc,
                           "context": f"{rating}-star App Store review ({cc.upper()}) of the app '{name}'"},
                )
            if too_old:
                return
