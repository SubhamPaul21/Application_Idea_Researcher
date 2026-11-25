from __future__ import annotations

from datetime import timedelta
from itertools import islice

from ..core.http import Http
from ..util import utcnow
from .apple_reviews import AppleReviews
from .base import FetchContext, Source
from .bluesky import Bluesky
from .cfpb import CFPB
from .github_issues import GitHubIssues
from .hackernews import HackerNews
from .stackexchange import StackExchange
from .youtube import YouTubeComments

_CLASSES: dict[str, type[Source]] = {
    cls.id: cls for cls in (HackerNews, StackExchange, GitHubIssues, AppleReviews, CFPB, Bluesky, YouTubeComments)
}


def implemented_ids() -> list[str]:
    return list(_CLASSES)


def get(source_id: str) -> Source:
    return _CLASSES[source_id]()


SMOKE_NICHE = {
    "id": "smoke",
    "label": "smoke test",
    "keywords": ["spreadsheet"],
    "github_keywords": ["export csv"],
    "stackexchange_sites": ["softwarerecs"],
    "app_store_terms": ["expense tracker"],
    "youtube_terms": ["excel tutorial"],
    "social_keywords": ["spreadsheet"],
    "cfpb_terms": ["credit report"],
}


def smoke(source_id: str, http: Http | None = None) -> tuple[bool, str]:
    """Fetch up to 5 real items to prove the endpoint works today."""
    src = get(source_id)
    ok, why = src.available()
    if not ok:
        return False, why
    messages: list[str] = []
    ctx = FetchContext(niche=SMOKE_NICHE, since=utcnow() - timedelta(days=90), max_results=5, max_keywords=1,
                       countries=["us"], http=http or Http(), log=messages.append)
    if source_id == "cfpb":
        ctx.max_results = 100  # aggregated: needs a page of complaints to form issue groups
    try:
        items = list(islice(src.fetch(ctx), 5))
    except Exception as e:  # noqa: BLE001 - report anything
        return False, f"{type(e).__name__}: {e}"
    if not items:
        return False, "no items returned" + (f" ({'; '.join(messages)})" if messages else "")
    return True, f"{len(items)} items, e.g. {items[0].url}"
