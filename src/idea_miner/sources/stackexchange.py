"""Stack Exchange official API v2.3 (Software Recommendations, Web Apps, Super User, ...).
Content is CC BY-SA 4.0: specs link back to every question used."""
from __future__ import annotations

import html
import time
from datetime import datetime, timezone
from typing import Iterator

from .. import config
from ..core.compliance import html_to_text
from ..core.http import HttpError
from ..core.models import RawItem
from .base import FetchContext, Source

URL = "https://api.stackexchange.com/2.3/search/advanced"
MAX_BACKOFF_S = 60


class StackExchange(Source):
    id = "stackexchange"

    def fetch(self, ctx: FetchContext) -> Iterator[RawItem]:
        key = config.env("STACKAPPS_KEY")
        sites = ctx.niche.get("stackexchange_sites") or ["softwarerecs", "webapps", "superuser"]
        for site in sites:
            for term in ctx.terms("keywords"):
                page, got = 1, 0
                while got < ctx.max_results:
                    params = {
                        "site": site, "q": term, "fromdate": int(ctx.since.timestamp()), "order": "desc",
                        "sort": "creation", "pagesize": min(100, ctx.max_results), "page": page,
                        "filter": "withbody",
                    }
                    if key:
                        params["key"] = key
                    try:
                        data = ctx.http.get_json(URL, params)
                    except HttpError as e:
                        ctx.log(f"Stack Exchange {site} '{term}': {e}")
                        break
                    items = data.get("items", [])
                    for q in items:
                        owner = q.get("owner") or {}
                        yield RawItem(
                            source=self.id,
                            source_id=f"{site}-{q['question_id']}",
                            url=q.get("link", ""),
                            title=html.unescape(q.get("title", "")),
                            text=html_to_text(q.get("body", "")),
                            created_at=datetime.fromtimestamp(q.get("creation_date", 0), tz=timezone.utc),
                            engagement={"score": q.get("score", 0), "answers": q.get("answer_count", 0),
                                        "views": q.get("view_count", 0)},
                            author=str(owner.get("user_id") or owner.get("display_name") or ""),
                            extra={"site": site, "tags": q.get("tags", []),
                                   "context": f"Stack Exchange question on {site}.stackexchange.com"},
                        )
                    got += len(items)
                    if data.get("backoff"):
                        wait = int(data["backoff"])
                        if wait > MAX_BACKOFF_S:  # obey the API, but don't hang the whole run on a huge value
                            ctx.log(f"Stack Exchange asked to wait {wait}s; skipping the rest of this source.")
                            return
                        time.sleep(wait)
                    if data.get("quota_remaining", 1) <= 2:
                        ctx.log("Stack Exchange daily quota nearly used up; add STACKAPPS_KEY for 10k/day.")
                        return
                    if not data.get("has_more") or not items:
                        break
                    page += 1
