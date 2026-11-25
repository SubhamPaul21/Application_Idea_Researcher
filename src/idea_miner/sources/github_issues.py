"""GitHub feature requests: open issues labelled enhancement with many reactions."""
from __future__ import annotations

from datetime import timedelta
from typing import Iterator

from .. import config
from ..core.http import HttpError
from ..core.models import RawItem
from ..util import parse_iso, utcnow
from .base import FetchContext, Source

URL = "https://api.github.com/search/issues"


def github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = config.env("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_search_interval() -> float:
    # Search API: 30 req/min with a token, 10 req/min anonymous
    return 2.1 if config.env("GITHUB_TOKEN") else 6.5


class GitHubIssues(Source):
    id = "github"

    def fetch(self, ctx: FetchContext) -> Iterator[RawItem]:
        # Long-ignored, highly-upvoted requests are the most valuable, so look back at least a year.
        since = min(ctx.since, utcnow() - timedelta(days=365))
        min_reactions = int(ctx.niche.get("github_min_reactions", 5))
        labels = self.meta.get("labels") or ["enhancement"]
        for term in ctx.terms("github_keywords"):
            for label in labels:
                label_q = f'label:"{label}"' if " " in label else f"label:{label}"
                q = (f"{term} is:issue is:open {label_q} reactions:>={min_reactions} "
                     f"created:>={since.date().isoformat()}")
                params = {"q": q, "sort": "reactions", "order": "desc", "per_page": min(100, ctx.max_results)}
                try:
                    data = ctx.http.get_json(URL, params, headers=github_headers(),
                                             min_interval=github_search_interval())
                except HttpError as e:
                    if e.status in (403, 429):
                        ctx.log("GitHub rate limit reached; add GITHUB_TOKEN to .env for higher limits.")
                        return
                    ctx.log(f"GitHub '{term}': {e}")
                    continue
                for it in data.get("items", []):
                    if "pull_request" in it:
                        continue
                    repo = it.get("repository_url", "").split("/repos/")[-1]
                    reactions = (it.get("reactions") or {}).get("total_count", 0)
                    yield RawItem(
                        source=self.id,
                        source_id=str(it["id"]),
                        url=it.get("html_url", ""),
                        title=it.get("title", ""),
                        text=(it.get("body") or "")[:6000],
                        created_at=parse_iso(it.get("created_at")) or utcnow(),
                        engagement={"reactions": reactions, "comments": it.get("comments", 0)},
                        author=(it.get("user") or {}).get("login", ""),
                        extra={"repo": repo,
                               "context": f"GitHub feature request in {repo} ({reactions} reactions)"},
                    )
