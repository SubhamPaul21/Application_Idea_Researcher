"""YouTube Data API v3: comments on tutorial videos ("how do I do X" pain).
Quota: search.list has its own 100 calls/day bucket; commentThreads.list costs 1 unit."""
from __future__ import annotations

import html
from typing import Iterator

from .. import config
from ..core.http import HttpError
from ..core.models import RawItem
from ..util import parse_iso, utcnow
from .base import FetchContext, Source

API = "https://www.googleapis.com/youtube/v3"


class YouTubeComments(Source):
    id = "youtube"

    def fetch(self, ctx: FetchContext) -> Iterator[RawItem]:
        # Key in a header, not the URL, so it never appears in URLs that end up in logs, proxies or error text.
        auth = {"X-Goog-Api-Key": config.env("YOUTUBE_API_KEY") or ""}
        videos_per_term = int(self.meta.get("videos_per_term", 3))
        for term in ctx.terms("youtube_terms"):
            params = {"part": "snippet", "q": term, "type": "video", "maxResults": videos_per_term,
                      "order": "relevance", "relevanceLanguage": "en"}
            if len(ctx.countries) == 1:
                params["regionCode"] = ctx.countries[0].upper()
            try:
                found = ctx.http.get_json(f"{API}/search", params, headers=auth)
            except HttpError as e:
                ctx.log(f"YouTube search '{term}': HTTP {e.status}")
                if e.status == 403:
                    return  # quota exhausted or key invalid
                continue
            for v in found.get("items", []):
                video_id = (v.get("id") or {}).get("videoId")
                if not video_id:
                    continue
                title = html.unescape((v.get("snippet") or {}).get("title", ""))
                try:
                    threads = ctx.http.get_json(f"{API}/commentThreads", {
                        "part": "snippet", "videoId": video_id, "maxResults": min(100, ctx.max_results),
                        "order": "relevance", "textFormat": "plainText",
                    }, headers=auth)
                except HttpError:
                    continue  # comments disabled on this video
                for t in threads.get("items", []):
                    s = t["snippet"]["topLevelComment"]["snippet"]
                    published = parse_iso(s.get("publishedAt")) or utcnow()
                    if published < ctx.since:
                        continue
                    yield RawItem(
                        source=self.id,
                        source_id=t["id"],
                        url=f"https://www.youtube.com/watch?v={video_id}&lc={t['id']}",
                        title="",
                        text=s.get("textOriginal") or s.get("textDisplay", ""),
                        created_at=published,
                        engagement={"likes": s.get("likeCount", 0),
                                    "replies": t["snippet"].get("totalReplyCount", 0)},
                        author=s.get("authorDisplayName", ""),
                        extra={"video": title, "context": f"YouTube comment on the video '{title}'"},
                    )
