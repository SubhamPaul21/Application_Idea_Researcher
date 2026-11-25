"""Bluesky post search (AppView). Uses the public endpoint; if that requires auth, logs in with
BLUESKY_HANDLE + BLUESKY_APP_PASSWORD (an app password, never the main password)."""
from __future__ import annotations

from typing import Iterator

from .. import config
from ..core.http import HttpError
from ..core.models import RawItem
from ..util import parse_iso, utcnow
from .base import FetchContext, Source

PUBLIC_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
PDS = "https://bsky.social/xrpc"


class Bluesky(Source):
    id = "bluesky"

    def _login(self, ctx: FetchContext) -> str | None:
        handle, password = config.env("BLUESKY_HANDLE"), config.env("BLUESKY_APP_PASSWORD")
        if not (handle and password):
            return None
        try:
            data = ctx.http.post_json(f"{PDS}/com.atproto.server.createSession",
                                      {"identifier": handle, "password": password})
        except HttpError as e:
            ctx.log(f"Bluesky login failed: HTTP {e.status}")
            return None
        return data.get("accessJwt")

    def fetch(self, ctx: FetchContext) -> Iterator[RawItem]:
        url, headers, tried_login = PUBLIC_URL, None, False
        if config.env("BLUESKY_APP_PASSWORD"):  # public search returns 403 as of Nov 2026: log in up front
            tried_login = True
            token = self._login(ctx)
            if token:
                url, headers = f"{PDS}/app.bsky.feed.searchPosts", {"Authorization": f"Bearer {token}"}
        for term in ctx.terms("social_keywords"):
            cursor, got = None, 0
            while got < ctx.max_results:
                params = {"q": term, "limit": min(100, ctx.max_results), "sort": "latest",
                          "since": ctx.since.strftime("%Y-%m-%dT%H:%M:%SZ")}
                if cursor:
                    params["cursor"] = cursor
                try:
                    data = ctx.http.get_json(url, params, headers=headers)
                except HttpError as e:
                    if e.status in (401, 403) and not tried_login:
                        tried_login = True
                        token = self._login(ctx)
                        if not token:
                            ctx.log("Bluesky search needs login: set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD.")
                            return
                        url = f"{PDS}/app.bsky.feed.searchPosts"
                        headers = {"Authorization": f"Bearer {token}"}
                        continue
                    ctx.log(f"Bluesky '{term}': {e}")
                    break
                posts = data.get("posts", [])
                for p in posts:
                    record = p.get("record", {})
                    handle = (p.get("author") or {}).get("handle", "")
                    rkey = p.get("uri", "").rsplit("/", 1)[-1]
                    yield RawItem(
                        source=self.id,
                        source_id=p.get("uri", rkey),
                        url=f"https://bsky.app/profile/{handle}/post/{rkey}",
                        title="",
                        text=record.get("text", ""),
                        created_at=parse_iso(record.get("createdAt")) or utcnow(),
                        engagement={"likes": p.get("likeCount", 0), "replies": p.get("replyCount", 0),
                                    "reposts": p.get("repostCount", 0)},
                        author=handle,
                        extra={"context": "Bluesky post"},
                    )
                got += len(posts)
                cursor = data.get("cursor")
                if not cursor or not posts:
                    break
