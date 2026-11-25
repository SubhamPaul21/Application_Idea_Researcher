"""Stage 4a: "does this already exist?" Searches GitHub repos, the App Store and Show HN, then asks
the LLM to judge. Alternatives the LLM names must come from the search results (no invented tools).

A failed search is never treated as "no results": if every search fails nothing is saved, and if some fail the
stored result is marked incomplete."""
from __future__ import annotations

import json
from typing import Any, Callable

from .. import config
from ..core import db
from ..core.compliance import scrub
from ..core.http import Http, HttpError
from ..core.safe_text import cap, one_line
from ..core.schemas import ExistenceVerdict
from ..llm.base import LLMClient
from ..sources.github_issues import github_headers, github_search_interval, public_terms
from ..util import iso, utcnow


class ExistenceCheckFailed(RuntimeError):
    """Every search failed, so no verdict can be given."""


def gather(http: Http, terms: list[str], country: str,
           log: Callable[[str], None]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (candidates, failed searches)."""
    out: list[dict[str, Any]] = []
    failed: list[str] = []
    seen: set[str] = set()

    def add(c: dict[str, Any]) -> None:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            out.append(c)

    def fail(name: str, term: str, e: HttpError) -> None:
        failed.append(f"{name} '{cap(term, 40)}': HTTP {e.status or 'connection error'}")
        log(f"{name} search failed: {e}")

    for term in terms[:3]:
        gh_term = public_terms(term)
        if gh_term:
            try:
                data = http.get_json("https://api.github.com/search/repositories",
                                     {"q": f"{gh_term} is:public", "sort": "stars", "order": "desc", "per_page": 5},
                                     headers=github_headers(), min_interval=github_search_interval())
                for r in data.get("items", []):
                    if r.get("private"):
                        continue
                    add({"source": "GitHub", "name": r["full_name"], "url": r["html_url"],
                         "description": (r.get("description") or "")[:200],
                         "signal": f"{r.get('stargazers_count', 0)} stars"})
            except HttpError as e:
                fail("GitHub", term, e)
        try:
            data = http.get_json("https://itunes.apple.com/search",
                                 {"term": term, "country": country, "entity": "software", "limit": 5})
            for a in data.get("results", []):
                add({"source": "App Store", "name": a.get("trackName", ""), "url": a.get("trackViewUrl", ""),
                     "description": (a.get("description") or "")[:200].replace("\n", " "),
                     "signal": f"{a.get('averageUserRating', 0):.1f}★ from {a.get('userRatingCount', 0)} ratings, "
                               f"{a.get('formattedPrice', '')}"})
        except HttpError as e:
            fail("App Store", term, e)
        try:
            data = http.get_json("https://hn.algolia.com/api/v1/search",
                                 {"query": term, "tags": "show_hn", "hitsPerPage": 5})
            for h in data.get("hits", []):
                add({"source": "Show HN", "name": h.get("title", ""),
                     "url": h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
                     "description": "", "signal": f"{h.get('points') or 0} points"})
        except HttpError as e:
            fail("Show HN", term, e)
    return out, failed


def check(cluster_id: int, writer: LLMClient, country: str = "us",
          log: Callable[[str], None] = print) -> dict[str, Any]:
    with db.session() as conn:
        cluster = db.get_cluster(conn, cluster_id)
    if not cluster:
        raise ValueError(f"cluster {cluster_id} not found")

    terms = json.loads(cluster.get("search_terms") or "[]") or [cluster["title"]]
    candidates, failed = gather(Http(), terms, country, log)
    if not candidates and failed:
        raise ExistenceCheckFailed("All searches failed, so the previous result was kept. "
                                   f"Failures: {'; '.join(failed[:3])}")
    if not candidates:
        result: dict[str, Any] = {"verdict": "unsolved", "closest_alternatives": [],
                                  "opportunity": "No matching tools found on GitHub, the App Store or Show HN."}
    else:
        listing = "\n".join(f"- [{c['source']}] {one_line(c['name'])} | {c['url']} | {c['signal']} | "
                            f"{one_line(c['description'])}" for c in candidates)
        note = ("\n\nNOTE: some searches failed, so these results are incomplete. Prefer partially_solved over "
                "unsolved." if failed else "")
        prompt = (f"IDEA: {cluster['title']}\nPROBLEM: {cluster['problem']}\nTARGET USER: {cluster['target_user']}\n"
                  f"CONCEPT: {cluster['app_concept']}\n\nSEARCH RESULTS:\n{listing}{note}")
        verdict = writer.complete_json(config.prompt("existence"), prompt, ExistenceVerdict, max_tokens=4000,
                                       task="existence")
        result = verdict.model_dump()
        by_url = {c["url"]: c for c in candidates}
        # Alternatives must be real search results; their names come from the result, not from the model.
        result["closest_alternatives"] = [
            {"name": by_url[a["url"]]["name"], "url": a["url"], "gap": cap(scrub(one_line(a["gap"])), 300)}
            for a in result["closest_alternatives"] if a["url"] in by_url
        ]
        result["opportunity"] = cap(scrub(one_line(result["opportunity"])), 600)
    result["incomplete"] = bool(failed)
    result["failed_searches"] = failed
    result["candidates"] = candidates
    result["checked_at"] = iso(utcnow())
    with db.session() as conn:
        db.set_existence(conn, cluster_id, result)
    return result
