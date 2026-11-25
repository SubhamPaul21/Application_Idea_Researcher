"""Stage 4a: "does this already exist?" Searches GitHub repos, the App Store and Show HN, then asks
the LLM to judge. Alternatives the LLM names must come from the search results (no invented tools)."""
from __future__ import annotations

from typing import Any, Callable

from .. import config
from ..core import db
from ..core.http import Http, HttpError
from ..core.schemas import ExistenceVerdict
from ..llm.base import LLMClient
from ..sources.github_issues import github_headers, github_search_interval
from ..util import iso, utcnow



def gather(http: Http, terms: list[str], country: str, log: Callable[[str], None]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(c: dict[str, Any]) -> None:
        if c["url"] and c["url"] not in seen:
            seen.add(c["url"])
            out.append(c)

    for term in terms[:3]:
        try:
            data = http.get_json("https://api.github.com/search/repositories",
                                 {"q": term, "sort": "stars", "order": "desc", "per_page": 5},
                                 headers=github_headers(), min_interval=github_search_interval())
            for r in data.get("items", []):
                add({"source": "GitHub", "name": r["full_name"], "url": r["html_url"],
                     "description": (r.get("description") or "")[:200],
                     "signal": f"{r.get('stargazers_count', 0)} stars"})
        except HttpError as e:
            log(f"GitHub repo search: {e}")
        try:
            data = http.get_json("https://itunes.apple.com/search",
                                 {"term": term, "country": country, "entity": "software", "limit": 5})
            for a in data.get("results", []):
                add({"source": "App Store", "name": a.get("trackName", ""), "url": a.get("trackViewUrl", ""),
                     "description": (a.get("description") or "")[:200].replace("\n", " "),
                     "signal": f"{a.get('averageUserRating', 0):.1f}★ from {a.get('userRatingCount', 0)} ratings, "
                               f"{a.get('formattedPrice', '')}"})
        except HttpError as e:
            log(f"App Store search: {e}")
        try:
            data = http.get_json("https://hn.algolia.com/api/v1/search",
                                 {"query": term, "tags": "show_hn", "hitsPerPage": 5})
            for h in data.get("hits", []):
                add({"source": "Show HN", "name": h.get("title", ""),
                     "url": h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
                     "description": "", "signal": f"{h.get('points') or 0} points"})
        except HttpError as e:
            log(f"Show HN search: {e}")
    return out


def check(cluster_id: int, writer: LLMClient, country: str = "us",
          log: Callable[[str], None] = print) -> dict[str, Any]:
    with db.session() as conn:
        cluster = db.get_cluster(conn, cluster_id)
    if not cluster:
        raise ValueError(f"cluster {cluster_id} not found")
    import json

    terms = json.loads(cluster.get("search_terms") or "[]") or [cluster["title"]]
    candidates = gather(Http(), terms, country, log)
    if not candidates:
        result: dict[str, Any] = {"verdict": "unsolved", "closest_alternatives": [],
                                  "opportunity": "No matching tools found on GitHub, the App Store or Show HN."}
    else:
        listing = "\n".join(f"- [{c['source']}] {c['name']} | {c['url']} | {c['signal']} | {c['description']}"
                            for c in candidates)
        prompt = (f"IDEA: {cluster['title']}\nPROBLEM: {cluster['problem']}\nTARGET USER: {cluster['target_user']}\n"
                  f"CONCEPT: {cluster['app_concept']}\n\nSEARCH RESULTS:\n{listing}")
        verdict = writer.complete_json(config.prompt("existence"), prompt, ExistenceVerdict, max_tokens=4000, task="existence")
        result = verdict.model_dump()
        valid = {c["url"] for c in candidates}
        result["closest_alternatives"] = [a for a in result["closest_alternatives"] if a["url"] in valid]
    result["candidates"] = candidates
    result["checked_at"] = iso(utcnow())
    with db.session() as conn:
        db.set_existence(conn, cluster_id, result)
    return result
