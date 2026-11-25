"""US CFPB Consumer Complaint Database (free official API).

As of Nov 2026 the public API returns only structured fields (product, issue, sub-issue, company); free-text
narratives are no longer included. A single categorical complaint says little, but complaint VOLUME per issue is
a strong signal, so this adapter rolls complaints up into one item per (product, issue, sub-issue) with the
complaint count as engagement."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Iterator

from ..core.http import HttpError
from ..core.models import RawItem
from ..util import parse_iso, utcnow
from .base import FetchContext, Source

URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
PAGE = 100


class CFPB(Source):
    id = "cfpb"

    def applies_to(self, niche: dict[str, Any]) -> bool:
        return bool(niche.get("cfpb_terms"))

    def _search(self, ctx: FetchContext, term: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        while len(out) < ctx.max_results:
            params = {"search_term": term, "size": PAGE, "frm": len(out), "sort": "created_date_desc",
                      "date_received_min": ctx.since.date().isoformat(), "no_aggs": "true"}
            try:
                data = ctx.http.get_json(URL, params)
            except HttpError as e:
                ctx.log(f"CFPB '{term}': {e}")
                break
            hits = data.get("hits", {}).get("hits", []) if isinstance(data, dict) else []
            out.extend(h.get("_source", {}) for h in hits)
            if len(hits) < PAGE:
                break
        return out[: ctx.max_results]

    def fetch(self, ctx: FetchContext) -> Iterator[RawItem]:
        days = max(1, (utcnow() - ctx.since).days)
        for term in ctx.terms("cfpb_terms"):
            groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for s in self._search(ctx, term):
                product = " / ".join(x for x in (s.get("product"), s.get("sub_product")) if x)
                groups[(product, s.get("issue") or "", s.get("sub_issue") or "")].append(s)
            for (product, issue, sub_issue), rows in groups.items():
                companies = Counter(r.get("company") for r in rows if r.get("company")).most_common(3)
                latest = max((parse_iso(r.get("date_received")) or utcnow() for r in rows), default=utcnow())
                key = hashlib.sha1(f"{product}|{issue}|{sub_issue}".encode()).hexdigest()[:16]
                problem = issue + (f": {sub_issue}" if sub_issue else "")
                yield RawItem(
                    source=self.id,
                    source_id=key,
                    url=f"https://www.consumerfinance.gov/data-research/consumer-complaints/search/"
                        f"?search_term={term.replace(' ', '+')}&date_received_min={ctx.since.date().isoformat()}",
                    title=problem,
                    text=(f"{len(rows)} US consumers filed complaints in the last {days} days about "
                          f"{product or 'a financial product'}. Issue: {problem}. Most-named companies: "
                          f"{', '.join(c for c, _ in companies) or 'various'}."),
                    created_at=latest,
                    engagement={"complaints": len(rows)},
                    author="",
                    extra={"product": product, "complaints": len(rows),
                           "context": "Aggregated US consumer-finance complaints (CFPB)"},
                )
