"""Stage 1: metadata score in [0, 1] = engagement percentile within its source + recency + phrase hits.
Decides which items are worth spending LLM tokens on."""
from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Any

from .. import config
from ..core import db
from ..util import parse_iso, utcnow

HALF_LIFE_DAYS = 30.0


def compute(rows: list[dict[str, Any]], weights: dict[str, dict[str, float]]) -> dict[str, float]:
    now = utcnow()
    per_source: dict[str, list[tuple[str, float, float, int]]] = defaultdict(list)
    for r in rows:
        eng = json.loads(r.get("engagement") or "{}")
        w = weights.get(r["source"], {})
        value = sum(w.get(k, 1.0) * float(v or 0) for k, v in eng.items())
        created = parse_iso(r.get("created_at")) or now
        age_days = max(0.0, (now - created).total_seconds() / 86400)
        hits = len([h for h in json.loads(r.get("prefilter_hits") or "[]") if h not in ("duplicate",)])
        per_source[r["source"]].append((r["id"], math.log1p(max(value, 0.0)), age_days, hits))

    scores: dict[str, float] = {}
    for entries in per_source.values():
        values = sorted(e[1] for e in entries)
        n = len(values)
        for item_id, v, age, hits in entries:
            if n == 1 or values[0] == values[-1]:
                pct = 0.5
            else:
                pct = ((bisect_left(values, v) + bisect_right(values, v) - 1) / 2) / (n - 1)
            recency = 0.5 ** (age / HALF_LIFE_DAYS)
            scores[item_id] = round(0.5 * pct + 0.3 * recency + 0.2 * min(hits, 3) / 3, 4)
    return scores


def run() -> int:
    weights = {k: v.get("engagement_weights") or {} for k, v in config.sources_registry().items()}
    with db.session() as conn:
        scores = compute(db.items_for_scoring(conn), weights)
        db.set_meta_scores(conn, scores)
    return len(scores)
