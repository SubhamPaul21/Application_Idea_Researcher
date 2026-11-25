"""Stage 3: embed problem cards, cluster with HDBSCAN, rank clusters, and have the LLM name each one."""
from __future__ import annotations

import json
import math
import secrets
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from .. import config
from ..core import db
from ..core.schemas import ClusterSummary
from ..llm.base import LLMClient, LLMError
from ..util import utcnow
from . import embed

WTP_WEIGHT = {"explicit": 1.5, "implied": 1.2, "none": 1.0}



def cluster_labels(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    n = X.shape[0]
    mcs = max(2, min_cluster_size)
    if n < max(4, mcs * 2):
        return np.full(n, -1)
    from sklearn.cluster import HDBSCAN

    if n <= 3000:
        D = np.clip(1.0 - X @ X.T, 0.0, 2.0)
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2
        return HDBSCAN(min_cluster_size=mcs, min_samples=1, metric="precomputed", copy=True).fit_predict(D)
    return HDBSCAN(min_cluster_size=mcs, min_samples=1, copy=True).fit_predict(X)


def group_stats(members: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(members)
    authors = {m.get("author_hash") or m["item_id"] for m in members}
    sources = sorted({m["source"] for m in members})
    severity = sum(m["severity"] or 1 for m in members) / n
    wtp = sum(WTP_WEIGHT.get(m["willingness_to_pay"], 1.0) for m in members) / n
    explicit = sum(m["willingness_to_pay"] == "explicit" for m in members) / n
    any_wtp = sum(m["willingness_to_pay"] in ("explicit", "implied") for m in members) / n
    solutions = sum(bool(json.loads(m.get("existing_solutions") or "[]")) for m in members) / n
    engagement = sum(m.get("meta_score") or 0 for m in members) / n
    score = (math.log2(1 + len(authors)) * severity * wtp * (1 + 0.15 * (len(sources) - 1))
             * (1 - 0.25 * solutions) * (0.75 + 0.5 * engagement))
    majority = lambda key: Counter(m[key] for m in members).most_common(1)[0][0]  # noqa: E731
    return {
        "size": n, "distinct_authors": len(authors), "sources": sources,
        "mean_severity": round(severity, 2), "wtp_score": round(wtp, 3),
        "wtp_explicit_share": round(explicit, 3), "wtp_any_share": round(any_wtp, 3),
        "solutions_share": round(solutions, 3), "score": round(score, 3),
        "app_type": majority("app_type"), "audience": majority("audience"), "build_effort": majority("build_effort"),
    }


def _summary_prompt(members: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    lines = [f"- {m['problem_statement']} (who: {m['who_has_it']}; severity {m['severity']}; via {m['source']})"
             for m in sorted(members, key=lambda m: -(m["severity"] or 0))[:15]]
    types = Counter(m["app_type"] for m in members).most_common(3)
    return (f"{stats['size']} reports from {stats['distinct_authors']} different people.\n"
            f"Suggested product forms so far: {', '.join(f'{t} ({c})' for t, c in types)}.\n\n"
            "Problem statements:\n" + "\n".join(lines))


def _fallback_summary(members: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    top = max(members, key=lambda m: m["severity"] or 0)
    words = top["problem_statement"].split()
    return {
        "title": " ".join(words[:8]) + ("…" if len(words) > 8 else ""),
        "problem": top["problem_statement"], "target_user": top["who_has_it"], "app_concept": "",
        "app_type": stats["app_type"], "audience": stats["audience"], "build_effort": stats["build_effort"],
        "search_terms": [" ".join(words[:5])],
    }


def summarize(writer: LLMClient, members: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    s = writer.complete_json(config.prompt("cluster_summary"), _summary_prompt(members, stats), ClusterSummary, max_tokens=3000,
                             task="summarize")
    data = s.model_dump()
    if data["app_type"] == "not_software":
        data["app_type"] = stats["app_type"]
    return data


def build(writer: LLMClient, niche_ids: list[str], *, min_cluster_size: int, max_ideas: int, workers: int,
          log: Callable[[str], None], progress: Callable[[float, str], None] | None = None) -> str | None:
    with db.session() as conn:
        rows = db.cards_for_clustering(conn, niche_ids)
    if not rows:
        log("Clustering: no classified, software-solvable problems yet for these niches.")
        return None

    texts = [f"{r['problem_statement']} (affects {r['who_has_it']})" for r in rows]
    X, backend_name = embed.embed([r["item_id"] for r in rows], texts)
    labels = cluster_labels(X, min_cluster_size)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[("c", int(label)) if label >= 0 else ("s", idx)].append(rows[idx])
    n_clusters = sum(1 for k in groups if k[0] == "c")
    log(f"Clustering: {len(rows)} problems embedded with {backend_name} -> {n_clusters} clusters "
        f"+ {len(groups) - n_clusters} single reports.")

    ranked = sorted(((group_stats(m), m) for m in groups.values()), key=lambda g: g[0]["score"], reverse=True)
    ranked = ranked[:max_ideas]

    summaries: list[dict[str, Any] | None] = [None] * len(ranked)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(summarize, writer, m, s): i for i, (s, m) in enumerate(ranked)}
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                summaries[i] = fut.result()
            except (LLMError, Exception) as e:  # noqa: BLE001
                log(f"  summary failed, using a fallback title: {e}")
                summaries[i] = _fallback_summary(ranked[i][1], ranked[i][0])
            if progress:
                progress(done / len(ranked), f"Named {done} of {len(ranked)} ideas")

    # Time prefix keeps batches sorted; the random suffix stops two runs in the same second from merging.
    batch_id = f"{utcnow().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    with db.session() as conn:
        for (stats, members), summary in zip(ranked, summaries):
            db.insert_cluster(conn, batch_id, niche_ids, stats, summary or _fallback_summary(members, stats),
                              [m["item_id"] for m in members])
    log(f"Ideas: saved {len(ranked)} ranked ideas (batch {batch_id}).")
    return batch_id
