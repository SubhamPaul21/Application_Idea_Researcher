"""Orchestrates the stages. Used by both the Streamlit app and the CLI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any, Callable

from .. import config
from ..core import db
from ..core.http import Http
from ..llm.base import LLMClient
from ..sources import registry
from ..sources.base import FetchContext
from ..util import utcnow
from . import classify, cluster, metadata, prefilter

REGIONS: dict[str, list[str]] = {"Global": ["us"], "US": ["us"], "India": ["in"], "US + India": ["us", "in"]}

Log = Callable[[str], None]
Progress = Callable[[float, str], None]


@dataclass
class RunParams:
    niche_ids: list[str] = field(default_factory=list)
    custom_keywords: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=lambda: registry.implemented_ids())
    days: int = 60
    max_per_query: int = 100
    max_keywords: int = 4
    region: str = "Global"
    max_to_classify: int = 150

    def niches(self) -> list[dict[str, Any]]:
        out = [n for n in (config.niche_by_id(i) for i in self.niche_ids) if n]
        if self.custom_keywords:
            out.append(config.custom_niche(self.custom_keywords))
        return out

    def all_niche_ids(self) -> list[str]:
        return [n["id"] for n in self.niches()]

    @property
    def countries(self) -> list[str]:
        return REGIONS.get(self.region, ["us"])


def fetch_stage(p: RunParams, log: Log, progress: Progress | None = None) -> dict[str, int]:
    http = Http()
    since = utcnow() - timedelta(days=p.days)
    jobs = [(n, registry.get(s)) for n in p.niches() for s in p.source_ids]
    fetched = new = 0
    for i, (niche, src) in enumerate(jobs, 1):
        ok, why = src.available()
        if not ok:
            log(f"Skipping {src.label}: {why}")
            continue
        if not src.applies_to(niche):
            continue
        if progress:
            progress((i - 1) / len(jobs), f"Fetching {src.label} for {niche['label']}")
        ctx = FetchContext(niche=niche, since=since, max_results=p.max_per_query, max_keywords=p.max_keywords,
                           countries=p.countries, http=http, log=lambda m: log(f"  {m}"))
        items = []
        try:
            for item in src.fetch(ctx):
                items.append(item)
        except Exception as e:  # noqa: BLE001 - one broken source must not stop the run
            log(f"  {src.label} stopped early: {type(e).__name__}: {e}")
        skipped: list[str] = []
        try:
            with db.session() as conn:
                added = db.upsert_items(conn, items, niche["id"], src.meta,
                                        on_skip=lambda it, e: skipped.append(type(e).__name__))
        except Exception as e:  # noqa: BLE001 - a storage problem in one source must not stop the others
            log(f"  {src.label}: could not store items: {type(e).__name__}")
            continue
        fetched += len(items)
        new += added
        note = f", {len(skipped)} malformed skipped" if skipped else ""
        log(f"{src.label} · {niche['label']}: {len(items)} fetched, {added} new{note}")
    if progress:
        progress(1.0, "Fetch complete")
    return {"fetched": fetched, "new": new}


def purge_stage(log: Log) -> int:
    """Enforce the retention window before any work, so expired raw text is never processed or sent anywhere."""
    n = db.purge_expired(db.retention_days())
    if n:
        log(f"Retention: deleted raw text of {n} items older than {db.retention_days()} days.")
    return n


def filter_stage(log: Log) -> dict[str, int]:
    stats = prefilter.run(log)
    stats["scored"] = metadata.run()
    return stats


def classify_stage(p: RunParams, llm: LLMClient, log: Log, progress: Progress | None = None) -> dict[str, int]:
    pipe = config.settings().get("pipeline", {})
    return classify.run(llm, p.all_niche_ids(), max_items=p.max_to_classify,
                        batch_size=int(pipe.get("classify_batch_size", 8)),
                        workers=int(pipe.get("classify_workers", 4)),
                        max_chars=int(pipe.get("max_input_chars", 1500)), log=log, progress=progress)


def cluster_stage(p: RunParams, writer: LLMClient, log: Log, progress: Progress | None = None) -> str | None:
    pipe = config.settings().get("pipeline", {})
    return cluster.build(writer, p.all_niche_ids(), min_cluster_size=int(pipe.get("min_cluster_size", 2)),
                         max_ideas=int(pipe.get("max_ideas", 30)), workers=int(pipe.get("summarize_workers", 4)),
                         log=log, progress=progress)


def run_all(p: RunParams, classifier: LLMClient, writer: LLMClient, log: Log,
            progress: Progress | None = None, stages: tuple[str, ...] = ("fetch", "filter", "classify", "cluster"),
            helper: LLMClient | None = None) -> dict[str, Any]:
    """`helper` (optional, usually a smaller model) names the idea clusters; defaults to `writer`."""
    stats: dict[str, Any] = {}
    with db.session() as conn:
        run_id = db.start_run(conn, asdict(p) | {"stages": list(stages), "classifier": classifier.name,
                                                 "writer": writer.name,
                                                 "helper": (helper or writer).name})
    status = "ok"
    try:
        stats["purged"] = purge_stage(log)
        if "fetch" in stages:
            stats["fetch"] = fetch_stage(p, log, progress)
        if "filter" in stages:
            stats["filter"] = filter_stage(log)
        if "classify" in stages:
            stats["classify"] = classify_stage(p, classifier, log, progress)
        if "cluster" in stages:
            stats["batch_id"] = cluster_stage(p, helper or writer, log, progress)
    except BaseException:  # also record interrupted runs (Streamlit stop, Ctrl+C) as not ok
        status = "error"
        raise
    finally:
        with db.session() as conn:
            db.finish_run(conn, run_id, stats, status)
    return stats
