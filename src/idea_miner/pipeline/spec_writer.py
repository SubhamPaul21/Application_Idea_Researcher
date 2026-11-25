"""Stage 4b: PRD spec for one idea. The LLM writes the product sections; evidence, links and
attribution are inserted by code so they can't be hallucinated."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .. import config
from ..core import db
from ..llm.base import LLMClient
from ..util import iso, slugify, utcnow


SOURCE_LABELS = {k: v.get("label", k) for k, v in config.sources_registry().items()}


def _prompt(c: dict[str, Any], members: list[dict[str, Any]], existence: dict[str, Any] | None) -> str:
    statements = "\n".join(f"- {m['problem_statement']} (who: {m['who_has_it']}; severity {m['severity']}; "
                           f"pays: {m['willingness_to_pay']}; frequency: {m['frequency']})" for m in members[:25])
    mentioned = Counter(s for m in members for s in json.loads(m.get("existing_solutions") or "[]"))
    parts = [
        f"IDEA: {c['title']}", f"PROBLEM: {c['problem']}", f"TARGET USER: {c['target_user']}",
        f"CONCEPT: {c['app_concept']}", f"PREFERRED FORM: {c['app_type']} for {c['audience']}, "
        f"build effort {c['build_effort']}",
        f"EVIDENCE: {c['size']} reports from {c['distinct_authors']} people; mean severity {c['mean_severity']}/10",
        f"PROBLEM STATEMENTS:\n{statements}",
    ]
    if mentioned:
        parts.append("TOOLS PEOPLE MENTIONED: " + ", ".join(f"{k} ({v})" for k, v in mentioned.most_common(10)))
    if existence:
        alts = "\n".join(f"- {a['name']} ({a['url']}): {a['gap']}" for a in existence.get("closest_alternatives", []))
        parts.append(f"EXISTENCE CHECK: verdict {existence.get('verdict')}. {existence.get('opportunity', '')}\n{alts}")
    return "\n\n".join(parts)


def _evidence(c: dict[str, Any], members: list[dict[str, Any]]) -> str:
    sources = Counter(SOURCE_LABELS.get(m["source"], m["source"]) for m in members)
    lines = [
        "## Evidence",
        "",
        "| Signal | Value |", "|---|---|",
        f"| Reports | {c['size']} |",
        f"| Distinct people | {c['distinct_authors']} |",
        f"| Mean severity | {c['mean_severity']}/10 |",
        f"| Willing to pay (explicit / any) | {c['wtp_explicit_share']:.0%} / {c['wtp_any_share']:.0%} |",
        f"| Sources | {', '.join(f'{k} ({v})' for k, v in sources.most_common())} |",
        f"| Opportunity score | {c['score']:.1f} |",
        "",
        "Paraphrased reports (links go to the original public posts):",
        "",
    ]
    for m in members[:40]:
        lines.append(f"- {m['problem_statement']} [{SOURCE_LABELS.get(m['source'], m['source'])}]({m['url']})")
    return "\n".join(lines)


def _attribution(members: list[dict[str, Any]]) -> str:
    notes = ["## Sources and attribution", "",
             "Reports are paraphrased; no original text is reproduced. Each link credits the original author."]
    if any(m["source"] == "stackexchange" for m in members):
        notes.append("Stack Exchange content is licensed CC BY-SA 4.0; see linked questions for authorship.")
    return "\n".join(notes)


def generate(cluster_id: int, writer: LLMClient) -> int:
    with db.session() as conn:
        c = db.get_cluster(conn, cluster_id)
        members = db.cluster_members(conn, cluster_id)
    if not c:
        raise ValueError(f"cluster {cluster_id} not found")
    existence = json.loads(c["existence"]) if c.get("existence") else None
    cap = int(config.settings().get("pipeline", {}).get("spec_max_tokens", 5000))
    body = writer.complete_text(config.prompt("spec"), _prompt(c, members, existence), max_tokens=cap, task="spec")
    header = f"# {c['title']}\n\n> {c['app_concept']}\n\n_App type: {c['app_type']} · Audience: {c['audience']} · " \
             f"Build effort: {c['build_effort']} · Generated {iso(utcnow())[:10]} by {writer.name}_"
    markdown = "\n\n".join([header, body, _evidence(c, members), _attribution(members)]) + "\n"
    with db.session() as conn:
        return db.insert_spec(conn, cluster_id, c["title"], markdown, writer.provider, writer.model)


def approve(spec_id: int) -> Path:
    """Write the approved spec to specs/ as Markdown plus a JSON contract for build agents."""
    with db.session() as conn:
        spec = db.get_spec(conn, spec_id)
        if not spec:
            raise ValueError(f"spec {spec_id} not found")
        c = db.get_cluster(conn, spec["cluster_id"]) or {}
        members = db.cluster_members(conn, spec["cluster_id"])
    stem = f"{utcnow().strftime('%Y%m%d')}-{slugify(spec['title'])}"
    md_path = config.specs_dir() / f"{stem}.md"
    md_path.write_text(spec["markdown"], encoding="utf-8")
    contract = {
        "title": spec["title"], "problem": c.get("problem"), "target_user": c.get("target_user"),
        "app_concept": c.get("app_concept"), "app_type": c.get("app_type"), "audience": c.get("audience"),
        "build_effort": c.get("build_effort"), "opportunity_score": c.get("score"),
        "existence": json.loads(c["existence"]).get("verdict") if c.get("existence") else None,
        "evidence": [{"source": m["source"], "url": m["url"], "problem_statement": m["problem_statement"],
                      "severity": m["severity"]} for m in members],
        "spec_markdown_file": md_path.name, "generated_by": f"{spec['provider']}/{spec['model']}",
        "approved_at": iso(utcnow()),
    }
    (config.specs_dir() / f"{stem}.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    with db.session() as conn:
        db.update_spec(conn, spec_id, status="approved", file_path=str(md_path))
    return md_path


def reject(spec_id: int) -> None:
    with db.session() as conn:
        db.update_spec(conn, spec_id, status="rejected")
