"""Stage 4b: PRD spec for one idea. The LLM writes the product sections; evidence, links and
attribution are inserted by code so they can't be hallucinated.

Approval integrity:
- the JSON build contract is built when the draft is written and stored with it, so the operator reviews exactly
  what will be written (Markdown, raw source and contract are all shown on the Specs tab);
- approve() only accepts a draft whose content hash matches what the operator was shown;
- approved files are named with the spec ID and never overwrite an existing file;
- rejecting an approved spec moves its files to specs/revoked/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .. import config
from ..core import db
from ..core.safe_text import clean_markdown, md_link, one_line
from ..llm.base import LLMClient
from ..util import iso, slugify, utcnow

SOURCE_LABELS = {k: v.get("label", k) for k, v in config.sources_registry().items()}
EVIDENCE_IN_MARKDOWN = 40


class SpecError(RuntimeError):
    """An approval or rejection that must not go ahead; the message is shown to the operator."""


def _prompt(c: dict[str, Any], members: list[dict[str, Any]], existence: dict[str, Any] | None) -> str:
    statements = "\n".join(f"- {one_line(m['problem_statement'])} (who: {one_line(m['who_has_it'])}; "
                           f"severity {m['severity']}; pays: {m['willingness_to_pay']}; frequency: {m['frequency']})"
                           for m in members[:25])
    mentioned = Counter(s for m in members for s in json.loads(m.get("existing_solutions") or "[]"))
    parts = [
        f"IDEA: {one_line(c['title'])}", f"PROBLEM: {one_line(c['problem'])}",
        f"TARGET USER: {one_line(c['target_user'])}", f"CONCEPT: {one_line(c['app_concept'])}",
        f"PREFERRED FORM: {c['app_type']} for {c['audience']}, build effort {c['build_effort']}",
        f"EVIDENCE: {c['size']} reports from {c['distinct_authors']} people; mean severity {c['mean_severity']}/10",
        f"PROBLEM STATEMENTS:\n{statements}",
    ]
    if mentioned:
        parts.append("TOOLS PEOPLE MENTIONED: " + ", ".join(f"{one_line(k)} ({v})" for k, v in mentioned.most_common(10)))
    if existence:
        alts = "\n".join(f"- {one_line(a['name'])} ({a['url']}): {one_line(a['gap'])}"
                         for a in existence.get("closest_alternatives", []))
        note = " (some searches failed, so this check is incomplete)" if existence.get("incomplete") else ""
        parts.append(f"EXISTENCE CHECK{note}: verdict {existence.get('verdict')}. "
                     f"{one_line(existence.get('opportunity', ''))}\n{alts}")
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
    for m in members[:EVIDENCE_IN_MARKDOWN]:
        lines.append(f"- {one_line(m['problem_statement'])} "
                     f"{md_link(SOURCE_LABELS.get(m['source'], m['source']), m['url'])}")
    if len(members) > EVIDENCE_IN_MARKDOWN:
        lines.append(f"- …and {len(members) - EVIDENCE_IN_MARKDOWN} more (all listed in the JSON build contract)")
    return "\n".join(lines)


def _attribution(members: list[dict[str, Any]]) -> str:
    notes = ["## Sources and attribution", "",
             "Reports are paraphrased; no original text is reproduced. Each link credits the original author."]
    if any(m["source"] == "stackexchange" for m in members):
        notes.append("Stack Exchange content is licensed CC BY-SA 4.0; see linked questions for authorship.")
    return "\n".join(notes)


def _contract(title: str, c: dict[str, Any], members: list[dict[str, Any]], provider: str, model: str) -> str:
    existence = json.loads(c["existence"]) if c.get("existence") else None
    return json.dumps({
        "title": title, "problem": c.get("problem"), "target_user": c.get("target_user"),
        "app_concept": c.get("app_concept"), "app_type": c.get("app_type"), "audience": c.get("audience"),
        "build_effort": c.get("build_effort"), "opportunity_score": c.get("score"),
        "existence": existence.get("verdict") if existence else None,
        "existence_incomplete": bool(existence.get("incomplete")) if existence else None,
        "evidence": [{"source": m["source"], "url": m["url"], "problem_statement": m["problem_statement"],
                      "severity": m["severity"]} for m in members],
        "generated_by": f"{provider}/{model}",
    }, indent=2, ensure_ascii=False)


def content_hash(spec: dict[str, Any]) -> str:
    """Fingerprint of everything an approval writes: the Markdown and the build contract."""
    blob = f"{spec.get('markdown') or ''}\x00{spec.get('contract') or ''}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def ensure_contract(spec_id: int) -> dict[str, Any]:
    """Older drafts were stored without a contract snapshot: build and store it once, before it is displayed."""
    with db.session() as conn:
        spec = db.get_spec(conn, spec_id)
        if spec and not spec.get("contract") and spec["status"] == "draft":
            c = db.get_cluster(conn, spec["cluster_id"]) or {}
            members = db.cluster_members(conn, spec["cluster_id"])
            contract = _contract(spec["title"], c, members, spec["provider"], spec["model"])
            db.update_spec(conn, spec_id, contract=contract)
            spec["contract"] = contract
    if not spec:
        raise SpecError(f"Spec {spec_id} not found.")
    return spec


def generate(cluster_id: int, writer: LLMClient) -> int:
    with db.session() as conn:
        c = db.get_cluster(conn, cluster_id)
        members = db.cluster_members(conn, cluster_id)
    if not c:
        raise ValueError(f"cluster {cluster_id} not found")
    existence = json.loads(c["existence"]) if c.get("existence") else None
    cap = int(config.settings().get("pipeline", {}).get("spec_max_tokens", 5000))
    body = writer.complete_text(config.prompt("spec"), _prompt(c, members, existence), max_tokens=cap, task="spec")
    if not body.strip():
        raise ValueError("The model returned an empty spec; nothing was saved.")
    title = one_line(c["title"])
    header = (f"# {title}\n\n> {one_line(c['app_concept'])}\n\n_App type: {c['app_type']} · Audience: "
              f"{c['audience']} · Build effort: {c['build_effort']} · Generated {iso(utcnow())[:10]} by {writer.name}_")
    # clean_markdown drops images, hidden constructs and non-http(s) links, so the saved file and the
    # preview contain the same visible content.
    markdown = clean_markdown("\n\n".join([header, body, _evidence(c, members), _attribution(members)])) + "\n"
    contract = _contract(title, c, members, writer.provider, writer.model)
    with db.session() as conn:
        return db.insert_spec(conn, cluster_id, title, markdown, writer.provider, writer.model, contract)


def _write_new(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8") as f:  # never replace an existing file
        f.write(text)


def approve(spec_id: int, expected_hash: str | None = None) -> Path:
    """Write the reviewed draft to specs/ as Markdown plus its JSON build contract.

    expected_hash: content_hash() of the spec as shown to the operator. If given and the stored spec differs,
    nothing is written."""
    spec = ensure_contract(spec_id)
    if spec["status"] != "draft":
        raise SpecError(f"Spec #{spec_id} is {spec['status']}; only drafts can be approved.")
    if expected_hash is not None and content_hash(spec) != expected_hash:
        raise SpecError(f"Spec #{spec_id} changed after it was displayed. Review it again before approving.")
    stem = f"{utcnow().strftime('%Y%m%d')}-{spec_id}-{slugify(spec['title'])}"
    md_path = config.specs_dir() / f"{stem}.md"
    json_path = md_path.with_suffix(".json")
    contract = json.loads(spec["contract"])
    contract.update({"spec_markdown_file": md_path.name, "approved_at": iso(utcnow())})
    try:
        _write_new(md_path, spec["markdown"])
        _write_new(json_path, json.dumps(contract, indent=2, ensure_ascii=False))
    except FileExistsError as e:
        raise SpecError(f"{Path(e.filename).name} already exists in specs/; nothing was overwritten.") from e
    except Exception:
        md_path.unlink(missing_ok=True)
        raise
    with db.session() as conn:
        db.update_spec(conn, spec_id, status="approved", file_path=str(md_path))
    return md_path


def reject(spec_id: int) -> Path | None:
    """Mark a spec rejected. If it was approved, move its files out of specs/ into specs/revoked/ so build agents
    reading the folder no longer see it. Returns the new location of the Markdown file, if any."""
    with db.session() as conn:
        spec = db.get_spec(conn, spec_id)
    if not spec:
        raise SpecError(f"Spec {spec_id} not found.")
    moved: Path | None = None
    if spec["status"] == "approved" and spec.get("file_path"):
        revoked = config.specs_dir() / "revoked"
        revoked.mkdir(exist_ok=True)
        md = Path(spec["file_path"])
        for src in (md, md.with_suffix(".json")):
            if src.exists():
                dest = revoked / src.name
                if dest.exists():
                    dest = revoked / f"{src.stem}-{utcnow().strftime('%H%M%S')}{src.suffix}"
                shutil.move(str(src), str(dest))
                if src.suffix == ".md":
                    moved = dest
    with db.session() as conn:
        db.update_spec(conn, spec_id, status="rejected",
                       file_path=str(moved) if moved else (None if spec["status"] == "approved" else spec.get("file_path")))
    return moved
