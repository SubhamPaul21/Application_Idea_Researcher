"""Stage 2: LLM classifier. Sends batches of scrubbed items to any provider and stores problem cards.

Isolation between posts in one request:
- each item is sent as an escaped JSON object, so a post cannot fake the end of its block or a new item;
- items are labelled with random per-call labels instead of their real IDs, so a post cannot name another item;
- exactly one result per label is required; anything missing, duplicated or copied from the post is retried alone.
Failures are isolated: a failing batch is split in half and the good halves are kept; an item that keeps failing
is counted, and after `max_classify_attempts` it is set aside instead of being re-sent (and re-billed) every run.
"""
from __future__ import annotations

import json
import re
import secrets
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .. import config
from ..core import db
from ..core.compliance import scrub
from ..core.schemas import ClassificationBatch, ProblemCard
from ..llm.base import LLMClient, LLMError
from ..util import chunks

COPY_RUN_WORDS = 12  # a problem statement sharing this many consecutive words with the post counts as copied
_WORD = re.compile(r"[a-z0-9]+")

Failure = tuple[str, str]  # (item id, reason)


def _payload(row: dict[str, Any], max_chars: int) -> str:
    extra = json.loads(row.get("extra") or "{}")
    data = {
        "source": scrub(str(extra.get("context") or row["source"]))[:500],
        "title": scrub((row.get("title") or "")[:500]),
        "text": scrub(row.get("text") or "")[:max_chars],
    }
    # Escaped JSON: quotes and newlines are encoded, and no '<' or '>' can appear to close or open an item.
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def _labels(n: int) -> list[str]:
    out: set[str] = set()
    while len(out) < n:
        out.add("i" + secrets.token_hex(3))
    return list(out)


def build_batch(rows: list[dict[str, Any]], max_chars: int) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return the user prompt and the label -> row mapping for one request."""
    label_to_row = dict(zip(_labels(len(rows)), rows))
    blocks = "\n\n".join(f'<item id="{label}">\n{_payload(row, max_chars)}\n</item>'
                         for label, row in label_to_row.items())
    prompt = (f"Classify these {len(rows)} items. Each item is a JSON object with source, title and text. "
              f"Return an object with a \"results\" list containing exactly one entry per item id.\n\n{blocks}")
    return prompt, label_to_row


def build_prompt(rows: list[dict[str, Any]], max_chars: int) -> str:
    return build_batch(rows, max_chars)[0]


def copied_from_source(statement: str, source_text: str, run: int = COPY_RUN_WORDS) -> bool:
    words = _WORD.findall((statement or "").lower())
    src = _WORD.findall((source_text or "").lower())
    if len(words) < run or len(src) < run:
        return False
    shingles = {tuple(src[i:i + run]) for i in range(len(src) - run + 1)}
    return any(tuple(words[i:i + run]) in shingles for i in range(len(words) - run + 1))


def _classify_once(llm: LLMClient, rows: list[dict[str, Any]],
                   max_chars: int) -> tuple[list[ProblemCard], list[tuple[dict[str, Any], str]]]:
    prompt, label_to_row = build_batch(rows, max_chars)
    out = llm.complete_json(config.prompt("classify"), prompt, ClassificationBatch,
                            max_tokens=1000 + 350 * len(rows), task="classify")
    counts = Counter(c.item_id for c in out.results)
    by_label = {c.item_id: c for c in out.results if c.item_id in label_to_row and counts[c.item_id] == 1}
    cards: list[ProblemCard] = []
    retry: list[tuple[dict[str, Any], str]] = []
    for label, row in label_to_row.items():
        card = by_label.get(label)
        if card is None:
            retry.append((row, "no single result returned for this item"))
        elif card.is_problem and copied_from_source(card.problem_statement,
                                                     f"{row.get('title') or ''} {row.get('text') or ''}"):
            retry.append((row, "problem statement copied the post"))
        else:
            cards.append(card.model_copy(update={"item_id": row["id"]}))
    return cards, retry


def classify_rows(llm: LLMClient, rows: list[dict[str, Any]], max_chars: int,
                  failures: list[Failure] | None = None) -> list[ProblemCard]:
    """Classify rows; returns the cards that succeeded. When `failures` is given, items that could not be
    classified are appended to it instead of raising."""
    try:
        cards, retry = _classify_once(llm, rows, max_chars)
    except LLMError as e:
        if len(rows) == 1:
            if failures is None:
                raise
            failures.append((rows[0]["id"], str(e)))
            return []
        mid = len(rows) // 2
        return (classify_rows(llm, rows[:mid], max_chars, failures)
                + classify_rows(llm, rows[mid:], max_chars, failures))
    for row, reason in retry:
        if len(rows) > 1:
            cards += classify_rows(llm, [row], max_chars, failures)
        elif failures is not None:
            failures.append((row["id"], reason))
    return cards


def _run_batch(llm: LLMClient, rows: list[dict[str, Any]], max_chars: int) -> tuple[list[ProblemCard], list[Failure]]:
    failures: list[Failure] = []
    return classify_rows(llm, rows, max_chars, failures), failures


def _clear_text_sources() -> set[str]:
    return {sid for sid, meta in config.sources_registry().items() if meta.get("clear_text_after_classify")}


def run(llm: LLMClient, niche_ids: list[str], *, max_items: int, batch_size: int, workers: int, max_chars: int,
        log: Callable[[str], None], progress: Callable[[float, str], None] | None = None) -> dict[str, int]:
    with db.session() as conn:
        rows = db.classification_candidates(conn, niche_ids, max_items)
    if not rows:
        log("Classifier: nothing new to classify.")
        return {"classified": 0, "problems": 0, "failed_batches": 0, "failed_items": 0}
    source_of = {r["id"]: r["source"] for r in rows}
    clear_sources = _clear_text_sources()
    batches = list(chunks(rows, batch_size))
    log(f"Classifier: {len(rows)} items in {len(batches)} batches with {llm.name}.")
    classified = problems = failed_batches = failed_items = 0
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futures = {pool.submit(_run_batch, llm, b, max_chars): b for b in batches}
        for i, fut in enumerate(as_completed(futures), 1):
            batch = futures[fut]
            try:
                cards, failures = fut.result()
            except Exception as e:  # noqa: BLE001 - one bad batch must not stop the run
                cards, failures = [], [(r["id"], f"{type(e).__name__}: {e}") for r in batch]
                log(f"  batch failed unexpectedly: {type(e).__name__}")
            if failures:
                failed_batches += 1
                failed_items += len(failures)
                log(f"  {len(failures)} item(s) could not be classified: {failures[0][1][:160]}")
            with db.session() as conn:
                if cards:
                    db.save_cards(conn, cards, llm.provider, llm.model)
                    done = [c.item_id for c in cards if source_of.get(c.item_id) in clear_sources]
                    if done:
                        db.clear_text(conn, done)  # e.g. App Store reviews: keep only the paraphrased card
                if failures:
                    db.record_classify_failures(conn, failures)
            classified += len(cards)
            problems += sum(1 for c in cards if c.is_problem)
            if progress:
                progress(i / len(batches), f"Classified {classified} of {len(rows)} items")
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)  # don't keep paying for calls whose results are discarded
        raise
    pool.shutdown()
    log(f"Classifier: {classified} classified, {problems} are real problems, {failed_items} items failed "
        f"(set aside after {db.max_classify_attempts()} failed attempts).")
    return {"classified": classified, "problems": problems, "failed_batches": failed_batches,
            "failed_items": failed_items}
