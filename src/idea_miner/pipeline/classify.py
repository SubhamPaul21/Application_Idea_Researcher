"""Stage 2: LLM classifier. Sends batches of scrubbed items to any provider and stores problem cards."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .. import config
from ..core import db
from ..core.compliance import scrub
from ..core.schemas import ClassificationBatch, ProblemCard
from ..llm.base import LLMClient, LLMError, LLMTruncated
from ..util import chunks



def _item_block(row: dict[str, Any], max_chars: int) -> str:
    extra = json.loads(row.get("extra") or "{}")
    context = extra.get("context") or row["source"]
    title = scrub(row.get("title") or "")
    text = scrub(row.get("text") or "")[:max_chars]
    return f'<item id="{row["id"]}">\nSOURCE: {context}\nTITLE: {title}\nTEXT: {text}\n</item>'


def build_prompt(rows: list[dict[str, Any]], max_chars: int) -> str:
    blocks = "\n\n".join(_item_block(r, max_chars) for r in rows)
    return (f"Classify these {len(rows)} items. Return an object with a \"results\" list containing exactly "
            f"one entry per item id.\n\n{blocks}")


def classify_rows(llm: LLMClient, rows: list[dict[str, Any]], max_chars: int) -> list[ProblemCard]:
    try:
        out = llm.complete_json(config.prompt("classify"), build_prompt(rows, max_chars), ClassificationBatch,
                                max_tokens=1000 + 350 * len(rows), task="classify")
    except LLMTruncated:
        if len(rows) == 1:
            raise
        mid = len(rows) // 2
        return classify_rows(llm, rows[:mid], max_chars) + classify_rows(llm, rows[mid:], max_chars)
    wanted = {r["id"] for r in rows}
    cards: list[ProblemCard] = []
    for card in out.results:
        if card.item_id in wanted:
            cards.append(card)
            wanted.discard(card.item_id)
    return cards


def run(llm: LLMClient, niche_ids: list[str], *, max_items: int, batch_size: int, workers: int, max_chars: int,
        log: Callable[[str], None], progress: Callable[[float, str], None] | None = None) -> dict[str, int]:
    with db.session() as conn:
        rows = db.classification_candidates(conn, niche_ids, max_items)
    if not rows:
        log("Classifier: nothing new to classify.")
        return {"classified": 0, "problems": 0, "failed_batches": 0}
    batches = list(chunks(rows, batch_size))
    log(f"Classifier: {len(rows)} items in {len(batches)} batches with {llm.name}.")
    classified = problems = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(classify_rows, llm, b, max_chars): b for b in batches}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                cards = fut.result()
            except LLMError as e:
                failed += 1
                cards = []
                log(f"  batch failed: {e}")
            except Exception as e:  # noqa: BLE001 - one bad batch must not stop the run
                failed += 1
                cards = []
                log(f"  batch failed unexpectedly: {type(e).__name__}: {e}")
            if cards:
                with db.session() as conn:
                    db.save_cards(conn, cards, llm.provider, llm.model)
            classified += len(cards)
            problems += sum(1 for c in cards if c.is_problem)
            if progress:
                progress(i / len(batches), f"Classified {classified} of {len(rows)} items")
    log(f"Classifier: {classified} classified, {problems} are real problems, {failed} batches failed.")
    return {"classified": classified, "problems": problems, "failed_batches": failed}
