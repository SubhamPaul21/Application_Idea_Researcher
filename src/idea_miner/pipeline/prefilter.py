"""Stage 0: free regex prefilter + exact-duplicate removal."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .. import config
from ..core import db

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


class Prefilter:
    def __init__(self, rules: dict[str, Any] | None = None) -> None:
        rules = rules if rules is not None else config.prefilter_rules()
        self.patterns = [(p["name"], re.compile(p["regex"], re.I)) for p in rules.get("patterns", [])]
        self.negatives = [re.compile(r, re.I) for r in rules.get("negatives", [])]
        self.min_chars = int(rules.get("min_chars", 25))

    def hits(self, text: str) -> list[str]:
        t = text.translate(_QUOTES)
        return [name for name, rx in self.patterns if rx.search(t)]

    def evaluate(self, text: str, bypass: bool = False) -> tuple[bool, list[str]]:
        t = (text or "").translate(_QUOTES)
        if len(t.strip()) < self.min_chars:
            return False, ["too_short"]
        if any(rx.search(t) for rx in self.negatives):
            return False, ["negative"]
        found = self.hits(t)
        return bool(found) or bypass, found


def run(log: Callable[[str], None]) -> dict[str, int]:
    pf = Prefilter()
    registry = config.sources_registry()
    with db.session() as conn:
        rows = db.items_needing_prefilter(conn)
        seen = db.known_hashes(conn)
        updates: list[tuple[str, int, str]] = []
        passed = dupes = 0
        for r in rows:
            h = r.get("text_hash") or ""
            if h and h in seen:
                updates.append((json.dumps(["duplicate"]), 0, r["id"]))
                dupes += 1
                continue
            if h:
                seen.add(h)
            bypass = bool(registry.get(r["source"], {}).get("prefilter_bypass", False))
            ok, found = pf.evaluate(f"{r.get('title') or ''}\n{r.get('text') or ''}", bypass)
            updates.append((json.dumps(found), int(ok), r["id"]))
            passed += int(ok)
        db.set_prefilter(conn, updates)
    log(f"Prefilter: {passed} of {len(rows)} new items look like problems ({dupes} duplicates dropped).")
    return {"checked": len(rows), "passed": passed, "duplicates": dupes}
