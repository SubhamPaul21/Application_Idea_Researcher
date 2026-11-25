"""SQLite storage. One file at data/idea_miner.db. Every function takes an open connection
except the few that manage their own session (record_usage, purge_expired)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from .. import config
from ..util import chunks, iso, utcnow
from .compliance import author_hash, scrub, text_hash
from .models import RawItem
from .safe_text import cap, one_line

TITLE_MAX = 500
TEXT_MAX = 8000
EXTRA_STRING_MAX = 2000

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    url TEXT,
    title TEXT,
    text TEXT,
    text_hash TEXT,
    created_at TEXT,
    fetched_at TEXT,
    engagement TEXT,
    author_hash TEXT,
    extra TEXT,
    license TEXT,
    tos_status TEXT,
    prefilter_hits TEXT,
    prefilter_pass INTEGER,
    meta_score REAL DEFAULT 0,
    text_purged INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_items_hash ON items(text_hash);
CREATE INDEX IF NOT EXISTS idx_items_pass ON items(prefilter_pass);
CREATE TABLE IF NOT EXISTS item_niches (
    item_id TEXT NOT NULL,
    niche TEXT NOT NULL,
    PRIMARY KEY (item_id, niche)
);
CREATE INDEX IF NOT EXISTS idx_item_niches_niche ON item_niches(niche);
CREATE TABLE IF NOT EXISTS cards (
    item_id TEXT PRIMARY KEY,
    is_problem INTEGER,
    problem_statement TEXT,
    who_has_it TEXT,
    audience TEXT,
    frequency TEXT,
    severity INTEGER,
    willingness_to_pay TEXT,
    existing_solutions TEXT,
    software_solvable INTEGER,
    app_type TEXT,
    build_effort TEXT,
    provider TEXT,
    model TEXT,
    classified_at TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
    item_id TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (item_id, model)
);
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    niches TEXT,
    title TEXT,
    problem TEXT,
    target_user TEXT,
    app_concept TEXT,
    app_type TEXT,
    audience TEXT,
    build_effort TEXT,
    search_terms TEXT,
    size INTEGER,
    distinct_authors INTEGER,
    sources TEXT,
    mean_severity REAL,
    wtp_score REAL,
    wtp_explicit_share REAL,
    wtp_any_share REAL,
    solutions_share REAL,
    score REAL,
    existence TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_clusters_batch ON clusters(batch_id);
CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    PRIMARY KEY (cluster_id, item_id)
);
CREATE TABLE IF NOT EXISTS specs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER,
    title TEXT,
    markdown TEXT,
    status TEXT,
    provider TEXT,
    model TEXT,
    created_at TEXT,
    file_path TEXT
);
CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    role TEXT,
    task TEXT,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    finished_at TEXT,
    params TEXT,
    stats TEXT,
    status TEXT
);
"""


def db_path() -> Path:
    return config.data_dir() / "idea_miner.db"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA secure_delete=ON")  # deleted text is overwritten, not left in free pages
    return conn


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init() -> None:
    with session() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive schema changes plus one-time data clean-ups. Safe to run on every start."""
    if "contract" not in _columns(conn, "specs"):
        conn.execute("ALTER TABLE specs ADD COLUMN contract TEXT")
    item_cols = _columns(conn, "items")
    if "classify_attempts" not in item_cols:
        conn.execute("ALTER TABLE items ADD COLUMN classify_attempts INTEGER DEFAULT 0")
    if "classify_error" not in item_cols:
        conn.execute("ALTER TABLE items ADD COLUMN classify_error TEXT")
    conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)")
    done = {r[0] for r in conn.execute("SELECT key FROM schema_meta").fetchall()}
    if "privacy_v2" not in done:
        _privacy_v2(conn)
        conn.execute("INSERT INTO schema_meta (key, value) VALUES ('privacy_v2', ?)", (iso(utcnow()),))


def _privacy_v2(conn: sqlite3.Connection) -> None:
    """Apply the 2026-11 scrubbing rules to rows stored before them."""
    from ..sources import bluesky

    for r in _rows(conn.execute("SELECT id, title, text, extra, text_purged FROM items")):
        if r["text_purged"]:
            conn.execute("UPDATE items SET extra=NULL WHERE id=?", (r["id"],))
            continue
        title = scrub((r["title"] or "")[:TITLE_MAX])
        text = scrub(r["text"] or "")[:TEXT_MAX]
        extra = json.dumps(scrub_tree(json.loads(r["extra"] or "{}")), default=str)
        conn.execute("UPDATE items SET title=?, text=?, extra=?, text_hash=? WHERE id=?",
                     (title, text, extra, text_hash(f"{title} {text}"), r["id"]))
    for r in _rows(conn.execute("SELECT * FROM cards")):
        fields = clean_card_fields(r["problem_statement"], r["who_has_it"], json.loads(r["existing_solutions"] or "[]"))
        conn.execute("UPDATE cards SET problem_statement=?, who_has_it=?, existing_solutions=? WHERE item_id=?",
                     (*fields[:2], json.dumps(fields[2]), r["item_id"]))
    # Bluesky rows used the account DID in the ID and the handle in the link.
    for r in _rows(conn.execute("SELECT id, source_id, url FROM items WHERE source='bluesky' AND source_id LIKE 'at://%'")):
        new_sid = bluesky.post_key(r["source_id"])
        new_id = f"bluesky:{new_sid}"
        if conn.execute("SELECT 1 FROM items WHERE id=?", (new_id,)).fetchone():
            continue
        conn.execute("UPDATE items SET id=?, source_id=?, url=? WHERE id=?",
                     (new_id, new_sid, bluesky.post_url(r["source_id"], r["url"]), r["id"]))
        for table in ("item_niches", "cards", "cluster_members", "embeddings"):
            conn.execute(f"UPDATE {table} SET item_id=? WHERE item_id=?", (new_id, r["id"]))


def scrub_tree(value: Any) -> Any:
    """Scrub every string inside a nested structure (the per-source `extra` details)."""
    if isinstance(value, dict):
        return {k: scrub_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_tree(v) for v in value]
    if isinstance(value, str):
        return scrub(value[:EXTRA_STRING_MAX])
    return value


def clean_card_fields(problem_statement: Any, who_has_it: Any,
                      existing_solutions: list[Any]) -> tuple[str, str, list[str]]:
    """Model-written card fields: one line, personal data removed, bounded length."""
    return (
        cap(scrub(one_line(problem_statement)), 400),
        cap(scrub(one_line(who_has_it)), 100),
        [cap(scrub(one_line(s)), 60) for s in list(existing_solutions or [])[:10] if str(s or "").strip()],
    )


def _rows(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


def _ph(n: int) -> str:
    return ",".join("?" * n)


# ---------------------------------------------------------------- items

def upsert_items(conn: sqlite3.Connection, items: Iterable[RawItem], niche_id: str, meta: dict[str, Any],
                 on_skip: Any = None) -> int:
    """Store fetched items. One malformed item is skipped (reported through on_skip) instead of failing the batch."""
    now = iso(utcnow())
    new = 0
    for it in items:
        try:
            # Cut to length before scrubbing so very long remote text cannot make the regexes slow.
            title = scrub(it.title[:TITLE_MAX])
            text = scrub(it.text[: TEXT_MAX + 200])[:TEXT_MAX]
            row = (
                it.id, it.source, it.source_id, it.url, title, text, text_hash(f"{title} {text}"),
                iso(it.created_at), now, json.dumps(it.engagement), author_hash(it.source, it.author),
                json.dumps(scrub_tree(it.extra), default=str), meta.get("license", ""), meta.get("tos_status", ""),
            )
        except Exception as e:  # noqa: BLE001 - one malformed item must not fail the whole batch
            if on_skip:
                on_skip(it, e)
            continue
        cur = conn.execute(
            "INSERT OR IGNORE INTO items (id, source, source_id, url, title, text, text_hash, created_at, fetched_at,"
            " engagement, author_hash, extra, license, tos_status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        if cur.rowcount:
            new += 1
        else:
            conn.execute("UPDATE items SET engagement=? WHERE id=?", (json.dumps(it.engagement), it.id))
        conn.execute("INSERT OR IGNORE INTO item_niches (item_id, niche) VALUES (?,?)", (it.id, niche_id))
    return new


def items_needing_prefilter(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(conn.execute(
        "SELECT id, source, title, text, text_hash FROM items"
        " WHERE prefilter_pass IS NULL AND text_purged=0 ORDER BY rowid"
    ))


def known_hashes(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT text_hash FROM items WHERE prefilter_pass IS NOT NULL AND text_hash != ''")
    return {r[0] for r in cur.fetchall()}


def set_prefilter(conn: sqlite3.Connection, updates: list[tuple[str, int, str]]) -> None:
    conn.executemany("UPDATE items SET prefilter_hits=?, prefilter_pass=? WHERE id=?", updates)


def items_for_scoring(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(conn.execute(
        "SELECT id, source, created_at, engagement, prefilter_hits FROM items WHERE prefilter_pass=1"
    ))


def set_meta_scores(conn: sqlite3.Connection, scores: dict[str, float]) -> None:
    conn.executemany("UPDATE items SET meta_score=? WHERE id=?", [(s, i) for i, s in scores.items()])


def retention_days() -> int:
    return int(config.settings().get("retention_days", 30))


def max_classify_attempts() -> int:
    return int(config.settings().get("pipeline", {}).get("max_classify_attempts", 3))


def _eligible_sql(niche_ids: list[str]) -> tuple[str, tuple[Any, ...]]:
    """Items still waiting for a card: in these niches, passed the prefilter, text not purged, fetched inside the
    retention window, and not set aside after repeated failures."""
    cutoff = iso(utcnow() - timedelta(days=retention_days()))
    sql = f"""FROM items i
              JOIN item_niches n ON n.item_id = i.id
              LEFT JOIN cards c ON c.item_id = i.id
              WHERE n.niche IN ({_ph(len(niche_ids))})
                AND i.prefilter_pass = 1 AND i.text_purged = 0 AND c.item_id IS NULL
                AND i.fetched_at >= ? AND COALESCE(i.classify_attempts, 0) < ?"""
    return sql, (*niche_ids, cutoff, max_classify_attempts())


def classification_candidates(conn: sqlite3.Connection, niche_ids: list[str], limit: int) -> list[dict[str, Any]]:
    if not niche_ids:
        return []
    where, args = _eligible_sql(niche_ids)
    return _rows(conn.execute(
        f"""SELECT DISTINCT i.id, i.source, i.url, i.title, i.text, i.extra, i.meta_score {where}
            ORDER BY i.meta_score DESC LIMIT ?""",
        (*args, max(0, limit)),
    ))


def pending_count(conn: sqlite3.Connection, niche_ids: list[str]) -> int:
    if not niche_ids:
        return 0
    where, args = _eligible_sql(niche_ids)
    return conn.execute(f"SELECT COUNT(DISTINCT i.id) {where}", args).fetchone()[0]


def record_classify_failures(conn: sqlite3.Connection, failures: list[tuple[str, str]]) -> None:
    conn.executemany(
        "UPDATE items SET classify_attempts = COALESCE(classify_attempts, 0) + 1, classify_error = ? WHERE id = ?",
        [(cap(one_line(reason), 300), item_id) for item_id, reason in failures],
    )


def set_aside_items(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Items excluded from classification after repeated failures."""
    return _rows(conn.execute(
        """SELECT i.id, i.source, i.url, i.classify_attempts AS attempts, i.classify_error AS last_error
           FROM items i LEFT JOIN cards c ON c.item_id = i.id
           WHERE c.item_id IS NULL AND COALESCE(i.classify_attempts, 0) >= ? ORDER BY i.id""",
        (max_classify_attempts(),),
    ))


def clear_text(conn: sqlite3.Connection, ids: list[str]) -> int:
    """Delete raw text and source details for these items now (used for sources whose terms say so)."""
    n = 0
    for part in chunks(ids, 500):
        n += conn.execute(
            f"UPDATE items SET text=NULL, title=NULL, extra=NULL, text_purged=1 WHERE id IN ({_ph(len(part))})",
            part,
        ).rowcount
    return n


# ---------------------------------------------------------------- cards

def save_cards(conn: sqlite3.Connection, cards: Iterable[Any], provider: str, model: str) -> None:
    now = iso(utcnow())
    rows = []
    for c in cards:
        statement, who, solutions = clean_card_fields(c.problem_statement, c.who_has_it, c.existing_solutions)
        rows.append((
            c.item_id, int(c.is_problem), statement, who, c.audience, c.frequency,
            c.severity, c.willingness_to_pay, json.dumps(solutions), int(c.software_solvable),
            c.app_type, c.build_effort, provider, model, now,
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO cards (item_id, is_problem, problem_statement, who_has_it, audience, frequency,"
        " severity, willingness_to_pay, existing_solutions, software_solvable, app_type, build_effort,"
        " provider, model, classified_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def cards_for_clustering(conn: sqlite3.Connection, niche_ids: list[str]) -> list[dict[str, Any]]:
    if not niche_ids:
        return []
    return _rows(conn.execute(
        f"""SELECT DISTINCT c.*, i.source, i.url, i.author_hash, i.meta_score, i.created_at
            FROM cards c
            JOIN items i ON i.id = c.item_id
            JOIN item_niches n ON n.item_id = i.id
            WHERE n.niche IN ({_ph(len(niche_ids))})
              AND c.is_problem = 1 AND c.software_solvable = 1 AND c.app_type != 'not_software'
            ORDER BY c.item_id""",
        niche_ids,
    ))


def cards_table(conn: sqlite3.Connection, niche_ids: list[str] | None = None) -> list[dict[str, Any]]:
    where, args = "", []
    if niche_ids:
        where = f"WHERE c.item_id IN (SELECT item_id FROM item_niches WHERE niche IN ({_ph(len(niche_ids))}))"
        args = list(niche_ids)
    return _rows(conn.execute(
        f"""SELECT c.item_id, i.source, c.is_problem, c.problem_statement, c.who_has_it, c.audience,
                   c.severity, c.willingness_to_pay, c.app_type, c.build_effort, c.frequency,
                   c.existing_solutions, i.url, i.created_at, c.model,
                   (SELECT group_concat(niche, ', ') FROM item_niches WHERE item_id = c.item_id) AS niches
            FROM cards c JOIN items i ON i.id = c.item_id {where}
            ORDER BY c.is_problem DESC, c.severity DESC""",
        args,
    ))


# ---------------------------------------------------------------- embeddings

def get_embeddings(conn: sqlite3.Connection, ids: list[str], model: str) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {}
    for part in chunks(ids, 500):
        cur = conn.execute(
            f"SELECT item_id, vector FROM embeddings WHERE model=? AND item_id IN ({_ph(len(part))})",
            (model, *part),
        )
        for item_id, blob in cur.fetchall():
            out[item_id] = np.frombuffer(blob, dtype=np.float32)
    return out


def put_embeddings(conn: sqlite3.Connection, vectors: dict[str, Any], model: str) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (item_id, model, vector) VALUES (?,?,?)",
        [(k, model, v.astype("float32").tobytes()) for k, v in vectors.items()],
    )


# ---------------------------------------------------------------- clusters

def insert_cluster(
    conn: sqlite3.Connection, batch_id: str, niche_ids: list[str], stats: dict[str, Any],
    summary: dict[str, Any], member_ids: list[str],
) -> int:
    clean = lambda key, limit: cap(scrub(one_line(summary.get(key))), limit)  # noqa: E731
    summary = {**summary, "title": clean("title", 120), "problem": clean("problem", 600),
               "target_user": clean("target_user", 200), "app_concept": clean("app_concept", 500),
               "search_terms": [cap(scrub(one_line(t)), 80) for t in list(summary.get("search_terms") or [])[:4]
                                if str(t or "").strip()]}
    cur = conn.execute(
        """INSERT INTO clusters (batch_id, niches, title, problem, target_user, app_concept, app_type, audience,
               build_effort, search_terms, size, distinct_authors, sources, mean_severity, wtp_score,
               wtp_explicit_share, wtp_any_share, solutions_share, score, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            batch_id, json.dumps(niche_ids), summary["title"], summary["problem"], summary["target_user"],
            summary["app_concept"], summary["app_type"], summary["audience"], summary["build_effort"],
            json.dumps(summary["search_terms"]), stats["size"], stats["distinct_authors"],
            json.dumps(stats["sources"]), stats["mean_severity"], stats["wtp_score"],
            stats["wtp_explicit_share"], stats["wtp_any_share"], stats["solutions_share"], stats["score"],
            iso(utcnow()),
        ),
    )
    cid = int(cur.lastrowid)
    conn.executemany("INSERT OR IGNORE INTO cluster_members (cluster_id, item_id) VALUES (?,?)",
                     [(cid, m) for m in member_ids])
    return cid


def cluster_batches(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(conn.execute(
        "SELECT batch_id, MIN(created_at) AS created_at, MAX(niches) AS niches, COUNT(*) AS n"
        " FROM clusters GROUP BY batch_id ORDER BY batch_id DESC"
    ))


def clusters_in_batch(conn: sqlite3.Connection, batch_id: str) -> list[dict[str, Any]]:
    return _rows(conn.execute("SELECT * FROM clusters WHERE batch_id=? ORDER BY score DESC", (batch_id,)))


def get_cluster(conn: sqlite3.Connection, cluster_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM clusters WHERE id=?", (cluster_id,)).fetchone()
    return dict(row) if row else None


def cluster_members(conn: sqlite3.Connection, cluster_id: int) -> list[dict[str, Any]]:
    return _rows(conn.execute(
        """SELECT c.*, i.source, i.url, i.created_at FROM cluster_members m
           JOIN cards c ON c.item_id = m.item_id JOIN items i ON i.id = m.item_id
           WHERE m.cluster_id=? ORDER BY c.severity DESC""",
        (cluster_id,),
    ))


def set_existence(conn: sqlite3.Connection, cluster_id: int, data: dict[str, Any]) -> None:
    conn.execute("UPDATE clusters SET existence=? WHERE id=?", (json.dumps(data), cluster_id))


# ---------------------------------------------------------------- specs

def insert_spec(conn: sqlite3.Connection, cluster_id: int, title: str, markdown: str, provider: str, model: str,
                contract: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO specs (cluster_id, title, markdown, status, provider, model, created_at, contract)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (cluster_id, title, markdown, "draft", provider, model, iso(utcnow()), contract),
    )
    return int(cur.lastrowid)


def list_specs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(conn.execute(
        "SELECT id, cluster_id, title, status, provider, model, created_at, file_path FROM specs ORDER BY id DESC"
    ))


def get_spec(conn: sqlite3.Connection, spec_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM specs WHERE id=?", (spec_id,)).fetchone()
    return dict(row) if row else None


def update_spec(conn: sqlite3.Connection, spec_id: int, **fields: Any) -> None:
    allowed = {"status", "file_path", "markdown", "contract"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if sets:
        conn.execute(
            f"UPDATE specs SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?", (*sets.values(), spec_id)
        )


# ---------------------------------------------------------------- usage, runs, housekeeping

def record_usage(role: str, task: str, provider: str, model: str, input_tokens: int, output_tokens: int,
                 cache_read_tokens: int = 0) -> None:
    with session() as conn:
        conn.execute(
            "INSERT INTO llm_usage (ts, role, task, provider, model, input_tokens, output_tokens, cache_read_tokens)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (iso(utcnow()), role, task, provider, model, input_tokens, output_tokens, cache_read_tokens),
        )


def usage_summary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(conn.execute(
        """SELECT provider, model, task, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens,
                  SUM(output_tokens) AS output_tokens, SUM(cache_read_tokens) AS cache_read_tokens
           FROM llm_usage GROUP BY provider, model, task ORDER BY provider, model, task"""
    ))


def start_run(conn: sqlite3.Connection, params: dict[str, Any]) -> int:
    cur = conn.execute("INSERT INTO runs (started_at, params, status) VALUES (?,?,?)",
                       (iso(utcnow()), json.dumps(params), "running"))
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, stats: dict[str, Any], status: str) -> None:
    conn.execute("UPDATE runs SET finished_at=?, stats=?, status=? WHERE id=?",
                 (iso(utcnow()), json.dumps(stats), status, run_id))


def recent_runs(conn: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    return _rows(conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)))


def purge_expired(retention_days: int) -> int:
    """Delete raw post text and source details older than the retention window. Problem cards are kept."""
    cutoff = iso(utcnow() - timedelta(days=retention_days))
    with session() as conn:
        n = conn.execute(
            "UPDATE items SET text=NULL, title=NULL, extra=NULL, text_purged=1 WHERE text_purged=0 AND fetched_at < ?",
            (cutoff,),
        ).rowcount
    if n:
        conn = connect()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # don't leave old pages in the write-ahead log
        finally:
            conn.close()
    return n


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    return {
        "items": q("SELECT COUNT(*) FROM items"),
        "passed": q("SELECT COUNT(*) FROM items WHERE prefilter_pass=1"),
        "cards": q("SELECT COUNT(*) FROM cards"),
        "problems": q("SELECT COUNT(*) FROM cards WHERE is_problem=1"),
        "clusters": q("SELECT COUNT(*) FROM clusters"),
        "specs": q("SELECT COUNT(*) FROM specs"),
        "approved": q("SELECT COUNT(*) FROM specs WHERE status='approved'"),
    }
