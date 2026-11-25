"""End-to-end pipeline with stored items and a fake LLM (no network)."""
import json
import re
from datetime import timedelta

from idea_miner.core import db
from idea_miner.core.models import RawItem
from idea_miner.llm.base import LLMClient, LLMResult
from idea_miner.pipeline import classify, cluster, metadata, prefilter, spec_writer
from idea_miner.util import utcnow

POSTS = [
    ("Is there an app to reconcile UPI payments with invoices? I'd pay for it.", "shop owners", "upi"),
    ("I'm so tired of manually matching UPI credits to GST invoices every week.", "shop owners", "upi"),
    ("Anyone know a tool that matches UPI transactions to bills automatically?", "shopkeepers", "upi"),
    ("Is there a tool to track which tenants paid rent this month? Spreadsheet hell.", "landlords", "rent"),
    ("I wish there was an app that reminds tenants about rent and logs payments.", "landlords", "rent"),
    ("There has to be a better way to collect rent from 12 tenants than WhatsApp.", "landlords", "rent"),
    ("Lovely weather today, went for a walk by the lake with my dog", "", "noise"),
]


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self):
        super().__init__("fake-1")

    def _complete(self, system, user, *, max_tokens, task, schema):
        if task == "classify":
            results = []
            for item_id, body in re.findall(r'<item id="([^"]+)">(.*?)</item>', user, re.S):
                topic = "upi" if "UPI" in body else "rent" if "rent" in body else "noise"
                results.append({
                    "item_id": item_id, "is_problem": topic != "noise",
                    "problem_statement": {"upi": "Matching UPI payments to invoices is slow manual work",
                                          "rent": "Tracking and chasing tenant rent payments is manual",
                                          "noise": "none"}[topic],
                    "who_has_it": {"upi": "small shop owners", "rent": "small landlords", "noise": "n/a"}[topic],
                    "audience": "small_businesses", "frequency": "weekly", "severity": 7 if topic != "noise" else 1,
                    "willingness_to_pay": "explicit" if "pay" in body else "implied",
                    "existing_solutions": ["Excel"] if topic == "upi" else [],
                    "software_solvable": topic != "noise",
                    "app_type": "web_app" if topic != "noise" else "not_software", "build_effort": "weeks",
                })
            return LLMResult(text=json.dumps({"results": results}), input_tokens=100, output_tokens=50)
        if task == "summarize":
            topic = "UPI reconciliation" if "UPI" in user else "Rent tracking"
            return LLMResult(text=json.dumps({
                "title": topic, "problem": "p", "target_user": "t", "app_concept": "c", "app_type": "web_app",
                "audience": "small_businesses", "build_effort": "weeks", "search_terms": [topic]}))
        return LLMResult(text="## Problem\nSomething.\n## MVP scope\n1. Feature")


def _store_items():
    items = [RawItem(source="hackernews", source_id=str(i), url=f"https://news.ycombinator.com/item?id={i}",
                     text=text, created_at=utcnow() - timedelta(days=i), engagement={"points": i},
                     author=f"user{i}") for i, (text, _, _) in enumerate(POSTS)]
    with db.session() as conn:
        assert db.upsert_items(conn, items, "small_business", {"license": "x", "tos_status": "y"}) == len(POSTS)
        # second insert of the same items is a no-op
        assert db.upsert_items(conn, items, "small_business", {}) == 0


def test_full_pipeline(tmp_path):
    db.init()
    _store_items()
    logs = []
    pf = prefilter.run(logs.append)
    assert pf["passed"] == 6  # the weather post fails the prefilter
    assert metadata.run() == 6

    llm = FakeLLM()
    stats = classify.run(llm, ["small_business"], max_items=50, batch_size=4, workers=2, max_chars=500,
                         log=logs.append)
    assert stats == {"classified": 6, "problems": 6, "failed_batches": 0}
    # nothing left to classify on a second run
    assert classify.run(llm, ["small_business"], max_items=50, batch_size=4, workers=2, max_chars=500,
                        log=logs.append)["classified"] == 0

    batch_id = cluster.build(llm, ["small_business"], min_cluster_size=2, max_ideas=10, workers=2, log=logs.append)
    assert batch_id
    with db.session() as conn:
        ideas = db.clusters_in_batch(conn, batch_id)
    titles = {c["title"] for c in ideas}
    assert {"UPI reconciliation", "Rent tracking"} <= titles
    assert all(c["distinct_authors"] == 3 for c in ideas)

    spec_id = spec_writer.generate(ideas[0]["id"], llm)
    path = spec_writer.approve(spec_id)
    md = path.read_text(encoding="utf-8")
    assert md.startswith(f"# {ideas[0]['title']}")
    assert "## Evidence" in md and "news.ycombinator.com/item?id=" in md
    assert path.with_suffix(".json").exists()

    with db.session() as conn:
        usage = db.usage_summary(conn)
    assert usage == []  # usage is only recorded for clients built by the factory


def test_group_stats_rewards_more_people_and_payment():
    base = {"source": "hackernews", "severity": 6, "willingness_to_pay": "none", "existing_solutions": "[]",
            "meta_score": 0.5, "app_type": "web_app", "audience": "consumers", "build_effort": "weeks"}
    one = [dict(base, item_id="a", author_hash="a")]
    three_paying = [dict(base, item_id=k, author_hash=k, willingness_to_pay="explicit") for k in "abc"]
    assert cluster.group_stats(three_paying)["score"] > cluster.group_stats(one)["score"]
