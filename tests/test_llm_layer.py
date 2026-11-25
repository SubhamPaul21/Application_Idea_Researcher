import json

import pytest

from idea_miner.core.schemas import ClassificationBatch, ClusterSummary, ProblemCard
from idea_miner.llm.base import LLMClient, LLMError, LLMResult, extract_json, parse_model
from idea_miner.llm.schema import strict_schema


def test_strict_schema_inlines_refs_and_keeps_title_property():
    s = strict_schema(ClusterSummary)
    assert "title" in s["properties"] and "title" in s["required"]
    assert s["additionalProperties"] is False
    batch = strict_schema(ClassificationBatch)
    item = batch["properties"]["results"]["items"]
    assert "$ref" not in json.dumps(batch)
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(ProblemCard.model_fields)
    assert item["properties"]["app_type"]["enum"][0] == "script"


def test_extract_json_handles_fences_and_think_tags():
    assert extract_json('<think>hmm {"x": 0}</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_lenient_validation_normalises_values():
    raw = [{"item_id": "x", "is_problem": "yes", "problem_statement": "p", "who_has_it": "w",
            "audience": "Small Businesses", "frequency": "every day", "severity": "14",
            "willingness_to_pay": "Implied", "existing_solutions": "Excel", "software_solvable": True,
            "app_type": "Web App", "build_effort": "Weeks"}]
    batch = parse_model(json.dumps(raw), ClassificationBatch)  # bare list is wrapped into {"results": [...]}
    card = batch.results[0]
    assert card.is_problem is True
    assert card.audience == "small_businesses"
    assert card.frequency == "unknown"  # unknown value -> fallback
    assert card.severity == 10  # clamped
    assert card.existing_solutions == ["Excel"]
    assert card.app_type == "web_app"


class _ScriptedLLM(LLMClient):
    provider = "fake"

    def __init__(self, replies):
        super().__init__("scripted")
        self.replies = list(replies)
        self.calls = 0

    def _complete(self, system, user, *, max_tokens, task, schema):
        self.calls += 1
        return LLMResult(text=self.replies.pop(0))


def test_complete_json_repairs_once_then_fails():
    good = json.dumps({"verdict": "unsolved", "closest_alternatives": [], "opportunity": "wide open"})
    from idea_miner.core.schemas import ExistenceVerdict

    llm = _ScriptedLLM(["not json", good])
    assert llm.complete_json("s", "u", ExistenceVerdict).verdict == "unsolved"
    assert llm.calls == 2
    with pytest.raises(LLMError):
        _ScriptedLLM(["bad", "still bad"]).complete_json("s", "u", ExistenceVerdict)
