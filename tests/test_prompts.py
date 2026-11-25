"""Guards against prompts.yaml drifting away from the code. If an enum value in schemas.py is missing from the
prompt that asks the model for it, the model never learns it exists and the lenient validator would silently
replace unknown answers with defaults."""
from typing import get_args

import pytest
import yaml

from idea_miner import config
from idea_miner.core import schemas

NAMES = ["classify", "cluster_summary", "existence", "spec"]


@pytest.mark.parametrize("name", NAMES)
def test_prompt_exists_and_is_nonempty(name):
    assert len(config.prompt(name)) > 200


def test_prompts_file_has_no_unknown_entries():
    raw = yaml.safe_load((config.CONFIG_DIR / "prompts.yaml").read_text(encoding="utf-8"))
    assert set(raw) == set(NAMES)


def test_missing_prompt_raises_clear_error():
    with pytest.raises(KeyError, match="no_such_prompt"):
        config.prompt("no_such_prompt")


def _values(literal) -> list[str]:
    return list(get_args(literal))


def test_classify_prompt_lists_every_allowed_value():
    text = config.prompt("classify")
    expected = (_values(schemas.AppType) + _values(schemas.Audience) + _values(schemas.Frequency)
                + _values(schemas.WTP) + _values(schemas.Effort))
    missing = [v for v in expected if v not in text]
    assert not missing, f"classify prompt does not mention: {missing}"
    for field in schemas.ProblemCard.model_fields:
        assert field in text or field in ("item_id",), f"classify prompt never names field '{field}'"
    assert "item_id" in text and "results" in text


def test_cluster_summary_prompt_lists_every_allowed_value():
    text = config.prompt("cluster_summary")
    expected = ([v for v in _values(schemas.AppType) if v != "not_software"]
                + _values(schemas.Audience) + _values(schemas.Effort))
    missing = [v for v in expected if v not in text]
    assert not missing, f"cluster_summary prompt does not mention: {missing}"
    for field in schemas.ClusterSummary.model_fields:
        assert field in text, f"cluster_summary prompt never names field '{field}'"


def test_existence_prompt_lists_every_verdict_and_field():
    text = config.prompt("existence")
    assert not [v for v in _values(schemas.Verdict) if v not in text]
    for field in schemas.ExistenceVerdict.model_fields:
        assert field in text, f"existence prompt never names field '{field}'"
    for field in schemas.Alternative.model_fields:
        assert field in text, f"existence prompt never names alternative field '{field}'"


def test_spec_prompt_has_all_required_sections():
    text = config.prompt("spec")
    for heading in ("## Problem", "## Target users", "## Existing alternatives and gaps", "## MVP scope",
                    "## Non-goals", "## Suggested architecture", "## Build plan", "## Risks and open questions"):
        assert heading in text, f"spec prompt is missing the section '{heading}'"
    # code appends these itself; the model must not write them
    assert "## Evidence" not in text and "## Sources and attribution" not in text
