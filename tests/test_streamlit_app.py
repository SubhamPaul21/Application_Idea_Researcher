"""Headless UI test: the app renders, the niche dropdown and idea-type presets work, and ideas show up."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from idea_miner.core import db

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    niche = at.sidebar.multiselect[0]
    assert niche.label == "Niche" and len(niche.options) >= 10
    labels = [s.label for s in at.sidebar.selectbox]
    assert "Idea type" in labels
    assert any(b.label == "Find app ideas" for b in at.button)


def test_idea_type_preset_changes_filters_and_ideas_tab_lists_clusters():
    db.init()
    stats = {"size": 3, "distinct_authors": 3, "sources": ["hackernews"], "mean_severity": 7.0, "wtp_score": 1.5,
             "wtp_explicit_share": 1.0, "wtp_any_share": 1.0, "solutions_share": 0.0, "score": 20.0}
    base = {"problem": "p", "target_user": "t", "app_concept": "c", "audience": "small_businesses",
            "build_effort": "weeks", "search_terms": ["x"]}
    with db.session() as conn:
        db.insert_cluster(conn, "20261105-000000", ["small_business"], stats,
                          base | {"title": "Invoice reconciler", "app_type": "web_app"}, [])
        db.insert_cluster(conn, "20261105-000000", ["small_business"], stats,
                          base | {"title": "Receipt CLI", "app_type": "cli", "build_effort": "weekend"}, [])

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    page = "\n".join(m.value for m in at.markdown)
    assert "Invoice reconciler" in page and "Receipt CLI" in page

    idea_type = next(s for s in at.sidebar.selectbox if s.label == "Idea type")
    idea_type.select("Quick wins: scripts & CLIs you can build in a weekend").run()
    assert not at.exception
    page = "\n".join(m.value for m in at.markdown)
    assert "Receipt CLI" in page and "Invoice reconciler" not in page
