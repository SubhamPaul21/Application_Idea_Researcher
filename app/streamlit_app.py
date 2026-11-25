"""App Idea Researcher: Streamlit UI.  Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without `pip install -e .`

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from idea_miner import config  # noqa: E402
from idea_miner.core import db  # noqa: E402
from idea_miner.core.schemas import AUDIENCES, EFFORTS, SOFTWARE_APP_TYPES  # noqa: E402
from idea_miner.llm import factory  # noqa: E402
from idea_miner.llm.base import LLMClient, LLMError  # noqa: E402
from idea_miner.pipeline import embed, existence, runner, spec_writer  # noqa: E402
from idea_miner.sources import registry  # noqa: E402

st.set_page_config(page_title="App Idea Researcher", page_icon="💡", layout="wide")


@st.cache_resource
def _bootstrap() -> int:
    db.init()
    return db.purge_expired(int(config.settings().get("retention_days", 30)))


_bootstrap()

APP_TYPE_LABELS = {
    "script": "Script / automation", "cli": "Command-line tool", "web_app": "Web app",
    "browser_extension": "Browser extension", "mobile_app": "Mobile app", "desktop_app": "Desktop app",
    "api_service": "API / backend service", "bot_or_integration": "Bot / integration", "not_software": "Not software",
}
AUDIENCE_LABELS = {
    "consumers": "Consumers", "professionals": "Professionals", "small_businesses": "Small businesses",
    "developers": "Developers", "enterprises": "Enterprises",
}
EFFORT_LABELS = {"weekend": "A weekend", "weeks": "A few weeks", "months": "Months"}
WTP_OPTIONS = ["Any", "Implied or explicit", "Explicit only"]
VERDICT_BADGE = {"unsolved": ":green[**Unsolved**]", "partially_solved": ":orange[**Partially solved**]",
                 "solved": ":red[**Already solved**]"}

IDEA_TYPES: dict[str, dict] = {
    "All app ideas": {},
    "Quick wins: scripts & CLIs you can build in a weekend": {
        "app_types": ["script", "cli", "bot_or_integration"], "efforts": ["weekend"]},
    "Micro-SaaS: web apps businesses would pay for": {
        "app_types": ["web_app", "api_service", "bot_or_integration"],
        "audiences": ["small_businesses", "professionals", "enterprises"], "wtp": "Implied or explicit"},
    "Consumer apps (mobile & web)": {"app_types": ["mobile_app", "web_app"], "audiences": ["consumers"]},
    "Developer tools": {"audiences": ["developers"]},
    "Browser extensions & integrations": {"app_types": ["browser_extension", "bot_or_integration"]},
    "Desktop utilities": {"app_types": ["desktop_app", "script", "cli"]},
    "High-pain problems people pay for": {"wtp": "Explicit only", "min_sev": 7},
}

NICHES = {n["id"]: n["label"] for n in config.niches()}
REGISTRY = config.sources_registry()


# ------------------------------------------------------------------ sidebar
def llm_picker(role: str, prefix: str, default_provider: str, default_model: str) -> dict:
    providers = list(factory.PROVIDERS)
    prov = st.selectbox(f"{role}: provider", providers, index=providers.index(default_provider),
                        format_func=lambda p: factory.PROVIDERS[p]["label"], key=f"{prefix}_prov")
    preset = factory.PROVIDERS[prov]
    models_key = f"{prefix}_models_{prov}"
    initial = default_model if prov == default_provider else factory.default_model(prov)

    key_override = None
    if not factory.key_available(prov):
        key_override = st.text_input(f"{preset['key_env']} (this session only)", type="password",
                                     key=f"{prefix}_key_{prov}") or None
    base_url = None
    if prov == "custom" or preset["kind"] == "openai" and prov in ("ollama", "lmstudio"):
        base_url = st.text_input("Base URL", value=factory.default_base_url(prov), key=f"{prefix}_url_{prov}")

    if st.session_state.get(models_key):
        options = st.session_state[models_key]
        model = st.selectbox(f"{role}: model", options,
                             index=options.index(initial) if initial in options else 0, key=f"{prefix}_msel_{prov}")
    else:
        model = st.text_input(f"{role}: model ID", value=initial, key=f"{prefix}_model_{prov}",
                              placeholder="exact model ID from the provider")
        if preset.get("models"):
            st.caption("Suggested: " + ", ".join(preset["models"]))
    if st.button("Load model list", key=f"{prefix}_load_{prov}"):
        try:
            client = factory.make_llm(prov, model or "x", api_key=key_override, base_url=base_url, track_usage=False)
            st.session_state[models_key] = client.list_models()
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not list models: {e}")
    if not factory.key_available(prov, key_override):
        st.warning(f"No API key: set {preset['key_env']} in .env or paste it above.")
    return {"provider": prov, "model": (model or "").strip(), "api_key": key_override, "base_url": base_url}


with st.sidebar:
    st.title("💡 App Idea Researcher")

    st.subheader("1. What to look for")
    sel_niches = st.multiselect("Niche", list(NICHES), default=[list(NICHES)[0]], format_func=NICHES.get,
                                help="Each niche has its own search terms per source (config/niches.yaml).")
    custom_kw = st.text_input("Custom keywords (comma-separated)", placeholder="e.g. dental clinic, tuition centre")
    idea_type = st.selectbox("Idea type", list(IDEA_TYPES))
    preset = IDEA_TYPES[idea_type]
    with st.expander("Fine-tune idea filters", expanded=False):
        f_app_types = st.multiselect("App types", SOFTWARE_APP_TYPES, default=preset.get("app_types", SOFTWARE_APP_TYPES),
                                     format_func=APP_TYPE_LABELS.get, key=f"f_types_{idea_type}")
        f_audiences = st.multiselect("Audience", AUDIENCES, default=preset.get("audiences", AUDIENCES),
                                     format_func=AUDIENCE_LABELS.get, key=f"f_aud_{idea_type}")
        f_efforts = st.multiselect("Build effort", EFFORTS, default=preset.get("efforts", EFFORTS),
                                   format_func=EFFORT_LABELS.get, key=f"f_eff_{idea_type}")
        f_wtp = st.selectbox("Willingness to pay", WTP_OPTIONS, index=WTP_OPTIONS.index(preset.get("wtp", "Any")),
                             key=f"f_wtp_{idea_type}")
        f_min_sev = st.slider("Minimum severity", 1, 10, preset.get("min_sev", 3), key=f"f_sev_{idea_type}")
        f_min_people = st.slider("Minimum distinct people reporting it", 1, 10, 1, key=f"f_ppl_{idea_type}")

    st.subheader("2. Where to look")
    region = st.radio("Region", list(runner.REGIONS), horizontal=True,
                      help="Sets the App Store country and YouTube region.")
    src_ids = registry.implemented_ids()
    src_status = {s: registry.get(s).available() for s in src_ids}
    usable = [s for s in src_ids if src_status[s][0]]
    sel_sources = st.multiselect("Sources", src_ids, default=usable,
                                 format_func=lambda s: REGISTRY[s]["label"] + ("" if src_status[s][0] else " (needs key)"))
    days = st.slider("Look-back window (days)", 7, 365, 60)
    max_kw = st.slider("Search terms per source", 1, 8, 3, help="More terms = more data and more API calls.")
    max_per_query = st.slider("Max results per search", 10, 200, 50, step=10)

    st.subheader("3. AI model")
    d = config.settings()["llm"]["defaults"]
    clf_cfg = llm_picker("Classifier", "clf", d["classifier"]["provider"], d["classifier"]["model"])
    same_writer = st.checkbox("Use the same model for idea names, existence checks and specs",
                              value=d["classifier"] == d["writer"])
    wr_cfg = clf_cfg if same_writer else llm_picker("Writer", "wr", d["writer"]["provider"], d["writer"]["model"])
    hd = d.get("helper")
    use_helper = False
    hp_cfg = wr_cfg
    if hd:
        use_helper = st.checkbox("Use a lighter model to name ideas (saves tokens)", value=True,
                                 help="Naming idea clusters is simple summarizing, so a smaller model is enough.")
        if use_helper:
            hp_cfg = llm_picker("Helper", "hp", hd["provider"], hd["model"])
    max_classify = st.slider("Max items to classify per run (cost cap)", 10, 1000, 120, step=10)

params = runner.RunParams(
    niche_ids=sel_niches,
    custom_keywords=[k for k in (custom_kw or "").split(",") if k.strip()],
    source_ids=[s for s in sel_sources if src_status[s][0]],
    days=days, max_per_query=max_per_query, max_keywords=max_kw, region=region, max_to_classify=max_classify,
)
filters = dict(app_types=f_app_types, audiences=f_audiences, efforts=f_efforts, wtp=f_wtp,
               min_sev=f_min_sev, min_people=f_min_people)
country = params.countries[0]


def make(cfg: dict, role: str) -> LLMClient | None:
    try:
        return factory.make_llm(cfg["provider"], cfg["model"], api_key=cfg["api_key"], base_url=cfg["base_url"],
                                role=role)
    except LLMError as e:
        st.error(str(e))
        return None


def idea_passes(c: dict, f: dict) -> bool:
    if c["app_type"] not in f["app_types"] or c["audience"] not in f["audiences"]:
        return False
    if c["build_effort"] not in f["efforts"] or (c["mean_severity"] or 0) < f["min_sev"]:
        return False
    if (c["distinct_authors"] or 0) < f["min_people"]:
        return False
    if f["wtp"] == "Implied or explicit" and not c["wtp_any_share"]:
        return False
    if f["wtp"] == "Explicit only" and not c["wtp_explicit_share"]:
        return False
    return True


# ------------------------------------------------------------------ header
with db.session() as conn:
    counts = db.counts(conn)
    pending = db.pending_count(conn, params.all_niche_ids())
m = st.columns(5)
m[0].metric("Posts collected", counts["items"])
m[1].metric("Passed prefilter", counts["passed"])
m[2].metric("Real problems found", counts["problems"])
m[3].metric("Ideas ranked", counts["clusters"])
m[4].metric("Specs approved", counts["approved"])

tab_run, tab_ideas, tab_specs, tab_cards, tab_sources = st.tabs(
    ["Discover", "Ideas", "Specs", "Problem cards", "Sources & compliance"])

# ------------------------------------------------------------------ Discover
with tab_run:
    niches = params.niches()
    if not niches:
        st.warning("Pick at least one niche or enter custom keywords in the sidebar.")
    else:
        st.markdown("**Niches:** " + " · ".join(n["label"] for n in niches))
        st.markdown(f"**Sources:** {', '.join(REGISTRY[s]['label'] for s in params.source_ids) or 'none'}  \n"
                    f"**Window:** last {days} days · **Region:** {region} · "
                    f"**Classifier:** {clf_cfg['provider']}/{clf_cfg['model'] or '?'} · "
                    f"**Writer:** {wr_cfg['provider']}/{wr_cfg['model'] or '?'}")
        st.caption(f"{pending} collected items in these niches are waiting for classification. Each run sends at "
                   f"most {max_classify} items (about {max(1, max_classify // 8)} LLM calls) to the classifier.")

    ready = bool(niches and params.source_ids)
    col_a, col_b = st.columns([1, 2])
    run_full = col_a.button("Find app ideas", type="primary", disabled=not ready, width="stretch")
    with col_b.expander("Run one stage at a time"):
        s_cols = st.columns(4)
        run_fetch = s_cols[0].button("1. Fetch", disabled=not ready)
        run_filter = s_cols[1].button("2. Prefilter")
        run_classify = s_cols[2].button("3. Classify", disabled=not niches)
        run_cluster = s_cols[3].button("4. Rank ideas", disabled=not niches)

    stages: tuple[str, ...] = ()
    if run_full:
        stages = ("fetch", "filter", "classify", "cluster")
    elif run_fetch:
        stages = ("fetch", "filter")
    elif run_filter:
        stages = ("filter",)
    elif run_classify:
        stages = ("classify",)
    elif run_cluster:
        stages = ("cluster",)

    if stages:
        classifier = make(clf_cfg, "classifier") if "classify" in stages or "cluster" in stages else None
        writer = make(wr_cfg, "writer") if "classify" in stages or "cluster" in stages else None
        helper = make(hp_cfg, "helper") if use_helper and "cluster" in stages else None
        needs_llm = "classify" in stages or "cluster" in stages
        if needs_llm and not (classifier and writer) or (use_helper and "cluster" in stages and not helper):
            st.stop()
        with st.status("Working…", expanded=True) as status:
            bar = st.progress(0.0)
            log = status.write
            prog = lambda frac, text: bar.progress(min(max(frac, 0.0), 1.0), text=text)  # noqa: E731
            try:
                if needs_llm:
                    result = runner.run_all(params, classifier, writer, log, prog, stages=stages, helper=helper)
                else:
                    result = {}
                    if "fetch" in stages:
                        result["fetch"] = runner.fetch_stage(params, log, prog)
                    if "filter" in stages:
                        result["filter"] = runner.filter_stage(log)
                status.update(label="Done", state="complete")
                if result.get("batch_id"):
                    st.success("New ideas are ready in the **Ideas** tab.")
            except Exception as e:  # noqa: BLE001
                status.update(label="Failed", state="error")
                st.exception(e)

    with db.session() as conn:
        runs = db.recent_runs(conn, 8)
    if runs:
        st.markdown("##### Recent runs")
        st.dataframe(pd.DataFrame([{
            "started": r["started_at"], "status": r["status"],
            "stages": ", ".join(json.loads(r["params"] or "{}").get("stages", [])),
            "niches": ", ".join(config.niche_label(n) for n in json.loads(r["params"] or "{}").get("niche_ids", [])),
            "result": r["stats"],
        } for r in runs]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Ideas
with tab_ideas:
    with db.session() as conn:
        batches = db.cluster_batches(conn)
    if not batches:
        st.info("No ideas yet. Run **Find app ideas** on the Discover tab.")
    else:
        def batch_label(b: dict) -> str:
            names = ", ".join(config.niche_label(n) for n in json.loads(b["niches"] or "[]"))
            return f"{b['created_at'][:16].replace('T', ' ')} · {names} · {b['n']} ideas"

        top = st.columns([3, 1])
        batch = top[0].selectbox("Idea batch", batches, format_func=batch_label)
        sort_by = top[1].selectbox("Sort by", ["Score", "People reporting", "Severity", "Willing to pay"])
        with db.session() as conn:
            clusters = db.clusters_in_batch(conn, batch["batch_id"])
        shown = [c for c in clusters if idea_passes(c, filters)]
        key = {"Score": "score", "People reporting": "distinct_authors", "Severity": "mean_severity",
               "Willing to pay": "wtp_any_share"}[sort_by]
        shown.sort(key=lambda c: c[key] or 0, reverse=True)
        st.caption(f"Showing {len(shown)} of {len(clusters)} ideas that match **{idea_type}** and your filters.")

        for rank, c in enumerate(shown, 1):
            with st.container(border=True):
                head = st.columns([6, 1])
                head[0].markdown(f"#### {rank}. {c['title']}")
                head[1].metric("Score", f"{c['score']:.1f}")
                st.caption(" · ".join([
                    APP_TYPE_LABELS.get(c["app_type"], c["app_type"]),
                    AUDIENCE_LABELS.get(c["audience"], c["audience"]),
                    f"build: {EFFORT_LABELS.get(c['build_effort'], c['build_effort']).lower()}",
                    f"{c['distinct_authors']} people", f"severity {c['mean_severity']:.1f}/10",
                    f"pays: {c['wtp_any_share']:.0%}",
                    "via " + ", ".join(REGISTRY.get(s, {}).get("label", s) for s in json.loads(c["sources"] or "[]")),
                ]))
                st.markdown(f"**Problem:** {c['problem']}")
                if c["app_concept"]:
                    st.markdown(f"**Idea:** {c['app_concept']}")
                st.markdown(f"**For:** {c['target_user']}")

                if c.get("existence"):
                    ex = json.loads(c["existence"])
                    st.markdown(f"**Already exists?** {VERDICT_BADGE.get(ex['verdict'], ex['verdict'])}: "
                                f"{ex.get('opportunity', '')}")
                    for a in ex.get("closest_alternatives", []):
                        st.markdown(f"- [{a['name']}]({a['url']}): {a['gap']}")

                with st.expander(f"Evidence ({c['size']} reports)"):
                    with db.session() as conn:
                        members = db.cluster_members(conn, c["id"])
                    for mem in members:
                        label = REGISTRY.get(mem["source"], {}).get("label", mem["source"])
                        st.markdown(f"- {mem['problem_statement']} · *severity {mem['severity']}, "
                                    f"pays: {mem['willingness_to_pay']}* · [{label}]({mem['url']})")

                b = st.columns([1, 1, 3])
                if b[0].button("Check if it exists", key=f"ex_{c['id']}"):
                    w = make(wr_cfg, "writer")
                    if w:
                        with st.spinner("Searching GitHub, the App Store and Show HN…"):
                            try:
                                existence.check(c["id"], w, country=country, log=lambda _m: None)
                                st.rerun()
                            except Exception as e:  # noqa: BLE001
                                st.error(f"Existence check failed: {e}")
                if b[1].button("Write PRD spec", key=f"spec_{c['id']}", type="primary"):
                    w = make(wr_cfg, "writer")
                    if w:
                        with st.spinner("Writing the spec…"):
                            try:
                                spec_writer.generate(c["id"], w)
                                st.success("Draft spec saved: review it on the **Specs** tab.")
                            except Exception as e:  # noqa: BLE001
                                st.error(f"Spec generation failed: {e}")

# ------------------------------------------------------------------ Specs
with tab_specs:
    with db.session() as conn:
        specs = db.list_specs(conn)
    if not specs:
        st.info("No specs yet. Click **Write PRD spec** on an idea.")
    else:
        spec_row = st.selectbox(
            "Spec", specs,
            format_func=lambda s: f"#{s['id']} {s['title']} [{s['status']}] · {s['created_at'][:10]} · {s['model']}")
        with db.session() as conn:
            spec = db.get_spec(conn, spec_row["id"])
        actions = st.columns([1, 1, 1, 3])
        if actions[0].button("Approve & save", type="primary", disabled=spec["status"] == "approved"):
            path = spec_writer.approve(spec["id"])
            st.success(f"Saved {path} (+ JSON contract for build agents)")
            st.rerun()
        if actions[1].button("Reject", disabled=spec["status"] == "rejected"):
            spec_writer.reject(spec["id"])
            st.rerun()
        actions[2].download_button("Download .md", spec["markdown"], file_name=f"spec-{spec['id']}.md",
                                   mime="text/markdown")
        if spec.get("file_path"):
            actions[3].caption(f"Saved to {spec['file_path']}")
        st.divider()
        st.markdown(spec["markdown"])

# ------------------------------------------------------------------ Problem cards
with tab_cards:
    only_selected = st.toggle("Only the niches selected in the sidebar", value=True)
    only_problems = st.toggle("Only real problems", value=True)
    with db.session() as conn:
        rows = db.cards_table(conn, params.all_niche_ids() if only_selected else None)
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No classified items yet.")
    else:
        if only_problems:
            df = df[df["is_problem"] == 1]
        df = df.drop(columns=["is_problem"])
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("link", display_text="open"),
                                    "severity": st.column_config.NumberColumn(format="%d")})
        st.download_button("Download CSV", df.to_csv(index=False), file_name="problem_cards.csv", mime="text/csv")

# ------------------------------------------------------------------ Sources & compliance
with tab_sources:
    st.markdown("##### Data sources")
    table = []
    for sid, meta in REGISTRY.items():
        if meta.get("implemented"):
            ok, why = src_status.get(sid, (False, "not implemented"))
            key_env = meta.get("key_env")
            note = why if not ok else (f"optional {key_env} not set (lower limits)"
                                       if key_env and not meta.get("key_required") and not config.env(key_env) else "")
            status_txt = "Active" if ok else "Unavailable"
        else:
            status_txt, note = "Excluded", meta.get("reason", "")
        table.append({"source": meta.get("label", sid), "status": status_txt, "note": note,
                      "access": meta.get("access", ""), "license": meta.get("license", ""),
                      "terms": meta.get("tos_status", "")})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

    if st.button("Run live health check on active sources"):
        results = []
        with st.spinner("Fetching a few items from each source…"):
            for sid in src_ids:
                ok, msg = registry.smoke(sid)
                results.append({"source": REGISTRY[sid]["label"], "ok": "✅" if ok else "❌", "detail": msg})
        st.dataframe(pd.DataFrame(results), hide_index=True, width="stretch")

    st.markdown("##### Privacy & retention")
    retention = int(config.settings().get("retention_days", 30))
    st.markdown(
        f"- Usernames are stored only as salted hashes; emails, phone numbers, IDs and profile links are removed "
        f"before storage and before any LLM call.\n"
        f"- Raw post text is deleted after **{retention} days**. Problem cards keep only paraphrases and links.\n"
        f"- Specs cite links and paraphrases, never original text.\n"
        f"- Embeddings backend: **{embed.backend()}** (runs locally, no data leaves your machine).")
    if st.button("Purge expired raw text now"):
        st.success(f"Purged {db.purge_expired(retention)} items.")

    st.markdown("##### LLM usage")
    with db.session() as conn:
        usage = db.usage_summary(conn)
    if usage:
        st.dataframe(pd.DataFrame(usage), hide_index=True, width="stretch")
    else:
        st.caption("No LLM calls yet.")
