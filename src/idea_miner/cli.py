"""Command line:  idea-miner --help"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Optional

import typer

from . import config
from .core import db
from .llm import factory
from .llm.base import LLMError
from .pipeline import runner
from .sources import registry

app = typer.Typer(help="Mine real-world problems from legal public sources and turn them into app ideas.",
                  no_args_is_help=True)


UI_SECURITY_FLAGS = [
    "--server.address=127.0.0.1",  # only this computer can reach the app (it has no login)
    "--server.allowedHosts=localhost", "--server.allowedHosts=127.0.0.1",  # block DNS rebinding
    "--browser.gatherUsageStats=false",
]


@app.command()
def ui() -> None:
    """Launch the Streamlit app (bound to this computer only, protected against being framed by other sites)."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(config.ROOT / "app" / "server.py"),
                    *UI_SECURITY_FLAGS], cwd=config.ROOT, check=False)


@app.command()
def smoke() -> None:
    """Fetch a few live items from every source to prove the endpoints work today."""
    for sid in registry.implemented_ids():
        ok, msg = registry.smoke(sid)
        echo_plain(f"{'OK  ' if ok else 'FAIL'} {sid:<14} {msg}")


@app.command()
def niches() -> None:
    """List available niche IDs."""
    for n in config.niches():
        typer.echo(f"{n['id']:<26} {n['label']}")


@app.command()
def run(
    niche: list[str] = typer.Option([], "--niche", "-n", help="Niche ID (repeatable). See `idea-miner niches`."),
    keyword: list[str] = typer.Option([], "--keyword", "-k", help="Custom keyword (repeatable)."),
    source: list[str] = typer.Option([], "--source", "-s", help="Source ID (repeatable). Default: all available."),
    days: int = 60,
    region: str = "Global",
    max_classify: int = 120,
    provider: Optional[str] = typer.Option(None, help="LLM provider, e.g. anthropic, openai, groq, openrouter."),
    model: Optional[str] = typer.Option(None, help="Model ID for that provider."),
    stages: str = typer.Option("fetch,filter,classify,cluster", help="Comma-separated stages to run."),
) -> None:
    """Run the pipeline headless (same as 'Find app ideas' in the UI)."""
    stage_list = validate_run_options(niche, keyword, source, days, region, max_classify, stages, provider)
    db.init()
    defaults = config.settings()["llm"]["defaults"]["classifier"]
    prov = provider or defaults["provider"]
    mdl = model or (defaults["model"] if prov == defaults["provider"] else factory.default_model(prov))
    try:
        llm = factory.make_llm(prov, mdl, role="classifier")
    except LLMError as e:
        raise typer.BadParameter(str(e), param_hint="--provider/--model") from None
    params = runner.RunParams(
        niche_ids=niche, custom_keywords=keyword,
        source_ids=source or [s for s in registry.implemented_ids() if registry.get(s).available()[0]],
        days=days, region=region, max_to_classify=max_classify,
    )
    stats = runner.run_all(params, llm, llm, echo_plain, stages=stage_list)
    echo_plain(f"Done: {stats}")


STAGES = ("fetch", "filter", "classify", "cluster")


def validate_run_options(niche: list[str], keyword: list[str], source: list[str], days: int, region: str,
                         max_classify: int, stages: str, provider: str | None) -> tuple[str, ...]:
    """Fail on any unknown or out-of-range option instead of silently ignoring it."""
    def bad(msg: str, hint: str) -> typer.BadParameter:
        return typer.BadParameter(msg, param_hint=hint)

    known_niches = {n["id"] for n in config.niches()}
    if unknown := [n for n in niche if n not in known_niches]:
        raise bad(f"unknown niche {unknown}; see `idea-miner niches`", "--niche")
    if not niche and not [k for k in keyword if k.strip()]:
        raise bad("give at least one --niche or --keyword", "--niche/--keyword")
    if unknown := [s for s in source if s not in registry.implemented_ids()]:
        raise bad(f"unknown source {unknown}; choose from {registry.implemented_ids()}", "--source")
    if region not in runner.REGIONS:
        raise bad(f"unknown region {region!r}; choose from {list(runner.REGIONS)}", "--region")
    if not 1 <= days <= 3650:
        raise bad("must be between 1 and 3650", "--days")
    if not 1 <= max_classify <= 5000:
        raise bad("must be between 1 and 5000", "--max-classify")
    if provider and provider not in factory.PROVIDERS:
        raise bad(f"unknown provider {provider!r}; choose from {list(factory.PROVIDERS)}", "--provider")
    stage_list = tuple(s.strip() for s in stages.split(",") if s.strip())
    if not stage_list or (unknown := [s for s in stage_list if s not in STAGES]):
        raise bad(f"choose from {list(STAGES)}", "--stages")
    return stage_list


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]")


def echo_plain(message: str) -> None:
    """Print text that may contain remote content without terminal control or text-direction characters."""
    typer.echo(_CONTROL.sub("", str(message)))


@app.command("llm-test")
def llm_test(provider: str, model: str) -> None:
    """Classify two sample posts to check a provider/model works with structured output (costs a few tokens)."""
    from .pipeline.classify import classify_rows

    rows = [
        {"id": "test:1", "source": "test", "title": "Is there a tool to reconcile UPI payments with my GST invoices?",
         "text": "I run a small shop and spend 3 hours every week matching UPI credits to invoices in Excel. "
                 "I'd happily pay for something that does this.", "extra": "{}"},
        {"id": "test:2", "source": "test", "title": "What a great sunset today", "text": "Just sharing a photo.",
         "extra": "{}"},
    ]
    db.init()
    try:
        llm = factory.make_llm(provider, model, role="test")
        cards = classify_rows(llm, rows, 1500)
    except LLMError as e:
        typer.echo(_CONTROL.sub("", f"FAILED: {e}"), err=True)
        raise typer.Exit(1) from None
    for card in cards:
        typer.echo(card.model_dump_json(indent=2))


@app.command()
def purge() -> None:
    """Delete raw post text older than the retention window."""
    db.init()
    typer.echo(f"Purged {db.purge_expired(int(config.settings().get('retention_days', 30)))} items.")


if __name__ == "__main__":
    app()
