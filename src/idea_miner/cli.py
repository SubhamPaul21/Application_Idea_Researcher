"""Command line:  idea-miner --help"""
from __future__ import annotations

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


@app.command()
def ui() -> None:
    """Launch the Streamlit app."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(config.ROOT / "app" / "streamlit_app.py")],
                   check=False)


@app.command()
def smoke() -> None:
    """Fetch a few live items from every source to prove the endpoints work today."""
    for sid in registry.implemented_ids():
        ok, msg = registry.smoke(sid)
        typer.echo(f"{'OK  ' if ok else 'FAIL'} {sid:<14} {msg}")


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
    db.init()
    defaults = config.settings()["llm"]["defaults"]["classifier"]
    prov = provider or defaults["provider"]
    mdl = model or (defaults["model"] if prov == defaults["provider"] else factory.default_model(prov))
    llm = factory.make_llm(prov, mdl, role="classifier")
    params = runner.RunParams(
        niche_ids=niche, custom_keywords=keyword,
        source_ids=source or [s for s in registry.implemented_ids() if registry.get(s).available()[0]],
        days=days, region=region, max_to_classify=max_classify,
    )
    if not params.niches():
        raise typer.BadParameter("give at least one --niche or --keyword")
    stats = runner.run_all(params, llm, llm, typer.echo, stages=tuple(s.strip() for s in stages.split(",")))
    typer.echo(f"Done: {stats}")


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
        typer.echo(f"FAILED: {e}", err=True)
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
