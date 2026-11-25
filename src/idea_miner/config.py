"""Configuration loading: YAML files in config/, secrets from .env / environment."""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .util import slugify

ROOT = Path(os.environ.get("IDEA_MINER_HOME") or Path(__file__).resolve().parents[2])
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env")


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def settings() -> dict[str, Any]:
    return _load("settings.yaml")


@lru_cache
def niches() -> list[dict[str, Any]]:
    return _load("niches.yaml").get("niches", [])


@lru_cache
def sources_registry() -> dict[str, dict[str, Any]]:
    return _load("sources.yaml").get("sources", {})


@lru_cache
def prefilter_rules() -> dict[str, Any]:
    return _load("prefilter.yaml")


def prompt(name: str) -> str:
    """System prompt from config/prompts.yaml. Read on every call (the file is tiny), so edits take effect
    on the next run without restarting the app."""
    prompts = _load("prompts.yaml")
    text = prompts.get(name)
    if not isinstance(text, str) or not text.strip():
        raise KeyError(f"config/prompts.yaml has no prompt named '{name}'")
    return text.strip()


def env(name: str | None, default: str | None = None) -> str | None:
    if not name:
        return default
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def data_dir() -> Path:
    d = ROOT / settings().get("data_dir", "data")
    d.mkdir(parents=True, exist_ok=True)
    return d


def specs_dir() -> Path:
    d = ROOT / settings().get("specs_dir", "specs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def author_salt() -> str:
    salt = env("IDEA_MINER_SALT")
    if salt:
        return salt
    path = data_dir() / ".salt"
    if not path.exists():
        path.write_text(secrets.token_hex(16), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def niche_by_id(niche_id: str) -> dict[str, Any] | None:
    for n in niches():
        if n["id"] == niche_id:
            return n
    return None


def custom_niche(keywords: list[str]) -> dict[str, Any]:
    kws = [k.strip() for k in keywords if k and k.strip()]
    return {
        "id": "custom:" + slugify(" ".join(kws), 40),
        "label": "Custom: " + ", ".join(kws),
        "keywords": kws,
        "github_keywords": kws,
        "app_store_terms": kws,
        "social_keywords": kws,
        "youtube_terms": [f"how to {k}" for k in kws],
        "stackexchange_sites": ["softwarerecs", "webapps", "superuser"],
    }


def niche_label(niche_id: str) -> str:
    n = niche_by_id(niche_id)
    if n:
        return n["label"]
    if niche_id.startswith("custom:"):
        return "Custom: " + niche_id[len("custom:") :].replace("-", " ")
    return niche_id
