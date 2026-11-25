"""Data-minimisation helpers: PII scrubbing, author hashing, HTML cleanup."""
from __future__ import annotations

import hashlib
import html
import re

from .. import config

# The lookbehind makes a match start only at the beginning of a run, and the possessive local part never
# backtracks, so matching time grows linearly with text length (the old pattern was quadratic).
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"https?://(?:www\.)?([^\s/<>\"')\]]+)[^\s<>\"')\]]*", re.I)
# Links written without http(s)://: any www. address, and common profile hosts with or without a path.
_PROFILE_HOSTS = (r"(?:linkedin\.com|github\.com|twitter\.com|x\.com|facebook\.com|fb\.com|instagram\.com|"
                  r"youtube\.com|bsky\.app|medium\.com|tiktok\.com|reddit\.com|threads\.net|t\.me)")
_BARE_URL = re.compile(
    rf"(?<![\w.:/@\[-])(?:www\.([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)|((?:[a-z]{{1,3}}\.)?{_PROFILE_HOSTS}))"
    r"(?:/[^\s<>\"')\]]*)?",
    re.I,
)
_HANDLE = re.compile(r"(?<![\w.@\[])@[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,38}[A-Za-z0-9_])?")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
# Any run of digits with separators (phone numbers, Aadhaar, card numbers, several numbers in a row). Runs with
# 10 or more digits are removed; digits touching letters (e.g. "ph9876543210") count too.
_LONG_NUMBER = re.compile(r"(?<!\d)\+?\d[\d\s().-]{8,}\d(?!\d)")
_DATE_PREFIX = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T]\d{1,2})?$")


def _number_repl(m: re.Match[str]) -> str:
    s = m.group(0)
    digits = re.sub(r"\D", "", s)
    if len(digits) < 10 or _DATE_PREFIX.match(s.strip()):
        return s
    return "[number]"


def scrub(text: str | None) -> str:
    """Remove personal data before storage and before any LLM call."""
    if not text:
        return ""
    t = _EMAIL.sub("[email]", text)
    t = _URL.sub(lambda m: f"[link:{m.group(1).lower()}]", t)
    t = _BARE_URL.sub(lambda m: f"[link:{(m.group(1) or m.group(2)).lower()}]", t)
    t = _HANDLE.sub("[user]", t)
    t = _PAN.sub("[id]", t)
    t = _LONG_NUMBER.sub(_number_repl, t)
    return t


def author_hash(source: str, author: str | None) -> str:
    if not author:
        return ""
    raw = f"{config.author_salt()}|{source}|{author.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def text_hash(text: str | None) -> str:
    norm = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16] if norm else ""


_BLOCK_TAGS = re.compile(r"(?i)<\s*(br|/p|p|/li|/div|/pre|/h\d)\b[^>]*>")
_TAG = re.compile(r"<[^>]+>")


def html_to_text(s: str | None) -> str:
    if not s:
        return ""
    s = _BLOCK_TAGS.sub("\n", s)
    s = _TAG.sub("", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()
