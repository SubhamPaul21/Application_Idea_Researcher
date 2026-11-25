"""Data-minimisation helpers: PII scrubbing, author hashing, HTML cleanup."""
from __future__ import annotations

import hashlib
import html
import re

from .. import config

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"https?://(?:www\.)?([^\s/<>\"')\]]+)[^\s<>\"')\]]*", re.I)
_HANDLE = re.compile(r"(?<![\w.@\[])@[A-Za-z0-9_](?:[A-Za-z0-9_.-]{0,38}[A-Za-z0-9_])?")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
# Runs of 10-19 digits with optional separators: phone numbers, Aadhaar, card numbers.
_LONG_NUMBER = re.compile(r"(?<![\w])\+?\d[\d\s().-]{8,24}\d(?![\w])")


def _number_repl(m: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[number]" if 10 <= len(digits) <= 19 else m.group(0)


def scrub(text: str | None) -> str:
    """Remove personal data before storage and before any LLM call."""
    if not text:
        return ""
    t = _EMAIL.sub("[email]", text)
    t = _URL.sub(lambda m: f"[link:{m.group(1).lower()}]", t)
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
