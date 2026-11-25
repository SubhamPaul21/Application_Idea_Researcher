"""Helpers that make untrusted text (posts, model output, search results) safe to display, export or save.

- one_line / cap:      collapse whitespace and bound length for single-line fields
- md_escape:           show text literally inside st.markdown (no links, images, headings, colours, maths)
- md_link:             a link only for http(s) URLs, with characters that would break Markdown encoded
- clean_markdown:      keep normal Markdown but remove images, hidden constructs and non-http(s) links
- neutralize_formulas: stop spreadsheet apps from evaluating exported cells
"""
from __future__ import annotations

import re
from typing import Any

_WS = re.compile(r"\s+")
# Every ASCII punctuation character CommonMark lets us backslash-escape, plus ':' (Streamlit colour/emoji
# directives) and '$' (Streamlit maths).
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:=&\"'^])")


def one_line(text: Any, limit: int | None = None) -> str:
    s = _WS.sub(" ", str(text or "")).strip()
    return cap(s, limit) if limit else s


def cap(text: Any, limit: int) -> str:
    s = str(text or "")
    return s if len(s) <= limit else s[: max(0, limit - 1)].rstrip() + "…"


def md_escape(text: Any, limit: int | None = None) -> str:
    """Render text literally in st.markdown: one line, every Markdown control character escaped."""
    return _MD_SPECIAL.sub(r"\\\1", one_line(text, limit))


def safe_url(url: Any) -> str | None:
    """Return the URL with Markdown-breaking characters encoded, or None unless it is an absolute http(s) URL."""
    u = str(url or "").strip()
    if not re.match(r"^https?://[^\s/]+", u, re.I):
        return None
    for ch, enc in (("(", "%28"), (")", "%29"), (" ", "%20"), ("<", "%3C"), (">", "%3E"), ("[", "%5B"),
                    ("]", "%5D"), ('"', "%22"), ("\n", ""), ("\r", "")):
        u = u.replace(ch, enc)
    return u


def md_link(label: Any, url: Any) -> str:
    text = md_escape(label) or "link"
    u = safe_url(url)
    return f"[{text}]({u})" if u else text


_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|$)", re.S)
_LINK_REF_DEF = re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*\S.*$", re.M)
_IMAGE = re.compile(r"!\[([^\]\n]*)\]\([^)\n]*\)")
_IMAGE_REF = re.compile(r"!\[([^\]\n]*)\]\[[^\]\n]*\]")
_LINK = re.compile(r"(?<!!)\[([^\]\n]*)\]\(\s*<?([^)\s>]*)>?(?:\s+[^)\n]*)?\)")
_RAW_IMG_TAG = re.compile(r"<img\b[^>]*>", re.I)


def clean_markdown(md: Any) -> str:
    """Remove constructs that are invisible when rendered or that load remote content: HTML comments, link
    reference definitions, images (kept as their alt text), raw <img> tags, and links whose target is not an
    absolute http(s) URL (kept as plain text)."""
    s = str(md or "")
    s = _HTML_COMMENT.sub("", s)
    s = _LINK_REF_DEF.sub("", s)
    s = _IMAGE.sub(lambda m: m.group(1), s)
    s = _IMAGE_REF.sub(lambda m: m.group(1), s)
    s = _RAW_IMG_TAG.sub("", s)

    def _link(m: re.Match[str]) -> str:
        u = safe_url(m.group(2))
        return f"[{m.group(1)}]({u})" if u else m.group(1)

    s = _LINK.sub(_link, s)
    return re.sub(r"\n{3,}", "\n\n", s)


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def neutralize_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


def neutralize_formulas(df: Any) -> Any:
    """Return a copy of a DataFrame whose text cells cannot be evaluated as spreadsheet formulas."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("str"):
            out[col] = out[col].map(neutralize_cell)
    return out
