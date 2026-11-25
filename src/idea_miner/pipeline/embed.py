"""Embeddings for clustering. Local only (no API cost): sentence-transformers when installed
(`pip install idea-miner[embeddings]`), otherwise TF-IDF + SVD from scikit-learn."""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import config
from ..core import db

_models: dict[str, Any] = {}


def _cfg() -> dict[str, Any]:
    return config.settings().get("embeddings", {})


def backend() -> str:
    pref = _cfg().get("backend", "auto")
    if pref in ("auto", "sentence-transformers"):
        try:
            import sentence_transformers  # noqa: F401
            return "sentence-transformers"
        except ImportError:
            pass
    return "tfidf"


def embed(ids: list[str], texts: list[str]) -> tuple[np.ndarray, str]:
    if backend() == "sentence-transformers":
        name = _cfg().get("model", "paraphrase-multilingual-MiniLM-L12-v2")
        return _embed_st(ids, texts, name), f"sentence-transformers ({name})"
    return _embed_tfidf(texts), "TF-IDF"


def _embed_st(ids: list[str], texts: list[str], name: str) -> np.ndarray:
    with db.session() as conn:
        cached = db.get_embeddings(conn, ids, name)
    missing = [i for i, item_id in enumerate(ids) if item_id not in cached]
    if missing:
        from sentence_transformers import SentenceTransformer

        model = _models.get(name) or SentenceTransformer(name)
        _models[name] = model
        vecs = model.encode([texts[i] for i in missing], normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False)
        new = {ids[i]: np.asarray(vecs[k], dtype=np.float32) for k, i in enumerate(missing)}
        with db.session() as conn:
            db.put_embeddings(conn, new, name)
        cached.update(new)
    return np.vstack([cached[i] for i in ids]).astype(np.float64)


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    try:
        X = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit_transform(texts)
    except ValueError:  # empty vocabulary
        return np.eye(len(texts))
    n, f = X.shape
    if n >= 10 and f > 50:
        X = TruncatedSVD(n_components=max(2, min(100, n - 1, f - 1)), random_state=42).fit_transform(X)
    else:
        X = X.toarray()
    return normalize(X)
