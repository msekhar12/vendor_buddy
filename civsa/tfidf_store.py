"""
On-disk TF-IDF index rebuilt on every add.

Fast enough for Phase 1's small corpus (rebuild time < 1 s per 1000 chunks).
Used as the cheap first-stage filter before the vector store re-ranks.
"""
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

from .config import INDEX_DIR

_stemmer = SnowballStemmer("english")

_STORE = INDEX_DIR / "tfidf.pkl"


def _stem_tokens(text):
    import re
    return [_stemmer.stem(t) for t in re.findall(r"\b\w{2,}\b", text.lower())]

def _load() -> dict:
    if _STORE.exists():
        return pickle.loads(_STORE.read_bytes())
    return {"chunks": [], "metas": []}


def _save(data: dict) -> None:
    _STORE.write_bytes(pickle.dumps(data))


def add_chunks(chunks: list[dict], meta: dict) -> None:
    data = _load()
    for c in chunks:
        data["chunks"].append(c["text"])
        data["metas"].append({**meta, "para": c["para_index"]})

    # Rebuild the vectorizer on the full corpus (small enough in Phase 1)
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                          tokenizer=_stem_tokens,
                          lowercase=True,
                          analyzer="word",
                          token_pattern=None,
                          stop_words="english",)         # drops "which", "can", "the" etc.
    matrix = vec.fit_transform(data["chunks"])
    data["vectorizer"] = vec
    data["matrix"] = matrix
    _save(data)


def query(text: str, k: int = 50) -> list[int]:
    """Return the top-k chunk indices by TF-IDF cosine."""
    data = _load()
    if not data["chunks"]:
        return []
    vec = data["vectorizer"]
    q = vec.transform([text])
    scores = (data["matrix"] @ q.T).toarray().ravel()
    top = scores.argsort()[::-1][:k]
    return [int(i) for i in top if scores[i] > 0]


def get_chunk(idx: int) -> tuple[str, dict]:
    data = _load()
    return data["chunks"][idx], data["metas"][idx]

def remove_by_source(source: str) -> int:
    """Drop chunks whose source == the given path. Rebuilds the vectorizer."""
    return _remove(lambda m: m.get("source") == source)


def remove_by_vendor(vendor: str) -> int:
    """Drop all chunks belonging to a vendor. Rebuilds the vectorizer."""
    return _remove(lambda m: m.get("vendor") == vendor)


def _remove(match) -> int:
    """Internal helper: drop matching chunks and rebuild the TF-IDF matrix."""
    data = _load()
    keep = [i for i, m in enumerate(data["metas"]) if not match(m)]
    removed = len(data["metas"]) - len(keep)
    if removed == 0:
        return 0
    data["chunks"] = [data["chunks"][i] for i in keep]
    data["metas"]  = [data["metas"][i]  for i in keep]
    if data["chunks"]:
        vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                                  tokenizer=_stem_tokens,
                                  lowercase=True,
                                  analyzer="word",
                                  token_pattern=None,
                                  stop_words="english",)         # drops "which", "can", "the" etc.
        data["vectorizer"] = vec
        data["matrix"] = vec.fit_transform(data["chunks"])
    else:
        data.pop("vectorizer", None)
        data.pop("matrix", None)
    _save(data)
    return removed