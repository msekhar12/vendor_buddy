"""
ChromaDB persistent vector index.

Metadata carries vendor + date + labels + source so retrieval can pre-filter
cheaply before ANN search — this is what makes label-filtered and
TF-IDF-shortlist-filtered queries fast.
"""
from typing import Any, Mapping, Sequence, Union, cast

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL_PATH, INDEX_DIR

# Persistent client so vectors survive process restarts
_client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
_collection = _client.get_or_create_collection("civsa_chunks")
_encoder = SentenceTransformer(EMBED_MODEL_PATH)


def add_chunks(chunks: list[dict], meta: dict) -> None:
    if not chunks:
        return
    ids  = [f"{meta['source']}#{c['chunk_index']}" for c in chunks]
    docs = [c["text"] for c in chunks]

    metas: Sequence[Mapping[str, Union[str, int, float, bool]]] = []
    for c in chunks:
        m = {
            "vendor": meta["vendor"],
            "date":   meta["date"],
            "labels": "|".join(meta["labels"]),
            "source": meta["source"],
            "para":   int(c.get("para_index", 0)),
        }
        # Pass through any extra scalar fields on the chunk itself
        # (chunk_type, region_index, row_index, table_header, prose_index).
        # Chroma metadata must be scalar — anything non-scalar is skipped.
        for k, v in c.items():
            if k in ("text", "chunk_index", "para_index"):
                continue
            if isinstance(v, (str, int, float, bool)):
                m[k] = v
        metas.append(m)

    embeddings = np.asarray(
        _encoder.encode(docs, show_progress_bar=False, convert_to_numpy=True)
    ).tolist()
    _collection.upsert(
        ids=ids, documents=docs,
        embeddings=embeddings,
        metadatas=cast(Any, metas),
    )


def query(text: str, k: int = 5,
          label_filter: list[str] | None = None,
          source_filter: list[str] | None = None):
    """
    Semantic search with optional label and/or source pre-filters.
    Combining both narrows the search to chunks whose document is on the
    TF-IDF shortlist AND whose label is relevant to the intent.
    """
    conditions: list[dict] = []
    if label_filter:
        conditions.append(
            {"$or": [{"labels": {"$eq": lbl}} for lbl in label_filter]}
        )
    if source_filter:
        conditions.append({"source": {"$in": source_filter}})

    where: dict | None = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    return _collection.query(
        query_texts=[text], n_results=k,
        where=cast(Any, where),
    )


def remove_by_source(source: str) -> int:
    """Delete all chunks whose source == the given file path."""
    got = _collection.get(where={"source": {"$eq": source}})
    ids = got["ids"] if got else []
    if ids:
        _collection.delete(ids=ids)
    return len(ids)


def remove_by_vendor(vendor: str) -> int:
    """Delete all chunks belonging to a vendor."""
    got = _collection.get(where={"vendor": {"$eq": vendor}})
    ids = got["ids"] if got else []
    if ids:
        _collection.delete(ids=ids)
    return len(ids)

def reset() -> None:
    """Delete every vector in the collection. Used by scripts/reindex.py."""
    result = _collection.get()          # fetch all IDs
    ids = result.get("ids") or []
    if ids:
        _collection.delete(ids=ids)