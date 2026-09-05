"""
Post-upload indexing pipeline: extract → chunk → index into stores +
extract structured facts into SQLite.

Called as a FastAPI BackgroundTask so uploads return in ~200 ms; this
function runs after the HTTP response has been flushed to the browser.
Structured extraction adds ~1-3s of LLM latency to that background task.
"""
from pathlib import Path

from . import structured, tfidf_store, vector_store
from .chunker import chunk_document
from .extract import extract_text
from .llm_client import chat as llm_chat
from .vendor_index import get_vendor_index


def index_document(path: Path, vendor: str, date: str, labels: list[str]) -> int:
    text = extract_text(path)
    chunks = chunk_document(text)
    meta = {
        "vendor": vendor,
        "date":   date,
        "labels": labels,
        "source": str(path),
    }
    vector_store.add_chunks(chunks, meta)
    tfidf_store.add_chunks(chunks, meta)

    # -------- Phase 2 Step 3: structured facts extraction --------
    # Never let extraction failure break indexing — the retrieval path
    # still works without it.
    try:
        resolved = get_vendor_index().resolve(vendor)
        canonical = resolved.canonical or vendor
        facts = structured.extract_facts(
            text,
            vendor=vendor,
            source=str(path),
            llm_chat_fn=llm_chat,
        )
        if facts:
            summary = structured.store_extraction(
                vendor=vendor,
                canonical_vendor=canonical,
                source=str(path),
                facts=facts,
            )
            print(f"[indexer] structured extraction for {path.name}: {summary}",
                  flush=True)
        else:
            print(f"[indexer] structured extraction returned nothing for "
                  f"{path.name}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[indexer] structured extraction failed for {path.name}: {e}",
              flush=True)

    return len(chunks)