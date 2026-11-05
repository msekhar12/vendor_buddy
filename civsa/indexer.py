"""
Post-upload indexing pipeline: extract → chunk → index into stores +
extract structured facts into SQLite.

Called as a FastAPI BackgroundTask so uploads return in ~200 ms; this
function runs after the HTTP response has been flushed to the browser.
Structured extraction adds ~1-3s of LLM latency to that background task.
"""
import threading
from pathlib import Path

from . import structured, tfidf_store, vector_store
from .chunker import chunk_document
from .extract import extract_text
from .llm_client import chat as llm_chat
from .vendor_index import get_vendor_index


# ---- In-flight tracking -------------------------------------------
# Indexing runs in the background for tens of seconds (LLM fact
# extraction) and writes rows keyed by the document path. Operations that
# change those paths (vendor rename) must not run meanwhile, so every
# scheduled index job is counted per vendor folder from the moment it is
# queued until it finishes.
_active: dict[str, int] = {}
_active_lock = threading.Lock()


def _folder_key(path: Path) -> str:
    return path.parent.parent.name.lower()


def mark_queued(path: Path) -> None:
    with _active_lock:
        k = _folder_key(path)
        _active[k] = _active.get(k, 0) + 1


def _mark_done(path: Path) -> None:
    with _active_lock:
        k = _folder_key(path)
        _active[k] = max(0, _active.get(k, 0) - 1)
        if not _active[k]:
            del _active[k]


def indexing_in_progress(vendor_folder: str) -> int:
    with _active_lock:
        return _active.get(vendor_folder.lower(), 0)


def index_document_tracked(path: Path, vendor: str, date: str, labels: list[str],
                           text: str | None = None) -> int:
    """index_document for jobs registered with mark_queued()."""
    try:
        return index_document(path, vendor, date, labels, text)
    finally:
        _mark_done(path)


def index_document(path: Path, vendor: str, date: str, labels: list[str],
                   text: str | None = None) -> int:
    """
    `text` overrides extraction: OCR text cached at upload, or text a
    reviewer corrected. PAN/GST documents are never passed here (they live
    in structured.vendor_identity_docs only).
    """
    if text is None:
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

    # A quote's PAN/GSTIN can conflict with an uploaded PAN/GST document;
    # re-check the vendor's identity documents now that its facts are stored.
    try:
        from .review import evaluate_vendor
        changed = evaluate_vendor(get_vendor_index().resolve(vendor).canonical or vendor)
        if changed:
            print(f"[indexer] identity review changes: {changed}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[indexer] identity re-check failed for {path.name}: {e}", flush=True)

    return len(chunks)