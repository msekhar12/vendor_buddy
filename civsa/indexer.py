"""
Post-upload indexing pipeline: extract → chunk → index into both stores.
Called as a FastAPI BackgroundTask so uploads return in ~200 ms; this
function runs after the HTTP response has been flushed to the browser.
"""
from pathlib import Path

from . import tfidf_store, vector_store
from .chunker import chunk_paragraphs
from .extract import extract_text


def index_document(path: Path, vendor: str, date: str,
                   labels: list[str]) -> None:
    print(f"[index] {vendor}/{date}/{path.name} → extract", flush=True)
    text = extract_text(path)
    if not text.strip():
        print(f"[index] {path.name} — no text extracted, skipping", flush=True)
        return

    print(f"[index] {path.name} → chunk", flush=True)
    chunks = chunk_paragraphs(text)

    meta = {"vendor": vendor, "date": date, "labels": labels,
            "source": str(path)}

    print(f"[index] {path.name} → tfidf ({len(chunks)} chunks)", flush=True)
    tfidf_store.add_chunks(chunks, meta)

    print(f"[index] {path.name} → chroma ({len(chunks)} chunks)", flush=True)
    vector_store.add_chunks(chunks, meta)

    print(f"[index] {path.name} DONE", flush=True)