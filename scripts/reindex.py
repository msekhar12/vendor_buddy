"""
Wipe the vector store + TF-IDF store, then reindex every document already
on disk with the current chunker. Run after changing chunker/indexer logic.
"""

from __future__ import annotations

import json
from pathlib import Path

from civsa import structured, tfidf_store, vector_store
from civsa.api import STORAGE  # reuse the same instance
from civsa.indexer import index_document
from civsa.ocr_quality import ID_DOC_TYPES
from civsa.review import recheck_identity_names, sidecar_path
from civsa.vendor_index import get_vendor_index, refresh_vendor_index


def main() -> None:
    root = STORAGE.root
    if not root.exists():
        print(f"Storage root {root} does not exist — nothing to reindex.")
        return

    # 1) Wipe both stores
    print("Wiping vector store…")
    vector_store.reset()      # or your equivalent "empty everything" call
    print("Wiping TF-IDF store…")
    tfidf_store.reset()
    print("Wiping structured facts DB…")
    structured.reset()
    # 2) Walk vendor/date/filename and reindex each doc
    n_docs = n_chunks = 0
    identity_vendors: set[str] = set()
    refresh_vendor_index()
    for vendor_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        print(f"Reindexing vendor {vendor_dir.name}…")
        for date_dir in sorted(p for p in vendor_dir.iterdir() if p.is_dir()):
            print(f"  Reindexing date {date_dir.name}…")
            mf_path = date_dir / "manifest.json"
            if not mf_path.exists():
                continue
            mf = json.loads(mf_path.read_text(encoding="utf-8"))
            for doc in mf.get("docs", []):
                path = date_dir / doc["filename"]
                print(f"  Reindexing {vendor_dir.name}/{date_dir.name}/{doc['filename']}…")
                if not path.exists():
                    continue
                status = doc.get("review_status", "auto_ok")
                # PAN / GST documents: restore the identity row from the
                # manifest (no OCR, no RAG indexing)
                if doc.get("doc_type") in ID_DOC_TYPES:
                    ident = doc.get("identity") or {}
                    canonical = get_vendor_index().resolve(vendor_dir.name).canonical or vendor_dir.name
                    structured.upsert_identity_doc(
                        vendor=vendor_dir.name, canonical_vendor=canonical,
                        source=str(path), doc_type=doc["doc_type"],
                        id_type=ident.get("id_type") or "pan",
                        id_value=ident.get("id_value"),
                        holder_name=(ident.get("holder_names") or [None])[0],
                        review_status=status)
                    identity_vendors.add(canonical)
                    print(f"  ✓ identity document ({status})")
                    continue
                if status in ("needs_review", "rejected"):
                    print(f"  – skipped ({status})")
                    continue
                sc = sidecar_path(path)
                try:
                    added = index_document(
                        path=path,
                        vendor=vendor_dir.name,
                        date=date_dir.name,
                        labels=doc.get("labels", []),
                        text=sc.read_text() if sc.exists() else None,
                    )
                    n_docs += 1
                    n_chunks += added
                    print(f"  ✓ {vendor_dir.name}/{date_dir.name}/{doc['filename']} → {added} chunks")
                    if n_docs % 5 == 0:
                        print(f"  …reindexed {n_docs} docs so far…")
                except Exception as e:
                    print(f"  ✗ {vendor_dir.name}/{date_dir.name}/{doc['filename']}: {e}")

    for canonical in sorted(identity_vendors):
        recheck_identity_names(canonical)      # also runs evaluate_vendor
    refresh_vendor_index()
    print(f"\nDone. Reindexed {n_docs} docs into {n_chunks} chunks.")


if __name__ == "__main__":
    main()