"""
Rebuild TF-IDF and Chroma indexes from documents already on disk.
Reads each folder's manifest.json to preserve vendor/date/label metadata.

Usage:
  PYTHONPATH=. python scripts/reindex.py
"""
import shutil
import sys
from pathlib import Path

# ---- Locate paths BEFORE importing civsa (so we can delete indexes cleanly) ----
ROOT      = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "indexes"
DOC_STORE = ROOT / "documents"

def wipe_indexes():
    tfidf  = INDEX_DIR / "tfidf.pkl"
    chroma = INDEX_DIR / "chroma"
    if tfidf.exists():
        tfidf.unlink()
        print(f"  · deleted {tfidf}")
    if chroma.exists():
        shutil.rmtree(chroma)
        print(f"  · deleted {chroma}")

# ---- Wipe first, then import (fresh Chroma collection is created on import) ----
print("Wiping existing indexes...")
wipe_indexes()

sys.path.insert(0, str(ROOT))
from civsa.manifest import load_manifest       # noqa: E402
from civsa.indexer import index_document       # noqa: E402


def main():
    if not DOC_STORE.exists():
        print("No documents/ folder — nothing to index.")
        return

    vendors = sorted([p for p in DOC_STORE.iterdir() if p.is_dir()])
    if not vendors:
        print("No vendors under documents/ — nothing to index.")
        return

    print(f"\nFound {len(vendors)} vendor(s). Reindexing:\n")

    total = 0
    skipped = 0
    for vendor_dir in vendors:
        vendor = vendor_dir.name
        for date_dir in sorted(vendor_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            manifest = load_manifest(date_dir)
            for doc in manifest.get("docs", []):
                doc_path = date_dir / doc["filename"]
                if not doc_path.exists():
                    print(f"  ! MISSING FILE — skipping: {doc_path}")
                    skipped += 1
                    continue
                labels = doc.get("labels", [])
                print(f"  → {vendor}/{date}/{doc['filename']}   labels={labels}")
                index_document(doc_path, vendor, date, labels)
                total += 1

    print(f"\nDone. Reindexed {total} document(s), skipped {skipped}.")


if __name__ == "__main__":
    main()