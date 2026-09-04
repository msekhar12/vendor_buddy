"""
Per-folder manifest.json management + content-hash duplicate detection.

Each vendor/date folder gets one manifest.json summarising what's inside.
Every uploaded document is also hashed so re-uploads of the same file
link to the existing entry instead of reindexing.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

from .storage import StorageDriver


def sha256_of(content: bytes) -> str:
    """Content-address hash — same file always hashes the same."""
    return hashlib.sha256(content).hexdigest()


def load_manifest(folder: Path) -> dict:
    """Read a folder's manifest, or return a fresh empty one."""
    mf = folder / "manifest.json"
    if mf.exists():
        return json.loads(mf.read_text())
    return {
        "vendor": folder.parent.name,
        "date":   folder.name,
        "docs":   [],
    }


def save_manifest(folder: Path, manifest: dict) -> None:
    """Persist a manifest back to disk (pretty-printed for human review)."""
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))


def find_duplicate(storage: StorageDriver, vendor: str,
                   file_hash: str) -> Path | None:
    """
    Scan every date-folder for this vendor. If any doc has the same content
    hash, return the existing path. Otherwise None (this is a new file).
    """
    for folder in storage.list_folders(vendor=vendor):
        mf = load_manifest(folder)
        for doc in mf["docs"]:
            if doc["sha256"] == file_hash:
                return folder / doc["filename"]
    return None


def append_doc(folder: Path, entry: dict) -> None:
    """Add a new doc entry to a folder's manifest and persist."""
    manifest = load_manifest(folder)
    manifest["docs"].append(entry)
    save_manifest(folder, manifest)