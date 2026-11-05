"""
Vendor-level maintenance operations.

rename_vendor(old, new) renames a vendor everywhere it is stored. The vendor
folder name is part of every document path, and that path is the key in all
indexes, so a rename has to rewrite:

  documents/<old>/                -> documents/<new>/   (files, manifests, .ocr.txt)
  manifest.json                   vendor field + ack_group paths
  Chroma                          ids (<source>#<n>) and source/vendor metadata;
                                  embeddings are copied, not recomputed
  TF-IDF                          source/vendor metadata (matrix unchanged)
  SQLite                          source_doc / vendor / canonical_vendor in
                                  vendor_facts, quote_items, commercial_terms,
                                  vendor_identity_docs
  documents/aliases.json          aliases of the old name now point to the new
                                  one, and the old name itself becomes an alias,
                                  so questions using the old name still resolve

Renaming onto an existing vendor is refused (that would be a merge).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import structured, tfidf_store, vector_store
from .config import DOC_STORE
from .manifest import load_manifest, save_manifest
from .review import vendor_lock
from .storage import _safe
from .vendor_index import refresh_vendor_index


class RenameError(ValueError):
    status = 400


class RenameConflict(RenameError):
    status = 409


class RenameNotFound(RenameError):
    status = 404


def _validate(old: str, new: str) -> str:
    new = " ".join((new or "").split())          # trim + collapse spaces
    if not new:
        raise RenameError("New name is empty.")
    if len(new) > 100:
        raise RenameError("New name is longer than 100 characters.")
    try:
        if _safe(new) != new:
            raise RenameError("Name cannot contain '/' or '\\', or start/end with '.'.")
    except ValueError as e:
        raise RenameError("Name cannot contain '/' or '\\', or start/end with '.'.") from e
    if new == old:
        raise RenameError("New name is the same as the current name.")
    existing = {p.name.lower(): p.name for p in DOC_STORE.iterdir() if p.is_dir()}
    clash = existing.get(new.lower())
    if clash and clash != old:
        raise RenameConflict(f'A vendor called "{clash}" already exists. '
                             "Renaming onto it would merge two vendors, which is not supported.")
    return new


def _swap(path: str, old_prefix: str, new_prefix: str) -> str:
    return new_prefix + path[len(old_prefix):] if path.startswith(old_prefix) else path


def rename_vendor(old: str, new: str) -> dict:
    old_dir = DOC_STORE / old
    if not old_dir.is_dir():
        raise RenameNotFound(f'Vendor "{old}" not found.')
    new = _validate(old, new)
    new_dir = DOC_STORE / new
    old_prefix, new_prefix = str(old_dir) + os.sep, str(new_dir) + os.sep

    with vendor_lock(old), vendor_lock(new):
        from .indexer import indexing_in_progress
        busy = indexing_in_progress(old)
        if busy:
            raise RenameConflict(
                f'{busy} document(s) of "{old}" are still being indexed (text '
                "extraction and fact extraction can take a minute). Try again shortly.")
        # 1. Folder (also handles case-only renames on case-insensitive disks)
        os.rename(old_dir, new_dir)
        summary = {"old": old, "new": new}
        try:
            summary.update(_rewrite_indexes(old, new, old_prefix, new_prefix))
        except Exception as e:
            raise RenameError(
                f"The folder was renamed to \"{new}\", but updating the indexes failed "
                f"({e}). Run `make reindex` to rebuild them.") from e
        refresh_vendor_index()
        # The name on PAN/GST documents is checked against the vendor name
        from .review import recheck_identity_names
        summary["review_changes"] = recheck_identity_names(new)
    return summary


def _rewrite_indexes(old: str, new: str, old_prefix: str, new_prefix: str) -> dict:
    out = {}

    # 2. Manifests
    n_docs = 0
    for mf_path in sorted((DOC_STORE / new).glob("*/manifest.json")):
        folder = mf_path.parent
        mf = load_manifest(folder)
        mf["vendor"] = new
        for d in mf.get("docs", []):
            n_docs += 1
            if d.get("ack_group"):
                d["ack_group"] = [_swap(p, old_prefix, new_prefix) for p in d["ack_group"]]
        save_manifest(folder, mf)
    out["documents"] = n_docs

    # 3. SQLite facts + identity documents
    n_rows = 0
    with structured._connect() as conn:
        for table in ("vendor_facts", "quote_items", "commercial_terms",
                      "vendor_identity_docs"):
            cur = conn.execute(
                f"UPDATE {table} SET source_doc = ? || substr(source_doc, ?), "
                f"vendor = ?, canonical_vendor = ? "
                f"WHERE substr(source_doc, 1, ?) = ? OR canonical_vendor = ?",
                (new_prefix, len(old_prefix) + 1, new, new,
                 len(old_prefix), old_prefix, old))
            n_rows += cur.rowcount
    out["sql_rows"] = n_rows

    # 4. TF-IDF metadata (vectors are unchanged)
    tf = tfidf_store._load()
    n_tf = 0
    for m in tf.get("metas", []):
        if str(m.get("source", "")).startswith(old_prefix):
            m["source"] = _swap(m["source"], old_prefix, new_prefix)
            m["vendor"] = new
            n_tf += 1
    if n_tf:
        tfidf_store._save(tf)
    out["tfidf_chunks"] = n_tf

    # 5. Chroma: ids embed the source path, so copy each chunk (with its
    #    embedding) to a new id and delete the old one.
    coll = vector_store._collection
    allm = coll.get(include=["metadatas"])
    ids = [i for i, m in zip(allm["ids"], allm["metadatas"] or [])
           if str((m or {}).get("source", "")).startswith(old_prefix)]
    if ids:
        got = coll.get(ids=ids, include=["embeddings", "documents", "metadatas"])
        new_ids, metas = [], []
        for cid, m in zip(got["ids"], got["metadatas"]):
            src = _swap(m["source"], old_prefix, new_prefix)
            new_ids.append(src + cid[len(m["source"]):])     # keep the "#<n>" suffix
            metas.append({**m, "source": src, "vendor": new})
        coll.add(ids=new_ids, embeddings=got["embeddings"],
                 documents=got["documents"], metadatas=metas)
        coll.delete(ids=got["ids"])
    out["chroma_chunks"] = len(ids)

    # 6. Aliases: keep the old name (and its aliases) resolving to the vendor
    alias_file = DOC_STORE / "aliases.json"
    data = {}
    if alias_file.exists():
        try:
            data = json.loads(alias_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    data = {a: (new if c == old else c) for a, c in data.items()}
    if old.lower() != new.lower():
        data[old] = new
    data.pop(new, None)                     # an alias equal to the name is pointless
    alias_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    out["alias_added"] = old if old.lower() != new.lower() else None
    return out
