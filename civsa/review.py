"""
Upload assessment, vendor-level identity rules and the manual-review queue.

Every document's review state lives in its manifest entry:

  review_status         auto_ok | needs_review | approved | rejected
                        (entries written before this feature have no field
                        and count as auto_ok)
  review_reasons        current reason codes (see ocr_quality.REASON_TEXT)
  doc_reasons           reasons from the document on its own (OCR, label,
                        ID readability, name) as last assessed
  acknowledged_reasons  reasons a reviewer had in front of them when they
                        approved; only NEW reasons re-open an approved doc
  ack_group             for PAN/GST docs: the set of same-type documents the
                        reviewer saw; a different set re-opens review
  doc_type, extraction, identity

PAN/GST documents are mirrored into structured.vendor_identity_docs; they
are never indexed into Chroma / TF-IDF (questions about PAN/GSTIN are
answered by the SQL router from that table).

Vendor rules (evaluate_vendor), run after any change to a vendor's docs:
  * 2+ PAN documents  -> every PAN document of that vendor needs review
  * 2+ GST documents  -> every GST document of that vendor needs review
  * PAN != PAN inside a GSTIN (other ID document)  -> both need review
  * ID document disagrees with PAN/GSTIN on the vendor's quote -> the ID
    document needs review (the quote stays live)
A flag is never cleared automatically (not even by deleting the extra
document); only a reviewer decision clears it.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from . import structured
from .config import DOC_STORE
from .extract import extract_with_quality
from .identity import name_matches, pan_from_gstin, validate_gstin, validate_pan
from .manifest import load_manifest, save_manifest
from .ocr_quality import ID_DOC_TYPES, REASON_TEXT, assess_document

USABLE = {"auto_ok", "approved"}
_GROUP_REASON = {"pan": "multiple_pan_docs", "gstin": "multiple_gst_docs"}

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def vendor_lock(key: str) -> threading.RLock:
    """One lock per vendor: parallel uploads for the same vendor (the UI
    sends several files at once) are evaluated one at a time."""
    with _locks_guard:
        return _locks.setdefault(key.lower(), threading.RLock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(vendor: str) -> str:
    from .vendor_index import get_vendor_index
    return get_vendor_index().resolve(vendor).canonical or vendor


def sidecar_path(path: Path) -> Path:
    """Where OCR'd (or reviewer-corrected) text is kept next to the file."""
    return path.with_name(path.name + ".ocr.txt")


def messages(reasons: list[str]) -> list[str]:
    return [REASON_TEXT.get(r, r) for r in reasons]


# ------------------------------------------------------------------
# Manifest entry helpers
# ------------------------------------------------------------------

def _update_entry(path: Path, **fields) -> dict:
    folder = path.parent
    mf = load_manifest(folder)
    for d in mf["docs"]:
        if d["filename"] == path.name:
            d.update(fields)
            save_manifest(folder, mf)
            return d
    raise KeyError(f"{path.name} not in manifest of {folder}")


def get_entry(path: Path) -> dict | None:
    for d in load_manifest(path.parent)["docs"]:
        if d["filename"] == path.name:
            return d
    return None


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

def process_upload(path: Path, vendor: str, labels: list[str],
                   entry: dict, use_llm: bool = True) -> dict:
    """
    Extract + assess a newly stored file, write its manifest entry, register
    PAN/GST documents and re-run the vendor rules. Returns what the caller
    needs: review status/reasons, doc_type, whether to index, and the text
    to index (None = let the indexer extract natively).
    """
    res = extract_with_quality(path)
    a = assess_document(res, vendor, labels, use_llm=use_llm)

    text_file = None
    if res.method in ("ocr", "mixed"):
        sidecar_path(path).write_text(res.text)
        text_file = sidecar_path(path).name

    canonical = _canonical(vendor)
    status = "needs_review" if a.reasons else "auto_ok"
    entry = {
        **entry,
        "doc_type": a.doc_type,
        "review_status": status,
        "review_reasons": a.reasons,
        "doc_reasons": a.reasons,
        "acknowledged_reasons": [],
        "extraction": {"method": a.method, "ocr": a.ocr, "text_file": text_file},
    }
    is_id = a.doc_type in ID_DOC_TYPES
    if is_id:
        entry["identity"] = {"id_type": a.id_type, "id_value": a.id_value,
                             "holder_names": a.holder_names}

    with vendor_lock(canonical):
        mf = load_manifest(path.parent)
        mf["docs"].append(entry)
        save_manifest(path.parent, mf)
        if is_id:
            structured.upsert_identity_doc(
                vendor=vendor, canonical_vendor=canonical, source=str(path),
                doc_type=a.doc_type, id_type=a.id_type or "pan",
                id_value=a.id_value,
                holder_name=(a.holder_names or [None])[0],
                review_status=status)
            evaluate_vendor(canonical)
        final = get_entry(path) or entry

    return {
        "review_status": final["review_status"],
        "review_reasons": final["review_reasons"],
        "review_messages": messages(final["review_reasons"]),
        "doc_type": a.doc_type,
        "identity": entry.get("identity"),
        "ocr": a.ocr,
        # ID documents are never indexed for RAG; others only when clean
        "index": (not is_id) and final["review_status"] in USABLE,
        "text": res.text if text_file else None,
    }


# ------------------------------------------------------------------
# Vendor-level identity rules
# ------------------------------------------------------------------

def _id_pan(id_type: str, value: str | None) -> str | None:
    if not value:
        return None
    return value if id_type == "pan" else pan_from_gstin(value)


def identity_reasons(doc: dict, others: list[dict], quotes: list[dict]) -> set[str]:
    """
    Vendor-level reasons for one PAN/GST document. `others` = the vendor's
    other non-rejected ID documents; `quotes` = PAN/GSTIN printed on its
    other documents. Pure function (unit-tested).
    """
    reasons: set[str] = set()
    same = [o for o in others if o["id_type"] == doc["id_type"]]
    if same:
        reasons.add(_GROUP_REASON[doc["id_type"]])
    mine = _id_pan(doc["id_type"], doc.get("id_value"))
    if not mine:
        return reasons
    for o in others:
        if o["id_type"] != doc["id_type"]:
            theirs = _id_pan(o["id_type"], o.get("id_value"))
            if theirs and theirs != mine:
                reasons.add("pan_gstin_mismatch")
    for q in quotes:
        q_pans = set()
        if q.get("pan"):
            q_pans.add(q["pan"])
        if q.get("gstin") and len(q["gstin"]) == 15:
            q_pans.add(pan_from_gstin(q["gstin"]))
        if q_pans and mine not in q_pans:
            reasons.add("conflicts_with_quote")
    return reasons


def evaluate_vendor(canonical: str) -> list[dict]:
    """Re-apply the vendor rules to every non-rejected PAN/GST document of
    a vendor. Returns the documents whose status changed."""
    changed = []
    with vendor_lock(canonical):
        docs = [d for d in structured.identity_docs_for_vendor(canonical)
                if d["review_status"] != "rejected"]
        quotes = structured.quote_identity_values(canonical)
        for d in docs:
            path = Path(d["source_doc"])
            entry = get_entry(path)
            if entry is None:
                continue
            others = [o for o in docs if o["source_doc"] != d["source_doc"]]
            group = sorted([d["source_doc"]] + [o["source_doc"] for o in others
                                                 if o["id_type"] == d["id_type"]])
            reasons = set(entry.get("doc_reasons", [])) | identity_reasons(d, others, quotes)
            acknowledged = set(entry.get("acknowledged_reasons", []))
            new = reasons - acknowledged
            if (_GROUP_REASON[d["id_type"]] in reasons
                    and entry.get("ack_group") not in (None, group)):
                new.add(_GROUP_REASON[d["id_type"]])     # group changed since approval
            status = entry.get("review_status", "auto_ok")
            if status in USABLE and new:
                status = "needs_review"
            fields = {"review_reasons": sorted(reasons), "review_status": status}
            if status != entry.get("review_status") or sorted(reasons) != entry.get("review_reasons"):
                _update_entry(path, **fields)
                structured.set_identity_status(d["source_doc"], status)
                if status != entry.get("review_status"):
                    changed.append({"source": d["source_doc"], "status": status,
                                    "reasons": sorted(reasons)})
    return changed


def recheck_identity_names(canonical: str | None = None) -> list[dict]:
    """
    Re-run the holder-name check on stored PAN/GST documents (for one vendor,
    or all) with the current name_matches() and vendor name, then re-apply
    the vendor rules. Used after the matching rule changes, after a rename,
    and by reindex. Like any new reason, a new name_mismatch re-opens an
    auto-accepted document, but not one a reviewer approved with that
    mismatch in front of them; a name that now matches never auto-clears.
    """
    vendors = [canonical] if canonical else sorted(
        {d["canonical_vendor"] for v in structured.all_vendors()
         for d in structured.identity_docs_for_vendor(v)})
    changed = []
    for v in vendors:
        with vendor_lock(v):
            for d in structured.identity_docs_for_vendor(v):
                if d["review_status"] == "rejected":
                    continue
                path = Path(d["source_doc"])
                entry = get_entry(path)
                names = ((entry or {}).get("identity") or {}).get("holder_names") or []
                if not entry or not names:
                    continue
                own = set(entry.get("doc_reasons", []))
                if any(name_matches(n, v) for n in names):
                    own.discard("name_mismatch")
                else:
                    own.add("name_mismatch")
                if sorted(own) != sorted(entry.get("doc_reasons", [])):
                    _update_entry(path, doc_reasons=sorted(own))
            changed += evaluate_vendor(v)
    return changed


# ------------------------------------------------------------------
# Review queue
# ------------------------------------------------------------------

def _iter_entries(root: Path = DOC_STORE):
    if not root.exists():
        return
    for mf_path in sorted(root.glob("*/*/manifest.json")):
        folder = mf_path.parent
        for d in load_manifest(folder)["docs"]:
            yield folder, d


def review_queue() -> list[dict]:
    """Vendors with documents awaiting review, with every same-type ID
    document shown alongside a flagged one (so the reviewer can compare)."""
    by_vendor: dict[str, dict] = {}
    for folder, d in _iter_entries():
        status = d.get("review_status", "auto_ok")
        vendor = folder.parent.name
        rec = {
            "vendor": vendor, "date": folder.name, "filename": d["filename"],
            "labels": d.get("labels", []), "doc_type": d.get("doc_type"),
            "review_status": status, "review_reasons": d.get("review_reasons", []),
            "review_messages": messages(d.get("review_reasons", [])),
            "identity": d.get("identity"), "ocr": (d.get("extraction") or {}).get("ocr"),
            "uploaded_at": d.get("uploaded_at"),
        }
        v = by_vendor.setdefault(vendor, {"vendor": vendor, "flagged": [], "id_docs": []})
        if status == "needs_review":
            v["flagged"].append(rec)
        if d.get("doc_type") in ID_DOC_TYPES and status != "rejected":
            v["id_docs"].append(rec)
    return [v for v in by_vendor.values() if v["flagged"]]


def review_count() -> int:
    return sum(len(v["flagged"]) for v in review_queue())


# ------------------------------------------------------------------
# Reviewer decisions
# ------------------------------------------------------------------

class ReviewError(ValueError):
    pass


def _doc_path(vendor: str, date: str, filename: str) -> Path:
    """Path of a stored document, spelled exactly as at upload time
    (DOC_STORE/vendor/date/filename) because that string is the key in the
    indexes. resolve() is only used for the traversal check: it can change
    the spelling (macOS /var -> /private/var) and would create a second key."""
    p = DOC_STORE / vendor / date / filename
    if DOC_STORE.resolve() not in p.resolve().parents:
        raise ReviewError("invalid path")
    if not p.exists() or get_entry(p) is None:
        raise ReviewError("document not found")
    return p


def resolve(vendor: str, date: str, filename: str, action: str,
            text: str | None = None, id_value: str | None = None,
            reviewer: str = "anonymous") -> dict:
    """
    action:
      approve      accept as is (indexes a non-ID document with its OCR text)
      submit_text  replace the OCR text with the reviewer's text, then approve
      set_id       (PAN/GST docs) type the correct PAN/GSTIN, then approve
      reject       mark irrelevant / unusable; nothing from it is used
    Returns {"entry", "index": bool, "text"}; the caller schedules indexing.
    """
    path = _doc_path(vendor, date, filename)
    canonical = _canonical(vendor)
    with vendor_lock(canonical):
        entry = get_entry(path) or {}
        is_id = entry.get("doc_type") in ID_DOC_TYPES
        identity = dict(entry.get("identity") or {})
        stamp = {"reviewed_by": reviewer, "reviewed_at": _now()}

        if action == "reject":
            entry = _update_entry(path, review_status="rejected", **stamp)
            if is_id:
                structured.set_identity_status(str(path), "rejected")
            else:
                _unindex(path)
            evaluate_vendor(canonical)
            return {"entry": entry, "index": False, "text": None}

        if action == "submit_text":
            if not (text and text.strip()):
                raise ReviewError("text is required")
            sidecar_path(path).write_text(text)
            entry["extraction"] = {**(entry.get("extraction") or {}),
                                   "text_file": sidecar_path(path).name,
                                   "method": "manual"}
            if is_id:
                from .identity import find_ids
                found = find_ids(text)[identity.get("id_type") or "pan"]
                if not found:
                    raise ReviewError("no valid PAN/GSTIN found in the text")
                identity["id_value"] = found[0]

        if action == "set_id":
            if not is_id:
                raise ReviewError("set_id only applies to PAN/GST documents")
            v = (id_value or "").strip().upper()
            ok, why = (validate_pan(v) if identity.get("id_type") == "pan"
                       else validate_gstin(v))
            if not ok:
                raise ReviewError(f"invalid {identity.get('id_type')}: {why}")
            identity["id_value"] = v

        if action not in ("approve", "submit_text", "set_id"):
            raise ReviewError(f"unknown action {action!r}")

        if is_id and not identity.get("id_value"):
            raise ReviewError("no PAN/GSTIN on this document yet; use set_id")

        # Reviewer has seen the current reasons; the document's own problems
        # are considered resolved by their decision.
        reasons = entry.get("review_reasons", [])
        fields = {"review_status": "approved", "acknowledged_reasons": reasons,
                  "extraction": entry.get("extraction"), **stamp}
        if is_id:
            fields["identity"] = identity
            fields["doc_reasons"] = []
            fields["ack_group"] = _group_sources(canonical, identity["id_type"], str(path))
            structured.upsert_identity_doc(
                vendor=vendor, canonical_vendor=canonical, source=str(path),
                doc_type=entry["doc_type"], id_type=identity["id_type"],
                id_value=identity["id_value"],
                holder_name=(identity.get("holder_names") or [None])[0],
                review_status="approved")
        entry = _update_entry(path, **fields)
        if is_id:
            evaluate_vendor(canonical)
            entry = get_entry(path) or entry
            return {"entry": entry, "index": False, "text": None}

    sc = sidecar_path(path)
    return {"entry": entry, "index": True,
            "text": sc.read_text() if sc.exists() else None}


def _group_sources(canonical: str, id_type: str, include: str) -> list[str]:
    srcs = {d["source_doc"] for d in structured.identity_docs_for_vendor(canonical)
            if d["id_type"] == id_type and d["review_status"] != "rejected"}
    srcs.add(include)
    return sorted(srcs)


def resolve_group(vendor: str, id_type: str, action: str,
                  keep: list[tuple[str, str]] | None = None,
                  reviewer: str = "anonymous") -> list[dict]:
    """
    Decide a vendor's multiple-PAN / multiple-GST case in one step.
      keep_one     keep = [(date, filename)]: approve it, reject the others
      approve_all  (GST only) approve every GST document; allowed only when
                   all share the same PAN (one company, several state
                   registrations)
    """
    if id_type not in ("pan", "gstin"):
        raise ReviewError("id_type must be pan or gstin")
    canonical = _canonical(vendor)
    with vendor_lock(canonical):
        docs = [d for d in structured.identity_docs_for_vendor(canonical)
                if d["id_type"] == id_type and d["review_status"] != "rejected"]
        if len(docs) < 2:
            raise ReviewError("vendor does not have multiple documents of this type")
        out = []
        if action == "approve_all":
            if id_type != "gstin":
                raise ReviewError("approve_all is only allowed for GST documents")
            pans = {pan_from_gstin(d["id_value"]) for d in docs if d["id_value"]}
            if len(pans) != 1 or any(not d["id_value"] for d in docs):
                raise ReviewError("GSTINs do not share one PAN; keep one instead")
            for d in docs:
                p = Path(d["source_doc"])
                out.append(resolve(p.parent.parent.name, p.parent.name, p.name,
                                   "approve", reviewer=reviewer)["entry"])
            return out
        if action == "keep_one":
            if not keep or len(keep) != 1:
                raise ReviewError("keep_one needs exactly one (date, filename)")
            keep_src = str(_doc_path(vendor, keep[0][0], keep[0][1]))
            if keep_src not in {d["source_doc"] for d in docs}:
                raise ReviewError("kept document is not in this group")
            for d in docs:
                p = Path(d["source_doc"])
                if d["source_doc"] != keep_src:
                    out.append(resolve(p.parent.parent.name, p.parent.name, p.name,
                                       "reject", reviewer=reviewer)["entry"])
            p = Path(keep_src)
            out.append(resolve(p.parent.parent.name, p.parent.name, p.name,
                               "approve", reviewer=reviewer)["entry"])
            return out
        raise ReviewError(f"unknown action {action!r}")


def _unindex(path: Path) -> None:
    from . import tfidf_store, vector_store
    vector_store.remove_by_source(str(path))
    tfidf_store.remove_by_source(str(path))
    structured.remove_by_source(str(path))
