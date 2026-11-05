"""
CIVSA FastAPI backend.

Endpoints:
  GET  /                                       - web UI
  POST /api/upload                             - upload a vendor document
  POST /api/suggest-label                      - LLM label suggestion
  POST /api/query                              - gated Q&A
  GET  /api/vendors                            - list vendor folders
  GET  /api/vendor/{vendor}/manifest           - vendor's manifests
  PATCH /api/vendor/{vendor}/doc/labels        - update labels on a doc
  DELETE /api/vendor/{vendor}                  - delete a vendor
  DELETE /api/vendor/{vendor}/doc              - delete a single doc
  GET  /api/debug/stats                        - corpus stats
  POST /api/debug/query                        - probe TF-IDF + Chroma
  POST /api/debug/gate                         - probe the gate stages
"""
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import review, tfidf_store, vector_store, vendor_score
from .config import ALLOWED_LABELS, DOC_STORE, MAX_UPLOAD_MB, SQLITE_PATH
from .extract import SUPPORTED_EXTS, extract_text
from .gate.pipeline import process as gate
from .indexer import index_document_tracked, mark_queued
from .manifest import (
    find_duplicate,
    load_manifest,
    save_manifest,
    sha256_of,
)
from .qa import answer as qa_answer
from .storage import LocalDriver
from .vendor_index import get_vendor_index, refresh_vendor_index

# ================================================================
# App + storage
# ================================================================
app = FastAPI(title="CIVSA")
STORAGE = LocalDriver(root=DOC_STORE)

from . import structured

structured.init_db(SQLITE_PATH)

def queue_index(background: BackgroundTasks, path: Path, vendor: str, date: str,
                labels: list[str], text: str | None) -> None:
    """Schedule background indexing, counted as in-flight from now on
    (vendor rename refuses to run while a vendor has jobs in flight)."""
    mark_queued(path)
    background.add_task(index_document_tracked, path, vendor, date, labels, text)


@app.get("/api/labels")
def list_labels():
    return {"labels": sorted(ALLOWED_LABELS)}
# ================================================================
# Upload
# ================================================================
@app.post("/api/upload")
async def upload(
    background: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    vendor: Annotated[str, Form(min_length=1, max_length=100)],
    labels: Annotated[list[str], Form()],
    uploaded_by: Annotated[str, Form()] = "anonymous",
):
    if not file.filename:
        raise HTTPException(400, "Uploaded file must have a filename.")
    filename = file.filename
    if Path(filename).suffix.lower() not in SUPPORTED_EXTS:
        raise HTTPException(400, f"Unsupported file type. Allowed: "
                                 f"{', '.join(sorted(SUPPORTED_EXTS))}")

    bad = [l for l in labels
           if l not in ALLOWED_LABELS and not l.startswith("other:")]
    if bad:
        raise HTTPException(400, f"Unknown labels: {bad}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Exceeds {MAX_UPLOAD_MB} MB")

    file_hash = sha256_of(content)
    dup = find_duplicate(STORAGE, vendor, file_hash)
    if dup:
        prev = load_manifest(dup.parent)
        rejected = any(d["filename"] == dup.name
                       and d.get("review_status") == "rejected"
                       for d in prev["docs"])
        return {"status": "duplicate", "linked_to": str(dup),
                "hash": file_hash, "rejected": rejected}

    date = datetime.now().astimezone().strftime("%Y-%m-%d")
    path = STORAGE.put(vendor, date, filename, content)
    refresh_vendor_index()          # new vendor folders must resolve below

    # Extraction (OCR for images / scanned PDFs) + quality, relevance and
    # PAN/GSTIN checks run now, so the upload card can show a flag at once.
    # This also writes the manifest entry.
    result = await run_in_threadpool(review.process_upload, path, vendor, labels, {
        "filename":     filename,
        "labels":       labels,
        "label_source": "user",
        "sha256":       file_hash,
        "size_bytes":   len(content),
        "uploaded_by":  uploaded_by,
        "uploaded_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
    })
    if result["index"]:
        queue_index(background, path, vendor, date, labels, result["text"])
    return {"status": "stored", "path": str(path), "hash": file_hash,
            "labels": labels,
            # where it was stored, so the UI can re-check its review status
            "vendor_folder":   path.parent.parent.name,
            "date":            date,
            "filename":        path.name,
            "review_status":   result["review_status"],
            "review_reasons":  result["review_reasons"],
            "review_messages": result["review_messages"],
            "doc_type":        result["doc_type"],
            "identity":        result["identity"],
            "ocr":             result["ocr"]}


# ================================================================
# Auto-suggest labels
# ================================================================
# ================================================================
# Label hint dictionary and few-shot for the suggest-label prompt.
# Extend _LABEL_HINTS as your vocabulary grows.
# ================================================================

_LABEL_HINTS = {
    "quote":           "vendor's priced offer with SKUs, unit prices, quantities, and validity",
    "quotation":       "same as quote",
    "invoice":         "bill for goods/services already delivered, with GST and total due",
    "purchase-order":  "buyer's formal order to a vendor listing items and prices",
    "rfq":             "buyer's request for quotation soliciting prices from vendors",
    "contract":        "signed multi-page agreement with legal clauses",
    "amendment":       "modification or addendum to an existing contract or order",
    "coa":             "certificate of analysis with test results and specifications",
    "msds":            "material safety data sheet with hazard information",
    "safety-data":     "safety data sheet or hazard-handling document",
    "iso-certificate": "generic ISO certification document",
    "iso-9001":        "specifically ISO 9001 quality management certificate",
    "iso-14001":       "specifically ISO 14001 environmental management certificate",
    "iso-17025":       "specifically ISO/IEC 17025 laboratory certificate",
    "compliance":      "compliance statement, regulatory declaration, or audit report",
    "vendor-profile":  "company overview or capability statement",
    "brochure":        "product marketing material with descriptions and images",
    "catalog":         "product catalog listing many items with codes",
    "pricing":         "generic price list or rate card (not a specific quote to a buyer)",
    "delivery-terms":  "delivery schedule, lead times, shipping conditions",
    "payment-terms":   "payment schedule, credit terms, invoicing conditions",
    "warranty":        "warranty terms and product guarantees",
    "specifications":  "technical specifications or datasheets",
    "packaging":       "packaging requirements or descriptions",
    "gst":             "GST registration or tax document",
    "certificate":     "generic certificate not covered by a more specific label",
    "chemical":        "chemical product datasheet or specification",
    "reagent":         "reagent product information",
    "solvent":         "solvent product information",
    "unit-price":      "single-item price sheet",
    "quantity":        "quantity list or bill of quantities",
    "other":           "does not fit ANY label above — LAST RESORT ONLY",
}

_FEW_SHOT = """
EXAMPLES OF CORRECT CLASSIFICATION:

Example 1 —
Filename: kaveri_scientific_quote.pdf
Excerpt starts: "Kaveri Scientific Supplies · QUOTATION · Quotation No. KSS/QT/26-27/198 · Priced Schedule of Items · Sodium Hydroxide pellets AR grade 500g — INR 1050/bottle · Payment: 45 days net · Delivery: 5-7 working days"
Label: quote

Example 2 —
Filename: acme_invoice_00423.pdf
Excerpt starts: "TAX INVOICE · Invoice No. INV/2026/00423 · Date: 15 Aug 2026 · To: IIIT Dharwad · Total: INR 45,600 · GST @ 18%: INR 6,955 · Amount Due"
Label: invoice

Example 3 —
Filename: bureau_veritas_certificate.pdf
Excerpt starts: "Certificate of Registration · Bureau Veritas certifies that Gangotri Reagents Pvt Ltd operates a Quality Management System which complies with ISO 9001:2015 · valid until 30 June 2027"
Label: iso-9001
""".strip()


def _build_label_menu(labels: list[str]) -> str:
    """Render 'label: hint' bullets. 'other' is always the last line."""
    rest = sorted(l for l in labels if l != "other")
    ordered = rest + (["other"] if "other" in labels else [])
    out = []
    for lbl in ordered:
        hint = _LABEL_HINTS.get(lbl)
        out.append(f"- {lbl}: {hint}" if hint else f"- {lbl}")
    return "\n".join(out)


# ================================================================
# Auto-suggest labels
# ================================================================
@app.post("/api/suggest-label")
async def suggest_label(file: Annotated[UploadFile, File()]):
    if not file.filename:
        raise HTTPException(400, "Filename required.")
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file.")

    suffix = Path(file.filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        text = extract_text(tmp_path)[:2500]
    except Exception as e:  # noqa: BLE001
        return {"suggested_label": "other", "reason": f"extract-failed: {e}"}
    finally:
        tmp_path.unlink(missing_ok=True)

    if not text.strip():
        return {"suggested_label": "other", "reason": "no-text"}

    menu = _build_label_menu(list(ALLOWED_LABELS))

    prompt = f"""You are classifying a procurement document into exactly ONE label.

RULES:
1. Pick the MOST SPECIFIC label that fits the excerpt below.
2. Prefer specific over generic. A vendor's priced offer is "quote", not "other".
   An ISO 9001 certificate is "iso-9001", not just "certificate".
3. Use "other" ONLY when NO label in the menu fits. It is a last resort.
4. Return ONLY the label word — lowercase, exactly as it appears in the menu.
   No explanation, no punctuation, no quotes.

LABEL MENU:
{menu}

{_FEW_SHOT}

NOW CLASSIFY THIS DOCUMENT:
Filename: {file.filename}
Excerpt (first ~2500 chars):
{text}

Label:"""

    from .llm_client import chat as llm_chat
    try:
        raw = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model_size="fast", temperature=0, max_tokens=15,
        ).strip().lower()
        first = raw.split()[0].strip(".,:;-*\"'()[]{}") if raw.split() else ""
    except Exception as e:  # noqa: BLE001
        return {"suggested_label": "other", "reason": f"llm-failed: {e}"}

    guess = first if first in ALLOWED_LABELS else ""

    # Fallback 1: search the raw response for any known label (longest first,
    # so "iso-9001" wins over "iso"). Skip "other" so we never fall back to it.
    if not guess:
        for label in sorted(ALLOWED_LABELS, key=len, reverse=True):
            if label == "other":
                continue
            if label in raw:
                guess = label
                break

    # Fallback 2: filename heuristic (cheap, works surprisingly well on well-
    # named vendor files like "nirmala_chemicals_quote.pdf").
    if not guess:
        fname = file.filename.lower()
        for label in sorted(ALLOWED_LABELS, key=len, reverse=True):
            if label == "other":
                continue
            if label.replace("-", "") in fname.replace("-", "").replace("_", ""):
                guess = label
                break

    if not guess:
        guess = "other"

    return {"suggested_label": guess, "raw": raw}


# ================================================================
# Q&A
# ================================================================
@app.post("/api/query")
def query(
    q: Annotated[str, Form()],
    user: Annotated[str, Form()] = "anonymous",
):
    gate_result = gate(q, user)
    if not gate_result["allowed"]:
        return {"allowed": False, **gate_result}
    ans = qa_answer(q)
    return {"allowed": True, **ans}


# ================================================================
# Vendors
# ================================================================
@app.get("/api/vendors")
def vendors():
    if not DOC_STORE.exists():
        return {"vendors": []}
    return {"vendors": sorted(p.name for p in DOC_STORE.iterdir()
                              if p.is_dir())}


@app.get("/api/vendor/{vendor}/manifest")
def get_vendor_manifest(vendor: str):
    folders = STORAGE.list_folders(vendor=vendor)
    return {
        "vendor": vendor,
        "folders": [{**load_manifest(f), "path": str(f)} for f in folders],
    }


# ================================================================
# Edit labels
# ================================================================
@app.patch("/api/vendor/{vendor}/doc/labels")
def update_doc_labels(
    vendor: str,
    date: Annotated[str, Form()],
    filename: Annotated[str, Form()],
    labels: Annotated[list[str], Form()],
):
    bad = [l for l in labels
           if l not in ALLOWED_LABELS and not l.startswith("other:")]
    if bad:
        raise HTTPException(400, f"Unknown labels: {bad}")
    if not labels:
        raise HTTPException(400, "At least one label required.")

    folder = STORAGE.root / vendor / date
    if not folder.exists():
        raise HTTPException(404, "Folder not found.")

    manifest = load_manifest(folder)
    matching = [d for d in manifest["docs"] if d["filename"] == filename]
    if not matching:
        raise HTTPException(404, "Document not found in manifest.")

    matching[0]["labels"] = labels
    matching[0]["label_source"] = "user_updated"
    matching[0]["updated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    save_manifest(folder, manifest)

    source = str(folder / filename)
    coll = vector_store._get_collection() if hasattr(vector_store, "_get_collection") \
                                          else vector_store._collection
    got = coll.get(where={"source": source})
    if got["ids"]:
        joined = "|".join(labels)
        new_metas = [{**m, "labels": joined} for m in got["metadatas"]] # type: ignore
        coll.update(ids=got["ids"], metadatas=cast(Any, new_metas))

    tf = tfidf_store._load()
    for m in tf["metas"]:
        if m.get("source") == source:
            m["labels"] = labels
    tfidf_store._save(tf)

    return {"status": "updated", "vendor": vendor, "date": date,
            "filename": filename, "labels": labels}


# ================================================================
# Delete
# ================================================================
@app.post("/api/vendor/{vendor}/rename")
def rename_vendor(vendor: str, new_name: Annotated[str, Form()]):
    """Rename a vendor everywhere (folder, manifests, Chroma, TF-IDF, SQLite,
    PAN/GST records). The old name stays resolvable as an alias."""
    from .vendor_ops import RenameError
    from .vendor_ops import rename_vendor as do_rename
    try:
        result = do_rename(vendor, new_name)
    except RenameError as e:
        raise HTTPException(e.status, str(e)) from e
    return {"status": "renamed", **result}


@app.delete("/api/vendor/{vendor}")
def delete_vendor(vendor: str):
    vendor_root = STORAGE.root / vendor
    if not vendor_root.exists():
        raise HTTPException(404, "Vendor not found.")

    n_chroma = vector_store.remove_by_vendor(vendor)
    n_tfidf  = tfidf_store.remove_by_vendor(vendor)
    structured.remove_by_vendor(vendor)
    from .vendor_index import get_vendor_index
    canonical = get_vendor_index().resolve(vendor).canonical
    if canonical and canonical != vendor:
        structured.remove_by_vendor(canonical)
    shutil.rmtree(vendor_root)
    refresh_vendor_index()

    return {"status": "deleted", "vendor": vendor,
            "chroma_chunks_removed": n_chroma,
            "tfidf_chunks_removed":  n_tfidf}


@app.delete("/api/vendor/{vendor}/doc")
def delete_doc(vendor: str, date: str, filename: str):
    folder = STORAGE.root / vendor / date
    file_path = folder / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found.")

    source = str(file_path)
    n_chroma = vector_store.remove_by_source(source)
    n_tfidf  = tfidf_store.remove_by_source(source)
    structured.remove_by_source(source)

    mf = load_manifest(folder)
    mf["docs"] = [d for d in mf["docs"] if d["filename"] != filename]
    save_manifest(folder, mf)
    file_path.unlink()
    review.sidecar_path(file_path).unlink(missing_ok=True)
    # Re-check the vendor's PAN/GST rules. Deleting an extra document does
    # not clear existing flags; a reviewer has to approve what is left.
    from .vendor_index import get_vendor_index
    review.evaluate_vendor(get_vendor_index().resolve(vendor).canonical or vendor)

    # NOTE: folders are intentionally retained when empty.
    # The vendor row stays visible in the Vendors tab until the user
    # explicitly clicks Delete Vendor.

    return {"status": "deleted", "vendor": vendor, "date": date,
            "filename": filename,
            "chroma_chunks_removed": n_chroma,
            "tfidf_chunks_removed":  n_tfidf}


# ================================================================
# Debug endpoints
# ================================================================
@app.get("/api/debug/stats")
def debug_stats():
    coll = vector_store._get_collection() if hasattr(vector_store, "_get_collection") \
                                          else vector_store._collection
    n_chroma = coll.count()
    tf = tfidf_store._load()
    n_tfidf = len(tf["chunks"])
    sources = sorted({m["source"] for m in tf["metas"]})
    per_vendor: dict[str, int] = {}
    for m in tf["metas"]:
        per_vendor[m["vendor"]] = per_vendor.get(m["vendor"], 0) + 1
    return {
        "chroma_chunks":     n_chroma,
        "tfidf_chunks":      n_tfidf,
        "distinct_sources":  len(sources),
        "sources":           sources[:50],
        "chunks_per_vendor": per_vendor,
    }


@app.post("/api/debug/query")
def debug_query(
    q: Annotated[str, Form()],
    label_filter: Annotated[str, Form()] = "",
):
    labels = [l.strip() for l in label_filter.split(",") if l.strip()] or None

    tf = tfidf_store._load()
    tfidf_hits = []
    if tf["chunks"]:
        vec = tf["vectorizer"]
        v = vec.transform([q])
        scores = (tf["matrix"] @ v.T).toarray().ravel()
        top = scores.argsort()[::-1][:10]
        for i in top:
            if scores[i] <= 0:
                continue
            m = tf["metas"][i]
            tfidf_hits.append({
                "score":        float(scores[i]),
                "source":       m["source"],
                "para":         m.get("para"),
                "labels":       m.get("labels"),
                "text_preview": tf["chunks"][i][:220],
            })

    chroma_hits = []
    r = vector_store.query(q, k=10, label_filter=labels)
    if r["documents"] and r["documents"][0]:
        for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]):
            chroma_hits.append({
                "distance":     float(dist),
                "source":       meta["source"],
                "para":         meta["para"],
                "labels":       meta["labels"],
                "text_preview": doc[:220],
            })

    return {"query": q, "label_filter": labels,
            "tfidf": tfidf_hits, "chroma": chroma_hits}


@app.post("/api/debug/final")
def debug_final(
    q: Annotated[str, Form()],
    mode: Annotated[str, Form()] = "",
):
    """
    Final ranked chunks exactly as qa.answer() would hand them to the LLM
    (same vendor detection, LLM query expansion and RRF/legacy ranking),
    without generating an answer. Also reports whether the SQL router would
    have answered the query instead. Label filter does not apply here.
    """
    from . import config
    from .qa import retrieve

    if mode not in ("", "rrf", "legacy"):
        raise HTTPException(400, "mode must be 'rrf' or 'legacy'")
    r = retrieve(q, mode=mode or None, stop_at_structured=False)
    s = r["structured"]
    chunks = []
    for i, (doc, m) in enumerate(zip(r["docs"], r["metas"]), start=1):
        chunks.append({
            "rank":         i,
            "in_prompt":    i <= r["ctx_cap"],
            "vendor":       m.get("vendor"),
            "source":       m.get("source"),
            "para":         m.get("para"),
            "labels":       m.get("labels"),
            "chunk_type":   m.get("chunk_type"),
            "rrf_score":    m.get("rrf_score"),
            "vec_rank":     m.get("vec_rank"),
            "distance":     m.get("distance"),
            "tfidf_rank":   m.get("tfidf_rank"),
            "tfidf_score":  m.get("tfidf_score"),
            "text_preview": doc[:220],
        })
    return {
        "query":        q,
        "mode":         r["mode"],
        "default_mode": config.RETRIEVAL_MODE,
        "stage":        r["stage"],
        "route":        r["route"],
        "structured_route": s["route_method"] if s else None,
        "expanded":     r["expanded"],
        "vendor_hint":  r["vendor_hint"],
        "is_multi":     r["is_multi"],
        "ctx_cap":      r["ctx_cap"],
        "chunks":       chunks,
    }


@app.post("/api/debug/gate")
def debug_gate(
    q: Annotated[str, Form()],
    user: Annotated[str, Form()] = "debug",
):
    from .gate import rules, domain, llm_intent

    out: dict = {"query": q}

    ok1, reason1 = rules.check(q, user)
    out["stage1_rules"] = {"passed": ok1, "reason": reason1}
    if not ok1:
        out["final_verdict"] = f"REJECTED · {reason1}"
        return out

    ok2, p_dom = domain.is_in_domain(q)
    out["stage2_domain"] = {"passed": ok2, "p_in_domain": p_dom}
    if ok2:
        out["final_verdict"] = "APPROVED · classifier confident"
        return out

    is_proc = llm_intent.is_procurement(q)
    out["stage2_5_llm_domain_check"] = {"called": True,
                                        "is_procurement": is_proc}
    if is_proc:
        out["final_verdict"] = "APPROVED · via LLM domain check"
        return out

    out["final_verdict"] = "REJECTED · off_topic"
    return out


@app.get("/api/debug/chunks")
def api_debug_chunks(vendor: str, date: str, filename: str):
    """Return the chunks the current chunker would produce for one document."""
    from .chunker import chunk_document
    path = STORAGE.root / vendor / date / filename
    if not path.exists():
        raise HTTPException(404, "File not found.")
    text = extract_text(path)
    chunks = chunk_document(text)
    return {
        "vendor": vendor,
        "date": date,
        "filename": filename,
        "n_chunks": len(chunks),
        "chunks": chunks,
    }

@app.get("/api/debug/facts")
def api_debug_facts(vendor: str):
    """Return every extracted structured fact we hold for one vendor."""
    from . import structured
    from .vendor_index import get_vendor_index

    resolved = get_vendor_index().resolve(vendor)
    if not resolved.canonical:
        raise HTTPException(404, f"No vendor found for '{vendor}'.")
    return {
        "query": vendor,
        "canonical": resolved.canonical,
        "resolve_method": resolved.method,
        "facts": structured.get_all_facts_for_vendor(resolved.canonical),
    }

@app.get("/api/vendor-resolve")
def api_vendor_resolve(q: str):
    """Resolve a raw vendor mention. Useful for the Debug tab."""
    idx = get_vendor_index()
    r = idx.resolve(q)
    return {
        "query": q,
        "canonical": r.canonical,
        "confidence": round(r.confidence, 3),
        "method": r.method,
        "candidates": [
            {"vendor": v, "score": round(s, 3)} for v, s in r.candidates
        ],
    }


@app.post("/api/vendor-alias")
def api_vendor_alias(alias: Annotated[str, Form()], canonical: Annotated[str, Form()]):
    """Teach the resolver a new alias like 'nirmala chem' -> canonical."""
    idx = get_vendor_index()
    try:
        idx.add_alias(alias, canonical)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "added", "alias": alias, "canonical": canonical}

@app.get("/api/rankings/items")
def api_rankings_items():
    return {"items": vendor_score.list_items()}


class RankRequest(BaseModel):
    mode: str = "single"
    items: list[str] = []
    weights: dict[str, float] | None = None


@app.post("/api/rankings/rank")
def api_rankings_rank(req: RankRequest):
    result = vendor_score.compute_ranking(
        mode=req.mode, items=req.items, weights=req.weights,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    for row in result["rows"]:
        row["narrative"] = vendor_score.build_narrative(row, result["baseline"])
    return result

# ================================================================
# Manual review (flagged uploads, PAN/GST documents)
# ================================================================
@app.get("/api/review")
def api_review_queue():
    q = review.review_queue()
    return {"vendors": q, "count": sum(len(v["flagged"]) for v in q)}


@app.get("/api/review/count")
def api_review_count():
    return {"count": review.review_count()}


def _serve_doc(vendor: str, date: str, filename: str) -> FileResponse:
    """Serve a stored document inline (opens in the browser rather than
    downloading where the browser can display it: PDF, images, text)."""
    try:
        p = review._doc_path(vendor, date, filename)
    except review.ReviewError as e:
        raise HTTPException(404, str(e)) from e
    media = ("text/plain; charset=utf-8"
             if p.suffix.lower() in (".txt", ".md") else None)
    return FileResponse(p, media_type=media)


@app.get("/api/vendor/{vendor}/doc/file")
def api_vendor_doc_file(vendor: str, date: str, filename: str):
    return _serve_doc(vendor, date, filename)


@app.get("/api/review/file")
def api_review_file(vendor: str, date: str, filename: str):
    return _serve_doc(vendor, date, filename)


@app.get("/api/review/text")
def api_review_text(vendor: str, date: str, filename: str):
    try:
        p = review._doc_path(vendor, date, filename)
    except review.ReviewError as e:
        raise HTTPException(404, str(e)) from e
    sc = review.sidecar_path(p)
    text = sc.read_text() if sc.exists() else extract_text(p)
    return {"text": text}


def _schedule_index(background: BackgroundTasks, vendor: str, date: str,
                    filename: str, result: dict) -> None:
    if result["index"]:
        entry = result["entry"]
        queue_index(background, STORAGE.root / vendor / date / filename,
                    vendor, date, entry.get("labels", []), result["text"])


@app.post("/api/review/resolve")
def api_review_resolve(
    background: BackgroundTasks,
    vendor: Annotated[str, Form()],
    date: Annotated[str, Form()],
    filename: Annotated[str, Form()],
    action: Annotated[str, Form()],
    text: Annotated[str | None, Form()] = None,
    id_value: Annotated[str | None, Form()] = None,
    reviewer: Annotated[str, Form()] = "anonymous",
):
    try:
        result = review.resolve(vendor, date, filename, action, text=text,
                                id_value=id_value, reviewer=reviewer)
    except review.ReviewError as e:
        raise HTTPException(400, str(e)) from e
    _schedule_index(background, vendor, date, filename, result)
    return {"status": result["entry"].get("review_status"),
            "indexing": result["index"], "entry": result["entry"]}


@app.post("/api/review/resolve-group")
def api_review_resolve_group(
    vendor: Annotated[str, Form()],
    id_type: Annotated[str, Form()],
    action: Annotated[str, Form()],
    keep_date: Annotated[str | None, Form()] = None,
    keep_filename: Annotated[str | None, Form()] = None,
    reviewer: Annotated[str, Form()] = "anonymous",
):
    keep = [(keep_date, keep_filename)] if keep_date and keep_filename else None
    try:
        entries = review.resolve_group(vendor, id_type, action, keep=keep,
                                       reviewer=reviewer)
    except review.ReviewError as e:
        raise HTTPException(400, str(e)) from e
    return {"status": "resolved", "entries": entries}


# ================================================================
# Static web UI
# ================================================================
class _RevalidatingStatic(StaticFiles):
    """
    Static files with Cache-Control: no-cache, so the browser revalidates
    (cheap 304 via ETag) on every load instead of serving a stale app.js
    from heuristic cache after the UI has been edited.
    """
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/static", _RevalidatingStatic(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html",
                        headers={"Cache-Control": "no-cache"})