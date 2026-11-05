# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CIVSA is a procurement Q&A system over vendor documents (quotes, ISO certs, MTCs, invoices). Users upload PDFs, images (OCR'd with Tesseract), DOCX or XLSX per vendor; questions are answered either deterministically from an LLM-extracted SQLite facts DB or via RAG over Chroma + TF-IDF. It is an M.Tech project; `my_file_readme.txt` holds the author's working notes and sample queries.

## Commands

Python ≥3.11, venv in `.venv/`. All commands run from the repo root; modules import as `civsa.*` so `PYTHONPATH=.` is needed for scripts (the Makefile sets it).

```bash
pip install -r requirements.txt
make setup          # spaCy model + saves all-MiniLM-L6-v2 into models/
make gate           # build gate training data + train classifier -> data/gate_*.pkl
make seed           # generate demo docs (scripts/create_demo_docs.py)
make api            # uvicorn civsa.api:app on :8000 (serves static/ web UI at /)
make reindex        # wipe Chroma + TF-IDF + SQLite facts, reindex everything under documents/
make test           # pytest -v (~95 s: OCR quality tests + vendor-rename end-to-end test)
PYTHONPATH=. pytest tests/test_gate.py::test_stage1_injection_blocked -v   # single test
make images         # regenerate SPECIMEN test images + TEST-DATA/images/expected.json
make eval-images    # end-to-end image / PAN / GST / review checks (see below)
make recheck-identity  # re-run the holder-name check on stored PAN/GST docs (after changing name_matches)
```

Tesseract must be installed (`brew install tesseract`; uses the `eng` and `osd` packs).

Two tests fail and were already failing before the recent work: `tests/test_gate.py::test_stage1_normal_query_allowed` and `tests/test_storage.py::test_safe_path_traversal`. A clean run is 56 passed, 2 failed.

Eval harness (`scripts/eval.py`, queries in `scripts/eval_queries.json`, results to `eval_results/`):

```bash
python -m scripts.eval                    # all queries, in-process (no server needed)
python -m scripts.eval --family sql       # families: sql, vendor, rag, gate, edge
python -m scripts.eval --id corp-list-all # one query
python -m scripts.eval --http             # go through the running API instead
python -m scripts.eval --verbose          # full answer on failures
```

Each eval case checks `expected_route` (the `route_method` returned by `qa.answer`) plus `must_contain_all` / `must_contain_any`. The current baseline is 60/60 passing; run the eval after any change to `qa.py`, `structured.py`, `vendor_index.py` or the gate.

`scripts/eval_images.py` uploads every file in `TEST-DATA/images/expected.json` through `/api/upload`, then checks review statuses, PAN/GSTIN answers and reviewer actions. It runs in a throwaway folder: it re-runs itself with `CIVSA_DATA_ROOT` pointed at a temp dir and refuses to run against the real corpus. By default the LLM is disabled and quote fact extraction is replaced by a rule-based stand-in, so results are deterministic; pass `--llm` to use Ollama. `CIVSA_DATA_ROOT` relocates `documents/` and `indexes/` for any process, but it must be set before `civsa` is imported. Tests that need a populated store (`test_vendor_rename.py`) run a subprocess with it set.

To try UI changes without touching the real corpus (the user's `make api` on :8000 serves it), start a second server on a throwaway root: `CIVSA_DATA_ROOT=/tmp/x PYTHONPATH=. uvicorn civsa.api:app --port 8010`. Then seed it with `curl -F vendor=… -F labels=quote -F file=@… localhost:8010/api/upload`.

Linting/type config is in `pyproject.toml` (ruff, mypy, pyright); no Makefile target for them.

## LLM providers

`civsa/llm_client.chat()` is the single LLM entrypoint. It tries providers in `config.LLM_PROVIDER_ORDER` (default Ollama first, then Groq) and returns the first that doesn't raise. Ollama models: `llama3.2:3b` (fast/classification) and `qwen2.5:7b` (answers), at `localhost:11434`. Groq needs `GROQ_API_KEY`. `model_size` chooses fast vs. full model. On this machine Ollama runs on CPU, so the background fact extraction after an upload (`qwen2.5:7b`) can take several minutes. Anything that depends on a document's facts has to wait for the `[indexer] structured extraction` log line. Set `CIVSA_SQL_TRACE=0` to silence the SQL statement trace printed by `structured.py`.

## Architecture

**Ingest** (`api.py /api/upload` → `review.process_upload`, synchronous → `indexer.index_document`, run as a FastAPI BackgroundTask):
1. `storage.LocalDriver` writes the file to `documents/<Vendor>/<YYYY-MM-DD>/<file>`. Each date folder has a `manifest.json` (labels, sha256, uploader, and the review fields below). Duplicates are detected per vendor by sha256 (`manifest.py`). Labels must come from `config.ALLOWED_LABELS` or have the form `other:<text>`. File types are limited to `extract.SUPPORTED_EXTS`.
2. `review.process_upload` runs during the request, so the upload card can show a flag immediately:
   - `extract.extract_with_quality` extracts the text. Images, and PDF pages with almost no native text, go through `extract.ocr_image`: orientation fix via Tesseract OSD, upscaling, `--psm 4` (keeps table rows on one line), an `OCR_TIMEOUT_S` limit, and metrics (word confidence, sharpness, size, junk ratio). OCR text is cached next to the file as `<file>.ocr.txt`, which is reused by indexing and reindex, and overwritten when a reviewer corrects it.
   - `ocr_quality.assess_document` checks the document on its own. Readability problems come from the `OCR_MIN_*` thresholds in `config.py`, which were calibrated on `TEST-DATA/images/quality`. OCR'd non-ID documents also get a relevance check: a procurement-keyword count, with a fast-LLM yes/no only for borderline text. Other checks are label vs. content (`label_mismatch`) and, for PAN/GST documents, ID readability and name match. Reason codes and their messages are in `ocr_quality.REASON_TEXT`.
   - `needs_review` documents are stored but **not indexed** until a reviewer approves them. PAN cards and GST certificates are never indexed into Chroma or TF-IDF; they go only into `structured.vendor_identity_docs`.
3. `extract.extract_text` (or the cached or reviewed `text` argument) → `chunker.chunk_document`. The chunker is table-aware: `table_regions.py` finds priced-schedule tables and emits one chunk per row carrying its header. Prose goes through a sliding window (700 chars, 120 overlap).
4. Chunks go into **both** `vector_store` (Chroma at `indexes/chroma`, MiniLM embeddings) and `tfidf_store` (`indexes/tfidf.pkl`). Metadata includes vendor/date/labels/source.
5. `structured.extract_facts` asks the LLM for JSON facts, which `store_extraction` writes to `indexes/facts.sqlite` (tables `vendor_facts`, `quote_items`, `commercial_terms`), keyed by **canonical** vendor name. A failure here is logged and swallowed, so RAG keeps working. Afterwards the indexer calls `review.evaluate_vendor`, because a quote's PAN/GSTIN can conflict with an uploaded PAN/GST document.

If you change the chunker, the extraction prompt or the schema, existing data has to be rebuilt with `make reindex`. Reindex skips `needs_review` and `rejected` documents, uses `.ocr.txt` sidecars, and restores PAN/GST rows from the manifests without OCR.

**PAN / GSTIN and review** (`identity.py`, `review.py`):
- **Rule-based reading (`identity.py`, not the LLM):** format checks, GSTIN state code and mod-36 check character, the GSTIN's embedded PAN (characters 3–12), and position-aware OCR repair capped at 2 changes so ordinary words can't become IDs. When no valid ID is read, `extract.reread_value_after_label` re-OCRs only the value after "Account Number:" or "Registration Number:". Holder-name check (`identity.name_matches`): after dropping corporate suffixes (Pvt, Ltd, Co, …), every word on each side must pair with a word on the other side, allowing a typo or an abbreviation of 3+ letters. Any extra word means a different company, so "Konkan" doesn't match "Konkan Marine Exports". `review.recheck_identity_names` re-applies the check to stored documents; it runs after a rename and during reindex. GSTINs printed on the existing demo quotes are made up and fail the check character, so strict validation applies only to PAN/GST documents.
- **`structured.get_vendor_identity(vendor)`** resolves PAN and GSTIN per vendor:
  1. An approved or auto-accepted PAN/GST document wins (a PAN can also be derived from an approved GSTIN).
  2. Otherwise the value printed on a quote is used.
  3. Documents awaiting review are never used; they come back under `pending`.

  `qa._format_identity`, `vendors_with_gstin` and `vendors_without_gstin` go through it. `vendor_facts` has one row per document, so never read a vendor field with an unordered `LIMIT 1`; `get_vendor_field` picks the newest non-empty value.
- **`review.evaluate_vendor`** runs under a per-vendor lock after every upload, deletion, reviewer decision and quote indexing. It flags:
  - every PAN document when a vendor has 2+ of them, and the same for GST documents
  - a PAN that doesn't match the PAN inside a GSTIN
  - an ID document that disagrees with the vendor's quote

  Flags never clear automatically, not even when the extra document is deleted. A reviewer approval records `acknowledged_reasons` and `ack_group`, and only a new reason or a changed document group re-opens review.
- **Reviewer actions:** `/api/review/resolve` (`approve`, `submit_text`, `set_id`, `reject`) and `/api/review/resolve-group` (`keep_one`; `approve_all` for GST only, and only when every GSTIN shares one PAN).

**Vendor resolution** (`vendor_index.py`): `get_vendor_index().resolve(str)` maps free text to the canonical vendor folder name. It tries, in order: exact match, suffix-stripped normalised match, `documents/aliases.json`, fuzzy match, substring match. Call `refresh_vendor_index()` after vendors are added or removed. Ingest and query must both use this so that SQL rows and folder names agree.

**Query** (`/api/query`):
1. `gate/pipeline.process` runs Stage 1 `rules.check` (empty, too-long, injection, rate limit), then a vocabulary fast-path (`rules.looks_like_procurement`), then Stage 2, the trained domain classifier (`data/gate_domain.pkl`, threshold `DOMAIN_THRESHOLD`), then Stage 2.5, an LLM yes/no tie-break. Anything still rejected gets an `off_topic` refusal.
2. `qa.answer` is a thin wrapper: `qa.retrieve()` does everything up to generation and returns a dict (`stage`, `structured`, `mode`, `route`, `docs`, `metas`, `expanded`, `vendor_hint`, `is_multi`, `ctx_cap`), and `_generate` produces the answer. Any change to retrieval belongs in `retrieve()`, because the debug probe calls it too.
   - **Path 0, the SQL router**: `_try_structured` is a long ordered chain of regex patterns that call `structured.*` helpers (cheapest_for, list_iso_certified, get_vendor_field, …) and return deterministic answers with `route_method` values like `structured_sql`, `structured_corpus` and `structured_sql_empty`. Order matters: most specific patterns go first and generic catch-alls go last. Vendor synonyms (supplier/company/firm) are normalised to "vendor" before matching. If no pattern matches, `_expand_query` asks the LLM to rewrite the query for retrieval.
   - **RAG ranking** depends on `config.RETRIEVAL_MODE` (or the `mode` argument to `retrieve()`):
     - `"rrf"` (default): `_rrf_retrieve` fuses a Chroma ranked list (with the table-row boost applied) and a `tfidf_store.query_ranked` list using `fusion.rrf` (Reciprocal Rank Fusion; `RRF_K`, `RRF_W_VECTOR` and `RRF_W_TFIDF` in config). Both lists are vendor-filtered when a single vendor is detected (k=15). Otherwise they cover the whole corpus (k=20, or 40 for multi-vendor). Chunks are matched across the two stores by `(source, text)` because their IDs differ. Each chunk's meta gets `rrf_score`, `vec_rank`, `distance`, `tfidf_rank` and `tfidf_score`.
     - `"legacy"`: Chroma alone does the ranking. The vendor-scoped path filters Chroma to that vendor. The broad path either runs a wide k=40 search (multi-vendor) or uses a TF-IDF source shortlist to pre-filter Chroma, dropping the filter if it is too narrow or too sparse.
   - `_generate` sends the first `ctx_cap(is_multi)` chunks (10, or 20 for multi-vendor) to the LLM and runs a hallucination check.

The 60-case eval barely exercises RAG ranking, because nearly every case is answered by the SQL router. To compare `rrf` and `legacy`, call `qa.retrieve(q, mode=...)` directly with `_try_structured` and `_expand_query` stubbed out, and measure where the ground-truth chunk ranks.

**Vendor rename** (`vendor_ops.rename_vendor`, `POST /api/vendor/{vendor}/rename`, "Rename vendor" in the Vendors tab):
- Moves `documents/<old>/` and rewrites every path-keyed store in place: manifests (`vendor`, `ack_group`), Chroma (ids are `<source>#<n>`, so chunks are copied to new ids with their embeddings, then the old ids are deleted), TF-IDF metadata, and the four SQLite tables.
- The old name is written to `documents/aliases.json`, so questions that use it still resolve.
- It refuses (409) to rename onto an existing vendor (that would be a merge), and refuses while any of the vendor's documents are still being indexed. Background jobs are counted from the moment they're queued: `indexer.mark_queued` / `index_document_tracked`, scheduled through `api.queue_index`. Always schedule indexing through `queue_index`, never `background.add_task(index_document, ...)`, or a rename can race the job and leave rows under the old path.

**Paths are index keys.** Document paths are always spelled `DOC_STORE/vendor/date/filename`, which is the key in Chroma, TF-IDF and SQLite. Don't `resolve()` them: on macOS that turns `/var` into `/private/var` and creates a second key.

**Rankings** (`vendor_score.py`, `/api/rankings/*`): a weighted composite score per vendor (price over an item basket, delivery, credit, …) computed from the SQLite facts, with a per-criterion attribution narrative.

**Frontend**: `static/index.html` + `static/app.js` is the main UI that FastAPI serves at `/`. It has upload, Q&A, vendor management, a Review tab (flagged uploads, with grouped PAN/GST decisions and a count badge), debug views (`/api/debug/*` for chunks, facts, gate and retrieval probes) and rankings. `civsa/ui.py` is a legacy Streamlit prototype.
- The Debug tab's query probe makes two requests. `/api/debug/query` returns the raw TF-IDF and Chroma top 10. `/api/debug/final` (with an optional `mode` of `rrf` or `legacy`) runs `retrieve(..., stop_at_structured=False)` and returns the final ranked chunks, an `in_prompt` flag on each, and `structured_route` if the SQL router would have answered instead. The second request is slower because it includes LLM query expansion.
- Upload history rows on the Upload tab are a client-side snapshot. `/api/upload` returns `vendor_folder`, `date` and `filename`, so flagged rows can re-check their manifest status (`refreshUploadLogStatuses`) when the Upload tab opens and after each Review decision.
- The page and `/static` are served with `Cache-Control: no-cache`, and `index.html` links `app.js?v=N` and `style.css?v=N`. Bump N whenever you edit the JS or CSS so browsers that cached the old files pick up the change.

## Data directories

- `documents/` holds uploaded vendor files and manifests. It is runtime data, but sample vendors are committed.
- `indexes/` holds Chroma, TF-IDF and SQLite. It's git-ignored and rebuilt from `documents/` with `make reindex`, which is needed on a fresh clone. The Chroma `.bin` files change on every upload, so don't commit them. `models/` holds the cached embedding model. `data/` holds gate training data and pickles.
- `TEST-DATA/` contains sample PDFs for manual upload. `TEST-DATA/images/` (generated by `make images`) holds SPECIMEN images for fictional vendors: `uploads/<scenario>/` contains quotes, PAN and GST documents for each edge case, and `quality/` contains a degradation ladder plus irrelevant images. `expected.json` is the ground truth used by the tests and `eval_images.py`. Images are saved as grayscale, optimised PNG or JPEG (about 3 MB in total). Avoid per-pixel random noise in new fixtures: it doesn't compress, and a single 8 MB file made `git push` over HTTPS fail.
- `make clean` deletes `indexes/*` and `documents/*` after a prompt.
