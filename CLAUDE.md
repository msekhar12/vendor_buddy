# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CIVSA is a procurement Q&A system over vendor documents (quotes, ISO certs, MTCs, invoices). Users upload PDFs/DOCX/XLSX per vendor; questions are answered either deterministically from an LLM-extracted SQLite facts DB or via RAG over Chroma + TF-IDF. It is an M.Tech project; `my_file_readme.txt` holds the author's working notes and sample queries.

## Commands

Python ≥3.11, venv in `.venv/`. All commands run from the repo root; modules import as `civsa.*` so `PYTHONPATH=.` is needed for scripts (the Makefile sets it).

```bash
pip install -r requirements.txt
make setup          # spaCy model + saves all-MiniLM-L6-v2 into models/
make gate           # build gate training data + train classifier -> data/gate_*.pkl
make seed           # generate demo docs (scripts/create_demo_docs.py)
make api            # uvicorn civsa.api:app on :8000 (serves static/ web UI at /)
make reindex        # wipe Chroma + TF-IDF + SQLite facts, reindex everything under documents/
make test           # pytest -v
PYTHONPATH=. pytest tests/test_gate.py::test_stage1_injection_blocked -v   # single test
```

`tests/test_gate.py::test_stage1_normal_query_allowed` and `tests/test_storage.py::test_safe_path_traversal` are failing on `main`. Those failures existed before the RRF work.

Eval harness (`scripts/eval.py`, queries in `scripts/eval_queries.json`, results to `eval_results/`):

```bash
python -m scripts.eval                    # all queries, in-process (no server needed)
python -m scripts.eval --family sql       # families: sql, vendor, rag, gate, edge
python -m scripts.eval --id corp-list-all # one query
python -m scripts.eval --http             # go through the running API instead
python -m scripts.eval --verbose          # full answer on failures
```

Each eval case checks `expected_route` (the `route_method` returned by `qa.answer`) plus `must_contain_all` / `must_contain_any`. The current baseline is 60/60 passing; run the eval after any change to `qa.py`, `structured.py`, `vendor_index.py` or the gate.

Linting/type config is in `pyproject.toml` (ruff, mypy, pyright); no Makefile target for them.

## LLM providers

`civsa/llm_client.chat()` is the single LLM entrypoint. It tries providers in `config.LLM_PROVIDER_ORDER` (default Ollama first, then Groq) and returns the first that doesn't raise. Ollama models: `llama3.2:3b` (fast/classification) and `qwen2.5:7b` (answers), at `localhost:11434`. Groq needs `GROQ_API_KEY`. `model_size` chooses fast vs. full model. Set `CIVSA_SQL_TRACE=0` to silence the SQL statement trace printed by `structured.py`.

## Architecture

**Ingest** (`api.py /api/upload` → `indexer.index_document`, run as a FastAPI BackgroundTask):
1. `storage.LocalDriver` writes the file to `documents/<Vendor>/<YYYY-MM-DD>/<file>`. Each date folder has a `manifest.json` (labels, sha256, uploader). Duplicates are detected per vendor by sha256 (`manifest.py`). Labels must come from `config.ALLOWED_LABELS` or have the form `other:<text>`.
2. `extract.extract_text` (PyMuPDF / docx / openpyxl / OCR) → `chunker.chunk_document`. The chunker is table-aware: `table_regions.py` finds priced-schedule tables and emits one chunk per row carrying its header. Prose goes through a sliding window (700 chars, 120 overlap).
3. Chunks go into **both** `vector_store` (Chroma at `indexes/chroma`, MiniLM embeddings) and `tfidf_store` (`indexes/tfidf.pkl`). Metadata includes vendor/date/labels/source.
4. `structured.extract_facts` asks the LLM for JSON facts, which `store_extraction` writes to `indexes/facts.sqlite` (tables `vendor_facts`, `quote_items`, `commercial_terms`), keyed by **canonical** vendor name. A failure here is logged and swallowed, so RAG keeps working.

If you change the chunker, the extraction prompt or the schema, existing data has to be rebuilt with `make reindex`.

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

**Rankings** (`vendor_score.py`, `/api/rankings/*`): a weighted composite score per vendor (price over an item basket, delivery, credit, …) computed from the SQLite facts, with a per-criterion attribution narrative.

**Frontend**: `static/index.html` + `static/app.js` is the main UI that FastAPI serves at `/`. It has upload, Q&A, vendor management, debug views (`/api/debug/*` for chunks, facts, gate and retrieval probes) and rankings. `civsa/ui.py` is a legacy Streamlit prototype.
- The Debug tab's query probe makes two requests. `/api/debug/query` returns the raw TF-IDF and Chroma top 10. `/api/debug/final` (with an optional `mode` of `rrf` or `legacy`) runs `retrieve(..., stop_at_structured=False)` and returns the final ranked chunks, an `in_prompt` flag on each, and `structured_route` if the SQL router would have answered instead. The second request is slower because it includes LLM query expansion.
- The page and `/static` are served with `Cache-Control: no-cache`, and `index.html` links `app.js?v=N` and `style.css?v=N`. Bump N whenever you edit the JS or CSS so browsers that cached the old files pick up the change.

## Data directories

- `documents/` holds uploaded vendor files and manifests. It is runtime data, but sample vendors are committed.
- `indexes/` holds Chroma, TF-IDF and SQLite. `models/` holds the cached embedding model. `data/` holds gate training data and pickles.
- `TEST-DATA/` contains sample PDFs for manual upload.
- `make clean` deletes `indexes/*` and `documents/*` after a prompt.
