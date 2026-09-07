# 1. Install deps (once)
pip install -r requirements.txt
make setup                   # spacy model + sentence-BERT

# 2. Set your Groq API key
export GROQ_API_KEY="gsk_..."

# 3. Build demo docs and train the gate
make demo                    # runs seed + gate

# 4. In terminal A
make api                     # FastAPI at :8000

# 5. In terminal B
make ui                      # Streamlit at :8501

# 6. Open browser
open http://localhost:8501


# For OLLAMA installation
brew install ollama

# Start Ollama as a background service (runs on http://localhost:11434)
brew services start ollama

# Pull a small model for classification + a mid-size one for RAG answers
ollama pull llama3.2:3b       # ~2 GB — fast, good for gate + label suggest
ollama pull qwen2.5:7b        # ~5 GB — better quality for RAG answers

# Verify
ollama list
ollama run llama3.2:3b "Say hello"

# To turn-off sql tracing:
CIVSA_SQL_TRACE=0 make api


# wipe out and reindex:
python -c "
from civsa import structured
from civsa.config import SQLITE_PATH
structured.init_db(SQLITE_PATH)
structured.reset()
"

make reindex

# SQL Test queries:

sqlite3 indexes/facts.sqlite (for interactive)

sqlite3 indexes/facts.sqlite \
  "SELECT canonical_vendor FROM vendor_facts WHERE LOWER(canonical_vendor) LIKE '%kaveri%';"

sqlite3 indexes/facts.sqlite \
  "SELECT canonical_vendor FROM vendor_facts WHERE LOWER(canonical_vendor) LIKE '%nirmala%';"  



is Nirmala ISO 9001 certified?	A · single-vendor ISO
what is Kaveri's GSTIN? --NOGSTIN for Kaveri
what is nirmala's GSTIN?	
price of methanol from Nirmala?
what did Aditya quote?
which vendors are ISO 14001 certified?
which suppliers do not have GST?
vendors from Mumbai
quotes in the last 2 days
quotes in the last 10 days
vendors delivering under 7 days
cheapest ethanol
most expensive methanol. 
who can supply methanol?
least expensive methanol
cheapest methanol?
fastest delivery
slowest delivery
longest credit period
shortest credit
who can supply potassium phosphate?
who can supply methanol?
how many vendors?
how many items did Nirmala quote?
list ISO certified vendors
list all vendors
what is Kaveri's GSTIN?
what is the GSTIN of Nirmala Chemicals?
"give me Deccan's PAN" → Deccan Chemicals & Reagents — pan: AAAFD3320H
"tell me the address of Gangotri" → the address string
"show me Aditya's quote number" → the quote reference number
"what is the quote of gangotri?" → Answer (structured_sql); lists Gangotri's line items
"give me Nirmala's quote" → same route (possessive branch)
"show me the quotes from Aditya" → same route (of-form branch, "from" synonym)
"tell me Kaveri's quote" → same route
"what did Gangotri quote?" → still SQL (existing block, unchanged)
"what is Gangotri's quote number?" → still SQL, but via the field-lookup block (different, more specific answer)

email IDs from hyderabad vendors.









Phase B — Next 3: Deployment + viva prep (1–2 days)

Same three deliverables from the earlier roadmap. Precise scope:

B1. One-command install (Makefile + scripts/setup.sh)

make setup creates the venv, pip install -r requirements.txt, downloads NLTK data, pulls the two Ollama models, creates empty documents/ and indexes/ folders, seeds nothing.
make api starts uvicorn on port 8000.
make demo re-uploads the 5 sample PDFs I generated, runs the eval, opens the browser.
Passes clean on a fresh clone → clean venv → make setup && make demo in under 5 minutes.

B2. Demo script (docs/DEMO.md)

Ten canned queries in an exact order, each hitting a different pillar:
Upload flow (drag-drop a new PDF)
Auto-tag (show "other" being suggested)
Gate fast-path (what is Nirmala's GSTIN? — no LLM tie-break)
Corpus lookup (list all vendors)
Single-vendor ISO SQL (is Aditya ISO 9001 certified? → answers NOT)
Multi-vendor SQL (cheapest ethanol)
Vendor resolution (is rajshree iso certified? — first-word match)
Table-aware chunking (fact viewer for Nirmala → 5 line items)
Vendor-scoped RAG (what warranty does Nirmala offer?)
Off-topic gate rejection (who won the world cup?)
Each with 2 sentences of narration for the committee.

B3. Architecture one-pager (docs/ARCHITECTURE.md + published as artifact)

Single diagram: upload → chunker → dual store (Chroma + TF-IDF) + LLM extractor → SQLite. Query: input → gate → SQL router → SQL / vendor-scoped RAG / broad RAG → LLM synthesis → answer + citations.
Table of the 15 SQL router patterns.
Eval scoreboard with your final pass rate.
Ready to paste into your report.

I can publish B3 as an updated HTML artifact when you're ready — it becomes your "current" technical design doc and supersedes the old scoping doc.

Phase C — Report writing & submission (owned by you, I assist)

The M.Tech report needs sections your codebase now supports evidence for:

Motivation / problem statement (Phase 1 material)
Related work (RAG limitations, negation blindness, need for structured routing)
Architecture (from B3)
Implementation (walkthrough of the 4 pillars: vendor resolver, table chunker, structured DB, gate)
Evaluation (from the eval harness — before/after numbers, per-family breakdown)
Limitations & future work (LLM query router, cross-encoder re-ranker, more data)

Send me the report template your institute uses and I'll help fill each section with what we've built, citations included.

Suggested sequence
This week: Run eval → Phase A triage → land at ≥85% pass rate.
Next week: Phase B (make setup, demo script, architecture doc). Reply "start Phase B" and I'll deliver all three in one code drop.
After viva: if you want to keep improving, add an LLM query router (catches phrasings the regex misses) and a cross-encoder re-ranker (better RAG quality). Both are ~2 days each and non-blocking for the viva.

Right now, the single most valuable thing to do next: run python -m scripts.eval and paste me the summary + any failure lines. From that I can tell you exactly which routes need tightening before you invest time in Phase B.


===========

Concrete next step, if you want to build this

Reply "scaffold vendor scoring" and I'll give you:

civsa/vendor_score.py — the composite score computation and per-feature attribution using your existing SQL helpers
civsa/api.py — a new GET /api/vendor-score/{vendor} endpoint
Vendors tab UI update — attribution bar per vendor
One row in eval_queries.json — "score aditya" → structured_sql

Roughly one code drop, one day of your time.

--PROJECT DOCUMENTATION GENERATION
Continuation marker

For any follow-up in this document-generation series (after you've reviewed Doc 1 and want me to make changes, or when you say "ready for Doc 2"), use the tag:

[CIVSA-DOC-GEN]

Just include it anywhere in your message. Example:

"[CIVSA-DOC-GEN] make the Objectives section shorter"
"[CIVSA-DOC-GEN] ready for Doc 2"
"[CIVSA-DOC-GEN] Doc 1 has an error on section 5.3"

That tag anchors us in this specific work stream so I know to keep the same style guide, section structure, project name (CIVSA), and delivery format we're establishing now, without you having to re-explain.

I'll also add this in the docx footer of Doc 1 for easy reference.

-----

When you've reviewed it, reply with [CIVSA-DOC-GEN] doc1 ok to start Doc 2, or [CIVSA-DOC-GEN] doc1: <changes> with any edits.

---------

============================================================
Overall:  60/60 passed (100%)
Wall:     227.6s (3.79s/query)
============================================================
By family:
  edge         5/ 5  (100%)
  gate         3/ 3  (100%)
  rag          2/ 2  (100%)
  sql         43/43  (100%)
  vendor       7/ 7  (100%)