"""
Central config for CIVSA.

Every path and constant used by more than one module lives here so that
changing a location (e.g., moving the documents folder) is a one-line edit.
"""
from pathlib import Path

# ---- Paths ----
ROOT       = Path(__file__).resolve().parent.parent   # project root
DOC_STORE  = ROOT / "documents"     # where uploaded vendor files live
INDEX_DIR  = ROOT / "indexes"       # TF-IDF matrix + ChromaDB
DATA_DIR   = ROOT / "data"          # gate training data, demo assets
MODEL_DIR  = ROOT / "models"        # cached sentence-BERT

# ---- Embedding model ----
EMBED_MODEL_PATH = str(MODEL_DIR / "all-MiniLM-L6-v2")
EMBED_DIM        = 384              # dims of MiniLM-L6-v2

# ---- Structured facts DB (Phase 2 Step 3) ----
SQLITE_PATH = INDEX_DIR / "facts.sqlite"

# ---- Groq LLM configuration ----
# See https://console.groq.com/docs/models for the current model roster.
# 70B is best quality; 8B-instant is faster and cheaper, use it for bulk work.

# You can also run this command to find the allowed models in your Groq account:
# curl -s "https://api.groq.com/openai/v1/models"   -H "Authorization: Bearer $GROQ_API_KEY"   | python -m json.tool | grep '"id"'

# ---- LLM provider fallback chain ----
# Tried in order. First provider that returns without raising wins.
# To force Groq-only: LLM_PROVIDER_ORDER = ["groq"]
# To force Ollama-only: LLM_PROVIDER_ORDER = ["ollama"]
LLM_PROVIDER_ORDER = ["ollama", "groq"]

# ---- Ollama (primary — local) ----
OLLAMA_HOST       = "http://localhost:11434"
OLLAMA_MODEL_FAST = "llama3.2:3b"      # classification tasks
OLLAMA_MODEL      = "qwen2.5:7b"       # RAG answers

# ---- Groq (backup — cloud) ----
GROQ_MODEL        = "openai/gpt-oss-20b"       # RAG answers
GROQ_MODEL_FAST   = "llama-3.1-8b-instant"     # classification (non-reasoning)


# ---- Upload / query limits ----
MAX_UPLOAD_MB    = 25
MAX_QUERY_TOKENS = 2000

# ---- Controlled vocabulary for document labels (Appendix D of scoping) ----
ALLOWED_LABELS = {
    "quote", "pricing_schedule", "invoice", "purchase_order",
    "terms_and_conditions", "master_service_agreement",
    "non_disclosure_agreement", "contract",
    "iso_9001_certification", "iso_14001_certification",
    "iso_45001_certification", "bis_certification", "msme_certificate",
    "certification", "msds", "mtc", "test_report", "product_datasheet",
    "chemical_composition", "gst_certificate", "pan_card",
    "udyam_certificate", "compliance_letter",
    "delivery_schedule", "goods_received_note", "inspection_certificate",
    "company_profile", "reference_letter", "audited_financials", "other",
}

# ---- Q&A intents ----
# All five defined; only the first three are wired up in Phase 1.
INTENTS = ["vendor_attribute", "cert_filter", "price_compare",
           "clause_lookup", "delivery_history"]
PHASE1_INTENTS = {"vendor_attribute", "cert_filter", "price_compare"}

# ---- Gate thresholds ----
DOMAIN_THRESHOLD = 0.48   # Stage 2: below this = treat as off-topic
INTENT_THRESHOLD = 0.60   # Stage 3: below this = ambiguous intent

# Ensure runtime folders exist so first-run doesn't blow up
for p in (DOC_STORE, INDEX_DIR, DATA_DIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)