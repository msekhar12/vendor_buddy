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