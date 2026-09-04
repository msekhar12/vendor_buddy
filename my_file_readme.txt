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