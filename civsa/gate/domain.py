"""
Stage 2: binary in-domain classifier.

Trained by scripts/train_gate.py on sentence-BERT embeddings of ~500 in-domain
+ ~500 out-of-domain queries. Loaded here once at import time.
"""
import pickle

from sentence_transformers import SentenceTransformer

from ..config import DATA_DIR, DOMAIN_THRESHOLD, EMBED_MODEL_PATH

_encoder = SentenceTransformer(EMBED_MODEL_PATH)
_model_path = DATA_DIR / "gate_domain.pkl"
_model = pickle.loads(_model_path.read_bytes()) if _model_path.exists() else None


def is_in_domain(query: str) -> tuple[bool, float]:
    """Returns (allowed, probability_in_domain)."""
    if _model is None:
        # Model not trained yet — allow everything so the tool works before gate is trained
        return True, 1.0
    v = _encoder.encode([query])
    p_in = float(_model.predict_proba(v)[0, 1])
    return p_in >= DOMAIN_THRESHOLD, p_in