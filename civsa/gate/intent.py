"""
Stage 3: multi-class intent router over the same sentence-BERT embeddings.
"""
import pickle

from sentence_transformers import SentenceTransformer

from ..config import DATA_DIR, EMBED_MODEL_PATH, INTENT_THRESHOLD, INTENTS

_encoder = SentenceTransformer(EMBED_MODEL_PATH)
_path = DATA_DIR / "gate_intent.pkl"
_model = pickle.loads(_path.read_bytes()) if _path.exists() else None


def route(query: str) -> tuple[str | None, float]:
    """Returns (intent_name_or_None, confidence). None means ambiguous."""
    if _model is None:
        return None, 0.0
    v = _encoder.encode([query])
    probs = _model.predict_proba(v)[0]
    idx = int(probs.argmax())
    p = float(probs[idx])
    if p < INTENT_THRESHOLD:
        return None, p
    # _model.classes_ are integer indices that map into INTENTS
    return INTENTS[_model.classes_[idx]], p