"""
Train the Stage-2 domain classifier and Stage-3 intent router from
data/gate_train.jsonl.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from civsa.config import EMBED_MODEL_PATH, INTENTS

DATA = Path("data")
rows = [json.loads(l) for l in (DATA / "gate_train.jsonl").open()]

manual_path = DATA / "gate_train_manual.jsonl"
if manual_path.exists():
    manual_rows = [json.loads(l) for l in manual_path.open()]
    rows.extend(manual_rows)
    print(f"Merged {len(manual_rows)} manual training rows.")

encoder = SentenceTransformer(EMBED_MODEL_PATH)
texts = [r["text"] for r in rows]
# Ensure X is a numpy array (works for boolean-mask indexing later)
X = np.asarray(
    encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
)

# --- Stage 2: binary in-domain classifier ---
y_dom = [r["label"] for r in rows]
Xtr, Xte, ytr, yte = train_test_split(X, y_dom, test_size=0.2,
                                      random_state=42, stratify=y_dom)
dom = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print("=== Stage 2 (in-domain vs off-topic) ===")
print(classification_report(yte, dom.predict(Xte)))
(DATA / "gate_domain.pkl").write_bytes(pickle.dumps(dom))

# --- Stage 3: multi-class intent router (in-domain rows only) ---
in_mask = np.array([r["label"] == 1 for r in rows])
Xin = X[in_mask]
yin = [INTENTS.index(r["intent"]) for r, m in zip(rows, in_mask) if m]

Xtr, Xte, ytr, yte = train_test_split(Xin, yin, test_size=0.2,
                                      random_state=42, stratify=yin)
intent = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print("=== Stage 3 (intent router) ===")
present = sorted(set(yin))
print(classification_report(
    yte, intent.predict(Xte),
    labels=present,
    target_names=[INTENTS[i] for i in present]))
(DATA / "gate_intent.pkl").write_bytes(pickle.dumps(intent))

print("\nWrote data/gate_domain.pkl and data/gate_intent.pkl")