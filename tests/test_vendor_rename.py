"""
Vendor rename, end to end, in a throwaway CIVSA_DATA_ROOT (a subprocess,
because civsa reads the data root at import time). LLM disabled; quote facts
come from a rule-based stand-in. Needs tesseract + TEST-DATA/images.
"""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "TEST-DATA" / "images"

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None or not (IMAGES / "expected.json").exists(),
    reason="needs tesseract and TEST-DATA/images")

SCRIPT = textwrap.dedent(r'''
    import json, os
    from pathlib import Path
    from civsa import config, identity, llm_client, structured
    assert config.DATA_ROOT != config.ROOT
    structured.extract_facts = lambda text, vendor, source, llm_chat_fn: {
        "vendor_facts": {"gstin": (identity.find_ids(text)["gstin"] or [None])[0]},
        "commercial_terms": {}, "quote_items": []}
    def no_llm(*a, **k): raise RuntimeError("LLM disabled")
    llm_client.chat = no_llm

    from fastapi.testclient import TestClient
    from civsa import tfidf_store, vector_store
    from civsa.api import app
    from civsa.qa import _try_structured, retrieve
    c = TestClient(app)
    IMG = Path(os.environ["IMAGES"])
    OLD, NEW = "Coromandel Solvents Pvt Ltd", "Coromandel Solvents India Pvt Ltd"
    for rel, lab in [("uploads/coromandel/coromandel_quote.png", "quote"),
                     ("uploads/coromandel/coromandel_pan.png", "pan_card")]:
        r = c.post("/api/upload", data={"vendor": OLD, "labels": [lab]},
                   files={"file": (Path(rel).name, (IMG / rel).read_bytes())})
        assert r.status_code == 200, r.text
    c.post("/api/upload", data={"vendor": "Other Vendor", "labels": ["quote"]},
           files={"file": ("q.png", (IMG / "uploads/hooghly/hooghly_quote.png").read_bytes())})

    n_chroma = vector_store._collection.count()
    n_tfidf = len(tfidf_store._load()["metas"])

    # invalid / conflicting names
    assert c.post(f"/api/vendor/{OLD}/rename", data={"new_name": "a/b"}).status_code == 400
    assert c.post(f"/api/vendor/{OLD}/rename", data={"new_name": "other vendor"}).status_code == 409
    assert c.post(f"/api/vendor/Nope/rename", data={"new_name": "X"}).status_code == 404

    r = c.post(f"/api/vendor/{OLD}/rename", data={"new_name": f"  {NEW} "})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["new"] == NEW and res["chroma_chunks"] > 0 and res["sql_rows"] >= 2

    D = config.DOC_STORE
    assert not (D / OLD).exists() and (D / NEW).is_dir()
    date = next((D / NEW).iterdir()).name
    files = sorted(p.name for p in (D / NEW / date).iterdir())
    assert "coromandel_pan.png.ocr.txt" in files and "coromandel_quote.png" in files, files
    mf = json.loads((D / NEW / date / "manifest.json").read_text())
    assert mf["vendor"] == NEW

    old_prefix = str(D / OLD) + os.sep
    metas = vector_store._collection.get(include=["metadatas"])
    assert vector_store._collection.count() == n_chroma
    assert not any(m["source"].startswith(old_prefix) for m in metas["metadatas"])
    assert not any(i.startswith(old_prefix) for i in metas["ids"])
    assert any(m["vendor"] == NEW for m in metas["metadatas"])
    tf = tfidf_store._load()["metas"]
    assert len(tf) == n_tfidf and not any(m["source"].startswith(old_prefix) for m in tf)
    with structured._connect() as conn:
        for t in ("vendor_facts", "vendor_identity_docs"):
            rows = conn.execute(f"SELECT source_doc, canonical_vendor FROM {t}").fetchall()
            assert not any(r[0].startswith(old_prefix) for r in rows), t
            assert any(r[1] == NEW for r in rows), t

    # The PAN card says "COROMANDEL SOLVENTS PRIVATE LIMITED", which no longer
    # matches the new name ("India"), so it goes back to review. Questions
    # under the new name and the old one (alias) both reach the vendor.
    assert "name_mismatch" in str(res["review_changes"]), res["review_changes"]
    for q in ("what is Coromandel Solvents India's PAN?", f"what is {OLD}'s PAN?"):
        a = _try_structured(q)["answer"]
        assert NEW in a and "pending manual review" in a and "AABCC4821K" in a, (q, a)
    assert "33AABCC4821K1ZW" in _try_structured(f"what is {NEW}'s GSTIN?")["answer"]
    import civsa.qa as qa
    qa._expand_query = lambda q: q
    r = retrieve(f"what are {NEW}'s payment terms?", stop_at_structured=False)
    assert r["vendor_hint"] == NEW and r["docs"], r["vendor_hint"]

    # files open under the new name
    assert c.get(f"/api/vendor/{NEW}/doc/file",
                 params={"date": date, "filename": "coromandel_quote.png"}).status_code == 200

    # refused while one of the vendor's documents is still being indexed
    from civsa import indexer
    busy = D / NEW / date / "coromandel_quote.png"
    indexer.mark_queued(busy)
    r = c.post(f"/api/vendor/{NEW}/rename", data={"new_name": "Something Else"})
    assert r.status_code == 409 and "still being indexed" in r.text, r.text
    indexer._mark_done(busy)

    # case-only rename
    r = c.post(f"/api/vendor/{NEW}/rename", data={"new_name": NEW.upper()})
    assert r.status_code == 200, r.text
    assert sorted(p.name for p in D.iterdir() if p.is_dir()) == sorted([NEW.upper(), "Other Vendor"])
    print("RENAME-OK")
''')


def test_vendor_rename_end_to_end(tmp_path):
    env = {**os.environ, "CIVSA_DATA_ROOT": str(tmp_path), "PYTHONPATH": str(ROOT),
           "IMAGES": str(IMAGES), "CIVSA_SQL_TRACE": "0"}
    p = subprocess.run([sys.executable, "-c", SCRIPT], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert "RENAME-OK" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]
