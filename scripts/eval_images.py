"""
End-to-end check of image uploads, PAN/GST documents and manual review.

Uploads every file in TEST-DATA/images/expected.json through POST /api/upload
into a THROWAWAY data folder (never the real documents/ or indexes/), then
checks review statuses, PAN/GSTIN answers and the reviewer actions.

    PYTHONPATH=. python scripts/eval_images.py            # deterministic, no LLM
    PYTHONPATH=. python scripts/eval_images.py --llm      # real LLM fact extraction
    PYTHONPATH=. python scripts/eval_images.py --keep     # keep the temp folder

Without --llm, quote fact extraction is replaced by a rule-based stand-in
(PAN/GSTIN read with civsa.identity) so the identity rules are tested
without depending on model output.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "TEST-DATA" / "images"


def _reexec_isolated(args: argparse.Namespace) -> int:
    tmp = tempfile.mkdtemp(prefix="civsa-eval-images-")
    env = {**os.environ, "CIVSA_DATA_ROOT": tmp, "PYTHONPATH": str(ROOT),
           "CIVSA_SQL_TRACE": "0"}
    cmd = [sys.executable, __file__, "--inner"] + (["--llm"] if args.llm else [])
    print(f"Data root: {tmp}")
    rc = subprocess.call(cmd, env=env, cwd=ROOT)
    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"Kept {tmp}")
    return rc


def main_inner(use_llm: bool) -> int:
    from civsa import config
    assert config.DATA_ROOT != config.ROOT, "refusing to run against the real corpus"

    if not use_llm:
        from civsa import identity, llm_client, structured

        def fake_extract(text, vendor, source, llm_chat_fn):
            ids = identity.find_ids(text)
            return {"vendor_facts": {"gstin": (ids["gstin"] or [None])[0],
                                     "pan": (ids["pan"] or [None])[0]},
                    "commercial_terms": {}, "quote_items": []}
        structured.extract_facts = fake_extract

        def no_llm(*a, **k):
            raise RuntimeError("LLM disabled in eval")
        llm_client.chat = no_llm

    from fastapi.testclient import TestClient

    from civsa import review
    from civsa.api import app
    from civsa.qa import answer

    client = TestClient(app)
    exp = json.loads((IMAGES / "expected.json").read_text())
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))

    def upload(vendor, rel, labels, extra: bytes = b"", name: str | None = None):
        data = (IMAGES / rel).read_bytes() + extra
        r = client.post("/api/upload",
                        data={"vendor": vendor, "labels": labels, "uploaded_by": "eval"},
                        files={"file": (name or Path(rel).name, data)})
        r.raise_for_status()
        return r.json()

    def entry(vendor, filename):
        for folder in sorted((config.DOC_STORE / vendor).iterdir()):
            for d in json.loads((folder / "manifest.json").read_text())["docs"]:
                if d["filename"] == filename:
                    return folder.name, d
        raise KeyError(filename)

    # ---- 1. upload everything, scenario by scenario (quotes first) ----
    print("\n== Uploads ==")
    for u in sorted(exp["uploads"], key=lambda u: (u["scenario"], u["labels"] != ["quote"])):
        upload(u["vendor"], u["file"], u["labels"])

    # ---- 2. final per-document status (after all vendor rules ran) ----
    print("\n== Review status per document ==")
    for u in exp["uploads"]:
        _date, d = entry(u["vendor"], Path(u["file"]).name)
        e = u["expect"]
        want = "needs_review" if e["status"] == "needs_review" else "auto_ok"
        ok = d["review_status"] == want and d.get("doc_type") == e["doc_type"]
        if e.get("reasons_any"):
            ok = ok and bool(set(d.get("review_reasons", [])) & set(e["reasons_any"]))
        check(f"{u['scenario']:20s} {Path(u['file']).name}", ok,
              f"got {d['review_status']} {d.get('review_reasons')} type={d.get('doc_type')}")

    # ---- 3. questions ----
    print("\n== PAN / GSTIN answers ==")
    def ask(q, must=(), must_not=()):
        a = answer(q)["answer"]
        ok = all(m.lower() in a.lower() for m in must) and not any(m.lower() in a.lower() for m in must_not)
        check(f"Q: {q}", ok, a.replace("\n", " | "))
    ask("what is Coromandel's PAN?", ["AABCC4821K", "PAN card"])
    ask("what is Hooghly's GSTIN?", ["19AADCH7310M1ZD"])
    ask("what is Hooghly's PAN?", ["AADCH7310M", "derived from the GSTIN"])
    ask("what is Rajshree's PAN?", ["AAAAA0000A"])
    ask("what is Pink City's PAN?", ["pending manual review", "2 PAN card documents"],
        must_not=["PAN: AAFCP"])
    ask("what is Malabar's GSTIN?", ["pending manual review"])
    ask("what is Narmada's PAN?", ["pending manual review", "AAKCN6618B", "unverified"])
    ask("what is Konkan's PAN?", ["pending manual review"], must_not=["AALCK2290F ("])

    # ---- 4. reviewer actions ----
    print("\n== Reviewer actions ==")
    v_pink = "Pink City Fasteners Pvt Ltd"
    d1, _ = entry(v_pink, "pinkcity_pan_1.png")
    d2, _ = entry(v_pink, "pinkcity_pan_2.png")
    # deleting the extra PAN document must NOT clear the remaining flag
    r = client.delete(f"/api/vendor/{v_pink}/doc",
                      params={"date": d2, "filename": "pinkcity_pan_2.png"})
    check("delete extra PAN doc -> 200", r.status_code == 200, r.text)
    _, e1 = entry(v_pink, "pinkcity_pan_1.png")
    check("remaining PAN doc stays needs_review after delete",
          e1["review_status"] == "needs_review", str(e1["review_status"]))
    r = client.post("/api/review/resolve", data={"vendor": v_pink, "date": d1,
                    "filename": "pinkcity_pan_1.png", "action": "approve", "reviewer": "eval"})
    check("approve remaining PAN doc", r.status_code == 200 and r.json()["status"] == "approved", r.text)
    ask("what is Pink City's PAN?", ["AAFCP5562Q", "PAN card"], must_not=["pending"])

    v_mal = "Malabar Lab Supplies Pvt Ltd"
    r = client.post("/api/review/resolve-group", data={"vendor": v_mal, "id_type": "pan",
                    "action": "approve_all"})
    check("approve_all rejected for PAN group", r.status_code == 400, r.text)
    r = client.post("/api/review/resolve-group", data={"vendor": v_mal, "id_type": "gstin",
                    "action": "approve_all", "reviewer": "eval"})
    check("approve_all for same-PAN GST group", r.status_code == 200, r.text)
    ask("what is Malabar's GSTIN?", ["32AAJCM9034E1ZM", "29AAJCM9034E1Z9"], must_not=["pending"])
    # a third GST document re-opens review for all of them
    upload(v_mal, "uploads/malabar/malabar_gst_kerala.png", ["gst_certificate"],
           extra=b"\0", name="malabar_gst_kerala_copy.png")
    statuses = [entry(v_mal, f)[1]["review_status"] for f in
                ("malabar_gst_kerala.png", "malabar_gst_karnataka.png", "malabar_gst_kerala_copy.png")]
    check("3rd GST doc re-flags all (incl. approved)", statuses == ["needs_review"] * 3, str(statuses))

    v_narm = "Narmada Scientific Pvt Ltd"
    dn, _ = entry(v_narm, "narmada_pan.png")
    r = client.post("/api/review/resolve", data={"vendor": v_narm, "date": dn,
                    "filename": "narmada_pan.png", "action": "set_id", "id_value": "AAKC6618B"})
    check("set_id with badly formatted PAN rejected", r.status_code == 400, r.text)
    r = client.post("/api/review/resolve", data={"vendor": v_narm, "date": dn,
                    "filename": "narmada_pan.png", "action": "set_id", "id_value": "AAKCN6618B"})
    check("set_id with corrected PAN approves", r.status_code == 200 and r.json()["status"] == "approved", r.text)
    ask("what is Narmada's PAN?", ["AAKCN6618B", "PAN card"], must_not=["pending"])

    v_kon = "Konkan Chemicals Pvt Ltd"
    dk, _ = entry(v_kon, "konkan_pan.png")
    r = client.post("/api/review/resolve", data={"vendor": v_kon, "date": dk,
                    "filename": "konkan_pan.png", "action": "reject"})
    check("reject wrong-company PAN", r.status_code == 200 and r.json()["status"] == "rejected", r.text)
    ask("what is Konkan's PAN?", ["No pan on file"])

    # flagged quote-type image: approve indexes it
    v_q = "Coromandel Solvents Pvt Ltd"
    up = upload(v_q, "quality/irrelevant_menu.png", ["quote"])
    check("menu uploaded as quote is flagged", up["review_status"] == "needs_review"
          and "not_procurement" in up["review_reasons"], str(up["review_reasons"]))
    dm, _ = entry(v_q, "irrelevant_menu.png")
    r = client.post("/api/review/resolve", data={"vendor": v_q, "date": dm,
                    "filename": "irrelevant_menu.png", "action": "reject"})
    check("reject irrelevant image", r.json().get("status") == "rejected", r.text)
    q = client.get("/api/review").json()
    check("review queue lists vendors with flags", q["count"] >= 1, str(q["count"]))

    # ---- 5. two PAN docs arriving at the same moment ----
    print("\n== Concurrency ==")
    from civsa.storage import LocalDriver
    from civsa.vendor_index import refresh_vendor_index
    store = LocalDriver(config.DOC_STORE)
    v_c = "Concurrent Test Vendor Pvt Ltd"
    paths = []
    for i, rel in enumerate(["uploads/coromandel/coromandel_pan.png",
                             "uploads/pinkcity/pinkcity_pan_1.png"]):
        paths.append(store.put(v_c, "2026-11-04", f"pan_{i}.png", (IMAGES / rel).read_bytes()))
    refresh_vendor_index()
    threads = [threading.Thread(target=review.process_upload,
                                args=(p, v_c, ["pan_card"], {"filename": p.name, "labels": ["pan_card"],
                                                             "sha256": str(i)}, False))
               for i, p in enumerate(paths)]
    for t in threads: t.start()
    for t in threads: t.join()
    sts = [review.get_entry(p)["review_status"] for p in paths]
    check("simultaneous PAN uploads both flagged", sts == ["needs_review", "needs_review"], str(sts))

    n_fail = sum(1 for _n, ok, _d in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    sys.exit(main_inner(a.llm) if a.inner else _reexec_isolated(a))
