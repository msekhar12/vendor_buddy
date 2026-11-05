"""
OCR assessment against the generated SPECIMEN images
(PYTHONPATH=. python scripts/create_demo_images.py). Needs Tesseract;
skipped otherwise. Takes ~40 s (the heavy-noise image runs into the OCR
timeout on purpose).
"""
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "TEST-DATA" / "images"
EXPECTED = ROOT / "expected.json"

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None or not EXPECTED.exists(),
    reason="needs tesseract and TEST-DATA/images (run scripts/create_demo_images.py)")

VENDOR_LEVEL = {"multiple_pan_docs", "multiple_gst_docs", "conflicts_with_quote",
                "pan_gstin_mismatch"}


def _cases(kind):
    if not EXPECTED.exists():
        return []
    return json.loads(EXPECTED.read_text())[kind]


@pytest.mark.parametrize("case", _cases("quality"), ids=lambda c: c["file"])
def test_quality_ladder(case):
    from civsa.extract import extract_with_quality
    from civsa.ocr_quality import assess_document
    a = assess_document(extract_with_quality(ROOT / case["file"]),
                        "Coromandel Solvents Pvt Ltd", [], use_llm=False)
    assert a.status == case["expect_status"], a.reasons
    if case["expect_reasons_any"]:
        assert set(a.reasons) & set(case["expect_reasons_any"]), a.reasons


@pytest.mark.parametrize("case", _cases("uploads"), ids=lambda c: c["file"])
def test_upload_documents_on_their_own(case):
    """Per-document result; vendor-level reasons are covered in test_review_flow."""
    from civsa.extract import extract_with_quality
    from civsa.ocr_quality import assess_document
    a = assess_document(extract_with_quality(ROOT / case["file"]),
                        case["vendor"], case["labels"], use_llm=False)
    e = case["expect"]
    own = [r for r in e.get("reasons_any", []) if r not in VENDOR_LEVEL]
    assert a.doc_type == e["doc_type"]
    assert a.status == ("needs_review" if own else "ok"), a.reasons
    if own:
        assert set(a.reasons) & set(own), a.reasons
    if e["doc_type"] == "pan_card" and e.get("pan") and not own:
        assert a.id_value == e["pan"]
    if e["doc_type"] == "gst_certificate" and e.get("gstin"):
        assert a.id_value == e["gstin"]
