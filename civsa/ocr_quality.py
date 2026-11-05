"""
Per-document checks for OCR'd uploads (images + scanned PDF pages).

`assess_document` decides, for ONE document on its own, whether its text
can be trusted. Vendor-level identity rules (two PAN cards, PAN/GSTIN
conflicts) live in civsa/review.py because they need the other documents.

Reason codes (all mean "needs manual review"):
  ocr_timeout, no_text, low_confidence, too_little_text, blurry,
  low_resolution, garbled          - image could not be read reliably
  not_procurement                  - readable, but not a procurement document
  label_mismatch                   - user label disagrees with the content
  id_unreadable                    - PAN/GST document without a valid ID number
  multiple_ids_in_doc              - one PAN/GST document showing several IDs
  name_mismatch                    - name on PAN/GST document is not this vendor
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .extract import ExtractionResult, OcrMetrics, reread_value_after_label
from .identity import (
    detect_doc_type,
    doc_type_from_labels,
    extract_holder_names,
    find_ids,
    name_matches,
)

ID_DOC_TYPES = {"pan_card", "gst_certificate"}

REASON_TEXT = {
    "ocr_timeout":         "OCR timed out (image is probably very noisy)",
    "no_text":             "no readable text found",
    "low_confidence":      "low OCR confidence",
    "too_little_text":     "too little text for this kind of document",
    "blurry":              "image is blurry",
    "low_resolution":      "image resolution is too low",
    "garbled":             "OCR output is mostly symbols",
    "not_procurement":     "does not look like a procurement document",
    "label_mismatch":      "chosen label does not match the content",
    "id_unreadable":       "no valid PAN/GSTIN could be read",
    "multiple_ids_in_doc": "document shows more than one PAN/GSTIN",
    "name_mismatch":       "name on the document does not match the vendor",
    "multiple_pan_docs":   "vendor has more than one PAN document",
    "multiple_gst_docs":   "vendor has more than one GST document",
    "pan_gstin_mismatch":  "PAN does not match the PAN inside the GSTIN",
    "conflicts_with_quote": "PAN/GSTIN differs from the one printed on the vendor's quote",
}

_ID_LABEL = {
    "pan":   re.compile(r"account\s+number|\bpan\b\s*(?:no|number)?\s*:", re.IGNORECASE),
    "gstin": re.compile(r"registration\s+number|gstin", re.IGNORECASE),
}

_PROCUREMENT_TERMS = re.compile(
    r"\b(quotation|quote|invoice|purchase\s+order|p\.?o\.?\s*no|gstin|gst|hsn|sku|"
    r"unit\s+price|rate|qty|quantity|subtotal|grand\s+total|total|payment\s+terms?|"
    r"delivery|lead\s+time|validity|freight|warranty|inr|certificate|iso|"
    r"supplier|vendor|buyer|tax|pan|cin|specification|batch|consignee)\b",
    re.IGNORECASE)


@dataclass
class Assessment:
    status: str                         # "ok" | "needs_review"
    reasons: list[str] = field(default_factory=list)
    doc_type: str = "other"             # pan_card | gst_certificate | quote | other
    method: str = "native"              # native | ocr | mixed
    ocr: dict | None = None
    id_type: str | None = None          # "pan" | "gstin" for identity documents
    id_value: str | None = None
    holder_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def ocr_reasons(m: OcrMetrics, id_doc: bool) -> list[str]:
    """Readability problems from OCR metrics alone."""
    if m.timed_out:
        return ["ocr_timeout"]
    if m.n_words == 0:
        return ["no_text"]
    r = []
    if m.mean_conf < config.OCR_MIN_CONF:
        r.append("low_confidence")
    if not id_doc and m.n_words < config.OCR_MIN_WORDS:
        r.append("too_little_text")
    if m.sharpness < config.OCR_MIN_SHARPNESS:
        r.append("blurry")
    if max(m.width, m.height) < config.OCR_MIN_LONG_SIDE:
        r.append("low_resolution")
    if m.junk_ratio > config.OCR_MAX_JUNK_RATIO:
        r.append("garbled")
    return r


def is_procurement(text: str, use_llm: bool = True) -> tuple[bool, str]:
    """Keyword score first; ask the fast LLM only for borderline text."""
    hits = len(_PROCUREMENT_TERMS.findall(text))
    if hits >= config.RELEVANCE_MIN_HITS:
        return True, f"keywords:{hits}"
    if hits == 0 or not use_llm:
        return False, f"keywords:{hits}"
    from .llm_client import chat
    try:
        raw = chat(messages=[{"role": "user", "content":
            "Is the following text from a business or procurement document "
            "(quotation, invoice, purchase order, certificate, tax/company "
            "registration, delivery note)? Answer only yes or no.\n\n"
            f"{text[:1500]}\n\nAnswer:"}],
            model_size="fast", temperature=0, max_tokens=3)
    except Exception:  # noqa: BLE001
        return False, f"keywords:{hits},llm:error"
    yes = raw.strip().lower().startswith("y")
    return yes, f"keywords:{hits},llm:{'yes' if yes else 'no'}"


def assess_document(res: ExtractionResult, vendor: str, labels: list[str],
                    use_llm: bool = True) -> Assessment:
    text = res.text or ""
    content_type = detect_doc_type(text)
    label_type = doc_type_from_labels(labels)
    # Content wins for identity documents (it is unambiguous); otherwise
    # trust the user's label, falling back to what the content suggests.
    doc_type = (content_type if content_type in ID_DOC_TYPES
                else label_type or content_type)
    a = Assessment(status="ok", doc_type=doc_type, method=res.method,
                   ocr=res.ocr.to_dict() if res.ocr else None)
    is_id = doc_type in ID_DOC_TYPES

    if res.ocr:
        a.reasons += ocr_reasons(res.ocr, id_doc=is_id)

    if label_type and (
            (content_type in ID_DOC_TYPES and label_type != content_type)
            or (label_type in ID_DOC_TYPES and content_type == "quote")):
        a.reasons.append("label_mismatch")

    if is_id:
        ids = find_ids(text)
        a.id_type = "pan" if doc_type == "pan_card" else "gstin"
        found = ids[a.id_type]
        if not found and res.layout is not None:
            # Targeted second pass on the ID field (e.g. "0000" read as "OQOO")
            for reading in reread_value_after_label(res.layout, _ID_LABEL[a.id_type]):
                found = find_ids(reading)[a.id_type]
                if found:
                    break
        if not found:
            a.reasons.append("id_unreadable")
        else:
            a.id_value = found[0]
            if len(found) > 1:
                a.reasons.append("multiple_ids_in_doc")
        a.holder_names = extract_holder_names(text, doc_type)
        if a.holder_names and not any(name_matches(n, vendor) for n in a.holder_names):
            a.reasons.append("name_mismatch")
    elif res.ocr and not a.reasons:
        # Only OCR'd documents get the relevance check: native PDFs/DOCX
        # behave exactly as before.
        ok, _why = is_procurement(text, use_llm=use_llm)
        if not ok:
            a.reasons.append("not_procurement")

    a.reasons = list(dict.fromkeys(a.reasons))
    a.status = "needs_review" if a.reasons else "ok"
    return a
