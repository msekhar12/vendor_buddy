"""
PAN / GSTIN detection, validation and OCR repair.

Rule-based on purpose: these IDs have strict formats and a GSTIN carries a
check character, so regex + checksum is more reliable than an LLM and
lets us correct common OCR confusions safely.

  PAN   : AAAAA9999A  (5 letters, 4 digits, 1 letter; 4th letter = holder type)
  GSTIN : 99 AAAAA9999A 9 Z C
          state code (01-38) + PAN + entity no. + 'Z' + check character
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

PAN_RE   = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")

# 4th PAN character: holder type
_PAN_HOLDER_TYPES = set("CPHFATBLJG")

# Position masks: L = letter, D = digit, X = either
_PAN_MASK   = "LLLLLDDDDL"
_GSTIN_MASK = "DD" + _PAN_MASK + "XLX"    # 13th any, 14th must be 'Z', 15th any

# Common OCR confusions, applied only where the mask demands the other class
_TO_DIGIT  = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "1",
              "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
              "4": "A", "7": "T"}


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def gstin_check_char(first14: str) -> str:
    """Check character for the first 14 chars of a GSTIN (mod-36 scheme)."""
    total = 0
    for i, ch in enumerate(first14.upper()):
        prod = _ALNUM.index(ch) * (1 if i % 2 == 0 else 2)
        total += prod // 36 + prod % 36
    return _ALNUM[(36 - total % 36) % 36]


def validate_pan(pan: str) -> tuple[bool, str]:
    pan = (pan or "").upper()
    if not PAN_RE.match(pan):
        return False, "bad_format"
    if pan[3] not in _PAN_HOLDER_TYPES:
        return False, "bad_holder_type"
    return True, "ok"


def validate_gstin(gstin: str) -> tuple[bool, str]:
    g = (gstin or "").upper()
    if not GSTIN_RE.match(g):
        return False, "bad_format"
    if not 1 <= int(g[:2]) <= 38:
        return False, "bad_state_code"
    ok, why = validate_pan(g[2:12])
    if not ok:
        return False, f"embedded_pan_{why}"
    if gstin_check_char(g[:14]) != g[14]:
        return False, "bad_checksum"
    return True, "ok"


def pan_from_gstin(gstin: str) -> str:
    return gstin.upper()[2:12]


# ------------------------------------------------------------------
# OCR repair
# ------------------------------------------------------------------

# A real ID misread by OCR needs a couple of fixes at most; an ordinary
# word ("TECHNOLOGY") would need many. Cap the repairs so words never
# turn into IDs.
_MAX_REPAIRS = 2


def _apply_mask(token: str, mask: str) -> str | None:
    out, repairs = [], 0
    for ch, m in zip(token.upper(), mask):
        if m == "D" and not ch.isdigit():
            ch, repairs = _TO_DIGIT.get(ch, ch), repairs + 1
        elif m == "L" and not ch.isalpha():
            ch, repairs = _TO_LETTER.get(ch, ch), repairs + 1
        out.append(ch)
    return "".join(out) if repairs <= _MAX_REPAIRS else None


def repair_pan(token: str) -> str | None:
    t = re.sub(r"[^A-Za-z0-9]", "", token)
    if len(t) != 10:
        return None
    fixed = _apply_mask(t, _PAN_MASK)
    return fixed if fixed and validate_pan(fixed)[0] else None


def repair_gstin(token: str) -> str | None:
    t = re.sub(r"[^A-Za-z0-9]", "", token).upper()
    if len(t) != 15:
        return None
    masked = _apply_mask(t, _GSTIN_MASK)
    if masked is None:
        return None
    fixed = list(masked)
    if fixed[13] in ("2", "7"):          # 'Z' misread as 2 / 7
        fixed[13] = "Z"
    cand = "".join(fixed)
    if validate_gstin(cand)[0]:
        return cand
    # Entity number (13th) and check character (15th) can be letter or
    # digit, so the mask can't fix them. Try the usual confusions there
    # and accept only a checksum-valid result.
    swap = {**_TO_DIGIT, **_TO_LETTER}
    for e in {cand[12], swap.get(cand[12], cand[12])}:
        for c in {cand[14], swap.get(cand[14], cand[14])}:
            cand2 = cand[:12] + e + "Z" + c
            if validate_gstin(cand2)[0]:
                return cand2
    return None


def find_ids(text: str) -> dict[str, list[str]]:
    """
    All valid PANs and GSTINs in a text, after OCR repair. A PAN that is
    only the embedded part of a found GSTIN is not reported separately.
    Order of first appearance is kept; duplicates removed.
    """
    up = text.upper()
    gstins: list[str] = []
    pans: list[str] = []
    # Allow single spaces/dashes inside an ID ("33 AABCC 4821 K 1Z5")
    for m in re.finditer(r"[A-Z0-9](?:[A-Z0-9]|[ \-](?=[A-Z0-9])){9,40}", up):
        chunk = m.group(0)
        compact = re.sub(r"[ \-]", "", chunk)
        for i in range(0, max(1, len(compact) - 14)):
            g = repair_gstin(compact[i:i + 15])
            if g and g not in gstins:
                gstins.append(g)
    embedded = {pan_from_gstin(g) for g in gstins}
    for tok in re.findall(r"[A-Z0-9]{10}", re.sub(r"[ \-]", " ", up)):
        p = repair_pan(tok)
        if p and p not in pans and p not in embedded:
            pans.append(p)
    # Also catch PANs split by a single space ("AABCC 4821K")
    for m in re.finditer(r"\b([A-Z0-9]{5})\s([A-Z0-9]{4,5})\s?([A-Z0-9]?)\b", up):
        p = repair_pan("".join(m.groups()))
        if p and p not in pans and p not in embedded:
            pans.append(p)
    return {"gstin": gstins, "pan": pans}


# ------------------------------------------------------------------
# Document type + holder name
# ------------------------------------------------------------------

_PAN_DOC_SIGNALS = re.compile(
    r"permanent\s+account\s+number|income\s*tax\s+department|\bpan\s+card\b",
    re.IGNORECASE)
_GST_DOC_SIGNALS = re.compile(
    r"goods\s+and\s+services?\s+tax|form\s+gst\s+reg|gst\s+reg[\s\-]*0?6|"
    r"registration\s+certificate|certificate\s+of\s+registration",
    re.IGNORECASE)
_QUOTE_SIGNALS = re.compile(
    r"quotation|\bquote\b|unit\s+price|line\s+total|grand\s+total|"
    r"payment\s+terms|delivery", re.IGNORECASE)


def detect_doc_type(text: str) -> str:
    """'pan_card' | 'gst_certificate' | 'quote' | 'other' from content alone."""
    n_quote = len(_QUOTE_SIGNALS.findall(text))
    if _GST_DOC_SIGNALS.search(text) and n_quote < 3:
        return "gst_certificate"
    if _PAN_DOC_SIGNALS.search(text) and n_quote < 3:
        return "pan_card"
    if n_quote >= 2:
        return "quote"
    return "other"


def doc_type_from_labels(labels: list[str]) -> str | None:
    s = set(labels or [])
    if "pan_card" in s:
        return "pan_card"
    if "gst_certificate" in s:
        return "gst_certificate"
    if s & {"quote", "pricing_schedule"}:
        return "quote"
    return None


_NAME_PATTERNS = {
    "pan_card": [r"^\s*name(?:\s+of\s+(?:the\s+)?(?:holder|company|entity))?\s*[:\-]?\s*(.+)$"],
    "gst_certificate": [r"^\s*legal\s+name(?:\s+of\s+business)?\s*[:\-]?\s*(.+)$",
                        r"^\s*trade\s+name(?:,?\s+if\s+any)?\s*[:\-]?\s*(.+)$"],
}


def extract_holder_names(text: str, doc_type: str) -> list[str]:
    names = []
    for line in text.splitlines():
        for pat in _NAME_PATTERNS.get(doc_type, []):
            m = re.match(pat, line, re.IGNORECASE)
            if m and m.group(1).strip():
                names.append(m.group(1).strip())
    return names


_SUFFIX = re.compile(
    r"\b(private|pvt|limited|ltd|llp|inc|co|company|corp|corporation|"
    r"and|&|the|m/s)\b\.?", re.IGNORECASE)


def _norm_name(s: str) -> str:
    s = _SUFFIX.sub(" ", s.lower())
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens_match(x: str, y: str) -> bool:
    """One word against one word: equal, an abbreviation ("chem" /
    "chemicals", shorter side >= 3 letters), or a typo ("fastners" /
    "fasteners")."""
    if x == y:
        return True
    short, long_ = sorted((x, y), key=len)
    if len(short) >= 3 and long_.startswith(short):
        return True
    return min(len(x), len(y)) >= 4 and SequenceMatcher(None, x, y).ratio() >= 0.8


def name_matches(name: str, vendor: str) -> bool:
    """
    Company-name match on significant words, ignoring corporate suffixes
    (Pvt, Ltd, Private, Limited, Co, ...) and word order.

    Every word on EACH side must pair with a word on the other side, so an
    extra word means a different company: "Konkan" does not match "Konkan
    Marine Exports", while "Rajshree Fastners" matches "RAJSHREE FASTENERS
    PRIVATE LIMITED" and "Nirmala Chem" matches "Nirmala Chemicals".
    """
    a, b = _norm_name(name).split(), _norm_name(vendor).split()
    if not a or not b or len(a) != len(b):
        return False
    unused = list(b)
    for tok in a:
        hit = next((u for u in unused if _tokens_match(tok, u)), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True
