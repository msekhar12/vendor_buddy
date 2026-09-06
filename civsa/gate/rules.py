"""
Stage 1 of the query gate: cheap deterministic checks.

Rejects obvious garbage and prompt-injection attempts in ~1 ms so we don't
spend any LLM tokens on them.
"""
import re
from collections import defaultdict, deque
from time import time

from ..config import MAX_QUERY_TOKENS

_PROCUREMENT_ALLOW = re.compile(
    r"\b("
    r"vendor|supplier|company|firm|manufacturer|"
    r"quote|quotation|invoice|contract|po|purchase\s*order|rfq|"
    r"gstin|gst|pan|cin|hsn|sac|"
    r"iso[\s\-]?\d+|iso\s+certified|certification|"
    r"price|cost|rate|amount|total|payment|credit|"
    r"delivery|lead\s*time|freight|packaging|warranty|"
    r"chemical|reagent|solvent|acid|"
    r"address|location|phone|email|contact|manufacturer"   # ← field words
    r")\b",
    re.IGNORECASE,
)

_CAPITALISED_PHRASE = re.compile(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}")


def looks_like_procurement(query: str) -> bool:
    """
    True if the query is obviously in-domain. Two signals:
      (a) A word from the procurement vocab appears.
      (b) A capitalised phrase in the query resolves to a known vendor.
    """
    if _PROCUREMENT_ALLOW.search(query):
        return True

    # Vendor-name signal: any capitalised phrase that maps to a real vendor
    try:
        from ..vendor_index import get_vendor_index
        idx = get_vendor_index()
        for candidate in _CAPITALISED_PHRASE.findall(query):
            if idx.resolve(candidate).canonical:
                return True
    except Exception:  # noqa: BLE001
        pass

    return False

# Pattern list gathered from HackAPrompt + common jailbreak phrasings
_INJECTION = re.compile(
    r"(ignore\s+(previous|prior|all)|"
    r"you\s+are\s+now|"
    r"disregard\s+(previous|above|the)|"
    r"system\s*:|"
    r"</?system>|```system|"
    r"reveal\s+(the|your)?\s*(system\s+)?prompt|"
    r"jailbreak|"
    r"as\s+an?\s+ai)", re.IGNORECASE)

# Per-user sliding window for rate limit
_rate: defaultdict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT = 30              # queries per minute per user


def check(query: str, user: str = "anonymous") -> tuple[bool, str]:
    """Returns (passed, reason). reason is a code mapped to UI text elsewhere."""
    print(f"[gate] checking query (user={user}): {query[:50]}...", flush=True)
    if not query or not query.strip():
        print("[gate] empty query", flush=True)
        return False, "empty"
    if len(query.split()) > MAX_QUERY_TOKENS:
        print(f"[gate] query too long: {len(query.split())} words", flush=True)
        return False, "too_long"
    if _INJECTION.search(query):
        print("[gate] prompt-injection detected", flush=True)
        return False, "injection"
    if looks_like_procurement(query):
        print("[gate] fast-path: obvious procurement query", flush=True)
        return True, "obvious_procurement_vocab"

    # Sliding 60-second rate limit per user
    now = time()
    q = _rate[user]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False, "rate_limited"
    q.append(now)

    return True, "ok"