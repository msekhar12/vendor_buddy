"""
Stage 1 of the query gate: cheap deterministic checks.

Rejects obvious garbage and prompt-injection attempts in ~1 ms so we don't
spend any LLM tokens on them.
"""
import re
from collections import defaultdict, deque
from time import time

from ..config import MAX_QUERY_TOKENS

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
    if not query or not query.strip():
        return False, "empty"
    if len(query.split()) > MAX_QUERY_TOKENS:
        return False, "too_long"
    if _INJECTION.search(query):
        return False, "injection"

    # Sliding 60-second rate limit per user
    now = time()
    q = _rate[user]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False, "rate_limited"
    q.append(now)

    return True, "ok"