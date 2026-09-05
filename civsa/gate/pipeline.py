"""
Two-stage gate + LLM tie-break.

Stage 1 - deterministic rules (rules.py): injection, length, rate limit
Stage 2 - fast in-domain classifier (domain.py): trained on procurement queries
Stage 2.5 - LLM tie-break (llm_intent.is_procurement) for borderline cases

No intent classification - retrieval and generation are now handled by qa.py
without needing an intent label.
"""
from . import rules, domain, llm_intent
from .rules import looks_like_procurement

REFUSALS = {
    "empty":        "Empty query. Please type a question.",
    "too_long":     "Queries longer than 2000 words are not accepted.",
    "injection":    "This looks like a prompt-injection attempt. I only "
                    "answer procurement questions from your document store.",
    "rate_limited": "Too many queries. Please wait a minute and try again.",
    "off_topic":    "I only answer questions about your vendors, quotes, "
                    "contracts, POs and delivery history.",
}


def process(query: str, user: str = "anonymous") -> dict:
    # Stage 1 - rules
    ok, reason = rules.check(query, user)
    if not ok:
        return {"allowed": False, "reason": reason,
                "message": REFUSALS[reason]}
    
    # ---- NEW: fast-path bypass ----
    # If the query obviously contains procurement vocabulary, skip the
    # classifier and the LLM tie-break entirely.
    if looks_like_procurement(query):
        print("[gate] fast-path: obvious procurement, bypassing classifier",
              flush=True)
        return {"allowed": True, "route_method": "obvious_procurement_vocab"}
    
    # Stage 2 - fast in-domain classifier
    in_domain, p_dom = domain.is_in_domain(query)
    if in_domain:
        return {"allowed": True, "confidence_domain": p_dom,
                "route_method": "fast_classifier"}

    # Stage 2.5 - LLM tie-break
    print(f"[gate] classifier uncertain (p={p_dom:.2f}), asking LLM...",
          flush=True)
    if llm_intent.is_procurement(query):
        print("[gate] LLM confirmed: procurement", flush=True)
        return {"allowed": True, "confidence_domain": p_dom,
                "route_method": "llm_confirmed"}

    return {"allowed": False, "reason": "off_topic",
            "message": REFUSALS["off_topic"], "confidence": p_dom}