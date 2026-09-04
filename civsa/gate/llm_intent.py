"""LLM confirmation that a borderline query is procurement-related."""
from ..llm_client import chat as llm_chat

_PROMPT = """Is the following query about procurement, vendors, quotes,
purchase orders, contracts, certifications, prices, deliveries, or any
business supplier topic?

Answer with ONE word: "yes" or "no". No explanation.

Query: {query}

Answer:"""


def is_procurement(query: str) -> bool:
    try:
        raw = llm_chat(
            messages=[{"role": "user",
                       "content": _PROMPT.format(query=query)}],
            model_size="fast", temperature=0, max_tokens=10,
        ).strip().lower()
    except Exception as e:
        print(f"[llm_intent] error: {e}", flush=True)
        return False
    print(f"[llm_intent] raw response: {raw!r}", flush=True)
    return raw.startswith("y")