"""
Robust RAG Q&A with query-shape-aware retrieval and prompting.

Query shapes handled today (Phase 1 + retrieval-first Phase 2 lite):
  - Single-subject:   "what did Rajshree quote for hex bolts?"
  - Existence:        "do we have vendor X?"
  - Multi-vendor list:"which vendors sell acetone?"
  - Comparison:       "compare acetone prices across vendors"
  - Filter/find:      "who can supply acetone fastest?", "non-ISO vendors"

Query shapes better served by Phase 2 add-on A (structured facts DB):
  - Date-scoped:      "quotes in past week"
  - Numeric filter:   "vendors with price under 500", "purity >= 99%"
"""
import re
from typing import Any, Optional, cast

from . import tfidf_store, vector_store
from .config import DOC_STORE
from .llm_client import chat as llm_chat
from .vendor_index import get_vendor_index

# ---------- Prompts ----------

_EXPAND_PROMPT = """Rewrite the user's procurement question to improve
document search. Rules:

1. Check the domain of the question. If it is NOT about procurement, vendors, quotes, purchase orders, contracts, certifications, prices, deliveries, or any business supplier topic, return "Unrelated to procurement" and do NOT attempt to expand it.
2. KEEP vendor names, company names, product codes, part numbers,
   certification names (ISO, BIS), HSN codes, and any capitalised proper
   nouns EXACTLY as written.
3. Only add synonyms for GENERIC common nouns (e.g. "bolt" ->
   "bolt bolts fastener screw").
4. Do NOT invent related entities.
5. Keep the expansion under 20 words.

Return ONLY the expanded search string on one line. No explanation.

Question: {q}

Expanded search string:"""


_ANSWER_PROMPT = """You are CIVSA, an assistant answering procurement questions
strictly from the excerpts below. Multiple vendors' documents may be present.

CORE RULES:
- Use ONLY the excerpts. Never invent.
- Answer the SPECIFIC question asked. Do not add unrelated summaries.
- Cite each fact as [source: filename].
- If the excerpts don't contain info to answer, say "Not found in the documents."

RESPONSE SHAPE - pick based on what the question is asking:

(A) SINGLE-SUBJECT question about ONE specific vendor/product/document
    (e.g. "what did Rajshree quote?", "brief overview of Nirmala"):
    - 2-4 short sentences about ONLY that subject.
    - Ignore excerpts about other vendors.
    - Prose, not lists.

(B) EXISTENCE / yes-no question ("do we have vendor X?", "is there a quote for Y?"):
    - Answer in ONE short sentence starting with "Yes" or "No".

(C) MULTI-VENDOR LIST or FILTER question ("which vendors sell X?",
    "list all non-ISO vendors", "who supplies acetone?"):
    - Return a bulleted list, one line per vendor.
    - Format: "- <VendorName>: <one-line relevant fact> [source: filename]"
    - Include ONLY vendors that match the filter/criterion.
    - If no vendors match, say so plainly.

(D) COMPARISON question ("compare prices", "who is cheapest/fastest?"):
    - List each vendor with the compared attribute on one line each.
    - End with a one-sentence conclusion ("Vendor X is cheapest at ...").

(E) OVERVIEW of a specific entity: 3-5 short sentences about ONLY that entity.

For (C) and (D), use ALL relevant excerpts across all vendors. Do not
restrict to a single vendor unless the question names one specifically.

Question: {q}

Excerpts:
{ctx}

Answer:"""


# ---------- Query shape detection ----------

# Signals that the query wants cross-vendor / aggregation / filtering
# behaviour — even if a vendor name is mentioned, we should NOT scope
# retrieval to that one vendor.
_MULTI_VENDOR_SIGNALS = [
    "all companies", "all vendors", "all suppliers",
    "list of vendors", "list of companies", "list of suppliers", "list all",
    "which vendors", "which companies", "which suppliers",
    "who supplies", "who sells", "who can supply", "who has", "who provides",
    "who don't have", "who does not have", "who dont have", "who lack",
    "compare", "comparison", "vs ", " versus ",
    "cheapest", "lowest price", "most expensive", "highest price",
    "fastest", "quickest", "asap", "shortest lead", "longest lead",
    "soonest", "earliest",
    "non-iso", "non iso", "without iso", "without cert", "no cert",
    "price range", "between ", " and rs", "under rs", "above rs",
    "purity range", "purity above", "purity below",
    "in the past", "in last", "in the last", "last week", "last month",
    "recent quotes", "recently",
    "average price", "median price", "total spend",
]


def _is_multi_vendor_query(query: str) -> bool:
    ql = query.lower()
    return any(pat in ql for pat in _MULTI_VENDOR_SIGNALS)


def _is_simple_query(query: str) -> bool:
    """Simple = short, factual → use fast model."""
    ql = query.lower()
    return any(pat in ql for pat in [
        "do we have", "is there", "does exist", "does it have",
        "gstin", "gst number", "payment terms", "payment days",
        "hsn code", "on file", "in our system",
    ])


# ---------- Vendor detection ----------

def _list_vendors() -> list[str]:
    if not DOC_STORE.exists():
        return []
    out = []
    for p in DOC_STORE.iterdir():
        if p.is_dir() and any(
            any(d.iterdir()) for d in p.iterdir() if d.is_dir()
        ):
            out.append(p.name)
    return sorted(out)





def _detect_vendor(query: str) -> Optional[str]:
    """
    Look for a vendor mention inside a free-text query.

    Strategy:
      1. Try the whole query as a candidate — usually wrong but cheap.
      2. Try each capitalised phrase in the query (proper-noun heuristic).
      3. Try each n-gram of length 2-4 words.
      Return the highest-confidence hit above the resolver's threshold.
    """
    idx = get_vendor_index()
    q = query.strip()
    if not q:
        return None

    candidates: list[str] = []

    # Capitalised phrases: consecutive Capitalised-Words.
    caps = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", q)
    candidates.extend(caps)

    # 2-, 3-, 4-word windows.
    words = q.split()
    for n in (4, 3, 2):
        for i in range(len(words) - n + 1):
            candidates.append(" ".join(words[i : i + n]))

    # De-duplicate while preserving order.
    seen = set()
    ordered = []
    for c in candidates:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            ordered.append(c)

    best: Optional[tuple[str, float]] = None
    for c in ordered:
        r = idx.resolve(c)
        if r.canonical:
            if best is None or r.confidence > best[1]:
                best = (r.canonical, r.confidence)

    return best[0] if best else None


# ---------- Query expansion ----------

def _expand_query(query: str) -> str:
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query)
    if caps:
        print(f"[qa] skipping expansion - proper nouns detected: {caps}",
              flush=True)
        return query
    try:
        raw = llm_chat(
            messages=[{"role": "user",
                       "content": _EXPAND_PROMPT.format(q=query)}],
            model_size="fast", temperature=0, max_tokens=60,
        ).strip()
        return raw if raw else query
    except Exception:
        return query


# ---------- Vector search helpers ----------

def _get_collection():
    return (vector_store._get_collection() # type: ignore
            if hasattr(vector_store, "_get_collection")
            else vector_store._collection)


def _vector_query_vendor(expanded: str, vendor: str, k: int = 15):
    """Chroma query restricted to one vendor."""
    return _get_collection().query(
        query_texts=[expanded], n_results=k,
        where=cast(Any, {"vendor": {"$eq": vendor}}),
    )


# ---------- Main entry ----------

def answer(query: str, intent: str | None = None) -> dict:
    is_multi   = _is_multi_vendor_query(query)
    is_simple  = _is_simple_query(query)
    # Only scope to one vendor when the query names it AND is NOT a
    # multi-vendor question (comparison, list, filter, aggregation).
    vendor_hint = None if is_multi else _detect_vendor(query)

    print(f"[qa] shape: multi_vendor={is_multi} simple={is_simple} "
          f"vendor_hint={vendor_hint}", flush=True)

    expanded = _expand_query(query)
    if expanded.lower() == "unrelated to procurement":
        return {"answer": "Not procurement-related.", "sources": [],
                "route_method": "unrelated"}
    print(f"[qa] original: {query!r}", flush=True)
    print(f"[qa] expanded: {expanded!r}", flush=True)

    # ---- Path A: vendor-scoped (single-subject question) ----
    if vendor_hint:
        hits = _vector_query_vendor(expanded, vendor_hint, k=15)
        docs  = hits["documents"][0]  if hits["documents"]  else [] # type: ignore
        metas = hits["metadatas"][0] if hits["metadatas"] else [] # type: ignore
        if docs:
            print(f"[qa] vendor-scoped: {len(docs)} chunks from "
                  f"{vendor_hint}", flush=True)
            return _generate(query, is_simple, is_multi,
                             docs, metas, route="vendor_scoped")
        print(f"[qa] vendor {vendor_hint} has no chunks - falling back",
              flush=True)

    # ---- Path B: broad retrieval (multi-vendor OR no vendor detected) ----
    #
    # For multi-vendor queries we want as much cross-vendor coverage as
    # possible. Skip the TF-IDF prefilter's narrow-set trap by retrieving
    # from the WHOLE corpus and only using TF-IDF hits as a hint bias.

    if is_multi:
        # Broader k, no source filter — need enough vendors represented
        hits = vector_store.query(expanded, k=40)
        docs  = hits["documents"][0]  if hits["documents"]  else []
        metas = hits["metadatas"][0] if hits["metadatas"] else []
        print(f"[qa] multi-vendor broad retrieval: {len(docs)} chunks",
              flush=True)
    else:
        # Standard path with TF-IDF prefilter + safety fallbacks
        tfidf_top = tfidf_store.query(expanded, k=100)
        candidate_sources = None
        if tfidf_top:
            candidate_sources = sorted({
                tfidf_store.get_chunk(i)[1]["source"] for i in tfidf_top
            })
            print(f"[qa] tfidf shortlist: {len(candidate_sources)} doc(s)",
                  flush=True)

        if candidate_sources and len(candidate_sources) < 3:
            print(f"[qa] tfidf too narrow ({len(candidate_sources)}) - "
                  f"dropping source filter", flush=True)
            candidate_sources = None

        hits = vector_store.query(expanded, k=20,
                                  source_filter=candidate_sources)
        docs  = hits["documents"][0]  if hits["documents"]  else []
        metas = hits["metadatas"][0] if hits["metadatas"] else []

        if candidate_sources and len(docs) < 10:
            print("[qa] broadening to whole corpus", flush=True)
            hits2 = vector_store.query(expanded, k=20)
            d2 = hits2["documents"][0]  if hits2["documents"]  else []
            m2 = hits2["metadatas"][0] if hits2["metadatas"] else []
            seen = {(m["source"], m.get("para")) for m in metas}
            for d, m in zip(d2, m2):
                key = (m["source"], m.get("para"))
                if key not in seen:
                    docs.append(d); metas.append(m); seen.add(key)
            print(f"[qa] union total: {len(docs)}", flush=True)

    if not docs:
        return {"answer": "No relevant documents found.", "sources": [],
                "route_method": "rag_empty"}

    return _generate(query, is_simple, is_multi, docs, metas, route="rag")


# ---------- Generation ----------

def _generate(query: str, is_simple: bool, is_multi: bool,
              docs: list, metas: list, route: str) -> dict:
    print(f"[qa] retrieved {len(docs)} candidates", flush=True)

    # Show a per-vendor coverage summary in the log so you can see whether
    # multi-vendor queries actually retrieved from multiple vendors.
    by_vendor: dict[str, int] = {}
    for m in metas:
        by_vendor[m["vendor"]] = by_vendor.get(m["vendor"], 0) + 1
    print(f"[qa] vendor coverage: {by_vendor}", flush=True)

    # Multi-vendor queries need more context to see across vendors
    ctx_cap = 20 if is_multi else 10
    ctx = "\n\n".join(
        f"[{i+1}] {d}\n(source: {m['source']}, vendor: {m['vendor']}, "
        f"para {m.get('para', '?')}, labels: {m.get('labels', 'unlabelled')})"
        for i, (d, m) in enumerate(zip(docs[:ctx_cap], metas[:ctx_cap]))
    )

    # Model + token budget
    if is_multi:
        model_size = "big"    # lists/comparisons need reasoning
        max_tokens = 600
    elif is_simple:
        model_size = "fast"
        max_tokens = 200
    else:
        model_size = "big"
        max_tokens = 400
    print(f"[qa] model_size={model_size} max_tokens={max_tokens}",
          flush=True)

    try:
        text = llm_chat(
            messages=[{"role": "user",
                       "content": _ANSWER_PROMPT.format(q=query, ctx=ctx)}],
            model_size=model_size, temperature=0, max_tokens=max_tokens,
        )
    except Exception as e:  # noqa: BLE001
        return {"answer": f"LLM error: {e}", "sources": metas[:5]}

    if not text or not text.strip():
        return {"answer": "The model returned an empty response.",
                "sources": metas[:5]}

    if _hallucination(text, docs):
        text = "Answer could not be verified against the documents."

    return {"answer": text, "sources": metas[:5], "route_method": route}


# ---------- Faithfulness ----------

def _hallucination(answer: str, ctx: list[str]) -> bool:
    quoted = re.findall(r'"([^"]+)"', answer)
    return any(len(q) > 15 and not any(q in c for c in ctx) for q in quoted)