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

# ---- Table-row boost for price/quantity/spec questions -------------
_TABLE_QUERY_HINTS = re.compile(
    r"\b(price|cost|cheapest|expensive|quote|rate|amount|total|"
    r"quantity|qty|unit|delivery|lead\s*time|days|payment)\b",
    re.IGNORECASE,
)

def _boost_table_rows(results, query: str, boost: float = 0.15):
    """
    If the query looks like it's asking for a fact that typically lives in
    a table row, reduce the Chroma distance (better rank) of table_row /
    table_terminal chunks and re-sort the parallel arrays.

    Chroma's results are column-oriented:
      {documents:[[...]], metadatas:[[...]], distances:[[...]], ids:[[...]]}
    Lower distance = better match.
    """
    if not _TABLE_QUERY_HINTS.search(query):
        return results

    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    if not metas or not dists:
        return results

    # Apply the boost (as a distance reduction)
    for i, m in enumerate(metas):
        ctype = (m or {}).get("chunk_type", "")
        if ctype in ("table_row", "table_terminal"):
            dists[i] = max(0.0, float(dists[i]) - boost)

    # Re-sort every parallel array by the new distances
    docs = (results.get("documents") or [[]])[0]
    ids  = (results.get("ids")       or [[]])[0]
    order = sorted(range(len(dists)), key=lambda k: dists[k])

    results["documents"] = [[docs[k]  for k in order]] if docs else results.get("documents")
    results["metadatas"] = [[metas[k] for k in order]]
    results["distances"] = [[dists[k] for k in order]]
    if ids:
        results["ids"]   = [[ids[k]   for k in order]]

    return results


# ============================================================
# Structured-query router (Phase 2 Step 3)
# ============================================================
def _try_structured(query: str) -> dict | None:
    """
    Match the query against structured-query patterns. If one fires, hit
    SQLite and return a deterministic answer with citations. Return None
    to fall through to the RAG path.

    Ordering rule: MOST SPECIFIC patterns first, GENERIC catch-alls last.
    A specific pattern that succeeds returns immediately, so a query like
    "list vendors from Mumbai" is caught by the location filter before it
    reaches the bare "list all vendors" block.
    """
    from . import structured
    from .vendor_index import get_vendor_index

    ql = query.lower().strip()

    # Strip wrapping quotes so trailing-anchor regexes still work when
    # users paste a query enclosed in "..." or '...' or curly quotes.
    ql = ql.strip('"\'\u201c\u201d\u2018\u2019')
    ql = ql.strip()   # remove any whitespace freed by the quote strip

    # Normalise vendor synonyms so downstream patterns only match "vendor".
    ql = re.sub(
        r"\b(suppliers?|companies|company|firms?|manufacturers?)\b",
        "vendor", ql,
    )

    # ================================================================
    # GROUP A · Specific single-vendor queries
    # ================================================================

    # ---- Single-vendor ISO certification check ("is X iso certified?") ----
    if re.search(r"\biso\b", ql) and re.search(
        r"\b(is|does|do|has|have|got)\b", ql
    ):
        m = re.search(
            r"\b(?:is|does|do|has|have)\s+(.+?)\s+"
            r"(?:an?\s+|hold(?:s|ing)?\s+|got\s+|obtained\s+)?iso",
            ql, re.IGNORECASE,
        )
        if m:
            vendor_str = m.group(1).strip()
            vendor_str = re.sub(
                r"\s+(hold(s|ing)?|have|has|got|obtained)$",
                "", vendor_str, flags=re.IGNORECASE,
            )
            resolved = get_vendor_index().resolve(vendor_str)

            if not resolved.canonical:
                return {
                    "answer": f'No records for a vendor called "{vendor_str}" '
                              f"in our corpus.",
                    "sources": [],
                    "route_method": "structured_sql_empty",
                }
            canonical = resolved.canonical

            std_match = re.search(r"\biso[\s\-]*(9001|14001|17025)\b", ql)
            requested = std_match.group(1) if std_match else None

            facts = structured.get_all_facts_for_vendor(canonical)
            vf_rows = facts["vendor_facts"]
            if not vf_rows:
                return {
                    "answer": f"No structured facts extracted yet for "
                              f"{canonical}. Try re-indexing.",
                    "sources": [],
                    "route_method": "structured_sql_empty",
                }
            row = vf_rows[0]

            def _describe(val: int | None, label: str) -> str | None:
                if val == 1: return f"holds {label}"
                if val == 0: return f"is NOT {label} certified"
                return None

            if requested:
                key = f"iso_{requested}"
                label = (f"ISO {requested}" if requested != "17025"
                         else "ISO/IEC 17025")
                desc = _describe(row.get(key), label)
                if desc:
                    answer_txt = f"{canonical} {desc}."
                else:
                    answer_txt = (f"{canonical}'s document does not explicitly "
                                  f"mention {label} certification.")
            else:
                holds = [lbl for k, lbl in [("iso_9001",  "ISO 9001"),
                                            ("iso_14001", "ISO 14001"),
                                            ("iso_17025", "ISO/IEC 17025")]
                         if row.get(k) == 1]
                nope  = [lbl for k, lbl in [("iso_9001",  "ISO 9001"),
                                            ("iso_14001", "ISO 14001"),
                                            ("iso_17025", "ISO/IEC 17025")]
                         if row.get(k) == 0]
                if holds:
                    answer_txt = f"Yes — {canonical} holds {', '.join(holds)}."
                elif nope:
                    answer_txt = (f"No — {canonical}'s document explicitly "
                                  f"states they are NOT ISO certified "
                                  f"({', '.join(nope)}).")
                else:
                    answer_txt = (f"{canonical}'s document does not mention "
                                  f"ISO certification.")

            return {
                "answer": answer_txt,
                "sources": [{"source": row["source_doc"], "vendor": canonical}],
                "route_method": "structured_sql",
            }

    # ---- Specific vendor field lookup ("what is X's GSTIN?") ----
    # ---- Specific vendor field lookup ----
    #   Handles both phrasings:
    #     "what is Kaveri's GSTIN?"              (possessive)
    #     "what is the GSTIN of Kaveri?"         (of-form)
    #     "give me Nirmala's address"            (possessive)
    #     "show me the PAN for Deccan"           (for-form)
    field_lookup: tuple[str, str] | None = None   # (field_raw, vendor_str)

    # Possessive: "<leader> <vendor>'s <field>"
    # Accept both straight ' and curly ’
    m = re.search(
        r"\b(?:what\s+is|what(?:'|\u2019)?s|give\s+me|show\s+me|tell\s+me)\s+"
        r"(.+?)(?:'|\u2019)s?\s+"
        r"(gstin|gst\s*number|pan|cin|address|location|"
        r"quote\s*number|quote\s*no\.?|quote\s*date)"
        r"\s*\??\s*$",
        ql,
    )
    if m:
        field_lookup = (m.group(2).strip(), m.group(1).strip())
    else:
        # Of/for form: "<leader> the <field> of <vendor>"
        m = re.search(
            r"\b(?:what\s+is|what(?:'|\u2019)?s|give\s+me|show\s+me|tell\s+me)\s+"
            r"(?:the\s+)?"
            r"(gstin|gst\s*number|pan|cin|address|location|"
            r"quote\s*number|quote\s*no\.?|quote\s*date)\s+"
            r"(?:of|for)\s+(.+?)(?:\?|$)",
            ql,
        )
        if m:
            field_lookup = (m.group(1).strip(), m.group(2).strip())

    if field_lookup:
        field_raw, vendor_str = field_lookup
        field_raw = re.sub(r"\s+", " ", field_raw)
        vendor_str = vendor_str.strip().strip('"\'')

        resolved = get_vendor_index().resolve(vendor_str)
        if not resolved.canonical:
            return {"answer": f'No records for a vendor called "{vendor_str}".',
                    "sources": [], "route_method": "structured_sql_empty"}

        field_map = {
            "gstin": "gstin", "gst number": "gstin",
            "pan": "pan", "cin": "cin",
            "address": "address", "location": "address",
            "quote number": "quote_number", "quote no": "quote_number",
            "quote no.": "quote_number", "quote date": "quote_date",
        }
        field = field_map.get(field_raw)
        if field:
            row = structured.get_vendor_field(resolved.canonical, field)
            val = (row or {}).get(field)
            if not val:
                return {"answer": f"No {field_raw} on file for "
                                  f"{resolved.canonical}.",
                        "sources": [], "route_method": "structured_sql_empty"}
            return {"answer": f"{resolved.canonical} — {field_raw}: {val}",
                    "sources": [{"source": row["source_doc"],
                                 "vendor": resolved.canonical}],
                    "route_method": "structured_sql"}

    # ---- Price of specific item from specific vendor ----
    m = re.search(
        r"\b(?:price|cost|rate)\s+of\s+(.+?)\s+from\s+(.+?)(?:\?|$)", ql
    )
    if m:
        item, vendor_str = m.group(1).strip(), m.group(2).strip()
        resolved = get_vendor_index().resolve(vendor_str)
        if not resolved.canonical:
            return {"answer": f'No records for a vendor called "{vendor_str}".',
                    "sources": [], "route_method": "structured_sql_empty"}
        row = structured.price_of_item_from_vendor(resolved.canonical, item)
        if not row:
            return {"answer": f"{resolved.canonical} did not quote for "
                              f"anything matching '{item}'.",
                    "sources": [], "route_method": "structured_sql_empty"}
        price = (f"INR {row['unit_price_inr']:,.0f}"
                 if row['unit_price_inr'] else "N/A")
        unit = f"/{row['unit']}" if row.get('unit') else ""
        return {"answer": f"{resolved.canonical} quoted {row['description']} "
                          f"at {price}{unit}.",
                "sources": [{"source": row["source_doc"],
                             "vendor": resolved.canonical}],
                "route_method": "structured_sql"}

    # ---- Items list from a vendor ("what did X quote?") ----
    m = re.search(
        r"\b(?:what|which\s+items?)\s+(?:did|does|has|have|do)\s+"
        r"(.+?)\s+(?:quote|offer|supply|provide|sell|stock)", ql
    )
    if m:
        vendor_str = m.group(1).strip()
        resolved = get_vendor_index().resolve(vendor_str)
        if not resolved.canonical:
            return {"answer": f'No records for a vendor called "{vendor_str}".',
                    "sources": [], "route_method": "structured_sql_empty"}
        rows = structured.items_from_vendor(resolved.canonical)
        if not rows:
            return {"answer": f"{resolved.canonical} has no extracted line items.",
                    "sources": [], "route_method": "structured_sql_empty"}
        lines = [f"{resolved.canonical} — {len(rows)} line item"
                 f"{'s' if len(rows) != 1 else ''}:", ""]
        for r in rows:
            price = (f"INR {r['unit_price_inr']:,.0f}"
                     if r['unit_price_inr'] else "N/A")
            unit = f"/{r['unit']}" if r.get('unit') else ""
            lines.append(f"• {r['description']} — {price}{unit}")
        return {"answer": "\n".join(lines),
                "sources": [{"vendor": resolved.canonical}],
                "route_method": "structured_sql"}


    # ---- Vendor's quote(s): "what is the quote of X" / "X's quote" ----
    m_of  = re.search(
        r"\b(?:what\s+is|what(?:'|\u2019)?s|give\s+me|show\s+me|tell\s+me)\s+"
        r"(?:the\s+)?quotes?\s+"
        r"(?:of|for|from|by)\s+(.+?)(?:\?|$)",
        ql,
    )
    m_pos = re.search(
        r"\b(?:what\s+is|what(?:'|\u2019)?s|give\s+me|show\s+me|tell\s+me)\s+"
        r"(.+?)(?:'|\u2019)s?\s+quotes?\s*\??\s*$",
        ql,
    )
    m = m_of or m_pos
    if m:
        vendor_str = m.group(1).strip().strip('"\'')
        resolved = get_vendor_index().resolve(vendor_str)
        if not resolved.canonical:
            return {"answer": f'No records for a vendor called "{vendor_str}".',
                    "sources": [], "route_method": "structured_sql_empty"}
        rows = structured.items_from_vendor(resolved.canonical)
        if not rows:
            return {"answer": f"{resolved.canonical} has no extracted line items.",
                    "sources": [], "route_method": "structured_sql_empty"}
        lines = [f"{resolved.canonical} — {len(rows)} line item"
                 f"{'s' if len(rows) != 1 else ''}:", ""]
        for r in rows:
            price = (f"INR {r['unit_price_inr']:,.0f}"
                     if r['unit_price_inr'] else "N/A")
            unit = f"/{r['unit']}" if r.get('unit') else ""
            lines.append(f"• {r['description']} — {price}{unit}")
        return {"answer": "\n".join(lines),
                "sources": [{"vendor": resolved.canonical}],
                "route_method": "structured_sql"}

    # ================================================================
    # GROUP B · Multi-vendor filter queries
    # ================================================================

    # ---- Specific ISO cert (9001 / 14001 / 17025) ----
    for cert_num, cert_key, cert_label in [
        ("9001",  "iso_9001",  "ISO 9001"),
        ("14001", "iso_14001", "ISO 14001"),
        ("17025", "iso_17025", "ISO/IEC 17025"),
    ]:
        if re.search(rf"iso[\s\-]*{cert_num}", ql):
            negated = bool(re.search(r"\b(not|non|without|missing|no)\b", ql))
            rows = (structured.list_non_iso(cert_key)
                    if negated
                    else structured.list_iso_certified(cert_key))
            heading = (f"Vendors NOT holding {cert_label} certification:"
                       if negated
                       else f"Vendors holding {cert_label} certification:")
            return _format_vendor_list(heading, rows)

    # ---- GST / GSTIN presence ----
    if re.search(r"\bgst(in)?\b", ql):
        negated = bool(re.search(
            r"\b(not|no|without|missing|don'?t|do\s+not|lack(?:ing)?)\b", ql
        ))
        if negated:
            rows = structured.vendors_without_gstin()
            heading = "Vendors without a registered GSTIN:"
        else:
            rows = structured.vendors_with_gstin()
            heading = "Vendors with a registered GSTIN:"
        if not rows:
            return {"answer": f"{heading}\n\n(No matching entries.)",
                    "sources": [], "route_method": "structured_sql_empty"}
        seen: set[str] = set()
        lines = [heading, ""]
        sources: list[dict] = []
        for r in rows:
            v = r["canonical_vendor"]
            if v in seen: continue
            seen.add(v)
            g = r.get("gstin")
            lines.append(f"• {v}" + (f" — GSTIN {g}" if g else ""))
            sources.append({"source": r["source_doc"], "vendor": v})
        return {"answer": "\n".join(lines), "sources": sources,
                "route_method": "structured_sql"}

    # ---- Vendor location filter ("vendors from Mumbai") ----
    m = re.search(
        r"\bvendors?\s+(?:in|from|based\s+in|located\s+in)\s+(.+?)(?:\?|$)", ql
    )
    if m:
        place = m.group(1).strip().strip('"\'')
        rows = structured.vendors_from(place)
        return _format_vendor_list(f"Vendors based in {place}:", rows)

    # ---- Recent quotes by issue date ("last N days/weeks/months") ----
    #  Matches:  "last 30 days", "past 2 weeks", "in the last week"
    m = re.search(
        r"\b(?:last|past|previous|within|in\s+the\s+last)\s+"
        r"(?:(\d+)\s+)?"                                  # ← digit now optional
        r"(day|days|week|weeks|month|months)\b",
        ql,
    )
    if m and re.search(r"\bquot|submit|received|issued|dated\b", ql):
        n = int(m.group(1)) if m.group(1) else 1          # ← default to 1
        unit_mul = {"day": 1, "days": 1,
                    "week": 7, "weeks": 7,
                    "month": 30, "months": 30}[m.group(2)]
        days = n * unit_mul
        rows = structured.recent_quotes(days)
        return _format_recent(rows, days)

    # ---- Delivery under N days ----
    m = re.search(
        r"\bdeliver(?:y|s|ing)?.*(?:under|below|within|less\s+than|"
        r"in\s+under|max)\s+(\d+)\s+(?:working\s+)?days?\b",
        ql,
    )
    if m:
        n = int(m.group(1))
        rows = structured.delivery_under(n)
        return _format_delivery(f"Vendors with delivery under {n} days:", rows)

    # ================================================================
    # GROUP C · Ranking queries
    # ================================================================

    # ---- Cheapest / lowest price ----
    if re.search(r"\b(cheapest|lowest\s+price|least\s+expensive|best\s+price)\b", ql):
        item = _extract_item_from_query(query)
        if item:
            rows = structured.cheapest_for(item, top_n=5)
            return _format_price_list(f"Cheapest offers for {item}:", rows)

    # ---- Most expensive / highest price ----
    if re.search(r"\b(most\s+expensive|highest\s+price|priciest|dearest)\b", ql):
        item = _extract_item_from_query(query)
        if item:
            rows = structured.most_expensive_for(item, top_n=5)
            return _format_price_list(f"Most expensive offers for {item}:", rows)

    # ---- Fastest delivery ----
    if re.search(r"\b(fastest|quickest|shortest\s+(?:delivery|lead))\b", ql):
        rows = structured.fastest_delivery(top_n=5)
        return _format_delivery(
            "Vendors ranked by delivery lead time (fastest first):", rows
        )

    # ---- Slowest / longest delivery ----
    if re.search(r"\b(slowest|longest\s+(?:delivery|lead))\b", ql):
        rows = structured.slowest_delivery(top_n=5)
        return _format_delivery(
            "Vendors ranked by delivery lead time (slowest first):", rows
        )

    # ---- Longest credit / best payment terms ----
    if re.search(
        r"\blongest\s+(?:credit|payment)|most\s+credit\s+days|"
        r"best\s+payment\s+terms\b",
        ql,
    ):
        rows = structured.longest_credit(top_n=5)
        return _format_credit(
            "Vendors ranked by credit period (longest first):", rows
        )

    # ---- Shortest credit / worst payment terms ----
    if re.search(
        r"\bshortest\s+(?:credit|payment)|worst\s+payment\s+terms\b", ql
    ):
        rows = structured.shortest_credit(top_n=5)
        return _format_credit(
            "Vendors ranked by credit period (shortest first):", rows
        )

    # ================================================================
    # GROUP D · Item availability
    # ================================================================

    # ---- Who supplies / quoted / offers item X ----
    m = re.search(
        r"\b(?:who|which\s+vendors?|any\s+vendors?|any\s+vendor)\s+"
        r"(?:can|could|would|may|does|do|will|has|have)?\s*"
        r"(?:quoted?|offered?|supply|supplies|supplied|sells?|sold|"
        r"provides?|provided|stocks?|stocked|carries|carry|has|have)\s+"
        r"(?:for\s+|us\s+with\s+|the\s+)?(.+?)(?:\?|$)",
        ql,
    )
    if m:
        item = m.group(1).strip().strip('"\'')
        if item:
            rows = structured.vendors_with_item(item)
            return _format_price_list(f"Vendors who quoted for {item}:", rows)

    # ================================================================
    # GROUP E · Generic catch-alls (must be LAST)
    # ================================================================

    # ---- Corpus counts ("how many …?") — items first, quotes last ----
    if re.search(r"\bhow\s+many\b", ql):
        # 1) items — needs to run before quotes so "items ... quote?" wins
        if re.search(r"\bitems?|line\s*items?|skus?\b", ql):
            vh = _detect_vendor(query)
            n = structured.item_count(vh)
            if vh:
                return {"answer": f"{vh} quoted {n} line item"
                                  f"{'s' if n != 1 else ''}.",
                        "sources": [], "route_method": "structured_sql"}
            return {"answer": f"The corpus has {n} line item"
                              f"{'s' if n != 1 else ''} across all quotes.",
                    "sources": [], "route_method": "structured_sql"}
        # 2) quotes as a noun (not the verb "to quote")
        if re.search(r"\bquotes?|quotations?\b", ql):
            n = structured.quote_count()
            return {"answer": f"The corpus has {n} quote document"
                              f"{'s' if n != 1 else ''}.",
                    "sources": [], "route_method": "structured_sql"}
        # 3) vendors — broadest
        if re.search(r"\bvendor", ql):
            n = structured.vendor_count()
            return {"answer": f"The corpus has {n} vendor"
                              f"{'s' if n != 1 else ''}.",
                    "sources": [], "route_method": "structured_sql"}

    # ---- Generic "iso certified?" without a specific standard ----
    if re.search(r"\biso\b", ql) and re.search(r"which|what|list|vendors?", ql):
        negated = bool(re.search(r"\b(not|non|without|no)\b", ql))
        rows = (structured.list_non_iso("iso_9001")
                if negated
                else structured.list_iso_certified("iso_9001"))
        heading = ("Vendors NOT ISO 9001 certified:" if negated
                   else "ISO 9001 certified vendors:")
        return _format_vendor_list(heading, rows)

    # ---- Corpus list: "list all vendors" (widest catch-all) ----
    if re.search(r"\b(list|show|what|which|all)\b.*\bvendors?\b", ql) \
       and not re.search(
           r"\b(iso|price|cheapest|fastest|slowest|delivery|payment|"
           r"credit|gst|from|in|based|located|quote)\b", ql
       ):
        vendors = get_vendor_index().vendors
        if not vendors:
            return {"answer": "The corpus is empty — no vendors yet.",
                    "sources": [], "route_method": "structured_corpus"}
        lines = [f"The corpus contains {len(vendors)} vendor"
                 f"{'s' if len(vendors) != 1 else ''}:", ""]
        lines += [f"• {v}" for v in vendors]
        return {"answer": "\n".join(lines), "sources": [],
                "route_method": "structured_corpus"}

    return None


def _format_vendor_list(heading: str, rows: list[dict]) -> dict:
    if not rows:
        return {"answer": f"{heading}\n\n(No matching entries in the structured facts DB. "
                          f"Try re-indexing to populate it, or the doc may not have stated this explicitly.)",
                "sources": [], "route_method": "structured_sql_empty"}
    seen: set[str] = set()
    lines = [heading, ""]
    sources: list[dict] = []
    for r in rows:
        v = r["canonical_vendor"]
        if v in seen:
            continue
        seen.add(v)
        lines.append(f"• {v}")
        sources.append({"source": r["source_doc"], "vendor": v})
    return {"answer": "\n".join(lines), "sources": sources,
            "route_method": "structured_sql"}


def _format_price_list(heading: str, rows: list[dict]) -> dict:
    if not rows:
        return {"answer": f"{heading}\n\n(No matching entries in the structured facts DB.)",
                "sources": [], "route_method": "structured_sql_empty"}
    lines = [heading, ""]
    sources: list[dict] = []
    for r in rows:
        price = (f"INR {r['unit_price_inr']:,.0f}"
                 if r.get("unit_price_inr") is not None else "price N/A")
        unit = f"/{r['unit']}" if r.get("unit") else ""
        lines.append(f"• {r['canonical_vendor']}: {r['description']} — {price}{unit}")
        sources.append({"source": r["source_doc"], "vendor": r["canonical_vendor"]})
    return {"answer": "\n".join(lines), "sources": sources,
            "route_method": "structured_sql"}


def _format_delivery(heading: str, rows: list[dict]) -> dict:
    if not rows:
        return {"answer": f"{heading}\n\n(No delivery info in the structured facts DB.)",
                "sources": [], "route_method": "structured_sql_empty"}
    lines = [heading, ""]
    sources: list[dict] = []
    for r in rows:
        mn, mx = r.get("delivery_days_min"), r.get("delivery_days_max")
        rng = (f"{mn}-{mx} days" if mn != mx else f"{mn} days") if mn is not None else "N/A"
        lines.append(f"• {r['canonical_vendor']}: {rng}")
        sources.append({"source": r["source_doc"], "vendor": r["canonical_vendor"]})
    return {"answer": "\n".join(lines), "sources": sources,
            "route_method": "structured_sql"}


def _format_credit(heading: str, rows: list[dict]) -> dict:
    if not rows:
        return {"answer": f"{heading}\n\n(No payment terms in the structured facts DB.)",
                "sources": [], "route_method": "structured_sql_empty"}
    lines = [heading, ""]
    sources: list[dict] = []
    for r in rows:
        days = r.get("payment_days_net")
        raw = r.get("payment_terms_raw") or ""
        detail = f"{days} days net" if days is not None else "N/A"
        if raw:
            detail += f" ({raw})"
        lines.append(f"• {r['canonical_vendor']}: {detail}")
        sources.append({"source": r["source_doc"], "vendor": r["canonical_vendor"]})
    return {"answer": "\n".join(lines), "sources": sources,
            "route_method": "structured_sql"}


def _extract_item_from_query(query: str) -> str | None:
    """
    Extract an item name from a price-superlative query. Handles both
    families (cheapest/lowest AND most-expensive/priciest).
    """
    m = re.search(
        r"(?:cheapest|lowest\s+price|least\s+expensive|best\s+price|"
        r"most\s+expensive|highest\s+price|priciest|dearest)\s+"
        r"(?:for|of|on)?\s*(.+?)(?:\?|\.|$|\s+(?:from|among|between|by)\b)",
        query, re.IGNORECASE,
    )
    if not m:
        return None
    item = m.group(1).strip()
    item = re.sub(r"^(the|a|an)\s+", "", item, flags=re.IGNORECASE)
    item = item.rstrip(".?!, ").strip()
    return item or None

def _format_recent(rows: list[dict], days: int) -> dict:
    if not rows:
        return {
            "answer": f"No quotes with an extracted quote_date in the last "
                      f"{days} day{'s' if days != 1 else ''}.",
            "sources": [],
            "route_method": "structured_sql_empty",
        }
    lines = [
        f"Vendors with quotes issued in the last {days} day"
        f"{'s' if days != 1 else ''} (most recent first):",
        "",
    ]
    sources: list[dict] = []
    for r in rows:
        num = r.get("quote_number") or "—"
        date = r.get("quote_date") or "?"
        lines.append(
            f"• {r['canonical_vendor']}: {date} (quote #{num})"
        )
        sources.append({
            "source": r["source_doc"],
            "vendor": r["canonical_vendor"],
        })
    return {
        "answer": "\n".join(lines),
        "sources": sources,
        "route_method": "structured_sql",
    }
# ---------- Main entry ----------

def answer(query: str, intent: str | None = None) -> dict:
    is_multi   = _is_multi_vendor_query(query)
    is_simple  = _is_simple_query(query)
    # Only scope to one vendor when the query names it AND is NOT a
    # multi-vendor question (comparison, list, filter, aggregation).
    vendor_hint = None if is_multi else _detect_vendor(query)

    print(f"[qa] shape: multi_vendor={is_multi} simple={is_simple} "
          f"vendor_hint={vendor_hint}", flush=True)

    # ---- Path 0: structured DB FIRST — no LLM expansion required ----
    structured_result = _try_structured(query)
    if structured_result:
        print(f"[qa] structured route: "
              f"{structured_result['route_method']}", flush=True)
        return structured_result

    # Only if SQL didn't fire, do the LLM expansion for RAG
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
        hits = _boost_table_rows(hits, query)
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
        hits = _boost_table_rows(hits, query)
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