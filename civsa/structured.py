"""
Structured facts DB for CIVSA — SQLite tables for deterministic queries
(price comparison, ISO certification, delivery time, payment terms) that
pure RAG cannot reliably answer.

Populated at index time by an LLM extractor forced into a JSON template.
Queried before hitting the retriever, with citations back to the source
document.

Public API:
    init_db(path)                       — call once at app startup
    extract_facts(text, vendor, ...)    — LLM → dict
    store_extraction(vendor, ..., dict) — dict → SQLite
    remove_by_source(source_path)       — on doc delete
    remove_by_vendor(canonical_vendor)  — on vendor delete
    reset()                             — for reindex
    list_iso_certified(cert)            — SQL query
    list_non_iso()                      — SQL query
    cheapest_for(item)                  — SQL query
    fastest_delivery()                  — SQL query
    vendors_with_item(item)             — SQL query
    get_all_facts_for_vendor(vendor)    — SQL query (debug)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_TRACE_SQL = os.getenv("CIVSA_SQL_TRACE", "1") == "1"

def _trace(stmt: str) -> None:
    s = stmt.strip()
    if s.upper().startswith("SELECT"):
        print(f"[structured.sql]\n{s}\n", flush=True)

# ============================================================
# Schema
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS vendor_facts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor            TEXT    NOT NULL,
    canonical_vendor  TEXT    NOT NULL,
    source_doc        TEXT    NOT NULL,
    iso_9001          INTEGER,
    iso_14001         INTEGER,
    iso_17025         INTEGER,
    other_certs       TEXT,
    address           TEXT,
    gstin             TEXT,
    pan               TEXT,
    cin               TEXT,
    quote_date        TEXT,          -- ← NEW, ISO 8601 (YYYY-MM-DD)
    quote_number      TEXT,          -- ← NEW
    updated_at        TEXT    NOT NULL,
    UNIQUE(canonical_vendor, source_doc)
);
CREATE INDEX IF NOT EXISTS idx_vf_canonical ON vendor_facts(canonical_vendor);
CREATE INDEX IF NOT EXISTS idx_vf_iso9001   ON vendor_facts(iso_9001);
CREATE INDEX IF NOT EXISTS idx_vf_iso14001  ON vendor_facts(iso_14001);
CREATE INDEX IF NOT EXISTS idx_vf_iso17025  ON vendor_facts(iso_17025);
CREATE INDEX IF NOT EXISTS idx_vf_quote_date ON vendor_facts(quote_date);

CREATE TABLE IF NOT EXISTS quote_items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor            TEXT    NOT NULL,
    canonical_vendor  TEXT    NOT NULL,
    source_doc        TEXT    NOT NULL,
    sku               TEXT,
    description       TEXT    NOT NULL,
    unit              TEXT,
    quantity          REAL,
    unit_price_inr    REAL,
    line_total_inr    REAL,
    currency          TEXT    DEFAULT 'INR',
    extracted_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qi_canonical ON quote_items(canonical_vendor);
CREATE INDEX IF NOT EXISTS idx_qi_desc      ON quote_items(description);
CREATE INDEX IF NOT EXISTS idx_qi_source    ON quote_items(source_doc);
CREATE INDEX IF NOT EXISTS idx_qi_price     ON quote_items(unit_price_inr);

CREATE TABLE IF NOT EXISTS commercial_terms (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor                TEXT    NOT NULL,
    canonical_vendor      TEXT    NOT NULL,
    source_doc            TEXT    NOT NULL,
    delivery_days_min     INTEGER,
    delivery_days_max     INTEGER,
    payment_days_net      INTEGER,
    payment_terms_raw     TEXT,
    freight_terms         TEXT,
    warranty_terms        TEXT,
    gst_percentage        REAL,
    quote_validity_days   INTEGER,
    extracted_at          TEXT    NOT NULL,
    UNIQUE(canonical_vendor, source_doc)
);
CREATE INDEX IF NOT EXISTS idx_ct_canonical    ON commercial_terms(canonical_vendor);
CREATE INDEX IF NOT EXISTS idx_ct_delivery_max ON commercial_terms(delivery_days_max);
CREATE INDEX IF NOT EXISTS idx_ct_payment_net  ON commercial_terms(payment_days_net);
"""


# ============================================================
# Connection
# ============================================================

_DB_PATH: Optional[Path] = None


def init_db(db_path: Path) -> None:
    """Create the SQLite file and tables. Call once at app startup."""
    global _DB_PATH
    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def _connect():
    """
    Yield a SQLite connection. Auto-initialises the DB from config.SQLITE_PATH
    on first use so scripts and shells don't have to call init_db() explicitly.
    FastAPI's startup still calls init_db() eagerly, which is fine — the second
    call is a no-op.
    """
    global _DB_PATH
    if _DB_PATH is None:
        from .config import SQLITE_PATH
        init_db(SQLITE_PATH)
    conn = sqlite3.connect(_DB_PATH) # type: ignore
    conn.row_factory = sqlite3.Row
    if _TRACE_SQL:
        conn.set_trace_callback(_trace) 
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ============================================================
# LLM extraction
# ============================================================

EXTRACTION_PROMPT = """You are extracting STRUCTURED FACTS from a procurement document.
Output valid JSON matching this exact schema. Use null for anything you can't determine — never guess.

{{
  "vendor_facts": {{
    "iso_9001":   true | false | null,
    "iso_14001":  true | false | null,
    "iso_17025":  true | false | null,
    "other_certs": [ "string", ... ],
    "address":    "string or null",
    "gstin":      "string or null",
    "pan":        "string or null",
    "cin":        "string or null",
    "quote_date":   "YYYY-MM-DD or null (the date the quotation was issued)",
    "quote_number": "string or null (the quotation reference number, e.g. NCP/QT/2026-27/0431)"
  }},
  "commercial_terms": {{
    "delivery_days_min":   integer or null,
    "delivery_days_max":   integer or null,
    "payment_days_net":    integer or null,
    "payment_terms_raw":   "string or null",
    "freight_terms":       "string or null",
    "warranty_terms":      "string or null",
    "gst_percentage":      number or null,
    "quote_validity_days": integer or null
  }},
  "quote_items": [
    {{
      "sku":            "string or null",
      "description":    "string",
      "unit":           "string or null",
      "quantity":       number or null,
      "unit_price_inr": number or null,
      "line_total_inr": number or null
    }}
  ]
}}

CRITICAL RULES:
1. iso_9001 / iso_14001 / iso_17025: true ONLY if the doc explicitly says the vendor HOLDS
   that specific certification. If it says "not certified", "under process", or "targeted for
   next year", set false. If the cert is not mentioned at all, use null.
2. Prices are in INR. Strip commas: "1,050" -> 1050. If a price is missing, use null.
3. Delivery: "5 to 7 working days" -> min=5, max=7. Single value "10 days" -> min=10, max=10.
4. Payment: "30 days net" -> 30. "45 days net" -> 45. "advance" -> 0.
5. quote_items: one entry per PRICED row in a schedule/table. Empty list if no priced items.
6. Return ONLY the JSON object. No prose, no markdown fences, no explanation.
7. quote_date: convert human-written dates ("22 August 2026", "22-Aug-26",
   "22/08/2026") to strict ISO format YYYY-MM-DD ("2026-08-22"). If no
   date is visible, use null. Never guess a year — if only "22 Aug" is
   shown with no year, use null.

DOCUMENT:
Vendor as recorded: {vendor}
Source path:        {source}

{text}

JSON:"""


def extract_facts(
    text: str,
    vendor: str,
    source: str,
    llm_chat_fn,
) -> Optional[dict]:
    """
    Run the LLM extractor, parse JSON, return dict (or None on failure).

    `llm_chat_fn` is your llm_client.chat function, injected so this module
    stays testable and free of import cycles.
    """
    if not text or not text.strip():
        return None
    prompt = EXTRACTION_PROMPT.format(
        vendor=vendor,
        source=source,
        text=text[:6000],   # cap so local 3B models don't crawl
    )
    try:
        raw = llm_chat_fn(
            messages=[{"role": "user", "content": prompt}],
            model_size="big",       # extraction wants the smarter model
            temperature=0,
            max_tokens=1500,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[structured] LLM call failed for {source}: {e}")
        return None

    # Strip markdown fences if the model wrapped its output
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?\s*```\s*$", "", raw)

    # Some models emit trailing prose after the JSON — clip at last }
    last_brace = raw.rfind("}")
    if last_brace != -1:
        raw = raw[: last_brace + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[structured] JSON parse failed for {source}: {e}")
        print(f"[structured] raw (first 400 chars): {raw[:400]}")
        return None


# ============================================================
# Insertion / deletion
# ============================================================

def store_extraction(
    vendor: str,
    canonical_vendor: str,
    source: str,
    facts: dict,
) -> dict:
    """
    Persist an extraction across the three tables. Idempotent per source.

    Returns a summary like {"vendor_facts": 1, "commercial_terms": 1, "quote_items": 5}.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {"vendor_facts": 0, "commercial_terms": 0, "quote_items": 0}

    with _connect() as conn:
        # Idempotent — remove anything previously extracted from this source
        conn.execute("DELETE FROM vendor_facts     WHERE source_doc = ?", (source,))
        conn.execute("DELETE FROM commercial_terms WHERE source_doc = ?", (source,))
        conn.execute("DELETE FROM quote_items      WHERE source_doc = ?", (source,))

        vf = facts.get("vendor_facts") or {}
        conn.execute(
       """INSERT INTO vendor_facts
       (vendor, canonical_vendor, source_doc,
        iso_9001, iso_14001, iso_17025,
        other_certs, address, gstin, pan, cin,
        quote_date, quote_number, updated_at)
       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
        vendor, canonical_vendor, source,
        _bool_to_int(vf.get("iso_9001")),
        _bool_to_int(vf.get("iso_14001")),
        _bool_to_int(vf.get("iso_17025")),
        json.dumps(vf.get("other_certs") or []),
        vf.get("address"), vf.get("gstin"),
        vf.get("pan"), vf.get("cin"),
        vf.get("quote_date"), vf.get("quote_number"),   # ← NEW
        now,
        ),
        )
        summary["vendor_facts"] = 1

        ct = facts.get("commercial_terms") or {}
        conn.execute(
            """INSERT INTO commercial_terms
               (vendor, canonical_vendor, source_doc,
                delivery_days_min, delivery_days_max,
                payment_days_net, payment_terms_raw,
                freight_terms, warranty_terms,
                gst_percentage, quote_validity_days, extracted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                vendor, canonical_vendor, source,
                _to_int(ct.get("delivery_days_min")),
                _to_int(ct.get("delivery_days_max")),
                _to_int(ct.get("payment_days_net")),
                ct.get("payment_terms_raw"),
                ct.get("freight_terms"),
                ct.get("warranty_terms"),
                _to_float(ct.get("gst_percentage")),
                _to_int(ct.get("quote_validity_days")),
                now,
            ),
        )
        summary["commercial_terms"] = 1

        for item in facts.get("quote_items") or []:
            desc = (item.get("description") or "").strip()
            if not desc:
                continue
            conn.execute(
                """INSERT INTO quote_items
                   (vendor, canonical_vendor, source_doc,
                    sku, description, unit, quantity,
                    unit_price_inr, line_total_inr, extracted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    vendor, canonical_vendor, source,
                    item.get("sku"), desc,
                    item.get("unit"), _to_float(item.get("quantity")),
                    _to_float(item.get("unit_price_inr")),
                    _to_float(item.get("line_total_inr")),
                    now,
                ),
            )
            summary["quote_items"] += 1

    return summary


def remove_by_source(source: str) -> None:
    """Remove all extracted rows for a source path. Called on doc delete."""
    with _connect() as conn:
        conn.execute("DELETE FROM vendor_facts     WHERE source_doc = ?", (source,))
        conn.execute("DELETE FROM commercial_terms WHERE source_doc = ?", (source,))
        conn.execute("DELETE FROM quote_items      WHERE source_doc = ?", (source,))


def remove_by_vendor(canonical_vendor: str) -> None:
    """Remove all extracted rows for a canonical vendor. Called on vendor delete."""
    with _connect() as conn:
        conn.execute("DELETE FROM vendor_facts     WHERE canonical_vendor = ?", (canonical_vendor,))
        conn.execute("DELETE FROM commercial_terms WHERE canonical_vendor = ?", (canonical_vendor,))
        conn.execute("DELETE FROM quote_items      WHERE canonical_vendor = ?", (canonical_vendor,))


def reset() -> None:
    """Drop and recreate all tables. Used by scripts/reindex.py."""
    with _connect() as conn:
        conn.executescript(
            "DROP TABLE IF EXISTS vendor_facts; "
            "DROP TABLE IF EXISTS commercial_terms; "
            "DROP TABLE IF EXISTS quote_items;"
        )
        conn.executescript(SCHEMA)


# ============================================================
# Query API
# ============================================================

_ISO_COL = {"iso_9001": "iso_9001", "iso_14001": "iso_14001", "iso_17025": "iso_17025"}


def list_iso_certified(cert: str = "iso_9001") -> list[dict]:
    """Vendors that hold a specific ISO certification (true only, never null)."""
    col = _ISO_COL.get(cert)
    if not col:
        return []
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT canonical_vendor, source_doc "
            f"FROM vendor_facts WHERE {col} = 1"
        ).fetchall()
        return [dict(r) for r in rows]


def list_non_iso(cert: str = "iso_9001") -> list[dict]:
    """Vendors explicitly NOT holding a specific ISO cert (false only, never null)."""
    col = _ISO_COL.get(cert)
    if not col:
        return []
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT canonical_vendor, source_doc "
            f"FROM vendor_facts WHERE {col} = 0"
        ).fetchall()
        return [dict(r) for r in rows]


def cheapest_for(item_keyword: str, top_n: int = 5) -> list[dict]:
    """Cheapest priced offers for items matching a keyword."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT canonical_vendor, description, unit, quantity,
                      unit_price_inr, source_doc
               FROM quote_items
               WHERE unit_price_inr IS NOT NULL
                 AND LOWER(description) LIKE ?
               ORDER BY unit_price_inr ASC
               LIMIT ?""",
            (f"%{item_keyword.lower()}%", top_n),
        ).fetchall()
        return [dict(r) for r in rows]


def fastest_delivery(top_n: int = 5) -> list[dict]:
    """Vendors ranked by shortest maximum delivery lead time."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT canonical_vendor,
                      delivery_days_min, delivery_days_max, source_doc
               FROM commercial_terms
               WHERE delivery_days_max IS NOT NULL
               ORDER BY delivery_days_max ASC, delivery_days_min ASC
               LIMIT ?""",
            (top_n,),
        ).fetchall()
        return [dict(r) for r in rows]


def longest_credit(top_n: int = 5) -> list[dict]:
    """Vendors ranked by longest payment credit period."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT canonical_vendor, payment_days_net, payment_terms_raw, source_doc
               FROM commercial_terms
               WHERE payment_days_net IS NOT NULL
               ORDER BY payment_days_net DESC
               LIMIT ?""",
            (top_n,),
        ).fetchall()
        return [dict(r) for r in rows]


def recent_quotes(days: int) -> list[dict]:
    """Vendors whose quotes were issued within the last N days (by quote_date)."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT canonical_vendor, quote_date, quote_number, source_doc
               FROM vendor_facts
               WHERE quote_date IS NOT NULL
                 AND quote_date >= date('now', ?)
               ORDER BY quote_date DESC""",
            (f"-{days} days",),
        ).fetchall()
        return [dict(r) for r in rows]

def vendors_with_item(item_keyword: str) -> list[dict]:
    """All vendors who quoted for items matching a keyword, cheapest first."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT canonical_vendor, description, unit,
                      unit_price_inr, source_doc
               FROM quote_items
               WHERE LOWER(description) LIKE ?
               ORDER BY (unit_price_inr IS NULL), unit_price_inr ASC""",
            (f"%{item_keyword.lower()}%",),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_facts_for_vendor(canonical_vendor: str) -> dict:
    """Everything we know about one vendor — used by the Debug tab."""
    with _connect() as conn:
        vf = conn.execute(
            "SELECT * FROM vendor_facts WHERE canonical_vendor = ?",
            (canonical_vendor,),
        ).fetchall()
        ct = conn.execute(
            "SELECT * FROM commercial_terms WHERE canonical_vendor = ?",
            (canonical_vendor,),
        ).fetchall()
        qi = conn.execute(
            "SELECT * FROM quote_items WHERE canonical_vendor = ?",
            (canonical_vendor,),
        ).fetchall()
        return {
            "vendor_facts":     [dict(r) for r in vf],
            "commercial_terms": [dict(r) for r in ct],
            "quote_items":      [dict(r) for r in qi],
        }


def stats() -> dict:
    """Counts across all tables — for debug."""
    with _connect() as conn:
        vf = conn.execute("SELECT COUNT(*) AS n FROM vendor_facts").fetchone()["n"]
        ct = conn.execute("SELECT COUNT(*) AS n FROM commercial_terms").fetchone()["n"]
        qi = conn.execute("SELECT COUNT(*) AS n FROM quote_items").fetchone()["n"]
        return {"vendor_facts": vf, "commercial_terms": ct, "quote_items": qi}

def vendors_without_gstin() -> list[dict]:
    """Vendors whose extracted GSTIN is NULL or empty."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT canonical_vendor, source_doc
               FROM vendor_facts
               WHERE gstin IS NULL OR TRIM(gstin) = ''"""
        ).fetchall()
        return [dict(r) for r in rows]


def vendors_with_gstin() -> list[dict]:
    """Vendors whose extracted GSTIN is present."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT canonical_vendor, gstin, source_doc
               FROM vendor_facts
               WHERE gstin IS NOT NULL AND TRIM(gstin) != ''"""
        ).fetchall()
        return [dict(r) for r in rows]



# ---- Vendor-facts single-field lookups --------------------------------

def get_vendor_field(canonical_vendor: str, field: str) -> Optional[str]:
    """Fetch one field from vendor_facts for one vendor."""
    allowed = {"address", "gstin", "pan", "cin", "quote_number", "quote_date"}
    if field not in allowed:
        return None
    with _connect() as conn:
        row = conn.execute(
            f"SELECT {field}, source_doc FROM vendor_facts "
            f"WHERE canonical_vendor = ? LIMIT 1",
            (canonical_vendor,),
        ).fetchone()
        return dict(row) if row else None


# ---- Item queries -----------------------------------------------------

def most_expensive_for(item_keyword: str, top_n: int = 5) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT canonical_vendor, description, unit, quantity,
                      unit_price_inr, source_doc
               FROM quote_items
               WHERE unit_price_inr IS NOT NULL
                 AND LOWER(description) LIKE ?
               ORDER BY unit_price_inr DESC
               LIMIT ?""",
            (f"%{item_keyword.lower()}%", top_n),
        ).fetchall()
        return [dict(r) for r in rows]


def items_from_vendor(canonical_vendor: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT sku, description, unit, quantity,
                      unit_price_inr, line_total_inr
               FROM quote_items
               WHERE canonical_vendor = ?
               ORDER BY unit_price_inr DESC NULLS LAST""",
            (canonical_vendor,),
        ).fetchall()
        return [dict(r) for r in rows]


def price_of_item_from_vendor(canonical_vendor: str, item: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """SELECT description, unit_price_inr, unit, quantity,
                      line_total_inr, source_doc
               FROM quote_items
               WHERE canonical_vendor = ?
                 AND LOWER(description) LIKE ?
               LIMIT 1""",
            (canonical_vendor, f"%{item.lower()}%"),
        ).fetchone()
        return dict(row) if row else None


# ---- Commercial-term filters ------------------------------------------

def slowest_delivery(top_n: int = 5) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT canonical_vendor,
                      delivery_days_min, delivery_days_max, source_doc
               FROM commercial_terms
               WHERE delivery_days_max IS NOT NULL
               ORDER BY delivery_days_max DESC, delivery_days_min DESC
               LIMIT ?""",
            (top_n,),
        ).fetchall()
        return [dict(r) for r in rows]


def delivery_under(days: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT canonical_vendor,
                      delivery_days_min, delivery_days_max, source_doc
               FROM commercial_terms
               WHERE delivery_days_max IS NOT NULL
                 AND delivery_days_max <= ?
               ORDER BY delivery_days_max ASC""",
            (days,),
        ).fetchall()
        return [dict(r) for r in rows]


def shortest_credit(top_n: int = 5) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT canonical_vendor, payment_days_net, payment_terms_raw,
                      source_doc
               FROM commercial_terms
               WHERE payment_days_net IS NOT NULL
               ORDER BY payment_days_net ASC
               LIMIT ?""",
            (top_n,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---- Location filter --------------------------------------------------

def vendors_from(city_or_state: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT canonical_vendor, address, source_doc
               FROM vendor_facts
               WHERE address IS NOT NULL
                 AND LOWER(address) LIKE ?""",
            (f"%{city_or_state.lower()}%",),
        ).fetchall()
        return [dict(r) for r in rows]


# ---- Corpus counters --------------------------------------------------

def vendor_count() -> int:
    with _connect() as conn:
        r = conn.execute(
            "SELECT COUNT(DISTINCT canonical_vendor) AS n FROM vendor_facts"
        ).fetchone()
        return r["n"]


def quote_count() -> int:
    with _connect() as conn:
        r = conn.execute(
            "SELECT COUNT(DISTINCT source_doc) AS n FROM vendor_facts"
        ).fetchone()
        return r["n"]


def item_count(vendor: Optional[str] = None) -> int:
    with _connect() as conn:
        if vendor:
            r = conn.execute(
                "SELECT COUNT(*) AS n FROM quote_items WHERE canonical_vendor = ?",
                (vendor,),
            ).fetchone()
        else:
            r = conn.execute("SELECT COUNT(*) AS n FROM quote_items").fetchone()
        return r["n"]


    
# ============================================================
# Helpers
# ============================================================

def _bool_to_int(v) -> Optional[int]:
    """None -> NULL, True -> 1, False -> 0. Coerces truthy/falsy safely."""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if v else 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1"):    return 1
        if s in ("false", "no", "n", "0"):    return 0
    return None


def _to_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, str):
        v = v.replace(",", "").strip()
    try:
        return float(v)
    except (ValueError, TypeError):
        return None