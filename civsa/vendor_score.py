"""
Vendor scoring for CIVSA — MCDM composite + SHAP-style attribution.

Currently implements S1 (Price) end-to-end from `quote_items`. S2-S5 are
placeholders returning None until certificates, POs/GRNs, and post-award
feedback are ingested — the framework and UI are ready to slot them in.
"""

from __future__ import annotations
from . import structured

CRITERIA = [
    {"key": "S1", "label": "Price",          "impl": True,  "default_weight": 0.35},
    {"key": "S2", "label": "Quality",        "impl": False, "default_weight": 0.20},
    {"key": "S3", "label": "Delivery",       "impl": False, "default_weight": 0.20},
    {"key": "S4", "label": "Financial Risk", "impl": False, "default_weight": 0.15},
    {"key": "S5", "label": "ESG",            "impl": False, "default_weight": 0.10},
]
DEFAULT_WEIGHTS = {c["key"]: c["default_weight"] for c in CRITERIA}


def list_items() -> list[str]:
    """All distinct item descriptions across quote_items."""
    with structured._connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT description FROM quote_items "
            "WHERE description IS NOT NULL AND TRIM(description) != '' "
            "ORDER BY LOWER(description)"
        ).fetchall()
        return [r["description"] for r in rows]


def _score_price_for_item(item_keyword: str) -> list[dict]:
    """
    S1 = 100 * cheapest_price / vendor_price. Cheapest → 100.
    Groups by vendor to take the min price per vendor for the item.
    """
    with structured._connect() as conn:
        rows = conn.execute(
            """SELECT canonical_vendor, description, unit,
                      MIN(unit_price_inr) AS unit_price_inr, source_doc
               FROM quote_items
               WHERE unit_price_inr IS NOT NULL
                 AND LOWER(description) LIKE ?
               GROUP BY canonical_vendor""",
            (f"%{item_keyword.lower()}%",),
        ).fetchall()
    if not rows:
        return []
    min_price = min(r["unit_price_inr"] for r in rows)
    return [{
        "vendor":      r["canonical_vendor"],
        "S1":          round(100.0 * min_price / r["unit_price_inr"], 1),
        "S2": None, "S3": None, "S4": None, "S5": None,
        "price":       r["unit_price_inr"],
        "unit":        r["unit"],
        "source":      r["source_doc"],
        "description": r["description"],
    } for r in rows]


def _combine_basket(item_scores: dict[str, list[dict]]) -> list[dict]:
    """Average S1 across items per vendor."""
    agg: dict[str, dict] = {}
    for item, scores in item_scores.items():
        for s in scores:
            v = s["vendor"]
            slot = agg.setdefault(v, {
                "vendor": v, "S1_sum": 0.0, "S1_count": 0,
                "items": [], "sources": set(),
            })
            if s["S1"] is not None:
                slot["S1_sum"] += s["S1"]
                slot["S1_count"] += 1
            slot["items"].append({
                "item": item, "price": s["price"],
                "unit": s["unit"], "S1": s["S1"],
            })
            slot["sources"].add(s["source"])
    out = []
    for v, slot in agg.items():
        S1 = round(slot["S1_sum"] / slot["S1_count"], 1) if slot["S1_count"] else None
        out.append({
            "vendor":  v,
            "S1":      S1,
            "S2": None, "S3": None, "S4": None, "S5": None,
            "items":   slot["items"],
            "n_items": slot["S1_count"],
            "sources": sorted(slot["sources"]),
            "price":   None,
            "unit":    None,
            "source":  next(iter(slot["sources"]), None),
        })
    return out


def compute_ranking(
    mode: str = "single",
    items: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    mode: "single" | "basket" | "overall"
    """
    weights = weights or dict(DEFAULT_WEIGHTS)
    items   = items or []

    if mode == "single":
        if not items:
            return {"error": "single mode requires exactly one item"}
        rows = _score_price_for_item(items[0])
    elif mode == "basket":
        if not items:
            return {"error": "basket mode requires at least one item"}
        rows = _combine_basket({it: _score_price_for_item(it) for it in items})
    elif mode == "overall":
        all_items = list_items()
        rows = _combine_basket({it: _score_price_for_item(it) for it in all_items})
    else:
        return {"error": f"unknown mode: {mode}"}

    if not rows:
        return {"criteria": CRITERIA, "weights_used": weights,
                "baseline": {}, "rows": []}

    baseline = {}
    for k in ("S1", "S2", "S3", "S4", "S5"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        baseline[k] = round(sum(vals) / len(vals), 2) if vals else None

    for r in rows:
        composite = 0.0
        contribs  = {}
        for k in ("S1", "S2", "S3", "S4", "S5"):
            w, s, b = weights.get(k, 0), r.get(k), baseline.get(k)
            if s is None or b is None:
                contribs[k] = None
            else:
                contribs[k] = round(w * (s - b), 2)
                composite  += w * s
        r["composite"]     = round(composite, 1)
        r["contributions"] = contribs

    rows.sort(key=lambda x: x["composite"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    return {
        "criteria":     CRITERIA,
        "weights_used": weights,
        "baseline":     baseline,
        "rows":         rows,
    }


def build_narrative(row: dict, baseline: dict) -> str:
    """Deterministic Algorithm-J-style narrative — no LLM cost."""
    contribs = row.get("contributions", {})
    ranked = sorted(
        [(k, v) for k, v in contribs.items() if v is not None],
        key=lambda t: abs(t[1]), reverse=True,
    )
    if not ranked:
        return f"{row['vendor']} has no computable criterion scores."
    labels = {c["key"]: c["label"] for c in CRITERIA}
    top_k, top_v = ranked[0]
    direction = "above" if top_v > 0 else "below"
    parts = [
        f"{row['vendor']} scores {row['composite']} on the composite — "
        f"driven mainly by {labels[top_k]}, "
        f"which sits {direction} the shortlist baseline."
    ]
    if row.get("S1") is not None and row.get("price") and baseline.get("S1"):
        # Invert S1 to recover a price context the buyer thinks in.
        # S1 = 100 * min_price / price   →   min_price = price * S1 / 100
        # baseline_S1 = 100 * min_price / avg_price → avg_price = min_price / (baseline_S1/100)
        min_price = row["price"] * row["S1"] / 100
        avg_price = min_price / (baseline["S1"] / 100)
        delta     = avg_price - row["price"]
        if delta > 1:
            price_note = f"INR {delta:,.0f} cheaper than the shortlist average"
        elif delta < -1:
            price_note = f"INR {-delta:,.0f} more expensive than the shortlist average"
        else:
            price_note = "priced at the shortlist average"
        parts.append(
            f"Priced at INR {row['price']:,.0f}/{row.get('unit', 'unit')} — "
            f"{price_note} (S₁ = {row['S1']})."
        )
    elif row.get("S1") is not None and row.get("price"):
        parts.append(
            f"Priced at INR {row['price']:,.0f}/{row.get('unit', 'unit')} "
            f"(S₁ = {row['S1']})."
        )
    unavailable = [c["label"] for c in CRITERIA if not c["impl"]]
    if unavailable:
        parts.append(
            f"Criteria not yet computable from quote-only data: {', '.join(unavailable)}."
        )
    return " ".join(parts)