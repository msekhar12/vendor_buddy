"""
Lightweight table-region detector for extracted PDF text.

A "table" here is: >=1 header line containing column-header hints (SKU /
Description / Price / Total / Qty / etc.) followed by >=2 consecutive data
lines that each look row-shaped — currency tokens, comma-grouped numbers,
or SKU-style identifiers.

Good enough for the priced-schedule tables that appear in vendor quotes,
invoices, and BOMs. Not for scanned tables (needs OCR upstream) or for
free-form paragraphs with numbers embedded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ------------ patterns ------------------------------------------------

# INR 1,050  ·  ₹ 850  ·  Rs. 2500  ·  Rs 12,500.00
_PRICE_TOKEN = re.compile(
    r"(?:INR|Rs\.?|₹)\s*[\d,]+(?:\.\d+)?",
    re.IGNORECASE,
)

# 1,050  ·  17,250.00  ·  1050.75  — but NOT bare small ints like "3"
_MONEY_ISH = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d{2}\b",
)

# NCP-2011  ·  ALS-A11  ·  DCR-01  — SKU-style identifier
_SKU_LIKE = re.compile(r"\b[A-Z]{2,}[-\s]?\d{1,}\b")

# Column-header hint words (need at least 2 distinct hits in one line)
_HEADER_HINTS = re.compile(
    r"\b(sku|description|unit|qty|quantity|price|total|amount|rate|item|line\s*total|unit\s*price|hsn|sac)\b",
    re.IGNORECASE,
)

# Terminal rows we still want to keep with the table
_TERMINAL_HINTS = re.compile(
    r"\b(subtotal|sub\s*total|gst|cgst|sgst|igst|tax|grand\s*total|total)\b",
    re.IGNORECASE,
)


# ------------ data class ---------------------------------------------

@dataclass
class TableRegion:
    start_line: int
    end_line: int
    header_lines: list[str]
    data_lines: list[str]
    terminal_lines: list[str] = field(default_factory=list)


# ------------ helpers -------------------------------------------------

def _looks_like_row(line: str) -> bool:
    """True if the line reads like a table row (has money + at least one other signal)."""
    if not line:
        return False
    has_money = bool(_PRICE_TOKEN.search(line)) or bool(_MONEY_ISH.search(line))
    has_sku   = bool(_SKU_LIKE.search(line))
    long_enough = len(line.split()) >= 3
    return (has_money and long_enough) or (has_sku and long_enough)


def _looks_like_header(line: str) -> bool:
    """True if the line reads like a table header — 2+ column-word hits."""
    if not line:
        return False
    hits = _HEADER_HINTS.findall(line)
    return len(hits) >= 2


def _looks_like_terminal(line: str) -> bool:
    return bool(_TERMINAL_HINTS.search(line))


# ------------ main API ------------------------------------------------

def find_table_regions(lines: list[str]) -> list[TableRegion]:
    """
    Find zero-or-more table regions in the given lines (already split on \\n).

    Each region is: an optional 1-2 line header, 2+ consecutive data lines,
    plus any subtotal/GST/total lines immediately after that we want to keep
    attached.
    """
    regions: list[TableRegion] = []
    i = 0
    n = len(lines)

    print("In find_table_regions: scanning", n, "lines…")

    while i < n:
        stripped = lines[i].strip()

        # Try to start a region: header found?
        if _looks_like_header(stripped):
            header_lines = [stripped]
            j = i + 1
            # Headers sometimes wrap into a second row of column names
            if j < n and _looks_like_header(lines[j].strip()):
                header_lines.append(lines[j].strip())
                j += 1

            # Collect rows until we hit 2 blank lines or a non-row non-terminal line
            data_lines: list[str] = []
            terminal_lines: list[str] = []
            blanks = 0
            print(f"In find_table_region  · Found header at line {i}: {header_lines}")
            while j < n:
                print(f"In find_table_region  · Scanning line {j}: {lines[j]!r}")
                cand = lines[j].strip()
                if not cand:
                    blanks += 1
                    if blanks >= 2:
                        break
                    j += 1
                    continue
                blanks = 0
                if _looks_like_row(cand):
                    data_lines.append(cand)
                    j += 1
                    continue
                # Once data rows have started, keep grabbing terminal-ish lines
                # (subtotal, GST, grand total) — they belong to the same region.
                if data_lines and _looks_like_terminal(cand):
                    terminal_lines.append(cand)
                    j += 1
                    continue
                break

            if len(data_lines) >= 2:
                regions.append(TableRegion(
                    start_line=i,
                    end_line=j - 1,
                    header_lines=header_lines,
                    data_lines=data_lines,
                    terminal_lines=terminal_lines,
                ))
                i = j
                continue

        # No table started here — try next line
        i += 1

    return regions


def is_in_region(line_idx: int, regions: list[TableRegion]) -> bool:
    return any(r.start_line <= line_idx <= r.end_line for r in regions)