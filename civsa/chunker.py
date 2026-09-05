"""
Table-aware document chunker.

Returns list[dict] so it plugs straight into
  vector_store.add_chunks(chunks, meta)
  tfidf_store.add_chunks(chunks, meta)

Each chunk dict carries the legacy keys (text, chunk_index, para_index)
that both stores already read, plus optional extras that describe where
the chunk came from (chunk_type, region_index, row_index, table_header,
prose_index). The stores will pass those extras through as scalar
metadata once you apply the small patch in 2c below.
"""

from __future__ import annotations

import re

from .table_regions import find_table_regions, is_in_region

# --- tuning ----------------------------------------------------------

CHUNK_SIZE     = 700
CHUNK_OVERLAP  = 120
MIN_CHUNK_SIZE = 60


# --- public API ------------------------------------------------------

def chunk_document(text: str) -> list[dict]:
    """
    Yield table-aware chunks for one document.

    Returned dicts always have:
        text        (str)
        chunk_index (int)   — position in the returned list
        para_index  (int)   — legacy field, kept for store compatibility

    Table-row chunks additionally have:
        chunk_type    = "table_row" | "table_terminal"
        region_index  (int)
        row_index     (int)
        table_header  (str)

    Prose chunks additionally have:
        chunk_type    = "prose"
        prose_index   (int)
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    regions = find_table_regions(lines)
    chunks: list[dict] = []

    # 1) Table row chunks (each row + header) and terminal-row chunks
    for r_idx, region in enumerate(regions):
        header_str = " | ".join(l for l in region.header_lines if l)

        for row_idx, row in enumerate(region.data_lines):
            row_text = f"{header_str}\n{row}" if header_str else row
            chunks.append({
                "text":         row_text,
                "chunk_index":  len(chunks),
                "para_index":   0,                # legacy, unused for tables
                "chunk_type":   "table_row",
                "region_index": r_idx,
                "row_index":    row_idx,
                "table_header": header_str,
            })

        for t_idx, term in enumerate(region.terminal_lines):
            term_text = f"{header_str}\n{term}" if header_str else term
            chunks.append({
                "text":         term_text,
                "chunk_index":  len(chunks),
                "para_index":   0,
                "chunk_type":   "table_terminal",
                "region_index": r_idx,
                "row_index":    t_idx,
                "table_header": header_str,
            })

    # 2) Prose chunks — everything outside detected tables
    prose_lines = [
        line for i, line in enumerate(lines)
        if not is_in_region(i, regions)
    ]
    prose_text = "\n".join(prose_lines).strip()
    if prose_text:
        for i, chunk_str in enumerate(_sliding_window(prose_text)):
            if len(chunk_str) < MIN_CHUNK_SIZE:
                continue
            chunks.append({
                "text":        chunk_str,
                "chunk_index": len(chunks),
                "para_index":  i,                # keep this useful for prose
                "chunk_type":  "prose",
                "prose_index": i,
            })

    return chunks


# --- prose windowing ------------------------------------------------

_SENT_BOUNDARY = re.compile(r"[.!?]\s+|\n\n+")


def _sliding_window(text: str) -> list[str]:
    """Char-based sliding window that prefers to break at sentence ends."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    out: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + CHUNK_SIZE, n)

        # Try to end the chunk at a sentence boundary within the last ~120 chars
        if end < n:
            window_start = max(start, end - 120)
            window = text[window_start:end]
            m = None
            for m_ in _SENT_BOUNDARY.finditer(window):
                m = m_
            if m:
                end = window_start + m.end()

        out.append(text[start:end].strip())

        # If we've reached the end of the text, we're done.
        if end >= n:
            break

        # Advance start; guarantee forward progress even if overlap would
        # otherwise leave us at or before the previous start.
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return out