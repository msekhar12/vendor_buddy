"""
Vendor entity resolution for CIVSA.

A single source of truth for turning a raw string ("nirmala chem",
"Nirmala Chemicals Pvt. Ltd.", "Nirmala") into the canonical vendor
name stored on disk ("Nirmala Chemicals Pvt Ltd").

Strategy (in order):
  1. Exact case-insensitive match against the vendor directory list.
  2. Normalised match (strip suffixes like "Pvt Ltd", "Inc", punctuation).
  3. Alias file lookup (vendors/aliases.json — user-editable).
  4. Fuzzy match via difflib.SequenceMatcher on normalised names.
  5. Substring match (rare but useful for very short queries like "nirmala").

Returns:
  ResolveResult(canonical, confidence, method, candidates)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from .config import DOC_STORE

# Common corporate suffixes we strip when normalising.
_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt ltd.", "p ltd",
    "pvt ltd company", "limited", "ltd", "ltd.",
    "llp", "llc", "inc", "inc.", "incorporated", "corp", "corp.",
    "corporation", "company", "co", "co.",
    "gmbh", "s.a.", "sa",
    "chemicals", "chem", "scientific", "reagents", "lab", "labs",
    "solutions", "supplies", "industries", "enterprises", "traders",
]

_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _SUFFIXES) + r")\b\.?",
    re.IGNORECASE,
)
_PUNCT_PATTERN = re.compile(r"[^\w\s]")
_WS_PATTERN = re.compile(r"\s+")


def _normalise(name: str) -> str:
    """Lowercase, strip suffixes, punctuation, and collapse whitespace."""
    s = name.lower().strip()
    s = _SUFFIX_PATTERN.sub(" ", s)
    s = _PUNCT_PATTERN.sub(" ", s)
    s = _WS_PATTERN.sub(" ", s).strip()
    return s


@dataclass
class ResolveResult:
    canonical: Optional[str]           # matched vendor name (as stored)
    confidence: float                  # 0.0 – 1.0
    method: str                        # "exact" | "normalised" | "alias" | "fuzzy" | "substring" | "none"
    candidates: list[tuple[str, float]] = field(default_factory=list)  # top-3 for "did you mean"


class VendorIndex:
    """
    Live index of the vendors known to STORAGE, refreshed on demand.

    Cheap enough to call refresh() before every resolve() in a small
    corpus; if the vendor list grows past a few thousand, cache with a
    TTL or watch the filesystem.
    """

    # Tunables.
    FUZZY_ACCEPT = 0.82        # ratio above which fuzzy is treated as a hit
    SUBSTR_ACCEPT = 0.70       # ratio for substring hits

    def __init__(self, storage_root: Path, alias_file: Optional[Path] = None):
        self.root = Path(storage_root)
        self.alias_file = alias_file or (self.root / "aliases.json")
        self._canonical: list[str] = []
        self._norm_to_canonical: dict[str, str] = {}
        self._aliases: dict[str, str] = {}   # normalised alias → canonical
        self.refresh()

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Rescan the storage root and reload the alias file."""
        if self.root.exists():
            self._canonical = sorted(
                p.name for p in self.root.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        else:
            self._canonical = []
        self._norm_to_canonical = {
            _normalise(v): v for v in self._canonical
        }
        self._load_aliases()

    def _load_aliases(self) -> None:
        self._aliases = {}
        if not self.alias_file.exists():
            return
        try:
            data = json.loads(self.alias_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        # Alias file shape: {"nirmala chem": "Nirmala Chemicals Pvt Ltd", ...}
        for alias, canonical in data.items():
            if canonical in self._canonical:
                self._aliases[_normalise(alias)] = canonical

    def add_alias(self, alias: str, canonical: str) -> None:
        """Persist a user-taught alias like 'nirmala chem' -> canonical."""
        if canonical not in self._canonical:
            raise ValueError(f"Unknown canonical vendor: {canonical}")
        data = {}
        if self.alias_file.exists():
            try:
                data = json.loads(self.alias_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                data = {}
        data[alias] = canonical
        self.alias_file.parent.mkdir(parents=True, exist_ok=True)
        self.alias_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._aliases[_normalise(alias)] = canonical

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, query: str) -> ResolveResult:
        """
        Turn any raw vendor mention into a canonical vendor (or none).

        Never raises — returns ResolveResult(canonical=None, ...) if
        nothing meets the confidence bar.
        """
        if not query or not query.strip():
            return ResolveResult(None, 0.0, "none")

        q_raw = query.strip()
        q_lc = q_raw.lower()
        q_norm = _normalise(q_raw)
        if not q_norm:
            return ResolveResult(None, 0.0, "none")

        # 1. Exact case-insensitive match against canonical names.
        for v in self._canonical:
            if v.lower() == q_lc:
                return ResolveResult(v, 1.0, "exact")

        # 2. Normalised exact match.
        if q_norm in self._norm_to_canonical:
            return ResolveResult(self._norm_to_canonical[q_norm], 0.97, "normalised")

        # 3. Alias file lookup.
        if q_norm in self._aliases:
            return ResolveResult(self._aliases[q_norm], 0.95, "alias")

        # 3.5. First-word match — user typed the distinctive first token.
        #      Covers short queries like "rajshree" or "kumar" that would
        #      otherwise fail because the fuzzy/substring ratios penalise
        #      long trailing words the query didn't include.
        q_first_tokens = q_norm.split()
        if q_first_tokens:
            q_first = q_first_tokens[0]
            first_word_hits = [
                v for v in self._canonical
                if (_normalise(v).split() or [v.lower()])[0] == q_first
            ]
            if len(first_word_hits) == 1:
                return ResolveResult(first_word_hits[0], 0.90, "first_word")
            # Ambiguous (two vendors share a first word) → fall through to
            # fuzzy/substring, which will surface both in `candidates`.

        # 4. Fuzzy match (SequenceMatcher on normalised strings).
        scored: list[tuple[str, float]] = []
        for v in self._canonical:
            v_norm = _normalise(v)
            if not v_norm:
                continue
            ratio = SequenceMatcher(None, q_norm, v_norm).ratio()
            scored.append((v, ratio))
        scored.sort(key=lambda t: t[1], reverse=True)
        top3 = scored[:3]

        if scored and scored[0][1] >= self.FUZZY_ACCEPT:
            best, r = scored[0]
            return ResolveResult(best, r, "fuzzy", top3)

        # 5. Substring: query is contained in a canonical (or vice versa).
        substr_hits: list[tuple[str, float]] = []
        for v in self._canonical:
            v_norm = _normalise(v)
            if not v_norm:
                continue
            if q_norm in v_norm or v_norm in q_norm:
                # Score = length ratio, so short queries still surface a hit.
                r = min(len(q_norm), len(v_norm)) / max(len(q_norm), len(v_norm))
                substr_hits.append((v, r))
        substr_hits.sort(key=lambda t: t[1], reverse=True)
        if substr_hits and substr_hits[0][1] >= self.SUBSTR_ACCEPT:
            best, r = substr_hits[0]
            return ResolveResult(best, r, "substring", top3 or substr_hits[:3])

        # 6. Nothing confident — return the top-3 candidates for "did you mean".
        return ResolveResult(None, scored[0][1] if scored else 0.0, "none", top3)

    def suggest(self, query: str, n: int = 5) -> list[tuple[str, float]]:
        """Return top-N fuzzy candidates without a confidence threshold."""
        q_norm = _normalise(query)
        scored = [
            (v, SequenceMatcher(None, q_norm, _normalise(v)).ratio())
            for v in self._canonical
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:n]

    @property
    def vendors(self) -> list[str]:
        """The current canonical vendor list."""
        return list(self._canonical)


# ----------------------------------------------------------------------
# Module-level singleton — created lazily so tests can override STORAGE.
# ----------------------------------------------------------------------

_INSTANCE: Optional[VendorIndex] = None


def get_vendor_index(storage_root: Optional[Path] = None) -> VendorIndex:
    """Get or create the process-wide VendorIndex."""
    global _INSTANCE
    if _INSTANCE is None:
        if storage_root is None:       # ← new
            storage_root = DOC_STORE  
        _INSTANCE = VendorIndex(storage_root)
    return _INSTANCE


def refresh_vendor_index() -> None:
    """Force a rescan — call after any add/rename/delete of vendor folders."""
    if _INSTANCE is not None:
        _INSTANCE.refresh()