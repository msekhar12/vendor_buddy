"""
Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, SIGIR 2009).

Given several ranked lists of the same kind of item, each item gets

    RRF(d) = sum_over_lists  w_i / (k + rank_i(d))        (rank starts at 1)

and items are re-ordered by that score. Only *ranks* are used, never raw
scores, so TF-IDF cosine and Chroma distance (different scales) can be
combined without any normalisation. k (default 60) damps the influence of
the very top ranks; items missing from a list simply contribute nothing.

Items are identified by a caller-supplied key so the same chunk returned by
two different retrievers is recognised as one candidate.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

T = TypeVar("T")


def rrf(
    ranked_lists: Sequence[Sequence[T]],
    key: Callable[[T], Hashable],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[T, float]]:
    """
    Fuse ranked lists into one list of (item, score), best first.

    When an item appears in several lists, the first occurrence (in list
    order) is the representative returned. Ties keep first-seen order, so
    the output is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists in length")

    scores: dict[Hashable, float] = {}
    first: dict[Hashable, T] = {}
    for lst, w in zip(ranked_lists, weights):
        seen_in_list: set[Hashable] = set()
        for rank, item in enumerate(lst, start=1):
            kk = key(item)
            if kk in seen_in_list:          # duplicate inside one list:
                continue                    # only its best rank counts
            seen_in_list.add(kk)
            scores[kk] = scores.get(kk, 0.0) + w / (k + rank)
            first.setdefault(kk, item)

    order = sorted(scores, key=lambda kk: -scores[kk])   # stable sort
    return [(first[kk], scores[kk]) for kk in order]
