from civsa.fusion import rrf


def _ids(fused):
    return [item for item, _ in fused]


def test_item_in_both_lists_beats_single_list_leaders():
    a = ["x", "shared", "y"]
    b = ["z", "shared", "w"]
    fused = rrf([a, b], key=lambda s: s)
    assert _ids(fused)[0] == "shared"


def test_score_formula():
    fused = dict(rrf([["a", "b"], ["b"]], key=lambda s: s, k=60))
    assert abs(fused["a"] - 1 / 61) < 1e-12
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-12


def test_weights_shift_ranking():
    a, b = ["p", "q"], ["q", "p"]
    assert _ids(rrf([a, b], key=lambda s: s, weights=[2.0, 1.0]))[0] == "p"
    assert _ids(rrf([a, b], key=lambda s: s, weights=[1.0, 2.0]))[0] == "q"


def test_duplicate_within_list_counts_once_at_best_rank():
    fused = dict(rrf([["a", "a", "b"]], key=lambda s: s, k=60))
    assert abs(fused["a"] - 1 / 61) < 1e-12
    assert abs(fused["b"] - 1 / 63) < 1e-12


def test_custom_key_merges_and_keeps_first_representative():
    a = [("text1", {"source": "s", "from": "vec"})]
    b = [("text1", {"source": "s", "from": "tfidf"})]
    fused = rrf([a, b], key=lambda dm: (dm[1]["source"], dm[0]))
    assert len(fused) == 1
    assert fused[0][0][1]["from"] == "vec"


def test_empty_lists():
    assert rrf([[], []], key=lambda s: s) == []
