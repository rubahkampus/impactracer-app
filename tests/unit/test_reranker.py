"""Tests for indexer/reranker.py (FR-C3). Model calls are monkeypatched."""

from __future__ import annotations

import sys
import types

import pytest

from impactracer.shared.models import Candidate


def _make_candidate(node_id: str, snippet: str, rrf: float = 0.5) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_type="Function",
        collection="code_units",
        rrf_score=rrf,
        text_snippet=snippet,
    )


class _FakeFlagReranker:
    """Returns preset scores for each pair based on index."""

    def __init__(self, model_name, use_fp16=True, devices=None, **kwargs):
        pass

    def compute_score(self, pairs, normalize=True):
        # Scores are index-based: pair[0] → 0.9, pair[1] → 0.3, pair[2] → 0.7
        preset = [0.9, 0.3, 0.7, 0.5, 0.1]
        return [preset[i % len(preset)] for i in range(len(pairs))]


@pytest.fixture(autouse=True)
def patch_flag_embedding():
    flag_mod = types.ModuleType("FlagEmbedding")
    flag_mod.FlagReranker = _FakeFlagReranker
    sys.modules["FlagEmbedding"] = flag_mod
    yield
    # Leave the patch in place for the session; subsequent tests re-patch if needed


@pytest.fixture()
def reranker():
    from impactracer.indexer.reranker import Reranker
    return Reranker("BAAI/bge-reranker-v2-m3")


def test_empty_candidates_returns_empty(reranker):
    assert reranker.rerank("query", [], top_k=5) == []


def test_reranker_score_set_in_place(reranker):
    c1 = _make_candidate("n1", "snippet one")
    c2 = _make_candidate("n2", "snippet two")
    reranker.rerank("query", [c1, c2], top_k=5)
    assert c1.reranker_score == pytest.approx(0.9)
    assert c2.reranker_score == pytest.approx(0.3)


def test_sorted_descending(reranker):
    c1 = _make_candidate("n1", "snippet one")
    c2 = _make_candidate("n2", "snippet two")
    c3 = _make_candidate("n3", "snippet three")
    result = reranker.rerank("query", [c1, c2, c3], top_k=5)
    scores = [c.reranker_score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_top_k_truncation(reranker):
    candidates = [_make_candidate(f"n{i}", f"snippet {i}") for i in range(5)]
    result = reranker.rerank("query", candidates, top_k=2)
    assert len(result) == 2


def test_top_k_larger_than_candidates(reranker):
    candidates = [_make_candidate(f"n{i}", f"snippet {i}") for i in range(3)]
    result = reranker.rerank("query", candidates, top_k=10)
    assert len(result) == 3


def test_scores_in_zero_one_range(reranker):
    candidates = [_make_candidate(f"n{i}", f"snippet {i}") for i in range(5)]
    result = reranker.rerank("query", candidates, top_k=5)
    for c in result:
        assert 0.0 <= c.reranker_score <= 1.0


def test_normalize_true_passed_to_model():
    calls = []

    class _RecordingReranker:
        def __init__(self, model_name, use_fp16=True, devices=None, **kwargs):
            pass

        def compute_score(self, pairs, normalize=True):
            calls.append(normalize)
            return [0.5] * len(pairs)

    flag_mod = types.ModuleType("FlagEmbedding")
    flag_mod.FlagReranker = _RecordingReranker
    sys.modules["FlagEmbedding"] = flag_mod

    from importlib import reload
    import impactracer.indexer.reranker as mod
    reload(mod)

    r = mod.Reranker("model")
    r.rerank("q", [_make_candidate("n1", "s1")], top_k=1)
    assert calls == [True]


def test_single_candidate(reranker):
    c = _make_candidate("n1", "only snippet")
    result = reranker.rerank("query", [c], top_k=1)
    assert len(result) == 1
    assert result[0].node_id == "n1"
    assert result[0].reranker_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# rerank_multi_query — N4 tests
# ---------------------------------------------------------------------------

def test_multi_query_empty_candidates(reranker):
    result = reranker.rerank_multi_query(["q1", "q2"], "fallback", [], top_k=5)
    assert result == []


def test_multi_query_fallback_when_no_queries(reranker):
    """Falls back to fallback_query when queries is empty."""
    c = _make_candidate("n1", "snippet")
    result = reranker.rerank_multi_query([], "fallback", [c], top_k=5)
    assert len(result) == 1
    assert result[0].node_id == "n1"


def test_multi_query_max_across_queries():
    """reranker_score is set to the max score across all queries (N4)."""
    # _FakeFlagReranker returns index-based scores: 0.9, 0.3, 0.7, ...
    # With 2 queries and 1 candidate:
    #   pairs = [("q1", "snippet"), ("q2", "snippet")]  → scores [0.9, 0.3]
    # max score for n1 across q1 and q2 = max(0.9, 0.3) = 0.9
    from impactracer.indexer.reranker import Reranker
    r = Reranker("model")
    c = _make_candidate("n1", "snippet")
    result = r.rerank_multi_query(["q1", "q2"], "fallback", [c], top_k=5)
    assert result[0].reranker_score == pytest.approx(0.9)


def test_multi_query_sorted_descending(reranker):
    """Results sorted descending by max score."""
    c1 = _make_candidate("n1", "s1")
    c2 = _make_candidate("n2", "s2")
    c3 = _make_candidate("n3", "s3")
    result = reranker.rerank_multi_query(["q1"], "fallback", [c1, c2, c3], top_k=5)
    scores = [c.reranker_score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_multi_query_top_k_respected(reranker):
    """top_k truncation works."""
    candidates = [_make_candidate(f"n{i}", f"s{i}") for i in range(5)]
    result = reranker.rerank_multi_query(["q"], "fallback", candidates, top_k=2)
    assert len(result) == 2


def test_multi_query_prefers_ila_over_snippet():
    """rerank_multi_query uses ILA when available for scoring text."""
    recorded_pairs = []

    class _RecordingReranker:
        def __init__(self, model_name, use_fp16=True, devices=None, **kwargs):
            pass

        def compute_score(self, pairs, normalize=True):
            recorded_pairs.extend(pairs)
            return [0.5] * len(pairs)

    import sys, types
    flag_mod = types.ModuleType("FlagEmbedding")
    flag_mod.FlagReranker = _RecordingReranker
    sys.modules["FlagEmbedding"] = flag_mod

    from importlib import reload
    import impactracer.indexer.reranker as mod
    reload(mod)
    r = mod.Reranker("model")

    c = _make_candidate("n1", "raw snippet")
    c.internal_logic_abstraction = "const x = doThing();"

    r.rerank_multi_query(["q1"], "fallback", [c], top_k=1)
    # The pair text should be the ILA, not the raw snippet
    assert recorded_pairs[0][1] == "const x = doThing();"
