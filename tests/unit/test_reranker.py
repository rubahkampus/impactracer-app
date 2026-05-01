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

    def __init__(self, model_name, use_fp16=True):
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
        def __init__(self, model_name, use_fp16=True):
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
