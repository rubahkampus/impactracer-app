"""Targeted regression tests for the Crucible refactor (Sprint 11).

Each block locks in one of the architectural-horizon invariants so a future
edit cannot silently regress them.
"""

from __future__ import annotations

import math

import pytest

from impactracer.evaluation.metrics import (
    compute_r_precision,
    compute_set_metrics,
)
from impactracer.pipeline.retriever import (
    _BM25_STOPWORDS,
    _tokenize_for_bm25,
    apply_negative_filter,
)
from impactracer.shared.constants import (
    EDGE_CONFIG,
    EXCLUDED_PROPAGATION_NODE_TYPES,
    NODE_TYPE_MAX_FAN_IN,
    UTILITY_FILE_CALLS_DEPTH_CAP,
)
from impactracer.shared.models import Candidate


# ---------------------------------------------------------------------------
# Crucible Fix 4 — set-level metrics
# ---------------------------------------------------------------------------


def test_set_metrics_perfect_match():
    metrics = compute_set_metrics({"a", "b", "c"}, {"a", "b", "c"})
    assert metrics["precision_set"] == 1.0
    assert metrics["recall_set"] == 1.0
    assert metrics["f1_set"] == 1.0


def test_set_metrics_disjoint():
    metrics = compute_set_metrics({"a", "b"}, {"c", "d"})
    assert metrics["precision_set"] == 0.0
    assert metrics["recall_set"] == 0.0
    assert metrics["f1_set"] == 0.0


def test_set_metrics_partial_overlap():
    # 2 of 4 predicted are correct; 2 of 3 GT recovered.
    metrics = compute_set_metrics({"a", "b", "c", "d"}, {"a", "b", "x"})
    assert metrics["precision_set"] == pytest.approx(0.5)
    assert metrics["recall_set"] == pytest.approx(2 / 3)
    assert metrics["f1_set"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_set_metrics_empty_predicted():
    metrics = compute_set_metrics(set(), {"a", "b"})
    assert metrics["precision_set"] == 0.0
    assert metrics["recall_set"] == 0.0
    assert metrics["f1_set"] == 0.0


def test_set_metrics_empty_gt_yields_nan_recall():
    metrics = compute_set_metrics({"a", "b"}, set())
    assert metrics["precision_set"] == 0.0
    assert math.isnan(metrics["recall_set"])
    assert math.isnan(metrics["f1_set"])


def test_set_metrics_distinguishes_graph_flood_from_focused():
    """The architectural-horizon failure mode: set-level metrics MUST give
    different f1 for a 50-node correct-heavy CIS vs a 372-node flood that
    has the same intersection with GT.

    Both predict the same 5 GT nodes, but the flood drowns in noise.
    """
    gt = {f"gt_{i}" for i in range(5)}
    focused = gt | {"noise_a", "noise_b"}
    flood = gt | {f"noise_{i}" for i in range(367)}
    m_focused = compute_set_metrics(focused, gt)
    m_flood = compute_set_metrics(flood, gt)
    # Recall identical (5 of 5 in both), precision must differ dramatically.
    assert m_focused["recall_set"] == 1.0
    assert m_flood["recall_set"] == 1.0
    assert m_focused["precision_set"] > m_flood["precision_set"] * 10
    assert m_focused["f1_set"] > m_flood["f1_set"] * 5


def test_r_precision():
    ranked = ["x", "a", "y", "b", "z", "c"]  # GT a, b, c are at ranks 1, 3, 5
    gt = {"a", "b", "c"}
    # top-3 of ranked = [x, a, y]; intersect with gt = {a} -> 1/3.
    assert compute_r_precision(ranked, gt) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Crucible Fix 5 — BM25 stop-words with len>=2
# ---------------------------------------------------------------------------


def test_bm25_keeps_2char_technical_identifiers():
    """`id`, `db`, `ts`, `js`, `ui` MUST survive tokenization."""
    text = "userId dbConnection tsConfig jsBundle uiKit"
    tokens = _tokenize_for_bm25(text)
    assert "id" in tokens
    assert "db" in tokens
    assert "ts" in tokens
    assert "js" in tokens
    assert "ui" in tokens


def test_bm25_drops_function_words():
    """The stop-word list MUST drop "the", "of", "and", "di", "ke"."""
    tokens = _tokenize_for_bm25("the lookup of the user and di api ke endpoint")
    assert "the" not in tokens
    assert "of" not in tokens
    assert "and" not in tokens
    assert "di" not in tokens
    assert "ke" not in tokens
    # Content tokens survive.
    assert "lookup" in tokens
    assert "user" in tokens
    assert "api" in tokens
    assert "endpoint" in tokens


def test_bm25_camelcase_decomposition():
    tokens = _tokenize_for_bm25("commissionListingPayload")
    assert "commission" in tokens
    assert "listing" in tokens
    assert "payload" in tokens


def test_bm25_stopwords_cover_indonesian_function_words():
    for w in {"di", "ke", "ya", "dan", "yang", "atau"}:
        assert w in _BM25_STOPWORDS, f"stop-word list missing {w!r}"


# ---------------------------------------------------------------------------
# Crucible Fix 13 — additive negative filter
# ---------------------------------------------------------------------------


def _candidate(node_id: str, name: str, snippet: str, raw_score: float) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_type="Function",
        collection="code_units",
        rrf_score=0.0,
        reranker_score=0.0,
        raw_reranker_score=raw_score,
        file_path="src/x.ts",
        name=name,
        text_snippet=snippet,
    )


def test_negative_filter_demotes_substring_match():
    a = _candidate("a", "loginUser", "user login flow", 4.0)
    b = _candidate("b", "logoutUser", "user logout flow", 3.0)
    apply_negative_filter(
        [a, b], out_of_scope_operations=["logout"], penalty=5.0
    )
    assert a.raw_reranker_score == 4.0   # untouched
    assert b.raw_reranker_score == 3.0 - 5.0  # additive penalty
    # Sanity: positive logit became negative, demonstrating the demotion is
    # actually monotonic across sign changes.
    assert b.raw_reranker_score < a.raw_reranker_score


def test_negative_filter_handles_negative_logits_correctly():
    """Crucible mathematical-correctness check: a multiplicative penalty
    on a negative logit would PROMOTE; the additive penalty must DEMOTE
    across the entire real line.
    """
    c = _candidate("c", "logout", "user logout", -4.0)
    apply_negative_filter([c], out_of_scope_operations=["logout"], penalty=5.0)
    # -4.0 * 0.5 = -2.0 (BUG: promotion). -4.0 - 5.0 = -9.0 (correct: demotion).
    assert c.raw_reranker_score == -9.0


def test_negative_filter_default_penalty_is_softer():
    """Apex Crucible V2: default penalty was hardened from -5.0 to -1.0
    after CR-02 forensics showed -5.0 obliterated legitimate domain
    candidates whose names contain a generic OOS substring like 'grace
    period'.
    """
    c = _candidate("c", "logoutHandler", "user logout flow", 3.0)
    apply_negative_filter([c], out_of_scope_operations=["logout"])
    assert c.raw_reranker_score == 2.0  # 3.0 - 1.0 default


def test_negative_filter_ignores_short_phrases():
    """Apex Crucible V2: needles shorter than 6 chars are dropped to avoid
    matching tokens like 'log' or 'add' that appear in many legitimate
    identifiers.
    """
    c = _candidate("c", "loggerHelper", "logger module", 3.0)
    apply_negative_filter([c], out_of_scope_operations=["log"], penalty=5.0)
    assert c.raw_reranker_score == 3.0  # untouched — needle "log" is < 6 chars


def test_negative_filter_matches_name_only_not_snippet():
    """Apex Crucible V2: substring matching on text_snippet caused too many
    false positives. Only the candidate's name is considered.
    """
    c = _candidate("c", "loginUser", "user logout subsystem", 3.0)
    apply_negative_filter([c], out_of_scope_operations=["logout"], penalty=5.0)
    assert c.raw_reranker_score == 3.0  # name doesn't contain "logout"


def test_negative_filter_no_op_when_empty():
    a = _candidate("a", "loginUser", "user login flow", 4.0)
    apply_negative_filter([a], out_of_scope_operations=[])
    assert a.raw_reranker_score == 4.0


# ---------------------------------------------------------------------------
# Crucible Fix 6 — CALLS depth 2
# ---------------------------------------------------------------------------


def test_calls_depth_locked_to_2():
    """EDGE_CONFIG['CALLS'] must specify max_depth=2 (Crucible Fix 6)."""
    assert EDGE_CONFIG["CALLS"]["max_depth"] == 2
    assert EDGE_CONFIG["CALLS"]["direction"] == "reverse"


# ---------------------------------------------------------------------------
# Crucible Fix 11 — propagation limits
# ---------------------------------------------------------------------------


def test_external_package_excluded_from_propagation():
    assert "ExternalPackage" in EXCLUDED_PROPAGATION_NODE_TYPES


def test_node_type_fan_in_caps_present_for_function_method_class():
    for nt in ("Function", "Method", "Class"):
        assert nt in NODE_TYPE_MAX_FAN_IN
        assert NODE_TYPE_MAX_FAN_IN[nt] > 0


def test_utility_file_calls_depth_cap_is_one():
    assert UTILITY_FILE_CALLS_DEPTH_CAP == 1
