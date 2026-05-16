"""Unit tests for pipeline/prevalidation_filter.py (FR-C4).

Blueprint: master_blueprint.md §4 Steps 3.5–3.7.
"""

from __future__ import annotations

import sqlite3

import pytest

from impactracer.pipeline.prevalidation_filter import (
    _matches_any_named,
    _primary_chunk_type,
    apply_prevalidation_gates,
    step_3_5_score_filter,
    step_3_6_semantic_dedup,
    step_3_7_plausibility_and_affinity,
)
from impactracer.shared.models import Candidate, CRInterpretation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cr(
    affected_layers=None,
    named_entry_points=None,
    out_of_scope_operations=None,
    change_type="ADDITION",
) -> CRInterpretation:
    return CRInterpretation(
        is_actionable=True,
        primary_intent="Test CR",
        change_type=change_type,
        affected_layers=affected_layers or ["requirement", "design", "code"],
        domain_concepts=["test"],
        search_queries=["test query", "another query"],
        named_entry_points=named_entry_points or [],
        out_of_scope_operations=out_of_scope_operations or [],
    )


def _make_code_candidate(
    node_id="src/lib/services/auth.service.ts::loginUser",
    file_path="src/lib/services/auth.service.ts",
    file_classification="UTILITY",
    reranker_score=0.8,
    raw_reranker_score: float | None = None,
    name="loginUser",
) -> Candidate:
    # Phase 2.6: raw_reranker_score is the authoritative quality signal.
    # If not explicitly provided, mirror reranker_score so existing tests
    # stay meaningful (they were written before the raw/normalized split).
    effective_raw = reranker_score if raw_reranker_score is None else raw_reranker_score
    return Candidate(
        node_id=node_id,
        node_type="Function",
        collection="code_units",
        rrf_score=0.5,
        reranker_score=reranker_score,
        raw_reranker_score=effective_raw,
        file_path=file_path,
        file_classification=file_classification,
        name=name,
        text_snippet="function loginUser() {...}",
    )


def _make_doc_candidate(
    node_id="sdd__v_1_autentikasi",
    chunk_type="Design",
    reranker_score=0.5,
) -> Candidate:
    return Candidate(
        node_id=node_id,
        node_type="DocChunk",
        collection="doc_chunks",
        rrf_score=0.3,
        reranker_score=reranker_score,
        file_path="docs/sdd.md",
        chunk_type=chunk_type,
        name=node_id,
        text_snippet="Authentication design section.",
    )


def _make_settings(
    min_reranker_score=-2.0,
    density_threshold=0.50,
    max_per_file=None,  # Crucible Fix 9: parameter retained for backwards-compat but unused.
):
    class _Settings:
        min_reranker_score_for_validation = min_reranker_score
        plausibility_gate_density_threshold = density_threshold
    return _Settings()


def _make_db_with_candidates(pairs: list[tuple[str, str, float]]) -> sqlite3.Connection:
    """Create in-memory DB with doc_code_candidates rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE doc_code_candidates (
            doc_id TEXT,
            code_id TEXT,
            weighted_similarity_score REAL
        )
    """)
    conn.executemany(
        "INSERT INTO doc_code_candidates VALUES (?, ?, ?)", pairs
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Step 3.5 — Score Floor
# ---------------------------------------------------------------------------

def test_3_5_keeps_above_threshold():
    c1 = _make_code_candidate(reranker_score=0.8)
    c2 = _make_code_candidate(node_id="b", reranker_score=0.3)
    result = step_3_5_score_filter([c1, c2], threshold=0.5)
    assert len(result) == 1
    assert result[0].node_id == c1.node_id


def test_3_5_zero_threshold_keeps_all():
    c1 = _make_code_candidate(reranker_score=0.0)
    c2 = _make_code_candidate(node_id="b", reranker_score=0.1)
    result = step_3_5_score_filter([c1, c2], threshold=0.0)
    assert len(result) == 2


def test_3_5_drops_all_below_threshold():
    c1 = _make_code_candidate(reranker_score=0.1)
    result = step_3_5_score_filter([c1], threshold=0.5)
    assert result == []


def test_3_5_exact_threshold_boundary_admits():
    c = _make_code_candidate(reranker_score=0.5)
    result = step_3_5_score_filter([c], threshold=0.5)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Step 3.6 — Semantic Dedup
# ---------------------------------------------------------------------------

def test_3_6_merges_doc_with_resolved_code():
    doc = _make_doc_candidate(node_id="sdd__v_1")
    code = _make_code_candidate(node_id="src/lib/services/auth.service.ts::loginUser")
    conn = _make_db_with_candidates([("sdd__v_1", "src/lib/services/auth.service.ts::loginUser", 0.7)])

    result = step_3_6_semantic_dedup([doc, code], conn)

    assert len(result) == 1
    assert result[0].node_id == code.node_id
    assert "sdd__v_1" in result[0].merged_doc_ids


def test_3_6_keeps_doc_with_no_resolution():
    doc = _make_doc_candidate(node_id="sdd__v_1")
    conn = _make_db_with_candidates([])  # no candidates

    result = step_3_6_semantic_dedup([doc], conn)
    assert len(result) == 1
    assert result[0].node_id == doc.node_id


def test_3_6_keeps_doc_resolved_to_absent_code():
    doc = _make_doc_candidate(node_id="sdd__v_1")
    # resolved code is NOT in candidates list
    conn = _make_db_with_candidates([("sdd__v_1", "src/other/node.ts::fn", 0.7)])

    result = step_3_6_semantic_dedup([doc], conn)
    assert len(result) == 1


def test_3_6_preserves_code_candidates():
    code = _make_code_candidate()
    conn = _make_db_with_candidates([])
    result = step_3_6_semantic_dedup([code], conn)
    assert len(result) == 1
    assert result[0].node_id == code.node_id


def test_3_6_multiple_docs_same_code():
    doc1 = _make_doc_candidate(node_id="sdd__a")
    doc2 = _make_doc_candidate(node_id="sdd__b")
    code = _make_code_candidate(node_id="src/lib/services/wallet.service.ts")
    conn = _make_db_with_candidates([
        ("sdd__a", "src/lib/services/wallet.service.ts", 0.7),
        ("sdd__b", "src/lib/services/wallet.service.ts", 0.65),
    ])

    result = step_3_6_semantic_dedup([doc1, doc2, code], conn)
    assert len(result) == 1
    assert set(result[0].merged_doc_ids) == {"sdd__a", "sdd__b"}


# ---------------------------------------------------------------------------
# Step 3.7 — Plausibility + Affinity
# ---------------------------------------------------------------------------

def test_3_7_affinity_rescores_doc_in_layer():
    cr = _make_cr(affected_layers=["design"])
    doc = _make_doc_candidate(chunk_type="Design", reranker_score=1.0)
    settings = _make_settings()
    result = step_3_7_plausibility_and_affinity([doc], cr, settings)
    # Design is in affected_layers → factor=1.0 → score unchanged
    assert result[0].reranker_score == pytest.approx(1.0)


def test_3_7_affinity_rescores_doc_out_of_layer():
    cr = _make_cr(affected_layers=["design"])  # only design
    doc = _make_doc_candidate(chunk_type="FR", reranker_score=1.0)
    settings = _make_settings()
    result = step_3_7_plausibility_and_affinity([doc], cr, settings)
    # FR not in ["Design"] → factor=0.7
    assert result[0].reranker_score == pytest.approx(0.7)


def test_3_7_affinity_rescores_code_candidate():
    cr = _make_cr(affected_layers=["requirement", "design", "code"])
    # Two candidates from different files so density gate doesn't fire.
    code1 = _make_code_candidate(
        node_id="src/lib/a.ts::fn",
        file_path="src/lib/a.ts",
        file_classification="UTILITY",
        reranker_score=1.0,
    )
    code2 = _make_code_candidate(
        node_id="src/lib/b.ts::fn",
        file_path="src/lib/b.ts",
        file_classification="UTILITY",
        reranker_score=0.5,
    )
    settings = _make_settings()
    result = step_3_7_plausibility_and_affinity([code1, code2], cr, settings)
    # UTILITY x FR = 1.0; reranker_score remains 1.0 after affinity.
    assert result[0].reranker_score == pytest.approx(1.0)
    assert len(result) == 2


def test_3_7_affinity_resorts_descending():
    cr = _make_cr(affected_layers=["design"])
    c_low = _make_doc_candidate(node_id="low", chunk_type="Design", reranker_score=0.3)
    c_high = _make_doc_candidate(node_id="high", chunk_type="Design", reranker_score=0.9)
    settings = _make_settings()
    result = step_3_7_plausibility_and_affinity([c_low, c_high], cr, settings)
    assert result[0].node_id == "high"
    assert result[1].node_id == "low"


def test_3_7_density_gate_drops_flooded_file():
    """Crucible Fix 9: flooded files now drop ALL code candidates (no max_per_file cap).

    Previous semantics: density > threshold AND admitted >= max_per_file -> drop.
    New semantics: density > threshold -> drop entire file's candidates unless
    matched by a named entry point. The fix removes the arbitrary 2-per-file
    cap; density alone is the gate.
    """
    cr = _make_cr()
    settings = _make_settings(density_threshold=0.3)
    # 4 candidates all from the same file -> fraction=1.0 > 0.3 -> all dropped.
    candidates = [
        _make_code_candidate(
            node_id=f"src/lib/svc.ts::fn{i}",
            file_path="src/lib/svc.ts",
            reranker_score=1.0 - i * 0.1,
        )
        for i in range(4)
    ]
    result = step_3_7_plausibility_and_affinity(candidates, cr, settings)
    assert len(result) == 0


def test_3_7_density_gate_named_entry_point_exempt():
    """Named entry points are exempt from density-based exclusion."""
    cr = _make_cr(named_entry_points=["createListing"])
    settings = _make_settings(density_threshold=0.3)
    candidates = [
        _make_code_candidate(
            node_id="src/lib/svc.ts::createListing",
            file_path="src/lib/svc.ts",
            name="createListing",
            reranker_score=0.9,
        ),
        _make_code_candidate(
            node_id="src/lib/svc.ts::updateListing",
            file_path="src/lib/svc.ts",
            name="updateListing",
            reranker_score=0.8,
        ),
    ]
    result = step_3_7_plausibility_and_affinity(candidates, cr, settings)
    ids = [c.node_id for c in result]
    # createListing is named -> always admitted; updateListing fails density.
    assert "src/lib/svc.ts::createListing" in ids
    assert "src/lib/svc.ts::updateListing" not in ids


def test_3_7_no_gate_when_below_density_threshold():
    """Files below density threshold pass entirely."""
    cr = _make_cr()
    settings = _make_settings(density_threshold=0.7)  # 67% < 70% -> no flood
    candidates = [
        _make_code_candidate(node_id="a/fn1", file_path="a.ts", reranker_score=0.9),
        _make_code_candidate(node_id="a/fn2", file_path="a.ts", reranker_score=0.8),
        _make_code_candidate(node_id="b/fn1", file_path="b.ts", reranker_score=0.7),
    ]
    result = step_3_7_plausibility_and_affinity(candidates, cr, settings)
    assert len(result) == 3


def test_3_7_empty_candidates_ok():
    cr = _make_cr()
    settings = _make_settings()
    result = step_3_7_plausibility_and_affinity([], cr, settings)
    assert result == []


# ---------------------------------------------------------------------------
# _primary_chunk_type helper
# ---------------------------------------------------------------------------

def test_primary_chunk_type_code_or_requirement():
    assert _primary_chunk_type(["code"]) == "FR"
    assert _primary_chunk_type(["requirement"]) == "FR"
    assert _primary_chunk_type(["requirement", "design"]) == "FR"


def test_primary_chunk_type_design_only():
    assert _primary_chunk_type(["design"]) == "Design"


def test_primary_chunk_type_fallback():
    assert _primary_chunk_type([]) == "General"


# ---------------------------------------------------------------------------
# _matches_any_named helper
# ---------------------------------------------------------------------------

def test_matches_any_named_pattern_in_name():
    """N3 fix: only p-in-name direction (pattern is a substring of the function name).

    The caller pre-lowercases patterns before passing them; the function
    lowercases the name internally for case-insensitive comparison.
    """
    # pattern "create" is a substring of name "createListing" → True
    assert _matches_any_named("createListing", ["create"])
    # pattern "createlisting" is NOT a substring of name "create" → False (N3 fix)
    assert not _matches_any_named("create", ["createlisting"])
    # completely unrelated name → False
    assert not _matches_any_named("deleteListing", ["createlisting"])
    # full match: pattern "createlisting" in name_lower "createlisting" → True
    assert _matches_any_named("createListing", ["createlisting"])


def test_matches_any_named_case_insensitive():
    # Caller lowercases patterns; function lowercases name
    assert _matches_any_named("CreateListing", ["createlisting"])


# ---------------------------------------------------------------------------
# apply_prevalidation_gates (integration)
# ---------------------------------------------------------------------------

def test_apply_gates_all_disabled():
    code = _make_code_candidate(reranker_score=0.05)
    conn = _make_db_with_candidates([])
    cr = _make_cr()
    settings = _make_settings(min_reranker_score=0.5)
    result = apply_prevalidation_gates(
        [code], cr, settings, conn,
        enable_score_floor=False, enable_dedup=False, enable_plausibility=False,
    )
    # Nothing dropped — all gates disabled
    assert len(result) == 1


def test_apply_gates_score_floor_active():
    c_low = _make_code_candidate(node_id="low", reranker_score=0.1)
    c_high = _make_code_candidate(node_id="high", reranker_score=0.8)
    conn = _make_db_with_candidates([])
    cr = _make_cr()
    settings = _make_settings(min_reranker_score=0.5)
    result = apply_prevalidation_gates(
        [c_low, c_high], cr, settings, conn,
        enable_score_floor=True, enable_dedup=False, enable_plausibility=False,
    )
    assert len(result) == 1
    assert result[0].node_id == "high"


def test_apply_gates_dedup_merges_doc():
    doc = _make_doc_candidate(node_id="sdd__v_1")
    code = _make_code_candidate()
    conn = _make_db_with_candidates([("sdd__v_1", code.node_id, 0.7)])
    cr = _make_cr()
    settings = _make_settings()
    result = apply_prevalidation_gates(
        [doc, code], cr, settings, conn,
        enable_score_floor=False, enable_dedup=True, enable_plausibility=False,
    )
    assert len(result) == 1
    assert "sdd__v_1" in result[0].merged_doc_ids


# ---------------------------------------------------------------------------
# New B1/B4/B3/N3 targeted tests
# ---------------------------------------------------------------------------

def test_3_6_merged_doc_contexts_populated():
    """B1: merged_doc_contexts carries (name, text_snippet) from the doc candidate."""
    doc = _make_doc_candidate(
        node_id="srs__v_1_pin",
        chunk_type="FR",
        reranker_score=0.5,
    )
    # Override text_snippet to something distinctive
    doc.text_snippet = "Users can pin listings to their profile."
    doc.name = "srs__v_1_pin"

    code = _make_code_candidate(node_id="src/lib/services/auth.service.ts::loginUser")
    conn = _make_db_with_candidates([
        ("srs__v_1_pin", "src/lib/services/auth.service.ts::loginUser", 0.8)
    ])

    result = step_3_6_semantic_dedup([doc, code], conn)

    assert len(result) == 1
    code_node = result[0]
    assert len(code_node.merged_doc_contexts) == 1
    section_title, section_text = code_node.merged_doc_contexts[0]
    # section_title comes from doc.name; section_text from doc.text_snippet
    assert "srs__v_1_pin" in section_title
    assert "Users can pin listings" in section_text


def test_3_5_uses_raw_reranker_score_not_normalized():
    """Phase 2.6: step_3_5 uses raw_reranker_score, not normalized reranker_score."""
    # c1: raw=0.3 (absolute quality below threshold=0.5) → drop
    c1 = _make_code_candidate(node_id="c1", reranker_score=0.8, raw_reranker_score=0.3)
    # c2: raw=0.7 (above threshold) → keep, despite low normalized score
    c2 = _make_code_candidate(node_id="c2", reranker_score=0.2, raw_reranker_score=0.7)

    result = step_3_5_score_filter([c1, c2], threshold=0.5)
    ids = [c.node_id for c in result]
    assert "c1" not in ids   # raw 0.3 < 0.5 → dropped
    assert "c2" in ids        # raw 0.7 ≥ 0.5 → kept


def test_3_5_v0_v2_raw_zero_passes_positive_threshold():
    """Phase 2.6: raw_reranker_score=0.0 (V0-V2, reranker not run) returns 0.0.

    The score floor default for V0-V2 is 0.0, so all candidates pass.
    When V3+ sets raw_reranker_score, the threshold is meaningful.
    This verifies that raw=0.0 fails a strict positive threshold (0.5),
    as intended — the floor gate is disabled for variants where the
    reranker was not run by setting threshold=0.0 in VariantFlags.
    """
    c = _make_code_candidate(reranker_score=0.8, raw_reranker_score=0.0)
    assert c.raw_reranker_score == 0.0
    # raw=0.0 does NOT pass a threshold=0.5; runner sets threshold=0.0 for V0-V2
    result_strict = step_3_5_score_filter([c], threshold=0.5)
    assert len(result_strict) == 0
    # With threshold=0.0 (V0-V2 setting), everything passes
    result_zero = step_3_5_score_filter([c], threshold=0.0)
    assert len(result_zero) == 1


def test_3_7_doc_chunks_exempt_from_density_gate():
    """B3: doc chunks are always admitted from the density gate."""
    cr = _make_cr()
    settings = _make_settings(density_threshold=0.01, max_per_file=1)
    # 3 doc candidates all with the same file_path; density would trigger if they counted
    docs = [
        _make_doc_candidate(node_id=f"srs__doc{i}", reranker_score=0.8 - i * 0.1)
        for i in range(3)
    ]
    result = step_3_7_plausibility_and_affinity(docs, cr, settings)
    # All 3 doc chunks admitted — they are exempt from the density gate
    assert len(result) == 3


def test_3_7_doc_chunks_and_code_independence():
    """B3: doc chunk exemption does not affect code-density calculation.

    Crucible Fix 9 semantics: when a file is flooded (>density_threshold of
    code candidates), ALL code candidates from that file are dropped, but
    doc chunks remain exempt. The previous max_per_file cap is gone.
    """
    cr = _make_cr()
    settings = _make_settings(density_threshold=0.3)
    # 2 code candidates from same file -> 100% density -> all dropped.
    code1 = _make_code_candidate(node_id="src/a.ts::fn1", file_path="src/a.ts", reranker_score=0.9)
    code2 = _make_code_candidate(node_id="src/a.ts::fn2", file_path="src/a.ts", reranker_score=0.7)
    # Doc chunk -> always passes.
    doc = _make_doc_candidate(node_id="srs__doc1", reranker_score=0.6)

    result = step_3_7_plausibility_and_affinity([code1, code2, doc], cr, settings)
    ids = [c.node_id for c in result]
    assert "srs__doc1" in ids                 # doc always admitted
    assert "src/a.ts::fn1" not in ids         # flooded -> dropped
    assert "src/a.ts::fn2" not in ids         # flooded -> dropped
