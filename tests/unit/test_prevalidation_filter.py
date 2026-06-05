"""Unit tests for pipeline/prevalidation_filter.py (FR-C4).

Blueprint: master_blueprint.md §4 Steps 3.5–3.7.
"""

from __future__ import annotations

import sqlite3

import pytest

from impactracer.pipeline.prevalidation_filter import (
    apply_prevalidation_gates,
    step_3_6_semantic_dedup,
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
    max_per_file=None,  # Retained for backwards-compat but unused (no per-file count cap).
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

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

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
        enable_dedup=False,
    )
    # Nothing dropped — all gates disabled
    assert len(result) == 1


def test_apply_gates_only_dedup_active():
    # Post-retirement contract: score floor (3.5) and plausibility (3.7) are
    # RETIRED (inert regardless of flag); only semantic dedup (3.6) is active.
    # The score floor would have dropped c_low (below 0.5) — it must NOT.
    # Dedup MUST still merge the doc into its resolved code candidate.
    c_low = _make_code_candidate(node_id="low", reranker_score=0.1)
    c_high = _make_code_candidate(node_id="high", reranker_score=0.8)
    doc = _make_doc_candidate(node_id="sdd__v_1")
    code = _make_code_candidate()
    conn = _make_db_with_candidates([("sdd__v_1", code.node_id, 0.7)])
    cr = _make_cr()
    settings = _make_settings(min_reranker_score=0.5)
    result = apply_prevalidation_gates(
        [c_low, c_high, doc, code], cr, settings, conn,
        enable_dedup=True,
    )
    # Score floor RETIRED: c_low (0.1 < 0.5) survives.
    assert any(c.node_id == "low" for c in result)
    # Plausibility RETIRED: no density drop.
    # Dedup ACTIVE: the doc is merged into its resolved code candidate and dropped.
    assert doc not in result
    assert "sdd__v_1" in code.merged_doc_ids


def test_apply_gates_dedup_can_be_disabled():
    # With enable_dedup=False, even dedup is a no-op (full pass-through).
    doc = _make_doc_candidate(node_id="sdd__v_1")
    code = _make_code_candidate()
    conn = _make_db_with_candidates([("sdd__v_1", code.node_id, 0.7)])
    cr = _make_cr()
    settings = _make_settings()
    candidates = [doc, code]
    result = apply_prevalidation_gates(
        candidates, cr, settings, conn,
        enable_dedup=False,
    )
    assert result == candidates
    assert "sdd__v_1" not in code.merged_doc_ids


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

