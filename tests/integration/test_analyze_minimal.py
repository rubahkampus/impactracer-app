"""Integration tests for the V0 and V3 online pipeline paths.

Requires:
  - OPENROUTER_API_KEY in env (or .env)
  - An already-indexed citrakara repo (data/impactracer.db + data/chroma_store)

Both tests are skipped when OPENROUTER_API_KEY is absent.
"""

from __future__ import annotations

import os

import pytest

from impactracer.evaluation.variant_flags import VariantFlags
from impactracer.pipeline.retriever import reciprocal_rank_fusion_adaptive
from impactracer.shared.config import Settings


# ---------------------------------------------------------------------------
# Adaptive RRF unit test (no LLM, no DB)
# ---------------------------------------------------------------------------


def test_arrf_formula_weights():
    """Adaptive RRF applies per-change_type path weights correctly."""
    # Three-list fusion for ADDITION change_type
    # dense_doc weight=1.2, bm25_doc=1.0, dense_code=1.0
    ranked_lists = [
        ("dense_doc", ["A", "B", "C"]),
        ("bm25_doc", ["B", "A", "D"]),
        ("dense_code", ["C", "A"]),
    ]
    scores = reciprocal_rank_fusion_adaptive(ranked_lists, "ADDITION", k=60)

    # A appears in all three lists: rank 0 in dense_doc, rank 1 in bm25_doc, rank 1 in dense_code
    # = 1.2/(60+0+1) + 1.0/(60+1+1) + 1.0/(60+1+1)
    expected_A = 1.2 / 61 + 1.0 / 62 + 1.0 / 62
    assert abs(scores["A"] - expected_A) < 1e-9

    # B appears in dense_doc rank 1 and bm25_doc rank 0
    expected_B = 1.2 / 62 + 1.0 / 61
    assert abs(scores["B"] - expected_B) < 1e-9

    # D appears only in bm25_doc rank 2
    expected_D = 1.0 / (60 + 2 + 1)
    assert abs(scores["D"] - expected_D) < 1e-9

    # Higher ARRF score should be ranked first
    assert scores["A"] > scores["B"] > scores["D"]


def test_arrf_single_list_degenerates():
    """With one list, RRF reduces to that list's order."""
    ranked_lists = [("bm25_doc", ["X", "Y", "Z"])]
    scores = reciprocal_rank_fusion_adaptive(ranked_lists, "MODIFICATION", k=60)
    assert scores["X"] > scores["Y"] > scores["Z"]


def test_arrf_modification_weights():
    """MODIFICATION gives higher weight to dense_code than dense_doc."""
    ranked_lists = [
        ("dense_doc", ["D"]),
        ("dense_code", ["D"]),
    ]
    # Both at rank 0 for node D
    scores = reciprocal_rank_fusion_adaptive(ranked_lists, "MODIFICATION", k=60)
    # dense_code weight=1.2, dense_doc weight=1.0 → both contribute
    expected = 1.0 / 61 + 1.2 / 61
    assert abs(scores["D"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# Live integration tests (require OPENROUTER_API_KEY + indexed DB)
# ---------------------------------------------------------------------------


_SKIP_REASON = "OPENROUTER_API_KEY not set — skipping live integration test"
_has_key = bool(os.environ.get("OPENROUTER_API_KEY") or Settings().openrouter_api_key)


@pytest.mark.skipif(not _has_key, reason=_SKIP_REASON)
def test_v3_analyze_commission_pin():
    """V3 end-to-end: pin commission listing CR returns non-empty report."""
    from impactracer.pipeline.runner import run_analysis

    settings = Settings()
    flags = VariantFlags.v3()

    cr = (
        "Tambahkan kemampuan bagi ilustrator untuk menyematkan (pin) maksimal 3 "
        "commission listing ke bagian atas halaman profil publiknya."
    )
    report = run_analysis(cr, settings, flags)

    assert report is not None, "run_analysis returned None"
    assert isinstance(report.impacted_nodes, list)
    assert len(report.impacted_nodes) > 0, (
        f"Expected non-empty impacted_nodes for a meaningful CR. "
        f"executive_summary={report.executive_summary!r}"
    )
    assert report.estimated_scope in ("terlokalisasi", "menengah", "ekstensif")
    assert report.executive_summary  # non-empty string


@pytest.mark.skipif(not _has_key, reason=_SKIP_REASON)
def test_v0_analyze_commission_pin():
    """V0 (BM25-only) also returns a parseable ImpactReport."""
    from impactracer.pipeline.runner import run_analysis

    settings = Settings()
    flags = VariantFlags.v0()

    cr = (
        "Tambahkan kemampuan bagi ilustrator untuk menyematkan (pin) maksimal 3 "
        "commission listing ke bagian atas halaman profil publiknya."
    )
    report = run_analysis(cr, settings, flags)

    assert report is not None
    assert isinstance(report.impacted_nodes, list)
    # V0 may return fewer nodes than V3 but must be parseable
    assert report.executive_summary


@pytest.mark.skipif(not _has_key, reason=_SKIP_REASON)
def test_non_actionable_cr_short_circuits():
    """A vague CR ('improve performance') produces zero impacted_nodes."""
    from impactracer.pipeline.runner import run_analysis

    settings = Settings()
    flags = VariantFlags.v3()

    report = run_analysis("improve performance", settings, flags)

    assert report is not None
    assert report.impacted_nodes == [], (
        f"Expected empty impacted_nodes for non-actionable CR, got: "
        f"{[n.node_id for n in report.impacted_nodes]}"
    )
    assert "reject" in report.executive_summary.lower() or len(report.impacted_nodes) == 0


@pytest.mark.skipif(not _has_key, reason=_SKIP_REASON)
def test_llm_calls_count_v3(tmp_path):
    """V3 uses exactly 2 LLM calls: interpret + synthesize.

    Uses a per-invocation tmp_path for the audit log so the test is
    fully isolated from accumulated entries in the shared ./data/llm_audit.jsonl.
    A stable before/after slice on a shared append-only file is fragile: any
    other test run or retry writing to the file between the snapshot and the
    analysis completion corrupts the count.  Temp-path isolation avoids this
    entirely — the file starts empty and exactly 2 entries appear after the run.
    """
    from impactracer.pipeline.runner import run_analysis

    import json

    # Isolated settings: same DB/chroma as production, but a fresh audit log
    isolated_audit = tmp_path / "llm_audit.jsonl"
    settings = Settings(llm_audit_log_path=str(isolated_audit))  # type: ignore[call-arg]
    flags = VariantFlags.v3()

    cr = "Ubah batas maksimal file attachment di pesan chat dari 5MB menjadi 10MB."

    run_analysis(cr, settings, flags)

    assert isolated_audit.exists(), "Audit log was not created"
    lines = isolated_audit.read_text(encoding="utf-8").strip().splitlines()
    new_entries = [json.loads(l) for l in lines if l.strip()]

    assert len(new_entries) == 2, f"Expected 2 audit entries for V3, got {len(new_entries)}"
    call_names = [e["call_name"] for e in new_entries]
    assert call_names == ["interpret", "synthesize"]

    # All entries share the same config_hash (NFR-05)
    hashes = {e["config_hash"] for e in new_entries}
    assert len(hashes) == 1, f"config_hash mismatch: {hashes}"
