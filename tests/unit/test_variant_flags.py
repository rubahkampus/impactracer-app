"""Sprint 11 acceptance test: variant flag consistency."""

from __future__ import annotations

from impactracer.evaluation.variant_flags import VariantFlags


def test_all_variants_constructable() -> None:
    for vid in VariantFlags.ALL_VARIANTS:
        v = VariantFlags.for_id(vid)
        assert v.variant_id == vid


def test_v7_has_everything_enabled() -> None:
    v7 = VariantFlags.v7_full()
    assert v7.enable_bm25
    assert v7.enable_dense
    assert v7.enable_rrf
    assert v7.enable_cross_encoder
    assert v7.enable_sis_validation
    assert v7.enable_trace_validation
    assert v7.enable_bfs
    assert v7.enable_propagation_validation


def test_v0_has_minimal_components() -> None:
    v0 = VariantFlags.v0()
    assert v0.enable_bm25
    assert not v0.enable_dense
    assert not v0.enable_rrf
    assert not v0.enable_cross_encoder
    assert not v0.enable_sis_validation
    assert not v0.enable_trace_validation
    assert not v0.enable_bfs
    assert not v0.enable_propagation_validation


def test_v3_5_isolation_variant() -> None:
    """N7: V3.5 enables all gates but NOT LLM #2, isolating gate contribution."""
    v = VariantFlags.v3_5()
    assert v.variant_id == "V3.5"
    # Has reranker
    assert v.enable_cross_encoder
    # Has all three gates
    assert v.enable_score_floor
    assert v.enable_dedup_gate
    assert v.enable_plausibility_gate
    # Does NOT have LLM #2 (SIS validation) — this is the isolation point
    assert not v.enable_sis_validation
    # No BFS or downstream LLMs
    assert not v.enable_trace_validation
    assert not v.enable_bfs
    assert not v.enable_propagation_validation


def test_v3_5_in_all_variants() -> None:
    assert "V3.5" in VariantFlags.ALL_VARIANTS


def test_variant_chain_is_strictly_additive() -> None:
    """V4 has everything V3.5 has, plus SIS validation (LLM #2)."""
    v3_5 = VariantFlags.v3_5()
    v4 = VariantFlags.v4()
    assert v4.enable_score_floor == v3_5.enable_score_floor
    assert v4.enable_dedup_gate == v3_5.enable_dedup_gate
    assert v4.enable_plausibility_gate == v3_5.enable_plausibility_gate
    # V4 adds SIS validation
    assert v4.enable_sis_validation and not v3_5.enable_sis_validation
