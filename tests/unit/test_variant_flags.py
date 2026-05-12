"""Variant flag consistency for the canonical 8-variant chain."""

from __future__ import annotations

from impactracer.evaluation.variant_flags import VariantFlags


def test_all_variants_constructable() -> None:
    for vid in VariantFlags.ALL_VARIANTS:
        v = VariantFlags.for_id(vid)
        assert v.variant_id == vid


def test_all_variants_is_canonical_eight() -> None:
    assert VariantFlags.ALL_VARIANTS == [
        "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"
    ]


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


def test_v3_includes_cross_encoder_and_all_three_gates() -> None:
    """V3 absorbs the former V3.5 — cross-encoder rerank plus all three
    deterministic gates, still no LLM #2. V3 is the deterministic-filtering peak.
    """
    v = VariantFlags.v3()
    assert v.variant_id == "V3"
    assert v.enable_cross_encoder
    assert v.enable_score_floor
    assert v.enable_dedup_gate
    assert v.enable_plausibility_gate
    # No LLM gating yet — that is V4.
    assert not v.enable_sis_validation
    assert not v.enable_trace_validation
    assert not v.enable_bfs
    assert not v.enable_propagation_validation


def test_v4_adds_llm2_on_top_of_v3() -> None:
    """V4 has everything V3 has, plus SIS validation (LLM #2)."""
    v3 = VariantFlags.v3()
    v4 = VariantFlags.v4()
    assert v4.enable_score_floor == v3.enable_score_floor
    assert v4.enable_dedup_gate == v3.enable_dedup_gate
    assert v4.enable_plausibility_gate == v3.enable_plausibility_gate
    assert v4.enable_cross_encoder == v3.enable_cross_encoder
    # V4 alone adds SIS validation.
    assert v4.enable_sis_validation and not v3.enable_sis_validation


def test_v3_5_and_v6_5_are_removed() -> None:
    """Diagnostic-only V3.5 / V6.5 have been folded into V3 / V7."""
    assert not hasattr(VariantFlags, "v3_5")
    assert not hasattr(VariantFlags, "v6_5")
    assert "V3.5" not in VariantFlags.ALL_VARIANTS
    assert "V6.5" not in VariantFlags.ALL_VARIANTS
