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
