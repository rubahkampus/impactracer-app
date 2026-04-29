"""V0..V7 variant definitions (authoritative; supersedes legacy B0-S2).

Each variant is a prefix of an additive chain:

    V0 BM25 only                         + blind resolution
    V1 Dense only                        + blind resolution
    V2 V1 + RRF fusion                   + blind resolution
    V3 V2 + cross-encoder rerank         + blind resolution
    V4 V3 + LLM #2 SIS validation        + blind resolution
    V5 V4 + LLM #3 trace validation      + validated SIS
    V6 V5 + BFS propagation              + blind propagation
    V7 V6 + LLM #4 propagation validation + full system

Reference: 09_ablation_harness.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class VariantFlags:
    """Per-variant feature toggles consumed by the pipeline runner."""

    variant_id: str

    # Retrieval
    enable_bm25: bool
    enable_dense: bool
    enable_rrf: bool
    enable_cross_encoder: bool

    # Gates (FR-C4)
    enable_dedup_gate: bool
    enable_plausibility_gate: bool

    # Validation LLMs
    enable_sis_validation: bool           # LLM #2
    enable_trace_validation: bool         # LLM #3
    enable_bfs: bool                      # Step 6
    enable_propagation_validation: bool   # LLM #4

    # Always-on
    run_llm_1: bool = True                # Interpret
    run_llm_5: bool = True                # Synthesize

    ALL_VARIANTS: ClassVar[list[str]] = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]

    @classmethod
    def v0(cls) -> "VariantFlags":
        return cls(
            variant_id="V0",
            enable_bm25=True, enable_dense=False, enable_rrf=False,
            enable_cross_encoder=False,
            enable_dedup_gate=False, enable_plausibility_gate=False,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v1(cls) -> "VariantFlags":
        return cls(
            variant_id="V1",
            enable_bm25=False, enable_dense=True, enable_rrf=False,
            enable_cross_encoder=False,
            enable_dedup_gate=False, enable_plausibility_gate=False,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v2(cls) -> "VariantFlags":
        return cls(
            variant_id="V2",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=False,
            enable_dedup_gate=False, enable_plausibility_gate=False,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v3(cls) -> "VariantFlags":
        return cls(
            variant_id="V3",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=False, enable_plausibility_gate=False,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v4(cls) -> "VariantFlags":
        return cls(
            variant_id="V4",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True, enable_plausibility_gate=True,
            enable_sis_validation=True, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v5(cls) -> "VariantFlags":
        return cls(
            variant_id="V5",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True, enable_plausibility_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v6(cls) -> "VariantFlags":
        return cls(
            variant_id="V6",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True, enable_plausibility_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=True, enable_propagation_validation=False,
        )

    @classmethod
    def v7_full(cls) -> "VariantFlags":
        return cls(
            variant_id="V7",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True, enable_plausibility_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=True, enable_propagation_validation=True,
        )

    @classmethod
    def for_id(cls, variant_id: str) -> "VariantFlags":
        """Return the VariantFlags instance for a given V0..V7 id."""
        return {
            "V0": cls.v0(),
            "V1": cls.v1(),
            "V2": cls.v2(),
            "V3": cls.v3(),
            "V4": cls.v4(),
            "V5": cls.v5(),
            "V6": cls.v6(),
            "V7": cls.v7_full(),
        }[variant_id]
