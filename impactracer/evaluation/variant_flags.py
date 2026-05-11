"""V0..V7 variant definitions (authoritative; supersedes legacy B0-S2).

Each variant is a prefix of an additive chain:

    V0   BM25 only                         + blind resolution
    V1   Dense only                        + blind resolution
    V2   V1 + RRF fusion                   + blind resolution
    V3   V2 + cross-encoder rerank         + blind resolution
    V3.5 V3 + pre-validation gates only    + blind resolution (no LLM #2)
    V4   V3 + gates + LLM #2 SIS validation + blind resolution
    V5   V4 + LLM #3 trace validation      + validated SIS
    V6   V5 + BFS propagation              + blind propagation
    V6.5 V6 + LLM #4 propagation validation (BFS only, no full synthesis changes)
    V7   V6 + LLM #4 propagation validation + full system

V3.5: gates-only isolation (measures gate contribution without LLM #2).
V6.5: BFS + LLM #4 isolation (measures LLM #4 precision contribution
  independently of V7's full system). Enables mechanistic attribution:
  V6→V6.5 isolates LLM #4 precision gain; V5→V6 isolates BFS recall gain.

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
    enable_score_floor: bool           # calibrated min reranker score
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

    # Forced-inclusion bypasses LLM #5 synthesis and builds ImpactReport directly
    # from all CIS nodes. Used to isolate LLM #5's selection contribution.
    force_include_all_cis_nodes: bool = False

    ALL_VARIANTS: ClassVar[list[str]] = [
        "V0", "V1", "V2", "V3", "V3.5", "V4", "V5", "V6", "V6.5", "V7"
    ]

    @classmethod
    def v0(cls) -> "VariantFlags":
        return cls(
            variant_id="V0",
            enable_bm25=True, enable_dense=False, enable_rrf=False,
            enable_cross_encoder=False,
            enable_score_floor=False,
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
            enable_score_floor=False,
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
            enable_score_floor=False,
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
            enable_score_floor=False,
            enable_dedup_gate=False, enable_plausibility_gate=False,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v3_5(cls) -> "VariantFlags":
        """V3.5: gates only (no LLM #2).

        Isolation variant to measure the independent contribution of the three
        deterministic gates (score floor, semantic dedup, plausibility+affinity)
        to precision/recall WITHOUT LLM #2. V3.5 vs V3 isolates gate contribution;
        V4 vs V3.5 isolates LLM #2 contribution.
        """
        return cls(
            variant_id="V3.5",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_score_floor=True,
            enable_dedup_gate=True, enable_plausibility_gate=True,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v4(cls) -> "VariantFlags":
        return cls(
            variant_id="V4",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_score_floor=True,
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
            enable_score_floor=True,
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
            enable_score_floor=True,
            enable_dedup_gate=True, enable_plausibility_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=True, enable_propagation_validation=False,
        )

    @classmethod
    def v6_5(cls) -> "VariantFlags":
        """V6.5: BFS + LLM #4 propagation validation isolation variant.

        Identical to V7; used to attribute LLM #4's precision contribution
        independently in the ablation table (V6.5 vs V6).
        """
        return cls(
            variant_id="V6.5",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_score_floor=True,
            enable_dedup_gate=True, enable_plausibility_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=True, enable_propagation_validation=True,
        )

    @classmethod
    def v7_full(cls) -> "VariantFlags":
        return cls(
            variant_id="V7",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_score_floor=True,
            enable_dedup_gate=True, enable_plausibility_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=True, enable_propagation_validation=True,
        )

    @classmethod
    def for_id(cls, variant_id: str) -> "VariantFlags":
        """Return the VariantFlags instance for a given V0..V7 (or V3.5/V6.5) id."""
        return {
            "V0": cls.v0(),
            "V1": cls.v1(),
            "V2": cls.v2(),
            "V3": cls.v3(),
            "V3.5": cls.v3_5(),
            "V4": cls.v4(),
            "V5": cls.v5(),
            "V6": cls.v6(),
            "V6.5": cls.v6_5(),
            "V7": cls.v7_full(),
        }[variant_id]
