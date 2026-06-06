"""V0..V7 variant definitions (authoritative).

Canonical 8-variant additive chain:

    V0   BM25-only retrieval                 + blind resolution
    V1   + dense retrieval                   + blind resolution
    V2   + RRF fusion                        + blind resolution
    V3   + cross-encoder rerank              + blind resolution
         V3 is the deterministic-filtering peak (no LLM gating).
    V4   + LLM #2 SIS validation             + blind resolution
    V5   + LLM #3 trace validation           + validated SIS
    V6   + deterministic propagation (both arms, pruned, unvalidated)
    V7   + LLM #4 propagation validation (both arms) + full system

Semantic dedup (Step 2.2) runs on every variant V0-V7 as the single
pre-validation gate; it is unconditional retrieval-level hygiene, not an
ablation boundary. No score floor or plausibility gate exists (both retired).

The dataclass keeps its full boolean flag surface so ad-hoc isolation
variants can still be hand-constructed for diagnostic purposes; only the
canonical 8 enumerated in ``ALL_VARIANTS`` are exercised by the ablation
matrix.

Reference: master_blueprint.md §6 (Ablation Harness).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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

    # Deterministic gate (FR-C4)
    enable_dedup_gate: bool               # Step 2.2 semantic dedup

    # Validation LLMs
    enable_sis_validation: bool           # LLM #2 (Step 3.1)
    enable_trace_validation: bool         # LLM #3 (Step 4.2)
    enable_bfs: bool                      # Phase 5 deterministic (both arms)
    enable_propagation_validation: bool   # LLM #4 (Steps 5.3a / 5.3b)

    # Always-on
    run_llm_1: bool = True                # Interpret
    run_llm_5: bool = True                # Synthesize

    # Forced-inclusion bypasses LLM #5 synthesis and builds ImpactReport directly
    # from all CIS nodes. Retained for diagnostic use; not exercised by the
    # canonical ablation matrix.
    force_include_all_cis_nodes: bool = False

    # LLM #1 anchor priming. When True (default), LLM #1's system
    # prompt instructs the model to populate CRInterpretation.anchor_candidates
    # with 1-3 likely host/sibling symbols from the codebase even when the CR
    # does not explicitly name them. The retriever then applies a soft BM25
    # boost (Path 4, settings.anchor_priming_bm25_boost) to candidates whose
    # name matches an anchor candidate. When False,
    # LLM #1 emits an empty anchor_candidates list and no boost is applied.
    # All V0-V7 variants inherit this default; toggling it produces the
    # before-and-after ablation reported in the thesis methodology amendment.
    anchor_priming: bool = True

    # Project-grounded two-stage interpretation. When True
    # (default), LLM #1 is split into two calls:
    #   1a (intent): actionability + change_type + layers + domain_concepts
    #      (no anchor reasoning, no project context).
    #   1b (anchors): receives stage-1 output PLUS the cached project
    #      skeleton, emits search_queries, layered_search_queries,
    #      named_entry_points, anchor_candidates, out_of_scope_operations.
    # The runner glues both outputs into one CRInterpretation. When False,
    # the legacy single-stage interpreter is used.
    two_stage_interpret: bool = True

    ALL_VARIANTS: ClassVar[list[str]] = [
        "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"
    ]

    #: Run-mode variant subsets (for `evaluate --mode`).
    #: RETRIEVAL_ONLY runs the deterministic + LLM #2/#3 chain (V0-V5) and
    #: populates the shared VariantCache (interp, SIS, trace, rerank_gated, …).
    #: PROPAGATE_ONLY runs only the graph/propagation variants (V6, V7),
    #: resuming the V0-V5 prefix from a cache produced by a prior RETRIEVAL_ONLY
    #: run (`--cache-from`). Their union is ALL_VARIANTS, so a RETRIEVAL_ONLY
    #: run followed by a PROPAGATE_ONLY run reproduces the full V0-V7 matrix.
    RETRIEVAL_ONLY: ClassVar[list[str]] = ["V0", "V1", "V2", "V3", "V4", "V5"]
    PROPAGATE_ONLY: ClassVar[list[str]] = ["V6", "V7"]

    @classmethod
    def variants_for_mode(cls, mode: str) -> list[str]:
        """Return the variant id list for an evaluate run-mode.

        mode in {"full", "retrieval-only", "propagate-only"}.
        """
        return {
            "full": list(cls.ALL_VARIANTS),
            "retrieval-only": list(cls.RETRIEVAL_ONLY),
            "propagate-only": list(cls.PROPAGATE_ONLY),
        }[mode]

    @classmethod
    def v0(cls) -> "VariantFlags":
        return cls(
            variant_id="V0",
            enable_bm25=True, enable_dense=False, enable_rrf=False,
            enable_cross_encoder=False,
            enable_dedup_gate=True,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v1(cls) -> "VariantFlags":
        return cls(
            variant_id="V1",
            enable_bm25=False, enable_dense=True, enable_rrf=False,
            enable_cross_encoder=False,
            enable_dedup_gate=True,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v2(cls) -> "VariantFlags":
        return cls(
            variant_id="V2",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=False,
            enable_dedup_gate=True,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v3(cls) -> "VariantFlags":
        """V3: cross-encoder rerank + semantic dedup - the deterministic
        pre-LLM filtering peak. The V3->V4 delta isolates LLM #2."""
        return cls(
            variant_id="V3",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True,
            enable_sis_validation=False, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v4(cls) -> "VariantFlags":
        return cls(
            variant_id="V4",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True,
            enable_sis_validation=True, enable_trace_validation=False,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v5(cls) -> "VariantFlags":
        return cls(
            variant_id="V5",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=False, enable_propagation_validation=False,
        )

    @classmethod
    def v6(cls) -> "VariantFlags":
        return cls(
            variant_id="V6",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=True, enable_propagation_validation=False,
        )

    @classmethod
    def v7_full(cls) -> "VariantFlags":
        return cls(
            variant_id="V7",
            enable_bm25=True, enable_dense=True, enable_rrf=True,
            enable_cross_encoder=True,
            enable_dedup_gate=True,
            enable_sis_validation=True, enable_trace_validation=True,
            enable_bfs=True, enable_propagation_validation=True,
        )

    @classmethod
    def for_id(cls, variant_id: str) -> "VariantFlags":
        """Return the VariantFlags instance for a canonical V0..V7 id."""
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


def with_anchor_priming(flags: VariantFlags, value: bool) -> VariantFlags:
    """Return a copy of ``flags`` with ``anchor_priming`` set to ``value``.

    ``VariantFlags`` is frozen, so this wraps ``dataclasses.replace``. Used by
    the anchor-priming before-and-after ablation: ``with_anchor_priming(flags,
    False)`` reproduces the baseline without anchor extraction, ``True`` is
    the production default.
    """
    return replace(flags, anchor_priming=value)


def with_two_stage_interpret(flags: VariantFlags, value: bool) -> VariantFlags:
    """Return a copy of ``flags`` with ``two_stage_interpret`` set to ``value``.

    Used by the two-stage-vs-single-stage ablation:
    ``with_two_stage_interpret(flags, False)`` reproduces the single-stage
    LLM #1 behaviour, ``True`` is the production default.
    """
    return replace(flags, two_stage_interpret=value)
