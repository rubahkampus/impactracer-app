"""Dual-path hybrid search with Adaptive RRF (FR-C1, FR-C2).

Four ranked lists per query: dense-doc, bm25-doc, dense-code, bm25-code.
Fused via Adaptive RRF where path weights depend on ``change_type``.

Reference: 07_online_pipeline.md §4.
"""

from __future__ import annotations

from impactracer.shared.models import Candidate, CRInterpretation


def hybrid_search(
    cr_interp: CRInterpretation,
    ctx: object,               # PipelineContext, forward-declared
    settings: object,          # Settings, forward-declared
) -> list[Candidate]:
    """Execute dual-path search and return RRF-sorted candidates.

    Honors ``cr_interp.affected_layers`` as a ChromaDB metadata filter
    on the doc collection and gates code-collection search.
    """
    raise NotImplementedError("Sprint 8")


def reciprocal_rank_fusion_adaptive(
    ranked_lists: list[tuple[str, list[str]]],
    change_type: str,
    k: int = 60,
) -> dict[str, float]:
    """Weighted RRF fusion.

    Each input entry is ``(path_label, ranked_ids)``; weights come from
    :data:`impactracer.shared.constants.RRF_PATH_WEIGHTS`.
    """
    raise NotImplementedError("Sprint 8")


def build_bm25_from_chroma(collection: object) -> tuple[object, list[str]]:
    """Construct a BM25Okapi index from all documents in a ChromaDB collection."""
    raise NotImplementedError("Sprint 8")
