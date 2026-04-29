"""LLM Call #4: Propagation validation (FR-D2).

For each node reached by BFS (except exempted-edge single-hop neighbors),
LLM #4 decides whether the structural path implies semantic impact.

Reference: 07_online_pipeline.md §11.
"""

from __future__ import annotations

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import CISResult, CRInterpretation


def validate_propagation(
    cis: CISResult,
    cr_interp: CRInterpretation,
    node_meta_by_id: dict[str, dict],
    client: LLMClient,
) -> CISResult:
    """Filter CIS by LLM #4 decision.

    Nodes reached at depth 1 via an edge in
    :data:`impactracer.shared.constants.PROPAGATION_VALIDATION_EXEMPT_EDGES`
    bypass the LLM call and are always kept.
    """
    raise NotImplementedError("Sprint 10")
