"""LLM Call #3: Trace resolution validation (FR-C7).

Per (doc_chunk, code_node) pair produced by blind resolution, LLM #3
emits one of three decisions: CONFIRMED, PARTIAL, REJECTED.

Reference: 07_online_pipeline.md §9.
"""

from __future__ import annotations

from impactracer.pipeline.llm_client import LLMClient


def validate_trace_resolutions(
    resolutions: list[dict],
    doc_text_by_id: dict[str, str],
    code_meta_by_id: dict[str, dict],
    client: LLMClient,
) -> tuple[list[str], dict[str, bool]]:
    """Run LLM Call #3 and return (validated_code_seeds, low_confidence_map).

    PARTIAL decisions mark their code seed with ``low_confidence_seed=True``.
    REJECTED pairs drop from SIS entirely.
    """
    raise NotImplementedError("Sprint 10")
