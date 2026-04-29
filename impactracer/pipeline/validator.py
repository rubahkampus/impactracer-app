"""LLM Call #2: SIS validation (FR-C5).

Reference: 07_online_pipeline.md §7.
"""

from __future__ import annotations

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import Candidate, CRInterpretation, SISValidationResult


def mitigate_lost_in_middle(candidates: list[Candidate]) -> list[Candidate]:
    """Reorder candidates to mitigate lost-in-the-middle positional bias.

    Highest reranker score at position 0, lowest at position N-1,
    middle positions filled in ascending order of score.
    """
    raise NotImplementedError("Sprint 9")


def build_validator_prompt(
    cr_interp: CRInterpretation,
    ordered: list[Candidate],
) -> str:
    """Construct the user-message string for LLM Call #2.

    Each candidate entry includes node_id, node_type, file_path,
    reranker_score, and the snippet (preferring
    ``internal_logic_abstraction`` over ``text_snippet``).
    """
    raise NotImplementedError("Sprint 9")


def validate_sis_candidates(
    cr_interp: CRInterpretation,
    candidates: list[Candidate],
    client: LLMClient,
) -> list[str]:
    """Run LLM Call #2 and return the list of confirmed node IDs."""
    raise NotImplementedError("Sprint 9")
