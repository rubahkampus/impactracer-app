"""LLM Call #1: Change Request interpretation (FR-B1, FR-B2).

Produces a :class:`CRInterpretation` with nine attributes. The
``is_actionable`` flag gates the downstream pipeline.

Reference: 07_online_pipeline.md §3.
"""

from __future__ import annotations

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import CRInterpretation


SYSTEM_PROMPT = """You are a software requirements analyst.

First, assess whether the Change Request (CR) is actionable. A CR is NOT
actionable when it is too vague (for example "improve performance"),
contains no identifiable change intent, or is less than one full sentence.
If not actionable, set is_actionable to false and provide a one-sentence
actionability_reason. If actionable, set is_actionable to true,
actionability_reason to null, and extract every remaining field.

For search_queries, produce 2 to 5 English technical phrases that would
match function signatures, class names, or API endpoints in code. The CR
may be in Indonesian; search_queries MUST be in English.

For domain_concepts, include both explicitly stated and implied business
concepts.

For named_entry_points, extract only specific function or component name
patterns that the CR explicitly describes. Do NOT infer.

For out_of_scope_operations, list business operations that share
vocabulary with the CR but are explicitly NOT being changed. Do NOT infer
beyond what the CR excludes.

Return valid JSON matching the schema exactly.
"""


def interpret_cr(cr_text: str, client: LLMClient) -> CRInterpretation:
    """Run LLM Call #1 and return the structured interpretation."""
    return client.call(
        system=SYSTEM_PROMPT,
        user=cr_text,
        response_schema=CRInterpretation,
        call_name="interpret",
    )
