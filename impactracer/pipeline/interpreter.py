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

For is_nfr, set true when the CR's PRIMARY concern is a non-functional
requirement: performance ("reduce latency", "speed up", "throughput"),
security ("authentication", "authorization", "vulnerability", "encryption"),
scalability ("horizontal scaling", "load handling"), accessibility ("a11y",
"screen reader", "WCAG"), reliability ("retry", "circuit breaker"). Set
false for ordinary feature additions, modifications, or deletions even when
they incidentally affect performance/security. (Crucible Fix 16.)

SCHEMA CONSTRAINTS (must be satisfied in every response):
- is_actionable: boolean, always required
- actionability_reason: string or null (null when actionable, one sentence when not)
- primary_intent: string, always required (use empty string "" when not actionable)
- change_type: one of "ADDITION", "MODIFICATION", "DELETION" (use "MODIFICATION" when not actionable)
- affected_layers: list containing any of "requirement", "design", "code" (use [] when not actionable)
- domain_concepts: list of 1 to 10 strings - MINIMUM 1 item always required even when not actionable
  (reason: zero domain concepts causes downstream retrieval to produce zero candidates, a pipeline failure)
- search_queries: list of 2 to 5 English strings - MINIMUM 2 items always required even when not actionable
  (reason: zero search queries causes downstream retrieval to produce zero candidates, a pipeline failure)
- named_entry_points: list of 0 to 4 strings, never null (use [] if none)
- out_of_scope_operations: list of 0 to 4 strings, never null (use [] if none)
- is_nfr: boolean, default false

CRITICAL: Never use null for list fields. Use [] for empty lists. Use "" for empty strings.
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
