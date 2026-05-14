"""LLM Call #1: Change Request interpretation (FR-B1, FR-B2).

Produces a :class:`CRInterpretation` with nine attributes. The
``is_actionable`` flag gates the downstream pipeline.

Reference: 07_online_pipeline.md §3.
"""

from __future__ import annotations

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import CRInterpretation


SYSTEM_PROMPT = """You are a software requirements analyst working on a
Next.js + TypeScript + MongoDB full-stack codebase.

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
they incidentally affect performance/security.

================================================================
LAYERED SEARCH QUERIES — CRITICAL (forensic finding: single-layer
queries on prior runs missed UI form components for backend-described
CRs; aggregate retrieval miss rate was 56%)
================================================================

Populate `layered_search_queries` as a JSON object with these EXACT keys:

  - "api_route": queries that match Next.js API route handlers
       (HTTP method functions in src/app/api/**/route.ts files).
       Use HTTP verbs and endpoint nouns: e.g. "PATCH commission listing handler",
       "POST update grace period endpoint".

  - "page_component": queries for top-level page components
       (src/app/**/page.tsx, src/components/**/Page.tsx). Use full feature
       names plus the word "page": e.g. "commission form page", "profile listing page".

  - "ui_component": queries for React components, forms, dialogs, cards,
       and section blocks (src/components/**). Use UI vocabulary:
       e.g. "deadline section form field", "listing item card", "template selector".
       INCLUDE form-section names when the CR adds or changes a form field —
       UI components are how illustrators/users actually expose the change.

  - "utility": queries for backend services, repositories, and helpers
       (src/lib/services/**, src/lib/db/repositories/**, src/lib/**). Use
       business-verb naming: e.g. "update listing service", "find active listings",
       "compute grace period deadline".

  - "type_definition": queries for Mongoose schemas, TypeScript interfaces,
       type aliases, payload/input types (src/lib/db/models/**, src/types/**,
       interfaces inside service files). Use the data shape name:
       e.g. "commission listing schema", "ICommissionListing interface",
       "CommissionListingUpdateInput type alias".

EACH key gets 1-2 English technical phrases. ALL FIVE keys are REQUIRED
and each list must have at least one phrase even when the CR seems to
target only one layer. Reasoning: a CR that describes a model change still
requires UI form updates to surface the new field, an API route to accept
it, and a service to persist it. Use your knowledge of full-stack Next.js
architecture to enumerate all five layers.

If you cannot in good faith generate a query for a layer, emit a single
phrase that uses the CR's primary domain concept plus the layer's archetype
(e.g. "<concept> page" for page_component). DO NOT skip a layer.

SCHEMA CONSTRAINTS (must be satisfied in every response):
- is_actionable: boolean, always required
- actionability_reason: string or null (null when actionable, one sentence when not)
- primary_intent: string, always required (use empty string "" when not actionable)
- change_type: one of "ADDITION", "MODIFICATION", "DELETION" (use "MODIFICATION" when not actionable)
- affected_layers: list containing any of "requirement", "design", "code" (use [] when not actionable)
- domain_concepts: list of 1 to 10 strings - MINIMUM 1 item always required even when not actionable
- search_queries: list of 2 to 5 English strings - MINIMUM 2 items always required even when not actionable
- layered_search_queries: object with 5 keys ("api_route", "page_component",
  "ui_component", "utility", "type_definition"), each mapped to a list of
  1-2 English phrases. ALL five keys MUST be present and non-empty when
  is_actionable is true. Use null when is_actionable is false.
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
