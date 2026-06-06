"""LLM Call #1: Change Request interpretation (FR-B1, FR-B2).

Two execution modes share one return shape (:class:`CRInterpretation`):

  Single-stage (legacy, pre-Amendment-3): one LLM call emits the full
  schema. Controlled by ``variant_flags.anchor_priming`` for whether
  ``anchor_candidates`` is populated.

  Two-stage (default-on via ``variant_flags.two_stage_
  interpret``): one cheap LLM call extracts intent (:class:`CRIntent`);
  a second call receives the stage-1 output plus the cached project
  skeleton and extracts anchors + search queries (:class:`CRAnchors`).
  The runner glues both outputs into one :class:`CRInterpretation`.

Both modes return the same shape so all downstream code is mode-
agnostic. The two-stage mode requires a non-empty ``project_skeleton``
argument; the runner passes ``None`` when no skeleton was found on
disk, in which case this module degrades gracefully to single-stage.

Reference: analysis_implementation.md §2 (LLM #1 — Interpret),
master_blueprint.md §4 Phase 1 (and §3.9 project skeleton).
"""

from __future__ import annotations

from impactracer.evaluation.variant_flags import VariantFlags
from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import CRAnchors, CRInterpretation, CRIntent

_PROMPT_BODY = """You are a software requirements analyst working on a
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

{ANCHOR_CANDIDATES_BLOCK}

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
- anchor_candidates: {ANCHOR_CANDIDATES_SCHEMA_LINE}
- out_of_scope_operations: list of 0 to 4 strings, never null (use [] if none)
- is_nfr: boolean, default false

CRITICAL: Never use null for list fields. Use [] for empty lists. Use "" for empty strings.
Return valid JSON matching the schema exactly.
"""


_ANCHOR_CANDIDATES_ON_BLOCK = """For anchor_candidates, the CR may PROPOSE a new function,
component, or page using imperative verbs ("Tambah", "Refactor",
"Implementasi", "Add", "Replace"). When that happens, list 1 to 3 bare
identifier names of likely HOST or SIBLING symbols in the existing
codebase that the new code will live next to, call, or modify. These are
HYPOTHESES about the codebase the CR does not explicitly name; retrieval
treats them as a soft additive boost, not a hard pin. Emit them even when
the CR does not name them.

Heuristics for picking anchor candidates:
  - When the CR adds a function, name a likely sibling function in the
    same likely host file (e.g. for "Tambah deleteReview" you might
    propose ["createClientReview", "updateReview"]).
  - When the CR adds a UI section, name a likely parent or peer
    component (e.g. for "Tambah price breakdown row" propose
    ["PriceBreakdownSection", "ProposalForm"]).
  - When the CR modifies a domain field, name the likely interface,
    schema, or service function holding that field.

If the CR is a pure modification or deletion of named symbols (no new
code proposed), emit an empty list.
"""

_ANCHOR_CANDIDATES_OFF_BLOCK = """For anchor_candidates, always emit an empty list []. This field is
populated only when the anchor-priming methodology amendment is enabled.
"""

_ANCHOR_CANDIDATES_ON_SCHEMA_LINE = (
    "list of 0 to 10 bare identifier names of likely host or sibling "
    "symbols. Emit 1 to 3 candidates when the CR proposes new code, "
    "else []. Never null."
)

_ANCHOR_CANDIDATES_OFF_SCHEMA_LINE = (
    "list of exactly 0 strings; always emit []. Never null."
)


def _build_system_prompt(anchor_priming: bool) -> str:
    """Assemble the LLM #1 single-stage system prompt."""
    if anchor_priming:
        block = _ANCHOR_CANDIDATES_ON_BLOCK
        schema_line = _ANCHOR_CANDIDATES_ON_SCHEMA_LINE
    else:
        block = _ANCHOR_CANDIDATES_OFF_BLOCK
        schema_line = _ANCHOR_CANDIDATES_OFF_SCHEMA_LINE
    return _PROMPT_BODY.format(
        ANCHOR_CANDIDATES_BLOCK=block,
        ANCHOR_CANDIDATES_SCHEMA_LINE=schema_line,
    )


# Pre-built default (anchor_priming=True) for callers that have no flags.
SYSTEM_PROMPT = _build_system_prompt(anchor_priming=True)


def interpret_cr_single_stage(
    cr_text: str,
    client: LLMClient,
    variant_flags: VariantFlags | None = None,
) -> CRInterpretation:
    """Single-stage LLM #1 (pre-Amendment-3 behaviour).

    One LLM call emits the full :class:`CRInterpretation`. The system
    prompt is assembled based on ``variant_flags.anchor_priming``.
    When ``variant_flags`` is None the default (anchor_priming=True) is
    used.
    """
    anchor_priming = True if variant_flags is None else variant_flags.anchor_priming
    system = _build_system_prompt(anchor_priming=anchor_priming)
    return client.call(
        system=system,
        user=cr_text,
        response_schema=CRInterpretation,
        call_name="interpret",
    )


# ----------------------------------------------------------------------
# Two-stage interpretation (intent + project-grounded anchors)
# ----------------------------------------------------------------------


_INTENT_PROMPT = """You are a software requirements analyst.

Read the Change Request (CR) text and extract its INTENT only. Do NOT
guess at codebase-specific symbols; a later stage will handle that with
project context. Focus on what is being changed, why, and at which
architectural layers.

A CR is NOT actionable when it is too vague (for example "improve
performance"), contains no identifiable change intent, or is less than
one full sentence. If not actionable, set is_actionable to false and
provide a one-sentence actionability_reason. If actionable, set
is_actionable to true and actionability_reason to null.

For affected_layers, include any of "requirement", "design", "code"
that the CR touches. Most code-level CRs include "code".

For domain_concepts, include both explicitly stated and implied
business concepts (e.g. "wallet", "escrow", "review", "commission").

For is_nfr, set true only when the CR's PRIMARY concern is a
non-functional requirement (performance, security, scalability,
accessibility, reliability). Default false for ordinary feature changes.

SCHEMA CONSTRAINTS:
- is_actionable: boolean
- actionability_reason: string or null
- primary_intent: string (use "" when not actionable)
- change_type: one of "ADDITION", "MODIFICATION", "DELETION".
  REQUIRED even when is_actionable=false; default to "MODIFICATION" when
  no clear change action can be extracted. Never emit an empty string.
- affected_layers: list of "requirement"/"design"/"code"
- domain_concepts: list of 1-30 strings, minimum 1 even when not actionable
- is_nfr: boolean

Return valid JSON matching the schema exactly. Never use null for list
fields; use [] for empty lists. Use "" for empty strings.
"""


_ANCHORS_PROMPT = """You are a software requirements analyst working on
a Next.js + TypeScript + MongoDB full-stack codebase. You have already
extracted the CR's intent. Now extract retrieval anchors using the
CACHED PROJECT SKELETON below as your source of truth for naming
conventions, file layout, and domain vocabulary.

================================================================
PROJECT SKELETON (snapshot of the indexed codebase)
================================================================
{PROJECT_SKELETON}
================================================================

Your job is to populate the search-side fields the retriever uses.
Ground every guess in the project skeleton above; do not invent
identifier shapes that the skeleton does not exemplify. In particular:

  - When the project uses camelCase function names (e.g. getTransactions,
    findContractById), your anchor_candidates MUST also be camelCase.
    Do NOT emit PascalCase classes (e.g. "WalletService") if the host
    file is a *.service.ts module that exports functions.

  - When the project uses PascalCase for components and interfaces,
    use PascalCase there.

  - When the CR proposes a new function/component/page, list the most
    likely SIBLING or HOST symbols in anchor_candidates by reading the
    domain vocabulary section above and applying the naming conventions.

  - When the CR clearly names an existing symbol, put it in
    named_entry_points.

For search_queries, produce 2 to 5 English technical phrases that match
function signatures, class names, or API endpoints as observed in the
project skeleton.

For layered_search_queries, populate ALL FIVE keys ("api_route",
"page_component", "ui_component", "utility", "type_definition") with
1-2 English phrases each that target THAT layer's vocabulary. Use the
path patterns shown in the skeleton to phrase each layer's query.

For named_entry_points, extract only symbols the CR explicitly names.
Do NOT infer beyond explicit mentions.

For anchor_candidates, emit 1-3 likely host or sibling symbols when the
CR proposes new code. Use the project skeleton's naming conventions.
Emit an empty list when the CR explicitly names every relevant symbol
or proposes no new code.

For out_of_scope_operations, list business operations that share
vocabulary with the CR but are explicitly NOT being changed.

SCHEMA CONSTRAINTS:
- search_queries: 2-5 English strings, minimum 2
- layered_search_queries: object with EXACTLY 5 keys, each value a list
  of 1-2 English strings; null only when is_actionable was false
- named_entry_points: 0-4 strings (extend up to 30 only if many are
  explicitly named); never null, use []
- anchor_candidates: 0-10 bare identifier names; never null, use []
- out_of_scope_operations: 0-4 strings; never null, use []

Return valid JSON matching the schema exactly.
"""


def _build_anchors_user_message(
    cr_text: str, intent: CRIntent, project_skeleton: str
) -> str:
    """Compose the user message for stage 1b.

    Includes the CR text and stage-1a's structured intent so the LLM
    has both the raw natural-language input and the disambiguated
    structural read.
    """
    return (
        "CR TEXT:\n"
        f"{cr_text}\n"
        "\n"
        "INTENT (stage 1a output):\n"
        f"  primary_intent: {intent.primary_intent}\n"
        f"  change_type: {intent.change_type}\n"
        f"  affected_layers: {intent.affected_layers}\n"
        f"  domain_concepts: {intent.domain_concepts}\n"
        f"  is_nfr: {intent.is_nfr}\n"
    )


def interpret_cr_two_stage(
    cr_text: str,
    client: LLMClient,
    project_skeleton: str,
) -> CRInterpretation:
    """Two-stage interpretation grounded in the cached project skeleton.

    Calls LLM #1a (intent) and LLM #1b (anchors), then glues both
    outputs into one :class:`CRInterpretation`. If stage 1a reports the
    CR as not actionable, stage 1b is skipped and a minimal
    :class:`CRInterpretation` is returned.
    """
    intent = client.call(
        system=_INTENT_PROMPT,
        user=cr_text,
        response_schema=CRIntent,
        call_name="interpret_intent",
    )

    if not intent.is_actionable:
        # Skip stage 1b; build a minimal CRInterpretation that satisfies
        # the schema's required fields. Downstream guards on
        # is_actionable=False short-circuit before the retrieval fields
        # are used. CRInterpretation requires domain_concepts to have
        # at least one entry; synthesise a placeholder when stage 1a
        # left it empty (legitimately, since the CR has no domain).
        domain_concepts = list(intent.domain_concepts) or ["not_actionable"]
        return CRInterpretation(
            is_actionable=False,
            actionability_reason=intent.actionability_reason,
            primary_intent=intent.primary_intent or "",
            change_type=intent.change_type,
            affected_layers=intent.affected_layers,
            domain_concepts=domain_concepts,
            search_queries=["not actionable", "no retrieval"],
            named_entry_points=[],
            anchor_candidates=[],
            out_of_scope_operations=[],
            layered_search_queries=None,
            is_nfr=intent.is_nfr,
        )

    anchors_system = _ANCHORS_PROMPT.format(PROJECT_SKELETON=project_skeleton)
    anchors = client.call(
        system=anchors_system,
        user=_build_anchors_user_message(cr_text, intent, project_skeleton),
        response_schema=CRAnchors,
        call_name="interpret_anchors",
    )

    # Edge case: stage 1a may emit empty domain_concepts even on
    # actionable CRs. CRInterpretation requires >= 1; fall back to the
    # CR's primary_intent split as a last-resort domain hint.
    domain_concepts = list(intent.domain_concepts)
    if not domain_concepts:
        domain_concepts = [intent.primary_intent[:40] or "unknown_domain"]
    affected_layers = list(intent.affected_layers) or ["code"]
    return CRInterpretation(
        is_actionable=True,
        actionability_reason=None,
        primary_intent=intent.primary_intent,
        change_type=intent.change_type,
        affected_layers=affected_layers,
        domain_concepts=domain_concepts,
        search_queries=anchors.search_queries,
        named_entry_points=anchors.named_entry_points,
        anchor_candidates=anchors.anchor_candidates,
        out_of_scope_operations=anchors.out_of_scope_operations,
        layered_search_queries=anchors.layered_search_queries,
        is_nfr=intent.is_nfr,
    )


def interpret_cr(
    cr_text: str,
    client: LLMClient,
    variant_flags: VariantFlags | None = None,
    project_skeleton: str | None = None,
) -> CRInterpretation:
    """Public dispatcher for LLM Call #1.

    Routes to ``interpret_cr_two_stage`` when both
    ``variant_flags.two_stage_interpret`` is True AND ``project_skeleton``
    is non-empty. Falls back to ``interpret_cr_single_stage`` otherwise.

    The fallback handles two cases:
      1. The caller explicitly set ``two_stage_interpret=False`` for the
         before-and-after Amendment-3 ablation.
      2. The project skeleton file is missing (e.g. pre-Amendment-3
         indexes). The pipeline must still work, just without grounded
         anchor extraction.
    """
    two_stage = True if variant_flags is None else variant_flags.two_stage_interpret
    if two_stage and project_skeleton:
        return interpret_cr_two_stage(cr_text, client, project_skeleton)
    return interpret_cr_single_stage(cr_text, client, variant_flags)
