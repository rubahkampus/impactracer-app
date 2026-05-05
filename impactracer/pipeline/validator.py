"""LLM Call #2: SIS validation (FR-C5).

Batches candidates into chunks of at most 5, calls LLM #2 per chunk,
then merges all SISValidationResult envelopes.

The validator prompt MUST NOT include retrieval scores (reranker score,
ARRF weight, cosine distance). The LLM judges structural relevance only.

Blueprint: master_blueprint.md §4 Step 4.
"""

from __future__ import annotations

from loguru import logger

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import (
    Candidate,
    CRInterpretation,
    SISValidationResult,
)

_BATCH_SIZE = 5

_SYSTEM_PROMPT = """\
You are a software impact analysis expert.

Given a Change Request and a list of code/documentation candidates, decide
which candidates are DIRECTLY impacted by the change.

RULES FOR CODE NODE CANDIDATES:
- Confirm a code node ONLY if modifying it is a structural requirement for
  implementing the CR — not merely because it is topically related.
- A function is directly impacted if its body, signature, or direct callees
  must change to implement the CR.
- Pay close attention to File path. A function in an unrelated service module
  is almost never directly impacted by a CR targeting a different feature.
- Reject candidates that share vocabulary with the CR but serve a different
  feature area.

RULES FOR DOC CHUNK CANDIDATES (node_type="DocChunk"):
- Confirm a doc chunk if the CR explicitly changes the requirement or design
  section it describes, or if the chunk describes behaviour that will be
  contradicted or extended by the CR.
- The standard is: would a developer need to update this documentation section
  to keep it consistent with the implemented CR?
- Reject doc chunks that merely mention related concepts but are not
  affected by the CR.

RULES FOR ADDITION CRs:
- When change_type is ADDITION, look forward: confirm candidates that are
  entry points for the new feature, callers that will need to invoke the
  new code, type definitions that need new fields, and doc sections that
  describe the feature being added.
- Do NOT require that an existing function already contains the logic —
  for ADDITION CRs, absence of logic IS the reason to confirm it (it will
  need to be added).
- Phase 2.5 ADDITION SCOPING CONSTRAINT: a node is impacted ONLY if it
  DIRECTLY exposes, accepts, or processes the new feature — not merely
  because it belongs to the same service or shares a related domain concept.
  Generic utility functions, shared middleware, and unrelated service modules
  that happen to be in the same codebase are NOT impacted unless they must
  explicitly handle the new feature.

OUTPUT FORMAT:
Return exactly one JSON object with this structure:
{"verdicts": [ ... ]}

Each verdict must have:
  node_id           – the EXACT node_id string from the candidate (<<NODE_ID_START>>...<<NODE_ID_END>>), copied verbatim
  function_purpose  – one sentence: what this node does
  mechanism_of_impact – concrete modification required (empty string if rejected)
  justification     – one-sentence confirmation or rejection summary
  confirmed         – true only if mechanism_of_impact is non-empty

CRITICAL: node_id must be copied VERBATIM from the <<NODE_ID_START>>...<<NODE_ID_END>> delimiters.
Do NOT paraphrase, truncate, or alter the node_id.
"""


def chunk_candidates(
    candidates: list[Candidate],
    batch_size: int = _BATCH_SIZE,
) -> list[list[Candidate]]:
    """Split candidates into batches of at most batch_size.

    Blueprint §4 Step 4 (Batching Mandate: max 5 per LLM call).
    """
    return [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]


def build_validator_prompt(
    cr_interp: CRInterpretation,
    batch: list[Candidate],
) -> str:
    """Construct the user-message string for one LLM Call #2 batch.

    Includes: CR intent, change type, domain concepts, out-of-scope
    operations, named entry points, and per-candidate node_id / type /
    file path / snippet (prefers internal_logic_abstraction, full length).

    B1: merged_doc_contexts are injected as "Business Context" blocks
    immediately after the code snippet, so the LLM sees what requirement
    makes each code node relevant.

    B5: DocChunk candidates are rendered with a DOCUMENTATION SECTION header
    and the specific confirmation criterion for doc chunks.

    B6: node_id is wrapped in <<NODE_ID_START>>...<<NODE_ID_END>> delimiters
    so the LLM cannot confuse it with surrounding prose.

    ILA is already a compact skeletonized reduction — truncation would
    drop call sites and return paths the LLM needs to judge impact.
    Batches of ≤5 nodes fit well within the 100k-token context window.

    DELIBERATELY EXCLUDES all retrieval scores (reranker_score, rrf_score,
    cosine_score). The LLM must judge structural relevance independently.

    Blueprint §4 Step 4 Anti-Circular Logic Mandate.
    """
    lines: list[str] = []
    lines.append(f"Change Request Intent: {cr_interp.primary_intent}")
    lines.append(f"Change Type: {cr_interp.change_type}")
    lines.append(f"Domain Concepts: {', '.join(cr_interp.domain_concepts)}")

    if cr_interp.out_of_scope_operations:
        lines.append("")
        lines.append(
            "OUT-OF-SCOPE OPERATIONS — these share vocabulary with the CR "
            "but are NOT being changed. Do NOT confirm any candidate that "
            "primarily serves one of these:"
        )
        for op in cr_interp.out_of_scope_operations:
            lines.append(f"  - {op}")

    if cr_interp.named_entry_points:
        lines.append("")
        lines.append(
            "NAMED ENTRY POINTS — the CR explicitly describes these:"
        )
        for pat in cr_interp.named_entry_points:
            lines.append(f"  - {pat}")

    lines.append("")
    lines.append(
        "Evaluate each candidate. Confirm ONLY if directly relevant. "
        "Reject topically related but functionally unaffected candidates."
    )
    lines.append("")

    for idx, c in enumerate(batch, start=1):
        is_doc = c.collection == "doc_chunks"

        if is_doc:
            lines.append(f"[{idx}] DOCUMENTATION SECTION")
            # B6: node_id wrapped in unambiguous delimiters
            lines.append(f"ID: <<NODE_ID_START>>{c.node_id}<<NODE_ID_END>>")
            lines.append(f"Type: {c.node_type}")
            snippet = c.text_snippet or ""
            lines.append(f"Content:\n{snippet}")
            lines.append(
                "Confirmation criterion: confirm if this documentation section "
                "describes behaviour that the CR changes or extends."
            )
        else:
            lines.append(f"[{idx}] CODE NODE")
            # B6: node_id wrapped in unambiguous delimiters
            lines.append(f"ID: <<NODE_ID_START>>{c.node_id}<<NODE_ID_END>>")
            lines.append(f"Type: {c.node_type}")
            lines.append(f"File: {c.file_path}")
            snippet = c.internal_logic_abstraction or c.text_snippet or ""
            lines.append(f"Snippet:\n{snippet}")

            # B1: inject merged doc contexts as "Business Context" blocks
            if c.merged_doc_contexts:
                lines.append("")
                lines.append("Business Context (requirement sections that reference this code node):")
                for section_title, section_text in c.merged_doc_contexts:
                    lines.append(f"  [Section: {section_title}]")
                    # Truncate long doc sections to avoid prompt bloat from very
                    # large doc chunks — 500 chars gives ample context.
                    preview = section_text[:500] if len(section_text) > 500 else section_text
                    lines.append(f"  {preview}")

        lines.append("")

    return "\n".join(lines)


def validate_sis_candidates_batched(
    cr_interp: CRInterpretation,
    candidates: list[Candidate],
    client: LLMClient,
) -> list[str]:
    """Run LLM Call #2 in batches and return confirmed node IDs.

    Splits candidates into batches of at most 5, calls LLM #2 per batch,
    merges SISValidationResult envelopes, and returns the list of node_ids
    whose confirmed=True.

    B2 fix: partial verdict coverage is handled per-node, not per-batch.
    Previously, the fail-open guard fired only when ``len(result.verdicts) == 0``
    (completely empty response).  If the LLM returned verdicts for 3 of 5
    candidates, the remaining 2 were silently dropped — a false negative trap.
    The new logic checks per-candidate: any node missing from the verdict
    response is admitted (fail-open) so we never silently discard candidates.

    Blueprint §4 Step 4 (Batching Mandate).
    """
    if not candidates:
        return []

    batches = chunk_candidates(candidates, _BATCH_SIZE)
    logger.info(
        "[validator] LLM #2: {} candidates -> {} batches of ≤{}",
        len(candidates), len(batches), _BATCH_SIZE,
    )

    confirmed_ids: list[str] = []

    for batch_idx, batch in enumerate(batches, start=1):
        prompt = build_validator_prompt(cr_interp, batch)
        logger.info(
            "[validator] Batch {}/{}: {} candidates",
            batch_idx, len(batches), len(batch),
        )

        result: SISValidationResult = client.call(
            system=_SYSTEM_PROMPT,
            user=prompt,
            response_schema=SISValidationResult,
            call_name="validate_sis",
        )

        batch_ids = {c.node_id for c in batch}

        # B2: build a lookup of verdicts keyed by node_id for O(1) coverage check.
        # Only accept verdicts whose node_id belongs to THIS batch (reject hallucinations).
        verdict_map: dict[str, bool] = {}
        for verdict in result.verdicts:
            if verdict.node_id in batch_ids:
                verdict_map[verdict.node_id] = verdict.confirmed
                if verdict.confirmed:
                    logger.debug(
                        "[validator] CONFIRMED: {} — {}",
                        verdict.node_id, verdict.mechanism_of_impact,
                    )
                else:
                    logger.debug(
                        "[validator] REJECTED: {} — {}",
                        verdict.node_id, verdict.justification,
                    )
            else:
                # Hallucinated node_id not in this batch — silently discard
                logger.debug(
                    "[validator] IGNORED hallucinated verdict for {} (not in batch)",
                    verdict.node_id,
                )

        # B2: per-node fail-open: any node with no verdict is admitted.
        for c in batch:
            if c.node_id not in verdict_map:
                # No verdict returned for this node — fail-open: admit it
                logger.warning(
                    "[validator] No verdict for {} — admitting (fail-open per-node)",
                    c.node_id,
                )
                confirmed_ids.append(c.node_id)
            elif verdict_map[c.node_id]:
                confirmed_ids.append(c.node_id)

    logger.info(
        "[validator] LLM #2 complete: {}/{} candidates confirmed",
        len(confirmed_ids), len(candidates),
    )
    return confirmed_ids
