"""LLM Call #2: SIS validation (FR-C5).

Batches candidates into chunks of at most 5, calls LLM #2 per chunk,
then merges all SISValidationResult envelopes.

Fail-closed: per-node missing verdict → DROP; batch-level exception → DROP
entire batch and continue. Runner sets degraded_run=True on any drop.

Returns (confirmed_ids, justifications_map, degraded) so the runner can
attach function_purpose, mechanism_of_impact, and justification to NodeTrace.

No retrieval scores in the prompt — the LLM judges structural relevance only.

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
- ADDITION SCOPING: a node is impacted ONLY if it DIRECTLY exposes,
  accepts, or processes the new feature — not merely because it belongs
  to the same service or shares a related domain concept. Generic utility
  functions, shared middleware, and unrelated service modules that happen
  to be in the same codebase are NOT impacted unless they must explicitly
  handle the new feature.

JUSTIFICATION QUALITY:
- mechanism_of_impact MUST describe a CONCRETE structural modification
  (e.g. "add `pin: boolean` field to the schema and propagate to the
  serializer", "change the rate-limit constant from 5/min to 10/min and
  update the test expectation"). Vague justifications such as "this
  function is related" or "this is a primary target" are forbidden.
- justification (one sentence) MUST cite the specific aspect of the CR
  that drives the impact, not the candidate's general purpose.

OUTPUT FORMAT:
Return exactly one JSON object with this structure:
{"verdicts": [ ... ]}

Each verdict must have:
  node_id           - the EXACT node_id string from the candidate (<<NODE_ID_START>>...<<NODE_ID_END>>), copied verbatim
  function_purpose  - one sentence: what this node does
  mechanism_of_impact - concrete modification required (empty string if rejected)
  justification     - one-sentence confirmation or rejection summary
  confirmed         - true only if mechanism_of_impact is non-empty

CRITICAL: Copy the node_id exactly from BETWEEN the delimiters
<<NODE_ID_START>>...<<NODE_ID_END>>. DO NOT include the << >> delimiter
markers themselves in your JSON output. Do NOT paraphrase, truncate, or
alter the node_id contents.
"""


def _strip_delimiters(s: str) -> str:
    """Remove leftover <<NODE_ID_START>>/<<NODE_ID_END>> markers some
    models (notably gemini-2.5-flash-lite) emit despite the prompt
    instruction. Also trims whitespace and surrounding angle brackets.
    """
    if s is None:
        return ""
    out = s
    for tok in (
        "<<NODE_ID_START>>", "<<NODE_ID_END>>",
        "<NODE_ID_START>", "<NODE_ID_END>",
    ):
        out = out.replace(tok, "")
    return out.strip().strip("<>").strip()


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
            lines.append(f"ID: <<NODE_ID_START>>{c.node_id}<<NODE_ID_END>>")
            lines.append(f"Type: {c.node_type}")
            lines.append(f"File: {c.file_path}")
            snippet = c.internal_logic_abstraction or c.text_snippet or ""
            lines.append(f"Snippet:\n{snippet}")

            if c.merged_doc_contexts:
                lines.append("")
                lines.append("Business Context (requirement sections that reference this code node):")
                for section_title, section_text in c.merged_doc_contexts:
                    lines.append(f"  [Section: {section_title}]")
                    preview = section_text[:500] if len(section_text) > 500 else section_text
                    lines.append(f"  {preview}")

        lines.append("")

    return "\n".join(lines)


def validate_sis_candidates_batched(
    cr_interp: CRInterpretation,
    candidates: list[Candidate],
    client: LLMClient,
) -> tuple[list[str], dict[str, dict[str, str]], bool]:
    """Run LLM Call #2 in batches and return (confirmed_ids, justifications, degraded).

    Fail-closed: missing per-node verdicts -> drop. Batch exception -> drop
    entire batch and continue. Returns degraded=True if any batch was dropped.

    Captures function_purpose, mechanism_of_impact, and justification per
    confirmed seed for downstream attachment to NodeTrace.

    Returns:
        confirmed_ids: list of node_ids the LLM marked confirmed=True.
        justifications: dict node_id -> {function_purpose, mechanism, justification}.
        degraded: True if any batch was dropped due to API exhaustion.

    Blueprint §4 Step 4 (Batching Mandate).
    """
    if not candidates:
        return [], {}, False

    batches = chunk_candidates(candidates, _BATCH_SIZE)
    logger.info(
        "[validator] LLM #2: {} candidates -> {} batches of <={}",
        len(candidates), len(batches), _BATCH_SIZE,
    )

    confirmed_ids: list[str] = []
    justifications: dict[str, dict[str, str]] = {}
    degraded: bool = False

    for batch_idx, batch in enumerate(batches, start=1):
        prompt = build_validator_prompt(cr_interp, batch)
        logger.info(
            "[validator] Batch {}/{}: {} candidates",
            batch_idx, len(batches), len(batch),
        )

        try:
            result: SISValidationResult = client.call(
                system=_SYSTEM_PROMPT,
                user=prompt,
                response_schema=SISValidationResult,
                call_name="validate_sis",
            )
        except Exception as exc:
            # Fail-closed: drop the batch but continue remaining batches.
            logger.error(
                "[validator] Batch {}/{} failed after retries: {} — "
                "DROPPING batch (fail-closed)",
                batch_idx, len(batches), exc,
            )
            degraded = True
            continue

        batch_ids = {c.node_id for c in batch}

        verdict_map: dict[str, "SISValidationResult.verdicts.__class__"] = {}
        for verdict in result.verdicts:
            clean_id = _strip_delimiters(verdict.node_id)
            if clean_id in batch_ids:
                verdict_map[clean_id] = verdict
            else:
                logger.debug(
                    "[validator] IGNORED hallucinated verdict for {!r} (cleaned: {!r}, not in batch)",
                    verdict.node_id, clean_id,
                )

        for c in batch:
            verdict = verdict_map.get(c.node_id)
            if verdict is None:
                logger.warning(
                    "[validator] No verdict for {} — DROPPING (fail-closed per-node)",
                    c.node_id,
                )
                continue
            if verdict.confirmed:
                confirmed_ids.append(c.node_id)
                justifications[c.node_id] = {
                    "function_purpose": verdict.function_purpose or "",
                    "mechanism_of_impact": verdict.mechanism_of_impact or "",
                    "justification": verdict.justification or "",
                }
                logger.debug(
                    "[validator] CONFIRMED: {} - {}",
                    c.node_id, verdict.mechanism_of_impact,
                )
            else:
                logger.debug(
                    "[validator] REJECTED: {} - {}",
                    c.node_id, verdict.justification,
                )

    logger.info(
        "[validator] LLM #2 complete: {}/{} candidates confirmed (degraded={})",
        len(confirmed_ids), len(candidates), degraded,
    )
    return confirmed_ids, justifications, degraded
