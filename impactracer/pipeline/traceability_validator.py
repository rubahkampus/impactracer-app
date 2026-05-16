"""LLM Call #3: Trace resolution validation (FR-C7).

Per (doc_chunk, code_node) pair produced by blind resolution, LLM #3
emits one of three decisions: CONFIRMED, PARTIAL, REJECTED.

Fail-closed: per-pair missing verdict → REJECTED; batch-level exception →
DROP entire batch, continue. Runner sets degraded_run=True on any drop.

Returns (validated_seeds, low_conf_map, justifications, degraded) so the
runner can attach LLM #3 reasoning to resolved code seed NodeTraces.

No retrieval scores in the prompt (anti-circular mandate).

Reference: master_blueprint.md §4 Step 5b.
"""

from __future__ import annotations

from loguru import logger

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import (
    CRInterpretation,
    TraceValidationResult,
)

_BATCH_SIZE = 5


def _strip_delimiters(s: str) -> str:
    """Remove leftover <<DOC_ID_*>>/<<CODE_ID_*>> markers from LLM output."""
    if s is None:
        return ""
    out = s
    for tok in (
        "<<DOC_ID_START>>", "<<DOC_ID_END>>",
        "<DOC_ID_START>", "<DOC_ID_END>",
        "<<CODE_ID_START>>", "<<CODE_ID_END>>",
        "<CODE_ID_START>", "<CODE_ID_END>",
        "<<NODE_ID_START>>", "<<NODE_ID_END>>",
    ):
        out = out.replace(tok, "")
    return out.strip().strip("<>").strip()

_SYSTEM_PROMPT = """\
You are a software traceability expert. Your task is to validate whether \
a code node genuinely implements or is directly relevant to a software \
documentation chunk (from an SRS or SDD).

For each (doc_chunk, code_node) pair you receive:
- CONFIRMED: The code node clearly implements, defines, or is directly \
required by the document section. The relationship is structural, not merely \
topical.
- PARTIAL: The code node is tangentially related or only partially implements \
the document section intent. Include it with a low-confidence marker.
- REJECTED: The code node has no meaningful implementation relationship to \
the document section. Vocabulary overlap alone is NOT sufficient.

CRITICAL RULES:
1. Judge based on AST structure and document semantics — NOT on any score.
2. A code node in a completely different feature area MUST be REJECTED even \
if its name sounds related.
3. For ADDITION changes: absence of current implementation does NOT mean \
REJECTED — it means the code node is where the feature SHOULD be added.
4. Return verdicts for EVERY pair in the batch using the exact node IDs given.
5. CRITICAL: Copy the doc_chunk_id and code_node_id exactly from BETWEEN \
the <<DOC_ID_START>>...<<DOC_ID_END>> and \
<<CODE_ID_START>>...<<CODE_ID_END>> delimiters. DO NOT include the \
<< >> delimiter markers themselves in your JSON output. Do NOT \
paraphrase or truncate the IDs.

JUSTIFICATION QUALITY:
The justification field MUST cite the SPECIFIC mechanism by which the code
node either implements or fails to implement the document section. Examples:
  GOOD: "uploadFile() reads the size limit constant defined in section 3.2.1
         which the CR raises from 5MB to 10MB."
  BAD:  "This function is related to file uploads."

OUTPUT FORMAT:
Return a JSON object matching TraceValidationResult with a "verdicts" list.
"""


def _build_trace_prompt(
    pairs: list[tuple[str, str]],
    doc_text_by_id: dict[str, str],
    code_meta_by_id: dict[str, dict],
    cr_interp: CRInterpretation,
) -> str:
    """Build the user prompt for one batch of (doc_id, code_id) pairs."""
    lines: list[str] = [
        f"Change Request Intent: {cr_interp.primary_intent}",
        f"Change Type: {cr_interp.change_type}",
        f"Domain Concepts: {', '.join(cr_interp.domain_concepts)}",
        "",
        "Validate each (document chunk, code node) pair below.",
        "",
    ]
    for i, (doc_id, code_id) in enumerate(pairs, start=1):
        doc_text = doc_text_by_id.get(doc_id, "")[:800]
        meta = code_meta_by_id.get(code_id, {})
        node_type = meta.get("node_type", "Unknown")
        file_path = meta.get("file_path", "")
        abstraction = meta.get("internal_logic_abstraction") or meta.get("source_code", "")
        if abstraction:
            abstraction = abstraction[:800]

        lines += [
            f"[{i}]",
            f"DOCUMENT CHUNK ID: <<DOC_ID_START>>{doc_id}<<DOC_ID_END>>",
            "Document Text:",
            doc_text,
            "",
            f"CODE NODE ID: <<CODE_ID_START>>{code_id}<<CODE_ID_END>>",
            f"Type: {node_type}",
            f"File: {file_path}",
        ]
        if abstraction:
            lines += [
                "Abstraction / Signature:",
                abstraction,
            ]
        lines.append("")

    lines += [
        "For each pair, decide: CONFIRMED, PARTIAL, or REJECTED.",
        "Return a JSON object: {\"verdicts\": [{\"doc_chunk_id\": ..., "
        "\"code_node_id\": ..., \"decision\": ..., \"justification\": ...}]}",
    ]
    return "\n".join(lines)


def validate_trace_resolutions(
    resolutions: list[dict],
    doc_text_by_id: dict[str, str],
    code_meta_by_id: dict[str, dict],
    client: LLMClient,
    cr_interp: CRInterpretation | None = None,
) -> tuple[list[str], dict[str, bool], dict[str, str], bool]:
    """Run LLM Call #3 and return (seeds, low_conf, justifications, degraded).

    Fail-closed: per-pair missing verdicts -> REJECTED. Batch exception ->
    all pairs in that batch REJECTED, continue with the next batch.

    Per-pair justifications are captured keyed by code_id. For code_ids
    resolved via multiple doc pairs, the highest-decision pair's justification
    wins (CONFIRMED > PARTIAL > REJECTED).

    Returns:
        validated_code_seeds: code_ids where any pair was CONFIRMED or PARTIAL.
        low_confidence_map:   code_id -> True if all pairs were PARTIAL.
        justifications:       code_id -> chosen justification text.
        degraded:             True if any batch was dropped due to API exhaustion.

    Blueprint §4 Step 5b.
    """
    if not resolutions:
        return [], {}, {}, False

    if cr_interp is None:
        from impactracer.shared.models import CRInterpretation as _CR
        cr_interp = _CR(
            is_actionable=True,
            primary_intent="",
            change_type="MODIFICATION",
            affected_layers=["code"],
            domain_concepts=["unknown"],
            search_queries=["unknown"],
        )

    all_pairs: list[tuple[str, str]] = []
    for r in resolutions:
        doc_id = r["doc_id"]
        for code_id in r["code_ids"]:
            all_pairs.append((doc_id, code_id))

    logger.info(
        "[traceability_validator] Validating {} pairs in batches of {}",
        len(all_pairs),
        _BATCH_SIZE,
    )

    decision_map: dict[tuple[str, str], str] = {}
    justification_map: dict[tuple[str, str], str] = {}
    degraded: bool = False

    for batch_start in range(0, len(all_pairs), _BATCH_SIZE):
        batch = all_pairs[batch_start : batch_start + _BATCH_SIZE]
        prompt = _build_trace_prompt(batch, doc_text_by_id, code_meta_by_id, cr_interp)

        try:
            result: TraceValidationResult = client.call(
                system=_SYSTEM_PROMPT,
                user=prompt,
                response_schema=TraceValidationResult,
                call_name="validate_trace",
            )
        except Exception as exc:
            # Fail-closed at batch level: drop the batch (all pairs REJECTED).
            logger.error(
                "[traceability_validator] Batch {}-{} failed after retries: {} - "
                "DROPPING batch (fail-closed)",
                batch_start, batch_start + len(batch), exc,
            )
            degraded = True
            for doc_id, code_id in batch:
                decision_map[(doc_id, code_id)] = "REJECTED"
            continue

        verdict_map: dict[tuple[str, str], tuple[str, str]] = {}
        for v in result.verdicts:
            doc_clean = _strip_delimiters(v.doc_chunk_id)
            code_clean = _strip_delimiters(v.code_node_id)
            verdict_map[(doc_clean, code_clean)] = (v.decision, v.justification or "")

        for doc_id, code_id in batch:
            key = (doc_id, code_id)
            if key in verdict_map:
                decision, justification = verdict_map[key]
                decision_map[key] = decision
                justification_map[key] = justification
            else:
                logger.warning(
                    "[traceability_validator] No verdict for ({}, {}) - "
                    "REJECTING (fail-closed per-pair)",
                    doc_id, code_id,
                )
                decision_map[key] = "REJECTED"

    # Aggregate per code_id.
    code_id_decisions: dict[str, list[tuple[str, str]]] = {}
    for (doc_id, code_id), decision in decision_map.items():
        just = justification_map.get((doc_id, code_id), "")
        code_id_decisions.setdefault(code_id, []).append((decision, just))

    validated_code_seeds: list[str] = []
    low_confidence_map: dict[str, bool] = {}
    justifications: dict[str, str] = {}

    seen: set[str] = set()
    for r in resolutions:
        for code_id in r["code_ids"]:
            if code_id in seen:
                continue
            seen.add(code_id)
            entries = code_id_decisions.get(code_id, [])
            decisions = [e[0] for e in entries]
            # Pick best (CONFIRMED > PARTIAL > REJECTED) and its justification.
            best: tuple[str, str] | None = None
            for entry in entries:
                d, j = entry
                if d == "CONFIRMED":
                    best = entry
                    break
                if d == "PARTIAL" and (best is None or best[0] != "CONFIRMED"):
                    best = entry
            if best is None:
                logger.debug(
                    "[traceability_validator] code_id={} REJECTED by all pairs - dropped",
                    code_id,
                )
                continue
            decision, justification = best
            validated_code_seeds.append(code_id)
            low_confidence_map[code_id] = (decision == "PARTIAL") or ("CONFIRMED" not in decisions)
            justifications[code_id] = justification

    logger.info(
        "[traceability_validator] Result: {} validated seeds ({} low-conf, degraded={})",
        len(validated_code_seeds),
        sum(1 for v in low_confidence_map.values() if v),
        degraded,
    )
    return validated_code_seeds, low_confidence_map, justifications, degraded
