"""LLM Call #3: Trace resolution validation (FR-C7).

Per (doc_chunk, code_node) pair produced by blind resolution, LLM #3
emits one of three decisions: CONFIRMED, PARTIAL, REJECTED.

Batching Mandate: max 5 pairs per LLM call. Fail-open per pair when
LLM omits a verdict.

Anti-Circular Mandate: NO retrieval scores in the prompt.

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
5. Use <<DOC_ID_START>>…<<DOC_ID_END>> and <<CODE_ID_START>>…<<CODE_ID_END>> \
delimiters when writing doc_chunk_id and code_node_id in your response. \
Copy them VERBATIM — do NOT paraphrase or truncate.

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
            f"Document Text:",
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
) -> tuple[list[str], dict[str, bool]]:
    """Run LLM Call #3 and return (validated_code_seeds, low_confidence_map).

    PARTIAL decisions mark their code seed with ``low_confidence_seed=True``.
    REJECTED pairs drop from SIS entirely. Fail-open: if LLM omits a verdict
    for a pair, that pair is admitted (fail-open per Sprint 9.1 mandate).

    Blueprint §4 Step 5b.
    """
    if not resolutions:
        return [], {}

    if cr_interp is None:
        # Construct a minimal placeholder so prompt building never crashes.
        from impactracer.shared.models import CRInterpretation as _CR
        cr_interp = _CR(
            is_actionable=True,
            primary_intent="",
            change_type="MODIFICATION",
            affected_layers=["code"],
            domain_concepts=["unknown"],
            search_queries=["unknown"],
        )

    # Flatten resolutions into a list of (doc_id, code_id) pairs for batching.
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

    # Map (doc_id, code_id) -> decision for aggregation.
    decision_map: dict[tuple[str, str], str] = {}

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
            logger.warning(
                "[traceability_validator] LLM call failed for batch {}-{}: {} — fail-open",
                batch_start, batch_start + len(batch), exc,
            )
            # Fail-open: admit all pairs in this batch as PARTIAL.
            for doc_id, code_id in batch:
                decision_map[(doc_id, code_id)] = "PARTIAL"
            continue

        # Build a lookup from returned verdicts.
        verdict_map: dict[tuple[str, str], str] = {}
        for v in result.verdicts:
            key = (v.doc_chunk_id, v.code_node_id)
            verdict_map[key] = v.decision

        # Per-pair coverage check: fail-open for any pair the LLM omitted.
        for doc_id, code_id in batch:
            key = (doc_id, code_id)
            if key in verdict_map:
                decision_map[key] = verdict_map[key]
            else:
                logger.warning(
                    "[traceability_validator] No verdict for ({}, {}) — fail-open as PARTIAL",
                    doc_id, code_id,
                )
                decision_map[key] = "PARTIAL"

    # Aggregate per code_id: if any pair for a code_id is CONFIRMED, it is
    # CONFIRMED. If only PARTIAL, it is PARTIAL. If all REJECTED, it is REJECTED.
    code_id_decisions: dict[str, list[str]] = {}
    for (doc_id, code_id), decision in decision_map.items():
        code_id_decisions.setdefault(code_id, []).append(decision)

    validated_code_seeds: list[str] = []
    low_confidence_map: dict[str, bool] = {}

    # Preserve resolution order for reproducibility.
    seen: set[str] = set()
    for r in resolutions:
        for code_id in r["code_ids"]:
            if code_id in seen:
                continue
            seen.add(code_id)
            decisions = code_id_decisions.get(code_id, ["PARTIAL"])
            if "CONFIRMED" in decisions:
                validated_code_seeds.append(code_id)
                low_confidence_map[code_id] = False
            elif "PARTIAL" in decisions:
                validated_code_seeds.append(code_id)
                low_confidence_map[code_id] = True
            # REJECTED-only: drop silently.
            else:
                logger.debug(
                    "[traceability_validator] code_id={} REJECTED by all pairs — dropped",
                    code_id,
                )

    logger.info(
        "[traceability_validator] Result: {} validated seeds ({} low-conf)",
        len(validated_code_seeds),
        sum(1 for v in low_confidence_map.values() if v),
    )
    return validated_code_seeds, low_confidence_map
