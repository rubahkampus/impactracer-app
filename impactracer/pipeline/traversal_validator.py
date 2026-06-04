"""LLM Call #4: Propagation validation (FR-D2).

For each node reached by BFS (except exempted-edge single-hop neighbors),
LLM #4 decides whether the structural path implies semantic impact.

Fail-closed: per-node missing verdict → DROP; batch-level exception → DROP
entire batch, continue. Same policy applies to collapsed-child batches.

The causal chain is shown to LLM #4 as factual context only. The prompt
explicitly forbids bare-topology justifications — the model must identify a
concrete contract breakage or behavioral anomaly, not just confirm edge presence.

Returns (filtered_cis, justifications_map, degraded). The runner attaches
justifications verbatim to propagated NodeTraces (distributed justification
principle — LLM #5 never re-justifies individual nodes).

No retrieval scores in the prompt (anti-circular mandate).

Reference: master_blueprint.md §4 Step 7.
"""

from __future__ import annotations

import random

from loguru import logger

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.constants import PROPAGATION_VALIDATION_EXEMPT_EDGES
from impactracer.shared.models import (
    CISResult,
    CRInterpretation,
    NodeTrace,
    PropagationValidationResult,
)

_BATCH_SIZE = 5


def _strip_delimiters(s: str) -> str:
    """Remove leftover <<NODE_ID_*>> markers from LLM output."""
    if s is None:
        return ""
    out = s
    for tok in (
        "<<NODE_ID_START>>", "<<NODE_ID_END>>",
        "<NODE_ID_START>", "<NODE_ID_END>",
    ):
        out = out.replace(tok, "")
    return out.strip().strip("<>").strip()

_SYSTEM_PROMPT = """\
You are a software impact analysis expert. Your task is to determine whether \
a code node is SEMANTICALLY IMPACTED by a given Change Request.

A node is semantically impacted if modifying the CR's target would require \
changes to that node, or if the node's behaviour would change as a side-effect.

YOU WILL SEE the structural causal chain (the sequence of edge types that \
connects the SIS seed to this node). The chain is provided as FACTUAL CONTEXT \
ONLY — it tells you HOW the BFS reached the node, not WHETHER the node is \
impacted.

CRITICAL ANTI-TAUTOLOGY RULE (default — for MODIFICATION and ADDITION CRs):
- Edge types are NOT impact evidence. A chain containing IMPLEMENTS, CALLS, \
or any other edge does NOT by itself confirm impact. Many chains terminate \
at nodes whose behaviour is unaffected by the CR despite a structurally \
present relationship.
- Reject any node where the chain merely describes a generic dependency that \
the CR does not actually disturb (e.g. a function CALLS a utility that the CR \
does not modify; a class IMPLEMENTS an interface whose contract is unchanged).

DELETION EXCEPTION (applies ONLY when Change Type == DELETION):
- When the CR REMOVES a symbol, the structural relationships ARE evidence. \
A direct CALLS/TYPED_BY/IMPLEMENTS/FIELDS_ACCESSED edge from the candidate \
to the deletion target means the candidate's code will lose a referenced \
contract and either fail to compile, raise at runtime, or carry dead \
references that must be cleaned up.
- ADMIT a depth-1 caller/consumer of the deletion target on the structural \
edge alone IF the candidate's code surface (signature, abstraction) shows \
the deleted symbol's identifier appearing in a position that affects \
behaviour (call site, type annotation, field access, schema field). The \
acceptable justification format here IS: "directly references the deleted \
symbol via <CALLS|TYPED_BY|...> at <location>; will break/dangle after \
removal." This is NOT a tautology — for DELETION the structural reference \
IS the breakage mechanism.
- REJECT a deletion-edge candidate ONLY if the reference is purely lexical \
(comment, dead import never used in body, generic type parameter that has \
its own default, string literal matching the symbol name).
- For depth >= 2 chains from a DELETION target, the anti-tautology rule \
applies as normal: behavioural cascade through intermediate symbols is NOT \
automatic and must be justified concretely.

GOOD JUSTIFICATIONS (state the contract breakage or behavioral anomaly):
- "Adding the `pin` attribute to CommissionListingPayload requires this form \
component to add a new reactive state to display the pinned indicator."
- "The rate-limit constant referenced here changes from 5/min to 10/min, so \
the test that asserts the old value will fail."
- "This callee returns a new error variant after the CR; the caller's switch \
statement must add a branch."

FORBIDDEN JUSTIFICATIONS (bare topology, generic relation, score-based):
- "function A calls function B" / "A implements B" / "A imports B"
- "this node is in the same module" / "this is a primary target"
- "the chain contains IMPLEMENTS so it must be impacted"

CRITICAL RULES:
1. Judge based on the node's TYPE, FILE PATH, SOURCE CODE / ABSTRACTION, and \
the CR intent. The causal chain is supporting context only.
2. A node is impacted only if it DIRECTLY uses or exposes the changed feature.
3. A node that merely happens to be in the same file as an impacted node \
is NOT automatically impacted.
4. For ADDITION changes: nodes that WOULD need to be extended to support the \
new feature are impacted even if no current code handles the feature.
5. Return a verdict for EVERY node in the batch using the exact node IDs given.
6. CRITICAL: Copy the node_id exactly from BETWEEN the
   <<NODE_ID_START>>...<<NODE_ID_END>> delimiters. DO NOT include the
   << >> delimiter markers themselves in your JSON output. Do NOT
   paraphrase or truncate the node_id contents.
7. The justification field is REQUIRED for both confirmations and rejections.

OUTPUT FORMAT:
Return a JSON object: {"verdicts": [{"node_id": ..., \
"semantically_impacted": true/false, "justification": "..."}]}
"""


def _build_propagation_prompt(
    batch: list[tuple[str, NodeTrace]],
    node_meta_by_id: dict[str, dict],
    cr_interp: CRInterpretation,
) -> str:
    """Build the user prompt for one batch of propagated nodes.

    The causal chain is shown to LLM #4 as factual context. Anti-tautology
    safety is enforced by the system prompt (forbidden bare-topology
    justifications), not by hiding the chain.
    """
    is_deletion = (cr_interp.change_type or "").upper() == "DELETION"
    lines: list[str] = [
        f"Change Request Intent: {cr_interp.primary_intent}",
        f"Change Type: {cr_interp.change_type}",
        f"Domain Concepts: {', '.join(cr_interp.domain_concepts)}",
        "",
    ]
    if is_deletion:
        lines.append(
            "NOTE — DELETION CR: the DELETION EXCEPTION clause in the system "
            "prompt applies. For depth-1 candidates that directly CALL, are "
            "TYPED_BY, IMPLEMENT, or access FIELDS_ACCESSED of the deletion "
            "target, the structural edge IS impact evidence (the reference "
            "will dangle/break after removal). Use the candidate's signature "
            "or abstraction to confirm the deleted symbol identifier appears "
            "in a behaviour-affecting position before admitting."
        )
    else:
        lines.append(
            "For each node below, determine whether it is semantically impacted "
            "by the Change Request. The causal chain is shown as factual context; "
            "do NOT use edge types as impact evidence."
        )
    lines.append("")

    for i, (node_id, trace) in enumerate(batch, start=1):
        meta = node_meta_by_id.get(node_id, {})
        node_type = meta.get("node_type", "Unknown")
        file_path = meta.get("file_path", "")
        abstraction = meta.get("internal_logic_abstraction") or meta.get("source_code", "")
        if abstraction:
            abstraction = abstraction[:2500]

        chain_display = " -> ".join(trace.causal_chain) if trace.causal_chain else "(direct seed)"

        lines += [
            f"[{i}]",
            f"NODE ID: <<NODE_ID_START>>{node_id}<<NODE_ID_END>>",
            f"Type: {node_type}",
            f"File: {file_path}",
            f"Causal chain (factual context, NOT impact evidence): {chain_display}",
            f"Reached from SIS seed: {trace.source_seed}",
        ]
        if abstraction:
            lines += [
                "Node Abstraction / Signature:",
                abstraction,
            ]
        lines.append("")

    lines.append(
        "Return: {\"verdicts\": [{\"node_id\": ..., "
        "\"semantically_impacted\": true/false, \"justification\": \"...\"}]}"
    )
    return "\n".join(lines)


def validate_propagation(
    cis: CISResult,
    cr_interp: CRInterpretation,
    node_meta_by_id: dict[str, dict],
    client: LLMClient,
) -> tuple[CISResult, dict[str, str], bool]:
    """Filter CIS by LLM #4 decision.

    Fail-closed: per-node missing verdict -> drop. Batch exception -> drop
    entire batch. Per-child missing/exception -> drop.

    Per-node justifications are keyed by node_id and returned alongside the
    filtered CIS so the runner can attach them to NodeTrace.justification
    with source='llm4_propagation'.

    Returns:
        filtered_cis: CIS with LLM-#4-rejected nodes removed.
        justifications: dict node_id -> LLM #4 justification text (for kept nodes).
        degraded: True if any batch (main or child) was dropped due to API exhaustion.

    Blueprint §4 Step 7.
    """
    if not cis.propagated_nodes:
        return cis, {}, False

    auto_kept: dict[str, NodeTrace] = {}
    to_validate: list[tuple[str, NodeTrace]] = []
    auto_kept_justifications: dict[str, str] = {}

    for node_id, trace in cis.propagated_nodes.items():
        if trace.justification_source == "sibling_candidate":
            # Raw file-local sibling candidates (the in-file arm of
            # propagation) are validated by the dedicated sibling validator
            # (Step 7.5), NOT this outward-BFS validator. Pass them through
            # untouched so the sibling pass can admit/reject them.
            auto_kept[node_id] = trace
        elif (
            trace.depth == 1
            and trace.causal_chain
            and trace.causal_chain[-1] in PROPAGATION_VALIDATION_EXEMPT_EDGES
        ):
            auto_kept[node_id] = trace
            # Synthetic justification for auto-exempt nodes.
            auto_kept_justifications[node_id] = (
                f"Direct {trace.causal_chain[-1]} contract from {trace.source_seed} - "
                f"auto-admitted exempt edge."
            )
        else:
            to_validate.append((node_id, trace))

    logger.info(
        "[traversal_validator] {} propagated: {} auto-kept (exempt), {} to validate",
        len(cis.propagated_nodes),
        len(auto_kept),
        len(to_validate),
    )

    justifications: dict[str, str] = dict(auto_kept_justifications)
    degraded: bool = False

    if not to_validate:
        return cis, justifications, degraded

    # Deterministic shuffle to remove BFS-order positional bias.
    to_validate_shuffled = list(to_validate)
    random.seed(42)
    random.shuffle(to_validate_shuffled)
    to_validate = to_validate_shuffled

    kept_propagated: dict[str, NodeTrace] = dict(auto_kept)

    for batch_start in range(0, len(to_validate), _BATCH_SIZE):
        batch = to_validate[batch_start : batch_start + _BATCH_SIZE]
        prompt = _build_propagation_prompt(batch, node_meta_by_id, cr_interp)

        try:
            result: PropagationValidationResult = client.call(
                system=_SYSTEM_PROMPT,
                user=prompt,
                response_schema=PropagationValidationResult,
                call_name="validate_propagation",
            )
        except Exception as exc:
            logger.error(
                "[traversal_validator] Batch {}-{} failed after retries: {} - "
                "DROPPING batch (fail-closed)",
                batch_start, batch_start + len(batch), exc,
            )
            degraded = True
            continue

        verdict_map: dict[str, tuple[bool, str]] = {}
        for v in result.verdicts:
            clean_id = _strip_delimiters(v.node_id)
            verdict_map[clean_id] = (v.semantically_impacted, v.justification or "")

        for node_id, trace in batch:
            verdict = verdict_map.get(node_id)
            if verdict is None:
                logger.warning(
                    "[traversal_validator] No verdict for {} - DROPPING (fail-closed)",
                    node_id,
                )
                continue
            impacted, justification = verdict
            if impacted:
                kept_propagated[node_id] = trace
                justifications[node_id] = justification
            else:
                logger.debug(
                    "[traversal_validator] node_id={} rejected by LLM #4: {}",
                    node_id, justification,
                )

    # Per-child collapse validation REMOVED 2026-06 with Step 6.5: it
    # re-validated each NodeTrace.collapsed_children entry, but graph-collapse
    # (the only producer of collapsed_children) is gone, so that list is always
    # empty and this loop was dead. LLM #4 now validates only the propagated
    # nodes themselves.
    final_propagated = kept_propagated

    logger.info(
        "[traversal_validator] After LLM #4: {} propagated nodes kept (was {}, degraded={})",
        len(final_propagated),
        len(cis.propagated_nodes),
        degraded,
    )

    filtered_cis = CISResult(sis_nodes=cis.sis_nodes, propagated_nodes=final_propagated)
    return filtered_cis, justifications, degraded


# =========================================================================
# Sibling validation via LLM #4 (Step 7.5)
# =========================================================================

_SIBLING_SYSTEM_PROMPT = """\
You are a software impact analysis expert. You are reviewing whether a set
of code symbols inside ONE file must change to support a Change Request.

INPUT:
  - A Change Request (CR) intent and change_type.
  - One or more confirmed-impacted ANCHOR nodes inside this file (already
    validated upstream — accept all of them as ground truth).
  - Each anchor's justification (why it is impacted).
  - A list of SIBLING symbols defined in the same file as the anchors.

YOUR TASK:
For EACH sibling, decide whether — given that the anchors ARE impacted by
this CR — the sibling must ALSO change to coherently support the CR.

ADMIT a sibling when ANY of these holds:
  - The CR adds/changes a domain field, and the sibling is a related
    interface, type alias, schema, or input/payload definition that must
    expose that field on the same data shape.
  - The CR adds/changes a behaviour, and the sibling is a function in the
    same file that produces, consumes, validates, sanitizes, or persists
    the same data shape.
  - The CR's change_type is ADDITION and the sibling is the matching
    creation / update / delete function that must accept the new field
    (e.g. updateX, createX, sanitizeX, validateX when the CR adds a field
    to X's payload).
  - The CR concerns a UI form/page and the sibling is a sub-component
    defined in the same file that renders or edits the affected field.
  - The sibling shares a tight contract with any anchor (e.g. anchor is
    `updateX` and sibling is `XUpdateInput` — the input type of `updateX`).

REJECT a sibling when:
  - The sibling is a helper/util that operates on unrelated data
    (e.g. a date formatter when the CR adds a boolean flag).
  - The sibling's domain is orthogonal to the CR (e.g. an analytics event
    helper when the CR is about pricing).
  - The sibling exists only for a legacy concern the CR does not touch.

ANTI-TAUTOLOGY RULE (softer than primary LLM #4 rule):
  Same-file co-location alone is NOT sufficient. But the bar here is lower
  than for general BFS propagation — these candidates are siblings of a
  CONFIRMED anchor in the same file, which is a strong prior. Use the
  CR's primary_intent and the anchor justifications to identify the
  matching CRUD partner / type partner / form partner.

OUTPUT (JSON):
{
  "verdicts": [
    {"node_id": "<verbatim sibling id>",
     "semantically_impacted": true | false,
     "justification": "<one sentence, concrete>"},
    ...
  ]
}
Return one verdict per sibling. Copy the node_id VERBATIM from between
the <<NODE_ID_START>> ... <<NODE_ID_END>> delimiters.
"""


def validate_siblings_for_file(
    *,
    file_path: str,
    anchors: list[tuple[str, str]],
    siblings: list[tuple[str, str]],
    cr_interp: CRInterpretation,
    node_meta_by_id: dict[str, dict],
    client: LLMClient,
) -> tuple[dict[str, str], bool]:
    """LLM #4 sibling-batch: admit/reject siblings given multi-anchor context.

    Multi-anchor framing: this function takes ALL confirmed anchors in the
    file plus each anchor's justification (not a single anchor). Previous
    single-anchor designs failed when a bad anchor's framing made LLM #4
    reject legitimate sibling GTs. With multi-anchor context, LLM #4 sees
    the full contract surface and can identify CRUD/type partners correctly.

    Args:
        file_path: the shared file path of anchors + siblings.
        anchors: list of (anchor_id, anchor_justification) tuples — every
            validated impacted node currently in that file. Justification
            text comes from whichever upstream validator admitted each
            anchor (LLM #2 / #3 / #4).
        siblings: list of (sibling_id, node_type) tuples to adjudicate.
        cr_interp: the active CRInterpretation (intent + change_type).
        node_meta_by_id: optional metadata for code snippets / abstractions.
        client: shared LLMClient.

    Returns:
        (justifications, degraded) where:
          - justifications maps admitted sibling_id -> per-node justification.
          - degraded is True if the batch failed all retries (fail-closed).
    """
    if not siblings:
        return {}, False

    lines: list[str] = [
        f"Change Request Intent: {cr_interp.primary_intent}",
        f"Change Type: {cr_interp.change_type}",
        f"Domain Concepts: {', '.join(cr_interp.domain_concepts)}",
        "",
        f"File under review: {file_path}",
        "Confirmed-impacted ANCHOR symbols in this file:",
    ]
    for anchor_id, anchor_just in anchors:
        truncated = (anchor_just or "Validated upstream; no per-anchor justification text available.")[:300]
        lines.append(f"  - {anchor_id}")
        lines.append(f"      justification: {truncated}")
    lines += [
        "",
        "Adjudicate each sibling in the same file:",
        "",
    ]

    for i, (sib_id, sib_type) in enumerate(siblings, start=1):
        meta = node_meta_by_id.get(sib_id, {})
        abstraction = meta.get("internal_logic_abstraction") or meta.get("source_code", "")
        if abstraction:
            abstraction = abstraction[:2500]
        lines += [
            f"[{i}]",
            f"NODE ID: <<NODE_ID_START>>{sib_id}<<NODE_ID_END>>",
            f"Type: {sib_type}",
        ]
        if abstraction:
            lines += ["Signature / Abstraction:", abstraction]
        lines.append("")

    lines.append(
        'Return: {"verdicts": [{"node_id": ..., '
        '"semantically_impacted": true/false, "justification": "..."}]}'
    )
    prompt = "\n".join(lines)

    try:
        result: PropagationValidationResult = client.call(
            system=_SIBLING_SYSTEM_PROMPT,
            user=prompt,
            response_schema=PropagationValidationResult,
            call_name="validate_siblings",
        )
    except Exception as exc:
        logger.error(
            "[traversal_validator] Sibling batch for {} failed after retries: {} "
            "- DROPPING (fail-closed)", file_path, exc,
        )
        return {}, True

    verdict_map: dict[str, tuple[bool, str]] = {}
    for v in result.verdicts:
        clean_id = _strip_delimiters(v.node_id)
        verdict_map[clean_id] = (v.semantically_impacted, v.justification or "")

    admitted: dict[str, str] = {}
    for sib_id, _sib_type in siblings:
        verdict = verdict_map.get(sib_id)
        if verdict is None:
            logger.warning(
                "[traversal_validator] No sibling verdict for {} - DROPPING (fail-closed)",
                sib_id,
            )
            continue
        impacted, justification = verdict
        if impacted:
            admitted[sib_id] = justification

    logger.info(
        "[traversal_validator] Sibling batch {}: {} admitted of {} candidates",
        file_path, len(admitted), len(siblings),
    )
    return admitted, False
