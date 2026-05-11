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

CRITICAL ANTI-TAUTOLOGY RULE:
- Edge types are NOT impact evidence. A chain containing IMPLEMENTS, CALLS, \
or any other edge does NOT by itself confirm impact. Many chains terminate \
at nodes whose behaviour is unaffected by the CR despite a structurally \
present relationship.
- Reject any node where the chain merely describes a generic dependency that \
the CR does not actually disturb (e.g. a function CALLS a utility that the CR \
does not modify; a class IMPLEMENTS an interface whose contract is unchanged).

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
    lines: list[str] = [
        f"Change Request Intent: {cr_interp.primary_intent}",
        f"Change Type: {cr_interp.change_type}",
        f"Domain Concepts: {', '.join(cr_interp.domain_concepts)}",
        "",
        "For each node below, determine whether it is semantically impacted "
        "by the Change Request. The causal chain is shown as factual context; "
        "do NOT use edge types as impact evidence.",
        "",
    ]

    for i, (node_id, trace) in enumerate(batch, start=1):
        meta = node_meta_by_id.get(node_id, {})
        node_type = meta.get("node_type", "Unknown")
        file_path = meta.get("file_path", "")
        abstraction = meta.get("internal_logic_abstraction") or meta.get("source_code", "")
        if abstraction:
            abstraction = abstraction[:1200]

        chain_display = " -> ".join(trace.causal_chain) if trace.causal_chain else "(direct seed)"

        lines += [
            f"[{i}]",
            f"NODE ID: <<NODE_ID_START>>{node_id}<<NODE_ID_END>>",
            f"Type: {node_type}",
            f"File: {file_path}",
            f"Causal chain (factual context, NOT impact evidence): {chain_display}",
            f"Reached from SIS seed: {trace.source_seed}",
        ]
        if trace.collapsed_children:
            lines.append(
                f"Contains {len(trace.collapsed_children)} collapsed child field(s): "
                + ", ".join(trace.collapsed_children[:20])
                + (" ..." if len(trace.collapsed_children) > 20 else "")
            )
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
        if (
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

    # Per-child collapse validation.
    _CHILD_SYSTEM = (
        "You are a software impact analysis expert. "
        "Given a parent code node that IS impacted by a Change Request, "
        "determine whether each of its named child fields/members is "
        "INDIVIDUALLY impacted (i.e. its value/type/behaviour must change). "
        "A child field that is unrelated to the CR's domain is NOT impacted. "
        "Return a justification citing the specific structural reason. "
        'Return: {"verdicts": [{"node_id": "<child_name>", '
        '"semantically_impacted": true/false, "justification": "..."}]}'
    )

    final_propagated: dict[str, NodeTrace] = {}
    for node_id, trace in kept_propagated.items():
        if not trace.collapsed_children:
            final_propagated[node_id] = trace
            continue

        children = trace.collapsed_children
        kept_children: list[str] = []

        for child_batch_start in range(0, len(children), _BATCH_SIZE):
            child_batch = children[child_batch_start: child_batch_start + _BATCH_SIZE]
            child_prompt_lines = [
                f"Change Request Intent: {cr_interp.primary_intent}",
                f"Change Type: {cr_interp.change_type}",
                f"Parent node: {node_id} (confirmed impacted)",
                "",
                "For each child field below, determine if it is individually impacted:",
                "",
            ]
            for j, child_name in enumerate(child_batch, start=1):
                child_prompt_lines.append(
                    f"[{j}] NODE ID: <<NODE_ID_START>>{child_name}<<NODE_ID_END>>"
                )
            child_prompt_lines.append(
                'Return: {"verdicts": [{"node_id": ..., '
                '"semantically_impacted": true/false, "justification": "..."}]}'
            )
            child_prompt = "\n".join(child_prompt_lines)

            try:
                child_result: PropagationValidationResult = client.call(
                    system=_CHILD_SYSTEM,
                    user=child_prompt,
                    response_schema=PropagationValidationResult,
                    call_name="validate_collapsed_children",
                )
            except Exception as exc:
                logger.error(
                    "[traversal_validator] Child validation failed for {} children "
                    "of {}: {} - DROPPING child batch (fail-closed)",
                    len(child_batch), node_id, exc,
                )
                degraded = True
                continue

            child_verdict_map: dict[str, tuple[bool, str]] = {}
            for v in child_result.verdicts:
                clean_id = _strip_delimiters(v.node_id)
                child_verdict_map[clean_id] = (v.semantically_impacted, v.justification or "")

            for child_name in child_batch:
                child_verdict = child_verdict_map.get(child_name)
                if child_verdict is None:
                    logger.warning(
                        "[traversal_validator] No child verdict for '{}' of '{}' - "
                        "DROPPING (fail-closed)",
                        child_name, node_id,
                    )
                    continue
                impacted, child_just = child_verdict
                if impacted:
                    kept_children.append(child_name)
                    # Store under "<parent>::<child>" key for distributed-justification audit.
                    justifications[f"{node_id}::{child_name}"] = child_just

        from dataclasses import replace as _dc_replace
        final_propagated[node_id] = _dc_replace(trace, collapsed_children=kept_children)

    logger.info(
        "[traversal_validator] After LLM #4: {} propagated nodes kept (was {}, degraded={})",
        len(final_propagated),
        len(cis.propagated_nodes),
        degraded,
    )

    filtered_cis = CISResult(sis_nodes=cis.sis_nodes, propagated_nodes=final_propagated)
    return filtered_cis, justifications, degraded
