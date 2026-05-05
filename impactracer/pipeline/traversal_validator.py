"""LLM Call #4: Propagation validation (FR-D2).

For each node reached by BFS (except exempted-edge single-hop neighbors),
LLM #4 decides whether the structural path implies semantic impact.

Batching Mandate: max 5 nodes per LLM call. Fail-open per node.
Anti-Circular Mandate: NO retrieval scores in the prompt.
Blind-Chain Mandate (Phase 2.3 / A-2/A-7): the causal chain that BFS used
  to reach a node is NOT shown to LLM #4. Showing the chain caused
  tautological confirmation bias — the LLM simply agreed that a node reached
  via IMPLEMENTS is impacted because IMPLEMENTS implies impact. Independent
  semantic assessment requires that the LLM reasons from node content alone.

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

_SYSTEM_PROMPT = """\
You are a software impact analysis expert. Your task is to determine whether \
a code node is SEMANTICALLY IMPACTED by a given Change Request.

A node is semantically impacted if modifying the CR's target would require \
changes to that node, or if the node's behaviour would change as a side-effect.

CRITICAL RULES:
1. Judge based on the node's TYPE, FILE PATH, and SOURCE CODE / ABSTRACTION \
against the CR intent — NOT on any graph path or retrieval score.
2. A node is impacted only if it DIRECTLY uses or exposes the changed feature.
3. A node that merely happens to be in the same file as an impacted node \
is NOT automatically impacted.
4. For ADDITION changes: nodes that WOULD need to be extended to support the \
new feature are impacted even if no current code handles the feature.
5. Return a verdict for EVERY node in the batch using the exact node IDs given.
6. Copy node_id VERBATIM using <<NODE_ID_START>>…<<NODE_ID_END>> delimiters.

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

    Phase 2.3 (A-2/A-7 — blind-chain): the causal chain (edge path from seed
    to node) is intentionally OMITTED from the prompt. Previous versions showed
    ``"Reached via: CALLS → IMPLEMENTS (depth 2)"`` which created a tautological
    loop: LLM #4 confirmed impact because the chain contains IMPLEMENTS, which
    the system-prompt rules declare as high-impact — circular reasoning, not
    independent semantic assessment. The seed ID is also omitted to prevent the
    LLM from inferring the chain indirectly.

    The LLM now judges solely on:
    - The CR intent and change type
    - The node's type, file path, and source abstraction
    - The collapsed children (structural metadata, not graph-path metadata)
    """
    lines: list[str] = [
        f"Change Request Intent: {cr_interp.primary_intent}",
        f"Change Type: {cr_interp.change_type}",
        f"Domain Concepts: {', '.join(cr_interp.domain_concepts)}",
        "",
        "For each node below, determine whether it is semantically impacted "
        "by the Change Request. Judge from the node content alone.",
        "",
    ]

    for i, (node_id, trace) in enumerate(batch, start=1):
        meta = node_meta_by_id.get(node_id, {})
        node_type = meta.get("node_type", "Unknown")
        file_path = meta.get("file_path", "")
        abstraction = meta.get("internal_logic_abstraction") or meta.get("source_code", "")
        if abstraction:
            abstraction = abstraction[:800]

        lines += [
            f"[{i}]",
            f"NODE ID: <<NODE_ID_START>>{node_id}<<NODE_ID_END>>",
            f"Type: {node_type}",
            f"File: {file_path}",
            # Phase 2.3: causal chain and seed ID intentionally omitted.
        ]
        # Collapsed children — structural fact about the node, not graph-path.
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
) -> CISResult:
    """Filter CIS by LLM #4 decision.

    Nodes reached at depth 1 via an edge in
    :data:`impactracer.shared.constants.PROPAGATION_VALIDATION_EXEMPT_EDGES`
    bypass the LLM call and are always kept.

    Blueprint §4 Step 7.
    """
    if not cis.propagated_nodes:
        return cis

    # Partition propagated nodes into auto-kept and to-validate.
    auto_kept: dict[str, NodeTrace] = {}
    to_validate: list[tuple[str, NodeTrace]] = []

    for node_id, trace in cis.propagated_nodes.items():
        # Exempt: depth=1 AND last edge in chain is in exempt set.
        if (
            trace.depth == 1
            and trace.causal_chain
            and trace.causal_chain[-1] in PROPAGATION_VALIDATION_EXEMPT_EDGES
        ):
            auto_kept[node_id] = trace
        else:
            to_validate.append((node_id, trace))

    logger.info(
        "[traversal_validator] {} propagated: {} auto-kept (exempt), {} to validate",
        len(cis.propagated_nodes),
        len(auto_kept),
        len(to_validate),
    )

    if not to_validate:
        # All propagated nodes were auto-kept — return unchanged CIS.
        return cis

    # Phase 2.4 (E-5): shuffle to_validate with a fixed seed before batching.
    # LLMs exhibit positional bias — items presented first in a batch tend to
    # receive more favourable verdicts. Shuffling with a fixed seed (42)
    # eliminates systematic ordering effects while remaining reproducible.
    to_validate_shuffled = list(to_validate)
    random.seed(42)
    random.shuffle(to_validate_shuffled)
    to_validate = to_validate_shuffled

    # Batch LLM #4 calls.
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
            logger.warning(
                "[traversal_validator] LLM call failed for batch {}-{}: {} — fail-open",
                batch_start, batch_start + len(batch), exc,
            )
            # Fail-open: keep all nodes in this batch.
            for node_id, trace in batch:
                kept_propagated[node_id] = trace
            continue

        # Build verdict lookup.
        verdict_map: dict[str, bool] = {
            v.node_id: v.semantically_impacted
            for v in result.verdicts
        }

        # Per-node coverage: fail-open for omitted nodes.
        for node_id, trace in batch:
            if node_id in verdict_map:
                if verdict_map[node_id]:
                    kept_propagated[node_id] = trace
                else:
                    logger.debug(
                        "[traversal_validator] node_id={} rejected by LLM #4", node_id
                    )
            else:
                logger.warning(
                    "[traversal_validator] No verdict for {} — fail-open (keep)", node_id
                )
                kept_propagated[node_id] = trace

    # Phase 2.2 (F-3): per-child collapse validation.
    # Sprint 10.1 attached collapsed children (InterfaceField IDs) to their
    # parent's NodeTrace. A single parent verdict was previously used to
    # implicitly admit ALL collapsed children without individual assessment —
    # bulk-admission. This block reassesses each child individually and removes
    # rejected children from the parent's collapsed_children list so the
    # context_builder and synthesizer do not render them.
    #
    # For each KEPT parent with collapsed_children, ask LLM #4 whether each
    # child name is individually impacted. We batch children into groups of 5
    # using the same prompt format. Rejected children are filtered out of the
    # parent's collapsed_children. This keeps the token economy of collapse
    # while restoring semantic per-child validation.
    _CHILD_SYSTEM = (
        "You are a software impact analysis expert. "
        "Given a parent code node that IS impacted by a Change Request, "
        "determine whether each of its named child fields/members is "
        "INDIVIDUALLY impacted (i.e. its value/type/behaviour must change). "
        "A child field that is unrelated to the CR's domain is NOT impacted. "
        'Return: {"verdicts": [{"node_id": "<child_name>", '
        '"semantically_impacted": true/false, "justification": "..."}]}'
    )

    final_propagated: dict[str, NodeTrace] = {}
    for node_id, trace in kept_propagated.items():
        if not trace.collapsed_children:
            final_propagated[node_id] = trace
            continue

        # Filter children by LLM #4.
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
                child_verdict_map: dict[str, bool] = {
                    v.node_id: v.semantically_impacted
                    for v in child_result.verdicts
                }
                for child_name in child_batch:
                    if child_verdict_map.get(child_name, True):  # fail-open
                        kept_children.append(child_name)
                    else:
                        logger.debug(
                            "[traversal_validator] Collapsed child '{}' of '{}' rejected",
                            child_name, node_id,
                        )
            except Exception as exc:
                logger.warning(
                    "[traversal_validator] Child validation failed for {} children of {}: {} — fail-open",
                    len(child_batch), node_id, exc,
                )
                kept_children.extend(child_batch)

        # Rebuild trace with filtered collapsed_children.
        from dataclasses import replace as _dc_replace
        final_propagated[node_id] = _dc_replace(trace, collapsed_children=kept_children)

    logger.info(
        "[traversal_validator] After LLM #4: {} propagated nodes kept (was {})",
        len(final_propagated),
        len(cis.propagated_nodes),
    )

    return CISResult(sis_nodes=cis.sis_nodes, propagated_nodes=final_propagated)
