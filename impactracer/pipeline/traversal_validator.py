"""LLM Call #4: Propagation validation (FR-D2).

For each node reached by BFS (except exempted-edge single-hop neighbors),
LLM #4 decides whether the structural path implies semantic impact.

Batching Mandate: max 5 nodes per LLM call. Fail-open per node.
Anti-Circular Mandate: NO retrieval scores in the prompt.

Reference: master_blueprint.md §4 Step 7.
"""

from __future__ import annotations

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
a code node propagated via a structural dependency chain is SEMANTICALLY \
IMPACTED by a given Change Request.

A node is semantically impacted if modifying the CR's target would require \
changes to that node, or if the node's behaviour would change as a side-effect.

CRITICAL RULES:
1. Judge based on the structural edge chain and the code semantics — NOT on \
any retrieval score or ranking.
2. A node reached only via module-composition edges (IMPORTS, RENDERS, \
CONTAINS) is impacted only if it DIRECTLY uses or exposes the changed feature.
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
    """Build the user prompt for one batch of propagated nodes."""
    lines: list[str] = [
        f"Change Request Intent: {cr_interp.primary_intent}",
        f"Change Type: {cr_interp.change_type}",
        f"Domain Concepts: {', '.join(cr_interp.domain_concepts)}",
        "",
        "For each node below, determine whether it is semantically impacted "
        "by the Change Request.",
        "",
    ]

    for i, (node_id, trace) in enumerate(batch, start=1):
        meta = node_meta_by_id.get(node_id, {})
        node_type = meta.get("node_type", "Unknown")
        file_path = meta.get("file_path", "")
        abstraction = meta.get("internal_logic_abstraction") or meta.get("source_code", "")
        if abstraction:
            abstraction = abstraction[:800]

        # Include the originating SIS seed info for context.
        seed_id = trace.source_seed
        seed_meta = node_meta_by_id.get(seed_id, {})
        seed_abstraction = (
            seed_meta.get("internal_logic_abstraction")
            or seed_meta.get("source_code", "")
        )
        if seed_abstraction:
            seed_abstraction = seed_abstraction[:400]

        chain_str = " -> ".join(trace.causal_chain) if trace.causal_chain else "(direct seed)"

        lines += [
            f"[{i}]",
            f"NODE ID: <<NODE_ID_START>>{node_id}<<NODE_ID_END>>",
            f"Type: {node_type}",
            f"File: {file_path}",
            f"Reached via: {chain_str} (depth {trace.depth})",
            f"Originating SIS seed: {seed_id}",
        ]
        # Sprint 10.1 — render collapsed CONTAINS children inline.
        if trace.collapsed_children:
            lines.append(
                f"Contains {len(trace.collapsed_children)} collapsed child field(s): "
                + ", ".join(trace.collapsed_children[:20])
                + (" ..." if len(trace.collapsed_children) > 20 else "")
            )
        if seed_abstraction:
            lines += [
                "Seed Abstraction:",
                seed_abstraction,
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

    logger.info(
        "[traversal_validator] After LLM #4: {} propagated nodes kept (was {})",
        len(kept_propagated),
        len(cis.propagated_nodes),
    )

    return CISResult(sis_nodes=cis.sis_nodes, propagated_nodes=kept_propagated)
