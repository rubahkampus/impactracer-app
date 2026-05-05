"""Backlink retrieval and synthesis context assembly (FR-E1, FR-E2).

Reference: master_blueprint.md §4 Steps 8.
"""

from __future__ import annotations

import json
import sqlite3

from loguru import logger

from impactracer.shared.constants import severity_for_chain
from impactracer.shared.models import CISResult, CRInterpretation


# ---------------------------------------------------------------------------
# Code node type set for backlink routing (FF-3)
# ---------------------------------------------------------------------------

_CODE_NODE_TYPES: frozenset[str] = frozenset({
    "File", "Class", "Function", "Method", "Interface",
    "TypeAlias", "Enum", "ExternalPackage", "InterfaceField",
})


# ---------------------------------------------------------------------------
# FR-E1: Bidirectional backlink retrieval (FF-3)
# ---------------------------------------------------------------------------


def fetch_backlinks(
    node_ids: list[str],
    node_types: dict[str, str],
    conn: sqlite3.Connection,
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Fetch top-K traceability backlinks per node, routing by node type.

    - Code nodes (Function, Method, etc.): query ``WHERE code_id IN (ids)``
      → returns [(doc_id, score), ...] — "which docs specify this code"
    - Doc chunk nodes: query ``WHERE doc_id IN (ids)``
      → returns [(code_id, score), ...] — "which code implements this doc"

    Both directions are meaningful for the synthesizer to establish confidence.
    """
    if not node_ids:
        return {}

    code_ids = [n for n in node_ids if node_types.get(n, "DocChunk") in _CODE_NODE_TYPES]
    doc_ids  = [n for n in node_ids if node_types.get(n, "DocChunk") not in _CODE_NODE_TYPES]

    result: dict[str, list[tuple[str, float]]] = {}

    # Code nodes → find their doc backlinks
    if code_ids:
        placeholders = ",".join("?" * len(code_ids))
        rows = conn.execute(
            f"SELECT code_id, doc_id, weighted_similarity_score "
            f"FROM doc_code_candidates "
            f"WHERE code_id IN ({placeholders}) "
            f"ORDER BY code_id, weighted_similarity_score DESC",
            code_ids,
        ).fetchall()
        for code_id, doc_id, score in rows:
            bucket = result.setdefault(code_id, [])
            if len(bucket) < top_k:
                bucket.append((doc_id, score))

    # Doc chunk nodes → find which code nodes they trace to
    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        rows = conn.execute(
            f"SELECT doc_id, code_id, weighted_similarity_score "
            f"FROM doc_code_candidates "
            f"WHERE doc_id IN ({placeholders}) "
            f"ORDER BY doc_id, weighted_similarity_score DESC",
            doc_ids,
        ).fetchall()
        for doc_id, code_id, score in rows:
            bucket = result.setdefault(doc_id, [])
            if len(bucket) < top_k:
                bucket.append((code_id, score))

    return result


# ---------------------------------------------------------------------------
# FR-E1: Source snippet retrieval (ED-8: includes doc chunk content)
# ---------------------------------------------------------------------------


_ILA_NODE_TYPES: frozenset[str] = frozenset({"Function", "Method"})


def fetch_snippets(
    node_ids: list[str],
    conn: sqlite3.Connection,
    doc_col: object | None = None,
) -> dict[str, str]:
    """Fetch source text for each node ID.

    - Code nodes (Function/Method): prefers ``internal_logic_abstraction``
      (skeletonized reduction) over ``source_code`` when ILA is available.
      N6 fix: ILA preserves call sites and signatures the synthesizer needs
      for structural_justification; raw source_code includes whitespace and
      comment noise that wastes token budget.
    - Other code nodes: ``source_code`` (no ILA generated for them).
    - Doc chunk nodes: fetched from ChromaDB ``doc_chunks`` documents
      (ED-8: previously returned empty string, causing content-free context
      blocks and hallucinated structural_justification in synthesis).
    """
    if not node_ids:
        return {}

    # Fetch code nodes from SQLite — fetch both source_code and ILA in one query
    placeholders = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT node_id, node_type, internal_logic_abstraction, source_code "
        f"FROM code_nodes WHERE node_id IN ({placeholders})",
        node_ids,
    ).fetchall()

    result: dict[str, str] = {}
    for node_id, node_type, ila, source_code in rows:
        # N6: prefer ILA for Function/Method nodes
        if node_type in _ILA_NODE_TYPES and ila:
            result[node_id] = ila
        else:
            result[node_id] = source_code or ""

    # Fetch doc chunk content from ChromaDB for IDs missing from SQLite (ED-8)
    missing_ids = [n for n in node_ids if n not in result]
    if missing_ids and doc_col is not None:
        try:
            chroma_result = doc_col.get(ids=missing_ids, include=["documents"])
            for cid, doc_text in zip(chroma_result["ids"], chroma_result["documents"]):
                result[cid] = doc_text or ""
        except Exception as exc:
            logger.warning("[context_builder] ChromaDB snippet fetch failed for {} doc ids: {}", len(missing_ids), exc)

    return result


# ---------------------------------------------------------------------------
# Token budget helper
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Sprint 10.1 — Strategy 2: depth-based hard-limit failsafe
# ---------------------------------------------------------------------------

# Hard ceiling enforced BEFORE the per-token-budget soft truncation.
# ~60 K tokens × 4 chars/token = 240 000 chars.
_HARD_CHAR_LIMIT: int = 240_000

# Warning block injected into the context when the hard limit activates.
_TRUNCATION_WARNING_TEMPLATE = (
    "[SYSTEM WARNING: The impact graph was too massive. "
    "Nodes beyond Depth {max_depth} were truncated from this context "
    "({n_dropped} nodes dropped). "
    "Explicitly mention this depth-based truncation limitation in your "
    "report scope section.]"
)


def _apply_hard_limit(
    sorted_ids: list[str],
    combined: dict,
    *,
    char_limit: int = _HARD_CHAR_LIMIT,
) -> tuple[list[str], str]:
    """Return (trimmed_ids, warning_text) after applying the hard char limit.

    Drops highest-depth propagated nodes first.  SIS seeds (depth=0) are
    immune and always survive the trim.

    Algorithm:
    1. Estimate rough per-node char cost = len(node_id) * 10 (conservative
       proxy — the real block is built later; we just need to pick which
       nodes to drop, not the exact budget).
    2. Sort candidates-for-dropping by depth descending, then node_id.
    3. Drop until the rough total is within char_limit.
    4. If the limit is already satisfied, return unchanged.
    """
    # Quick check: build a rough total using a fixed per-node char estimate.
    # We use 500 chars/node as a safe lower bound (real blocks are larger,
    # so this won't over-drop).
    if len(sorted_ids) * 500 <= char_limit:
        return sorted_ids, ""

    # Separate immune nodes (depth=0 SIS seeds) from droppable (depth>0).
    immune: list[str] = []
    droppable: list[tuple[int, str]] = []  # (depth, node_id) for sorting
    for nid in sorted_ids:
        trace = combined.get(nid)
        depth = trace.depth if trace is not None else 0
        if depth == 0:
            immune.append(nid)
        else:
            droppable.append((depth, nid))

    # Sort droppable by depth ascending so the shallowest (highest priority)
    # nodes are kept first.  Deepest nodes appear at the end and are dropped.
    droppable.sort(key=lambda t: (t[0], t[1]))

    # Allow up to char_limit chars; reserve space for immune nodes.
    # Each immune node gets a guaranteed budget slot.
    immune_budget = len(immune) * 800  # generous per-seed estimate
    available_for_droppable = max(char_limit - immune_budget, 0)
    per_node_budget = 800  # chars per droppable node (conservative estimate)

    max_keep = available_for_droppable // per_node_budget
    kept_droppable = droppable[:max_keep]  # shallowest max_keep nodes
    dropped = droppable[max_keep:]         # deepest nodes — these are dropped

    if not dropped:
        return sorted_ids, ""

    # Reconstruct sorted_ids preserving the original order among survivors.
    kept_set: set[str] = set(immune) | {nid for _, nid in kept_droppable}
    trimmed_ids = [nid for nid in sorted_ids if nid in kept_set]

    # max_kept_depth = the deepest surviving node's depth → the cutoff threshold.
    max_kept_depth = max((d for d, _ in kept_droppable), default=0)
    warning = _TRUNCATION_WARNING_TEMPLATE.format(
        max_depth=max_kept_depth,
        n_dropped=len(dropped),
    )
    logger.warning(
        "[context_builder] Hard-limit failsafe activated: dropped {} nodes "
        "(depth > {}), {} nodes remain",
        len(dropped), max_kept_depth, len(trimmed_ids),
    )
    return trimmed_ids, warning


# ---------------------------------------------------------------------------
# FR-E2: Context assembly
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"Tinggi": 0, "Menengah": 1, "Rendah": 2}


def build_context(
    cr_text: str,
    cr_interp: CRInterpretation,
    cis: CISResult,
    backlinks: dict[str, list[tuple[str, float]]],
    snippets: dict[str, str],
    settings: object,
    node_file_paths: dict[str, str] | None = None,
    node_types: dict[str, str] | None = None,
    candidate_scores: dict[str, float] | None = None,
) -> str:
    """Assemble the final LLM Call #5 user message (FR-E2).

    Enforces the token budget; truncates lowest-priority nodes first.

    Truncation priority (AV-4):
    1. Nodes with populated causal_chain (BFS-propagated, Sprint 10+) sorted by severity
    2. SIS seeds (empty causal_chain) sorted by retrieval score descending
    3. Within each group: depth ascending, then node_id alphabetic

    ``node_file_paths``: maps node_id to file_path for doc chunks.
    ``node_types``: maps node_id to node_type string.
    ``candidate_scores``: maps node_id to RRF/reranker score for pre-BFS ordering.
    """
    budget = (
        settings.llm_max_context_tokens
        - settings.synthesis_system_prompt_tokens
        - settings.output_reserve_tokens
    )

    combined = cis.combined()
    _candidate_scores = candidate_scores or {}

    # Pre-compute metadata for sort-truncation
    node_meta: dict[str, dict] = {}
    for nid, trace in combined.items():
        sev = severity_for_chain(trace.causal_chain)
        has_chain = len(trace.causal_chain) > 0
        node_meta[nid] = {
            "severity": sev,
            "depth": trace.depth,
            "causal_chain": trace.causal_chain,
            "path": trace.path,
            "source_seed": trace.source_seed,
            "has_chain": has_chain,
            "retrieval_score": _candidate_scores.get(nid, 0.0),
        }

    # AV-4: sort for inclusion
    # Group 1: BFS-propagated nodes (has_chain=True) — sort by severity, depth, id
    # Group 2: SIS seeds (has_chain=False) — sort by retrieval score desc, depth, id
    def _sort_key(nid: str) -> tuple:
        m = node_meta[nid]
        if m["has_chain"]:
            # Group 1 first (0), then severity rank, then depth, then id
            return (0, _SEVERITY_RANK[m["severity"]], m["depth"], nid)
        else:
            # Group 2 (1), then inverted retrieval score (higher score = lower sort key)
            return (1, -m["retrieval_score"], m["depth"], nid)

    sorted_ids = sorted(combined.keys(), key=_sort_key)

    # Sprint 10.1 — Strategy 2: apply hard char-limit failsafe BEFORE
    # the per-token-budget soft truncation.  Drops highest-depth propagated
    # nodes first; SIS seeds (depth=0) are immune.
    sorted_ids, hard_limit_warning = _apply_hard_limit(sorted_ids, combined)

    # Build the fixed header
    header_parts = [
        "=== CHANGE REQUEST ===",
        cr_text,
        "",
        "=== INTERPRETATION ===",
        json.dumps(cr_interp.model_dump(), ensure_ascii=False, indent=2),
        "",
        f"=== IMPACTED NODE COUNT: {len(sorted_ids)} ===",
        "",
    ]
    header = "\n".join(header_parts)
    header_tokens = _estimate_tokens(header)

    # Build node blocks
    node_blocks: list[tuple[str, str]] = []
    for nid in sorted_ids:
        meta = node_meta[nid]
        bl = backlinks.get(nid, [])
        snippet = snippets.get(nid, "")
        chain_display = " → ".join(meta["causal_chain"]) if meta["causal_chain"] else "[]"
        file_path = (node_file_paths or {}).get(nid, "")
        node_type = (node_types or {}).get(nid, "DocChunk")
        block_lines = [
            f"--- NODE: {nid} ---",
            f"node_type: {node_type}",
            f"file_path: {file_path if file_path else '(doc chunk — no source file)'}",
            f"Severity: {meta['severity']}",
            f"Depth: {meta['depth']}",
            f"Causal chain (JSON array for output): {json.dumps(meta['causal_chain'])}",
            f"Causal chain (readable): {chain_display}",
            f"Path from seed: {' → '.join(meta['path']) if meta['path'] else nid}",
            f"Source seed: {meta['source_seed']}",
        ]
        # Sprint 10.1 — render collapsed CONTAINS children inline.
        trace_obj = combined.get(nid)
        if trace_obj is not None and trace_obj.collapsed_children:
            collapsed = trace_obj.collapsed_children
            child_preview = ", ".join(collapsed[:20])
            if len(collapsed) > 20:
                child_preview += f" ... (+{len(collapsed) - 20} more)"
            block_lines.append(
                f"Collapsed CONTAINS children ({len(collapsed)}): {child_preview}"
            )
        if bl:
            block_lines.append(
                "Traceability backlinks: " + ", ".join(f"{d}({s:.3f})" for d, s in bl[:3])
            )
        if snippet:
            # N6: for Function/Method nodes, the SQLite source_code field is the
            # raw TypeScript body.  Prefer the ILA (internal_logic_abstraction)
            # when available — it is a skeletonized reduction that strips whitespace
            # noise and preserves the call graph the synthesizer needs for
            # structural_justification.  The raw source_code is a reliable
            # fallback when ILA was not generated (degenerate nodes, non-TS files).
            # No 1500-char truncation: ILA is already compact; truncating raw
            # source drops important call sites and field assignments.
            block_lines.append("Source snippet:")
            block_lines.append(snippet)
        block_lines.append("")
        node_blocks.append((nid, "\n".join(block_lines)))

    # Fit within budget
    used_tokens = header_tokens
    included_ids: list[str] = []
    included_blocks: list[str] = []
    truncated = 0
    for nid, block in node_blocks:
        block_tokens = _estimate_tokens(block)
        if used_tokens + block_tokens <= budget:
            included_ids.append(nid)
            included_blocks.append(block)
            used_tokens += block_tokens
        else:
            truncated += 1

    context_parts = [header] + included_blocks
    if truncated > 0:
        context_parts.append(
            f"[TRUNCATION NOTE: {truncated} lower-priority nodes omitted due to token budget. "
            f"Included {len(included_ids)} of {len(sorted_ids)} nodes. "
            f"BFS-propagated nodes (if any) were prioritized; SIS seeds ordered by retrieval score.]"
        )
        logger.info(
            "[context_builder] Truncated {} nodes (budget={} tokens, used={})",
            truncated, budget, used_tokens,
        )

    # Sprint 10.1 — prepend hard-limit warning if the failsafe activated.
    if hard_limit_warning:
        context_parts.insert(0, hard_limit_warning + "\n")

    result = "\n".join(context_parts)
    logger.info(
        "[context_builder] Context assembled: {} nodes included, ~{} tokens",
        len(included_ids), _estimate_tokens(result),
    )
    return result
