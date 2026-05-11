"""Backlink retrieval and synthesis context assembly (FR-E1, FR-E2).

Reference: master_blueprint.md §4 Steps 8.
"""

from __future__ import annotations

import json
import sqlite3

from loguru import logger

from impactracer.shared.constants import severity_for_chain
from impactracer.shared.models import CISResult, CRInterpretation


_CODE_NODE_TYPES: frozenset[str] = frozenset({
    "File", "Class", "Function", "Method", "Interface",
    "TypeAlias", "Enum", "ExternalPackage", "InterfaceField",
})


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
# FR-E1: Source snippet retrieval
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
      ILA preserves call sites and signatures while stripping whitespace and
      comment noise that wastes token budget.
    - Other code nodes: ``source_code`` (no ILA generated for them).
    - Doc chunk nodes: fetched from ChromaDB ``doc_chunks`` documents.
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
        if node_type in _ILA_NODE_TYPES and ila:
            result[node_id] = ila
        else:
            result[node_id] = source_code or ""

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
# Depth-based hard-limit failsafe
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


_SEVERITY_RANK_MAP = {"Tinggi": 0, "Menengah": 1, "Rendah": 2}


def _apply_hard_limit(
    sorted_ids: list[str],
    combined: dict,
    *,
    char_limit: int = _HARD_CHAR_LIMIT,
) -> tuple[list[str], str]:
    """Return (trimmed_ids, warning_text) after applying the hard char limit.

    Severity-aware truncation: sort key for droppable nodes is
    (severity_rank ASC, depth DESC, node_id).
    Tinggi (contract dependencies) always survive even at depth 3; Rendah
    (composition) at depth 2 is dropped before Tinggi at depth 3.

    SIS seeds (depth=0) remain immune (highest priority).

    Algorithm:
    1. Proxy per-node cost = 2000 chars (matches typical block size).
    2. If full set fits, return unchanged.
    3. Sort droppable by (severity_rank ASC, depth DESC, node_id) so the
       lowest-severity / deepest nodes are dropped first.
    4. Keep as many as the budget allows; report the worst surviving
       severity in the warning text so the LLM #5 prompt acknowledges it.
    """
    _PROXY_CHARS_PER_NODE = 2000
    if len(sorted_ids) * _PROXY_CHARS_PER_NODE <= char_limit:
        return sorted_ids, ""

    immune: list[str] = []
    droppable: list[tuple[int, int, str]] = []  # (severity_rank, depth, node_id)
    for nid in sorted_ids:
        trace = combined.get(nid)
        if trace is None or trace.depth == 0:
            immune.append(nid)
            continue
        sev_rank = _SEVERITY_RANK_MAP.get(severity_for_chain(trace.causal_chain), 2)
        droppable.append((sev_rank, trace.depth, nid))

    # Sort ascending by (severity_rank, depth, nid). Lowest severity_rank
    # = highest priority = head; deepest = tail. Among equal severity,
    # shallowest comes first (kept first), deepest is dropped first. Among
    # equal severity AND depth, alphabetical by node_id is the tie-breaker.
    droppable.sort(key=lambda t: (t[0], t[1], t[2]))

    immune_budget = len(immune) * _PROXY_CHARS_PER_NODE
    available_for_droppable = max(char_limit - immune_budget, 0)
    per_node_budget = _PROXY_CHARS_PER_NODE

    max_keep = available_for_droppable // per_node_budget
    kept_droppable = droppable[:max_keep]
    dropped = droppable[max_keep:]

    if not dropped:
        return sorted_ids, ""

    kept_set: set[str] = set(immune) | {nid for _, _, nid in kept_droppable}
    trimmed_ids = [nid for nid in sorted_ids if nid in kept_set]

    # Report the worst surviving severity and the count dropped.
    if kept_droppable:
        worst_sev_rank_kept = max(d[0] for d in kept_droppable)
    else:
        worst_sev_rank_kept = 0
    sev_label = {0: "Tinggi", 1: "Menengah", 2: "Rendah"}[worst_sev_rank_kept]
    warning = (
        f"[SYSTEM WARNING: The impact graph was too massive for the LLM "
        f"prompt. {len(dropped)} nodes were truncated from this prompt "
        f"(severity-aware: lowest-severity, deepest-first). The deepest "
        f"surviving severity tier is '{sev_label}'. Note: the FULL impact "
        f"set is still emitted in ImpactReport.impacted_nodes — this "
        f"truncation only affects the prompt window, not the report.]"
    )
    logger.warning(
        "[context_builder] Hard-limit failsafe (severity-aware): dropped {} "
        "nodes, {} nodes remain in prompt; worst surviving severity={}",
        len(dropped), len(trimmed_ids), sev_label,
    )
    return trimmed_ids, warning


# ---------------------------------------------------------------------------
# FR-E2: Context assembly
# ---------------------------------------------------------------------------

# Severity rank used by the pre-truncation prompt-priority sort (group 1).
# Identical to _SEVERITY_RANK_MAP defined earlier in the module — kept as a
# named alias so the existing _sort_key reads idiomatically.
_SEVERITY_RANK = _SEVERITY_RANK_MAP


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

    Truncation priority:
    1. Nodes with populated causal_chain (BFS-propagated) sorted by severity
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

    # Sort for inclusion — HIGHEST PRIORITY FIRST.
    # Group 0: SIS seeds (depth=0, empty causal_chain), sorted by retrieval score DESC.
    # Group 1: BFS-propagated nodes, sorted by severity rank ASC, depth ASC, then id.
    def _sort_key(nid: str) -> tuple:
        m = node_meta[nid]
        if not m["has_chain"]:
            # Group 0: SIS seeds — priority placement.
            # Inverted retrieval score: higher score → smaller sort key → earlier.
            return (0, -m["retrieval_score"], m["depth"], nid)
        else:
            # Group 1: BFS-propagated — after all seeds.
            # Severity rank ASC (Tinggi=0 before Menengah=1 before Rendah=2),
            # then depth ASC (shallower first), then id for determinism.
            return (1, _SEVERITY_RANK[m["severity"]], m["depth"], nid)

    sorted_ids = sorted(combined.keys(), key=_sort_key)

    sorted_ids, hard_limit_warning = _apply_hard_limit(sorted_ids, combined)

    file_set: list[str] = []
    seen_fp: set[str] = set()
    for nid in sorted_ids:
        fp = (node_file_paths or {}).get(nid, "")
        if fp and fp not in seen_fp:
            seen_fp.add(fp)
            file_set.append(fp)

    file_block_lines = [
        "",
        "=== IMPACTED FILES (write a file_justifications row for EACH of these) ===",
    ]
    for fp in file_set:
        file_block_lines.append(f"  - {fp}")

    # Build the fixed header
    header_parts = [
        "=== CHANGE REQUEST ===",
        cr_text,
        "",
        "=== INTERPRETATION ===",
        json.dumps(cr_interp.model_dump(), ensure_ascii=False, indent=2),
        "",
        f"=== IMPACTED NODE COUNT: {len(sorted_ids)} ===",
        *file_block_lines,
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
        chain_display = " -> ".join(meta["causal_chain"]) if meta["causal_chain"] else "[]"
        file_path = (node_file_paths or {}).get(nid, "")
        node_type = (node_types or {}).get(nid, "DocChunk")
        trace_obj = combined.get(nid)
        block_lines = [
            f"--- NODE: {nid} ---",
            f"node_type: {node_type}",
            f"file_path: {file_path if file_path else '(doc chunk - no source file)'}",
            f"Severity: {meta['severity']}",
            f"Depth: {meta['depth']}",
            f"Causal chain (readable): {chain_display}",
            f"Path from seed: {' -> '.join(meta['path']) if meta['path'] else nid}",
            f"Source seed: {meta['source_seed']}",
        ]
        if trace_obj is not None and trace_obj.justification:
            src_label = trace_obj.justification_source or "validator"
            block_lines.append(
                f"Justification ({src_label}): {trace_obj.justification}"
            )
        if trace_obj is not None and trace_obj.mechanism_of_impact:
            block_lines.append(
                f"Mechanism of impact (LLM #2): {trace_obj.mechanism_of_impact}"
            )
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

    if hard_limit_warning:
        context_parts.insert(0, hard_limit_warning + "\n")

    result = "\n".join(context_parts)
    logger.info(
        "[context_builder] Context assembled: {} nodes included, ~{} tokens",
        len(included_ids), _estimate_tokens(result),
    )
    return result
