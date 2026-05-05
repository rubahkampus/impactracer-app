"""Trace resolution: doc-chunk SIS -> code seeds (FR-C6).

Uses the precomputed ``doc_code_candidates`` table. Direct code-node SIS
entries bypass resolution. The output feeds LLM #3 when enabled.

Reference: master_blueprint.md §4 Step 5.
"""

from __future__ import annotations

import sqlite3


def resolve_doc_to_code(
    sis_ids: list[str],
    conn: sqlite3.Connection,
    top_k: int,
    code_node_ids: set[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Return (resolutions, direct_code_seeds).

    - ``resolutions``: list of ``{"doc_id": ..., "code_ids": [...]}``
      for doc-chunk SIS entries that resolved to at least one code node.
    - ``direct_code_seeds``: SIS entries that were already code nodes.

    Phase 1 fix (E-NEW-3 / E-6): the previous signature returned a third
    value ``doc_to_code_map`` that was silently discarded by runner.py on
    every call — dead return value eliminated. The function also previously
    executed a full ``SELECT node_id FROM code_nodes`` table scan on every
    invocation (160× during ablation). The optional ``code_node_ids``
    parameter lets the caller supply a pre-built set so the scan is a no-op.

    Blueprint §4 Step 5: direct code-node SIS entries pass through as
    direct_code_seeds; doc-chunk SIS entries query doc_code_candidates
    ORDER BY weighted_similarity_score DESC LIMIT top_k.
    """
    # Materialise the full set of known code node IDs for membership testing.
    # If the caller passes a pre-built set (E-6 cache), use it directly.
    if code_node_ids is None:
        code_node_set: set[str] = {
            row[0]
            for row in conn.execute("SELECT node_id FROM code_nodes").fetchall()
        }
    else:
        code_node_set = code_node_ids

    direct_code_seeds: list[str] = []
    resolutions: list[dict] = []  # [{"doc_id": x, "code_ids": [...]}]

    for nid in sis_ids:
        if nid in code_node_set:
            direct_code_seeds.append(nid)
            continue

        # Doc-chunk: resolve via precomputed traceability table.
        rows = conn.execute(
            "SELECT code_id FROM doc_code_candidates "
            "WHERE doc_id = ? "
            "ORDER BY weighted_similarity_score DESC "
            "LIMIT ?",
            (nid, top_k),
        ).fetchall()

        if rows:
            code_ids = [r[0] for r in rows]
            resolutions.append({"doc_id": nid, "code_ids": code_ids})
        # If no rows: doc chunk is stranded (no candidates above threshold).
        # Log is handled by runner; we silently skip here.

    return resolutions, direct_code_seeds
