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

    - ``resolutions``: ``[{"doc_id": ..., "code_ids": [...]}]`` for each
      doc-chunk SIS entry that resolved to at least one code node.
    - ``direct_code_seeds``: SIS entries that were already code nodes.

    The optional ``code_node_ids`` parameter lets the caller supply a
    pre-built set to skip the full table scan (used by the ablation harness).

    Blueprint §4 Step 5.
    """
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
