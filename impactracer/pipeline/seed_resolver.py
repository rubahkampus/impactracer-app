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
) -> tuple[list[dict], dict[str, list[str]], list[str]]:
    """Return (resolutions, doc_to_code_map, direct_code_seeds).

    - ``resolutions``: list of ``{"doc_id": ..., "code_ids": [...]}``
      for doc-chunk SIS entries that resolved to at least one code node.
    - ``doc_to_code_map``: ``{doc_id: [code_ids]}`` for synthesis backlinks.
    - ``direct_code_seeds``: SIS entries that were already code nodes.

    Blueprint §4 Step 5: direct code-node SIS entries pass through as
    direct_code_seeds; doc-chunk SIS entries query doc_code_candidates
    ORDER BY weighted_similarity_score DESC LIMIT top_k.
    """
    # Materialise the full set of known code node IDs for membership testing.
    code_node_set: set[str] = {
        row[0]
        for row in conn.execute("SELECT node_id FROM code_nodes").fetchall()
    }

    direct_code_seeds: list[str] = []
    resolutions: list[dict] = []          # [{"doc_id": x, "code_ids": [...]}]
    doc_to_code_map: dict[str, list[str]] = {}  # for synthesis backlinks

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
            doc_to_code_map[nid] = code_ids
        # If no rows: doc chunk is stranded (no candidates above threshold).
        # Log is handled by runner; we silently skip here.

    return resolutions, doc_to_code_map, direct_code_seeds
