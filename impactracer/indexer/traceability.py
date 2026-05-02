"""Layer-weighted doc<->code traceability precomputation (FR-A7).

Reference: master_blueprint.md §3.7.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np

from impactracer.shared.constants import layer_compat


def compute_and_store(
    code_vecs: dict[str, np.ndarray],
    doc_vecs: dict[str, np.ndarray],
    code_meta: dict[str, dict[str, Any]],
    doc_meta: dict[str, dict[str, Any]],
    top_k: int,
    min_similarity: float,
    conn: sqlite3.Connection,
) -> int:
    """Compute and persist the top-K layer-weighted similarity pairs.

    Algorithm (blueprint §3.7):
      1. L2-normalize all vectors.
      2. Compute cosine matrix via single matmul.
      3. For each code node, multiply row by per-doc layer_compat().
      4. Retain top-K pairs per code node above ``min_similarity``.
      5. DELETE existing rows for these code IDs, then INSERT new ones.

    Returns the number of pairs stored.
    """
    if not code_vecs or not doc_vecs:
        return 0

    code_ids = list(code_vecs.keys())
    doc_ids = list(doc_vecs.keys())

    # Stack into matrices (N_code x D) and (N_doc x D)
    code_matrix = np.stack([code_vecs[cid] for cid in code_ids]).astype(np.float32)
    doc_matrix = np.stack([doc_vecs[did] for did in doc_ids]).astype(np.float32)

    # Step 1: L2-normalize
    code_norms = np.linalg.norm(code_matrix, axis=1, keepdims=True)
    doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    code_norms = np.where(code_norms == 0, 1.0, code_norms)
    doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)
    code_matrix /= code_norms
    doc_matrix /= doc_norms

    # Step 2: cosine similarity matrix (N_code x N_doc)
    cos_matrix = code_matrix @ doc_matrix.T

    # Build per-doc layer_compat weight vector for each code node
    # shape: (N_code x N_doc)
    compat_matrix = np.empty((len(code_ids), len(doc_ids)), dtype=np.float32)
    for i, cid in enumerate(code_ids):
        classification = code_meta[cid].get("file_classification")
        for j, did in enumerate(doc_ids):
            chunk_type = doc_meta[did].get("chunk_type", "General")
            compat_matrix[i, j] = layer_compat(classification, chunk_type)

    # Step 3: weighted scores
    weighted = cos_matrix * compat_matrix

    # Step 4 + 5: collect pairs, delete stale rows, insert
    #
    # Dual-direction top-K pass (V1 fix):
    #   Forward pass  — for each CODE node, retain its top-K doc chunks.
    #   Reverse pass  — for each DOC chunk, retain its top-K code nodes.
    # Union of both passes ensures that low-LAYER_COMPAT chunk types
    # (NFR, General) are never silently squeeze-out when they have a
    # genuinely high cosine similarity to some code node.
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[str, str, float]] = []

    # --- Forward pass: per-code-node top-K ---
    for i, cid in enumerate(code_ids):
        row = weighted[i]
        order = np.argsort(row)[::-1]
        count = 0
        for j in order:
            score = float(row[j])
            if score < min_similarity:
                break
            key = (cid, doc_ids[j])
            if key not in seen:
                rows.append((cid, doc_ids[j], score))
                seen.add(key)
            count += 1
            if count >= top_k:
                break

    # --- Reverse pass: per-doc-chunk top-K ---
    for j, did in enumerate(doc_ids):
        col = weighted[:, j]
        order = np.argsort(col)[::-1]
        count = 0
        for i in order:
            score = float(col[i])
            if score < min_similarity:
                break
            key = (code_ids[i], did)
            if key not in seen:
                rows.append((code_ids[i], did, score))
                seen.add(key)
            count += 1
            if count >= top_k:
                break

    cur = conn.cursor()
    # DELETE existing rows for all code IDs being (re)computed
    placeholders = ",".join("?" * len(code_ids))
    cur.execute(
        f"DELETE FROM doc_code_candidates WHERE code_id IN ({placeholders})",
        code_ids,
    )
    cur.executemany(
        "INSERT OR REPLACE INTO doc_code_candidates "
        "(code_id, doc_id, weighted_similarity_score) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)
