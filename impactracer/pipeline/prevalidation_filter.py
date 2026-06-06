"""Pre-validation deterministic gate (FR-C4): semantic dedup.

The only active gate. Merges a doc chunk into its resolved code candidate,
carrying the doc's section text into ``merged_doc_contexts`` as Business Context
for the LLM #2 prompt (V4+), and prevents the same impact being double-counted
as both a doc and a code candidate. Retained on engineering grounds (measured
within-noise on F1, not an accuracy claim).

A contribution study found the former score-floor and plausibility gates inert
/ net-negative; they have been removed (see git history / stage3_contribution_study.md).
"""

from __future__ import annotations

import sqlite3

from loguru import logger

from impactracer.shared.models import Candidate


def apply_prevalidation_gates(
    candidates: list[Candidate],
    settings: object,
    conn: sqlite3.Connection,
    enable_dedup: bool = True,
) -> list[Candidate]:
    """Pre-validation gate: Step 2.2 semantic dedup (the only active gate).

    Merges each doc chunk into its resolved code candidate, attaching the doc's
    section text as Business Context for LLM #2 and avoiding doc/code
    double-counting. Active on all variants when ``enable_dedup`` is True.
    """
    if enable_dedup:
        attach_doc_contexts = not bool(getattr(settings, "code_only_mode", False))
        candidates = semantic_dedup(
            candidates, conn, attach_doc_contexts=attach_doc_contexts
        )
        logger.info("[gates] Post-2.2 (semantic dedup): {} candidates", len(candidates))
    return candidates


def semantic_dedup(
    candidates: list[Candidate],
    conn: sqlite3.Connection,
    attach_doc_contexts: bool = True,
) -> list[Candidate]:
    """ACTIVE — retained on engineering/robustness grounds (measured
    within-noise on F1, NOT an accuracy claim): provides LLM #2 Business
    Context and prevents doc/code double-counting. See apply_prevalidation_gates.

    Merge doc chunks whose top-1 code resolution is already in the list.

    For each doc_chunks candidate, look up top-1 code node from
    doc_code_candidates. If that code_id is already a candidate, append
    the doc chunk's ID to the code candidate's merged_doc_ids and also
    store the doc chunk's (section_title, text) in merged_doc_contexts so
    the LLM #2 validator prompt can inject it as "Business Context" (B1).

    Code-only evaluation mode (``attach_doc_contexts=False``): the merge
    behaviour is preserved (so the doc chunk is still dropped from the
    candidate list to avoid duplicate scoring), but the
    ``merged_doc_contexts`` field on the target code candidate is left
    empty. LLM-2 therefore receives no Business Context block. Used by
    the Sprint-24 supervisor experiment to isolate the documentation
    contribution.

    Blueprint §4 Step 2.2.
    """
    # Build index of current code node IDs for O(1) lookup
    code_candidate_idx: dict[str, Candidate] = {
        c.node_id: c for c in candidates if c.collection == "code_units"
    }

    doc_candidates = [c for c in candidates if c.collection == "doc_chunks"]
    doc_top1_map: dict[str, str] = {}  # doc_id → top-1 code_id

    if doc_candidates:
        doc_ids = [c.node_id for c in doc_candidates]
        placeholders = ",".join("?" * len(doc_ids))
        # Fetch all rows for these doc_ids, ordered by score desc per doc_id.
        rows = conn.execute(
            f"SELECT doc_id, code_id FROM doc_code_candidates "
            f"WHERE doc_id IN ({placeholders}) "
            f"ORDER BY doc_id, weighted_similarity_score DESC",
            doc_ids,
        ).fetchall()
        # Keep only the first (highest-score) code_id per doc_id.
        for doc_id, code_id in rows:
            if doc_id not in doc_top1_map:
                doc_top1_map[doc_id] = code_id

    merged: set[str] = set()
    result: list[Candidate] = []

    for c in candidates:
        if c.collection != "doc_chunks":
            result.append(c)
            continue

        top1_code = doc_top1_map.get(c.node_id)
        if top1_code is not None and top1_code in code_candidate_idx:
            # Merge: append this doc's ID to the code candidate and drop the doc candidate
            target_code = code_candidate_idx[top1_code]
            target_code.merged_doc_ids.append(c.node_id)

            if attach_doc_contexts:
                # B1: carry (section_title, text) so the validator prompt can show
                # "Business Context" explaining WHY this code node is relevant.
                section_title = c.name or c.node_id
                source_text = c.text_snippet or ""
                target_code.merged_doc_contexts.append((section_title, source_text))

            merged.add(c.node_id)
            logger.debug(
                "[gates 2.2] Merged doc {} -> code {} (merged_doc_ids={})",
                c.node_id, top1_code, target_code.merged_doc_ids,
            )
        else:
            result.append(c)

    if merged:
        logger.info("[gates 2.2] Merged {} doc chunks into existing code candidates", len(merged))

    return result
