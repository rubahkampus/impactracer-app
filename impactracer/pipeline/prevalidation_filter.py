"""Pre-validation deterministic gates (FR-C4).

Three sub-steps executed in order:

- Step 3.5 - Reranker score floor (configurable via settings.min_reranker_score_for_admission).
- Step 3.6 - Cross-collection semantic deduplication.
- Step 3.7 - Layer-aware affinity rescoring + file-density plausibility
             gate with named_entry_points exemption.

No LLM calls. Purely deterministic.

Blueprint: master_blueprint.md §4 Steps 3.5–3.7.
"""

from __future__ import annotations

import sqlite3
from collections import Counter

from loguru import logger

from impactracer.shared.constants import layer_compat
from impactracer.shared.models import Candidate, CRInterpretation


def apply_prevalidation_gates(
    candidates: list[Candidate],
    cr_interp: CRInterpretation,
    settings: object,
    conn: sqlite3.Connection,
    enable_score_floor: bool = True,
    enable_dedup: bool = True,
    enable_plausibility: bool = True,
) -> list[Candidate]:
    """Apply Steps 3.5, 3.6, 3.7 in order.

    Blueprint §4 Steps 3.5–3.7. Step 3.5 is skipped when enable_score_floor
    is False (V0–V3 maximum inclusivity mandate). Steps 3.6 and 3.7 are
    independently gated by their flags.
    """
    if enable_score_floor:
        threshold = settings.min_reranker_score_for_validation  # type: ignore[attr-defined]
        candidates = step_3_5_score_filter(candidates, threshold)
        logger.info("[gates] Post-3.5 (score floor ≥{}): {} candidates", threshold, len(candidates))
    else:
        logger.debug("[gates] Step 3.5 DISABLED (enable_score_floor=False)")

    if enable_dedup:
        candidates = step_3_6_semantic_dedup(candidates, conn)
        logger.info("[gates] Post-3.6 (semantic dedup): {} candidates", len(candidates))
    else:
        logger.debug("[gates] Step 3.6 DISABLED (enable_dedup=False)")

    if enable_plausibility:
        candidates = step_3_7_plausibility_and_affinity(candidates, cr_interp, settings)
        logger.info("[gates] Post-3.7 (plausibility+affinity): {} candidates", len(candidates))
    else:
        logger.debug("[gates] Step 3.7 DISABLED (enable_plausibility=False)")

    return candidates


def step_3_5_score_filter(
    candidates: list[Candidate],
    threshold: float,
) -> list[Candidate]:
    """Drop candidates whose absolute cross-encoder score is below threshold.

    Uses raw_reranker_score (absolute logit) not the min-max normalized
    reranker_score — normalization maps the worst candidate to 0.0 regardless
    of quality, making a normalized floor a rank filter, not a quality filter.

    Falls back to 0.0 for V0–V2 where the reranker was not run.

    Blueprint §4 Step 3.5.
    """
    def _effective_score(c: Candidate) -> float:
        if c.raw_reranker_score != 0.0:
            return c.raw_reranker_score
        return 0.0

    return [c for c in candidates if _effective_score(c) >= threshold]


def step_3_6_semantic_dedup(
    candidates: list[Candidate],
    conn: sqlite3.Connection,
) -> list[Candidate]:
    """Merge doc chunks whose top-1 code resolution is already in the list.

    For each doc_chunks candidate, look up top-1 code node from
    doc_code_candidates. If that code_id is already a candidate, append
    the doc chunk's ID to the code candidate's merged_doc_ids and also
    store the doc chunk's (section_title, text) in merged_doc_contexts so
    the LLM #2 validator prompt can inject it as "Business Context" (B1).

    Blueprint §4 Step 3.6.
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

            # B1: carry (section_title, text) so the validator prompt can show
            # "Business Context" explaining WHY this code node is relevant.
            section_title = c.name or c.node_id
            source_text = c.text_snippet or ""
            target_code.merged_doc_contexts.append((section_title, source_text))

            merged.add(c.node_id)
            logger.debug(
                "[gates 3.6] Merged doc {} -> code {} (merged_doc_ids={})",
                c.node_id, top1_code, target_code.merged_doc_ids,
            )
        else:
            result.append(c)

    if merged:
        logger.info("[gates 3.6] Merged {} doc chunks into existing code candidates", len(merged))

    return result


def step_3_7_plausibility_and_affinity(
    candidates: list[Candidate],
    cr_interp: CRInterpretation,
    settings: object,
) -> list[Candidate]:
    """Rescore by layer affinity, then enforce density-only plausibility gate.

    Phase A: multiply reranker_score by _affinity_factor(c, cr_interp).
    Phase B: drop CODE candidates from files whose fraction of total code
    candidates exceeds plausibility_gate_density_threshold. Named-entry-point
    matches are exempt. Doc chunk candidates always pass Phase B.

    Blueprint §4 Step 3.7.
    """
    density_threshold: float = settings.plausibility_gate_density_threshold  # type: ignore[attr-defined]

    # Phase A: affinity rescoring (all candidates, including doc chunks)
    for c in candidates:
        factor = _affinity_factor(c, cr_interp)
        c.reranker_score = c.reranker_score * factor

    # Re-sort descending after rescoring
    candidates = sorted(candidates, key=lambda c: c.reranker_score, reverse=True)

    if not candidates:
        return candidates

    # B3: only count CODE candidates toward the density denominator.
    code_candidates = [c for c in candidates if c.collection == "code_units"]
    total_code = len(code_candidates)

    named_patterns = [p.lower() for p in cr_interp.named_entry_points]

    if total_code == 0:
        return candidates

    file_density = Counter(c.file_path for c in code_candidates)
    flooded_files: set[str] = {
        fp for fp, count in file_density.items()
        if count / total_code > density_threshold
    }

    if not flooded_files:
        return candidates

    result: list[Candidate] = []
    for c in candidates:
        if c.collection == "doc_chunks":
            result.append(c)
            continue

        if c.file_path not in flooded_files:
            result.append(c)
            continue

        if named_patterns and _matches_any_named(c.name, named_patterns):
            result.append(c)
            continue

        logger.debug(
            "[gates 3.7] Dropped {} (file {} contains {:.0%} of code "
            "candidates, exceeds density threshold {:.0%})",
            c.node_id, c.file_path,
            file_density[c.file_path] / total_code, density_threshold,
        )

    return result


def _affinity_factor(c: Candidate, cr_interp: CRInterpretation) -> float:
    """Compute layer-affinity multiplier for a candidate.

    Doc candidates: 1.0 if chunk_type is in the affected layer's chunk types,
    else 0.7.
    Code candidates: layer_compat(file_classification, primary_chunk_type).

    Blueprint §4 Step 3.7 Phase A.
    """
    affected_layers = cr_interp.affected_layers

    if c.collection == "doc_chunks":
        # Map affected_layers to expected chunk types
        expected_chunk_types: set[str] = set()
        if "requirement" in affected_layers:
            expected_chunk_types.update(["FR", "NFR"])
        if "design" in affected_layers:
            expected_chunk_types.add("Design")
        # "code" does not correspond to a specific chunk type; use General as fallback
        if "code" in affected_layers:
            expected_chunk_types.add("General")

        return 1.0 if c.chunk_type in expected_chunk_types else 0.7

    # Code candidates
    primary_chunk_type = _primary_chunk_type(affected_layers)
    return layer_compat(c.file_classification, primary_chunk_type)


def _primary_chunk_type(affected_layers: list[str]) -> str:
    """Derive primary chunk type for layer_compat lookup.

    Blueprint §4 Step 3.7 Phase A:
    "FR" if "code" or "requirement" in layers, "Design" if "design", else "General".
    """
    if "code" in affected_layers or "requirement" in affected_layers:
        return "FR"
    if "design" in affected_layers:
        return "Design"
    return "General"


def _matches_any_named(name: str, named_patterns: list[str]) -> bool:
    """True if any named pattern is a substring of name (case-insensitive).

    Direction: pattern ∈ name (not name ∈ pattern). Pattern "createListing"
    matches "createListingHandler" but not the reverse — avoids false positives
    from short generic names like "get" matching long patterns.

    Blueprint §4 Step 3.7 Phase B.
    """
    name_lower = name.lower()
    return any(p in name_lower for p in named_patterns)
