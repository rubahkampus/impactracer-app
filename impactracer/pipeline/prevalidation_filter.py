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

    B4 fix: uses ``raw_reranker_score`` (absolute cross-encoder quality)
    rather than ``reranker_score`` (min-max normalized, rank-relative).
    Min-max normalization maps the worst candidate in a batch to 0.0
    regardless of its absolute quality, making a normalized score floor
    a rank filter rather than a quality filter.  Using the raw score
    preserves the intended semantics: "drop candidates the model considers
    low quality", not "drop the bottom N% of candidates".

    Falls back to ``reranker_score`` when ``raw_reranker_score`` is 0.0
    (V0–V2 where the reranker was not run, so raw is always 0).

    Blueprint §4 Step 3.5.
    """
    def _effective_score(c: Candidate) -> float:
        # Phase 2.6 (F-NEW-1/F-5): use raw_reranker_score as the authoritative
        # quality signal. If raw_reranker_score is None-equivalent (0.0, which
        # is the default for V0-V2 where the reranker was not run), return 0.0
        # directly — do NOT fall back to the normalized reranker_score.
        #
        # The previous fall-back to reranker_score was the bug: min-max
        # normalization always maps the lowest candidate in a batch to 0.0,
        # so the normalized score of a genuinely poor candidate could equal
        # 0.0 and still pass a 0.0 floor — the gate was permanently disabled.
        # With raw_reranker_score, a score of 0.0 means "reranker not run"
        # (V0-V2), which correctly passes a 0.0 default threshold.
        # For V3+ where the reranker IS run, raw_reranker_score is the
        # absolute cross-encoder logit — negative values indicate poor quality
        # and will fail a properly calibrated positive threshold (e.g. 0.15).
        if c.raw_reranker_score != 0.0:
            return c.raw_reranker_score
        # Reranker not run (V0-V2): treat as 0.0 so everything passes default floor.
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

    # Phase 2.8 (E-NEW-1): batch SQLite query for all doc chunk top-1 resolutions.
    # The previous implementation issued one SELECT per doc candidate in a loop —
    # N sequential round-trips for N doc chunks. With 15 candidates this is 15
    # sequential queries. Replace with a single WHERE doc_id IN (...) batch query
    # that returns all top-1 mappings at once, then build a lookup dict.
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
    Phase B (Crucible Fix 9): drop CODE candidates from files whose fraction
    of total code candidates exceeds plausibility_gate_density_threshold.
    Named-entry-point matches are exempt. The previous per-file admission
    cap (max_per_file=2) was removed because it arbitrarily rejected genuine
    multi-symbol impacts; density-based protection alone catches the
    pathological "one over-indexed file dominates retrieval" case.

    B3 fix: doc chunk candidates are ALWAYS exempt from the file-density cap.

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

    N3 fix: the previous implementation checked BOTH ``p in name_lower`` AND
    ``name_lower in p``, creating a false positive trap: a short name like
    "get" would match a pattern like "getCommission" (name_lower in p direction)
    even though "get" is not what the CR is targeting.

    The semantically correct direction is: the pattern is a substring of the
    name.  E.g. pattern "createListing" should match the function named
    "createListingHandler" but NOT match any function whose name happens to be
    a prefix of "createListing".

    Blueprint §4 Step 3.7 Phase B: named_entry_points are explicit patterns
    the CR describes — use them as search strings against node names.
    """
    name_lower = name.lower()
    return any(p in name_lower for p in named_patterns)
