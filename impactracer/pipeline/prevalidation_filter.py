"""Pre-validation deterministic gates (FR-C4).

Two sub-steps executed in order:

- Step 3.6 - Cross-collection semantic deduplication.
- Step 3.7 - Layer-aware affinity rescoring + file-density plausibility
             gate with named_entry_points exemption.

(Step 3.5, the reranker score floor, is RETIRED — found strictly inert by a
42-CR two-repo ablation. ``step_3_5_score_filter`` remains for archival only
and is never invoked.)

No LLM calls. Purely deterministic.
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
    """Apply Steps 3.6, 3.7 in order (semantic dedup, plausibility+affinity).

    NOTE: the Step 3.5 score floor is RETIRED. A 42-CR two-repo ablation
    (floor on vs off, all else fixed) found it strictly inert — it changed no
    candidate on any CR because the calibrated threshold admits all normalized
    cross-encoder scores. ``step_3_5_score_filter`` is retained below for
    archival only and is never invoked; ``enable_score_floor`` is accepted for
    signature/back-compat but has no effect. See the retirement note on
    ``step_3_5_score_filter``.
    """
    if enable_dedup:
        attach_doc_contexts = not bool(getattr(settings, "code_only_mode", False))
        candidates = step_3_6_semantic_dedup(
            candidates, conn, attach_doc_contexts=attach_doc_contexts
        )
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
    cr_interp: CRInterpretation | None = None,
    anchor_boost: float = 0.10,
) -> list[Candidate]:
    """RETIRED / ARCHIVAL — not called by the pipeline. A 42-CR two-repo
    ablation found the score floor strictly inert (no candidate changed on any
    CR at the calibrated threshold). Kept for reference; do not re-wire without
    re-evaluating. ``apply_prevalidation_gates`` no longer invokes this.

    Drop candidates whose absolute cross-encoder score is below threshold.

    Uses raw_reranker_score (absolute logit) not the min-max normalized
    reranker_score — normalization maps the worst candidate to 0.0 regardless
    of quality, making a normalized floor a rank filter, not a quality filter.

    Falls back to 0.0 for V0–V2 where the reranker was not run.

    Anchor priming: when ``cr_interp.anchor_candidates`` is non-empty, the
    candidate's effective score receives an additive boost of ``anchor_boost``
    if its ``name`` substring-matches any anchor candidate (case-insensitive).
    The boost is SOFT: a candidate whose boosted score is still below the
    floor is dropped. ``c.anchor_boost_applied`` is set True for trace
    inspection. Hallucinated anchor identifiers therefore cannot inject
    irrelevant candidates above the floor by themselves; the validator
    chain remains the source of truth.

    Blueprint §4 Step 3.5.
    """
    anchor_patterns: list[str] = []
    if cr_interp is not None and cr_interp.anchor_candidates:
        anchor_patterns = [p.lower() for p in cr_interp.anchor_candidates if p]

    def _effective_score(c: Candidate) -> float:
        base = c.raw_reranker_score if c.raw_reranker_score != 0.0 else 0.0
        if anchor_patterns and _matches_any_named(c.name, anchor_patterns):
            c.anchor_boost_applied = True
            return base + anchor_boost
        return base

    # Candidates pinned by named_entry_points
    # bypass the score floor. The cross-encoder body-similarity score is
    # not a meaningful quality signal for files the CR text explicitly
    # names; the pin is a stronger signal of intent.
    return [
        c for c in candidates
        if c.pinned_by_named_entry or _effective_score(c) >= threshold
    ]


def step_3_6_semantic_dedup(
    candidates: list[Candidate],
    conn: sqlite3.Connection,
    attach_doc_contexts: bool = True,
) -> list[Candidate]:
    """Merge doc chunks whose top-1 code resolution is already in the list.

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

        # Pinned doc chunks survive dedup.
        if c.pinned_by_named_entry:
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

        # Pinned candidates bypass density gate.
        if c.pinned_by_named_entry:
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
