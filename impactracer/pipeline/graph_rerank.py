"""Apex Crucible Proposal C — graph-aware label-propagation rerank.

Inserted between Step 3 (cross-encoder rerank) and Step 3.5 (gates).

ALGORITHM (2-iteration Personalized PageRank approximation):

    personalization[v] = max(0, cross_encoder_score[v])  for v in top-N seeds
                       = 0                                otherwise

    iter 1:
        prop1[v] = sum over edges (u → v) in structural graph:
                     edge_weight(edge_type) * personalization[u]
    iter 2:
        prop2[v] = sum over edges (u → v) in structural graph:
                     edge_weight(edge_type) * prop1[u]

    graph_score[v] = (prop1[v] + 0.5 * prop2[v])  # downweight 2-hop

After min-max normalising graph_score across the candidate pool, blend with
the cross-encoder score:

    blended_score[v] = alpha * cross_encoder_norm[v] + (1-alpha) * graph_norm[v]

Two operating modes:

    Mode A (re-rank existing pool):
        Re-score the existing RRF/cross-encoder pool by the blended score.
        Helps cases where GT entities are in the pool but ranked below the
        top-K cross-encoder cutoff because cross-encoder doesn't see their
        structural dependency on a confirmed seed.

    Mode B (graph-add):
        Discover NEW candidates by computing graph_score over the full
        structural graph (not just the candidate subgraph). The top-N
        previously-unseen nodes whose graph_score exceeds a threshold are
        ADDED to the candidate pool (with a synthetic cross-encoder score
        equal to the pool median). Helps cases like CR-02 where the GT
        files are structurally adjacent to confirmed seeds but never reach
        the RRF pool because the CR text never mentions them.

Edge weights come from a curated subset of EDGE_CONFIG semantics:
TYPED_BY (0.8), CALLS (0.9), RENDERS (0.8), CONTAINS (0.6), IMPORTS (0.5),
INHERITS (0.7), IMPLEMENTS (0.7), HOOK_DEPENDS_ON (0.6), DEFINES_METHOD
(0.6), PASSES_CALLBACK (0.5), CLIENT_API_CALLS (0.7), DYNAMIC_IMPORT (0.4),
FIELDS_ACCESSED (0.7), DEPENDS_ON_EXTERNAL (0.0 — externals never propagate).

Reference: master_blueprint.md is silent on graph rerank; this is a
post-W2 Apex Crucible extension justified in implementation_report.md
Sprint 15. The Apex Crucible Report Section 2 Proposal C is the original
design source.
"""

from __future__ import annotations

import networkx as nx
from loguru import logger

from impactracer.shared.models import Candidate


# Edge weights for label propagation. Higher = stronger structural signal.
# Chosen to match EDGE_CONFIG semantics:
#   - Contract edges (TYPED_BY, IMPLEMENTS, INHERITS) are high — they directly
#     bind callers to the changed contract.
#   - Behavioural edges (CALLS, HOOK_DEPENDS_ON) are high.
#   - Composition edges (IMPORTS, CONTAINS) are medium — co-location only.
#   - DEPENDS_ON_EXTERNAL is 0 — third-party deps never propagate.
_EDGE_WEIGHTS: dict[str, float] = {
    "TYPED_BY": 0.8,
    "CALLS": 0.9,
    "RENDERS": 0.8,
    "CONTAINS": 0.6,
    "IMPORTS": 0.5,
    "INHERITS": 0.7,
    "IMPLEMENTS": 0.7,
    "HOOK_DEPENDS_ON": 0.6,
    "DEFINES_METHOD": 0.6,
    "PASSES_CALLBACK": 0.5,
    "CLIENT_API_CALLS": 0.7,
    "DYNAMIC_IMPORT": 0.4,
    "FIELDS_ACCESSED": 0.7,
    "DEPENDS_ON_EXTERNAL": 0.0,
}

_HOP2_DECAY = 0.5  # weight of 2-hop signal relative to 1-hop


def _propagate_one_step(
    scores: dict[str, float],
    graph: nx.MultiDiGraph,
) -> dict[str, float]:
    """One label-propagation hop. Edges are traversed in BOTH directions
    (impact can flow from caller to callee or vice versa depending on edge
    semantics; we don't bake a direction policy here — the edge_weight
    captures relevance).
    """
    next_scores: dict[str, float] = {}
    seeds = [n for n, s in scores.items() if s > 0.0]
    for src in seeds:
        if src not in graph:
            continue
        src_score = scores[src]
        # Forward edges (src → tgt)
        for _, tgt, ed in graph.out_edges(src, data=True):
            w = _EDGE_WEIGHTS.get(ed.get("edge_type", ""), 0.0)
            if w <= 0.0:
                continue
            next_scores[tgt] = next_scores.get(tgt, 0.0) + w * src_score
        # Reverse edges (pred → src means src receives from pred; but here we
        # want to push src's mass to predecessors too — impact can flow
        # bidirectionally along the same edge type, with the SAME weight)
        for pred, _, ed in graph.in_edges(src, data=True):
            w = _EDGE_WEIGHTS.get(ed.get("edge_type", ""), 0.0)
            if w <= 0.0:
                continue
            next_scores[pred] = next_scores.get(pred, 0.0) + w * src_score
    return next_scores


def graph_rerank(
    candidates: list[Candidate],
    graph: nx.MultiDiGraph,
    *,
    alpha: float = 0.7,
    iterations: int = 2,
    personalization_top_n: int = 5,
    add_top_n: int = 10,
    add_min_score: float = 0.10,
    code_meta_by_id: dict[str, dict] | None = None,
) -> list[Candidate]:
    """Re-score the candidate pool by blending cross-encoder + graph propagation.

    Mutates each candidate's ``reranker_score`` (and ``raw_reranker_score``)
    to the blended value. Optionally appends up to ``add_top_n`` new
    candidates discovered via mode B (graph-add).

    Returns the merged candidate list (un-truncated). Caller is responsible
    for the final top-K cut.

    Args:
        candidates: existing pool after cross-encoder rerank (already has
            non-zero ``raw_reranker_score``).
        graph: the full structural graph from build_graph_from_sqlite.
        alpha: weight on cross-encoder; (1-alpha) on graph signal.
        iterations: label-propagation rounds (default 2).
        personalization_top_n: top-N candidates seed the personalization
            vector. Their cross-encoder scores are the "mass" injected
            into the graph.
        add_top_n: mode B — number of new candidates to discover and add.
            Set to 0 to disable mode B (rerank-only).
        add_min_score: mode B — minimum normalized graph_score for an
            added candidate to be admitted.
        code_meta_by_id: optional metadata to hydrate mode-B-added candidates.
            If None, mode B is disabled (we need metadata to construct a
            Candidate dataclass).
    """
    if not candidates:
        return candidates
    if iterations < 1:
        return candidates

    # Snapshot cross-encoder scores for blending. Use raw_reranker_score
    # (pre-normalization) so the blend is on absolute quality.
    ce_score: dict[str, float] = {
        c.node_id: max(0.0, c.raw_reranker_score) for c in candidates
    }

    # Personalization: top-N candidates by cross-encoder score.
    top_seeds = sorted(
        candidates,
        key=lambda c: c.raw_reranker_score,
        reverse=True,
    )[:personalization_top_n]
    personalization: dict[str, float] = {
        c.node_id: max(0.0, c.raw_reranker_score) for c in top_seeds
    }
    # Normalize personalization to sum=1 (so multiple seeds don't accumulate
    # unfairly relative to a 1-seed propagation).
    total_pers = sum(personalization.values())
    if total_pers > 0:
        personalization = {k: v / total_pers for k, v in personalization.items()}

    # Run propagation.
    current = dict(personalization)
    accumulated: dict[str, float] = {k: v for k, v in current.items()}
    hop_weight = 1.0
    for it in range(iterations):
        hop_weight = hop_weight * _HOP2_DECAY if it > 0 else hop_weight
        current = _propagate_one_step(current, graph)
        for k, v in current.items():
            accumulated[k] = accumulated.get(k, 0.0) + hop_weight * v

    # Remove the personalization mass from accumulated[seed] so seeds aren't
    # double-counted (they're already in cross-encoder score).
    for seed_id in personalization:
        accumulated[seed_id] = accumulated.get(seed_id, 0.0) - personalization[seed_id]
        if accumulated[seed_id] < 0:
            accumulated[seed_id] = 0.0

    # Mode A: re-score existing pool members.
    # Min-max normalize cross-encoder and graph scores across the pool.
    pool_ce = list(ce_score.values())
    pool_graph = [accumulated.get(c.node_id, 0.0) for c in candidates]
    ce_min, ce_max = min(pool_ce), max(pool_ce)
    g_min, g_max = min(pool_graph), max(pool_graph)
    ce_span = ce_max - ce_min if ce_max > ce_min else 1.0
    g_span = g_max - g_min if g_max > g_min else 1.0

    def _norm_ce(s: float) -> float:
        return (s - ce_min) / ce_span
    def _norm_g(s: float) -> float:
        return (s - g_min) / g_span

    for c in candidates:
        ce_n = _norm_ce(ce_score.get(c.node_id, 0.0))
        g_n = _norm_g(accumulated.get(c.node_id, 0.0))
        blended = alpha * ce_n + (1 - alpha) * g_n
        c.raw_reranker_score = blended
        c.reranker_score = blended

    # Mode B: graph-add. Discover graph-adjacent nodes not in the pool whose
    # propagated score is above threshold.
    added: list[Candidate] = []
    if add_top_n > 0 and code_meta_by_id:
        pool_ids = {c.node_id for c in candidates}
        # Find top accumulated nodes not in the pool, restricted to nodes
        # that have code_meta (i.e. real code_nodes, not external packages).
        external_neighbors = []
        for nid, score in accumulated.items():
            if nid in pool_ids:
                continue
            if nid not in code_meta_by_id:
                continue
            # Skip External packages (no useful Candidate metadata).
            meta = code_meta_by_id[nid]
            if meta.get("node_type") == "ExternalPackage":
                continue
            external_neighbors.append((nid, score))

        external_neighbors.sort(key=lambda kv: kv[1], reverse=True)
        median_pool_blended = (
            sorted(c.raw_reranker_score for c in candidates)[len(candidates) // 2]
            if candidates else 0.0
        )

        for nid, raw_g_score in external_neighbors[:add_top_n]:
            g_n = _norm_g(raw_g_score)
            if g_n < add_min_score:
                continue
            meta = code_meta_by_id[nid]
            # Construct a Candidate with the median pool blended score as a
            # synthetic cross-encoder proxy. This puts the new candidate
            # mid-pool — high enough to survive the gates, low enough that
            # cross-encoder-confirmed candidates still rank above it.
            name = nid.split("::")[-1] if "::" in nid else nid
            src_code = meta.get("source_code") or ""
            new_cand = Candidate(
                node_id=nid,
                node_type=meta.get("node_type") or "Function",
                collection="code_units",
                rrf_score=median_pool_blended * 0.5,  # half the median for ranking
                reranker_score=median_pool_blended,
                raw_reranker_score=median_pool_blended,
                file_path=meta.get("file_path") or "",
                file_classification=meta.get("file_classification"),
                name=name,
                text_snippet=src_code[:1200],
                internal_logic_abstraction=meta.get("internal_logic_abstraction"),
            )
            added.append(new_cand)

        if added:
            logger.info(
                "[graph_rerank] Mode B: added {} graph-discovered candidates "
                "(top accumulated scores; min_norm={:.2f})",
                len(added), add_min_score,
            )

    logger.info(
        "[graph_rerank] Mode A: re-scored {} pool candidates "
        "(alpha={:.2f}, iters={}, pers_top_n={})",
        len(candidates), alpha, iterations, personalization_top_n,
    )

    return candidates + added
