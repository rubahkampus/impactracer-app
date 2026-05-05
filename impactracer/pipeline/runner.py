"""Online pipeline orchestrator (nine steps, five LLM invocations).

Invoked by :func:`impactracer.cli.analyze`. Consumes a :class:`VariantFlags`
instance so the same code powers both full V7 analysis and the ablation
harness variants V0 through V6.

Sprint 8 implements Steps 0, 1, 2, 3, 8, 9 (V0–V3 capability).
Steps 4–7 are passthrough stubs filled in Sprints 9–10.

Reference: master_blueprint.md §4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
from loguru import logger

from impactracer.indexer.embedder import Embedder
from impactracer.indexer.reranker import Reranker
from impactracer.persistence.chroma_client import get_client, init_collections
from impactracer.persistence.sqlite_client import connect
from impactracer.pipeline.context_builder import (
    build_context,
    fetch_backlinks,
    fetch_snippets,
)
from impactracer.pipeline.interpreter import interpret_cr
from impactracer.pipeline.llm_client import LLMClient
from impactracer.pipeline.prevalidation_filter import apply_prevalidation_gates
from impactracer.pipeline.retriever import (
    build_bm25_from_chroma,
    build_metadata_cache,
    hybrid_search,
)
from impactracer.pipeline.seed_resolver import resolve_doc_to_code
from impactracer.pipeline.synthesizer import synthesize_report
from impactracer.pipeline.traceability_validator import validate_trace_resolutions
from impactracer.pipeline.traversal_validator import validate_propagation
from impactracer.pipeline.validator import validate_sis_candidates_batched
from impactracer.shared.config import Settings
from impactracer.shared.constants import severity_for_chain
from impactracer.shared.models import (
    CISResult,
    Candidate,
    ImpactReport,
    ImpactedNode,
    NodeTrace,
)

if TYPE_CHECKING:
    from impactracer.evaluation.variant_flags import VariantFlags


@dataclass
class PipelineContext:
    """Loaded persistent stores, shared across all pipeline steps."""

    conn: Any
    doc_col: Any
    code_col: Any
    graph: Any
    doc_bm25: Any
    doc_bm25_ids: list[str]
    doc_meta_cache: dict[str, dict]   # ED-3: pre-cached doc chunk metadata
    code_bm25: Any
    code_bm25_ids: list[str]
    embedder: Any
    reranker: Any
    llm_client: Any
    variant_flags: Any  # VariantFlags


def _build_graph_from_sqlite(conn: Any) -> nx.MultiDiGraph:
    """Materialize the structural edge graph from SQLite (Step 0).

    Delegates to graph_bfs.build_graph_from_sqlite — kept here so
    load_pipeline_context does not import graph_bfs at module level
    (avoids circular import).
    """
    from impactracer.pipeline.graph_bfs import build_graph_from_sqlite
    return build_graph_from_sqlite(conn)


def load_pipeline_context(
    settings: Settings,
    variant_flags: "VariantFlags | None" = None,
    shared_embedder: Any = None,
    shared_reranker: Any = None,
    shared_llm_client: Any = None,
) -> PipelineContext:
    """Step 0: load every persistent dependency into a PipelineContext.

    Accepts optional pre-constructed shared_embedder, shared_reranker,
    shared_llm_client so the evaluation harness can reuse a single set of
    heavy objects across 160 runs instead of instantiating 160 copies (ED-6).

    Blueprint §4 Step 0.
    """
    from impactracer.evaluation.variant_flags import VariantFlags

    if variant_flags is None:
        variant_flags = VariantFlags.v7_full()

    t0 = time.perf_counter()
    logger.info("[runner] Loading pipeline context (variant={})", variant_flags.variant_id)

    conn = connect(settings.db_path)
    chroma_client = get_client(settings.chroma_path)
    doc_col, code_col = init_collections(chroma_client)

    graph = _build_graph_from_sqlite(conn)

    doc_bm25, doc_bm25_ids = build_bm25_from_chroma(doc_col)
    # ED-3: build metadata cache to avoid N+1 ChromaDB queries in BM25 filtering
    doc_meta_cache = build_metadata_cache(doc_col)

    code_bm25, code_bm25_ids = build_bm25_from_chroma(code_col)

    embedder = shared_embedder or Embedder(
        settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
    )
    reranker = shared_reranker or Reranker(settings.reranker_model)
    llm_client = shared_llm_client or LLMClient(settings)

    elapsed = time.perf_counter() - t0
    logger.info("[runner] Context loaded in {:.1f}s", elapsed)

    return PipelineContext(
        conn=conn,
        doc_col=doc_col,
        code_col=code_col,
        graph=graph,
        doc_bm25=doc_bm25,
        doc_bm25_ids=doc_bm25_ids,
        doc_meta_cache=doc_meta_cache,
        code_bm25=code_bm25,
        code_bm25_ids=code_bm25_ids,
        embedder=embedder,
        reranker=reranker,
        llm_client=llm_client,
        variant_flags=variant_flags,
    )


def _candidates_to_cis(candidates: list[Candidate]) -> CISResult:
    """Wrap retrieval candidates as SIS seeds in a CISResult (no BFS)."""
    sis_nodes: dict[str, NodeTrace] = {}
    for c in candidates:
        sis_nodes[c.node_id] = NodeTrace(
            depth=0,
            causal_chain=[],
            path=[c.node_id],
            source_seed=c.node_id,
            low_confidence_seed=False,
        )
    return CISResult(sis_nodes=sis_nodes, propagated_nodes={})


def _minimal_rejection_report(reason: str) -> ImpactReport:
    """Return a minimal ImpactReport for non-actionable CRs (ED-4: removed dead cr_text param)."""
    return ImpactReport(
        executive_summary=f"CR rejected: {reason}",
        impacted_nodes=[],
        documentation_conflicts=[],
        estimated_scope="terlokalisasi",
        analysis_mode="retrieval_only",
    )


def _compute_scope(cis: CISResult) -> str:
    """Deterministically compute estimated_scope from CIS node counts.

    N8 fix: estimated_scope must NOT be hallucinated by LLM #5 from a pruned
    synthesis context.  The synthesizer sees only the context-budget subset;
    counting nodes from the full CIS gives a provably correct scope that is
    reproducible and thesis-defensible.

    Thresholds (thesis-calibrated, matches Scope Literal values):
        terlokalisasi  : ≤ 5 total CIS nodes
        menengah       : 6–15 total CIS nodes
        ekstensif      : > 15 total CIS nodes

    Returns one of "terlokalisasi", "menengah", "ekstensif".
    """
    n_nodes = len(cis.combined())
    if n_nodes <= 5:
        return "terlokalisasi"
    if n_nodes <= 15:
        return "menengah"
    return "ekstensif"


def run_analysis(
    cr_text: str,
    settings: Settings,
    variant_flags: "VariantFlags | None" = None,
    shared_embedder: Any = None,
    shared_reranker: Any = None,
    shared_llm_client: Any = None,
) -> ImpactReport:
    """End-to-end online analysis for one CR.

    Sprint 8: implements V0–V3 paths (Steps 0,1,2,3,8,9).
    Steps 4–7 stubs pass through transparently when their flags are False.
    Sprints 9–10 fill in Steps 3.5–7.

    Blueprint §4.
    """
    from impactracer.evaluation.variant_flags import VariantFlags

    if variant_flags is None:
        variant_flags = VariantFlags.v7_full()

    t_start = time.perf_counter()
    ctx = load_pipeline_context(
        settings, variant_flags,
        shared_embedder=shared_embedder,
        shared_reranker=shared_reranker,
        shared_llm_client=shared_llm_client,
    )

    # ------------------------------------------------------------------
    # Step 1 — Interpret CR (LLM #1, always-on)
    # Schema constraints now live in interpreter.py SYSTEM_PROMPT (AV-1).
    # ------------------------------------------------------------------
    logger.info("[runner] Step 1: Interpret CR")
    cr_interp = interpret_cr(cr_text, ctx.llm_client)

    logger.info(
        "[runner] === INTERPRETER OUTPUT ===\n"
        "  is_actionable : {}\n"
        "  change_type   : {}\n"
        "  affected_layers: {}\n"
        "  primary_intent: {}\n"
        "  domain_concepts: {}\n"
        "  search_queries  : {}\n"
        "  named_entry_points: {}\n"
        "  out_of_scope_operations: {}",
        cr_interp.is_actionable,
        cr_interp.change_type,
        cr_interp.affected_layers,
        cr_interp.primary_intent,
        cr_interp.domain_concepts,
        cr_interp.search_queries,
        cr_interp.named_entry_points,
        cr_interp.out_of_scope_operations,
    )

    if not cr_interp.is_actionable:
        logger.info("[runner] CR is NOT actionable — short-circuiting")
        return _minimal_rejection_report(
            cr_interp.actionability_reason or "CR was not actionable"
        )

    # ------------------------------------------------------------------
    # Step 2 — Adaptive RRF Hybrid Search (FR-C1, FR-C2)
    # Returns top_k_rrf_pool (=50) candidates for cross-encoder input.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 2: Hybrid search (variant={})", variant_flags.variant_id)
    candidates = hybrid_search(cr_interp, ctx, settings)
    logger.info("[runner] Post-RRF pool: {}", len(candidates))

    # FF-5: guard zero-candidate case — synthesizing with zero nodes produces
    # a hallucinated report; return a clean rejection instead.
    if not candidates:
        logger.warning("[runner] Zero candidates — returning empty report")
        return _minimal_rejection_report("No candidates retrieved — check index and affected_layers")

    # ------------------------------------------------------------------
    # Step 3 — Cross-Encoder Rerank (FR-C3)
    # N4: Score each candidate against ALL search_queries and take the max
    # cross-encoder score.  Using only primary_intent systematically
    # penalises BM25-code candidates whose embed_text is identifier-level
    # (low semantic overlap with natural-language intent).
    # ------------------------------------------------------------------
    if variant_flags.enable_cross_encoder:
        logger.info("[runner] Step 3: Cross-encoder rerank (multi-query max scoring)")
        candidates = ctx.reranker.rerank_multi_query(
            cr_interp.search_queries,
            cr_interp.primary_intent,
            candidates,
            settings.max_admitted_seeds,
        )
        logger.info("[runner] Post-rerank candidates: {}", len(candidates))

        # B4: Snapshot raw cross-encoder scores BEFORE normalization so the
        # score floor gate operates on absolute quality, not rank-within-15.
        for c in candidates:
            c.raw_reranker_score = c.reranker_score

        # FF-2: min-max normalize to [0,1] for relative sorting inside the
        # pipeline (gates, context priority).  Raw scores preserved above.
        if len(candidates) > 1:
            min_s = min(c.reranker_score for c in candidates)
            max_s = max(c.reranker_score for c in candidates)
            if max_s > min_s:
                span = max_s - min_s
                for c in candidates:
                    c.reranker_score = (c.reranker_score - min_s) / span
            logger.debug(
                "[runner] Reranker scores normalized: min_raw={:.4f} max_raw={:.4f}",
                min_s, max_s,
            )
    else:
        # V0–V2: no reranker — cap at max_admitted_seeds from the RRF pool
        candidates = candidates[:settings.max_admitted_seeds]
        logger.info("[runner] Step 3: Cross-encoder DISABLED ({})", variant_flags.variant_id)

    # ------------------------------------------------------------------
    # Steps 3.5, 3.6, 3.7 — Pre-validation gates (FR-C4)
    # B4: Score floor uses raw_reranker_score (absolute quality signal),
    # not the normalized relative-rank signal.
    # ------------------------------------------------------------------
    post_rerank_count = len(candidates)
    candidates = apply_prevalidation_gates(
        candidates,
        cr_interp,
        settings,
        ctx.conn,
        enable_score_floor=variant_flags.enable_score_floor,
        enable_dedup=variant_flags.enable_dedup_gate,
        enable_plausibility=variant_flags.enable_plausibility_gate,
    )

    # AV-3: structured admission summary for ablation log analysis
    logger.info(
        "[runner] admission_summary variant={} post_rerank={} post_gates={} admitted={}",
        variant_flags.variant_id,
        post_rerank_count,
        len(candidates),
        len(candidates),
    )

    if not candidates:
        logger.warning("[runner] Zero candidates after gates — returning empty report")
        return _minimal_rejection_report("All candidates rejected by pre-validation gates")

    # ------------------------------------------------------------------
    # Step 4 — SIS Validation (LLM #2, FR-C5)
    # Batched: max 5 candidates per LLM call (Batching Mandate).
    # Returns (confirmed_ids, score_ordered_ids) for metric ranking (N9).
    # ------------------------------------------------------------------
    if variant_flags.enable_sis_validation:
        logger.info("[runner] Step 4: SIS validation (batched, max 5 per call)")
        sis_ids = validate_sis_candidates_batched(cr_interp, candidates, ctx.llm_client)
        if not sis_ids:
            logger.warning("[runner] LLM #2 confirmed zero candidates — returning empty report")
            return _minimal_rejection_report("SIS validation rejected all candidates")
    else:
        sis_ids = [c.node_id for c in candidates]
        logger.info("[runner] Step 4: SIS validation DISABLED — {} seeds", len(sis_ids))

    # ------------------------------------------------------------------
    # Step 5 — Resolve doc-chunk SIS to code seeds (FR-C6)
    # N9: Preserve score order within admitted set for metric ranking.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 5: Seed resolution")
    sis_id_set = set(sis_ids)
    # Sort admitted candidates by raw_reranker_score desc (absolute quality);
    # fall back to rrf_score for V0-V2 where reranker was disabled.
    admitted_candidates = sorted(
        [c for c in candidates if c.node_id in sis_id_set],
        key=lambda c: c.raw_reranker_score if c.raw_reranker_score > 0.0 else c.rrf_score,
        reverse=True,
    )

    if not admitted_candidates:
        logger.warning("[runner] Zero admitted candidates after gates — returning empty report")
        return _minimal_rejection_report("All candidates rejected by validation gates")

    resolutions, doc_to_code_map, direct_code_seeds = resolve_doc_to_code(
        sis_ids=sis_ids,
        conn=ctx.conn,
        top_k=settings.top_k_traceability,
    )
    logger.info(
        "[runner] Step 5: {} direct code seeds, {} doc-chunk resolutions",
        len(direct_code_seeds), len(resolutions),
    )

    # ------------------------------------------------------------------
    # Step 5b — Trace validation (LLM #3, FR-C7)
    # ------------------------------------------------------------------
    # low_conf tracks which resolved code seeds have PARTIAL decision.
    low_conf: dict[str, bool] = {}

    if variant_flags.enable_trace_validation and resolutions:
        logger.info("[runner] Step 5b: Trace validation (LLM #3, batched max 5)")

        # Hydrate doc texts from ChromaDB doc_meta_cache (pre-cached in ctx).
        doc_text_by_id: dict[str, str] = {}
        for r in resolutions:
            doc_id = r["doc_id"]
            if doc_id in ctx.doc_meta_cache:
                doc_text_by_id[doc_id] = ctx.doc_meta_cache[doc_id].get("document", "")
            else:
                # Fallback: fetch from ChromaDB directly.
                try:
                    res = ctx.doc_col.get(ids=[doc_id], include=["documents"])
                    if res["documents"]:
                        doc_text_by_id[doc_id] = res["documents"][0]
                except Exception:
                    doc_text_by_id[doc_id] = ""

        # Collect all resolved code IDs and fetch their metadata from SQLite.
        all_resolved_code_ids: set[str] = set()
        for r in resolutions:
            all_resolved_code_ids.update(r["code_ids"])

        code_meta_by_id: dict[str, dict] = {}
        if all_resolved_code_ids:
            placeholders = ",".join("?" * len(all_resolved_code_ids))
            rows = ctx.conn.execute(
                f"SELECT node_id, node_type, file_path, "
                f"internal_logic_abstraction, source_code "
                f"FROM code_nodes WHERE node_id IN ({placeholders})",
                list(all_resolved_code_ids),
            ).fetchall()
            for row in rows:
                code_meta_by_id[row[0]] = {
                    "node_type": row[1],
                    "file_path": row[2],
                    "internal_logic_abstraction": row[3],
                    "source_code": row[4],
                }

        validated_code_seeds, low_conf = validate_trace_resolutions(
            resolutions=resolutions,
            doc_text_by_id=doc_text_by_id,
            code_meta_by_id=code_meta_by_id,
            client=ctx.llm_client,
            cr_interp=cr_interp,
        )
        logger.info(
            "[runner] Step 5b: {} validated seeds ({} low-conf)",
            len(validated_code_seeds), sum(1 for v in low_conf.values() if v),
        )
    elif resolutions:
        # Blind resolution: take top-1 of each resolution as seed (no LLM #3).
        validated_code_seeds = []
        for r in resolutions:
            if r["code_ids"]:
                top_id = r["code_ids"][0]
                validated_code_seeds.append(top_id)
                low_conf[top_id] = True  # Blind = low confidence
        logger.info(
            "[runner] Step 5b: Trace validation DISABLED — {} blind seeds",
            len(validated_code_seeds),
        )
    else:
        validated_code_seeds = []

    # Merge all code seeds (deduplicated, preserving direct_code_seeds order).
    all_code_seeds = list(dict.fromkeys(direct_code_seeds + validated_code_seeds))
    logger.info("[runner] Combined code seeds: {}", len(all_code_seeds))

    # ------------------------------------------------------------------
    # Step 6 — BFS propagation (FR-D1)
    # ------------------------------------------------------------------
    if variant_flags.enable_bfs and all_code_seeds:
        logger.info("[runner] Step 6: BFS propagation")
        from impactracer.pipeline.graph_bfs import bfs_propagate, compute_confidence_tiers

        # Build reranker score map for confidence tiering.
        # doc-chunk reranker scores propagate to resolved code seeds via setdefault.
        sis_reranker_map: dict[str, float] = {
            c.node_id: c.reranker_score
            for c in admitted_candidates
            if c.node_id in set(all_code_seeds)
        }
        for r in resolutions:
            for cid in r["code_ids"]:
                sis_reranker_map.setdefault(cid, 0.0)

        high_conf = compute_confidence_tiers(
            all_code_seeds, sis_reranker_map, settings.bfs_high_conf_top_n
        )
        logger.info(
            "[runner] High-confidence seeds (top-{}): {}",
            settings.bfs_high_conf_top_n,
            len(high_conf),
        )

        cis = bfs_propagate(
            ctx.graph,
            all_code_seeds,
            high_confidence=high_conf,
            low_confidence_seed_map=low_conf,
        )
        logger.info(
            "[runner] BFS: {} SIS seeds, {} propagated nodes",
            len(cis.sis_nodes), len(cis.propagated_nodes),
        )
    elif all_code_seeds:
        # BFS disabled: CIS = seeds only.
        cis = _candidates_to_cis(admitted_candidates)
        if validated_code_seeds:
            # Add validated code seeds that aren't already in admitted_candidates.
            existing = set(cis.sis_nodes.keys())
            for code_id in validated_code_seeds:
                if code_id not in existing:
                    cis.sis_nodes[code_id] = NodeTrace(
                        depth=0,
                        causal_chain=[],
                        path=[code_id],
                        source_seed=code_id,
                        low_confidence_seed=low_conf.get(code_id, True),
                    )
        logger.info("[runner] Step 6: BFS DISABLED — {} SIS seeds", len(cis.sis_nodes))
    else:
        cis = _candidates_to_cis(admitted_candidates)
        logger.info("[runner] Step 6: No code seeds — CIS from admitted candidates")

    # ------------------------------------------------------------------
    # Step 6.5 — Graph Collapse: CONTAINS sub-tree aggregation (Sprint 10.1)
    # Must run AFTER BFS and BEFORE LLM #4 to reduce prompt token count.
    # Only applies when BFS was enabled (propagated_nodes is non-empty).
    # ------------------------------------------------------------------
    if variant_flags.enable_bfs and cis.propagated_nodes:
        from impactracer.pipeline.graph_bfs import collapse_contains_subtrees

        # We need node_meta_by_id to check parent/child types.  Build a
        # lightweight version from the SQLite code_nodes table for all CIS ids.
        _collapse_ids = cis.all_node_ids()
        _collapse_meta: dict[str, dict] = {}
        if _collapse_ids:
            _placeholders = ",".join("?" * len(_collapse_ids))
            _rows = ctx.conn.execute(
                f"SELECT node_id, node_type FROM code_nodes "
                f"WHERE node_id IN ({_placeholders})",
                _collapse_ids,
            ).fetchall()
            for _row in _rows:
                _collapse_meta[_row[0]] = {"node_type": _row[1]}

        pre_collapse_propagated = len(cis.propagated_nodes)
        cis = collapse_contains_subtrees(cis, ctx.graph, _collapse_meta)
        logger.info(
            "[runner] Step 6.5: Graph collapse reduced propagated_nodes "
            "from {} to {} (removed {} CONTAINS-only leaves)",
            pre_collapse_propagated,
            len(cis.propagated_nodes),
            pre_collapse_propagated - len(cis.propagated_nodes),
        )

    # ------------------------------------------------------------------
    # Step 7 — Propagation validation (LLM #4, FR-D2)
    # ------------------------------------------------------------------
    if variant_flags.enable_propagation_validation and cis.propagated_nodes:
        logger.info(
            "[runner] Step 7: Propagation validation (LLM #4, {} propagated nodes)",
            len(cis.propagated_nodes),
        )
        # Fetch node metadata for all CIS nodes (SIS + propagated).
        all_cis_ids = cis.all_node_ids()
        node_meta_by_id: dict[str, dict] = {}
        if all_cis_ids:
            placeholders = ",".join("?" * len(all_cis_ids))
            rows = ctx.conn.execute(
                f"SELECT node_id, node_type, file_path, "
                f"internal_logic_abstraction, source_code "
                f"FROM code_nodes WHERE node_id IN ({placeholders})",
                all_cis_ids,
            ).fetchall()
            for row in rows:
                node_meta_by_id[row[0]] = {
                    "node_type": row[1],
                    "file_path": row[2],
                    "internal_logic_abstraction": row[3],
                    "source_code": row[4],
                }

        cis = validate_propagation(
            cis=cis,
            cr_interp=cr_interp,
            node_meta_by_id=node_meta_by_id,
            client=ctx.llm_client,
        )
        logger.info(
            "[runner] Post-LLM #4 CIS: {} SIS + {} propagated = {} total",
            len(cis.sis_nodes), len(cis.propagated_nodes),
            len(cis.sis_nodes) + len(cis.propagated_nodes),
        )
    elif variant_flags.enable_propagation_validation:
        logger.info("[runner] Step 7: Propagation validation SKIPPED (no propagated nodes)")

    # ------------------------------------------------------------------
    # Step 8 — Backlinks + Context (FR-E1, FR-E2)
    # ------------------------------------------------------------------
    logger.info("[runner] Step 8: Build context")
    all_node_ids = cis.all_node_ids()

    # Build node_types map from admitted candidates (needed for FF-3 routing).
    # Also query SQLite for propagated nodes not in admitted_candidates.
    node_types: dict[str, str] = {}
    node_file_paths: dict[str, str] = {}
    for c in admitted_candidates:
        node_types[c.node_id] = c.node_type
        node_file_paths[c.node_id] = c.file_path or ""

    # Propagated nodes may not be in admitted_candidates — fetch from SQLite.
    missing_ids = [nid for nid in all_node_ids if nid not in node_types]
    if missing_ids:
        placeholders = ",".join("?" * len(missing_ids))
        rows = ctx.conn.execute(
            f"SELECT node_id, node_type, file_path FROM code_nodes "
            f"WHERE node_id IN ({placeholders})",
            missing_ids,
        ).fetchall()
        for row in rows:
            node_types[row[0]] = row[1]
            node_file_paths[row[0]] = row[2] or ""

    # FF-3: bidirectional backlink routing by node type
    backlinks = fetch_backlinks(all_node_ids, node_types, ctx.conn, settings.top_k_backlinks_per_node)
    snippets = fetch_snippets(all_node_ids, ctx.conn, doc_col=ctx.doc_col)

    # FF-6/B4: Use raw_reranker_score (absolute quality) for context
    # truncation priority.  The affinity-modified normalized score would
    # mis-rank TYPE_DEFINITION nodes vs UTILITY nodes of equal raw quality.
    candidate_scores: dict[str, float] = {}
    for c in admitted_candidates:
        candidate_scores[c.node_id] = (
            c.raw_reranker_score if c.raw_reranker_score > 0.0 else c.rrf_score
        )

    context = build_context(
        cr_text, cr_interp, cis, backlinks, snippets, settings,
        node_file_paths=node_file_paths,
        node_types=node_types,
        candidate_scores=candidate_scores,
    )

    # ------------------------------------------------------------------
    # Step 9 — Synthesize (LLM #5, FR-E3)
    # N8: estimated_scope is computed DETERMINISTICALLY from admitted node
    # and file counts post-gates, not hallucinated by LLM #5 from a pruned
    # set.  The synthesizer is instructed NOT to override it.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 9: Synthesize report")
    report = synthesize_report(context, ctx.llm_client)

    # N8: override estimated_scope with deterministic computation
    report = report.model_copy(update={
        "estimated_scope": _compute_scope(cis),
    })

    # AV-5: set analysis_mode based on whether BFS actually ran
    bfs_ran = variant_flags.enable_bfs and len(cis.propagated_nodes) > 0
    analysis_mode = "retrieval_plus_propagation" if bfs_ran else "retrieval_only"
    report = report.model_copy(update={"analysis_mode": analysis_mode})

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[runner] Analysis complete: {} impacted nodes, scope={}, mode={}, elapsed={:.1f}s, LLM calls={}",
        len(report.impacted_nodes),
        report.estimated_scope,
        report.analysis_mode,
        elapsed,
        ctx.llm_client.call_counter,
    )
    return report
