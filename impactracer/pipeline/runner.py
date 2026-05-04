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
from impactracer.pipeline.synthesizer import synthesize_report
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
    """Materialize the structural edge graph from SQLite (Step 0)."""
    g = nx.MultiDiGraph()
    rows = conn.execute(
        "SELECT source_id, target_id, edge_type FROM structural_edges"
    ).fetchall()
    for src, tgt, etype in rows:
        g.add_edge(src, tgt, edge_type=etype)
    logger.info("[runner] Graph loaded: {} nodes, {} edges", g.number_of_nodes(), g.number_of_edges())
    return g


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
    # Step 5 — Resolve doc-chunk SIS to code seeds (Sprint 10 — gated)
    # N9: Preserve score order within admitted set for metric ranking.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 5: Seed resolution (blind passthrough this sprint)")
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

    # ------------------------------------------------------------------
    # Step 5b — Trace validation (LLM #3, Sprint 10 — gated)
    # ------------------------------------------------------------------
    if variant_flags.enable_trace_validation:
        logger.info("[runner] Step 5b: Trace validation (stub — Sprint 10)")

    # ------------------------------------------------------------------
    # Step 6 — BFS propagation (Sprint 10 — gated)
    # ------------------------------------------------------------------
    if variant_flags.enable_bfs:
        logger.info("[runner] Step 6: BFS propagation (stub — Sprint 10)")

    cis = _candidates_to_cis(admitted_candidates)
    logger.info("[runner] CIS: {} SIS seeds, 0 propagated", len(cis.sis_nodes))

    # ------------------------------------------------------------------
    # Step 7 — Propagation validation (LLM #4, Sprint 10 — gated)
    # ------------------------------------------------------------------
    if variant_flags.enable_propagation_validation:
        logger.info("[runner] Step 7: Propagation validation (stub — Sprint 10)")

    # ------------------------------------------------------------------
    # Step 8 — Backlinks + Context (FR-E1, FR-E2)
    # ------------------------------------------------------------------
    logger.info("[runner] Step 8: Build context")
    all_node_ids = cis.all_node_ids()

    # Build node_types map from admitted candidates (needed for FF-3 routing)
    node_types: dict[str, str] = {}
    node_file_paths: dict[str, str] = {}
    for c in admitted_candidates:
        node_types[c.node_id] = c.node_type
        node_file_paths[c.node_id] = c.file_path or ""

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
