"""Online pipeline orchestrator (nine steps, five LLM invocations).

Invoked by :func:`impactracer.cli.analyze`. Consumes a :class:`VariantFlags`
instance so the same code powers both full V7 analysis and the ablation
harness variants V0 through V6.

Reference: master_blueprint.md §4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
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
    apply_negative_filter,
    apply_traceability_bonus,
    build_bm25_from_chroma,
    build_metadata_cache,
    hybrid_search,
)
from impactracer.pipeline.seed_resolver import resolve_doc_to_code
from impactracer.pipeline.synthesizer import (
    assemble_impact_report,
    build_deterministic_impacted_entities,
    build_minimal_summary,
    synthesize_summary,
)
from impactracer.pipeline.traceability_validator import validate_trace_resolutions
from impactracer.pipeline.traversal_validator import validate_propagation
from impactracer.pipeline.validator import validate_sis_candidates_batched
from impactracer.shared.config import Settings
from impactracer.shared.constants import (
    PROPAGATION_VALIDATION_EXEMPT_EDGES,
)
from impactracer.shared.models import (
    CISResult,
    Candidate,
    CRInterpretation,
    ImpactReport,
    LLMSynthesisOutput,
    NodeTrace,
)

if TYPE_CHECKING:
    from impactracer.evaluation.variant_flags import VariantFlags
    from impactracer.pipeline.variant_cache import VariantCache


@dataclass
class PipelineContext:
    """Loaded persistent stores, shared across all pipeline steps."""

    conn: Any
    doc_col: Any
    code_col: Any
    graph: Any
    doc_bm25: Any
    doc_bm25_ids: list[str]
    doc_meta_cache: dict[str, dict]   # pre-cached doc chunk metadata
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
    heavy objects across all ablation runs.

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


def _minimal_rejection_report(reason: str, degraded: bool = False) -> ImpactReport:
    """Return a minimal ImpactReport for non-actionable CRs."""
    return ImpactReport(
        executive_summary=f"CR rejected: {reason}",
        impacted_files=[],
        impacted_entities=[],
        documentation_conflicts=[],
        estimated_scope="terlokalisasi",
        analysis_mode="retrieval_only",
        degraded_run=degraded,
    )


def _validate_cr_interpretation_coherence(ci: CRInterpretation) -> CRInterpretation:
    """Soft-fix change_type / affected_layers coherence.

    DELETION CRs without 'code' layer have 'code' added. ADDITION CRs with
    only 'code' (no requirement/design) have 'requirement' added so doc
    retrieval has a chance to pull in the new feature's spec.

    Never rejects — only broadens. The pipeline always proceeds.
    """
    layers = list(ci.affected_layers)
    mutated = False
    if ci.change_type == "DELETION" and "code" not in layers:
        layers.append("code")
        mutated = True
    if ci.change_type == "ADDITION" and layers == ["code"]:
        layers.append("requirement")
        mutated = True
    if mutated:
        logger.warning(
            "[runner] CR coherence: change_type={} -> broadened affected_layers from {} to {}",
            ci.change_type, ci.affected_layers, layers,
        )
        ci = ci.model_copy(update={"affected_layers": layers})
    return ci


def _compute_scope(cis: CISResult, settings: "Settings | None" = None) -> str:
    """Deterministically compute estimated_scope from CIS node counts.

    Counts the full CIS regardless of prompt-window truncation, so the scope
    is reproducible. Thresholds from Settings.scope_local_max / scope_medium_max.

    Returns one of "terlokalisasi", "menengah", "ekstensif".
    """
    local_max = getattr(settings, "scope_local_max", 10) if settings else 10
    medium_max = getattr(settings, "scope_medium_max", 30) if settings else 30
    n_nodes = len(cis.combined())
    if n_nodes <= local_max:
        return "terlokalisasi"
    if n_nodes <= medium_max:
        return "menengah"
    return "ekstensif"


def run_analysis(
    cr_text: str,
    settings: Settings,
    variant_flags: "VariantFlags | None" = None,
    shared_embedder: Any = None,
    shared_reranker: Any = None,
    shared_llm_client: Any = None,
    trace_sink: dict | None = None,
    variant_cache: "VariantCache | None" = None,
) -> ImpactReport:
    """End-to-end online analysis for one CR.

    When ``trace_sink`` is provided (a mutable dict), the runner populates it
    with every step's result so the CLI can write ``impact_report_full.json``
    for academic auditability. The dict is mutated in-place; None disables tracing.

    Trace keys: step_1_interpretation, step_2_rrf_pool, step_3_reranked,
    step_3_gates_survivors, step_4_llm2_verdicts, step_5_resolutions,
    step_5b_llm3_verdicts, step_6_bfs_raw_cis, step_7_llm4_verdicts, final_report.

    Blueprint §4.
    """
    from impactracer.evaluation.variant_flags import VariantFlags

    if variant_flags is None:
        variant_flags = VariantFlags.v7_full()

    def _trace(key: str, value: Any) -> None:
        if trace_sink is not None:
            trace_sink[key] = value

    t_start = time.perf_counter()
    ctx = load_pipeline_context(
        settings, variant_flags,
        shared_embedder=shared_embedder,
        shared_reranker=shared_reranker,
        shared_llm_client=shared_llm_client,
    )

    # ------------------------------------------------------------------
    # Step 1 — Interpret CR (LLM #1, always-on)
    #
    # When variant_flags.two_stage_interpret is True AND a project-skeleton
    # file is available on disk, LLM #1 is split into two calls (intent +
    # project-grounded anchors). The cached CRInterpretation memoises the
    # glued result so V1-V7 share it.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 1: Interpret CR")
    if variant_cache is not None:
        cached_interp = variant_cache.get_interp()
    else:
        cached_interp = None
    if cached_interp is not None:
        logger.info("[runner] [cache HIT] interp (LLM #1 skipped)")
        cr_interp = cached_interp
    else:
        # Load the project skeleton at most once per process. The runner
        # reads it from settings.project_skeleton_path; missing file is
        # tolerated (interpret_cr falls back to single-stage).
        project_skeleton: str | None = None
        if variant_flags is not None and variant_flags.two_stage_interpret:
            from impactracer.indexer.project_skeleton import read_project_skeleton
            project_skeleton = read_project_skeleton(
                Path(settings.project_skeleton_path)
            )
            if project_skeleton is None:
                logger.warning(
                    "[runner] two_stage_interpret enabled but no project "
                    "skeleton at {} — falling back to single-stage.",
                    settings.project_skeleton_path,
                )
        cr_interp = interpret_cr(
            cr_text,
            ctx.llm_client,
            variant_flags=variant_flags,
            project_skeleton=project_skeleton,
        )
        if variant_cache is not None:
            variant_cache.put_interp(cr_interp)

    logger.info(
        "[runner] === INTERPRETER OUTPUT ===\n"
        "  is_actionable : {}\n"
        "  change_type   : {}\n"
        "  affected_layers: {}\n"
        "  primary_intent: {}\n"
        "  domain_concepts: {}\n"
        "  search_queries  : {}\n"
        "  named_entry_points: {}\n"
        "  anchor_candidates: {}\n"
        "  out_of_scope_operations: {}",
        cr_interp.is_actionable,
        cr_interp.change_type,
        cr_interp.affected_layers,
        cr_interp.primary_intent,
        cr_interp.domain_concepts,
        cr_interp.search_queries,
        cr_interp.named_entry_points,
        cr_interp.anchor_candidates,
        cr_interp.out_of_scope_operations,
    )

    _trace("step_1_interpretation", cr_interp.model_dump())

    if not cr_interp.is_actionable:
        logger.info("[runner] CR is NOT actionable — short-circuiting")
        rej = _minimal_rejection_report(
            cr_interp.actionability_reason or "CR was not actionable"
        )
        _trace("final_report", rej.model_dump())
        return rej

    cr_interp = _validate_cr_interpretation_coherence(cr_interp)

    # Code-only mode (sensitivity-analysis toggle): coerce
    # affected_layers to ["code"] so the retriever skips both doc paths.
    # Placement matters: the coercion must run AFTER
    # _validate_cr_interpretation_coherence, which otherwise broadens
    # ADDITION CRs to include "requirement" and would re-enable doc
    # retrieval. By running last, this block has the final word.
    if getattr(settings, "code_only_mode", False) and cr_interp.is_actionable:
        if cr_interp.affected_layers != ["code"]:
            logger.info(
                "[runner] code_only_mode: coerced affected_layers from {} to ['code']",
                cr_interp.affected_layers,
            )
            cr_interp = cr_interp.model_copy(update={"affected_layers": ["code"]})

    # Aggregate degraded flag across all LLM batches in this run.
    degraded_run: bool = False

    # ------------------------------------------------------------------
    # Step 2 — Adaptive RRF Hybrid Search (FR-C1, FR-C2)
    #
    # Cache key: the hybrid_search output depends on which
    # of bm25/dense/rrf are enabled. V0 (bm25-only), V1 (dense-only), and
    # V2+ (bm25+dense+RRF) each produce a different candidate list and
    # cache under a separate key. V2-V7 share the V2+ retrieval key.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 2: Hybrid search (variant={})", variant_flags.variant_id)
    if variant_cache is not None:
        if variant_flags.variant_id == "V0":
            _retr_key = "retrieval_v0"
            _retr_get = variant_cache.get_retrieval_v0
            _retr_put = variant_cache.put_retrieval_v0
        elif variant_flags.variant_id == "V1":
            _retr_key = "retrieval_v1"
            _retr_get = variant_cache.get_retrieval_v1
            _retr_put = variant_cache.put_retrieval_v1
        else:
            _retr_key = "retrieval_v2plus"
            _retr_get = variant_cache.get_retrieval_v2plus
            _retr_put = variant_cache.put_retrieval_v2plus
        cached_candidates = _retr_get()
    else:
        cached_candidates = None
        _retr_put = None
    if cached_candidates is not None:
        logger.info("[runner] [cache HIT] {} ({} candidates skipped recompute)",
                    _retr_key, len(cached_candidates))
        candidates = cached_candidates
    else:
        candidates = hybrid_search(cr_interp, ctx, settings, cr_text=cr_text)
        if _retr_put is not None:
            _retr_put(candidates)
    logger.info("[runner] Post-RRF pool: {}", len(candidates))
    _trace("step_2_rrf_pool", [
        {"node_id": c.node_id, "collection": c.collection,
         "rrf_score": c.rrf_score, "file_path": c.file_path}
        for c in candidates
    ])

    # Guard zero-candidate case: synthesizing with zero nodes produces a hallucinated report.
    if not candidates:
        logger.warning("[runner] Zero candidates — returning empty report")
        return _minimal_rejection_report("No candidates retrieved — check index and affected_layers")

    # ------------------------------------------------------------------
    # Step 3 — Cross-Encoder Rerank (FR-C3) + optional graph-aware rerank
    # Graph-aware rerank inserted between the cross-encoder and the
    # top-K truncation. Cross-encoder scores ALL 200 RRF candidates;
    # graph propagation blends in a structural signal; the truncation
    # to max_admitted_seeds happens on the blended score.
    #
    # Cache key: V3-V7 all reach the same post-gate
    # candidate list (rerank + 3 gates is a pure function of the V2+
    # retrieval pool + cr_interp). Cache the post-gate result under
    # `rerank_gated`. On hit, skip the rerank, pinning, normalisation,
    # and the three gates entirely.
    # ------------------------------------------------------------------
    _rerank_gated_cache_hit = False
    if (
        variant_cache is not None
        and variant_flags.enable_cross_encoder
        and variant_flags.enable_score_floor
        and variant_flags.enable_dedup_gate
        and variant_flags.enable_plausibility_gate
    ):
        _cached_rerank_gated = variant_cache.get_rerank_gated()
        if _cached_rerank_gated is not None:
            logger.info(
                "[runner] [cache HIT] rerank_gated ({} candidates, "
                "rerank + 3 gates skipped)",
                len(_cached_rerank_gated),
            )
            candidates = _cached_rerank_gated
            _rerank_gated_cache_hit = True
            # Replay the trace logs the gates would have emitted.
            _trace("step_3_gates_survivors", [
                {"node_id": c.node_id, "collection": c.collection,
                 "raw_reranker_score": c.raw_reranker_score,
                 "reranker_score": c.reranker_score, "file_path": c.file_path,
                 "merged_doc_ids": list(c.merged_doc_ids)}
                for c in candidates
            ])

    if _rerank_gated_cache_hit:
        pass  # Skip the rerank + gates block entirely.
    elif variant_flags.enable_cross_encoder:
        logger.info("[runner] Step 3: Cross-encoder rerank (multi-query max scoring)")
        # Cross-encoder ALWAYS scores the full pool (not just top_k=
        # max_admitted_seeds). Three reasons:
        #   1. step_3_reranked_full trace needs the rank of every candidate,
        #      not just the top-K. The K-widening diagnostic
        #      (tools/diagnose_k_widening.py) reads this trace.
        #   2. The traceability bonus and negative filter (lines below) then
        #      operate on the full ranked pool; the final truncation at
        #      candidates[:max_admitted_seeds] is mathematically equivalent
        #      at the final SIS — the top-15 after sort is the same set
        #      whether you sort 15 or 200.
        #   3. Symmetry with the graph-rerank path (default-disabled) which
        #      already required full-pool scoring.
        # Performance cost: ~1s additional cross-encoder wall time per CR
        # (200 vs 15 candidates on bge-reranker-v2-m3); negligible vs LLM cost.
        _graph_rerank_on = getattr(settings, "enable_graph_rerank", False)
        candidates = ctx.reranker.rerank_multi_query(
            cr_interp.search_queries,
            cr_interp.primary_intent,
            candidates,
            len(candidates),
        )
        logger.info(
            "[runner] Post-rerank (full_pool={}): {} candidates",
            _graph_rerank_on, len(candidates),
        )

        # Snapshot raw cross-encoder logits before any post-rerank adjustment.
        for c in candidates:
            c.raw_reranker_score = c.reranker_score

        # Traceability bonus: additive +0.1 for code candidates associated with
        # any retrieved doc chunk in the offline traceability table.
        candidates = apply_traceability_bonus(candidates, ctx.conn)

        # Negative filter (penalty -1.0, name-only).
        candidates = apply_negative_filter(
            candidates, cr_interp.out_of_scope_operations
        )

        candidates.sort(key=lambda c: c.raw_reranker_score, reverse=True)

        # ----- Optional graph-aware label-propagation rerank (default off)
        # Default-disabled post-Sprint-15. The full-pool cross-encoder pass
        # above only fires when this flag is on, so the V4-canonical regime
        # has IDENTICAL behaviour to pre-Sprint-15 code.
        if _graph_rerank_on:
            from impactracer.pipeline.graph_rerank import graph_rerank

            # Mode B needs metadata for any node that could be graph-added.
            # Bulk-fetch all code_nodes once (fast on 3,150-node citrakara).
            code_meta_by_id: dict[str, dict] = {}
            for row in ctx.conn.execute(
                "SELECT node_id, node_type, file_path, file_classification, "
                "internal_logic_abstraction, source_code FROM code_nodes"
            ).fetchall():
                code_meta_by_id[row[0]] = {
                    "node_type": row[1],
                    "file_path": row[2],
                    "file_classification": row[3],
                    "internal_logic_abstraction": row[4],
                    "source_code": row[5],
                }

            pre_count = len(candidates)
            candidates = graph_rerank(
                candidates,
                ctx.graph,
                alpha=getattr(settings, "graph_rerank_alpha", 0.7),
                iterations=getattr(settings, "graph_rerank_iterations", 2),
                personalization_top_n=getattr(
                    settings, "graph_rerank_personalization_top_n", 5
                ),
                add_top_n=getattr(settings, "graph_rerank_add_top_n", 10),
                add_min_score=getattr(settings, "graph_rerank_add_min_score", 0.10),
                code_meta_by_id=code_meta_by_id,
            )
            logger.info(
                "[runner] Step 3 (Apex C): graph rerank produced {} candidates "
                "(was {}, added {})",
                len(candidates), pre_count, len(candidates) - pre_count,
            )

            # graph_rerank already wrote blended values to raw_reranker_score /
            # reranker_score. Re-sort by blended score.
            candidates.sort(key=lambda c: c.raw_reranker_score, reverse=True)

        # Step 3 trace: snapshot the FULL post-rerank
        # pool BEFORE the max_admitted_seeds truncation. Captures up to 200
        # candidates (or ~210 when graph-rerank mode B adds extras), letting
        # tools/diagnose_k_widening.py count GT entities at ranks 16-30 from
        # existing eval artefacts. Read-only trace; behaviour unchanged.
        _trace("step_3_reranked_full", [
            {"node_id": c.node_id, "collection": c.collection,
             "rrf_score": c.rrf_score, "reranker_score": c.reranker_score,
             "raw_reranker_score": c.raw_reranker_score, "file_path": c.file_path,
             "name": c.name}
            for c in candidates
        ])

        # Named-entry-point hard pass-through.
        # Before the top-K truncation, force-include any candidate whose
        # node_id, file_path, or name matches a token in
        # cr_interp.named_entry_points. The cross-encoder may rank the
        # CR-named file out of the top-K when the CR text body is short
        # (e.g. "delete formatters.ts") while service-method bodies score
        # higher on lexical density. The exemption ensures that files the
        # LLM #1 interpreter explicitly named survive into the SIS.
        named_patterns_raw = list(cr_interp.named_entry_points or [])
        named_patterns = [p.lower() for p in named_patterns_raw if p]

        def _is_named(c) -> bool:
            if not named_patterns:
                return False
            fields = [c.node_id or "", c.file_path or "", c.name or ""]
            haystacks = [f.lower() for f in fields]
            for p in named_patterns:
                for h in haystacks:
                    if p in h:
                        return True
            return False

        # Partition into pinned (named) and rest, preserving relative order.
        # Tag pinned candidates so downstream gates respect the pin.
        pinned = []
        rest = []
        for c in candidates:
            if _is_named(c):
                c.pinned_by_named_entry = True
                pinned.append(c)
            else:
                rest.append(c)

        if pinned:
            logger.info(
                "[runner] Named-entry-point pinning: {} candidates pinned by patterns={}",
                len(pinned), named_patterns_raw,
            )

        # Truncate to max_admitted_seeds AFTER all post-rerank adjustments.
        # Pinned candidates always survive; remaining slots filled by top
        # cross-encoder ranks. If pinned count exceeds the cap, all pinned
        # are kept (the cap is a soft floor in that case).
        seat_cap = settings.max_admitted_seeds
        if len(pinned) >= seat_cap:
            candidates = pinned
        else:
            remaining = seat_cap - len(pinned)
            candidates = pinned + rest[:remaining]

        _trace("step_3_reranked", [
            {"node_id": c.node_id, "collection": c.collection,
             "rrf_score": c.rrf_score, "reranker_score": c.reranker_score,
             "raw_reranker_score": c.raw_reranker_score, "file_path": c.file_path,
             "name": c.name,
             "named_entry_point_pinned": _is_named(c)}
            for c in candidates
        ])

        # Min-max normalize to [0,1] for relative sorting in gates and context.
        if len(candidates) > 1:
            min_s = min(c.raw_reranker_score for c in candidates)
            max_s = max(c.raw_reranker_score for c in candidates)
            for c in candidates:
                c.reranker_score = c.raw_reranker_score
            if max_s > min_s:
                span = max_s - min_s
                for c in candidates:
                    c.reranker_score = (c.raw_reranker_score - min_s) / span
            logger.debug(
                "[runner] Reranker scores normalized: min_raw={:.4f} max_raw={:.4f}",
                min_s, max_s,
            )
    else:
        # V0-V2: no reranker — cap at max_admitted_seeds from the RRF pool.
        # Named entry points are pinned before
        # truncation so they survive the cap.
        named_patterns_raw = list(cr_interp.named_entry_points or [])
        named_patterns = [p.lower() for p in named_patterns_raw if p]

        def _is_named_v012(c) -> bool:
            if not named_patterns:
                return False
            fields = [c.node_id or "", c.file_path or "", c.name or ""]
            haystacks = [f.lower() for f in fields]
            for p in named_patterns:
                for h in haystacks:
                    if p in h:
                        return True
            return False

        pinned = []
        rest = []
        for c in candidates:
            if _is_named_v012(c):
                c.pinned_by_named_entry = True
                pinned.append(c)
            else:
                rest.append(c)
        seat_cap = settings.max_admitted_seeds
        if len(pinned) >= seat_cap:
            candidates = pinned
        else:
            candidates = pinned + rest[: seat_cap - len(pinned)]
        if pinned:
            logger.info(
                "[runner] Named-entry-point pinning (no rerank): {} pinned",
                len(pinned),
            )
        logger.info("[runner] Step 3: Cross-encoder DISABLED ({})", variant_flags.variant_id)

    # ------------------------------------------------------------------
    # Steps 3.5, 3.6, 3.7 — Pre-validation gates (FR-C4)
    #
    # Step 3.6 dedup and 3.7 plausibility run universally
    # (V0-V7). 3.5 score floor stays gated on the cross-encoder because
    # it consumes raw_reranker_score that V0-V2 do not produce.
    #
    # Skipped on cache hit for the post-gate candidate list.
    # ------------------------------------------------------------------
    if not _rerank_gated_cache_hit:
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

        logger.info(
            "[runner] admission_summary variant={} post_rerank={} post_gates={} admitted={}",
            variant_flags.variant_id,
            post_rerank_count,
            len(candidates),
            len(candidates),
        )
        _trace("step_3_gates_survivors", [
            {"node_id": c.node_id, "collection": c.collection,
             "raw_reranker_score": c.raw_reranker_score,
             "reranker_score": c.reranker_score, "file_path": c.file_path,
             "merged_doc_ids": list(c.merged_doc_ids)}
            for c in candidates
        ])

        # Cache the post-gate candidates for V3-V7 sharing.
        if (
            variant_cache is not None
            and variant_flags.enable_cross_encoder
            and variant_flags.enable_score_floor
            and variant_flags.enable_dedup_gate
            and variant_flags.enable_plausibility_gate
        ):
            variant_cache.put_rerank_gated(candidates)

    if not candidates:
        logger.warning("[runner] Zero candidates after gates — returning empty report")
        return _minimal_rejection_report("All candidates rejected by pre-validation gates")

    # ------------------------------------------------------------------
    # Step 4 — SIS Validation (LLM #2, FR-C5)
    # Batched max 5. Returns (ids, justifications, degraded).
    #
    # Cache key: the LLM #2 verdicts on the post-gate
    # candidate list are a pure function of (candidates, cr_interp).
    # V4-V7 all share the same post-gate candidates (via rerank_gated
    # cache) and the same cr_interp, so the LLM #2 verdicts are
    # cacheable across V4-V7 under one key.
    # ------------------------------------------------------------------
    sis_justifications: dict[str, dict[str, str]] = {}
    if variant_flags.enable_sis_validation:
        logger.info("[runner] Step 4: SIS validation (batched, fail-closed)")
        _cached_sis = (
            variant_cache.get_sis_verdicts() if variant_cache is not None else None
        )
        if _cached_sis is not None:
            sis_ids, sis_justifications, llm2_degraded = _cached_sis
            logger.info(
                "[runner] [cache HIT] sis_verdicts ({} confirmed, LLM #2 skipped)",
                len(sis_ids),
            )
        else:
            sis_ids, sis_justifications, llm2_degraded = validate_sis_candidates_batched(
                cr_interp, candidates, ctx.llm_client
            )
            if variant_cache is not None:
                variant_cache.put_sis_verdicts(
                    sis_ids, sis_justifications, llm2_degraded
                )
        if llm2_degraded:
            degraded_run = True
        _trace("step_4_llm2_verdicts", {
            "confirmed_ids": list(sis_ids),
            "justifications": sis_justifications,
            "degraded": llm2_degraded,
        })
        if not sis_ids:
            logger.warning("[runner] LLM #2 confirmed zero candidates — returning empty report")
            rej = _minimal_rejection_report(
                "SIS validation rejected all candidates", degraded=degraded_run
            )
            _trace("final_report", rej.model_dump())
            return rej
    else:
        sis_ids = [c.node_id for c in candidates]
        logger.info("[runner] Step 4: SIS validation DISABLED — {} seeds", len(sis_ids))

    # ------------------------------------------------------------------
    # Step 5 — Resolve doc-chunk SIS to code seeds (FR-C6)
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

    # Build code_node_ids once per run so resolve_doc_to_code skips the full
    # table scan on every call during the ablation harness (160 calls per eval).
    _code_node_ids: set[str] = {
        row[0]
        for row in ctx.conn.execute("SELECT node_id FROM code_nodes").fetchall()
    }

    resolutions, direct_code_seeds = resolve_doc_to_code(
        sis_ids=sis_ids,
        conn=ctx.conn,
        top_k=settings.top_k_traceability,
        code_node_ids=_code_node_ids,
    )
    logger.info(
        "[runner] Step 5: {} direct code seeds, {} doc-chunk resolutions",
        len(direct_code_seeds), len(resolutions),
    )
    _trace("step_5_resolutions", {
        "direct_code_seeds": list(direct_code_seeds),
        "doc_resolutions": [
            {"doc_id": r["doc_id"], "code_ids": list(r["code_ids"])}
            for r in resolutions
        ],
    })

    # ------------------------------------------------------------------
    # Step 5b — Trace validation (LLM #3, FR-C7)
    #
    # Cache key: LLM #3 verdicts on the resolutions are a
    # pure function of (resolutions, cr_interp). V5-V7 share the same
    # resolutions (because they share SIS verdicts) and the same
    # cr_interp, so trace_verdicts is cacheable across V5-V7.
    # ------------------------------------------------------------------
    low_conf: dict[str, bool] = {}
    trace_justifications: dict[str, str] = {}

    if variant_flags.enable_trace_validation and resolutions:
        logger.info("[runner] Step 5b: Trace validation (LLM #3, batched max 5)")

        _cached_trace = (
            variant_cache.get_trace_verdicts() if variant_cache is not None else None
        )
        if _cached_trace is not None:
            validated_code_seeds, low_conf, trace_justifications, llm3_degraded = _cached_trace
            logger.info(
                "[runner] [cache HIT] trace_verdicts ({} seeds, LLM #3 skipped)",
                len(validated_code_seeds),
            )
        else:
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

            validated_code_seeds, low_conf, trace_justifications, llm3_degraded = (
                validate_trace_resolutions(
                    resolutions=resolutions,
                    doc_text_by_id=doc_text_by_id,
                    code_meta_by_id=code_meta_by_id,
                    client=ctx.llm_client,
                    cr_interp=cr_interp,
                )
            )
            if variant_cache is not None:
                variant_cache.put_trace_verdicts(
                    validated_code_seeds,
                    low_conf,
                    trace_justifications,
                    llm3_degraded,
                )

        if llm3_degraded:
            degraded_run = True
        logger.info(
            "[runner] Step 5b: {} validated seeds ({} low-conf, degraded={})",
            len(validated_code_seeds), sum(1 for v in low_conf.values() if v),
            llm3_degraded,
        )
        _trace("step_5b_llm3_verdicts", {
            "validated_code_seeds": list(validated_code_seeds),
            "low_confidence": dict(low_conf),
            "justifications": dict(trace_justifications),
            "degraded": llm3_degraded,
        })
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
    #
    # Cache key: the BFS output is a pure function of
    # (all_code_seeds, low_conf, graph, settings). V6 and V7 share all
    # of those, so the post-BFS-and-collapse CIS is cacheable across
    # the two variants under one `bfs_cis` key. The cache covers both
    # Step 6 (BFS) and Step 6.5 (graph collapse), since the collapse is
    # also deterministic given the BFS CIS.
    # ------------------------------------------------------------------
    _bfs_cis_cache_hit = False
    if variant_cache is not None and variant_flags.enable_bfs and all_code_seeds:
        _cached_bfs_cis = variant_cache.get_bfs_cis()
        if _cached_bfs_cis is not None:
            cis = _cached_bfs_cis
            logger.info(
                "[runner] [cache HIT] bfs_cis ({} sis + {} propagated, "
                "BFS + collapse skipped)",
                len(cis.sis_nodes), len(cis.propagated_nodes),
            )
            _trace("step_6_bfs_raw_cis", {
                "sis_seeds": [
                    {"node_id": k, "depth": v.depth, "source_seed": v.source_seed}
                    for k, v in cis.sis_nodes.items()
                ],
                "propagated_nodes": [
                    {"node_id": k, "depth": v.depth,
                     "causal_chain": v.causal_chain, "source_seed": v.source_seed}
                    for k, v in cis.propagated_nodes.items()
                ],
            })
            _bfs_cis_cache_hit = True

    if _bfs_cis_cache_hit:
        pass  # CIS already loaded from cache; skip Step 6 and Step 6.5.
    elif variant_flags.enable_bfs and all_code_seeds:
        logger.info("[runner] Step 6: BFS propagation")
        from impactracer.pipeline.graph_bfs import bfs_propagate, compute_confidence_tiers

        # Build reranker score map for confidence tiering.
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

        # Bulk-fetch (file_classification, node_type) for BFS depth cap and fan-in cap.
        all_graph_node_ids = list(ctx.graph.nodes())
        seed_file_classification: dict[str, str] = {}
        node_type_by_id: dict[str, str] = {}
        ids_to_fetch = list(set(all_graph_node_ids) | set(all_code_seeds))
        if ids_to_fetch:
            CHUNK = 500
            for i in range(0, len(ids_to_fetch), CHUNK):
                chunk = ids_to_fetch[i:i + CHUNK]
                placeholders = ",".join("?" * len(chunk))
                rows = ctx.conn.execute(
                    f"SELECT node_id, node_type, file_classification "
                    f"FROM code_nodes WHERE node_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    nid_, ntype_, fclass_ = row
                    if ntype_:
                        node_type_by_id[nid_] = ntype_
                    if fclass_:
                        seed_file_classification[nid_] = fclass_

        cis = bfs_propagate(
            ctx.graph,
            all_code_seeds,
            high_confidence=high_conf,
            low_confidence_seed_map=low_conf,
            seed_file_classification=seed_file_classification,
            node_type_by_id=node_type_by_id,
        )
        logger.info(
            "[runner] BFS: {} SIS seeds, {} propagated nodes",
            len(cis.sis_nodes), len(cis.propagated_nodes),
        )
        _trace("step_6_bfs_raw_cis", {
            "sis_seeds": [
                {"node_id": k, "depth": v.depth, "source_seed": v.source_seed}
                for k, v in cis.sis_nodes.items()
            ],
            "propagated_nodes": [
                {"node_id": k, "depth": v.depth,
                 "causal_chain": v.causal_chain, "source_seed": v.source_seed}
                for k, v in cis.propagated_nodes.items()
            ],
        })
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
    # Step 6.5 — Graph Collapse: CONTAINS sub-tree aggregation
    # Runs after BFS, before LLM #4, to reduce prompt token count.
    #
    # Skipped on cache hit; otherwise runs and the result is cached.
    # ------------------------------------------------------------------
    if not _bfs_cis_cache_hit and variant_flags.enable_bfs and cis.propagated_nodes:
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

    # Cache the post-BFS post-collapse CIS for V6-V7 sharing.
    if (
        not _bfs_cis_cache_hit
        and variant_cache is not None
        and variant_flags.enable_bfs
        and all_code_seeds
    ):
        variant_cache.put_bfs_cis(cis)

    # ------------------------------------------------------------------
    # Step 7 — Propagation validation (LLM #4, FR-D2)
    # ------------------------------------------------------------------
    llm4_justifications: dict[str, str] = {}
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

        _cached_llm4 = (
            variant_cache.get_llm4_verdicts() if variant_cache is not None else None
        )
        if _cached_llm4 is not None:
            cis, llm4_justifications, llm4_degraded = _cached_llm4
            logger.info(
                "[runner] [cache HIT] llm4_verdicts ({} kept, LLM #4 skipped)",
                len(cis.propagated_nodes),
            )
        else:
            cis, llm4_justifications, llm4_degraded = validate_propagation(
                cis=cis,
                cr_interp=cr_interp,
                node_meta_by_id=node_meta_by_id,
                client=ctx.llm_client,
            )
            if variant_cache is not None:
                variant_cache.put_llm4_verdicts(cis, llm4_justifications, llm4_degraded)
        if llm4_degraded:
            degraded_run = True
        _trace("step_7_llm4_verdicts", {
            "kept_node_ids": list(cis.propagated_nodes.keys()),
            "justifications": dict(llm4_justifications),
            "degraded": llm4_degraded,
        })
        logger.info(
            "[runner] Post-LLM #4 CIS: {} SIS + {} propagated = {} total (degraded={})",
            len(cis.sis_nodes), len(cis.propagated_nodes),
            len(cis.sis_nodes) + len(cis.propagated_nodes),
            llm4_degraded,
        )
    elif variant_flags.enable_propagation_validation:
        logger.info("[runner] Step 7: Propagation validation SKIPPED (no propagated nodes)")

    # ------------------------------------------------------------------
    # Step 7.5 — File-local sibling promotion
    # For each validated qualified node, fetch its in-file siblings via
    # CONTAINS and ask LLM #4 (sibling-batch mode) to admit/reject each.
    # Recovers GT entities that share a file with a confirmed seed — the
    # dominant V7-baseline failure mode (forensic audit: 7/8 missed entities
    # on CR-01, 4/6 on CR-03 lived in already-named files).
    # ------------------------------------------------------------------
    sibling_admitted_count = 0
    if (
        variant_flags.enable_propagation_validation
        and getattr(settings, "enable_sibling_promotion", True)
    ):
        from impactracer.pipeline.graph_bfs import collect_file_local_siblings
        from impactracer.pipeline.traversal_validator import validate_siblings_for_file

        anchors_for_sibling: list[str] = []
        anchor_justifications: dict[str, str] = {}
        # Anchor pool restricted to LLM #2 SIS-confirmed seeds with a
        # NON-EMPTY mechanism_of_impact. The worst per-file overshoot in
        # prior calibrations came from anchors that were SIS-confirmed but
        # whose LLM #2 verdict had no concrete mechanism (e.g. CRUD funcs
        # of an unrelated domain entity). When the anchor's mechanism is
        # articulate, downstream sibling-batch admits
        # share a real contract surface; when it's empty, sibling-batch
        # generalises by domain analogy and over-admits.
        for nid in list(cis.sis_nodes.keys()) + list(cis.propagated_nodes.keys()):
            if "::" not in nid:
                continue
            v2 = sis_justifications.get(nid) or {}
            mechanism = (v2.get("mechanism_of_impact") or "").strip()
            if not mechanism:
                # Anchor failed the "articulate LLM #2 mechanism" gate. Skip.
                continue
            anchors_for_sibling.append(nid)
            j = mechanism or (
                v2.get("justification") or ""
            )
            anchor_justifications[nid] = j

        already_in_cis_set = set(cis.sis_nodes.keys()) | set(cis.propagated_nodes.keys())

        per_file_candidates = collect_file_local_siblings(
            anchor_ids=anchors_for_sibling,
            conn=ctx.conn,
            already_in_cis=already_in_cis_set,
            max_per_file=getattr(settings, "sibling_promotion_max_per_file", 12),
        )

        if per_file_candidates:
            # Fetch metadata for all candidate siblings (one batched SELECT).
            all_sibling_ids = [
                sid for sibs in per_file_candidates.values() for sid, _t, _a in sibs
            ]
            sib_meta: dict[str, dict] = {}
            if all_sibling_ids:
                placeholders = ",".join("?" * len(all_sibling_ids))
                rows = ctx.conn.execute(
                    f"SELECT node_id, node_type, file_path, "
                    f"internal_logic_abstraction, source_code "
                    f"FROM code_nodes WHERE node_id IN ({placeholders})",
                    all_sibling_ids,
                ).fetchall()
                for row in rows:
                    sib_meta[row[0]] = {
                        "node_type": row[1],
                        "file_path": row[2],
                        "internal_logic_abstraction": row[3],
                        "source_code": row[4],
                    }

            # Build per-file anchor list (all CIS anchors in
            # the file, not just the first). This corrects the V1 bug where a
            # bad first-anchor caused LLM #4 to reject legitimate sibling GT
            # entities for off-target reasons.
            anchors_by_file: dict[str, list[tuple[str, str]]] = {}

            # Fetch anchor file_paths via SQLite.
            if anchors_for_sibling:
                placeholders_a = ",".join("?" * len(anchors_for_sibling))
                rows_a = ctx.conn.execute(
                    f"SELECT node_id, file_path FROM code_nodes "
                    f"WHERE node_id IN ({placeholders_a})",
                    anchors_for_sibling,
                ).fetchall()
                anchor_fp = {nid: fp for nid, fp in rows_a if fp}
            else:
                anchor_fp = {}
            for anc in anchors_for_sibling:
                fp = anchor_fp.get(anc, "")
                if not fp:
                    continue
                anchors_by_file.setdefault(fp, []).append(
                    (anc, anchor_justifications.get(anc, ""))
                )

            sibling_justifications: dict[str, str] = {}
            sibling_anchors: dict[str, str] = {}
            sibling_files: dict[str, str] = {}

            for file_path, sibs in per_file_candidates.items():
                if not sibs:
                    continue
                file_anchors = anchors_by_file.get(file_path, [])
                if not file_anchors:
                    # Fallback: synthesize from sib's recorded first-anchor.
                    fallback_anchor = sibs[0][2]
                    file_anchors = [(
                        fallback_anchor,
                        anchor_justifications.get(fallback_anchor, "") or
                        "Anchor was validated upstream; specific justification unavailable.",
                    )]
                pairs = [(sid, stype) for sid, stype, _a in sibs]
                admitted, sibling_degraded = validate_siblings_for_file(
                    file_path=file_path,
                    anchors=file_anchors,
                    siblings=pairs,
                    cr_interp=cr_interp,
                    node_meta_by_id=sib_meta,
                    client=ctx.llm_client,
                )
                if sibling_degraded:
                    degraded_run = True
                primary_anchor = file_anchors[0][0] if file_anchors else sibs[0][2]
                # Per-file cap: truncate per-file admissions to the configured
                # ceiling, preserving LLM #4's emission order (which correlates
                # with confidence — first-emitted siblings tend to be the
                # tightest contract matches).
                per_file_cap = getattr(settings, "sibling_admit_max_per_file", 2)
                if per_file_cap > 0 and len(admitted) > per_file_cap:
                    logger.info(
                        "[runner] Step 7.5: per-file cap — truncating {} admissions "
                        "to top-{} for {}", len(admitted), per_file_cap, file_path,
                    )
                    capped_items = list(admitted.items())[:per_file_cap]
                else:
                    capped_items = list(admitted.items())
                for sib_id, sib_just in capped_items:
                    sibling_justifications[sib_id] = sib_just
                    sibling_anchors[sib_id] = primary_anchor
                    sibling_files[sib_id] = file_path

            # Per-CR cap: enforce per-CR admission ceiling. Drops the lowest-
            # order admissions when the total exceeds the cap. The order is
            # insertion order from the per-file loop, which roughly tracks
            # anchor strength.
            per_cr_cap = getattr(settings, "sibling_admit_max_per_cr", 5)
            if per_cr_cap > 0 and len(sibling_justifications) > per_cr_cap:
                logger.info(
                    "[runner] Step 7.5: per-CR cap — truncating {} sibling "
                    "admissions to top-{}",
                    len(sibling_justifications), per_cr_cap,
                )
                kept_ids = list(sibling_justifications.keys())[:per_cr_cap]
                sibling_justifications = {k: sibling_justifications[k] for k in kept_ids}
                sibling_anchors = {k: sibling_anchors[k] for k in kept_ids}
                sibling_files = {k: sibling_files[k] for k in kept_ids}

            # Inject admitted siblings into the CIS as propagated nodes with
            # a CONTAINS causal chain (severity_for_chain maps CONTAINS -> Rendah,
            # which is fine — these are admitted on contract similarity, not
            # primary behavioural change).
            for sib_id, sib_just in sibling_justifications.items():
                if sib_id in cis.sis_nodes or sib_id in cis.propagated_nodes:
                    continue
                anchor = sibling_anchors[sib_id]
                cis.propagated_nodes[sib_id] = NodeTrace(
                    depth=1,
                    causal_chain=["CONTAINS"],
                    path=[anchor, sib_id],
                    source_seed=anchor,
                    low_confidence_seed=False,
                    justification=sib_just,
                    justification_source="llm4_sibling",
                )
                llm4_justifications[sib_id] = sib_just
                # Also register node_type / file_path so downstream maps see it.
                meta = sib_meta.get(sib_id, {})
                if meta:
                    pass  # node_types map is rebuilt below
                sibling_admitted_count += 1

            logger.info(
                "[runner] Step 7.5: sibling promotion admitted {} new siblings "
                "across {} files (anchors={})",
                sibling_admitted_count,
                sum(1 for sibs in per_file_candidates.values() if sibs),
                len(anchors_for_sibling),
            )
            _trace("step_7p5_sibling_promotion", {
                "anchors": list(anchors_for_sibling),
                "candidates_per_file": {
                    fp: [sid for sid, _t, _a in sibs]
                    for fp, sibs in per_file_candidates.items()
                },
                "admitted": list(sibling_justifications.keys()),
                "justifications": sibling_justifications,
            })

    # Generate synthetic justifications for propagated nodes not processed by
    # LLM #4 (BFS-only variants and exempt-edge auto-keeps). These are
    # deterministic renderings of the BFS chain, not LLM-generated text.
    for nid, trace in cis.propagated_nodes.items():
        if nid in llm4_justifications:
            continue
        chain = " -> ".join(trace.causal_chain) if trace.causal_chain else "(direct)"
        if (
            trace.depth == 1
            and trace.causal_chain
            and trace.causal_chain[-1] in PROPAGATION_VALIDATION_EXEMPT_EDGES
        ):
            llm4_justifications[nid] = (
                f"Direct {trace.causal_chain[-1]} contract from {trace.source_seed} - "
                f"auto-admitted exempt edge."
            )
        else:
            llm4_justifications[nid] = (
                f"BFS-propagated via {chain} (depth {trace.depth}) from "
                f"{trace.source_seed}. No semantic validation performed in this "
                f"variant; chain is structural only."
            )

    # Attach distributed justifications to every NodeTrace.
    # LLM #5 never re-justifies individual nodes — they carry verbatim
    # text from whichever LLM validated them (#2/#3/#4).
    from dataclasses import replace as _dc_replace

    for sid, trace in list(cis.sis_nodes.items()):
        # Priority: LLM #2 mechanism (richest) > LLM #2 justification
        # > LLM #3 justification (when seed came via doc resolution).
        v2 = sis_justifications.get(sid)
        if v2:
            cis.sis_nodes[sid] = _dc_replace(
                trace,
                justification=v2.get("mechanism_of_impact") or v2.get("justification") or "",
                justification_source="llm2_sis",
                function_purpose=v2.get("function_purpose", ""),
                mechanism_of_impact=v2.get("mechanism_of_impact", ""),
            )
            continue
        v3 = trace_justifications.get(sid)
        if v3:
            cis.sis_nodes[sid] = _dc_replace(
                trace,
                justification=v3,
                justification_source="llm3_trace",
            )
            continue
        # No LLM verdict associated (e.g. direct seed under V0-V3).
        cis.sis_nodes[sid] = _dc_replace(
            trace,
            justification=(
                "Direct retrieval seed (no LLM validation in this variant)."
            ),
            justification_source="retrieval_only",
        )

    for pid, trace in list(cis.propagated_nodes.items()):
        v4 = llm4_justifications.get(pid, "")
        # Determine source: auto_exempt vs llm4_propagation vs synthetic.
        if (
            trace.depth == 1
            and trace.causal_chain
            and trace.causal_chain[-1] in PROPAGATION_VALIDATION_EXEMPT_EDGES
            and variant_flags.enable_propagation_validation
        ):
            src = "auto_exempt"
        elif variant_flags.enable_propagation_validation:
            src = "llm4_propagation"
        else:
            src = "bfs_only"
        cis.propagated_nodes[pid] = _dc_replace(
            trace,
            justification=v4,
            justification_source=src,
        )

    # ------------------------------------------------------------------
    # Step 8 — Backlinks + Context (FR-E1, FR-E2)
    # ------------------------------------------------------------------
    logger.info("[runner] Step 8: Build context")
    all_node_ids = cis.all_node_ids()

    # Build node_types and node_file_paths maps — candidates plus SQLite for propagated nodes.
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

    backlinks = fetch_backlinks(all_node_ids, node_types, ctx.conn, settings.top_k_backlinks_per_node)
    snippets = fetch_snippets(all_node_ids, ctx.conn, doc_col=ctx.doc_col)

    # Use raw_reranker_score for context truncation priority — the normalized
    # score would mis-rank nodes modified by the affinity gate.
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
    # LLM #5 produces only executive_summary + documentation_conflicts.
    # impacted_entities is built deterministically from the full validated CIS;
    # every validated node appears in the report regardless of prompt truncation.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 9: Synthesize report (aggregator-only LLM #5)")

    impacted_entities_deterministic = build_deterministic_impacted_entities(
        cis=cis,
        node_types=node_types,
        node_file_paths=node_file_paths,
        justifications_extra=llm4_justifications,
        backlinks=backlinks,
    )

    # LLM #5: aggregator-only.
    forced_inclusion = getattr(variant_flags, "force_include_all_cis_nodes", False)
    if forced_inclusion or not impacted_entities_deterministic:
        # Forced-inclusion variant or empty CIS: skip the LLM call entirely.
        synthesis: LLMSynthesisOutput = build_minimal_summary(
            text=(
                f"Analysis produced {len(impacted_entities_deterministic)} "
                f"impacted nodes across the validated CIS."
            ),
            conflicts=[],
        )
        logger.info(
            "[runner] Step 9: LLM #5 SKIPPED ({})",
            "forced inclusion" if forced_inclusion else "empty CIS",
        )
    else:
        try:
            synthesis = synthesize_summary(context, ctx.llm_client)
        except Exception as exc:
            logger.error(
                "[runner] Step 9: LLM #5 failed after retries: {} - "
                "using minimal summary (degraded)",
                exc,
            )
            degraded_run = True
            synthesis = build_minimal_summary(
                text=(
                    "[degraded] Synthesis LLM call failed; impacted_nodes "
                    "are still populated from validated CIS with "
                    "per-node justifications."
                ),
                conflicts=[],
            )

    # Deterministic scope and analysis_mode.
    computed_scope = _compute_scope(cis, settings)
    bfs_ran = variant_flags.enable_bfs and len(cis.propagated_nodes) > 0
    analysis_mode = "retrieval_plus_propagation" if bfs_ran else "retrieval_only"

    # Step 9 augmentation: extra impacted_files come from File-type CIS nodes
    # (and bare path nodes) that we filtered out of impacted_entities. Their
    # paths still belong in the file-level report.
    extra_file_paths: list[str] = []
    for nid in cis.combined().keys():
        nt = node_types.get(nid, "")
        if nt == "File":
            fp = node_file_paths.get(nid) or nid
            if fp:
                extra_file_paths.append(fp)
        elif "::" not in nid:
            # Bare path CIS node (shouldn't happen often post-filter, but
            # belt-and-braces). Treat the node_id itself as a file path.
            extra_file_paths.append(nid)

    report = assemble_impact_report(
        summary=synthesis,
        impacted_entities=impacted_entities_deterministic,
        estimated_scope=computed_scope,
        analysis_mode=analysis_mode,
        degraded_run=degraded_run,
        extra_impacted_file_paths=extra_file_paths,
    )

    _trace("final_report", report.model_dump())

    elapsed = time.perf_counter() - t_start
    logger.info(
        "[runner] Analysis complete: {} impacted entities ({} files), "
        "scope={}, mode={}, elapsed={:.1f}s, LLM calls={}, degraded={}",
        len(report.impacted_entities),
        len(report.impacted_files),
        report.estimated_scope,
        report.analysis_mode,
        elapsed,
        ctx.llm_client.call_counter,
        degraded_run,
    )
    return report
