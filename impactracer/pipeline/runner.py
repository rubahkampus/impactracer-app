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
    severity_for_chain,
)
from impactracer.shared.models import (
    CISResult,
    Candidate,
    CRInterpretation,
    ImpactReport,
    ImpactedEntity,
    LLMSynthesisOutput,
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
    """Crucible Fix 14: soft-fix change_type / affected_layers coherence.

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

    N8 fix: estimated_scope must NOT be hallucinated by LLM #5 from a pruned
    synthesis context.  The synthesizer sees only the context-budget subset;
    counting nodes from the full CIS gives a provably correct scope that is
    reproducible and thesis-defensible.

    Phase 2.9 (F-NEW-5): thresholds are now read from Settings so they can
    be calibrated post-evaluation without changing source code.

    Thresholds (default, from Settings.scope_local_max / scope_medium_max):
        terlokalisasi  : ≤ scope_local_max total CIS nodes   (default 10)
        menengah       : scope_local_max+1 – scope_medium_max (default 11-30)
        ekstensif      : > scope_medium_max                   (default >30)

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
) -> ImpactReport:
    """End-to-end online analysis for one CR.

    Crucible E2E Task 2b: when ``trace_sink`` is provided (a mutable dict),
    the runner populates it with every step's result so the CLI can write
    ``impact_report_full.json`` for academic auditability. The trace_sink
    dict is mutated in-place; passing None disables tracing.

    Trace keys populated:
      step_1_interpretation,
      step_2_rrf_pool,
      step_3_reranked,
      step_3_gates_survivors,
      step_4_llm2_verdicts,
      step_5_resolutions,
      step_5b_llm3_verdicts,
      step_6_bfs_raw_cis,
      step_7_llm4_verdicts,
      final_report.

    Blueprint §4.
    """
    from impactracer.evaluation.variant_flags import VariantFlags

    if variant_flags is None:
        variant_flags = VariantFlags.v7_full()

    # Crucible E2E Task 2b: full-traceability sink. None means tracing
    # disabled (no-op). Otherwise mutate the dict in-place at each step.
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

    _trace("step_1_interpretation", cr_interp.model_dump())

    if not cr_interp.is_actionable:
        logger.info("[runner] CR is NOT actionable — short-circuiting")
        rej = _minimal_rejection_report(
            cr_interp.actionability_reason or "CR was not actionable"
        )
        _trace("final_report", rej.model_dump())
        return rej

    # Crucible Fix 14: enforce change_type / affected_layers coherence.
    cr_interp = _validate_cr_interpretation_coherence(cr_interp)

    # Aggregate degraded flag across all LLM batches in this run.
    degraded_run: bool = False

    # ------------------------------------------------------------------
    # Step 2 — Adaptive RRF Hybrid Search (FR-C1, FR-C2)
    # Returns top_k_rrf_pool (=50) candidates for cross-encoder input.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 2: Hybrid search (variant={})", variant_flags.variant_id)
    candidates = hybrid_search(cr_interp, ctx, settings)
    logger.info("[runner] Post-RRF pool: {}", len(candidates))
    _trace("step_2_rrf_pool", [
        {"node_id": c.node_id, "collection": c.collection,
         "rrf_score": c.rrf_score, "file_path": c.file_path}
        for c in candidates
    ])

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
        _trace("step_3_reranked", [
            {"node_id": c.node_id, "collection": c.collection,
             "rrf_score": c.rrf_score, "reranker_score": c.reranker_score,
             "raw_reranker_score": c.reranker_score, "file_path": c.file_path,
             "name": c.name}
            for c in candidates
        ])

        # B4: Snapshot raw cross-encoder scores BEFORE normalization so the
        # score floor gate operates on absolute quality, not rank-within-15.
        for c in candidates:
            c.raw_reranker_score = c.reranker_score

        # Crucible Fix 12.2: traceability bonus on raw_reranker_score for
        # code candidates that the offline doc-code matrix associates with
        # any retrieved doc chunk.
        candidates = apply_traceability_bonus(candidates, ctx.conn)

        # Crucible Fix 13: additive negative filter on raw_reranker_score
        # for candidates whose name/snippet contains an out-of-scope
        # operation. Mathematically sound across the entire real line of
        # cross-encoder logits (additive, not multiplicative).
        candidates = apply_negative_filter(
            candidates, cr_interp.out_of_scope_operations
        )

        # Re-sort after bonus/penalty so downstream score-floor gate sees
        # the corrected ordering.
        candidates.sort(key=lambda c: c.raw_reranker_score, reverse=True)

        # FF-2: min-max normalize to [0,1] for relative sorting inside the
        # pipeline (gates, context priority).  Raw scores preserved above.
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
    _trace("step_3_gates_survivors", [
        {"node_id": c.node_id, "collection": c.collection,
         "raw_reranker_score": c.raw_reranker_score,
         "reranker_score": c.reranker_score, "file_path": c.file_path,
         "merged_doc_ids": list(c.merged_doc_ids)}
        for c in candidates
    ])

    if not candidates:
        logger.warning("[runner] Zero candidates after gates — returning empty report")
        return _minimal_rejection_report("All candidates rejected by pre-validation gates")

    # ------------------------------------------------------------------
    # Step 4 — SIS Validation (LLM #2, FR-C5)
    # Batched: max 5 candidates per LLM call (Batching Mandate).
    # Returns (confirmed_ids, score_ordered_ids) for metric ranking (N9).
    # ------------------------------------------------------------------
    # Crucible Fix 1+3: validator returns (ids, justifications, degraded).
    # justifications maps confirmed node_id -> {function_purpose,
    # mechanism_of_impact, justification} — used to populate NodeTrace.
    sis_justifications: dict[str, dict[str, str]] = {}
    if variant_flags.enable_sis_validation:
        logger.info("[runner] Step 4: SIS validation (batched, fail-closed)")
        sis_ids, sis_justifications, llm2_degraded = validate_sis_candidates_batched(
            cr_interp, candidates, ctx.llm_client
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

    # Phase 1 (E-6): Build code_node_ids set once and pass it to the resolver
    # so resolve_doc_to_code can skip the full table scan on every call.
    # During a 20-CR ablation run (8 variants × 20 CRs = 160 calls), this
    # eliminates 160 full-table scans. We build it here in runner.py because
    # the runner already has the SQLite connection and this set is stable for
    # the lifetime of a single pipeline context.
    _code_node_ids: set[str] = {
        row[0]
        for row in ctx.conn.execute("SELECT node_id FROM code_nodes").fetchall()
    }

    # Phase 1 (E-NEW-3): resolve_doc_to_code now returns (resolutions,
    # direct_code_seeds) — the previously dead doc_to_code_map third return
    # value has been removed from the function signature.
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
    # ------------------------------------------------------------------
    # low_conf tracks which resolved code seeds have PARTIAL decision.
    low_conf: dict[str, bool] = {}
    # Crucible Fix 3: trace justifications keyed by code_id (LLM #3).
    trace_justifications: dict[str, str] = {}

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

        # Crucible Fix 1+3: trace validator returns 4-tuple
        # (seeds, low_conf, justifications, degraded). Justifications map
        # code_id -> LLM #3 verdict justification text (used to populate
        # NodeTrace.justification with source='llm3_trace').
        validated_code_seeds, low_conf, trace_justifications, llm3_degraded = (
            validate_trace_resolutions(
                resolutions=resolutions,
                doc_text_by_id=doc_text_by_id,
                code_meta_by_id=code_meta_by_id,
                client=ctx.llm_client,
                cr_interp=cr_interp,
            )
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
    # ------------------------------------------------------------------
    if variant_flags.enable_bfs and all_code_seeds:
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

        # Crucible Fix 11: bulk-fetch (file_classification, node_type) for
        # every node in the graph and seeds. The BFS uses this to apply the
        # UTILITY-CALLS depth cutoff and per-node-type fan-in cap.
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
    # Crucible Fix 3: justification map populated by LLM #4 for kept nodes
    # AND by synthetic 'auto_exempt' strings for IMPLEMENTS/DEFINES_METHOD/
    # TYPED_BY single-hop edges (assigned inside validate_propagation).
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

        # Crucible Fix 1+3: validate_propagation now returns
        # (filtered_cis, justifications, degraded). Justifications map
        # node_id -> LLM #4 verdict justification text.
        cis, llm4_justifications, llm4_degraded = validate_propagation(
            cis=cis,
            cr_interp=cr_interp,
            node_meta_by_id=node_meta_by_id,
            client=ctx.llm_client,
        )
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

    # Crucible Fix 3: BFS-only variants (V6) and exempt-edge auto-keeps
    # need a justification too. Generate synthetic chain-described strings
    # for any propagated node that was NOT processed by LLM #4. This is
    # NOT a retrospective LLM hallucination — it is a deterministic
    # rendering of the BFS chain that the runner has access to.
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

    # ------------------------------------------------------------------
    # Crucible Fix 3: attach distributed justifications to every NodeTrace
    # so context_builder can render them and synthesizer.build_deterministic
    # _impacted_nodes can copy them verbatim. This is the single source of
    # truth — LLM #5 never re-justifies.
    # ------------------------------------------------------------------
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
    #
    # Crucible Fix 3 (Full Demotion): LLM #5 is called only for the
    # executive summary + documentation_conflicts. The impacted_nodes
    # array is built deterministically from the FULL validated CIS using
    # pre-recorded justifications — Crucible Amendment 2 mandate: every
    # validated node appears in the report, regardless of whether it was
    # truncated from the LLM #5 prompt window.
    # ------------------------------------------------------------------
    logger.info("[runner] Step 9: Synthesize report (aggregator-only LLM #5)")

    # Build deterministic impacted_nodes FIRST (Amendment 2 invariant).
    # This is independent of any prompt-window truncation in build_context.
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
            # Crucible Amendment 1: fail-closed at LLM #5 too — if the
            # synthesis call exhausts retries, fall back to a minimal
            # summary. The deterministic impacted_nodes still carry the
            # full per-node justifications.
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

    report = assemble_impact_report(
        summary=synthesis,
        impacted_entities=impacted_entities_deterministic,
        estimated_scope=computed_scope,
        analysis_mode=analysis_mode,
        degraded_run=degraded_run,
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
