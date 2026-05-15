"""Settings class backed by ``pydantic-settings``.

All parameters are locked pre-evaluation; values here represent the
defaults. Overrides come from environment variables (see .env.template).

Reference: 11_configuration_and_cli.md §1.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for ImpacTracer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM (OpenRouter — exclusive provider) --------------------------
    openrouter_api_key: str = ""
    llm_model: str = "google/gemini-2.5-flash"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_max_output_tokens: int = 65536
    llm_retry_max_attempts: int = 10
    llm_retry_base_backoff: float = 2.0

    # ---- Embedding and Reranking -----------------------------------
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_batch_size: int = 32
    embedding_max_length: int = 512

    # ---- Storage ----------------------------------------------------
    db_path: str = "./data/impactracer.db"
    chroma_path: str = "./data/chroma_store"

    # ---- Indexer ----------------------------------------------------
    top_k_traceability: int = 5
    min_traceability_similarity: float = 0.40
    degenerate_embed_min_length: int = 50

    # ---- Retrieval --------------------------------------------------
    # top_k_per_query: per individual dense/BM25 query call (4 paths × N queries)
    # Sprint 13-W2A: widened from 15 to 30. Diagnostics on the calibration set
    # showed only 5 of 21 in-index GT entities entered the RRF pool at width 15.
    top_k_per_query: int = 30
    # top_k_rrf_pool: pool entering cross-encoder after RRF (was 50)
    # Sprint 13-W2A: widened to 200. Cross-encoder rerank on 200 candidates
    # adds < 1 s on the existing model and gives the rerank-top-15 funnel a
    # meaningful selection problem to solve.
    top_k_rrf_pool: int = 200
    # max_admitted_seeds: hard cap on seeds admitted to CIS after reranking
    max_admitted_seeds: int = 15
    rrf_k: int = 60

    # ---- Sprint 13-W2B: raw CR multilingual bridge ----------------
    # When True, the retriever runs one additional dense query against the
    # code collection using the raw (pre-interpretation) CR text. BGE-M3 is
    # multilingual; an Indonesian CR can natively reach English-identifier
    # code that no LLM #1 search_query happens to mention.
    enable_raw_cr_dense_pass: bool = True
    raw_cr_dense_top_k: int = 60

    # ---- Apex Crucible Proposal B: per-layer code retrieval --------
    # When CRInterpretation.layered_search_queries is populated, the retriever
    # runs an additional pass per architectural layer (api_route, page_component,
    # ui_component, utility, type_definition) against the code collection
    # scoped by file_classification. Each layer contributes up to
    # `per_layer_top_k` candidates to a new RRF path called "layered_code".
    # This guarantees no layer is starved when LLM #1's flat search_queries
    # are biased toward one architectural plane.
    per_layer_top_k: int = 12

    # ---- Apex Crucible Proposal C: graph-aware label-propagation rerank --
    # After cross-encoder rerank scores the full RRF pool, a 2-iteration label
    # propagation over the structural graph blends a per-node "graph_score"
    # with the cross-encoder score before the top-K truncation. Personalization
    # is the top-N cross-encoder candidates (no extra LLM call).
    #
    # **Disabled by default per Sprint 15 postmortem.** Calibration on 5 CRs
    # showed Proposal C trades file-level F1 for entity-level F1 with no
    # configuration that wins both metrics. Mode B added zero GT entities on
    # citrakara; mode A's entity-level precision tightening came at the cost
    # of dropping file TPs from the rerank pool. Keep the code path and tests
    # for future codebases where the structural graph more densely connects
    # CR-described seeds to GT files (e.g. monorepos where forms directly
    # import schemas). Re-enable by setting enable_graph_rerank=True or via
    # GRAPH_RERANK_ALPHA env var override.
    enable_graph_rerank: bool = False
    graph_rerank_alpha: float = 0.7              # weight on cross-encoder; (1-alpha) on graph
    graph_rerank_iterations: int = 2             # number of label-propagation rounds
    graph_rerank_personalization_top_n: int = 5  # seeds for PPR (top-N by cross-encoder)
    graph_rerank_add_top_n: int = 10             # mode B: add this many graph-discovered candidates
    graph_rerank_add_min_score: float = 0.10     # mode B: minimum normalized graph_score to admit

    # ---- Apex Crucible Proposal A: file-local sibling promotion -----
    # After LLM #4 validation, the runner enumerates every qualified sibling
    # of each validated node within the same file (via CONTAINS) and lets
    # LLM #4 admit/reject each sibling using the anchor's justification as
    # context. Recovers GT entities that share a file with a confirmed seed
    # (the dominant failure mode at the V7 baseline: 7/8 missed entities on
    # CR-01, 4/6 on CR-03 live in already-named files).
    #
    # Apex V3: per-file/per-CR admission caps prevent sibling-promotion
    # overshoot. Forensic on V2 calibration: CR-04 admitted 10 siblings in
    # one file (escrow repo) because the anchors were SIS-confirmed CRUD
    # functions of a single domain entity; LLM #4 correctly recognized all
    # 10 as similarly-shaped, but the GT only named the one caller. Capping
    # admissions truncates this overshoot while preserving the recall win
    # on CRs where 1-2 siblings per file are the right answer.
    enable_sibling_promotion: bool = True
    sibling_promotion_max_per_file: int = 12        # candidate ceiling per file
    # Apex V4: per-file admission cap softened from 2 (V3) to 4. V3 forensics
    # showed cap=2 dropped legitimate admissions on CR-03 and removed the TP
    # camouflage on CR-04. Cap=4 prevents the worst overshoot (CR-04 V2 had
    # 10 admits in one file) while preserving the 1-3 admits per file that
    # drive recall gains on CR-01 / CR-03. Per-CR cap removed (set to 0 =
    # disabled) — global throttling was too blunt on a 5-CR mix.
    sibling_admit_max_per_file: int = 4
    sibling_admit_max_per_cr: int = 0               # 0 = no global cap

    # ---- Sprint 13-W2C: traceability-matrix pool seeding -----------
    # After dense_doc retrieval, query doc_code_candidates for code-nodes
    # linked to those doc-chunks above this threshold and inject them into
    # the RRF pool with a synthetic rank. Promotes the offline traceability
    # precomputation from a rerank +0.1 bonus into a pool-seeding signal.
    enable_traceability_pool_seeding: bool = True
    traceability_seed_top_k_per_doc: int = 5
    traceability_seed_min_score: float = 0.40
    traceability_seed_synthetic_rank: int = 5

    # ---- Pre-Validation Gates (FR-C4) ------------------------------
    # Score floor is a sanity-only gate (-2.0 admits all candidates above
    # the BGE-reranker-v2-m3 "irrelevant" floor). LLM #2 is the real precision gate.
    min_reranker_score_for_validation: float = -2.0
    # Density threshold: rejects candidates when a single file exceeds this
    # fraction of the total pool. Density-only; no per-file count cap.
    plausibility_gate_density_threshold: float = 0.50

    # ---- BFS --------------------------------------------------------
    bfs_global_max_depth: int = 3
    bfs_high_conf_top_n: int = 5

    # ---- Context Assembly ------------------------------------------
    llm_max_context_tokens: int = 100_000
    synthesis_system_prompt_tokens: int = 1200
    output_reserve_tokens: int = 2000
    top_k_backlinks_per_node: int = 3

    # ---- Scope thresholds -------------------------------------------
    scope_local_max: int = 10    # ≤10 nodes → terlokalisasi
    scope_medium_max: int = 30   # 11-30 → menengah; >30 → ekstensif

    # ---- Audit and Evaluation ---------------------------------------
    llm_audit_log_path: str = "./data/llm_audit.jsonl"
    locked_parameters_path: str = "./data/locked_parameters.json"
    alpha: float = 0.05


def get_settings() -> Settings:
    """Construct a :class:`Settings` instance from the current env."""
    return Settings()  # type: ignore[call-arg]
