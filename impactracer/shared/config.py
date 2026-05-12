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
