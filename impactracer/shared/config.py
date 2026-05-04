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
    llm_max_output_tokens: int = 4096
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
    top_k_per_query: int = 15
    # top_k_rrf_pool: candidates entering cross-encoder after RRF fusion (FF-1)
    top_k_rrf_pool: int = 50
    # max_admitted_seeds: hard cap on seeds admitted to CIS after reranking
    max_admitted_seeds: int = 15
    rrf_k: int = 60

    # ---- Pre-Validation Gates (FR-C4) ------------------------------
    min_reranker_score_for_validation: float = 0.0
    plausibility_gate_density_threshold: float = 0.35
    plausibility_gate_max_per_file: int = 2

    # ---- BFS --------------------------------------------------------
    bfs_global_max_depth: int = 3
    bfs_high_conf_top_n: int = 5

    # ---- Context Assembly ------------------------------------------
    llm_max_context_tokens: int = 100_000
    synthesis_system_prompt_tokens: int = 1200
    output_reserve_tokens: int = 2000
    top_k_backlinks_per_node: int = 3

    # ---- Audit and Evaluation ---------------------------------------
    llm_audit_log_path: str = "./data/llm_audit.jsonl"
    locked_parameters_path: str = "./data/locked_parameters.json"
    eval_k_values: list[int] = [5, 10]
    alpha: float = 0.05


def get_settings() -> Settings:
    """Construct a :class:`Settings` instance from the current env."""
    return Settings()  # type: ignore[call-arg]
