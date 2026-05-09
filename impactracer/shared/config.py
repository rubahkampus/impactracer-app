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
    top_k_per_query: int = 15
    # top_k_rrf_pool: candidates entering cross-encoder after RRF fusion (FF-1)
    top_k_rrf_pool: int = 50
    # max_admitted_seeds: hard cap on seeds admitted to CIS after reranking
    max_admitted_seeds: int = 15
    rrf_k: int = 60

    # ---- Pre-Validation Gates (FR-C4) ------------------------------
    # Crucible Fix 9: score floor demoted to a sanity-only gate.
    # The previous 0.15 default was a calibration-fragile magic number
    # masking retrieval-quality issues; LLM #2 is the real precision gate.
    # -2.0 admits all candidates whose cross-encoder logit is at or above
    # the typical "irrelevant-candidate" floor for BGE-reranker-v2-m3,
    # while still rejecting structurally broken candidates whose logit is
    # strongly negative (e.g. encoded-as-binary or empty-text candidates).
    min_reranker_score_for_validation: float = -2.0
    # Crucible Fix 9: density threshold raised 0.35 -> 0.50.
    # The previous 0.35 was unjustified and rejected legitimate candidates
    # on dense service files (e.g. a 100-symbol service file triggers 35%
    # density at 35 candidates, but LLM #2 may still confirm those if the
    # CR is broad). 0.50 retains the gate's structural intent (catch
    # pathological cases where the retrieval is over-concentrating into
    # one file) while admitting more cases for LLM-level filtering.
    plausibility_gate_density_threshold: float = 0.50
    # Crucible Fix 9: plausibility_gate_max_per_file removed entirely.
    # The previous default of 2 was unjustified and arbitrarily capped
    # genuine multi-symbol impacts. Density-based protection above is
    # sufficient.

    # ---- BFS --------------------------------------------------------
    bfs_global_max_depth: int = 3
    bfs_high_conf_top_n: int = 5

    # ---- Context Assembly ------------------------------------------
    llm_max_context_tokens: int = 100_000
    synthesis_system_prompt_tokens: int = 1200
    output_reserve_tokens: int = 2000
    top_k_backlinks_per_node: int = 3

    # ---- Scope thresholds (F-NEW-5) ---------------------------------
    # Phase 2.9: _compute_scope thresholds are now calibrated parameters
    # rather than hardcoded magic numbers. Defaults match the thesis-defined
    # thresholds but can be overridden via env vars post-evaluation.
    # The indexed citrakara codebase has ~400 code nodes; 5/15 (old) was
    # too tight. These wider defaults better reflect a real-world codebase.
    scope_local_max: int = 10    # ≤10 nodes → terlokalisasi
    scope_medium_max: int = 30   # 11-30 nodes → menengah; >30 → ekstensif

    # ---- Audit and Evaluation ---------------------------------------
    # Crucible Fix 4: eval_k_values removed. Bounded F1@K metrics are no
    # longer computed; metrics are pure set-level (precision_set,
    # recall_set, f1_set) plus rank-aware r_precision. See
    # impactracer/evaluation/metrics.py.
    llm_audit_log_path: str = "./data/llm_audit.jsonl"
    locked_parameters_path: str = "./data/locked_parameters.json"
    alpha: float = 0.05


def get_settings() -> Settings:
    """Construct a :class:`Settings` instance from the current env."""
    return Settings()  # type: ignore[call-arg]
