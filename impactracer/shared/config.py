"""Settings class backed by ``pydantic-settings``.

All parameters are locked pre-evaluation; values here represent the
defaults. Overrides come from environment variables (see .env.template).

Reference: 11_configuration_and_cli.md §1.
"""

from __future__ import annotations

import os
from pathlib import Path

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

    # ---- Step 2: Retrieval (RRF) ------------------------------------
    # top_k_per_query: candidates kept per individual dense/BM25 query call.
    top_k_per_query: int = 30
    # top_k_rrf_pool: pool entering cross-encoder after RRF fusion. A wide
    # pool (200) gives the rerank-top-15 funnel a meaningful selection
    # problem and keeps the K-widening diagnostic well-resourced.
    top_k_rrf_pool: int = 200
    # max_admitted_seeds: hard cap on seeds admitted to SIS after reranking.
    max_admitted_seeds: int = 15
    rrf_k: int = 60

    # ---- Step 2: Raw-CR multilingual bridge -------------------------
    # When True, the retriever runs one additional dense query against the
    # code collection using the raw (pre-interpretation) CR text. BGE-M3 is
    # multilingual; an Indonesian CR can natively reach English-identifier
    # code that no LLM #1 search_query happens to mention.
    enable_raw_cr_dense_pass: bool = True
    raw_cr_dense_top_k: int = 60

    # ---- Step 2: Per-layer code retrieval ---------------------------
    # When CRInterpretation.layered_search_queries is populated, the retriever
    # runs an additional pass per architectural layer (api_route, page_component,
    # ui_component, utility, type_definition) against the code collection
    # scoped by file_classification. Each layer contributes up to
    # `per_layer_top_k` candidates to a new RRF path called "layered_code".
    # This guarantees no layer is starved when LLM #1's flat search_queries
    # are biased toward one architectural plane.
    per_layer_top_k: int = 12

    # ---- Step 7.5: File-local sibling promotion ---------------------
    # After LLM #4 validation, the runner enumerates every qualified sibling
    # of each validated node within the same file (via CONTAINS) and lets
    # LLM #4 admit/reject each sibling using the anchor's justification as
    # context. Recovers GT entities that share a file with a confirmed seed
    # (a frequent failure mode: many missed entities live in already-named
    # files).
    #
    # The per-file and per-CR admission caps prevent sibling-promotion
    # overshoot. Without caps, a sibling batch can over-admit when the
    # anchors are SIS-confirmed CRUD functions of a single domain entity
    # and LLM #4 correctly recognises every same-shape function as
    # similarly impacted, even when GT only names the one caller.
    # Eval-only: force the CR change_type to this value, overriding LLM #1's
    # classification. None (default) = use the LLM classification. Set to
    # "ADDITION"|"MODIFICATION"|"DELETION" to drive the change_type-dependent
    # treatments (RRF path weights, LLM #2/#3 framing) from a known label
    # instead. The evaluate harness sets this per-CR from the GT label when
    # --force-change-type-from-gt is passed. Not for production analyze.
    force_change_type: str | None = None


    enable_sibling_promotion: bool = True
    sibling_promotion_max_per_file: int = 12        # candidate ceiling per file
    # Per-file admission cap: 4 strikes the balance between preventing
    # worst-case overshoot (~10 admits per file) and preserving the 1-3
    # admits per file that drive legitimate recall gains.
    sibling_admit_max_per_file: int = 4
    sibling_admit_max_per_cr: int = 0               # 0 = no global cap

    # ---- Step 6.8: Weight-decay propagation prune -------------------
    # Deterministic flood control: after BFS + sibling expansion, rank the
    # propagated pool by edge-weight-aware decay (constants.propagation_decay_
    # score: prod(edge_weight)/(1+depth), weights = measured edge productivity)
    # and keep only the top-K. SIS seeds are never scored or cut. Always-on at
    # V6+ so V7's LLM #4 validates a shrunk, higher-precision pool. Selected
    # over PPR / semantic-cosine by an offline V6 sweep (removed ~68% of
    # propagated FPs retaining ~70% of TPs, more surgically than either).
    # Detachable: set enable_propagation_weight_prune=False to restore the raw
    # unpruned flood. 0 disables the cut (scores only).
    # K=10 is the RECALL-SAFE floor found by a live K-sweep on the V6 BFS pool:
    # BFS-GT recall is 100% at K>=10 and drops below it (the binding
    # CR, ADD-2, has its last BFS-GT at rank 9). Earlier K=20 was the safe
    # choice; K=10 is the tight choice — recall-identical on this corpus, with
    # marginally better precision and ~half the BFS budget on flood CRs. NOTE
    # the floor is set by one CR via insertion order, so it is corpus-specific;
    # raise toward 12-15 for a safety margin if generalising. Edge-weight tuning
    # canNOT lower it further: the binding CR is single-edge (all RENDERS) so GT
    # and noise are score-tied — a tie-breaker problem, not a weight problem.
    # As a PRE-V7 filter the goal is to hand V7 the full recall with less noise,
    # not to maximise V6 F1 (which peaks ~0.29 at K=3 by shedding GT). This is
    # an efficiency layer at zero recall cost, not an accuracy knob.
    # Env-overridable (bare name, like TOP_K_RRF_POOL): PROPAGATION_PRUNE_TOP_K=<n>.
    enable_propagation_weight_prune: bool = True
    propagation_prune_top_k: int = 10

    # ---- Step 6.9: Sibling-precision prune (in-file arm, parallel to 6.8) ----
    # Deterministic precision step for the in-file arm, mirroring how Step 6.8
    # prunes the outward-BFS arm. Siblings are EXEMPT from 6.8 (they tie under
    # decay scoring); instead they are ranked here by their ANCHOR's rrf_score
    # and cut to the top-K. A sibling-pool bake-off found anchor-rrf
    # the best deterministic sibling ranker — 83% sibling-GT @ top-5 vs 0%
    # semantic (GT siblings are mostly unembeddable type defs), 33% PPR, 50%
    # flat-tie, 17% random; combining signals only hurt. K=10 is the recall-
    # safe point (100% sibling-GT retained, ~53% sibling noise cut), matching
    # the 6.8 philosophy: hand V7's sibling validator the full recall with
    # less noise. Detachable; sibling_prune_top_k=0 disables the cut.
    # Env-overridable (bare name, like TOP_K_RRF_POOL): SIBLING_PRUNE_TOP_K=<n>.
    enable_sibling_precision_prune: bool = True
    sibling_prune_top_k: int = 10

    # ---- Step 2: Traceability-matrix pool seeding -------------------
    # After dense_doc retrieval, query doc_code_candidates for code-nodes
    # linked to those doc-chunks above this threshold and inject them into
    # the RRF pool with a synthetic rank. Promotes the offline traceability
    # precomputation from a rerank +0.1 bonus into a pool-seeding signal.
    enable_traceability_pool_seeding: bool = True
    traceability_seed_top_k_per_doc: int = 5
    traceability_seed_min_score: float = 0.40
    traceability_seed_synthetic_rank: int = 5

    # ---- Steps 3.5 / 3.6 / 3.7: Pre-Validation Gates (FR-C4) --------
    # Score floor is a sanity-only gate (-2.0 admits all candidates above
    # the BGE-reranker-v2-m3 "irrelevant" floor). LLM #2 is the real precision gate.
    min_reranker_score_for_validation: float = -2.0
    # Density threshold: rejects candidates when a single file exceeds this
    # fraction of the total pool. Density-only; no per-file count cap.
    plausibility_gate_density_threshold: float = 0.50
    # Additive boost applied to a candidate's raw_reranker_score at the
    # score-floor admission step when its name substring-matches any of
    # cr_interp.anchor_candidates. Soft semantics — a candidate whose
    # boosted score is still below min_reranker_score_for_validation is
    # dropped. Default 0.10 sits roughly one cross-encoder logit step above
    # the floor, enough to admit borderline anchors without overwhelming
    # the validator chain. Set to 0.0 to disable the boost while keeping
    # anchor_candidates extraction.
    anchor_priming_boost: float = 0.10

    # Anchor-priming BM25 boost (Step 2 Path 4). Multiplier applied to a
    # synthetic BM25 query built from each anchor_candidate identifier
    # inside retriever.hybrid_search Path 4. Default 1.5 = anchor-matching
    # candidates get a 0.5 weight bonus in the BM25 max-over-queries step.
    # Set to 1.0 to disable.
    anchor_priming_bm25_boost: float = 1.5

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

    # ---- Step 1: Project skeleton -----------------------------------
    # Cached project-skeleton text written at index time and consumed by
    # the two-stage interpreter at run time. Missing file is treated as
    # a graceful degrade signal (single-stage interpreter only).
    project_skeleton_path: str = "./data/project_skeleton.txt"

    # ---- Code-only evaluation mode (sensitivity-analysis toggle) ----
    # When True:
    #   - cr_interp.affected_layers is coerced to ["code"] post-LLM-1, which
    #     disables doc retrieval (dense_doc / bm25_doc paths).
    #   - enable_traceability_pool_seeding is treated as False (no doc->code
    #     neighbour injection into the RRF pool).
    #   - merged_doc_contexts attachment in step_3_6_semantic_dedup is
    #     skipped, so LLM-2 receives no "Business Context" block.
    # Together these three toggles isolate "what would ImpacTracer score if
    # the SRS / SDD did not exist". Default False = production behaviour.
    code_only_mode: bool = False


# =========================================================================
# Index profiles
# =========================================================================
#
# A "profile" names an independent on-disk index (one per target repo) so
# multiple repos (e.g. citrakara, nova) can be indexed and analyzed without
# overwriting each other. Each profile's stores live under ./data/<profile>/.
#
# Profile selection precedence (highest first):
#   1. explicit `profile` argument to get_settings (the CLI --profile flag)
#   2. the IMPACTRACER_PROFILE environment variable
#   3. DEFAULT_PROFILE
#
# Profile is deliberately NOT a Settings field: a field would itself be
# env/.env-driven and muddy the precedence story. It stays a function-level
# concern, applied AFTER Settings() resolves all other sources so the profile
# is authoritative over any DB_PATH/CHROMA_PATH/etc. left in .env.

DEFAULT_PROFILE = "citrakara"
KNOWN_PROFILES = ("citrakara", "nova")  # advisory only; arbitrary names are allowed
_PROFILE_ENV_VAR = "IMPACTRACER_PROFILE"

# Settings field -> basename under ./data/<profile>/. Basenames match the
# pre-profile defaults so moving the legacy ./data files into ./data/citrakara/
# is a 1:1 migration with no rebuild. llm_audit_log_path is scoped per profile
# so NFR-03/NFR-05 token/latency stats never mix two repos' runs;
# locked_parameters_path is declared-only but included for layout consistency.
_PROFILE_PATH_FIELDS: dict[str, str] = {
    "db_path": "impactracer.db",
    "chroma_path": "chroma_store",
    "project_skeleton_path": "project_skeleton.txt",
    "locked_parameters_path": "locked_parameters.json",
    "llm_audit_log_path": "llm_audit.jsonl",
}


def resolve_profile(profile: str | None = None) -> str:
    """Resolve the active profile name.

    Precedence: explicit ``profile`` arg > ``IMPACTRACER_PROFILE`` env >
    :data:`DEFAULT_PROFILE`.
    """
    if profile:
        return profile
    return os.environ.get(_PROFILE_ENV_VAR, "").strip() or DEFAULT_PROFILE


def get_settings(profile: str | None = None) -> Settings:
    """Construct a :class:`Settings` instance scoped to an index profile.

    Builds ``Settings()`` (full .env + OS-env resolution), then rewrites the
    storage path fields to live under ``./data/<profile>/`` via
    :meth:`model_copy`. The post-construction override is authoritative over
    any ``DB_PATH``/``CHROMA_PATH``/``LLM_AUDIT_LOG_PATH``/
    ``LOCKED_PARAMETERS_PATH`` set in ``.env``; every other env-driven setting
    is left untouched.
    """
    settings = Settings()  # type: ignore[call-arg]
    data_root = Path("./data") / resolve_profile(profile)
    overrides = {
        field: (data_root / basename).as_posix()
        for field, basename in _PROFILE_PATH_FIELDS.items()
    }
    return settings.model_copy(update=overrides)
