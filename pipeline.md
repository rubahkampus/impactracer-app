# ImpacTracer v4.0 — How It Works

> A three-layer explainer for the thesis author and committee.
> Layer 1 is the 60-second version. Layer 2 is the component tour.
> Layer 3 is the end-to-end trace of one Change Request.
> All claims are anchored to real code paths so any committee question
> can be drilled down to a file and a function.

---

## Layer 1 — The 60-Second Version

### What problem does this system solve?

Given a natural-language **Change Request** (CR) — typically a one- to
three-sentence description of a feature, fix, or refactor, written in
Indonesian or English — produce an **Impact Report** that names *every
code file and every code entity* (function, class, interface, method,
type, top-level variable) that will need to change, and explains why.

The target codebase is a full-stack TypeScript application
(`citrakara`) accompanied by a Software Requirements Specification (SRS)
and Software Design Document (SDD) in Markdown.

### The three big pieces

```
                            ┌─────────────────────────────────┐
   citrakara repo  ────►    │       OFFLINE INDEXER           │   one-time
   (.ts/.tsx + docs/*.md)   │   (impactracer index <repo>)    │   per repo
                            └─────────────────────────────────┘
                                          │
                              writes      ▼
                            ┌─────────────────────────────────┐
                            │   SQLite + ChromaDB on disk     │
                            │   (graph + embeddings)          │
                            └─────────────────────────────────┘
                                          │
                                          ▼
   "Add a feature           ┌─────────────────────────────────┐   per CR
    that lets users         │       ONLINE PIPELINE           │   ~30-60 s
    cancel an order"        │   (impactracer analyze "...")   │
                            └─────────────────────────────────┘
                                          │
                              writes      ▼
                            ┌─────────────────────────────────┐
                            │   ImpactReport (JSON)           │
                            │   impacted_files                │
                            │   impacted_entities             │
                            │   executive_summary             │
                            └─────────────────────────────────┘

   Ground-Truth JSONs       ┌─────────────────────────────────┐
   (calibration + eval) ──► │       EVALUATION HARNESS        │   per dataset
                            │ (impactracer evaluate --...)    │
                            └─────────────────────────────────┘
                                          │
                                          ▼
                       summary_table.csv · statistical_tests.json ·
                       per_cr_per_variant_metrics.csv · nfr_verification.json
```

### The five LLM calls (in V7 — the full system)

1. **`interpret`** — read the CR, decide if it is actionable, extract search
   terms.
2. **`validate_sis`** — given the retrieval top-K, decide which candidates are
   *structurally* required.
3. **`validate_trace`** — given a documentation chunk and a candidate code
   resolution for it, decide if the code genuinely implements the doc.
4. **`validate_propagation`** — given a code seed and a graph neighbour reached
   through structural edges (calls, inheritance, ...), decide if the
   neighbour is *semantically* impacted.
5. **`synthesize`** — write the executive summary and one-paragraph
   justifications for each impacted *file*. **This call is never permitted to
   author entity-level claims** (see the Distributed Justification Principle in
   §3.10).

Variants V0 through V7 selectively disable steps and LLMs to isolate
contributions. V7 is the full system; V5 is the strongest no-graph baseline.
The pre-registered statistical test compares V7 to V5 on entity-level F1.

### The shape of the final report

```json
{
  "executive_summary": "<one paragraph for non-technical stakeholders>",
  "impacted_files": [
    {"file_path": "src/lib/orders/cancel.ts",
     "justification": "<one or two sentences, by LLM #5>"}, ...
  ],
  "impacted_entities": [
    {"node": "src/lib/orders/cancel.ts::cancelOrder",
     "node_type": "Function",
     "file_path": "src/lib/orders/cancel.ts",
     "severity": "Tinggi",
     "causal_chain": [],
     "justification": "<verbatim from LLM #2 / #3 / #4 / auto_exempt>",
     "justification_source": "llm2_sis",
     "traceability_backlinks": ["srs#order-lifecycle"]}, ...
  ],
  "documentation_conflicts": ["..."],
  "estimated_scope": "menengah",
  "analysis_mode": "retrieval_plus_propagation",
  "degraded_run": false
}
```

Two arrays, two granularities — files and entities — aligned with the
Ground-Truth shape so the evaluation harness can compute set-level
Precision / Recall / F1 at both granularities independently.

If you remember nothing else: **retrieval finds candidate impacts, three
fail-closed validation LLMs filter and propagate them through the code graph,
and a final aggregator writes prose around the deterministic result.**

---

## Layer 2 — The Components

There are four packages under `impactracer/`, plus a thin CLI. Each package
has a single responsibility.

### 2.1 `shared/` — contracts everyone depends on

[shared/models.py](impactracer/shared/models.py) — every Pydantic schema. Three groups:

- **LLM-output schemas** (inherit `TruncatingModel`, which silently truncates
  overlong strings so the LLM cannot crash validation):
  `CRInterpretation`, `SISValidationResult`, `TraceValidationResult`,
  `PropagationValidationResult`, `LLMSynthesisOutput`.
- **Final-report schemas**: `ImpactReport`, `ImpactedFile`, `ImpactedEntity`.
  The legacy alias `ImpactedNode` still imports cleanly via a `model_validator`
  that maps old field names.
- **Internal DTOs** (plain dataclasses): `Candidate` (passed between retrieval,
  reranking, gates, and validation), `NodeTrace` and `CISResult` (BFS output).

[shared/constants.py](impactracer/shared/constants.py) — system-wide constants. The
canonical numbers live here:

- `RRF_PATH_WEIGHTS` — per `ChangeType` (ADDITION/MODIFICATION/DELETION),
  the relative weight of the four retrieval paths in RRF fusion.
- `LAYER_COMPAT` — a `FileClassification × ChunkType` matrix used by the
  traceability matrix to weight doc↔code similarity (e.g. an `API_ROUTE` file
  weighted highly against an `FR` doc chunk, less against a `Design` chunk).
- `EDGE_CONFIG` — **the 14 edge types** with their BFS direction and depth:
  `CALLS` (reverse, depth 2), `INHERITS`/`IMPLEMENTS`/`TYPED_BY` (reverse,
  depth 3), `FIELDS_ACCESSED` (reverse, depth 2),
  `DEFINES_METHOD`/`PASSES_CALLBACK` (forward, depth 1),
  `HOOK_DEPENDS_ON`/`IMPORTS`/`RENDERS`/`DEPENDS_ON_EXTERNAL`/`CLIENT_API_CALLS`/`DYNAMIC_IMPORT`/`CONTAINS`
  (reverse, depth 1).
- `NODE_TYPE_MAX_FAN_IN` — per node-type in-degree caps used to drop hub
  nodes from BFS (e.g. a `Function` with > 50 callers is excluded as noise).
- `SEVERITY_BY_EDGE_CHAIN_TYPE` — the rules that map a causal chain to one of
  three Indonesian severity labels (`Tinggi`, `Menengah`, `Rendah`).
- `BUILTIN_PATTERNS`, `PRIMITIVE_TYPES`, `HOOK_NAMES` — blacklists / whitelists
  used by the AST passes.

[shared/config.py](impactracer/shared/config.py) — the `Settings` class
(pydantic-settings, reads `.env`). The most thesis-relevant fields:

- `llm_model="google/gemini-2.5-flash"`, `llm_temperature=0.0`, `llm_seed=42` —
  determinism mandate.
- `top_k_per_query=30`, `top_k_rrf_pool=200`, `max_admitted_seeds=15` — the
  retrieval funnel widths. The wide RRF pool is intentional: a narrow pool
  starved the cross-encoder of GT entities.
- `enable_raw_cr_dense_pass=True`, `raw_cr_dense_top_k=60` — the multilingual
  bridge that lets an Indonesian CR retrieve English-identifier code without
  going through LLM #1's English search queries (NFR-04).
- `enable_traceability_pool_seeding=True` — promotes the offline doc↔code
  similarity matrix from a +0.1 rerank bonus into an actual pool-seeding signal.
- `enable_sibling_promotion=True`, `sibling_admit_max_per_file=4` — recovers
  GT entities that share a file with a confirmed seed but were not retrieved
  themselves (Step 7.5).
- `enable_graph_rerank=False` — ships off by default because it cannot win
  both file F1 and entity F1 on the target codebase's calibration set.
- `bfs_global_max_depth=3`, `bfs_high_conf_top_n=5` — BFS knobs.
- `alpha=0.05` — the single pre-registered Wilcoxon threshold.

### 2.2 `persistence/` — SQLite + ChromaDB

[persistence/sqlite_client.py](impactracer/persistence/sqlite_client.py) — opens the
SQLite connection (WAL mode, foreign-keys on) and creates six tables:

- `code_nodes` — every code entity. Key columns: `node_id` (PK),
  `node_type` (the 10-value CHECK constraint), `file_path`, `name`,
  `line_start/end`, `signature`, `docstring`, `source_code`,
  `internal_logic_abstraction` (the skeletonized body), `file_classification`,
  `embed_text`.
- `structural_edges` — `(source_id, target_id, edge_type)` triple as primary
  key, with the 14-value CHECK constraint on `edge_type`.
- `doc_code_candidates` — the precomputed doc↔code similarity table:
  `(code_id, doc_id, weighted_similarity_score)`.
- `file_hashes`, `file_dependencies`, `index_metadata` — bookkeeping for
  incremental indexing and the regime audit trail.

[persistence/chroma_client.py](impactracer/persistence/chroma_client.py) — opens the
ChromaDB persistent client and asserts cosine distance on both collections:

- `code_units` — 1024-dim BGE-M3 embeddings of every non-degenerate code node.
- `doc_chunks` — embeddings of every Markdown H2/H3 chunk.

Degenerate nodes (`len(embed_text) < 50`) live only in SQLite — they are still
BFS-reachable but cannot be retrieved by vector search.

### 2.3 `indexer/` — offline knowledge construction

Entrypoint: [indexer/runner.py](impactracer/indexer/runner.py)::`run_indexing()`.
Eight steps end-to-end:

1. Walk the repo, hash every `.ts/.tsx/.md` file, compare to `file_hashes` to
   determine the work set (changed + deleted + new). `--force` invalidates
   all hashes.
2. Purge SQLite rows for deleted files; recompute reverse-dependencies.
3. **Pass 1** ([indexer/code_indexer.py](impactracer/indexer/code_indexer.py)::`extract_nodes()`):
   Tree-Sitter AST walk emits the 10 node types
   (File, Class, Function, Method, Interface, TypeAlias, Enum, InterfaceField,
   Variable, ExternalPackage). Function/Method bodies are passed through
   [indexer/skeletonizer.py](impactracer/indexer/skeletonizer.py)::`skeletonize_node()`
   to produce a token-budgeted abstraction (preserves calls/returns/throws,
   folds JSX/arrays/long strings).
4. Document indexing ([indexer/doc_indexer.py](impactracer/indexer/doc_indexer.py)::`chunk_markdown()`):
   chunk SRS / SDD Markdown at H2/H3 boundaries using `mistune`; classify each
   chunk as FR / NFR / Design / General. **The classifier checks NFR before
   FR** because the substring "functional requirement" is contained inside
   "non-functional requirement".
5. **Pass 2** ([indexer/code_indexer.py](impactracer/indexer/code_indexer.py)::`extract_edges()`):
   second AST walk, this time with the global `known_node_ids` set in hand,
   emits the 14 edge types (CALLS, INHERITS, IMPLEMENTS, TYPED_BY,
   FIELDS_ACCESSED, DEFINES_METHOD, HOOK_DEPENDS_ON, PASSES_CALLBACK, IMPORTS,
   RENDERS, DEPENDS_ON_EXTERNAL, CLIENT_API_CALLS, DYNAMIC_IMPORT, CONTAINS).
   Special handling for Mongoose `model<T>(...)`, React hooks, JSX callback
   handlers, and `middleware.ts` config.matcher.
6. Embedding ([indexer/embedder.py](impactracer/indexer/embedder.py)::`Embedder.embed_batch()`):
   BGE-M3 in FP16 on CUDA when available; emits 1024-dim float32 vectors.
7. Traceability ([indexer/traceability.py](impactracer/indexer/traceability.py)::`compute_and_store()`):
   L2-normalize, cosine matrix multiply, weight by `LAYER_COMPAT`, take dual
   top-K (per-doc and per-code) to prevent low-compat chunk types being
   squeezed out, write `doc_code_candidates`.
8. Write `index_metadata`: edge_schema_version, embedding_model_name,
   traceability_k_parameter, total counts, timestamp.

A separate [indexer/auditor.py](impactracer/indexer/auditor.py) produces a
human-readable quality report: orphan analysis, degree
distribution, traceability score distribution, semantic benchmark sanity
checks, FK integrity checks, ChromaDB↔SQLite alignment.

On the current `citrakara` index: **333 files, 3,150 nodes, 8,179 edges.**

### 2.4 `pipeline/` — online analysis

Entrypoint: [pipeline/runner.py](impactracer/pipeline/runner.py)::`run_analysis()`.
Steps 0–9 (5 LLM calls in V7). Layer 3 walks each step in detail; this
section just inventories the modules.

- [pipeline/llm_client.py](impactracer/pipeline/llm_client.py) — **the only file that
  imports `httpx` or `openai`**. All five LLM calls go through
  `LLMClient.call(system, user, response_schema, call_name)`. Temperature 0,
  seed 42, JSON mode, Pydantic schema validation. Retries 429 / 5xx /
  connection errors with exponential backoff up to 10 attempts. Every call
  appended to `data/llm_audit.jsonl` (the NFR-05 source of truth) with
  `config_hash` = SHA-256 of (model + temperature + seed + provider). The
  audit log is the artefact the thesis cites for reproducibility.
- [pipeline/interpreter.py](impactracer/pipeline/interpreter.py) — LLM #1
  (Step 1). Returns `CRInterpretation`.
- [pipeline/retriever.py](impactracer/pipeline/retriever.py) — Step 2 hybrid
  search. Four RRF paths × N queries, plus the raw-CR multilingual pass and
  traceability-matrix pool seeding. Also hosts
  `rerank_multi_query` which is invoked at Step 3 with `top_k=len(candidates)`
  — the cross-encoder scores the entire 200-candidate pool, not a
  15-pre-truncated subset.
- [pipeline/prevalidation_filter.py](impactracer/pipeline/prevalidation_filter.py) —
  the three deterministic gates (3.5 score floor, 3.6 semantic dedup,
  3.7 plausibility + affinity). Each is independently toggleable via
  `VariantFlags`.
- [pipeline/validator.py](impactracer/pipeline/validator.py) — LLM #2
  (Step 4). Per-batch fail-closed: if a batch's structured-output validation
  fails after retries, the entire batch is dropped and `degraded_run` flips
  to `True`.
- [pipeline/seed_resolver.py](impactracer/pipeline/seed_resolver.py) — Step 5.
  Deterministic SQLite lookup: each doc-chunk SIS entry is resolved to its
  top-K code candidates via `doc_code_candidates`.
- [pipeline/traceability_validator.py](impactracer/pipeline/traceability_validator.py) —
  LLM #3 (Step 5b). Same fail-closed contract as LLM #2; rejected pairs are
  marked low-confidence (which then caps their BFS reverse-CALLS depth to 1).
- [pipeline/graph_bfs.py](impactracer/pipeline/graph_bfs.py) — Step 6 BFS over
  the `networkx.MultiDiGraph` built from `structural_edges`. Honours
  `EDGE_CONFIG` direction/depth, applies `NODE_TYPE_MAX_FAN_IN` caps, hub-node
  depth-1 cap (degree > 20), `UTILITY_FILE_CALLS_DEPTH_CAP=1`, and excludes
  `ExternalPackage` from the propagated set. Step 6.5 collapses CONTAINS-only
  subtrees into their parents to keep the LLM #4 prompt token-budgeted.
- [pipeline/traversal_validator.py](impactracer/pipeline/traversal_validator.py) —
  LLM #4 (Step 7). De-blinded: the prompt shows the causal chain as
  "factual context only" and explicitly tells the LLM that edge presence is
  not impact. Also hosts the Step 7.5 sibling-promotion batch mode.
- [pipeline/context_builder.py](impactracer/pipeline/context_builder.py) — Step 8.
  Fetches backlinks (doc chunks that specify each code node, via
  `doc_code_candidates` in reverse), source snippets (preferring
  `internal_logic_abstraction` for Function/Method), and assembles a
  token-budgeted Markdown prompt within `llm_max_context_tokens` (default
  100k).
- [pipeline/synthesizer.py](impactracer/pipeline/synthesizer.py) — LLM #5
  (Step 9) **as an aggregator only**. Produces `executive_summary`,
  `documentation_conflicts`, and per-file `file_justifications`. The entity
  array is built deterministically *outside* the LLM call by
  `build_deterministic_impacted_entities()` from the CIS and the validator
  justifications. **LLM #5 never authors entity-level prose** — this is the
  Distributed Justification Principle, see §3.10.

### 2.5 `evaluation/` — the ablation harness

- [evaluation/variant_flags.py](impactracer/evaluation/variant_flags.py) — the
  canonical 8 variants V0..V7 as `@dataclass(frozen=True)` factories. See the
  table in §3.11.
- [evaluation/metrics.py](impactracer/evaluation/metrics.py) — `compute_set_metrics`
  (set-level Precision / Recall / F1 against the full unpruned CIS, no top-K
  truncation), `compute_r_precision` (descriptive only), and
  `compute_dual_granularity_metrics` (entity- and file-level in one call).
  **There is no F1@K.** Bounded F1@K cannot distinguish a 50-node focused
  result from a 372-node graph flood; set-F1 is the only honest metric for a
  variable-output system.
- [evaluation/statistical.py](impactracer/evaluation/statistical.py) — `ALPHA=0.05`,
  `PRIMARY_COMPARISON=("V7","V5")`, `PRIMARY_METRIC="f1_set"`,
  `MIN_PAIRED_N=15`. One-sided paired Wilcoxon on entity-level f1_set. If
  fewer than 15 paired CRs are available, the harness emits
  `status="insufficient_pairs"` instead of an unreliable p-value.
- [evaluation/ablation.py](impactracer/evaluation/ablation.py) — `run_full_evaluation()`
  loops every CR in the dataset × all 8 variants, shares one heavy
  Embedder/Reranker/LLMClient across runs, and writes per-cell
  `impact_report.json` + `impact_report_full.json` (the latter contains the
  step-by-step trace_sink for academic auditability).
- [evaluation/nfr_verify.py](impactracer/evaluation/nfr_verify.py) — NFR-01
  (determinism: re-run V7 and compare the validated SIS code-seed sets), NFR-02
  (local-execution audit), NFR-03 (latency p50/p95 over the eval CSV), NFR-04
  (cross-lingual: an Indonesian CR retrieves English-identifier code), NFR-05
  (audit log invariants: all calls in a run share `config_hash`).
- [evaluation/report_builder.py](impactracer/evaluation/report_builder.py) — writes
  `summary_table.csv`, `per_cr_per_variant_metrics.csv`,
  `statistical_tests.json`, `nfr_verification.json`,
  `calibration_analysis.md`.
- [evaluation/schemas.py](impactracer/evaluation/schemas.py) — the GT side:
  `GTEntry`, `ImpactedFile`, `ImpactedEntity`. Helper methods
  `file_paths() -> set[str]` and `entity_node_ids() -> set[str]` are what the
  metric functions consume directly.

### 2.6 `cli.py` — the three commands

[cli.py](impactracer/cli.py), using Typer:

- `impactracer index <repo>` — offline indexing (calls indexer/runner).
- `impactracer analyze "<cr text>" [--variant V0..V7] [--output PATH]` — one
  CR, one variant, one report.
- `impactracer evaluate --dataset DIR --output DIR [--run-full-ablation] [--verify-nfr]` —
  the harness. Iterates every `<dataset>/cr*.json` × every variant in
  `ALL_VARIANTS`, produces the artefacts listed above.

---

## Layer 3 — One Change Request, End to End

Setting the stage: the offline index already exists. The user types:

```
impactracer analyze "Tambahkan fitur agar pengguna bisa membatalkan order
yang sudah dibayar tetapi belum dikirim." --variant V7
```

(Indonesian: "Add a feature that lets users cancel orders that have been paid
but not yet shipped.")

### 3.0 Offline prerequisites — what the index looks like

When `impactracer index <repo>` finished, the following exist on disk:

- **SQLite**: 3,150 rows in `code_nodes` (one per File / Function / Class /
  ...), 8,179 rows in `structural_edges`, ~10,000 rows in
  `doc_code_candidates` (precomputed doc↔code similarity pairs), plus
  metadata tables.
- **ChromaDB**: `code_units` has BGE-M3 embeddings for every non-degenerate
  code node (1024-dim, cosine metric). `doc_chunks` has embeddings for every
  H2/H3 Markdown chunk in `citrakara/docs/`.

`run_analysis()` ([pipeline/runner.py:228](impactracer/pipeline/runner.py)) is the
orchestrator. **Step 0** (`load_pipeline_context`) opens both stores,
constructs the NetworkX `MultiDiGraph` from `structural_edges`, builds the BM25
indices from the ChromaDB documents, and instantiates the Embedder, Reranker,
and LLMClient. All subsequent steps reuse these.

A `trace_sink: dict` is passed through so each step writes its intermediate
result for `impact_report_full.json` — every claim in the final report can be
traced back to its origin step.

### 3.1 Step 1 — Interpret CR (LLM #1)

[pipeline/interpreter.py](impactracer/pipeline/interpreter.py)::`interpret_cr(cr_text, llm_client)`
calls the LLM with the CR text and gets back a `CRInterpretation`:

```python
CRInterpretation(
    is_actionable=True,
    primary_intent="Add user-initiated cancellation for paid, unshipped orders",
    change_type="ADDITION",
    affected_layers=["requirement", "design", "code"],
    domain_concepts=["order", "cancellation", "payment", "shipment"],
    search_queries=["cancel order", "order cancellation flow", "refund paid order"],
    layered_search_queries={
        "api_route": ["POST /api/orders/.../cancel"],
        "page_component": ["order detail cancel button"],
        "ui_component": ["confirm cancellation modal"],
        "utility": ["refund handler"],
        "type_definition": ["OrderStatus enum"],
    },
    named_entry_points=["cancelOrder"],
    out_of_scope_operations=["delete order", "shipping update"],
    is_nfr=False,
)
```

**Fail-closed**: if `is_actionable=False`, the runner returns a minimal
rejection `ImpactReport` immediately (no retrieval, no LLM #2-5).

**Why this step matters for retrieval**: the `search_queries` are
**always English**, even for Indonesian CRs, because the cross-encoder is
strongest on English identifiers. The `out_of_scope_operations` will be
applied as a negative filter (−1.0 penalty) at step 2.

### 3.2 Step 2 — Adaptive RRF Hybrid Search

[pipeline/retriever.py](impactracer/pipeline/retriever.py)::`hybrid_search(cr_interp, ctx, settings, cr_text)`.

For each `search_query`, four ranked lists are produced:

1. **bm25_doc** — BM25 over doc-chunk text.
2. **dense_doc** — cosine similarity over `doc_chunks` embeddings.
3. **bm25_code** — BM25 over code-node `embed_text`.
4. **dense_code** — cosine similarity over `code_units` embeddings.

Each list has `top_k_per_query = 30` entries. Plus:

- **Raw-CR multilingual pass**: one extra `dense_code` query using the
  **original** Indonesian CR text against `code_units` (BGE-M3 is
  multilingual, so the Indonesian noun "pembatalan" can directly reach
  `cancelOrder()` without depending on LLM #1).
- **Layered queries**: when `layered_search_queries` is populated, an extra
  retrieval pass per architectural layer, scoped to that layer's
  `file_classification`, contributes to a new RRF path called
  `layered_code`. Prevents one architectural plane from monopolising the pool.
- **Traceability-matrix pool seeding**: for every doc chunk that survived
  `dense_doc`, look up the top-K linked code nodes in
  `doc_code_candidates` (above `traceability_seed_min_score=0.40`) and inject
  them into the RRF pool with a synthetic rank of 5. Promotes the offline
  similarity precomputation from a rerank +0.1 bonus to a true seeding signal.

All ranked lists fuse via **Reciprocal Rank Fusion** with weights from
`RRF_PATH_WEIGHTS[change_type]`. RRF score per candidate per path:
`1 / (rrf_k + rank)` (default `rrf_k=60`), then summed across paths and
weighted by path.

Post-fusion bonuses / penalties:

- **+0.1** for any code candidate linked via `doc_code_candidates` to a doc
  chunk that survived `dense_doc` (traceability bonus).
- **−1.0** ("hard demotion") for any candidate whose `name` or `text_snippet`
  contains a phrase from `out_of_scope_operations` (≥ 6 chars).

Output: `list[Candidate]` with up to **200** entries (`top_k_rrf_pool=200`).
Each `Candidate` carries `rrf_score`, `collection`, `node_type`, `file_path`,
`file_classification`, `text_snippet`, and `merged_doc_ids`/`merged_doc_contexts`
populated later by the dedup gate.

### 3.3 Step 3 — Cross-Encoder Rerank (Full-Pool)

[pipeline/retriever.py](impactracer/pipeline/retriever.py)::`rerank_multi_query()` is
invoked with `top_k=len(candidates)` — **every one of the 200 candidates is
scored by the BGE cross-encoder against every search query**, max-pooled per
candidate. Cost: ~1 s extra wall-time. Benefit: V4/V5/V6 entity F1 lifts on
the calibration set.

Each candidate gets:

- `raw_reranker_score` — absolute logit, preserved for the score-floor gate.
- `reranker_score` — min-max normalised to [0, 1], used for relative sorting.

Two diagnostic traces are emitted:

- `step_3_reranked_full` — the entire 200-candidate ranked list.
- `step_3_reranked` — the top-15 after `max_admitted_seeds` truncation.

Optional **graph-aware label-propagation rerank** (default off,
`enable_graph_rerank=False`) blends a structural-graph-walk score into
the rerank. Shipped off because the calibration trade-off doesn't
generalise on the target codebase.

### 3.4 Steps 3.5 / 3.6 / 3.7 — The Three Deterministic Gates

[pipeline/prevalidation_filter.py](impactracer/pipeline/prevalidation_filter.py).
All three gates run only when `VariantFlags` enables them.

**Gate 3.5 — Score floor** (`enable_score_floor`): drop any candidate with
`raw_reranker_score < min_reranker_score_for_validation` (−2.0 by default; a
"sanity-only" floor that removes only the BGE-reranker-v2-m3 "irrelevant"
class). LLM #2 is the real precision gate; this just stops garbage from
wasting a structured-output call.

**Gate 3.6 — Semantic dedup** (`enable_dedup_gate`): for every doc-chunk
candidate, look up its top-1 code resolution in `doc_code_candidates`. If that
code id is already in the pool, **merge** the doc into the code candidate
(append to `merged_doc_ids` and `merged_doc_contexts`) and **drop** the doc
candidate. Result: the LLM #2 validator will see the code node + the
business-context paragraphs that justify it, in one prompt.

**Gate 3.7 — Plausibility + Affinity** (`enable_plausibility_gate`): two
sub-gates.

- **Affinity rescoring**: layer-compatibility bonus per `(change_type,
  file_classification)` — e.g. ADDITION + UTILITY gets a small boost.
- **File-density plausibility**: if a single file contributes more than
  `plausibility_gate_density_threshold * pool_size` candidates AND none of
  them appear in `named_entry_points`, the file's contribution is rejected.
  Prevents one generic utility file from monopolising the SIS.

### 3.5 Step 4 — Validate SIS (LLM #2, fail-closed)

[pipeline/validator.py](impactracer/pipeline/validator.py)::`validate_sis_candidates_batched(cr_interp, candidates, llm_client)`.

The top ~15 surviving candidates are batched (max 5 per batch) and passed to
LLM #2. The prompt deliberately **does not show retrieval scores** — LLM #2
must judge each candidate on the code snippet alone, anti-circular.

Output schema: `SISValidationResult.verdicts: list[CandidateVerdict]`.
Each verdict:

```python
CandidateVerdict(
    node_id="src/lib/orders/cancel.ts::cancelOrder",
    function_purpose="Marks a paid order as cancelled and triggers refund.",
    mechanism_of_impact="Add a status-precondition check for 'shipped' and a "
                       "new branch that issues a partial refund.",
    justification="Direct named entry point matching CR intent.",
    confirmed=True,  # True iff mechanism_of_impact is non-empty
)
```

**Fail-closed contract**:

- A missing verdict for a candidate → that candidate is DROPPED (not silently
  admitted).
- A batch that fails structured-output validation after retries → the entire
  batch is DROPPED and `degraded_run = True`.

The `mechanism_of_impact` and `justification` strings are saved in the
runner's `justifications` map, keyed by `node_id`. **These strings become the
entity-level justification in the final report, verbatim.**

### 3.6 Step 5 + 5b — Resolve and Validate Traces

[pipeline/seed_resolver.py](impactracer/pipeline/seed_resolver.py)::`resolve_doc_to_code()`
walks the confirmed SIS:

- Code-node SIS entries → already code, passed through as `direct_code_seeds`.
- Doc-chunk SIS entries → look up top-K code resolutions in
  `doc_code_candidates`, return as `[{"doc_id": d, "code_ids": [c1, c2, ...]}]`.

[pipeline/traceability_validator.py](impactracer/pipeline/traceability_validator.py)::`validate_trace_resolutions()` (LLM #3)
batches the `(doc_id, code_id)` pairs (max 5 per batch). Each pair gets a
verdict: `CONFIRMED`, `PARTIAL`, or `REJECTED`. PARTIAL pairs are admitted
but **flagged low-confidence** — the BFS step caps their reverse-CALLS depth
to 1 to prevent low-confidence seeds spawning large neighbour clouds.

Same fail-closed contract: missing verdicts → REJECTED; batch errors → drop
batch and set `degraded_run = True`.

The validated code seeds are merged with `direct_code_seeds` to form the
final code-seed set that enters BFS.

### 3.7 Step 6 — BFS Propagation

[pipeline/graph_bfs.py](impactracer/pipeline/graph_bfs.py)::`bfs_propagate()`.

Starting from each code seed (depth 0), traverse the NetworkX MultiDiGraph
edge by edge, with five separate gating rules:

1. **Per-edge-type direction and depth** from `EDGE_CONFIG`. Example:
   `CALLS` is reverse, max-depth 2 — given seed `cancelOrder`, walk callers
   for up to two hops.
2. **Hub-node cap**: any node with total degree > 20 has all outgoing
   traversal capped to depth 1.
3. **Low-confidence-seed cap**: if the seed was a PARTIAL trace verdict from
   LLM #3, all `LOW_CONF_CAPPED_EDGES` (= `{CALLS}`) are capped to depth 1.
4. **Utility-file cap**: if the seed lives in a `UTILITY` file
   (`lib/format-date.ts`, etc.), reverse-CALLS is capped to `UTILITY_FILE_CALLS_DEPTH_CAP=1`.
   Utility functions are called from everywhere; deeper reverse-CALLS produces
   a guaranteed flood across unrelated features.
5. **Per-node-type fan-in cap**: any node whose in-degree exceeds
   `NODE_TYPE_MAX_FAN_IN[node_type]` is dropped from propagation (not from
   seeds). Default caps: 50 for Function/Method/Class; 100 for type
   definitions; 200 for files; **0 for `ExternalPackage`** (never propagate
   *into* third-party packages).

Output: a `CISResult` (Change Impact Set) with two dicts:

- `sis_nodes` — the depth-0 seeds, with `NodeTrace(depth=0, causal_chain=[])`.
- `propagated_nodes` — every BFS-discovered node, with
  `NodeTrace(depth, causal_chain, path, source_seed)`.

**Step 6.5 — Graph collapse**: `collapse_contains_subtrees()` walks
`propagated_nodes` and merges any node whose only inbound edge is `CONTAINS`
into its parent's `collapsed_children` list. Reduces the prompt token count
for LLM #4 without losing information.

### 3.8 Step 7 — Validate Propagation (LLM #4)

[pipeline/traversal_validator.py](impactracer/pipeline/traversal_validator.py)::`validate_propagation()`.

For each propagated node (in batches of 5), the prompt shows:

- The CR intent and `change_type`.
- The propagated node's source code or skeletonized abstraction.
- The **causal chain** — the ordered list of edge types from seed to this
  node — **explicitly framed as "factual context only"**. The prompt states
  that edge presence is not impact; the LLM must justify the node on contract
  breakage, behavioural anomaly, or downstream type-mismatch.

Verdict: `semantically_impacted: bool` + `justification: str (≤400 chars)`.

Same fail-closed contract: missing verdict → DROP, batch error → DROP batch
+ set `degraded_run = True`.

**Step 7.5 — Sibling promotion**: a separate LLM #4 invocation specifically
targets *missed* GT entities that live in the same file as a confirmed seed
but were not retrieved. The runner enumerates every CONTAINS-sibling of each
LLM-#2-confirmed seed; LLM #4 admits/rejects each sibling using the anchor's
`mechanism_of_impact` as context. Capped at `sibling_admit_max_per_file=4`.
Admitted siblings enter the CIS at depth 1 with `causal_chain=["CONTAINS"]`.

**Step 7.5 Anchor Gate**: only seeds where LLM #2 produced a non-empty
`mechanism_of_impact` qualify as sibling-promotion anchors. Prevents
over-confident propagation from weakly-justified seeds.

### 3.9 Step 8 + 9 — Synthesize (LLM #5, aggregator only)

**Step 8** ([pipeline/context_builder.py](impactracer/pipeline/context_builder.py)::`build_context()`):

- Fetch **backlinks** — for each CIS code node, the doc chunks that specify it
  (top-K from `doc_code_candidates` in reverse).
- Fetch **snippets** — for Function/Method, use `internal_logic_abstraction`
  (already skeletonized); for everything else, use `source_code`.
- Assemble a Markdown prompt: CR text + interpretation summary + full CIS
  with causal chains and validator justifications + backlinks + snippets,
  truncated to fit `llm_max_context_tokens=100000`.

**Step 9** ([pipeline/synthesizer.py](impactracer/pipeline/synthesizer.py)::`synthesize_summary()`):
LLM #5 returns an `LLMSynthesisOutput`:

```python
LLMSynthesisOutput(
    executive_summary="<one paragraph for non-technical stakeholders>",
    documentation_conflicts=["sdd#order-state-machine"],
    file_justifications=[
        FileJustificationItem(
            file_path="src/lib/orders/cancel.ts",
            justification="Holds the cancellation primitive; both the new "
                          "status precondition and the refund branch live here."),
        ...
    ],
)
```

**LLM #5 produces three things only**: `executive_summary`,
`documentation_conflicts`, and per-file `file_justifications`. It never sees
an entity-level slot.

The `impacted_entities` array is built deterministically by
`synthesizer.build_deterministic_impacted_entities(cis, node_types,
node_file_paths, justifications_extra, backlinks)`. For each node in the CIS,
this function pulls the justification from the right source:

- **SIS seeds (LLM #2)** → `mechanism_of_impact` if non-empty else
  `justification`; `justification_source="llm2_sis"`.
- **Trace-validated seeds (LLM #3)** → LLM #3 verdict justification;
  `justification_source="llm3_trace"`.
- **BFS-propagated nodes (LLM #4)** → LLM #4 verdict justification;
  `justification_source="llm4_propagation"`.
- **Auto-exempt edges (IMPLEMENTS, DEFINES_METHOD)** → synthetic short string;
  `justification_source="auto_exempt"`.
- **Fallback** when no validator wrote a justification (e.g. V6, where LLM #4
  is disabled) → synthetic chain string;
  `justification_source="bfs_only"` or `"retrieval_only"`.

`severity` is computed by `severity_for_chain(causal_chain)`:

- Contract chains (IMPLEMENTS, TYPED_BY, FIELDS_ACCESSED) → **Tinggi**.
- Behavioural chains (CALLS, INHERITS, DEFINES_METHOD, HOOK_DEPENDS_ON,
  PASSES_CALLBACK) → **Menengah**.
- Module-composition chains (IMPORTS, RENDERS, DEPENDS_ON_EXTERNAL,
  CLIENT_API_CALLS, DYNAMIC_IMPORT, CONTAINS) → **Rendah**.
- Empty chain (SIS seed at depth 0) → **Tinggi** (the change site itself).

The `impacted_files` array is built deterministically from the distinct
`file_path` values in `impacted_entities`. Its `justification` field is the
matching `FileJustificationItem` from LLM #5, or a fallback string if LLM #5
failed.

Finally:

- `estimated_scope` is set by `_compute_scope(cis, settings)`: `≤10` nodes →
  `terlokalisasi`, `11–30` → `menengah`, `>30` → `ekstensif`.
- `analysis_mode` is `"retrieval_only"` if no BFS ran or `propagated_nodes`
  is empty, else `"retrieval_plus_propagation"`.
- `degraded_run` aggregates the boolean flags from every LLM call.

The CLI writes the `ImpactReport` to `impact_report.json` and the
step-by-step trace to `impact_report_full.json`.

### 3.10 The Distributed Justification Principle

This is a load-bearing architectural invariant. The principle is:

> **Every entity in the final `impacted_entities` list carries a justification
> produced verbatim by the validator that admitted it.** No justification is
> generated retrospectively by LLM #5.

The reason: a single late-stage LLM call seeing 50+ entities cannot produce
calibrated, per-entity reasoning. It produces glossy but generic paragraphs,
which is fatal for a precision-critical artifact.

Each validator is *the* authority for its admissions:

| Validator | Admits | Writes |
| --- | --- | --- |
| LLM #2 | SIS seeds | `mechanism_of_impact` + `justification` (richest) |
| LLM #3 | Doc→code resolutions | trace verdict justification |
| LLM #4 | BFS-propagated nodes | contract-breakage justification |
| LLM #4 (sibling-batch) | File-local siblings of confirmed seeds | sibling-batch justification |
| AUTO | Single-hop IMPLEMENTS/DEFINES_METHOD edges | synthetic short string |

LLM #5's only entity-adjacent freedom is the *file-level* summary, and even
that is bounded to the deterministic file set computed from the CIS — the
runner enforces a 1:1 mapping between
`LLMSynthesisOutput.file_justifications[*].file_path` and the distinct files
in `impacted_entities`.

### 3.11 The 8 Variants — What Each Turns Off

| Variant | Retrieval | Cross-Enc | Gates | LLM #2 SIS | LLM #3 Trace | BFS | LLM #4 Prop | LLM Calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | BM25 only | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 1 (interpret + synth, but synth on raw retrieval) |
| V1 | Dense only | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 1 |
| V2 | RRF hybrid | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 1 |
| V3 | RRF hybrid | ✓ | ✓ (all 3) | ✗ | ✗ | ✗ | ✗ | 1 |
| V4 | RRF hybrid | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | 2 |
| V5 | RRF hybrid | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | 3 |
| V6 | RRF hybrid | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 3 (blind propagation) |
| **V7** | RRF hybrid | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **5** (full system) |

Notes:

- LLM #1 (`interpret`) and LLM #5 (`synthesize`) run in **every** variant.
- V0–V3 produce SIS = the post-gate retrieval top-K (no LLM gating).
- V6 ("blind propagation") is the diagnostic that isolates LLM #4's
  contribution. Comparing V6 to V7 tells you whether the propagation
  validator is a precision gain or noise.
- **V3.5 (gates without LLM #2) was folded into V3**; **V6.5 (BFS without
  LLM #5 synthesis) was deleted** when LLM #5 was demoted to aggregator
  (V7 with aggregator-only LLM #5 absorbed the old V6.5 contract).
- The canonical 8 are exercised by every `impactracer evaluate
  --run-full-ablation` invocation. 5 CRs × 8 variants × 5 LLM calls = 200
  Gemini calls per calibration sweep.

### 3.12 Metrics — Set-Level Dual-Granularity F1

[evaluation/metrics.py](impactracer/evaluation/metrics.py).

For each `(CR, variant)` pair, compute four numbers from
`compute_dual_granularity_metrics`:

- `entity_precision_set = |predicted_entities ∩ gt_entities| / |predicted_entities|`
- `entity_recall_set    = |predicted_entities ∩ gt_entities| / |gt_entities|`
- `entity_f1_set        = 2·P·R / (P+R)`
- ... same three at file granularity (`file_*`).

`predicted_entities = cis.all_node_ids()` — the **full unpruned validated
CIS**, no top-K truncation. Bounded F1@K (which the original blueprint
specified) is banned: it cannot distinguish a focused 50-node result from
a 372-node graph flood because both get truncated to the same top-K.

The pre-registered Wilcoxon test ([statistical.py](impactracer/evaluation/statistical.py))
operates exclusively on **entity-level f1_set, V7 vs V5**, one-sided paired,
`alpha=0.05`, no Bonferroni (single test). If fewer than 15 paired CRs exist,
the artefact emits `status="insufficient_pairs"` instead of an unreliable
p-value — that is why the test is "deferred" on the n=5 calibration set and
will run on the 20-CR evaluation set once GT is finalised.

### 3.13 The Five NFRs

[evaluation/nfr_verify.py](impactracer/evaluation/nfr_verify.py).

| NFR | Plain English | Verification |
| --- | --- | --- |
| NFR-01 | Structural components are deterministic | Run V7 twice on the same CR; compare the validated SIS code-seed set after step 5b. Sets must be identical. **Known limitation**: fails because of Gemini Flash Lite structured-output stochasticity at `temperature=0` — every non-LLM stage is bit-identical. Documented as a model-side limit. |
| NFR-02 | Indexing + all non-LLM stages run offline | Pull the network plug. Indexer + BFS + gates must complete. Only the 5 LLM calls require connectivity. |
| NFR-03 | End-to-end latency reasonable | p50 / p95 over the evaluation CSV. |
| NFR-04 | Cross-lingual retrieval works | Indonesian CR retrieves English-identifier code. The raw-CR multilingual pass (`enable_raw_cr_dense_pass=True`) is what makes this pass. |
| NFR-05 | All LLM calls in a run share `config_hash` | Every row in `data/llm_audit.jsonl` from a single run has the same `config_hash` (= SHA-256 of model + temperature + seed + provider). Audit log is the artefact the thesis cites. |

The current canonical sweep **passes NFR-02 / -03 / -04 / -05**; NFR-01
fails for the documented LLM-stochasticity reason.

---

## Appendix A — Defence Q&A (Likely Committee Questions)

**Q: Why is the headline F1 only ~0.25 — that's pretty low.**
A: Two reasons. (1) The metric is **unbounded set-level F1** computed against
the *full unpruned CIS*. There is no top-K truncation that would artificially
boost precision. F1@10 would report higher numbers but would not be
defensible — a 372-node graph flood and a focused 50-node result both
truncate to the same top-10. (2) Pool-attrition diagnostics showed that
**55.9% of GT entities are not in the RRF pool at any rank** on the
calibration set;
this is a retrieval-recall ceiling, not a validator failure. The validators
have correctly identified what's there to identify. The thesis claims an
architecture for CIA over full-stack TypeScript codebases; the *delta*
between variants is the thesis contribution, not an absolute F1 number.

**Q: Why does V5 beat V7 on entity F1 in the calibration draw?**
A: n=5 is below the pre-registered `MIN_PAIRED_N=15` for the Wilcoxon test,
so any V5/V7 ordering is descriptive only. The observed
δ ≈ 0.04 is within the documented LLM-stochasticity band: two runs of the
same V7 cell can differ by similar magnitudes because Gemini Flash Lite at
temperature 0 is not bit-deterministic on structured outputs. The
pre-registered hypothesis test runs on the 20-CR evaluation set and reports
its own decision.

**Q: Why not use F1@K, MAP, MRR — standard IR metrics?**
A: This is not an IR problem; it is a *set-prediction* problem. The output is
an unordered set of impacted entities; there is no meaningful rank for the
end user. Rank-aware metrics would reward retrieval ordering decisions that
the user never sees. R-Precision is reported as a *descriptive*
cross-comparison with prior CIA literature, but the hypothesis test runs on
set-F1.

**Q: Why was CONTAINS added as the 14th edge type after the blueprint specified
13?**
A: The File↔symbol membrane was breaking full-stack BFS — code seeds could
not reach their containing file or vice versa. Adding `CONTAINS` (forward
File→symbol, reverse for BFS) was the minimum-bloat fix. The canonical
edge-type count rose from 13 to 14, and both source-of-truth documents
(`master_blueprint.md`, `analysis_implementation.md`) reflect the new
edge type.

**Q: Why is LLM #5 not allowed to write entity-level justifications? Isn't that
under-using a powerful model?**
A: The **Distributed Justification Principle**. A single
late-stage LLM call seeing 50+ entities produces glossy but generic prose; it
cannot match the per-entity calibration that LLM #2/#3/#4 each have at the
point of admission. Each validator is the authority for what it admits;
LLM #5 is an aggregator that writes *file-level* summaries and the executive
summary only. This is enforced in code: the entity array is built by
`build_deterministic_impacted_entities()` outside the LLM call, and the
runner asserts a 1:1 mapping between LLM #5's `file_justifications` and the
deterministic file set.

**Q: Why fail-closed batching instead of best-effort?**
A: Silent admission of unvalidated candidates is the failure mode that
*looks* like high recall but is actually hallucinated. The contract is: if
the validator did not affirmatively confirm a candidate, the candidate is
dropped, and the run is flagged `degraded_run=True`. The user sees that the
report is potentially incomplete; they never see a falsely-confirmed
candidate.

**Q: Why is determinism (NFR-01) marked as failing?**
A: AST parsing, embedding, RRF fusion, BFS, gates, and metric computation
are **bit-identical** across runs. The only stochastic element is the LLM —
even at `temperature=0` and a fixed `seed`, Gemini Flash Lite's
structured-output mode produces non-identical verdict sets on identical
inputs. This is a model-provider property, not a system bug. Thesis Chapter V
reports it as a documented model-side limit; the rest of the pipeline is
deterministic and the LLM call inputs (the prompts) *are* bit-identical run
to run.

---

## Appendix B — Glossary

- **CR** — Change Request. The natural-language input.
- **CIS** — Change Impact Set. The output: every entity predicted to be
  affected by the CR. Two parts: `sis_nodes` (depth-0 seeds) and
  `propagated_nodes` (BFS-discovered).
- **SIS** — Starting Impact Set. The retrieval-and-validation result that
  becomes the seeds for BFS. SIS ⊆ CIS.
- **RRF** — Reciprocal Rank Fusion. Combines multiple ranked lists into one
  via `1 / (rrf_k + rank)`.
- **GT** — Ground Truth. Human-annotated `(impacted_files, impacted_entities)`
  pairs per CR.
- **NFR** — Non-Functional Requirement. The 5 properties verified by
  `nfr_verify.py`.
- **Fail-closed** — If a validator did not affirmatively confirm a candidate,
  drop the candidate. Opposite of fail-open (which would silently admit).
- **Distributed Justification Principle** — Each entity in the final report
  carries a justification produced verbatim by the validator that admitted
  it. LLM #5 never authors entity-level claims.
- **f1_set** — Unbounded set-level F1 against the full unpruned CIS. The
  single pre-registered hypothesis-test target.
- **Variant** — One of V0..V7, a feature-toggle configuration for ablation.
- **degraded_run** — Boolean on the final report. True if any validator batch
  failed all retries and was dropped fail-closed. The CR completed but the
  impact set may be incomplete.
- **analysis_mode** — Either `retrieval_only` (no BFS ran or no propagation
  happened) or `retrieval_plus_propagation` (BFS ran and produced ≥ 1
  propagated node).
