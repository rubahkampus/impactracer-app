# ImpacTracer v4.0 — Master Blueprint

**Status:** LOCKED. This is the authoritative design specification for the artefact contribution of a Master's thesis on RAG-based Change Impact Analysis. Every architectural decision in the codebase traces to a section of this document.

**Companions:**
- `index_implementation.md` — offline indexer operational reference (the "how the indexer actually runs" book, with citrakara-specific numbers and deltas).
- `analysis_implementation.md` — online pipeline operational reference (the "how the runtime pipeline actually runs" book, with per-LLM prompt contracts, fail-closed semantics, attrition data, and frozen invariants).
- `implementation_report.md` — append-only sprint history (the audit trail; not authoritative for current behaviour).

**Scaffold:** the modules below are mature. Any session that wants to modify them must STOP and report — these define the contract.

- `impactracer/shared/models.py` (all Pydantic schemas)
- `impactracer/shared/constants.py` (EDGE_CONFIG, LAYER_COMPAT, RRF_PATH_WEIGHTS, fan-in caps, blacklists)
- `impactracer/shared/config.py` (Settings)
- `impactracer/persistence/sqlite_client.py` (DDL)
- `impactracer/persistence/chroma_client.py` (collection init)
- `impactracer/pipeline/llm_client.py` (OpenRouter transport, audit log)
- `impactracer/pipeline/interpreter.py` (LLM #1 wrapper)
- `impactracer/pipeline/synthesizer.py` (LLM #5 wrapper)
- `impactracer/evaluation/variant_flags.py` (canonical 8 variants V0..V7)

---

## 0. Real Paths and Environment

```
ImpacTracer repo:        C:\Users\Haidar\Documents\thesis\impactracer-app
Target repo (citrakara): C:\Users\Haidar\Documents\thesis\citrakara
Target docs (SRS+SDD):   C:\Users\Haidar\Documents\thesis\citrakara\docs
SQLite DB:               C:\Users\Haidar\Documents\thesis\impactracer-app\data\impactracer.db
ChromaDB store:          C:\Users\Haidar\Documents\thesis\impactracer-app\data\chroma_store
LLM audit log:           C:\Users\Haidar\Documents\thesis\impactracer-app\data\llm_audit.jsonl
Locked parameters:       C:\Users\Haidar\Documents\thesis\impactracer-app\data\locked_parameters.json
Persistent memory:       C:\Users\Haidar\Documents\thesis\impactracer-app\implementation_report.md
HF cache:                ~/.cache/huggingface/hub  (OS-managed)
```

Forward slashes in all node IDs and file paths, even on Windows. `pathlib.Path.as_posix()` is the canonical form.

---

## 1. The 21 Atomic Functional Requirements

| FR | Name | Module | LLM |
|----|------|--------|-----|
| FR-A1 | Pemotongan Dokumen Deterministik | `indexer/doc_indexer.py` | — |
| FR-A2 | Klasifikasi Jenis Potongan Dokumen | `indexer/doc_indexer.py` | — |
| FR-A3 | Penguraian AST Kode Sumber | `indexer/code_indexer.py` (Pass 1) | — |
| FR-A4 | Ekstraksi Graf Ketergantungan Struktural | `indexer/code_indexer.py` (Pass 2) | — |
| FR-A5 | Pembangkitan Vektor Embedding | `indexer/embedder.py` | — |
| FR-A6 | Pembangkitan Abstraksi Logika Internal | `indexer/skeletonizer.py` | — |
| FR-A7 | Prakomputasi Tabel Kandidat Keterlacakan | `indexer/traceability.py` | — |
| FR-B1 | Validasi Kelayakan CR | `pipeline/interpreter.py` | #1 |
| FR-B2 | Pembangkitan Objek CRInterpretation | `pipeline/interpreter.py` | #1 |
| FR-C1 | Pencarian Hibrida Dua-Jalur | `pipeline/retriever.py` | — |
| FR-C2 | Pemeringkatan Kandidat (Adaptive RRF) | `pipeline/retriever.py` | — |
| FR-C3 | Pemeringkatan Ulang Cross-Encoder | `indexer/reranker.py` (used online) | — |
| FR-C4 | Penyaringan Pra-Validasi (3 gates) | `pipeline/prevalidation_filter.py` | — |
| FR-C5 | Validasi Hasil Pencarian Awal | `pipeline/validator.py` | #2 |
| FR-C6 | Resolusi Keterlacakan Dokumen | `pipeline/seed_resolver.py` | — |
| FR-C7 | Validasi Hasil Resolusi Keterlacakan | `pipeline/traceability_validator.py` | #3 |
| FR-D1 | Penelusuran Graf Ketergantungan | `pipeline/graph_bfs.py` | — |
| FR-D2 | Validasi Hasil Penelusuran Graf | `pipeline/traversal_validator.py` | #4 |
| FR-E1 | Pengambilan Tautan Keterlacakan Balik | `pipeline/context_builder.py` | — |
| FR-E2 | Konstruksi Paket Konteks Laporan | `pipeline/context_builder.py` | — |
| FR-E3 | Pembangkitan Objek ImpactReport | `pipeline/synthesizer.py` | #5 |

---

## 2. Architectural Invariants (Non-Negotiable)

1. **Five LLM invocations in V7.** Names: `interpret`, `validate_sis`, `validate_trace`, `validate_propagation`, `synthesize`. Step 7 additionally runs an internal child-validation call (parameterised with the same LLM #4 prompt) for collapsed children — this does NOT count as a sixth distinct stage.
2. **Deterministic structural pipeline.** AST extraction, embedding, RRF fusion, BFS propagation, and all three gates produce bit-identical output on identical input. Determinism in LLM steps is enforced by `temperature=0`, `seed=42`, Pydantic `response_schema`.
3. **All LLM outputs are Pydantic-schema-constrained** via `LLMClient.call(response_schema=...)`. Never free-form text. Every schema inherits from `TruncatingModel` (in `shared/models.py`).
4. **3 `change_type` values only:** `ADDITION`, `MODIFICATION`, `DELETION`. Uppercase, no others.
5. **14 structural edge types** (canonical and frozen): `CALLS, INHERITS, IMPLEMENTS, TYPED_BY, FIELDS_ACCESSED, DEFINES_METHOD, HOOK_DEPENDS_ON, PASSES_CALLBACK, IMPORTS, RENDERS, DEPENDS_ON_EXTERNAL, CLIENT_API_CALLS, DYNAMIC_IMPORT, CONTAINS`. Adding a new type requires the SQLite CHECK migration.
6. **10 node types** (canonical and frozen): `File, Class, Function, Method, Interface, TypeAlias, Enum, ExternalPackage, InterfaceField, Variable`. `Variable` covers `const NAME = <new_expression|object|array|call_expression>` (Mongoose schemas, factory results, large frozen objects, template arrays); arrow-function `const` declarations remain `Function`.
7. **8 canonical ablation variants V0..V7.** No V3.5, no V6.5. `VariantFlags.ALL_VARIANTS = ["V0","V1","V2","V3","V4","V5","V6","V7"]`. V3 represents the deterministic-filtering peak (cross-encoder + all three gates, no LLM gating). V7 is the full pipeline (BFS + LLM #4 + LLM #5 aggregator).
8. **Forward slashes everywhere.** `pathlib.Path.as_posix()` for any path written to SQLite, ChromaDB metadata, or `node_id`.
9. **Single pre-registered statistical test.** V7 vs V5, one-sided paired Wilcoxon signed-rank, on the **entity-level `f1_set`** metric. `MIN_PAIRED_N = 15`. `ALPHA = 0.05`. **No Bonferroni** (only one test exists).
10. **Set-level metrics only.** `f1_set`, `precision_set`, `recall_set`, computed against the full unpruned validated CIS. There is no `F1@K` in the codebase. Bounded top-K metrics cannot distinguish a graph-flood result (recall via flood) from a focused one and are therefore forbidden.
11. **Distributed Justification Principle.** Every entity in `ImpactReport.impacted_entities` carries a `justification` produced verbatim by the validator that admitted it (LLM #2 / LLM #3 / LLM #4) or by a synthetic `auto_exempt` string for depth-1 IMPLEMENTS / DEFINES_METHOD / TYPED_BY edges. LLM #5 may author the executive_summary and the per-file `justification` field of `impacted_files` — never per-entity claims.
12. **Fail-CLOSED at every validator.** Per-item (missing verdict → DROP) AND per-batch (uncaught exception after retries → DROP entire batch, continue). The runner sets `degraded_run=True` on any drop. No validator may raise into the runner.
13. **Truncation decoupled from output.** The LLM #5 prompt may be truncated to fit the token budget; `ImpactReport.impacted_entities` always contains the FULL validated CIS regardless.
14. **No git, no managed services.** Edges are extracted from AST + file content only. All stores are local files. No managed vector DB, no remote queue.

---

## 3. Offline Indexer (FR-A1 through FR-A7)

Operational detail (live behaviour, real numbers, deltas from this blueprint) lives in `index_implementation.md`. This section is the design contract.

### 3.1 Document Indexer (`indexer/doc_indexer.py`) — FR-A1, FR-A2

- Chunk Markdown at H2 and H3 boundaries via `mistune` 3.x AST (`renderer=None`). H1 is ignored as a boundary. H4+ content absorbs into the enclosing chunk. Chunk text is sliced from raw source lines — the AST is used only to enumerate boundary headings.
- Chunk ID: `{file_stem}__{slugify(section_title)}` (lowercase, non-alphanumeric → `_`). Deterministic across runs.
- Chunk type via case-insensitive substring match on the section title, **NFR rule evaluated before FR** (because "non-functional requirement" contains "functional requirement" as a substring):

```
NFR    ← ["non-fungsional", "non-functional", "kebutuhan non"]
FR     ← ["kebutuhan fungsional", "functional requirement", "use case"]
Design ← ["perancangan", "desain", "arsitektur", "design", "architecture"]
General← (fallback)
```

- Output to ChromaDB `doc_chunks` collection.

### 3.2 AST Pass 1 (`indexer/code_indexer.py::extract_nodes`) — FR-A3

Extract **10 node types** from `.ts` and `.tsx` via `tree-sitter-languages`. The `File` node is appended FIRST per file (index 0).

| Node | tree-sitter source |
|------|---------------------|
| `File` | one per source file |
| `Class` | `class_declaration` |
| `Function` | `function_declaration` OR `lexical_declaration` → `variable_declarator` → `arrow_function` |
| `Method` | `method_definition` inside class body |
| `Interface` | `interface_declaration` |
| `TypeAlias` | `type_alias_declaration` |
| `Enum` | `enum_declaration` |
| `InterfaceField` | one synthetic child per `property_signature` of `Interface`/`TypeAlias` (object shape only) |
| `ExternalPackage` | synthetic node per unique non-relative import (no leading `.` and no `@/` alias) |
| `Variable` | `lexical_declaration` → `variable_declarator` whose value is one of `new_expression`, `object`, `array`, `call_expression` AND whose name passes the canonical-name heuristic |

The `Variable` node_type captures `const FOO = new Schema(...)`, `const ROUTE_TABLE = [...]`, `const Mapping = { ... }`, `const X = createFooBar(...)`. Pure lowercase locals (`tmp`, `i`, `result`) are excluded by the canonical-name heuristic; PascalCase, SCREAMING_SNAKE_CASE, and names with internal capitals are admitted.

**embed_text composition (sent to BGE-M3):**
- Function/Method/Class/Interface/TypeAlias/Enum: `docstring\nsignature` (omit empty lines; fall back to `name`).
- File: `{filename} [{classification}] ({rel_dir})\nexports: {sorted_unique_exports}`.
- Variable: `signature\n{rhs-shape summary}` — for Mongoose-shaped schemas the summary lists field keys; for object literals the top-level keys; for arrays the keys of the first up to 20 element objects.
- ExternalPackage / InterfaceField: empty (not embedded).

**Degenerate-node rule:** nodes with `len(embed_text) < 50` are persisted to SQLite but NOT embedded into ChromaDB. They remain BFS-reachable via CONTAINS edges.

### 3.3 Skeletonizer (`indexer/skeletonizer.py`) — FR-A6

Two-pass tag-and-fold AST reduction on function bodies. Output goes to `code_nodes.internal_logic_abstraction`. Only Function and Method nodes get a non-NULL value.

- **Pass 1:** tag every node of type `call_expression`, `return_statement`, `throw_statement`, `import_declaration` AND every ancestor of those nodes as DO-NOT-ERASE.
- **Pass 2:** emit source bytes per node. Tagged nodes recurse verbatim. Untagged nodes apply fold rules (first match wins): JSX → `/* [JSX: N elements] */`; large arrays (>3 items, not a hook dep array) → `/* [array: N items] */`; large objects (>4 props, not tagged) → `/* [object: N props] */`; logic blocks without high-signal children → `/* [logic block] */`; long template strings/strings (>100/>80 chars) → length placeholder; comments deleted; import_declaration / call_expression always verbatim.

Hook dep arrays are detected by being the last argument to `useEffect`/`useCallback`/`useMemo`/`useLayoutEffect` and are not folded.

### 3.4 AST Pass 2 (`indexer/code_indexer.py::extract_edges`) — FR-A4

Emit **14 edge types**. Pass 2 runs after Pass 1 has populated `code_nodes` for ALL files in the work set, so cross-file resolution works.

| Edge | Source → Target | Direction | Max depth (BFS) |
|------|---|---|---|
| `CALLS` | Function/Method → Function/Method | reverse | **2** |
| `INHERITS` | Class → Class | reverse | 3 |
| `IMPLEMENTS` | Class → Interface | reverse | 3 |
| `TYPED_BY` | Function/Method → Interface/TypeAlias | reverse | 3 |
| `FIELDS_ACCESSED` | Function → InterfaceField | reverse | 2 |
| `DEFINES_METHOD` | Class → Method | forward | 1 |
| `PASSES_CALLBACK` | Function → Function | forward | 1 |
| `HOOK_DEPENDS_ON` | Function → Function/Interface | reverse | 1 |
| `IMPORTS` | File → File | reverse | 1 |
| `RENDERS` | Function → Function | reverse | 1 |
| `DEPENDS_ON_EXTERNAL` | File → ExternalPackage | reverse | 1 |
| `CLIENT_API_CALLS` | Function → API_ROUTE Function | reverse | 1 |
| `DYNAMIC_IMPORT` | File/Function → File | reverse | 1 |
| `CONTAINS` | File → {Function, Method, Interface, TypeAlias, Class, Enum, InterfaceField, Variable} and Interface → InterfaceField | reverse | 1 |

These depths and directions are encoded in `shared/constants.py::EDGE_CONFIG`. Use that constant; never hard-code per-edge BFS rules elsewhere.

**LLM #4 auto-exempt edges:** depth-1 IMPLEMENTS, DEFINES_METHOD, TYPED_BY skip propagation validation. They receive a synthetic justification `"Direct <edge> contract from <seed> — auto-admitted exempt edge."`.

**File classification (path glob, first match wins):**

| Pattern | Class |
|---------|-------|
| `src/app/**/route.{ts,tsx}` | `API_ROUTE` |
| `src/app/**/page.{ts,tsx}` / `layout.*` | `PAGE_COMPONENT` |
| `src/components/**` | `UI_COMPONENT` |
| `src/hooks/**` | `UI_COMPONENT` |
| `src/lib/stores/**` | `UI_COMPONENT` |
| `src/lib/test/**`, `__mocks__/**`, `__tests__/**` | (None — must precede the `src/lib/db/models/` rule) |
| `src/lib/db/models/**` | `TYPE_DEFINITION` |
| `src/lib/**`, `src/utils/**` | `UTILITY` |
| `src/types/**` | `TYPE_DEFINITION` |
| else | (None) |

Edge-specific implementation notes (alias resolution, positional wildcard route matching, middleware synthetic edges, Mongoose TYPED_BY) are documented in `index_implementation.md` §4.

### 3.5 Embedder (`indexer/embedder.py`) — FR-A5

`BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)`. Output dim 1024. Batch size 32, max length 512. Returns dense vectors only (sparse and ColBERT off). Pre-warm via `huggingface_hub.snapshot_download` on first construction so subsequent runs are offline-safe.

### 3.6 Reranker (`indexer/reranker.py`) — FR-C3

`FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)`. Used in **online Step 3 only**, never by the offline indexer. Sigmoid-normalised scores in `[0, 1]`. Supports multi-query MAX aggregation: each candidate is scored against every LLM #1 `search_query`, and the max is kept.

### 3.7 Traceability (`indexer/traceability.py`) — FR-A7

1. L2-normalise all code and doc vectors.
2. `cos_matrix = code_matrix @ doc_matrix.T` (single matmul).
3. For each `(code_id, doc_id)` pair compute `adjusted = cos_matrix[i,j] × layer_compat(code_classification, chunk_type)` where `layer_compat` is in `shared/constants.py::LAYER_COMPAT`.
4. **Forward pass:** per-code-node top-K above `settings.min_traceability_similarity` (=0.40).
5. **Reverse pass:** per-doc-chunk top-K above the same floor. The forward+reverse union prevents NFR / General chunks (low layer_compat) from being squeezed out by high-affinity competitors despite genuine cosine signal.
6. `INSERT OR REPLACE INTO doc_code_candidates(code_id, doc_id, weighted_similarity_score)`.

`top_k_traceability = 5` (default).

### 3.8 Indexer Runner (`indexer/runner.py`)

Ten ordered steps; see `index_implementation.md` §1 for the runtime detail.

```
0. Connect SQLite, init_schema. Init ChromaDB collections.
1. Scan repo for .md and .ts/.tsx files.
2. Compute SHA-256 of each file. Compare with file_hashes table.
   work_set = files where hash differs OR file is new.
   If --force: work_set = all files; truncate file_hashes.
3. For deleted files (in file_hashes but not in scan):
     purge rows from code_nodes / structural_edges / doc_code_candidates /
     file_hashes / file_dependencies and their ChromaDB ids.
4. Chunk Markdown files in work_set → ChromaDB doc_chunks.
5. Pass 1 over each work_set .ts/.tsx → INSERT OR REPLACE into code_nodes.
6. Pass 2 over each work_set .ts/.tsx + their reverse-dep neighbours
   (from file_dependencies) → INSERT OR IGNORE into structural_edges.
7. Embed pending non-degenerate code_nodes via Embedder.embed_batch (batch=32).
   Embed pending doc_chunks. Insert into ChromaDB code_units / doc_chunks.
8. Recompute traceability table (full recompute — layer-weighted scores
   depend on the population).
9. Update file_hashes for every successfully processed file.
10. Update index_metadata: edge_schema_version, indexing_timestamp,
    embedding_model_name, traceability_k_parameter, total_*.
```

The indexer is incremental by default; `--force` discards `file_hashes` and re-indexes everything.

---

## 4. Online Pipeline (FR-B1 through FR-E3)

Nine steps, orchestrated by `impactracer/pipeline/runner.py::run_analysis`. Variant flags toggle each step independently. The description below is V7. Operational detail (prompts, fail-closed semantics, attrition data) is in `analysis_implementation.md`.

### Step 0 — Load context

```python
@dataclass
class PipelineContext:
    conn: sqlite3.Connection
    doc_col: chromadb.Collection
    code_col: chromadb.Collection
    graph: nx.MultiDiGraph
    doc_bm25: BM25Okapi
    doc_bm25_ids: list[str]
    code_bm25: BM25Okapi
    code_bm25_ids: list[str]
    embedder: Embedder
    reranker: Reranker
    llm_client: LLMClient
    variant_flags: VariantFlags
```

`build_graph_from_sqlite(conn)` materialises a `MultiDiGraph` once per analysis session. `build_bm25_from_chroma(collection)` rebuilds BM25 from ChromaDB documents on every load.

### Step 1 — Interpret CR (LLM #1, FR-B1, FR-B2)

`pipeline/interpreter.py::interpret_cr`. Returns `CRInterpretation` with ten fields:

```
is_actionable: bool
actionability_reason: str | None
primary_intent: str
change_type: ADDITION | MODIFICATION | DELETION
affected_layers: list[Literal["requirement", "design", "code"]]
domain_concepts: list[str]
search_queries: list[str]   # English even when CR is Indonesian
named_entry_points: list[str]
out_of_scope_operations: list[str]
is_nfr: bool
```

If `is_actionable=False`, the runner short-circuits to a minimal rejection ImpactReport and exits. A coherence soft-fix is then applied: DELETION CRs must include `"code"` in `affected_layers`; ADDITION CRs must not be code-only.

### Step 2 — Adaptive RRF Hybrid Search (FR-C1, FR-C2)

`pipeline/retriever.py::hybrid_search(cr_interp, ctx, settings, cr_text)`.

Four ranked lists are assembled per LLM #1 search-query, then RRF-fused:

1. **dense_doc** — `BGE-M3 embedding × ChromaDB doc_chunks` (filtered by `doc_filter` derived from `affected_layers`).
2. **bm25_doc** — `rank_bm25` over chunked SRS/SDD. Tokenizer: camelCase split, length ≥ 2, English + Indonesian stop-word list.
3. **dense_code** — `BGE-M3 embedding × ChromaDB code_units` (when `"code" ∈ affected_layers`).
4. **bm25_code** — `rank_bm25` over code embed_text.

The `dense_code` path additionally:

- **Raw-CR multilingual dense pass** — when `settings.enable_raw_cr_dense_pass = True` and `cr_text` is provided, the retriever embeds the raw (Indonesian / English / mixed) CR text once and merges its nearest-neighbour code candidates into `dense_code` via score-max. BGE-M3's multilingual capability bridges the Indonesian-CR ↔ English-identifier gap without going through LLM #1.
- **Traceability pool seeding** — when `settings.enable_traceability_pool_seeding = True`, code nodes that the offline `doc_code_candidates` table links to any retrieved doc-chunk (above `settings.traceability_seed_min_score = 0.40`, capped at `settings.traceability_seed_top_k_per_doc = 5` per doc) are injected into `dense_code` with a synthetic rank. Promotes the offline traceability precomputation from a rerank +0.1 bonus into a pool-membership signal.

**Adaptive RRF formula:**
```
ARRF(d) = Σ_{p ∈ paths_present} W[change_type][p] / (rrf_k + rank_p(d) + 1)
```
where `W = RRF_PATH_WEIGHTS[change_type]` from `shared/constants.py`. `rrf_k = 60`.

After fusion: sort all unique node ids by ARRF score descending; take the top `settings.top_k_rrf_pool` (=200) candidates. Build `Candidate` DTOs by hydrating ChromaDB metadata + SQLite columns.

**Variant degenerations** of Step 2:
- V0: only `bm25_doc` + `bm25_code`.
- V1: only `dense_doc` + `dense_code` (raw-CR pass and traceability seeding still active because they target the dense_code list).
- V2: BM25 + Dense + RRF fusion across all four paths.
- V3+: same as V2 (the cross-encoder enters in Step 3).

### Step 3 — Cross-Encoder Rerank (FR-C3)

Skip entirely when `enable_cross_encoder=False` (V0–V2).

```python
candidates = ctx.reranker.rerank_multi_query(
    cr_interp.search_queries, candidates, top_k=settings.max_admitted_seeds,
)
```

`max_admitted_seeds = 15`. Each candidate retains its raw cross-encoder logit in `raw_reranker_score` and a sigmoid-normalised score in `reranker_score`.

Post-rerank score adjustments:
- **Traceability bonus** (+0.10) on `raw_reranker_score` for code candidates that any retrieved doc-chunk traceability-links to.
- **Negative filter** (additive −5.0 on `raw_reranker_score`) for candidates whose name or snippet contains an entry from `cr_interp.out_of_scope_operations`. Additive — multiplicative would invert sign on negative logits and inadvertently promote out-of-scope candidates.

### Step 3.5 / 3.6 / 3.7 — Pre-Validation Gates (FR-C4)

`pipeline/prevalidation_filter.py`. Each gate is independently toggleable.

- **3.5 Score floor** (`enable_score_floor`): drop candidates with `raw_reranker_score < settings.min_reranker_score_for_validation` (=-2.0). Sanity-only gate; the real precision gate is LLM #2.
- **3.6 Semantic dedup** (`enable_dedup_gate`): for each `doc_chunks` candidate, look up its top-1 code resolution via `doc_code_candidates`. If that code_id is already in the candidate list, collapse the doc into the code candidate (attach `(section_title, text)` as Business Context for the LLM #2 prompt) and drop the doc.
- **3.7 Plausibility — density only** (`enable_plausibility_gate`): if a single file accounts for more than `settings.plausibility_gate_density_threshold` (=0.50) of code candidates, drop those candidates UNLESS their name matches a `named_entry_point`. No per-file count cap.

### Step 4 — Validate SIS (LLM #2, FR-C5)

`pipeline/validator.py`. Batched ≤5 candidates per call. Returns `(confirmed_ids, justifications, degraded)`.

**Fail-CLOSED:**
- Per node: missing verdict → DROP that candidate.
- Per batch: any uncaught exception (after `LLMClient.call` retry exhaustion) is caught, the batch is dropped, `degraded=True` set, loop continues with next batch.

Captures `function_purpose`, `mechanism_of_impact`, `justification` per confirmed seed — propagated verbatim to `NodeTrace.justification` with `justification_source="llm2_sis"`.

Lost-in-the-middle reorder: position 0 = highest reranker_score, position N−1 = lowest, middle filled in ascending order. The prompt forbids any retrieval-score mention (anti-circular mandate).

### Step 5 — Resolve doc-chunk SIS to code seeds (FR-C6)

`pipeline/seed_resolver.py`. Direct code-node SIS entries pass through as `direct_code_seeds`. Doc-chunk SIS entries are looked up in `doc_code_candidates` for their top-K code nodes (`top_k_traceability=5`). Returns `(resolutions, doc_to_code_map, direct_code_seeds)`.

### Step 5b — Validate trace resolution (LLM #3, FR-C7)

`pipeline/traceability_validator.py`. Batched ≤5 pairs per call. Returns `(validated_code_seeds, low_confidence_map, justifications, degraded)`.

Skip if `enable_trace_validation=False` → take top-1 of each resolution as a blind seed (marked low_confidence_seed=True).

Three-way per-pair decision: CONFIRMED → seed admitted; PARTIAL → seed admitted with `low_confidence_seed=True`; REJECTED → dropped. Each surviving code_id keeps the BEST decision's justification (CONFIRMED > PARTIAL > REJECTED) → propagated to `NodeTrace.justification` with `justification_source="llm3_trace"`.

**Fail-CLOSED:** per-pair missing verdict → REJECTED; per-batch exception → entire batch REJECTED, continue.

### Step 6 — BFS Propagation (FR-D1)

`pipeline/graph_bfs.py`. Multi-source BFS over the materialised `MultiDiGraph`. Per-edge direction and max-depth from `EDGE_CONFIG`. Global cap `settings.bfs_global_max_depth = 3`.

Confidence-tier rule: top-N seeds by reranker score are "high-conf" (N = `settings.bfs_high_conf_top_n = 5`). Doc-chunk reranker scores propagate to resolved code seeds via `dict.setdefault`.

`LOW_CONF_CAPPED_EDGES = {CALLS}` — low-confidence seeds cap CALLS depth at 1 instead of 2.

`UTILITY_FILE_CALLS_DEPTH_CAP = 1` — seeds whose `file_classification == "UTILITY"` cap their reverse-CALLS chain at depth 1 regardless of confidence tier. Utility functions are called from everywhere; deeper reverse-CALLS from a UTILITY seed is a near-guaranteed flood.

`_HUB_DEGREE_THRESHOLD = 20` — nodes with total degree > 20 (generic interfaces, framework primitives like `ext::react`) cap at depth 1 for all edges when traversing FROM them.

`NODE_TYPE_MAX_FAN_IN` (from `shared/constants.py`): propagated neighbours with in-degree exceeding the type-specific cap are excluded from the CIS unless they are themselves SIS seeds.

| Node type | Max fan-in |
|---|---:|
| Function / Method / Class | 50 |
| Interface / TypeAlias / Enum | 100 |
| InterfaceField / File | 200 |
| Variable | 80 |
| ExternalPackage | 0 (always excluded — `EXCLUDED_PROPAGATION_NODE_TYPES`) |

Outputs `CISResult(sis_nodes, propagated_nodes)`. Each `NodeTrace` records `depth, causal_chain, path, source_seed, low_confidence_seed, collapsed_children, justification, justification_source, function_purpose, mechanism_of_impact`.

### Step 6.5 — CONTAINS sub-tree collapse

InterfaceField nodes that reach the CIS via CONTAINS-only paths are collapsed into their parent Interface's `NodeTrace.collapsed_children`. Reduces token cost for LLM #4 and LLM #5 without losing information; each collapsed child is individually re-validated in Step 7.

### Step 7 — Validate propagation (LLM #4, FR-D2)

`pipeline/traversal_validator.py`. Batched ≤5 propagated nodes per call. Returns `(filtered_cis, justifications, degraded)`.

Skip if `enable_propagation_validation=False`.

**Auto-exempt:** depth-1 IMPLEMENTS / DEFINES_METHOD / TYPED_BY edges bypass the LLM call (direct structural contracts whose impact is definitional). They receive a synthetic justification `"Direct <edge> contract from <seed> — auto-admitted exempt edge."` with `justification_source="auto_exempt"`.

**Per-child collapse validation:** for each surviving parent with `collapsed_children`, an additional LLM-#4-style call individually validates each child name. Same fail-closed semantics as the main batch.

The prompt shows the causal chain as factual context but forbids edge-type-as-evidence reasoning. Required justification format demands a contract-breakage, behavioural-anomaly, or downstream-type-mismatch sentence; generic "function A calls function B" / "in the same module" patterns are explicitly rejected.

`random.seed(42); random.shuffle(to_validate)` before batching to neutralise positional bias without compromising reproducibility.

**Fail-CLOSED** at per-node, per-batch, and per-child-batch levels.

### Step 8 — Context build (FR-E1, FR-E2)

`pipeline/context_builder.py`.

```python
backlinks = fetch_backlinks(cis.all_node_ids(), conn, top_k_backlinks_per_node)
snippets  = fetch_snippets(cis.all_node_ids(), conn)  # ILA preferred; source_code fallback
context   = build_context(cr_text, cr_interp, cis, backlinks, snippets, settings)
```

Context includes raw CR, CRInterpretation JSON, SIS list with reranker scores, CIS with BFS path + chain per node, depth, top-3 backlinks per code node, precomputed severity per node, and a canonical `=== IMPACTED FILES (write a file_justifications row for EACH of these) ===` header listing every distinct `file_path` in the deterministic CIS.

Severity comes from `severity_for_chain(causal_chain)` — the LAST edge in the chain (eliminating "severity laundering" where deep CALLS chains inherit a final IMPLEMENTS's Tinggi severity). SIS seeds (empty chain) are `Tinggi` by convention.

**Token-budget enforcement** (severity-aware truncation order):
1. SIS seeds first (depth=0).
2. Propagated nodes: severity ASC (Tinggi < Menengah < Rendah), then depth ASC, then alphabetic node_id.
3. Hard limit failsafe: 240,000 chars max.

Truncation affects only the LLM #5 prompt. The report's `impacted_entities` always contains the full validated CIS.

### Step 9 — Synthesize (LLM #5, FR-E3)

`pipeline/synthesizer.py`. Aggregator-only role.

Input: the truncated context + the canonical file set.

Output: `LLMSynthesisOutput`:
```
executive_summary: str          # one paragraph for non-technical stakeholders
documentation_conflicts: list[str]   # doc_chunk ids that may conflict with the change
file_justifications: list[FileJustificationItem]  # one row per file in the canonical set
```

LLM #5 NEVER produces an `impacted_entities` array. The runner reconciles `file_justifications` against the deterministic file set: hallucinated files are dropped silently; omitted files receive a deterministic fallback summarising the entity-level justifications inside that file.

**Fail-closed:** if `LLMClient.call` raises after retry exhaustion, the runner falls back to `build_minimal_summary` (deterministic; `degraded_run=True`). `impacted_entities` and `impacted_files` are emitted regardless — they exist independently of LLM #5.

---

## 5. `ImpactReport` Output Schema

The user-visible artefact aligns 1:1 with the Ground Truth dual-granularity schema.

```
ImpactReport
├── executive_summary: str              (LLM #5)
├── impacted_files: list[ImpactedFile]  (deterministic file_path set + LLM #5
│                                        justification or deterministic fallback)
├── impacted_entities: list[ImpactedEntity]
│        (deterministic; every validated CIS node; justification verbatim from
│         LLM #2/#3/#4 or synthetic auto_exempt)
├── documentation_conflicts: list[str]  (LLM #5)
├── estimated_scope: "local" | "menengah" | "luas"   (deterministic from CIS size)
├── analysis_mode: "retrieval_only" | "retrieval_plus_propagation"
└── degraded_run: bool                  (True if any LLM batch was dropped)
```

`ImpactedEntity` fields:
```
node: str                   # canonical node_id, matches GT format "file_path::EntityName"
node_type: NodeType
file_path: str
severity: "Tinggi" | "Menengah" | "Rendah"
causal_chain: list[str]     # ordered edge_types from SIS root to this entity
justification: str          # max 400 chars
justification_source: str   # "llm2_sis" | "llm3_trace" | "llm4_propagation" | "auto_exempt" | "bfs_only" | "retrieval_only"
traceability_backlinks: list[str]
```

`ImpactedFile` fields:
```
file_path: str
justification: str   # max 600 chars; LLM #5 prose or deterministic fallback
```

A backward-compatibility property `ImpactReport.impacted_nodes` returns `impacted_entities` (do not use in new code).

The runner also writes a per-step trace dict to `impact_report_full.json` when `trace_sink` is provided (the CLI always provides it). Trace keys:
```
step_1_interpretation, step_2_rrf_pool, step_3_reranked,
step_3_gates_survivors, step_4_llm2_verdicts, step_5_resolutions,
step_5b_llm3_verdicts, step_6_bfs_raw_cis, step_7_llm4_verdicts,
final_report
```

---

## 6. Ablation Harness (Canonical 8 Variants)

`evaluation/variant_flags.py`. `VariantFlags.ALL_VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]`.

| Variant | BM25 | Dense | RRF | XEnc | ScoreFlr | Dedup | Plaus | LLM #2 | LLM #3 | BFS | LLM #4 |
|---------|------|-------|-----|------|----------|-------|-------|--------|--------|-----|--------|
| V0 | ✓ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| V1 | ☐ | ✓ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| V2 | ✓ | ✓ | ✓ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| V3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ☐ | ☐ | ☐ | ☐ |
| V4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ☐ | ☐ | ☐ |
| V5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ☐ | ☐ |
| V6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ☐ |
| V7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

LLM #1 (`interpret`) and LLM #5 (`synthesize`) run for every variant.

V3 absorbs the former diagnostic V3.5: it represents the deterministic-filtering peak (cross-encoder + score floor + semantic dedup + density gate, no LLM gating). V7 absorbs the former V6.5: with LLM #5 demoted to aggregator-only, V6.5 and V7 are behaviourally identical.

**Cross-variant additive assertions** (the harness verifies these per CR):
- `|CIS(V7)| ≤ |CIS(V6)|` — LLM #4 only rejects.
- `|CIS(V6)| ≥ |CIS(V5)|` — BFS only adds.
- `|SIS(V5)| ≤ |SIS(V4)|` — LLM #3 only rejects.

The harness:
```
For each (cr_id, variant_id) in dataset × ALL_VARIANTS:
  flags = VariantFlags.for_id(variant_id)
  trace_sink = {"cr_text": cr_text, "variant": variant_id, "cr_id": cr_id}
  start = time.perf_counter()
  try:
    report = run_analysis(cr_text, settings, variant_flags=flags,
                          shared_embedder=..., shared_reranker=...,
                          shared_llm_client=..., trace_sink=trace_sink)
    metrics = compute_dual_granularity_metrics(
        predicted_nodes={e.node for e in report.impacted_entities},
        predicted_files={f.file_path for f in report.impacted_files},
        gt_nodes=gt_entry.entity_node_ids(),
        gt_files=gt_entry.file_paths(),
    )
  except Exception as e:
    record error row, status='error'
  Write <output_dir>/<cr_id>/<variant_id>/impact_report.json
  Write <output_dir>/<cr_id>/<variant_id>/impact_report_full.json
  Append metrics row to per_cr_per_variant_metrics.csv
```

**Locked parameters:** at the start of `run_full_evaluation`, the harness writes `data/locked_parameters.json` snapshotting `Settings`. If any parameter changes mid-run, the run is invalidated.

---

## 7. Evaluation Protocol (V7 vs V5 Only)

### 7.1 Metrics

`evaluation/metrics.py`:
- `compute_set_metrics(predicted, gt)` → `{precision_set, recall_set, f1_set, n_predicted, n_gt, n_intersect}`. Computed against the full unpruned predicted set. No top-K.
- `compute_r_precision(ranked, gt)` → diagnostic only; not a hypothesis-test target.
- `compute_dual_granularity_metrics(predicted_nodes, predicted_files, gt_nodes, gt_files)` → flat dict with `entity_*` and `file_*` prefixed keys.

Edge cases: empty predicted → `precision_set=0.0`, `recall_set=0.0`, `f1_set=0.0`. Empty gt → `recall_set=NaN` (CR excluded from the paired Wilcoxon test).

### 7.2 Statistical test

`evaluation/statistical.py`:
- `ALPHA = 0.05`
- `PRIMARY_COMPARISON = ("V7", "V5")`
- `PRIMARY_METRIC = "f1_set"` (entity-level)
- `MIN_PAIRED_N = 15`
- `run_primary_test(df)` runs one-sided paired Wilcoxon (`alternative="greater"`, `zero_method="wilcox"`, `correction=False`) on `V7_f1_set` vs `V5_f1_set`. Returns dict with `hypothesis, variant_a, variant_b, metric, p_value, statistic, cliffs_delta, median_diff, n, accepted, achieved_power_note`.
- Raises `InsufficientPairsError` when n<15.

No Bonferroni — only one test is run.

### 7.3 Reports written by `run_full_evaluation` + CLI orchestrator

| Artefact | Owner | Notes |
|---|---|---|
| `per_cr_per_variant_metrics.csv` | `evaluation/ablation.py` | One row per (cr_id, variant). Long form. |
| `per_cr_per_variant_metrics.jsonl` | `evaluation/ablation.py` | Same data, JSON Lines. |
| `<cr_id>/<variant>/impact_report.json` | `evaluation/ablation.py` | Per-cell user-visible report. |
| `<cr_id>/<variant>/impact_report_full.json` | `evaluation/ablation.py` | Per-cell step-by-step trace. |
| `summary_table.csv` | `evaluation/report_builder.py` | Macro-averaged set-level metrics per variant. |
| `summary_table.md` | `evaluation/report_builder.py` | Markdown rendering for thesis appendix. |
| `statistical_tests.json` | `cli.py::evaluate` orchestrator | The single V7 vs V5 Wilcoxon result. On n<15, written with `status="insufficient_pairs"` plus descriptive Cliff's δ. |
| `calibration_analysis.md` | `cli.py::evaluate` orchestrator | Written analytical commentary on graph-flood + LLM-#4-recovery + variant ranking. |
| `nfr_verification.json` | `evaluation/nfr_verify.py` | Five NFR check results + `all_passed`. Optional (`--verify-nfr`). |
| `locked_parameters.json` | `evaluation/ablation.py` | Settings snapshot. |
| `llm_audit.jsonl` | `pipeline/llm_client.py` | Append-only per-call audit (NFR-05 source). |

### 7.4 NFR verification procedures (`evaluation/nfr_verify.py`)

- **NFR-01 Determinism:** run V7 twice on the same CR; compare `trace_sink["step_5b_llm3_verdicts"]["validated_code_seeds"]` across the two runs. Equal sets ⇒ deterministic. The comparison target is the validated SIS (post-LLM-#3, pre-BFS), NOT `impacted_entities` — BFS and LLM #4 carry network-induced variance NFR-01 is not designed to test.
- **NFR-02 Local Execution:** stubbed per architect mandate (OS-level network disabling cannot be safely scripted from inside the test process). Manual verification: indexer + non-LLM stages run with the network adapter disabled.
- **NFR-03 Latency:** read `elapsed_s` per row from `per_cr_per_variant_metrics.csv`; report `median` and `p95` per variant and overall.
- **NFR-04 Cross-lingual:** run V7 on an Indonesian CR; confirm at least one impacted entity has a plain-ASCII English identifier suffix.
- **NFR-05 Config Consistency:** parse `data/llm_audit.jsonl`; filter entries to those within `[run_start_iso, now]`; assert all entries share a single `config_hash`.

---

## 8. Configuration (`shared/config.py::Settings`)

Defaults that drive the entire pipeline:

```
# ChromaDB / SQLite
db_path                          = "./data/impactracer.db"
chroma_path                      = "./data/chroma_store"
llm_audit_log_path               = "./data/llm_audit.jsonl"
locked_parameters_path           = "./data/locked_parameters.json"

# Embedder / reranker
embedding_model                  = "BAAI/bge-m3"
reranker_model                   = "BAAI/bge-reranker-v2-m3"
embedding_batch_size             = 32
embedding_max_length             = 512

# Indexer
top_k_traceability               = 5
min_traceability_similarity      = 0.40
degenerate_embed_min_length      = 50

# Retrieval
top_k_per_query                  = 30           # per dense/BM25 query per path
top_k_rrf_pool                   = 200          # candidates entering cross-encoder
max_admitted_seeds               = 15           # cap after rerank
rrf_k                            = 60

# Raw-CR multilingual bridge
enable_raw_cr_dense_pass         = True
raw_cr_dense_top_k               = 60

# Traceability pool seeding
enable_traceability_pool_seeding = True
traceability_seed_top_k_per_doc  = 5
traceability_seed_min_score      = 0.40
traceability_seed_synthetic_rank = 5

# Pre-validation gates (FR-C4)
min_reranker_score_for_validation     = -2.0    # sanity-only floor
plausibility_gate_density_threshold   = 0.50    # density-only; no max_per_file cap

# BFS
bfs_global_max_depth             = 3
bfs_high_conf_top_n              = 5

# Context-builder budgets
llm_max_context_tokens           = 100_000
synthesis_system_prompt_tokens   = 1_200
output_reserve_tokens            = 2_000
top_k_backlinks_per_node         = 3

# Scope thresholds
scope_local_max                  = 10
scope_medium_max                 = 30

# Statistical
alpha                            = 0.05

# LLM client
openrouter_api_key               = ""           # from environment
llm_model                        = "google/gemini-2.5-flash"
llm_temperature                  = 0.0
llm_seed                         = 42
llm_max_output_tokens            = 65_536
llm_retry_max_attempts           = 10
llm_retry_base_backoff           = 2.0          # exponential backoff base seconds
```

---

## 9. LLM Transport (`pipeline/llm_client.py`)

OpenRouter is the exclusive transport. The OpenAI SDK is NOT imported; `LLMClient` is a single `httpx`-based POST client targeting `https://openrouter.ai/api/v1/chat/completions`.

Every call:
- `temperature = 0.0`, `seed = 42`.
- Pydantic `response_schema` enforces the output shape via JSON-schema mode.
- `session_config_hash` is a SHA-256 digest of `model + temperature + seed` (NFR-05 audit anchor).
- Retry: exponential backoff `base_backoff ** (attempts − 1)` on 429 / 5xx / network timeout; respects the `Retry-After` header. Up to `llm_retry_max_attempts = 10`.
- Audit log: one JSONL line appended to `data/llm_audit.jsonl` per call with `call_index, call_name, status, config_hash, model, timestamp, prompt_hash, response_hash, retry_count, prompt_tokens?, completion_tokens?, error?`.

No other module in the codebase may import the HTTP layer directly.

---

## 10. CLI Surface (`cli.py`)

Three commands.

### `impactracer index <repo_path> [--force]`

Build or update the knowledge representation. Orchestrates the offline indexer (FR-A1 .. FR-A7). Prints stats: `files_scanned, files_reindexed, code_nodes, doc_chunks, edges, elapsed`.

### `impactracer analyze "<cr_text>" [--output PATH] [--variant V0..V7]`

Run the nine-step online pipeline on a single CR. Writes both `impact_report.json` (user-visible) and `impact_report_full.json` (step-by-step trace) to `--output` (and its `_full.json` sibling). Default `--variant V7`.

### `impactracer evaluate --dataset DIR --output DIR [--run-full-ablation] [--verify-nfr]`

Run the canonical 8-variant ablation matrix over every `*.json` GT entry in the `--dataset` directory. Each GT file is one `GTEntry` object (NOT an array) with `cr_id, cr_description, impacted_files, impacted_entities`.

Output side: every artefact listed in §7.3, plus the `<cr_id>/<variant_id>/` per-cell directories.

On startup the CLI captures `run_start_iso = datetime.now(timezone.utc).isoformat()` so NFR-05 can scope its audit-log filter to this run. The CLI orchestrator pivots the long-form metrics CSV to wide form before invoking `run_primary_test`; on `InsufficientPairsError` it writes `statistical_tests.json` with `status="insufficient_pairs"` plus descriptive Cliff's δ and median Δ (the same artefact path always exists, regardless of n). Standard-output prints of the summary table, statistical result, and calibration analysis are wrapped in a UTF-8 `TextIOWrapper` so Windows cp1252 consoles can render the `Δ` glyph without crashing.

### `impactracer report [--output PATH]`

Diagnostic. Generates a Markdown indexing-quality report (node/edge counts, type breakdowns, FK integrity, doc-chunk traceability, semantic benchmark top-1, BFS reachability sanity).

---

*End of master_blueprint.md. Operational detail in `index_implementation.md` (offline) and `analysis_implementation.md` (online). Sprint history in `implementation_report.md`.*
