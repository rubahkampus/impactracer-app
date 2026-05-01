# ImpacTracer v4.0 — Finalized Session-Agnostic Blueprint

**Status:** LOCKED. This is the authoritative spec for stateless multi-session implementation. Every new session reads this file.

**Scaffold reference (do NOT rewrite):** the existing `C:\Users\Haidar\Documents\thesis\impactracer-app` repository already contains:
- Full Pydantic schemas in `impactracer/shared/models.py`
- Full constants (RRF weights, LAYER_COMPAT, EDGE_CONFIG, severity mapping, blacklists) in `impactracer/shared/constants.py`
- Full SQLite DDL in `impactracer/persistence/sqlite_client.py`
- Full ChromaDB client in `impactracer/persistence/chroma_client.py`
- Full OpenRouter `LLMClient` in `impactracer/pipeline/llm_client.py`
- Full `interpreter.py` and `synthesizer.py` thin wrappers
- Full `VariantFlags` V0–V7 factories in `impactracer/evaluation/variant_flags.py`

These files are the contract. Sessions IMPLEMENT the empty stubs; they do NOT rewrite the contract.

---

## 0. Real Paths and Environment

```
ImpacTracer repo:    C:\Users\Haidar\Documents\thesis\impactracer-app
Target repo:         C:\Users\Haidar\Documents\thesis\citrakara
Target docs:         C:\Users\Haidar\Documents\thesis\citrakara\docs
Persistent memory:   C:\Users\Haidar\Documents\thesis\impactracer-app\implementation_report.md
SQLite DB:           C:\Users\Haidar\Documents\thesis\impactracer-app\data\impactracer.db
ChromaDB store:      C:\Users\Haidar\Documents\thesis\impactracer-app\data\chroma_store
HF cache:            ~/.cache/huggingface/hub  (default; OS-managed)
```

---

## 1. The 21 Atomic FRs and Their Modules

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
| FR-C3 | Pemeringkatan Ulang Cross-Encoder | `indexer/reranker.py` | — |
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

1. **Five LLM invocations in V7.** Names: `interpret`, `validate_sis`, `validate_trace`, `validate_propagation`, `synthesize`.
2. **Deterministic structural pipeline.** AST, embedding, RRF, BFS, gates produce bit-identical output on identical input.
3. **All LLM outputs are Pydantic-schema-constrained** via `LLMClient.call(response_schema=...)`. Never free-form text.
4. **TruncatingModel base.** Every LLM output schema inherits from it (already done in `shared/models.py`).
5. **Three change_type values only:** `ADDITION`, `MODIFICATION`, `DELETION`.
6. **13 edge types only.** Adding a new type requires schema CHECK migration (do NOT do this).
7. **9 node types only:** File, Class, Function, Method, Interface, TypeAlias, Enum, ExternalPackage, InterfaceField.
8. **Forward slashes in all node IDs and file paths**, even on Windows. Use `pathlib.Path.as_posix()`.
9. **Single statistical test:** V7 vs V5, one-sided paired Wilcoxon, no Bonferroni.
10. **No git, no managed services.** Edges from AST + file content only. All stores are local files.

---

## 3. Offline Indexer (FR-A1 through FR-A7)

### 3.1 Document Indexer (`doc_indexer.py`) — FR-A1, FR-A2

- Chunk Markdown at H2 and H3 boundaries via `mistune`. H1 is ignored as a boundary; H4+ content absorbs into the enclosing chunk.
- Chunk ID: `{file_stem}__{slugified_title}` (lowercase, non-alphanumeric → `_`). Deterministic across runs.
- Chunk type via case-insensitive substring match on the section title:
  - `FR` ← any of: `kebutuhan fungsional`, `functional requirement`, `use case`
  - `NFR` ← any of: `non-fungsional`, `non-functional`, `kebutuhan non`
  - `Design` ← any of: `perancangan`, `desain`, `arsitektur`, `design`, `architecture`
  - `General` ← else
- Output to ChromaDB `doc_chunks` collection: `id=chunk_id`, `document=text`, `metadata={chunk_id, source_file, section_title, chunk_type}`.

### 3.2 AST Indexer Pass 1 (`code_indexer.py`) — FR-A3

Extract 9 node types from `.ts` and `.tsx` via `tree-sitter-languages`. The File node is appended FIRST per file (index 0).

| Node | tree-sitter source |
|------|---------------------|
| `File` | one per source file |
| `Class` | `class_declaration` |
| `Function` | `function_declaration` OR `lexical_declaration` → `variable_declarator` → `arrow_function` |
| `Method` | `method_definition` inside class body |
| `Interface` | `interface_declaration` |
| `TypeAlias` | `type_alias_declaration` |
| `Enum` | `enum_declaration` |
| `ExternalPackage` | synthetic node per unique non-relative import |
| `InterfaceField` | one synthetic child per property of `Interface`/`TypeAlias` (object shape only) |

**React component flag:** uppercase first char + JSX in body → mark as UI/PAGE component (informs classification, not node_type).

**File classification (path glob):**
| Pattern | Class |
|---------|-------|
| `src/app/**/route.{ts,tsx}` | `API_ROUTE` |
| `src/app/**/page.{ts,tsx}`, `src/app/**/layout.{ts,tsx}` | `PAGE_COMPONENT` |
| `src/components/**` | `UI_COMPONENT` |
| `src/lib/**`, `src/utils/**` | `UTILITY` |
| `src/types/**` | `TYPE_DEFINITION` |
| else | NULL |

**route_path derivation** (API_ROUTE only): `src/app/api/commissions/[id]/route.ts` → `/api/commissions/{id}`.

**embed_text composition** (sent to BGE-M3):
- Function/Method/Class/Interface/TypeAlias/Enum: `docstring\nsignature` (omit empty lines; fall back to `name`).
- File: `{filename} [{classification}] ({rel_dir})\nexports: {sorted_unique_exports}`.
- ExternalPackage / InterfaceField: empty (NOT embedded).

**Synthetic UI docstring** (for exported UI_COMPONENT Functions without JSDoc):
```
{readable_name} UI component. Props: {readable_prop_types}
```
Where `readable_X` = CamelCase split into spaces. Prevents BGE-M3 starvation on sparse signatures.

**Degenerate node exclusion:** nodes where `len(embed_text) < 50` are written to SQLite but NOT embedded into ChromaDB. They remain BFS-reachable.

### 3.3 Skeletonizer (`skeletonizer.py`) — FR-A6

Two-pass tag-and-fold AST reduction. Output goes to `code_nodes.internal_logic_abstraction`. Only Function and Method nodes get a non-NULL value.

**Pass 1:** tag all nodes of type `call_expression`, `return_statement`, `throw_statement`, `import_declaration` AND every ancestor of those nodes as `DO_NOT_ERASE`.

**Pass 2:** emit source bytes per node. `DO_NOT_ERASE` nodes recurse; untagged nodes apply fold rules (first match wins):

| Node type | Condition | Replacement |
|-----------|-----------|-------------|
| `jsx_element`, `jsx_self_closing_element` | not tagged | `/* [JSX: N elements] */` |
| `array` | length > 3 AND not a hook dep array | `/* [array: N items] */` |
| `object` | length > 4 AND not tagged | `/* [object: N props] */` |
| `if_statement`, `switch_statement` | no high-signal child | `/* [logic block] */` |
| `template_string` | length > 100 | `` `/* [template: N chars] */` `` |
| `string` | length > 80 | `"/* [string: N chars] */"` |
| `comment` | always | (deleted) |
| `import_declaration`, `call_expression` | always | (verbatim) |

Hook dep arrays are detected by being the last argument to `useEffect`/`useCallback`/`useMemo`/`useLayoutEffect`.

### 3.4 AST Indexer Pass 2 (`code_indexer.py`) — FR-A4

Emit 13 edge types. Pass 2 runs after Pass 1 has populated `code_nodes` for ALL files in the work set, so cross-file resolution works.

**Builtin call blacklist** (skip CALLS with these as the root identifier):
```
console, Object, Array, Math, JSON, Promise,
setTimeout, setInterval, clearTimeout, clearInterval,
parseInt, parseFloat, String, Number, Boolean,
Error, Date, RegExp, Map, Set, WeakMap, WeakSet,
Symbol, Proxy, Reflect, Intl,
fetch, URL, URLSearchParams, FormData,
Headers, Request, Response,
Buffer, process, require,
window, document, globalThis
```

**Primitive type blacklist** (skip TYPED_BY targets):
```
string, number, boolean, void, any, unknown,
null, undefined, never, object, symbol, bigint
```

**Edge extraction order per file:**
1. Build import_map from `import_declaration` nodes.
2. Emit `IMPORTS` (relative) and `DEPENDS_ON_EXTERNAL` (non-relative). Populate `file_dependencies`.
3. Per Class: `INHERITS`, `IMPLEMENTS`, `DEFINES_METHOD`.
4. Per Function/Method body: `CALLS`, `TYPED_BY`, `FIELDS_ACCESSED`, `RENDERS`, `PASSES_CALLBACK`, `HOOK_DEPENDS_ON`, `DYNAMIC_IMPORT`, `CLIENT_API_CALLS`.

**Edge propagation (BFS direction + depth):**
| Edge | Dir | Depth |
|------|-----|-------|
| CALLS | reverse | 3 (capped to 1 for low-conf seeds) |
| INHERITS, IMPLEMENTS, TYPED_BY | reverse | 3 |
| FIELDS_ACCESSED | reverse | 2 |
| DEFINES_METHOD | forward | 3 |
| PASSES_CALLBACK | forward | 1 |
| HOOK_DEPENDS_ON | reverse | 1 |
| IMPORTS, RENDERS, DEPENDS_ON_EXTERNAL, CLIENT_API_CALLS, DYNAMIC_IMPORT | reverse | 1 |

These values are already in `shared/constants.py::EDGE_CONFIG`. Use that constant.

**LLM #4 exemption:** nodes reached at depth 1 via `IMPLEMENTS`, `DEFINES_METHOD`, or `TYPED_BY` skip propagation validation (already in `PROPAGATION_VALIDATION_EXEMPT_EDGES`).

**Edge-specific extraction notes:**
- **CALLS:** resolve callee via import_map first, then same-file scope. Skip if root in builtin blacklist or unresolvable.
- **CLIENT_API_CALLS:** scan `fetch()` and template-string args for `^/api/...` patterns; resolve to App Router file using `${...}` → `[id]` and `:param` → `[id]` normalization, then probe `src/app/api/<path>/route.ts` first, then `src/pages/api/<path>.ts`.
- **DYNAMIC_IMPORT:** match `dynamic(() => import('./X'))` and `React.lazy(() => import('./X'))`.
- **HOOK_DEPENDS_ON:** identifiers in the dep array literal of the four hooks.
- **PASSES_CALLBACK:** JSX attributes matching `^on[A-Z]` whose value is a resolvable function reference. Source = parent component, Target = referenced function.
- **FIELDS_ACCESSED:** `member_expression` where the object's annotated type is a known Interface and the accessed property exists as an `InterfaceField` node.

**Insertion:** use `INSERT OR IGNORE INTO structural_edges` to handle duplicate (source, target, edge_type) triples.

### 3.5 Embedder (`embedder.py`) — FR-A5

```python
from FlagEmbedding import BGEM3FlagModel

class Embedder:
    def __init__(self, model_name, batch_size=32, max_length=512):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self.batch_size = batch_size
        self.max_length = max_length

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        out = self.model.encode(
            texts, batch_size=self.batch_size, max_length=self.max_length,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        return out["dense_vecs"]   # (N, 1024)

    def embed_single(self, text: str) -> list[float]:
        return self.embed_batch([text])[0].tolist()
```

Pre-warm via `huggingface_hub.snapshot_download` on first construction.

### 3.6 Reranker (`reranker.py`) — FR-C3

```python
from FlagEmbedding import FlagReranker

class Reranker:
    def __init__(self, model_name):
        self.model = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        if not candidates: return []
        pairs = [(query, c.text_snippet) for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        for c, s in zip(candidates, scores):
            c.reranker_score = float(s)
        return sorted(candidates, key=lambda c: c.reranker_score, reverse=True)[:top_k]
```

### 3.7 Traceability Precompute (`traceability.py`) — FR-A7

```
1. L2-normalize all code and doc vectors.
2. cos_matrix = code_matrix @ doc_matrix.T   (single matmul)
3. For each code_id i:
     for each doc_id j:
       adjusted = cos_matrix[i,j] * layer_compat(code_classification, chunk_type)
     keep top-K (default 5) where adjusted >= 0.60
4. INSERT OR REPLACE INTO doc_code_candidates(code_id, doc_id, weighted_similarity_score).
```

`layer_compat` is in `shared/constants.py`. Skip code nodes with empty embed_text (degenerate).

### 3.8 Indexer Runner (`indexer/runner.py`)

Orchestrates the offline pipeline:

```
0. Connect SQLite, init_schema. Init ChromaDB collections.
1. Scan repo for .md and .ts/.tsx files.
2. Compute SHA-256 of each file. Compare with file_hashes table.
   work_set = files where hash differs OR file is new.
   If --force: work_set = all files; truncate file_hashes.
3. For deleted files (in file_hashes but not in scan):
     DELETE rows from code_nodes WHERE node_id LIKE '<file>%'
     DELETE rows from structural_edges referencing those node_ids
     DELETE rows from doc_code_candidates referencing those code_ids
     DELETE corresponding ChromaDB IDs
     DELETE file_hashes row, file_dependencies rows.
4. For each work_set Markdown file: chunk_markdown → ChromaDB doc_chunks insert.
5. Pass 1 over each work_set .ts/.tsx: extract_nodes → INSERT OR REPLACE into code_nodes.
6. Pass 2 over each work_set .ts/.tsx: extract_edges → INSERT OR IGNORE into structural_edges.
   Also populate file_dependencies on each IMPORTS edge.
7. Embed all pending non-degenerate code_nodes via Embedder.embed_batch (batch=32).
   Embed all pending doc_chunks. Insert into ChromaDB code_units / doc_chunks.
8. Recompute traceability table (full recompute is correct; layer-weighted scores
   depend on the population).
9. Update file_hashes for every successfully processed file.
10. Update index_metadata: edge_schema_version='4.0', skeletonization_enabled='true',
    incremental_indexing_enabled='true', total_*, indexing_timestamp, embedding_model_name,
    traceability_k_parameter, min_traceability_similarity.
11. Print stats: code_nodes, doc_chunks, edges, files_scanned, files_reindexed, elapsed.
```

Real target: `C:\Users\Haidar\Documents\thesis\citrakara`. Documents: `C:\Users\Haidar\Documents\thesis\citrakara\docs`.

---

## 4. Online Pipeline (FR-B1 through FR-E3)

Nine numbered steps. Variant flags toggle each. Implementation lives in `pipeline/runner.py`.

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
```

`build_graph_from_sqlite(conn)` materializes a `MultiDiGraph` once per analysis session. `build_bm25_from_chroma(collection)` rebuilds BM25 from ChromaDB documents on every load (~0.5s for typical repo).

### Step 1 — Interpret CR (LLM #1, FR-B1, FR-B2)

Already implemented in `pipeline/interpreter.py`. Returns `CRInterpretation`. If `is_actionable=False`, runner returns a rejection ImpactReport immediately and exits.

### Step 2 — Adaptive RRF Hybrid Search (FR-C1, FR-C2)

Four ranked lists per query:
1. Dense doc — `doc_col.query(query_embeddings=[qvec], n_results=top_k_per_query, where={"chunk_type": {"$in": doc_filter}})` if `doc_filter` non-empty.
2. BM25 doc — `doc_bm25.get_scores(tokens)` → top-K by score.
3. Dense code — only if `"code"` in `cr_interp.affected_layers`.
4. BM25 code — only if `"code"` in affected_layers.

`doc_filter` derives from `cr_interp.affected_layers`:
- `"requirement"` → `["FR", "NFR"]`
- `"design"` → `["Design"]`
- `"code"` → `[]` (empty filter → search code collection)

**Adaptive RRF formula:**
```
ARRF(d) = Σ_{p ∈ paths_present} W[change_type][p] / (rrf_k + rank_p(d) + 1)
```
where `W = RRF_PATH_WEIGHTS[change_type]` from `shared/constants.py`. `rrf_k = 60`.

After fusion, sort all unique node IDs by ARRF score descending, take top `max_candidates_post_rrf` (=15). Build `Candidate` DTOs by joining against ChromaDB metadata + SQLite for `internal_logic_abstraction`, `file_path`, `file_classification`, etc.

**Variant degenerations:**
- V0: only path 2 (BM25 doc) and 4 (BM25 code) lists; no fusion needed (single ranking) — but for code, still emit BM25 code; if both BM25 lists present, RRF still applies but only across two lists.
- V1: only path 1 + 3 (dense paths).
- V2 onward: all four paths active when `code` is in affected_layers.

In practice the safest implementation: always assemble the ranked lists according to flags (skip lists for disabled retrieval modes) and run the same RRF fusion code. With one list, RRF just reduces to that list's order.

### Step 3 — Cross-Encoder Rerank (FR-C3)

Skip entirely if `enable_cross_encoder=False`.
```python
candidates = ctx.reranker.rerank(cr_interp.primary_intent, candidates, settings.max_candidates_post_rerank)
```

### Step 3.5 — Score Floor (FR-C4 part 1)

Always active.
```python
candidates = [c for c in candidates if c.reranker_score >= settings.min_reranker_score_for_validation]
# Note: when reranker is disabled, reranker_score is 0.0 → all dropped.
# Therefore: when enable_cross_encoder=False, also skip Step 3.5.
```

When `enable_cross_encoder=False`, skip 3.5.

### Step 3.6 — Semantic Dedup (FR-C4 part 2)

Skip if `enable_dedup_gate=False`. Otherwise: for each `doc_chunks` candidate, look up top-1 code resolution from `doc_code_candidates`; if that code_id is already in the candidate list, append the doc_id to `merged_doc_ids` on the matching code candidate and drop the doc candidate.

### Step 3.7 — Plausibility Gate + Affinity (FR-C4 part 3)

Skip if `enable_plausibility_gate=False`. Otherwise:

**Phase A (rescore):** multiply each candidate's `reranker_score` by `_affinity_factor(c, cr_interp)`:
- For doc candidates: 1.0 if `chunk_type ∈ layer_to_chunk_types(affected_layers)` else 0.7.
- For code candidates: `layer_compat(file_classification, primary_chunk_type)` where `primary_chunk_type = "FR"` if "code" or "requirement" in layers, "Design" if "design", else "General".
Re-sort descending.

**Phase B (file-density gate):** total = `len(candidates)`. file_density = `Counter(c.file_path)`. For each candidate (in sorted order):
- If `c.name.lower()` substring-matches (either direction) any pattern in `cr_interp.named_entry_points`: ADMIT.
- Else if `file_density[c.file_path] / total > settings.plausibility_gate_density_threshold` AND admitted-from-this-file count >= `settings.plausibility_gate_max_per_file`: DROP.
- Else: ADMIT.

### Step 4 — Validate SIS (LLM #2, FR-C5)

`pipeline/validator.py`:
Skip if `enable_sis_validation=False`. Otherwise:

**Lost-in-the-middle reorder:** position 0 = highest reranker_score, position N-1 = lowest, middle filled in ascending order.

**Prompt body** (`pipeline/validator.py` builds it):
```
Change Request Intent: {primary_intent}
Change Type: {change_type}
Domain Concepts: {", ".join(domain_concepts)}

OUT-OF-SCOPE OPERATIONS — these share vocabulary with the CR but are NOT being changed.
Do NOT confirm any candidate that primarily serves one of these:
  - {op_1}
  - {op_2}

NAMED ENTRY POINTS — the CR explicitly describes these:
  - {pat_1}

Evaluate each candidate. Confirm ONLY if directly relevant. Reject topically related but
functionally unaffected candidates. Pay attention to File path: a function in an unrelated
service module is almost never directly impacted by a CR targeting a different feature.

[1] ID: {node_id}
Type: {node_type}
File: {file_path}
Reranker score: {reranker_score:.3f}
Snippet:
{internal_logic_abstraction OR text_snippet, truncated to ~1500 chars}

[2] ...
```

Call:
```python
result: SISValidationResult = ctx.llm_client.call(
    system="You are a software impact analysis expert...",
    user=prompt,
    response_schema=SISValidationResult,
    call_name="validate_sis",
)
sis_ids = [v.node_id for v in result.verdicts if v.confirmed]
```

When variant disables LLM #2: `sis_ids = [c.node_id for c in candidates]`.

### Step 5 — Resolve doc-chunk SIS to code seeds (FR-C6)

`pipeline/seed_resolver.py`:
- Direct code-node SIS entries pass through as `direct_code_seeds`.
- Doc-chunk SIS entries: query `doc_code_candidates` for top-K code nodes per doc (K = `top_k_traceability`).
- Return `(resolutions, doc_to_code_map, direct_code_seeds)` where `resolutions = [{doc_id, code_ids}]` and `doc_to_code_map = {doc_id: [code_ids]}`.

```python
def resolve_doc_to_code(sis_ids, conn, top_k):
    code_node_set = {r[0] for r in conn.execute("SELECT node_id FROM code_nodes").fetchall()}
    direct_code_seeds = []
    resolutions = []          # [{"doc_id": x, "code_ids": [...]}]
    doc_to_code_map = {}      # for synthesis backlinks
    for nid in sis_ids:
        if nid in code_node_set:
            direct_code_seeds.append(nid)
            continue
        rows = conn.execute(
            "SELECT code_id FROM doc_code_candidates WHERE doc_id = ? "
            "ORDER BY weighted_similarity_score DESC LIMIT ?",
            (nid, top_k),
        ).fetchall()
        if rows:
            code_ids = [r[0] for r in rows]
            resolutions.append({"doc_id": nid, "code_ids": code_ids})
            doc_to_code_map[nid] = code_ids
    return resolutions, doc_to_code_map, direct_code_seeds
```

### Step 5b — Validate trace resolution (LLM #3, FR-C7)

`pipeline/traceability_validator.py`:
- Return `(validated_code_seeds, low_confidence_map)`.
- `call_name="validate_trace"`.

Skip if `enable_trace_validation=False` → take top-1 of each resolution as code seed (blind resolution).

Otherwise: build prompt presenting each (doc_chunk, candidate code_node) pair with the doc text, the code signature + internal_logic_abstraction. LLM returns `TraceValidationResult` with three-way decision per pair measuring their relevancy to the doc_chunk and the CRInterpretation object. CONFIRMED → seed admitted. PARTIAL → seed admitted with `low_confidence_seed=True`. REJECTED → dropped.

### Step 6 — BFS propagation (FR-D1)

`pipeline/graph_bfs.py`:
1. Build NetworkX `MultiDiGraph` from `structural_edges`.
2. Compute high-confidence tier: top-N seeds by reranker score (N = `settings.bfs_high_conf_top_n`, default 5). Doc-chunk reranker scores propagate to their resolved code seeds via `dict.setdefault`.
3. Multi-source BFS. Per edge type, consult `EDGE_CONFIG[edge_type]` for direction (forward/reverse) and max_depth.
4. **Fix D:** for low-confidence origins (seed marked `low_confidence_seed=True` OR seed not in high-confidence set), CALLS depth is capped to 1 instead of 3 (`LOW_CONF_CAPPED_EDGES = {CALLS}`).
5. Global cap: `settings.bfs_global_max_depth` (default 3).
6. Output `CISResult` separating `sis_nodes` (the seeds themselves) from `propagated_nodes`. Each `NodeTrace` records `depth, causal_chain, path, source_seed, low_confidence_seed`.

Skip if `enable_bfs=False` → CIS = SIS only.

Otherwise:
```python
all_code_seeds = list(dict.fromkeys(direct_code_seeds + validated_code_seeds))
sis_reranker_map = {c.node_id: c.reranker_score for c in admitted_candidates if c.node_id in set(all_code_seeds)}
for r in resolutions:
    for cid in r["code_ids"]:
        sis_reranker_map.setdefault(cid, 0.0)

high_conf = compute_confidence_tiers(all_code_seeds, sis_reranker_map, settings.bfs_high_conf_top_n)
cis = bfs_propagate(ctx.graph, all_code_seeds, high_confidence=high_conf, low_confidence_seed_map=low_conf)
```

`compute_confidence_tiers`: top-N by reranker score → frozenset of node IDs.

`bfs_propagate`: per-edge-type direction and depth from `EDGE_CONFIG`. CALLS depth capped to 1 for low-confidence origins. Invariant: `len(sis_nodes) + len(propagated_nodes) == len(visited)`.

### Step 7 — Validate propagation (LLM #4, FR-D2)

`pipeline/traversal_validator.py`:
Skip if `enable_propagation_validation=False`.

Otherwise: partition `cis.propagated_nodes` into:
- `auto_kept`: nodes at depth=1 reached via an edge in `PROPAGATION_VALIDATION_EXEMPT_EDGES`.
- `to_validate`: the rest, send the code signature + internal_logic_abstraction.

Send `to_validate` to LLM #4 with their causal chains, the originating SIS seed (signature + internal_logic_abstraction),a dn the CRIterpretation object. Keep only those with `semantically_impacted=True`, plus all `auto_kept`, based on their relevancy to their originating SIS seed and the CRInterpretation object.

### Step 8 — Backlinks + Context (FR-E1, FR-E2)

`pipeline/context_builder.py`:
```python
backlinks = fetch_backlinks(cis.all_node_ids(), conn, top_k_backlinks_per_node)
snippets = fetch_snippets(cis.all_node_ids(), conn)  # source_code from SQLite
context = build_context(cr_text, cr_interp, cis, backlinks, snippets, settings)
```

Context includes: raw CR, CRInterpretation JSON, SIS list with reranker scores, CIS with BFS path + chain per node, depth, top-3 backlinks per code node, precomputed severity per node.

**Token budget enforcement:** if `estimate_tokens(context) > llm_max_context_tokens - synthesis_system_prompt_tokens - output_reserve_tokens`, drop nodes in this priority order:
1. Lowest severity first (Rendah > Menengah > Tinggi).
2. Within same severity: deepest depth first.
3. Within same depth: alphabetic node_id.

Inject a one-line truncation note into the context for the synthesizer to surface.

### Step 9 — Synthesize (LLM #5, FR-E3)

Already implemented in `pipeline/synthesizer.py`. Severity is precomputed per node via `severity_for_chain`; the system prompt instructs the LLM not to override it.

Output: `ImpactReport` written to `--output` path.

---

## 5. Ablation Harness (V0–V7)

**Variant Matrix (V0..V7):**

Already encoded in `evaluation/variant_flags.py`. Summary (✓ = enabled):

| Variant | BM25 | Dense | RRF | CrossEnc | Dedup | Plaus | LLM#2 | LLM#3 | BFS | LLM#4 |
|---------|------|-------|-----|----------|-------|-------|-------|-------|-----|-------|
| V0 | ✓ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| V1 | ☐ | ✓ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| V2 | ✓ | ✓ | ✓ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| V3 | ✓ | ✓ | ✓ | ✓ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| V4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ☐ | ☐ | ☐ |
| V5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ☐ | ☐ |
| V6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ☐ |
| V7 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

LLM #1 (interpret) and LLM #5 (synthesize) run for every variant.

`VariantFlags` factories already exist in `evaluation/variant_flags.py`. The harness:

```
For each (cr_id, variant_id) in dataset × ALL_VARIANTS:
  flags = VariantFlags.for_id(variant_id)
  start = time.perf_counter()
  try:
    report, ranked_cis = run_analysis_with_ranking(cr_text, settings, flags)
    metrics = compute_metrics(ranked_cis.ordered_ids, gt_node_ids, k_values=[5,10])
  except Exception as e:
    record error row
  Write {output_dir}/{cr_id}/{variant_id}.json
  Append metrics row to per_cr_per_variant_metrics.csv
```

**CIS ordering for metrics (per variant):**
| Variant | Sort key |
|---------|----------|
| V0 | bm25_score desc |
| V1 | cosine_score desc |
| V2 | rrf_score desc |
| V3, V4, V5 | reranker_score desc |
| V6, V7 | depth asc, then discovery order |

Tie-break: alphabetic node_id.

**Locked parameters:** at start of `run_full_evaluation`, write `locked_parameters.json` with current settings (model, temperature, seed, rrf_k, all gate thresholds, top_n_high_conf). If any of these change mid-run, raise.

**Additive property assertions** (run after evaluation):
- For every CR: `|CIS(V7)| <= |CIS(V6)|` (LLM #4 only rejects)
- `|CIS(V6)| >= |CIS(V5)|` (BFS only adds)
- `|SIS(V5)| <= |SIS(V4)|` (LLM #3 only rejects)

---

## 6. Evaluation (V7 vs V5 only)

Precision@10, Recall@10, F1@10. K=5 reported descriptively. 

```
P@K = |ranked[:K] ∩ gt| / K
R@K = |ranked[:K] ∩ gt| / |gt|
F1@K = 2PR/(P+R) or 0 if P+R==0

Edge cases: empty ranked → all 0. `|ranked| < K` → denominator is K (penalize under-retrieval).

`evaluation/statistical.py::run_primary_test(df)` — single one-sided paired Wilcoxon on F1@10. No Bonferroni. Output dict: `hypothesis, variant_a, variant_b, p_value, statistic, cliffs_delta, median_diff, n, accepted`.

`build_summary_artifacts` writes:
- `per_cr_per_variant_metrics.csv` — one row per (cr_id, variant_id) with all metric columns.
- `<cr_id>/<variant>.json` — per-variant ImpactReport.
- `summary_table.csv` — macro-averaged metrics per variant (all 8 variants present, descriptive).
- `per_category_table.csv` — metrics per variant per CR category (C1..C5).
- `statistical_tests.json` — single dict (V7 vs V5).
- `locked_parameters.json` — pre-run snapshot of `Settings`.
- `latency_distribution.png` — box plot.
- `llm_audit.jsonl` — append-only line per LLM call, including `config_hash` (NFR-05).
- (with `--verify-nfr`) `nfr_verification.json` — pass/fail for NFR-01 through NFR-05.

**NFR verification procedures** (in `evaluation/nfr_verify.py`):
- NFR-01: run V7 twice, compare validated SIS code node sets after Step 5b. Use a `ContextRecorder` snapshotting hook.
- NFR-02: subprocess `impactracer index` with network monitoring; non-LLM stages must work offline (HF cache pre-warmed).
- NFR-03: latency distribution across all 20 test CRs at V7. Median + p95.
- NFR-04: a designated Indonesian CR retrieves at least one English-identifier GT code node into the validated SIS.
- NFR-05: parse `llm_audit.jsonl`; assert all entries share `config_hash`.

---

## 7. Configuration

`shared/config.py` already defines `Settings`. Important defaults (from blueprint 11):
- `llm_model="google/gemini-2.5-flash"`, `llm_temperature=0.0`, `llm_seed=42`
- `llm_retry_max_attempts=10`, `llm_retry_base_backoff=2.0`
- `min_reranker_score_for_validation=0.01`
- `plausibility_gate_density_threshold=0.35`, `plausibility_gate_max_per_file=2`
- `bfs_high_conf_top_n=5`, `bfs_global_max_depth=3`
- `top_k_traceability=5`, `min_traceability_similarity=0.60`, `degenerate_embed_min_length=50`

**OpenRouter is the exclusive LLM transport.** All models (including Gemini, DeepSeek, etc.) are accessed via the OpenRouter API. `pipeline/llm_client.py` implements a single `httpx`-based POST client. Pydantic `response_schema` is the contract; `session_config_hash` is a digest of model + temperature + seed.

---

## 8. CLI Surface

Three commands in `cli.py`:
- `impactracer index <repo_path> [--force]`
- `impactracer analyze "<cr_text>" [--output PATH] [--variant V0..V7]`
- `impactracer evaluate --dataset <gt.json> --output <dir> [--run-full-ablation] [--verify-nfr]`

Tested by running against:
- `C:\Users\Haidar\Documents\thesis\citrakara` (target repo)
- `C:\Users\Haidar\Documents\thesis\citrakara\docs` (target docs)
- `C:\Users\Haidar\Documents\thesis\impactracer-app` (this repo)

---

## 9. Sprint Plan (12 Sessions)

| Sprint | Deliverable | Modules |
|--------|-------------|---------|
| 1 | Foundation verification | `tests/unit/test_models.py` |
| 2 | Persistence verification | `tests/unit/test_persistence.py` |
| 3 | Document indexer | `indexer/doc_indexer.py` + tests |
| 4 | AST Pass 1 + skeletonizer | `indexer/code_indexer.py` (Pass 1), `indexer/skeletonizer.py` |
| 5 | AST Pass 2 (13 edge types) | `indexer/code_indexer.py` (Pass 2) |
| 6 | Embedder + reranker + traceability | `indexer/{embedder,reranker,traceability}.py` |
| 7 | Indexer runner + `index` CLI | `indexer/runner.py`, `cli.py::index` |
| 8 | Online minimal (LLM #1, retriever, RRF, context, LLM #5) | `pipeline/{retriever,context_builder,runner}.py`, `cli.py::analyze` |
| 9 | Pre-validation gates + LLM #2 | `pipeline/{prevalidation_filter,validator}.py` |
| 10 | LLM #3, BFS, LLM #4 | `pipeline/{seed_resolver,traceability_validator,graph_bfs,traversal_validator}.py` |
| 11 | Ablation + metrics + Wilcoxon | `evaluation/{ablation,metrics,statistical,report_builder}.py`, `cli.py::evaluate` |
| 12 | NFR verification + release smoke | `evaluation/nfr_verify.py`, `evaluation/annotator_tool.py` |
