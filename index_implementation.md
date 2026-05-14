# ImpacTracer v4.0 — Offline Indexer: Implementation Reference

> Source of truth for the offline indexing phase (FR-A1 through FR-A7 + runner).
> All numbers reflect the live `citrakara` index. Companion to `master_blueprint.md`
> (the design specification) and `analysis_implementation.md` (the online pipeline).

---

## 1. Architecture Overview

The offline indexer builds a hybrid knowledge base from a target TypeScript repository and its Markdown documentation. It runs in a single CLI command (`impactracer index <repo_path>`) and produces three persistent stores:

| Store | Technology | Contents |
|---|---|---|
| SQLite (`data/impactracer.db`) | `sqlite_client.py` | `code_nodes`, `structural_edges`, `doc_code_candidates`, `file_hashes`, `file_dependencies`, `index_metadata` |
| ChromaDB (`data/chroma_store`) | `chroma_client.py` | Dense embedding vectors for non-degenerate code nodes (`code_units`) and doc chunks (`doc_chunks`) |

The runner (`indexer/runner.py`) orchestrates 10 ordered steps: scan → diff → purge → Markdown chunk → AST Pass 1 → AST Pass 2 → embed → traceability → update hashes → metadata.

---

## 2. Document Indexer (FR-A1, FR-A2)

**File:** `indexer/doc_indexer.py`

Chunks Markdown at H2 and H3 boundaries using the mistune 3.x AST (`renderer=None`). H1 is not a boundary; H4+ content absorbs into the enclosing chunk. Chunk text is sliced from raw source lines — the AST is used only to enumerate boundary headings. This preserves tables, code blocks, and raw Markdown fidelity.

**Chunk ID:** `{file_stem}__{slugify(section_title)}` — deterministic across runs (lowercase, non-alphanumeric → `_`).

**Classification (`CHUNK_TYPE_RULES`):** evaluated NFR → FR → Design → General. The order matters: "non-functional requirement" contains "functional requirement" as a substring, so the FR rule would otherwise misclassify NFR titles.

```python
CHUNK_TYPE_RULES = {
    "NFR":    ["non-fungsional", "non-functional", "kebutuhan non"],
    "FR":     ["kebutuhan fungsional", "functional requirement", "use case"],
    "Design": ["perancangan", "desain", "arsitektur", "design", "architecture"],
}
# General is the fallback when no rule matches.
```

**citrakara result:** 104 total chunks (`srs.md`: 51, `sdd.md`: 53). 94 embedded into ChromaDB; 10 degenerate (length < 50 chars and therefore not embedded).

---

## 3. AST Node Extraction — Pass 1 (FR-A3, FR-A6)

**File:** `indexer/code_indexer.py::extract_nodes`

Two-pass TypeScript/TSX parser via `tree-sitter-languages`. Pass 1 extracts **10 node types**:

| Node Type | Key details |
|---|---|
| `File` | Always index 0; classification, route_path, client_directive, exported names |
| `Class` | `class_declaration`; heritage (extends, implements) captured |
| `Function` | `function_declaration` OR `lexical_declaration → variable_declarator → arrow_function` |
| `Method` | `method_definition` inside a class body; qualified id `ClassName.methodName` |
| `Interface` | `interface_declaration` with optional extends clause |
| `InterfaceField` | One synthetic child per `property_signature` of an `Interface` or object-shape `TypeAlias`; always degenerate (`embed_text=""`) |
| `TypeAlias` | `type_alias_declaration`; object-shape aliases also produce InterfaceField children |
| `Enum` | `enum_declaration`; name only |
| `ExternalPackage` | Synthetic node per unique non-relative, non-`@/` import specifier (Pass 1 emits per-file; runner deduplicates globally) |
| `Variable` | `lexical_declaration → variable_declarator` whose value is `new_expression`, `object`, `array`, or `call_expression` AND whose name passes the canonical-name heuristic |

### 3.1 The `Variable` node type

Captures module-level constants the rest of the AST extractor was previously blind to:
- `const UserSchema = new Schema<IUser>(...)` (Mongoose schemas).
- `const TEMPLATES = [...]` (template arrays, frozen lookup tables).
- `const config = { ... }` (large object literals).
- `const useUserStore = createStore(...)` (factory-built singletons).

Arrow-function lexical declarations remain `Function`; the `Variable` branch handles all other RHS kinds.

**Canonical-name heuristic** (`_variable_name_is_canonical`):
- Accept PascalCase (`UserSchema`).
- Accept SCREAMING_SNAKE_CASE (`TEMPLATES`).
- Accept any name with at least one internal capital (`createUserStore`).
- Reject pure lowercase locals (`tmp`, `i`, `result`).

**embed_text composition** (`_summarize_variable_value`):
- `new Schema({...})`: signature + the schema's field keys (e.g. `"const UserSchema = new Schema<IUser>"` plus `email username roles pinnedCommissions`).
- Object literal: signature + top-level keys.
- Array literal: signature + the keys of up to the first 20 element objects.
- `call_expression`: signature + the called function's name and up to ~200 chars of the call argument list.

These tokens directly match the vocabulary that appears in CR descriptions ("expose the graceDays setting" matches the `graceDays` field token inside `CommissionListingSchema`'s embed_text).

### 3.2 File classification (path glob, first match wins)

| Pattern | Class |
|---|---|
| `src/app/**/route.{ts,tsx}` | `API_ROUTE` |
| `src/app/**/page.{tsx,ts}` / `src/app/**/layout.{tsx,ts}` | `PAGE_COMPONENT` |
| `src/components/**` | `UI_COMPONENT` |
| `src/hooks/**` | `UI_COMPONENT` |
| `src/lib/stores/**` | `UI_COMPONENT` |
| `src/lib/test/**`, `__mocks__/**`, `__tests__/**` | `None` *(must precede the models/ rule)* |
| `src/lib/db/models/**` | `TYPE_DEFINITION` |
| `src/lib/**`, `src/utils/**` | `UTILITY` |
| `src/types/**` | `TYPE_DEFINITION` |
| else | `None` |

**Degenerate-node rule:** nodes with `len(embed_text) < 50` go to SQLite but NOT ChromaDB. `InterfaceField` and `ExternalPackage` are always degenerate. They remain BFS-reachable via `CONTAINS` edges.

### 3.3 Skeletonizer (FR-A6)

`indexer/skeletonizer.py`. Two-pass tag-and-fold AST reduction. Output goes to `code_nodes.internal_logic_abstraction`. Only `Function`, `Method`, and (when the RHS is `arrow_function`) `Variable`-coalesced-into-Function nodes get a non-NULL ILA.

- **Pass 1:** tag `call_expression`, `return_statement`, `throw_statement`, `import_declaration` AND every ancestor of those nodes as DO-NOT-ERASE.
- **Pass 2:** emit source bytes per node. Tagged nodes recurse verbatim. Untagged nodes apply fold rules (first match wins):

| Node | Condition | Replacement |
|---|---|---|
| `jsx_element`, `jsx_self_closing_element` | always | `/* [JSX: N elements] */` |
| `array` | length > 3 AND not a hook dep array | `/* [array: N items] */` |
| `object` | length > 4 AND not tagged | `/* [object: N props] */` |
| `if_statement`, `switch_statement` | no high-signal descendant | `/* [logic block] */` |
| `template_string` | length > 100 | `` `/* [template: N chars] */` `` |
| `string` | length > 80 | `"/* [string: N chars] */"` |
| `comment` | always | (deleted) |
| `import_declaration`, `call_expression` | always | verbatim |

A position-tracking cursor preserves inter-token whitespace so the output stays parseable.

### 3.4 citrakara node breakdown (live)

3,150 total `code_nodes` — distribution:

| Node type | Count |
|---|---:|
| InterfaceField | 1,452 |
| Function | 771 |
| File | 331 |
| Interface | 313 |
| Variable | 148 |
| ExternalPackage | 92 |
| TypeAlias | 39 |
| Method | 2 |
| Class | 2 |

The Variable count (148) is dominated by Mongoose sub-schemas and frozen lookup tables. The Class/Method counts are low because citrakara is a functional/React codebase; the few classes are Mongoose-derived helpers.

---

## 4. AST Edge Extraction — Pass 2 (FR-A4)

**File:** `indexer/code_indexer.py::extract_edges`

Emits **14 edge types**. Pass 2 runs after Pass 1 has populated `code_nodes` for ALL files in the work set, so cross-file resolution works.

| Edge type | Source → Target | Mechanism |
|---|---|---|
| `IMPORTS` | File → File | Relative + `@/` alias imports |
| `DEPENDS_ON_EXTERNAL` | File → ExternalPackage | Non-relative, non-alias imports |
| `CALLS` | Function/Method → Function/Method | Call expressions; resolved via import_map then same-file lookup |
| `INHERITS` | Class → Class | `extends` clause |
| `IMPLEMENTS` | Class → Interface | `implements` clause |
| `DEFINES_METHOD` | Class → Method | Class body walk |
| `TYPED_BY` | Function/Method → Interface/TypeAlias | Parameter + variable type annotations + Mongoose `model<IFoo>()` generics + repository `ref: 'ModelName'` literals |
| `RENDERS` | Function → Function | JSX uppercase tag names |
| `PASSES_CALLBACK` | Function → Function | JSX `onX={importedFn}` (imported named function only) |
| `HOOK_DEPENDS_ON` | Function → Function/Interface | Hook dep array elements (`useEffect`, `useCallback`, `useMemo`, `useLayoutEffect`) |
| `FIELDS_ACCESSED` | Function → InterfaceField | `obj.field` where `obj` is import-map resolvable to an Interface |
| `CLIENT_API_CALLS` | Function → API_ROUTE Function | `fetch('/api/...')` or `axiosClient.<verb>(...)`; positional wildcard route matching |
| `DYNAMIC_IMPORT` | File/Function → File | `dynamic(() => import('./X'))`, `React.lazy(() => import('./X'))` |
| `CONTAINS` | File → {Function, Method, Interface, TypeAlias, Class, Enum, InterfaceField, Variable} **and** Interface → InterfaceField | Structural membership |

### 4.1 Key implementation choices

- **`@/` path aliases as IMPORTS, not ExternalPackage.** TypeScript's `tsconfig.json` paths map `@/` to `src/`. Treating aliases as `ExternalPackage` created hundreds of ghost edges terminating at dead-end `ext::@/...` nodes and severed the API_ROUTE → service CALLS chain.
- **`CLIENT_API_CALLS` precedes the builtins guard.** `fetch` is on the builtin blacklist (it would otherwise be silently skipped) but must still trigger this rule.
- **`resolve_api_route` uses positional wildcard matching.** Disk segment `[paramName]` matches any single URL segment regardless of param name (`[id]`, `[proposalId]`, `[listingId]` all match the same wildcard slot). The most-specific match (fewest wildcards) wins. Without this, routes with non-`[id]` param names produced 0 CLIENT_API_CALLS edges.
- **Inline JSX handlers walk their body.** `onX={() => doFn()}` triggers `_walk_body` on the arrow-function body, emitting transitive CALLS to imported functions called within. Preserves reachability even when PASSES_CALLBACK = 0.
- **Middleware synthetic CALLS.** `middleware.ts` has no static imports to route handlers; `_emit_middleware_edges` parses `export const config = { matcher: [...] }` and emits CALLS from the middleware function to every matched API_ROUTE node.
- **Mongoose TYPED_BY.** `_emit_mongoose_edges` extracts `model<IFoo>()` generic arguments (model → interface) and repository-side `ref: 'ModelName'` literals (repo → schema variable).
- **File → Variable CONTAINS.** Added when Variable became a node_type. Without it, `Variable` nodes are BFS-unreachable.
- **`INSERT OR IGNORE INTO structural_edges`** handles duplicate `(source, target, edge_type)` triples (same edge inferred from two extractors).

### 4.2 Builtin call blacklist (skip CALLS with these as the root identifier)

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

### 4.3 Primitive type blacklist (skip TYPED_BY targets)

```
string, number, boolean, void, any, unknown,
null, undefined, never, object, symbol, bigint
```

### 4.4 citrakara edge breakdown (live)

8,179 total `structural_edges`:

| Edge type | Count |
|---|---:|
| CONTAINS | 4,179 |
| CALLS | 1,382 |
| IMPORTS | 926 |
| DEPENDS_ON_EXTERNAL | 847 |
| TYPED_BY | 532 |
| RENDERS | 194 |
| CLIENT_API_CALLS | 110 |
| DYNAMIC_IMPORT | 7 |
| DEFINES_METHOD | 2 |

`INHERITS`, `IMPLEMENTS`, `FIELDS_ACCESSED`, `PASSES_CALLBACK`, `HOOK_DEPENDS_ON` count 0 on the current citrakara corpus (consistent with a functional/React codebase without class hierarchies and with most callbacks defined inline).

---

## 5. Embedder + Reranker (FR-A5, FR-C3)

**Files:** `indexer/embedder.py`, `indexer/reranker.py`

- **Embedder.** `BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)`. Output dim 1024. Batch size 32, max length 512. Returns dense vectors only (`return_dense=True`; sparse and ColBERT off). FlagEmbedding is imported lazily inside `__init__` so unit tests can monkeypatch the model.
- **Degenerate exclusion.** Nodes with `len(embed_text) < 50` are not embedded and not sent to ChromaDB. They remain BFS-reachable via `CONTAINS` edges.
- **Reranker.** `FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)`. Sigmoid-normalised scores in `[0, 1]`. Used in the **online pipeline Step 3 only**, never invoked by the offline indexer. Supports multi-query MAX aggregation: each candidate is scored against every LLM #1 `search_query` and the max is kept.

GPU acceleration: when CUDA is available, both models run FP16 on the GPU; on CPU-only systems they fall back to FP32 transparently.

---

## 6. Traceability Precomputation (FR-A7)

**File:** `indexer/traceability.py::compute_and_store`

Layer-weighted cosine similarity between all non-degenerate code vectors and all doc-chunk vectors. Full cross-product via a single matrix multiply.

### 6.1 Algorithm

```
1. L2-normalise all code and doc vectors.
2. cos_matrix = code_matrix @ doc_matrix.T          (single matmul, dense float32)
3. For each (code_id, doc_id):
       adjusted = cos_matrix[i,j] × LAYER_COMPAT[code_classification][chunk_type]
4. Forward pass:  per-code-node top-K above min_similarity floor.
5. Reverse pass:  per-doc-chunk top-K above min_similarity floor.
6. Union of (4) and (5) → INSERT OR REPLACE INTO doc_code_candidates(
                                   code_id, doc_id, weighted_similarity_score).
```

The reverse pass is the critical defence against squeeze-out: forward-only top-K silently strands NFR / General chunks whose layer_compat is low (they always lose the per-code-node competition to high-LAYER_COMPAT competitors despite genuine cosine signal). The reverse pass guarantees every doc chunk with a weighted score above the floor gets at least K candidates of its own.

### 6.2 LAYER_COMPAT matrix (calibrated values in `shared/constants.py`)

| Node class | FR | NFR | Design | General |
|---|---:|---:|---:|---:|
| API_ROUTE | 1.0 | 0.6 | 0.9 | 0.8 |
| PAGE_COMPONENT | 1.0 | 0.5 | 0.9 | 0.7 |
| UI_COMPONENT | 0.9 | 0.5 | 0.9 | 0.7 |
| UTILITY | 1.0 | 0.7 | 1.0 | 0.8 |
| TYPE_DEFINITION | 0.8 | 0.3 | **1.0** | 0.6 |
| `None` (uncategorised) | 0.8 | 0.5 | 0.8 | 0.6 |

`TYPE_DEFINITION × Design = 1.0` reflects that Mongoose schema files are the direct artefact of SDD "Perancangan Basis Data" sections — no semantic gap.

### 6.3 Threshold

`min_traceability_similarity = 0.40`. Calibrated to the empirical BGE-M3 cross-lingual cosine distribution over this corpus: raw scores live in `[0.24, 0.75]` with mean ≈ 0.45. The 0.40 floor captures the vast majority of meaningful pairs while excluding noise. A higher threshold (e.g. 0.60) retains less than 2 % of pairs — unusable for the online traceability pool-seeding signal.

### 6.4 citrakara result (live)

5,890 doc → code candidate pairs persisted. Score distribution typical: min = 0.40, mean ≈ 0.54, max ≈ 0.75, std ≈ 0.06. Roughly 84 % of doc chunks (79 of 94 embedded) end up with at least one above-threshold code neighbour.

---

## 7. Incremental Indexing

**File:** `indexer/runner.py`

SHA-256 file hashes stored in the `file_hashes` table (absolute POSIX path as key). On each run:

1. Diff `current_posix_set` vs `known_hashes` → `deleted`, `changed`, `unchanged` partitions.
2. Purge deleted files from SQLite + ChromaDB.
3. Re-extract Pass 1 nodes for changed TS/TSX files.
4. **Reverse-dependency expansion:** query `file_dependencies` for files that import any node in the changed set — those files also need Pass 2 edge re-extraction (without full node re-extraction). Without this, deleting a service file would leave dangling CALLS edges in its callers.
5. Re-extract Pass 2 edges for `changed ∪ reverse_deps`.
6. Embed only changed files' new code nodes (not the full set).
7. Traceability: **full recompute** always (population-dependent scores).
8. Update `file_hashes` and `index_metadata`.

### 7.1 Path format invariant

| Store | Path format |
|---|---|
| `file_hashes.file_path` | Absolute POSIX (`C:/Users/.../file.ts`) |
| `code_nodes.file_path` | `src/...` relative |
| `file_dependencies.{dependent,target}_file` | `src/...` relative |
| ChromaDB doc `source_file` metadata | Absolute POSIX |

Root-level TS files with no `src/` ancestor (e.g. `jest.config.ts`) fall back to just the filename.

### 7.2 Observed timings (citrakara, post Sprint 13-W1 reindex)

| Run | Trigger | Files reindexed | Elapsed |
|---|---|---:|---|
| Cold start `--force` | full reindex | 333 | ~60s (GPU) |
| No-change | identical repo | 0 | ~16s |
| Single-file modified | typical service file | 1 + 2 reverse-deps | ~10s |

The cold-start time is dominated by BGE-M3 embedding of ~1,150 non-degenerate code nodes plus 94 doc chunks. Subsequent runs amortise model load + index walking; the embedding cost is incremental.

---

## 8. Deltas From the Master Blueprint

The blueprint specifies the design contract; the indexer needed targeted operational deltas to make that contract work on a real Next.js/React/Mongoose codebase. Each delta is durable and codified in the relevant module.

| # | Delta | Location | Rationale |
|---|---|---|---|
| D1 | **CONTAINS is the 14th edge type** | `sqlite_client.py` CHECK, `constants.py::EDGE_CONFIG`, `code_indexer.py::_emit_contains_edges` | File ↔ symbol membrane was opaque to BFS. Without CONTAINS, BFS from a seed file could not reach its child symbols and vice versa. Schema change was the only defensible fix. |
| D2 | **`Variable` is the 10th node type** | `models.py::NodeType`, `sqlite_client.py` CHECK, `code_indexer.py::_build_variable_node`, `constants.py::NODE_TYPE_MAX_FAN_IN` | `const FOO = new Schema(...)` / `const TEMPLATES = [...]` / `const config = { ... }` were invisible. Mongoose schemas alone are a primary CR target on this codebase; without Variable, three of every five calibration GT entities in those files were unreachable. |
| D3 | **`@/` path alias resolution as IMPORTS** | `code_indexer.py::_build_import_map` | TypeScript `@/` maps to `src/` via `tsconfig.json`. Treating aliases as ExternalPackage created ~744 ghost edges and severed the full API_ROUTE → service CALLS chain. |
| D4 | **`CHUNK_TYPE_RULES` evaluation order: NFR before FR** | `doc_indexer.py` | "non-functional requirement" contains "functional requirement" as substring. FR evaluated first misclassified NFR titles. |
| D5 | **File classification: `src/hooks/` → UI_COMPONENT** | `code_indexer.py::_make_classifier` | React hooks are client-side UI logic. Classifying as UTILITY would misroute them in LAYER_COMPAT (FR / Design layers should resolve hooks). |
| D6 | **File classification: `src/lib/stores/` → UI_COMPONENT** | `code_indexer.py::_make_classifier` | Zustand stores manage client-side session state. Not backend logic. |
| D7 | **File classification: `src/lib/db/models/` → TYPE_DEFINITION** | `code_indexer.py::_make_classifier` | Mongoose schema files are the direct subject of SDD DB Design sections. `TYPE_DEFINITION × Design = 1.0` maximises traceability signal. |
| D8 | **`min_traceability_similarity = 0.40`** | `shared/config.py` | BGE-M3 cross-lingual cosine distribution peaks at 0.75 with mean ≈ 0.45. Any threshold above 0.50 retains < 2 % of all pairs. 0.40 is calibrated to the actual distribution. |
| D9 | **LAYER_COMPAT recalibrated** | `shared/constants.py` | Initial matrix assigned `UTILITY × General = 0.50`, suppressing the most traceable business-logic nodes. Recalibrated to `UTILITY × FR = 1.0`, `UTILITY × Design = 1.0`, `TYPE_DEFINITION × Design = 1.0`. |
| D10 | **Dual-direction top-K in traceability** | `indexer/traceability.py` | Forward-only top-K silently stranded NFR / General chunks. Reverse pass guarantees every chunk with a weighted score ≥ threshold gets at least K candidates. |
| D11 | **Positional wildcard route matching** | `code_indexer.py::resolve_api_route` | Blueprint assumed normalised `[id]` segments. citrakara uses named params (`[proposalId]`, `[listingId]`). String equality on normalised paths left those routes with 0 `CLIENT_API_CALLS` edges. |
| D12 | **InterfaceField CONTAINS edges** | `code_indexer.py::_emit_contains_edges` | InterfaceField nodes had 0 incoming edges, making them BFS dead-ends. `File → InterfaceField` plus `Interface → InterfaceField` CONTAINS edges added. |
| D13 | **Middleware synthetic CALLS edges** | `code_indexer.py::_emit_middleware_edges` | `middleware.ts` has no static imports to route handlers. Without synthetic edges, BFS from middleware reached only 2 nodes. |
| D14 | **Mongoose TYPED_BY edges** | `code_indexer.py::_emit_mongoose_edges` | `model<IFoo>()` calls and `ref: 'ModelName'` literals establish schema-to-interface relationships invisible in normal TYPED_BY extraction. |

---

## 9. Final Index State (citrakara, live)

| Metric | Value |
|---|---:|
| `code_nodes` (total) | 3,150 |
| non-degenerate (embedded) | ~1,150 |
| `structural_edges` (total) | 8,179 |
| `doc_chunks` (embedded) | 94 |
| `doc_code_candidates` (pairs) | 5,890 |
| FK violations | 0 |
| Edge-schema version | 4.0 |

### 9.1 Semantic benchmark top-1 (sanity)

| Benchmark | Doc chunk | Top-1 code node | Score |
|---|---|---|---|
| Auth | `sdd__v_1_…_autentikasi` | `auth.service.ts::loginUser` | ~0.67 |
| Wallet | `sdd__v_17_…_wallet_dan_akun` | `wallet.service.ts` | ~0.69 |
| Dispute | `sdd__iii_12_…_resolution_m06` | `ticket.service.ts::createResolutionTicket` | ~0.69 |
| DB Design | `sdd__iv_2_…_entitas_wallet` | `wallet.repository.ts` | ~0.68 |

### 9.2 Full-stack BFS sanity

A `wallet.model.ts` seed at depth 3 reaches four PAGE_COMPONENT nodes (`wallet/page.tsx`, `dashboard/page.tsx`, `contracts/tickets/page.tsx`, `wallet/transactions/page.tsx`) via the IMPORTS / CONTAINS / RENDERS chain. The full-stack route from model to page is intact.

---

## 10. Dual-Granularity Evaluation Strategy

**Schema:** `evaluation/schemas.py` — `GTEntry`, `ImpactedFile`, `ImpactedEntity`.

Each annotated Change Request carries two independent GT sets:

| Level | Field | Unit | Purpose |
|---|---|---|---|
| **File-level** | `impacted_files` | File path (`src/lib/services/auth.service.ts`) | Baseline routing accuracy — does the system surface the right files? Tolerant of entity-level noise. |
| **Entity-level** | `impacted_entities` | Node id (`src/lib/services/auth.service.ts::loginUser`) | AST precision — does the system pinpoint the exact function, interface, schema, or method? |

**Superset rule:** every file path referenced in `impacted_entities` MUST also appear in `impacted_files`. `impacted_files` may additionally contain file-only impacts (configuration files, barrel exports, etc.).

**GT JSON format (locked):**

```json
{
  "cr_id": "CR-01",
  "cr_description": "...",
  "impacted_files": [
    { "file_path": "src/components/auth/LoginForm.tsx", "justification": "..." }
  ],
  "impacted_entities": [
    { "node": "src/components/auth/LoginForm.tsx::LoginForm", "justification": "..." }
  ]
}
```

`GTEntry.file_paths()` returns the set of `file_path` strings; `GTEntry.entity_node_ids()` returns the set of `node` ids. These two sets feed `compute_dual_granularity_metrics` in the ablation harness.

**Rationale for the split:** a system that retrieves the correct file but misses the specific entity still provides useful routing information. Conflating both levels into a single GT set distorts the metric — entity-level false positives (extra functions in the right file) inflate the denominator and deflate recall unfairly. Separate evaluation surfaces this distinction cleanly.

**Set-level only:** the primary metric is `f1_set` against the full unpruned predicted set, computed twice per CR per variant (entity-level and file-level). No `F1@K` exists anywhere in the codebase — bounded top-K metrics cannot distinguish a graph-flood result from a focused one and are therefore forbidden. The pre-registered Wilcoxon test (V7 vs V5) targets the **entity-level `f1_set`**.

---

*End of index_implementation.md. Online pipeline detail in `analysis_implementation.md`. Sprint history in `implementation_report.md`.*
