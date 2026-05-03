# ImpacTracer v4.0 — Offline Indexer: Implementation Reference

> Source of truth for the offline indexing phase (FR-A1 through FR-A7 + runner).
> All numbers reflect the live index over `citrakara` as of Sprint 7.9.

---

## 1. Architecture Overview

The offline indexer builds a hybrid knowledge base from a target TypeScript repository and its Markdown documentation. It runs in a single CLI command (`impactracer index <repo_path>`) and produces three persistent stores:

| Store | Technology | Contents |
|---|---|---|
| SQLite (`data/impactracer.db`) | `sqlite_client.py` | Code nodes, structural edges, doc→code candidate pairs, file hashes, file dependencies, index metadata |
| ChromaDB (`data/chroma_store`) | `chroma_client.py` | Dense embedding vectors for non-degenerate code nodes (`code_units`) and doc chunks (`doc_chunks`) |

The runner (`indexer/runner.py`) orchestrates 10 steps: scan → diff → purge → Markdown chunk → AST Pass 1 → AST Pass 2 → embed → traceability → update hashes → metadata.

---

## 2. Document Indexer (FR-A1, FR-A2)

**File:** `indexer/doc_indexer.py`

Chunks Markdown at H2/H3 boundaries using the mistune 3.x AST (`renderer=None`). H1 is not a boundary; H4+ content absorbs into the enclosing chunk. Chunk text is sliced from raw source lines — the AST is used only to enumerate boundary headings. This preserves tables, code blocks, and raw Markdown fidelity.

**Chunk ID:** `{file_stem}__{slugify(section_title)}` — deterministic across runs.

**Classification (`CHUNK_TYPE_RULES`):** NFR → FR → Design → General. Evaluated in this order to avoid "non-functional requirement" being matched by the "functional requirement" FR rule (substring containment).

**Citrakara result:** 104 total chunks (srs.md: 51, sdd.md: 53). 94 embedded (non-degenerate); 10 degenerate (< 50 chars).

---

## 3. AST Node Extraction — Pass 1 (FR-A3, FR-A6)

**File:** `indexer/code_indexer.py` — `extract_nodes()`

Two-pass TypeScript/TSX parser via `tree-sitter-languages`. Pass 1 extracts **9 node types**:

| Node Type | Key Details |
|---|---|
| `File` | Always index 0; classification, route_path, client_directive, exported names |
| `Function` | Named functions and arrow-function `const` declarations |
| `Method` | Class methods; qualified ID: `ClassName.methodName` |
| `Class` | Extracts heritage (extends, implements) |
| `Interface` | With extends clause |
| `InterfaceField` | One per `property_signature`; always degenerate (embed_text = "") |
| `TypeAlias` | Object-shape aliases also produce InterfaceField children |
| `Enum` | Name only |
| `ExternalPackage` | One per unique npm specifier per file (Pass 1); global dedup in runner |

**File classification** (precedence order, first match wins):

1. `src/app/route.{ts,tsx}` → `API_ROUTE`
2. `src/app/page.{tsx,ts}` / `layout.*` → `PAGE_COMPONENT`
3. `src/components/` → `UI_COMPONENT`
4. `src/hooks/` → `UI_COMPONENT` *(delta from blueprint: React hooks are client-side UI logic)*
5. `src/lib/stores/` → `UI_COMPONENT` *(delta: Zustand stores are UI state)*
6. `src/lib/test/`, `__mocks__/`, `__tests__/` → `None` *(must precede models/ rule)*
7. `src/lib/db/models/` → `TYPE_DEFINITION` *(delta: Mongoose schemas classified as data definitions, not UTILITY)*
8. `src/lib/` → `UTILITY`
9. `src/types/` → `TYPE_DEFINITION`
10. Fallback → `None`

**Degenerate rule:** nodes with `len(embed_text) < 50` go to SQLite but NOT ChromaDB. InterfaceField and ExternalPackage are always degenerate. They remain BFS-reachable via CONTAINS edges.

**Skeletonizer (FR-A6):** `indexer/skeletonizer.py`. Two-pass tag-and-fold AST reduction on function bodies. Pass 1 tags HIGH_SIGNAL nodes (call, return, throw, import) and all their ancestors. Pass 2 emits tagged nodes verbatim; untagged nodes are folded by rule: JSX → `/* [JSX: N] */`, large arrays → `/* [array: N] */`, large objects, logic blocks, long strings. Position-tracking cursor preserves inter-token whitespace.

**Citrakara result:** 3,002 total nodes — InterfaceField: 1,452 | Function: 771 | File: 331 | Interface: 313 | ExternalPackage: 92 | TypeAlias: 39 | Method/Class: 2 each.

---

## 4. AST Edge Extraction — Pass 2 (FR-A4)

**File:** `indexer/code_indexer.py` — `extract_edges()`

The blueprint specifies 13 edge types. **We added a 14th: `CONTAINS`** (see §6 Deltas).

All 14 edge types emitted per file:

| Edge Type | Source → Target | Mechanism |
|---|---|---|
| `IMPORTS` | File → File | Relative + `@/` alias imports |
| `DEPENDS_ON_EXTERNAL` | File → ExternalPackage | Non-relative, non-alias imports |
| `CALLS` | Function/Method → Function/Method | Call expressions; resolved via import_map then same-file lookup |
| `INHERITS` | Class → Class | `extends` clause |
| `IMPLEMENTS` | Class → Interface | `implements` clause |
| `DEFINES_METHOD` | Class → Method | Class body walk |
| `TYPED_BY` | Function/Method → Interface/TypeAlias | Parameter + variable type annotations |
| `RENDERS` | Function → Function | JSX uppercase tag names |
| `PASSES_CALLBACK` | Function → Function | JSX `onX={importedFn}` (imported named function only) |
| `HOOK_DEPENDS_ON` | Function → Function/Interface | Hook dep array elements |
| `FIELDS_ACCESSED` | Function → InterfaceField | `obj.field` where obj is in import_map |
| `CLIENT_API_CALLS` | Function → API_ROUTE Function | `fetch('/api/...')` or `axiosClient.post(...)` |
| `DYNAMIC_IMPORT` | File/Function → File | `dynamic(() => import('./X'))`, `React.lazy(...)` |
| `CONTAINS` | File → {Function,Method,Interface,TypeAlias,Class,Enum,InterfaceField} + Interface → InterfaceField | Structural membership |

**Key implementation choices:**

- `@/` path aliases (TypeScript `tsconfig.json` paths) are resolved as intra-repo IMPORTS, not as ExternalPackage nodes. Without this, 744 internal edges were ghost edges terminating at dead-end `ext::@/...` nodes, severing the API_ROUTE → service CALLS chain entirely.
- `CLIENT_API_CALLS` check runs before the builtins guard (`fetch` is a builtin but must still trigger this rule).
- `resolve_api_route` uses positional wildcard matching: disk segment `[paramName]` matches any single URL segment. Selects the most-specific match (fewest wildcards). Without this, routes with non-`[id]` param names produced 0 CLIENT_API_CALLS edges.
- Inline JSX handlers (`onX={() => doFn()}`) trigger `_walk_body` on the arrow function body, emitting transitive CALLS to imported functions called within — preserving reachability even when PASSES_CALLBACK = 0.
- `_emit_middleware_edges`: parses `export const config = { matcher: [...] }` in `middleware.ts` and emits CALLS edges from the middleware function to all matched API_ROUTE nodes.
- `_emit_mongoose_edges`: emits TYPED_BY edges from model files to Interface nodes via `model<IFoo>()` and TYPED_BY from repositories via `ref: 'ModelName'` string literals.

**Citrakara result:** 8,004 total edges — CONTAINS: 4,031 | DEPENDS_ON_EXTERNAL: 847 | CALLS: 1,380 | TYPED_BY: 532 | IMPORTS: 926 | RENDERS: 169 | CLIENT_API_CALLS: 110 | DYNAMIC_IMPORT: 7 | DEFINES_METHOD: 2.

---

## 5. Embedder + Reranker (FR-A5, FR-C3)

**Files:** `indexer/embedder.py`, `indexer/reranker.py`

- **Model:** `BAAI/bge-m3` via `FlagEmbedding.BGEM3FlagModel`. Output dim: 1024 float32. FP16 on GPU (RTX 3050 active; `torch==2.6.0+cu124`).
- **Degenerate exclusion:** nodes with `len(embed_text) < 50` are not embedded and not sent to ChromaDB. They remain BFS-reachable via CONTAINS edges.
- **FlagEmbedding imports deferred inside `__init__`** — allows monkeypatching in tests without requiring the real GPU package.
- **Reranker:** `FlagReranker` (`BAAI/bge-reranker-v2-m3`). Used in online pipeline Step 3 (FR-C3); not invoked by the offline indexer.

---

## 6. Traceability Precomputation (FR-A7)

**File:** `indexer/traceability.py` — `compute_and_store()`

Layer-weighted cosine similarity between all non-degenerate code vectors and all doc chunk vectors. Full cross-product via single matrix multiply (`N_code × N_doc`).

**LAYER_COMPAT matrix (final calibrated values):**

| Node Class | FR | NFR | Design | General |
|---|---|---|---|---|
| API_ROUTE | 1.0 | 0.6 | 0.9 | 0.8 |
| PAGE_COMPONENT | 1.0 | 0.5 | 0.9 | 0.7 |
| UI_COMPONENT | 0.9 | 0.5 | 0.9 | 0.7 |
| UTILITY | 1.0 | 0.7 | 1.0 | 0.8 |
| TYPE_DEFINITION | 0.8 | 0.3 | **1.0** | 0.6 |
| None | 0.8 | 0.5 | 0.8 | 0.6 |

`TYPE_DEFINITION × Design = 1.0`: Mongoose schema files are the direct artifact of SDD "Perancangan Basis Data" sections — no semantic gap.

**Threshold:** `min_traceability_similarity = 0.40`. Calibrated to BGE-M3 cross-lingual anisotropy: raw cosine scores over this corpus live in [0.24, 0.75] with mean ≈ 0.45. The 0.40 floor captures 100% of meaningful pairs while excluding noise.

**Dual-direction top-K:** Forward pass (per-code-node top-K) + reverse pass (per-doc-chunk top-K). Union prevents NFR/General chunks from being squeeze-out by high-LAYER_COMPAT competitors despite genuine cosine signal.

**Citrakara result:** 5,167 pairs. Score distribution: min=0.40, mean=0.54, max=0.75, std=0.062. 79/94 doc chunks mapped (84%). NFR security chunk covered (1/2 NFR; the second has highest weighted score 0.396 — below floor, correctly excluded).

---

## 7. Incremental Indexing

**File:** `indexer/runner.py`

SHA-256 file hashes stored in `file_hashes` table (absolute posix path as key). On each run:

1. Diff `current_posix` vs `known_hashes` → deleted, changed, and unchanged files.
2. Purge deleted files from SQLite + ChromaDB.
3. Re-extract Pass 1 nodes for changed TS files.
4. **Reverse-dep expansion:** query `file_dependencies` for files that import the changed set — those files also need Pass 2 edge re-extraction (without full node re-extraction).
5. Re-extract Pass 2 edges for changed + reverse-dep files.
6. Embed only changed files' code nodes (not the full set).
7. Traceability: full recompute always (population-dependent scores).

**Path format invariant:**

| Store | Path format |
|---|---|
| `file_hashes.file_path` | Absolute posix (`C:/Users/.../file.ts`) |
| `code_nodes.file_path` | `src/...` relative (`src/lib/services/auth.service.ts`) |
| `file_dependencies.{dependent,target}_file` | `src/...` relative |
| ChromaDB doc `source_file` metadata | Absolute posix |

Root-level TS files with no `src/` ancestor (e.g. `jest.config.ts`) fall back to just the filename.

**Verified runs (citrakara):**

| Run | Trigger | Files reindexed | Elapsed |
|---|---|---|---|
| Cold start `--force` | — | 333 | ~45s (GPU) |
| No-change | identical repo | 0 | ~16s |
| 1-file modified | `wallet.service.ts` | 1 + 2 rev-deps | ~10s |

---

## 8. Deltas from Master Blueprint

| # | Delta | Location | Rationale |
|---|---|---|---|
| D1 | **CONTAINS = 14th edge type** (blueprint specifies 13) | `sqlite_client.py` CHECK, `constants.py` EDGE_CONFIG, `code_indexer.py` | File↔Function membrane was opaque to BFS. Without CONTAINS, BFS from a seed file could not reach its child Functions, and vice versa. Schema change was the only defensible fix. |
| D2 | **`@/` path alias resolution as IMPORTS** | `code_indexer.py::_build_import_map` | TypeScript `@/` maps to `src/` via `tsconfig.json` paths. Treating aliases as ExternalPackage created 744 ghost edges and severed the full API_ROUTE → service CALLS chain. |
| D3 | **CHUNK_TYPE_RULES evaluation order: NFR before FR** | `doc_indexer.py` | "non-functional requirement" contains "functional requirement" as substring. FR evaluated first misclassified NFR titles. |
| D4 | **File classification: `src/hooks/` → UI_COMPONENT** | `code_indexer.py::_make_classifier` | React hooks are client-side UI logic. Classifying as UTILITY misroutes them in LAYER_COMPAT (FR/Design should resolve hooks). |
| D5 | **File classification: `src/lib/stores/` → UI_COMPONENT** | `code_indexer.py::_make_classifier` | Zustand stores manage client-side session state. Not backend logic. |
| D6 | **File classification: `src/lib/db/models/` → TYPE_DEFINITION** | `code_indexer.py::_make_classifier` | Mongoose schema files are the direct subject of SDD DB Design sections. TYPE_DEFINITION×Design=1.0 maximises traceability signal. |
| D7 | **`min_traceability_similarity = 0.40`** (blueprint implied higher) | `shared/config.py` | BGE-M3 cross-lingual cosine distribution peaks at 0.75 with mean ≈ 0.45. Any threshold above 0.50 retains < 2% of all pairs. 0.40 is calibrated to the actual distribution. |
| D8 | **LAYER_COMPAT recalibrated** | `shared/constants.py` | Original matrix assigned UTILITY×General=0.50, suppressing the most traceable business-logic nodes. Recalibrated to UTILITY×FR=1.0, UTILITY×Design=1.0, UTILITY×General=0.8; TYPE_DEFINITION×Design=1.0. |
| D9 | **Dual-direction top-K in traceability** | `indexer/traceability.py` | Forward-only top-K silently stranded NFR/General chunks (15/94 stranded pre-fix). Reverse pass guarantees every chunk with a weighted score ≥ threshold gets at least K candidates. |
| D10 | **Positional wildcard route matching** | `code_indexer.py::resolve_api_route` | Blueprint assumed normalized `[id]` segments. Citrakara uses named params (`[proposalId]`, `[listingId]`). String equality on normalized paths left those routes with 0 CLIENT_API_CALLS. |
| D11 | **InterfaceField CONTAINS edges** | `code_indexer.py::_emit_contains_edges` | InterfaceField nodes had 0 incoming edges, making them BFS dead-ends. File→InterfaceField + Interface→InterfaceField CONTAINS edges added. |
| D12 | **Middleware synthetic CALLS edges** | `code_indexer.py::_emit_middleware_edges` | `middleware.ts` has no static imports to route handlers. Without synthetic edges, BFS from middleware reached only 2 nodes. |
| D13 | **Mongoose TYPED_BY edges** | `code_indexer.py::_emit_mongoose_edges` | `model<IFoo>()` calls and `ref: 'ModelName'` literals establish schema-to-interface relationships invisible in normal TYPED_BY extraction. |

---

## 9. Final Index State (citrakara, post Sprint 7.9)

| Metric | Value |
|---|---|
| code_nodes (total) | 3,002 |
| non-degenerate (embedded) | 1,003 |
| structural_edges | 8,004 |
| doc_chunks (embedded) | 94 |
| doc→code candidate pairs | 5,167 |
| orphan non-degenerate nodes | 23 (2.3%) |
| FK violations | 0 |
| Semantic benchmarks | 4/4 PASS |

**Semantic benchmark top-1:**

| Benchmark | Chunk | Top-1 Node | Score |
|---|---|---|---|
| Auth | `sdd__v_1_…_autentikasi` | `auth.service.ts::loginUser` | 0.6671 |
| Wallet | `sdd__v_17_…_wallet_dan_akun` | `wallet.service.ts` | 0.6902 |
| Dispute | `sdd__iii_12_…_resolution_m06` | `ticket.service.ts::createResolutionTicket` | 0.6893 |
| DB Design | `sdd__iv_2_…_entitas_wallet` | `wallet.repository.ts` | 0.6840 |

**Full-stack BFS confirmed:** `wallet.model.ts` seed at depth=3 reaches 4 PAGE_COMPONENT nodes (`wallet/page.tsx`, `dashboard/page.tsx`, `contracts/tickets/page.tsx`, `wallet/transactions/page.tsx`). Pre-alias-fix: 0 UI/PAGE nodes reachable.

---

## 10. Dual-Granularity Evaluation Strategy

**Schema:** `evaluation/schemas.py` — `GTEntry`, `ImpactedFile`, `ImpactedEntity`.

Each annotated Change Request carries two independent GT sets:

| Level | Field | Unit of measurement | Purpose |
|---|---|---|---|
| **File-level** | `impacted_files` | File path (`src/lib/services/auth.service.ts`) | Baseline routing accuracy — does the system surface the right files? Tolerant of entity-level noise. |
| **Entity-level** | `impacted_entities` | Node ID (`src/lib/services/auth.service.ts::loginUser`) | AST precision — does the system pinpoint the exact function, interface, or method? |

**Superset rule:** every file path referenced in `impacted_entities` must also appear in `impacted_files`. `impacted_files` may additionally contain file-only impacts (configuration files, barrel exports, etc.).

**Rationale for the split:** a system that retrieves the correct file but misses the specific function still provides useful routing information. Conflating both levels into a single GT set creates metric distortion: entity-level false positives (extra functions in the right file) inflate the denominator and deflate Recall unfairly. Separate evaluation surfaces this distinction cleanly.

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

**Metrics (Sprint 11 implementation):** P@K, R@K, F1@K computed **twice per CR per variant** — once against `GTEntry.file_paths()`, once against `GTEntry.entity_node_ids()`. The ranked CIS list is projected to the appropriate granularity before metric computation (file paths extracted from node IDs for the file-level pass). The primary statistical test (V7 vs V5, Wilcoxon on F1@10) runs at **entity-level** — this is the granularity with the most discriminative signal.
