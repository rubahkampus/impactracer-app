# Phase.Step renumbering map — SIGN-OFF DRAFT

Status: **PROPOSAL — no edits applied yet.** This is the frozen mapping to review
before any code/doc change. Two independent pipelines, each grouped into named
phases with contiguous steps. Parallel propagation arms carry an `a`/`b` suffix.
Function names lose embedded numbers (numbers live in docstrings/logs/docs only).

Legend: ⭐ = canonical LLM stage (ablation-relevant).  ‖ = runs as a parallel arm.

---

## ONLINE pipeline (`impactracer/pipeline/`)

### Phase 1 · Interpret
| New | Old | What | Notes |
|-----|-----|------|-------|
| **1.1** | Step 1 (interpret_intent) | LLM #1 intent | ⭐ |
| **1.2** | Step 1 (interpret_anchors) | LLM #1 skeleton-grounded anchors | ⭐ (2-stage default) |

### Phase 2 · Retrieve
| New | Old | What | Notes |
|-----|-----|------|-------|
| **2.1** | Step 2 | RRF hybrid search (unweighted) | deterministic |
| **2.2** | Step 3.6 | semantic dedup | **runs BEFORE rerank** — new # fixes the old "3.6 after 3" lie |
| **2.3** | Step 3 | cross-encoder rerank | deterministic; ablation V2→V3 |
| **2.4** | (top-K cut, currently unlabeled / part of Step 3) | top-K pool cut | deterministic |

> Retired & GONE from the scheme (no new number): **3.5 score floor**, **3.7 plausibility**.
> This is where the "why the gap" confusion disappears — Phase 2 is a clean 2.1→2.4.

### Phase 3 · Validate SIS
| New | Old | What | Notes |
|-----|-----|------|-------|
| **3.1** | Step 4 | validate Search Impact Set (LLM #2) | ⭐ — the gain driver |

### Phase 4 · Resolve & Trace
| New | Old | What | Notes |
|-----|-----|------|-------|
| **4.1** | Step 5 | seed-resolve doc SIS → code seeds | deterministic |
| **4.2** | Step 5b | trace-validate resolutions (LLM #3) | ⭐ |

### Phase 5 · Propagate  (TWO PARALLEL ARMS)
| New | Old | What | Notes |
|-----|-----|------|-------|
| **5.1a** | Step 6 | outward BFS expand | ‖ outward arm |
| **5.2a** | Step 6.8 | weight-decay prune (outward) | ‖ deterministic prune |
| **5.3a** | Step 7 | validate propagation (LLM #4) | ⭐ ‖ |
| **5.1b** | Step 6.7 | in-file sibling expand | ‖ in-file arm |
| **5.2b** | Step 6.9 | anchor-rrf prune (siblings) | ‖ deterministic prune |
| **5.3b** | Step 7.5 | validate siblings (LLM #4, distinct call) | ⭐ ‖ |

> The `a`/`b` split is the two-arms redesign made visible: each arm is
> expand → prune → validate. (Old `7.5a` was an internal sub-label of 7.5 — folds into 5.3b.)

### Phase 6 · Assemble
| New | Old | What | Notes |
|-----|-----|------|-------|
| **6.1** | Step 8 | context-build (backlinks, snippets, token budget) | deterministic |
| **6.2** | Step 9 | synthesize (LLM #5, aggregator-only) | ⭐ |

> Step 0 (setup/guards) → keep as **Step 0** (pre-phase bootstrap) OR fold into 1.1 prologue.
> **DECISION NEEDED — see Q below.**

---

## OFFLINE indexer (`impactracer/indexer/runner.py`)

Purely sequential (no parallel arms). Phase.Step mirrors the online style.

### Phase A1 · Ingest
| New | Old | What |
|-----|-----|------|
| **A1.1** | Step 0 | connect persistence |
| **A1.2** | Step 1 | scan repo |
| **A1.3** | Step 2 | hash diff |
| **A1.4** | Step 3 | purge deleted files |

### Phase A2 · Extract
| New | Old | What |
|-----|-----|------|
| **A2.1** | Step 4 | Markdown chunking |
| **A2.2** | Step 5 | AST Pass 1 (nodes) |
| **A2.3** | Step 6 | AST Pass 2 (edges) |

### Phase A3 · Embed & Link
| New | Old | What |
|-----|-----|------|
| **A3.1** | Step 7 | embed + upsert to ChromaDB |
| **A3.2** | Step 8 | traceability recompute |
| **A3.3** | (project-skeleton extraction) | skeleton for LLM #1 |

### Phase A4 · Finalize
| New | Old | What |
|-----|-----|------|
| **A4.1** | Step 9 | update file_hashes |
| **A4.2** | Step 10 | write index_metadata |

> **NOT renumbered** (out of scope — these are *algorithm-internal* labels, local to one
> function, not pipeline steps): `code_indexer.py` "Step 1–4d" (sub-passes inside A2.2/A2.3)
> and `traceability.py` "Step 1–5" (sub-steps inside A3.2). Leaving their internal numbering
> as-is; only their *Blueprint §* cross-refs get updated. **CONFIRM this scoping.**

---

## Function renames (numbers dropped from identifiers)
| Old identifier | New identifier |
|----------------|----------------|
| `step_3_6_semantic_dedup` | `semantic_dedup` |
| `apply_prevalidation_gates` | (unchanged — already unnumbered) |
| `validate_siblings_for_file` | (unchanged — already unnumbered) |
| `validate_propagation` | (unchanged) |
| *(no other `def step_N_*` exist — verified)* | |

## Blueprint §-number cross-refs in code
~30 `master_blueprint.md §4` / `§3.x` docstring refs → repoint to the new phase sections
once `master_blueprint.md` itself is restructured into phases. (Mechanical, last.)

## Report / log output strings
Calibration report prints "6.8 drops / 6.9 drops" (`report_builder.py`) → "5.2a drops /
5.2b drops". All `logger.info("[runner] Step X …")` → new labels. ~150 string sites total.

---

## RESOLVED (sign-off complete)
1. **Step 0:** KEEP as explicit pre-phase bootstrap in BOTH pipelines. Phases start at 1.1 / A1.1.
2. **Indexer-internal labels:** LEAVE `code_indexer.py` (Step 1–4d) + `traceability.py`
   (Step 1–5) untouched; only repoint their `§` cross-refs.
3. **Top-K cut:** PROMOTE to explicit **2.4**.

## EXECUTION ORDER (golden-gate after each file group)
1. Rename `step_3_6_semantic_dedup` → `semantic_dedup` (prevalidation_filter.py + runner + tests + __init__).
2. Online `runner.py` step-label strings/comments → Phase.Step.
3. Online per-module docstrings + log strings (validator, retriever, seed_resolver,
   traceability_validator, graph_bfs, traversal_validator, context_builder, synthesizer, variant_cache).
4. shared/ (config.py, constants.py, models.py) step refs.
5. Offline indexer runner.py Step 0–10 → A1.1–A4.2; fix § cross-refs in code_indexer/traceability.
6. report_builder.py output strings (6.8/6.9 → 5.2a/5.2b).
7. Docs: master_blueprint.md (restructure into phases) + CLAUDE.md + analysis_implementation.md
   + index_implementation.md + pipeline.md.
8. GOLDEN-GATE: verify_index_golden + propagate-replay online golden + full pytest.
9. Delete this map + RENUMBER scratch.
