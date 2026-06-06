# Online Analysis Pipeline — Source of Truth

> Operational reference for the V7 online pipeline as it currently runs.
> Every claim here quotes a constant from `shared/constants.py` /
> `shared/config.py` or cites an empirical number from a calibration run.
>
> Companion: `master_blueprint.md` is the design specification.
> `index_implementation.md` is the offline-indexer operational reference.
>
> The pipeline is organized into **six phases** plus a Step 0 bootstrap.
> The propagation phase (Phase 5) runs **two parallel arms** — outward
> dependency BFS and in-file siblings — each shaped expand → deterministic
> prune → LLM validate. Step labels are `P.S` (phase.step); the two arms
> carry an `a`/`b` suffix.

---

## 1. Pipeline Architecture — Six Phases

The online pipeline transforms an Indonesian / English Change Request (CR) into a structured `ImpactReport` with deterministic entity-level rows and LLM-assisted file-level summaries. It is orchestrated by `impactracer/pipeline/runner.py::run_analysis`. Variant flags (V0..V7) toggle each LLM call independently for the ablation; the description below is V7.

```
              CR text (Indonesian or English)
                       │
 Phase 1  ───────────▼──────────────────────  LLM #1 — interpret_cr
 Interpret           Two-stage by default: 1.1 interpret_intent +
 (always-on)         1.2 interpret_anchors (grounded in the cached
                     project skeleton); single-stage fallback when no
                     skeleton exists. CRInterpretation: is_actionable,
                     change_type, affected_layers, primary_intent,
                     domain_concepts, search_queries (EN),
                     layered_search_queries, named_entry_points,
                     anchor_candidates (soft signal),
                     out_of_scope_operations, is_nfr.
                     │   Coherence soft-fix:
                     │     DELETION ⇒ ensure 'code' in affected_layers
                     │     ADDITION ⇒ ensure not code-only
                     ▼
 Phase 2  ─── RRF Hybrid Search (unweighted) ──────────────────────────
 Retrieve   Step 2.1   Four ranked lists fused at equal weight:
 (V0+)                   • dense_doc   (BGE-M3 embedding × ChromaDB)
                         • bm25_doc    (rank_bm25 over chunked SRS/SDD)
                         • dense_code  (BGE-M3 embedding × ChromaDB)
                         • bm25_code   (rank_bm25 over code embed_text)
                       dense_code additionally incorporates a RAW-CR
                       multilingual dense pass and traceability pool
                       seeding (see §4). Output: top-K RRF pool
                       (top_k_rrf_pool = 200).
                       │
            Step 2.2   Semantic Dedup (POST-RETRIEVAL, on FULL pool,
                       BEFORE rerank + cut). Doc chunks whose top-1 code
                       resolution is already in the pool are merged into
                       that code candidate; (section_title, text) attached
                       as "Business Context" for LLM #2 (V4+). The only
                       deterministic pre-validation gate.
                       │
            Step 2.3   Cross-Encoder Rerank (V3+). BGE-Reranker-v2-m3,
                       multi-query MAX scoring on the deduped pool.
                       V0–V2: no reranker.
                       │
            Step 2.4   Top-K cut — plain top-15 by cross-encoder score
                       (V3+) / by RRF score (V0–V2).
                       │
                       ▼
 Phase 3  ─── LLM #2  validate_sis (V4+) ──────────────────────────────
 Validate   Step 3.1   Batched ≤ 5 candidates per call.
 SIS                   Per-node and batch-level fail-CLOSED:
                         • missing verdict  → DROP that node
                         • exception        → DROP entire batch, continue
                                              (degraded_run flag set).
                       Captures verdict.{function_purpose,
                       mechanism_of_impact, justification} for each
                       admitted seed → propagated to NodeTrace.
                       │
                       ▼
 Phase 4  ─── Resolve & Trace ─────────────────────────────────────────
 Resolve    Step 4.1   Doc → Code Resolution (V5+). Direct code seeds pass
 & Trace               through; doc-chunk seeds are looked up in
                       doc_code_candidates → top-K code candidates per doc.
                       │
            Step 4.2   LLM #3  validate_trace (V5+). Batched ≤ 5 (doc, code)
                       pairs. Two-standard test (link is structural AND CR
                       changes the code). Per-pair fail-CLOSED → REJECTED on
                       missing verdict. Each surviving code_id keeps the BEST
                       decision (CONFIRMED > PARTIAL > REJECTED).
                       │
                       ▼
 Phase 5  ─── Propagation (the Propagator) — TWO PARALLEL ARMS ─────────
 Propagate
   OUTWARD arm (dependency BFS)         IN-FILE arm (file-local siblings)
   Step 5.1a  BFS over structural       Step 5.1b  collect_file_local_
              graph; per-edge depth/                siblings injects RAW,
              direction (§3). V6+.                  unvalidated siblings of
              │                                     each mechanism-carrying
   Step 5.2a  Weight-decay prune                    anchor (V6+). NO LLM.
              (deterministic, K=10).      │
              Recall-safe flood          Step 5.2b  Anchor-RRF prune
              control. SIS never cut.               (deterministic,
              │                                     K=sibling_prune_top_k).
   Step 5.3a  LLM #4 validate_           │
              propagation (V7).          Step 5.3b  LLM #4 validate_siblings
              Prunes BFS nodes;                     _for_file (V7). DISTINCT
              passes siblings through.              call; admits/rejects raw
                                                    siblings per file. Caps
                                                    apply here (per_file=4).
                       │
                       ▼
 Phase 6  ─── Assemble ────────────────────────────────────────────────
 Assemble   Step 6.1   Context Build. Backlinks (bidirectional doc↔code),
                       source snippets (ILA preferred for Function/Method),
                       and a CANONICAL FILE LIST in a "=== IMPACTED FILES ==="
                       header. Token-budget truncation is severity-aware
                       (severity_rank ASC, depth ASC, node_id) and DECOUPLED
                       from output: only the LLM #5 prompt is trimmed; the
                       report's impacted_entities always contains the FULL
                       validated CIS.
                       │
            Step 6.2   LLM #5  synthesize (always-on). Aggregator-only:
                       LLMSynthesisOutput = { executive_summary,
                       documentation_conflicts, file_justifications }.
                       Hallucinated files dropped; omitted files get a
                       deterministic fallback. NEVER authors entity rows.
                       │
                       ▼
              ImpactReport
              ├── executive_summary           (LLM #5)
              ├── impacted_files              (deterministic file set +
              │                                LLM #5 justification or
              │                                deterministic fallback)
              ├── impacted_entities           (deterministic; every
              │                                validated CIS node;
              │                                justification verbatim from
              │                                LLM #2/#3/#4 or auto_exempt)
              ├── documentation_conflicts     (LLM #5)
              ├── estimated_scope             (deterministic from CIS size)
              ├── analysis_mode               ('retrieval_only' or
              │                                'retrieval_plus_propagation')
              └── degraded_run                (true if any LLM batch dropped)
```

The runner writes a per-step trace to `impact_report_full.json` when `trace_sink` is provided (always populated by the CLI). The trace keys retain stable identifiers for tooling continuity (they are functional data keys, not display labels):

```
step_1_interpretation, step_2_rrf_pool, step_3_reranked_full,
step_3_reranked, step_3_gates_survivors, step_4_llm2_verdicts,
step_5_resolutions, step_5b_llm3_verdicts, step_6_bfs_raw_cis,
step_7_llm4_verdicts, step_7p5_sibling_validation, final_report
```
(The deterministic propagation sub-steps `step_6p6_deletion_import_only_filter`, `step_6p7_sibling_expansion`, `step_6p8_weight_decay_prune`, and `step_6p9_sibling_precision_prune` are also emitted by `graph_bfs.py` and consumed by `report_builder.py`.)

These keys may be absent for variants that disable the corresponding phase or for CRs that resolve to zero seeds before a stage.

---

## 2. The Five LLM Invocations — Roles, Prompts, Fail-Closed Logic

### LLM #1 — Interpret (`interpret_cr`) — Phase 1

* **Role:** parse the CR into a structured `CRInterpretation`. Schema-constrained, always-on. Runs as **one OR two** calls depending on `variant_flags.two_stage_interpret` (default True) and whether a project skeleton exists on disk.
* **Output schema** (`shared/models.py::CRInterpretation`): `is_actionable`, `actionability_reason`, `primary_intent`, `change_type` ∈ {ADDITION, MODIFICATION, DELETION}, `affected_layers ⊆ {requirement, design, code}`, `domain_concepts`, `search_queries` (English even when CR is Indonesian), `layered_search_queries`, `named_entry_points`, **`anchor_candidates`**, `out_of_scope_operations`, `is_nfr`.
* **Two-stage interpretation (default, `interpret_cr_two_stage`):**
  * **Step 1.1** (`call_name="interpret_intent"`, schema `CRIntent`): actionability, change_type, affected_layers, domain_concepts, is_nfr — from the CR text alone, **no project context**. A not-actionable verdict here skips Step 1.2.
  * **Step 1.2** (`call_name="interpret_anchors"`, schema `CRAnchors`): receives Step-1.1's output PLUS the cached **project skeleton** (`indexer/project_skeleton.py`, read from `Settings.project_skeleton_path`) and emits the retrieval-side fields: `search_queries`, `layered_search_queries`, `named_entry_points`, `anchor_candidates`, `out_of_scope_operations`. The skeleton's naming-convention + domain-vocabulary sections keep anchors grounded in real symbols (camelCase functions vs PascalCase classes) rather than hallucinated shapes.
  * The runner glues both outputs into one `CRInterpretation`. The variant cache memoises the glued result so V0–V7 on the same CR share it.
* **Single-stage fallback (`interpret_cr_single_stage`, `call_name="interpret"`):** one call emitting the full schema. Used when `two_stage_interpret=False` (the methodology ablation via `with_two_stage_interpret`) OR when no skeleton file exists. Same output shape, so all downstream code is mode-agnostic.
* **Anchor priming (default-on, `variant_flags.anchor_priming`):** the interpreter populates `anchor_candidates` with 1–3 bare identifier guesses at likely host/sibling symbols even when the CR does not name them. These are HYPOTHESES applied as a SOFT signal only — a synthetic-BM25 boost (`anchor_priming_bm25_boost=1.5`, retriever Path 4). A hallucinated anchor still faces the full validator chain; `with_anchor_priming(flags, False)` reproduces the baseline.
* **Fail-closed:** a Pydantic ValidationError on either stage's response halts the run with a rejection report. The `is_actionable=False` branch short-circuits to a minimal rejection report with no downstream calls.
* **Distributed Justification role:** none. LLM #1 produces metadata; it does not validate any node.

### LLM #2 — Validate SIS (`validate_sis`) — Step 3.1

* **Role:** judge whether each retrieved candidate is DIRECTLY impacted by the CR. Operates on the top-15 cut survivors. Batches of ≤ 5 candidates per call.
* **Prompt constraints** (`pipeline/validator.py::_SYSTEM_PROMPT`):
  * No retrieval scores in the prompt — anti-circular mandate.
  * Distinguishes code-node vs doc-chunk verdict criteria.
  * ADDITION CRs get a forward-looking branch (absence of logic IS reason to confirm an entry point), plus an ADDITION-scoping guard against same-service over-confirmation.
  * Forces concrete `mechanism_of_impact` ("vague 'related' justifications forbidden"). This text becomes the seed-level Distributed Justification propagated verbatim to `ImpactedEntity.justification` with `justification_source="llm2_sis"`.
  * Delimiter contract: copy `node_id` from BETWEEN `<<NODE_ID_START>>...<<NODE_ID_END>>`, do NOT include the markers in the JSON output. The runner sanitises any leftover markers before lookup.
* **Fail-closed:** per node, missing verdict → DROP; per batch, any uncaught exception (after `LLMClient.call` retries) → batch DROPPED, `degraded=True`, loop continues.
* **Captures:** `function_purpose`, `mechanism_of_impact`, `justification` for every confirmed seed → attached to `NodeTrace`.

### LLM #3 — Validate Trace (`validate_trace`) — Step 4.2

* **Role:** for each `(doc_chunk, code_node)` pair produced by Step 4.1 resolution, apply the SAME two-standard test as LLM #2 — (1) the code implements the doc section AND (2) the CR structurally modifies the code — and decide CONFIRMED / PARTIAL / REJECTED. Batches of ≤ 5 pairs.
* **Prompt constraints** (`pipeline/traceability_validator.py::_SYSTEM_PROMPT`):
  * Judge by AST structure and document semantics — never by score.
  * **Two standards, both required for CONFIRMED.** A correct-but-unchanged implementation of an in-scope requirement is REJECTED — relevance to the doc is not, by itself, impact.
  * CONFIRMED → concrete `mechanism_of_impact`. PARTIAL → link holds, change unclear; empty mechanism. REJECTED → no structural change relationship.
  * For ADDITION CRs: absence of current implementation does NOT mean REJECTED — confirm with a mechanism describing what must be added.
* **Fail-closed:** per pair, missing verdict → REJECTED; per batch, exception → all pairs REJECTED, continue.
* **Captures:** the verdict justification AND `mechanism_of_impact` of the BEST decision per code_id → `NodeTrace` with `justification_source="llm3_trace"`. Returns `(seeds, low_conf, justifications, mechanisms, degraded)`. A CONFIRMED seed's non-empty mechanism makes it anchor-eligible for in-file sibling promotion (Step 5.1b), parity with direct LLM-#2 seeds; PARTIAL/REJECTED are not.

### LLM #4 — Validate Propagation (`validate_propagation`) + sibling sub-call — Steps 5.3a / 5.3b

* **Primary role (Step 5.3a, outward arm):** for each BFS-propagated node that is NOT auto-exempt, decide whether the structural reach implies semantic impact.
* **Sibling sub-call (Step 5.3b, in-file arm):** a **distinct** LLM #4 call (`validate_siblings_for_file`) that admits/rejects the raw siblings injected at Step 5.1b. Step 5.3a passes `sibling_candidate` nodes through untouched; 5.3b adjudicates them.
* **Prompt constraints** (`pipeline/traversal_validator.py::_SYSTEM_PROMPT`):
  * **De-blinded chain:** the causal chain IS shown as factual context. Tautology safety is enforced by explicit anti-tautology language: *"Edge types are NOT impact evidence … Reject any node where the relationship is structurally present but the target's behaviour is unaffected."*
  * Forbidden justification patterns enumerated ("function A calls function B" / "in the same module" rejected). Required format demands a contract-breakage / behavioural-anomaly / downstream-type-mismatch sentence.
  * Determinism: `random.seed(42); random.shuffle(...)` before batching to neutralise positional bias reproducibly.
* **Auto-exempt edges:** depth-1 `IMPLEMENTS` and `DEFINES_METHOD` (`PROPAGATION_VALIDATION_EXEMPT_EDGES`) bypass the LLM call, receiving `"Direct <edge> contract from <seed> — auto-admitted exempt edge."` with `justification_source="auto_exempt"`. `TYPED_BY` is intentionally NOT exempt — auto-exempt TYPED_BY admissions historically produced too many false positives; LLM #4 adjudicates depth-1 TYPED_BY like any deeper chain.
* **Sibling anchor qualification (5.3b):** anchor must be in `sis_justifications` OR `trace_mechanisms` AND have a non-empty `mechanism_of_impact`. The prompt receives ALL qualifying anchors in the file with their justifications. Per-file admission cap `settings.sibling_admit_max_per_file = 4`; admitted siblings receive `justification_source="llm4_sibling"`.
* **Fail-closed:** per node → DROP; per batch / per sibling-file-batch → DROP that batch, continue.
* **Captures:** verdict justification → `NodeTrace.justification` with `justification_source="llm4_propagation"` (outward) or `"llm4_sibling"` (in-file).

### LLM #5 — Synthesize (`synthesize_summary`) — Step 6.2

* **Role (aggregator-only):** produces the executive summary, documentation-conflicts list, and **per-file** justifications. NEVER produces per-entity justifications.
* **Distributed Justification Principle:**
  * `impacted_entities` is built deterministically by the runner from the validated CIS. Each entity's `justification` is propagated VERBATIM from the LLM (#2/#3/#4) that admitted it, or from the synthetic `auto_exempt` string. LLM #5 never sees nor authors these.
  * `impacted_files` is deterministic w.r.t. its `file_path` set: every distinct file referenced by `impacted_entities` MUST have exactly one row. The `justification` field of each file row may be written by LLM #5 (file-level summarisation is a summarisation task). Hallucinated files are dropped; omitted files receive a deterministic fallback summarising the entity-level justifications inside that file.
* **File-type filter:** `build_deterministic_impacted_entities` HARD-FILTERS every CIS node whose `node_type == "File"` or whose `node_id` lacks `::`. GT `impacted_entities` only ever contains qualified `file::symbol` ids. File-level impact is preserved separately via `assemble_impact_report(..., extra_impacted_file_paths=...)`.
* **Fail-closed:** if `LLMClient.call` raises after retry exhaustion, the runner falls back to `build_minimal_summary` (`degraded_run=True`). The deterministic `impacted_entities` / `impacted_files` lists are emitted regardless — they exist independently of LLM #5.

---

## 3. Graph Constraints — Phase 5 (Propagation) Rules

The structural graph is a `networkx.MultiDiGraph` materialised once per pipeline context from the SQLite `structural_edges` table. The outward BFS arm is governed by `EDGE_CONFIG` in `impactracer/shared/constants.py`.

### 3.1 Per-edge direction and max_depth (`EDGE_CONFIG`, 9 edge types)

| Edge type | Direction | Max depth | Rationale |
|---|---|---:|---|
| `CALLS` | reverse | **2** | Depth-3 fan-in regularly produces 200+ propagated nodes per seed in TS codebases. Depth-2 is the precision-recovery sweet spot. |
| `INHERITS` | reverse | 3 | Class hierarchies are typically shallow. |
| `IMPLEMENTS` | reverse | 3 | Interface contract graph. |
| `TYPED_BY` | reverse | 3 | Type-reference propagation. |
| `DEFINES_METHOD` | forward | 1 | Definitional containment, not semantic propagation. |
| `IMPORTS` | reverse | 1 | Module composition; no transitive impact assumed. |
| `RENDERS` | reverse | 1 | UI parent→child composition. The single most productive propagation edge on the evaluated corpora (caught 13/17 BFS true positives, all sole-credit). Extractor is JSX/TSX-specific but the relation generalises to other component frameworks. |
| `DYNAMIC_IMPORT` | reverse | 1 | Lazy/code-split module boundaries. |
| `CONTAINS` | reverse | 1 | File ↔ symbol containment. Reverse-only: given a changed symbol, find which files contain it. Sibling enumeration is handled by the separate in-file arm (5.1b), not by CONTAINS BFS. |

`INHERITS` / `IMPLEMENTS` have no instances on the citrakara corpus (no class hierarchies) but remain defined for TypeScript generality.

### 3.2 Confidence-tier CALLS cap

`LOW_CONF_CAPPED_EDGES = {CALLS}`: low-confidence seeds (not in the top-N reranker scores AND not directly retrieved) cap CALLS depth to 1, preventing low-quality seeds from emitting deep chains.

### 3.3 Hub mitigation

`_HUB_DEGREE_THRESHOLD = 20`: nodes whose total degree > 20 (generic interfaces, framework primitives) are capped at depth 1 for ALL edges when traversing FROM them. Prevents combinatorial explosion through hubs.

### 3.4 UTILITY-file CALLS cutoff

Seeds whose `file_classification == "UTILITY"` cap their reverse-CALLS chain at `UTILITY_FILE_CALLS_DEPTH_CAP = 1`. Utility functions are called from everywhere; deeper reverse-CALLS from a UTILITY seed floods unrelated features.

### 3.5 Per-node-type fan-in cap

`NODE_TYPE_MAX_FAN_IN`: a propagated neighbour with in-degree exceeding the type-specific cap is excluded from the CIS unless it is itself a SIS seed.

| Node type | Max fan-in |
|---|---:|
| Function / Method / Class | 50 |
| Interface / TypeAlias / Enum | 100 |
| File | 200 |
| Variable | 80 |

### 3.6 Severity (last-hop rule)

`severity_for_chain(causal_chain)` returns the severity of the LAST edge in the chain. SIS seeds (empty chain) are `Tinggi` by convention. This eliminates "severity laundering" where a chain like `CALLS → CALLS → IMPLEMENTS` would inherit IMPLEMENTS's severity from an otherwise speculative path.

### 3.7 Weight-decay prune (Step 5.2a, deterministic, default-ON)

After the outward BFS, the propagated pool (SIS seeds + BFS nodes) is ranked by `prod(edge_weight)/(1+depth)` (RENDERS=1.0 … IMPORTS=0.3) and the top-K kept (`propagation_prune_top_k=10`). SIS seeds are never cut. Recall-safe flood control: K=10 is the recall-safe floor found by a live K-sweep on the V6 BFS pool (BFS-GT recall is 100% at K≥10, and the binding CR has its last BFS-GT at rank 9). The earlier K=20 was the conservative choice; K=10 is recall-identical on this corpus with marginally better precision and ~half the BFS budget on flood CRs. Overridable via `PROPAGATION_PRUNE_TOP_K`.

### 3.8 Graph isolation invariant

`propagate` does NOT mutate the shared graph. Sequential ablation runs (V0 → V7 over the same CR) produce identical CIS results given identical inputs. Seeds absent from the graph are recorded as SIS-only terminal nodes (no expansion).

### 3.9 In-file siblings — split across V6 (expand) / V7 (validate)

The in-file arm mirrors outward BFS (expand at V6, prune at V7):

**Step 5.1b — Sibling EXPANSION (V6+, gated `enable_bfs`, deterministic, no LLM).** `runner.py` invokes `collect_file_local_siblings` (`graph_bfs.py`) to enumerate qualified siblings of every qualifying anchor. **Anchor qualification:** in `sis_justifications` OR `trace_mechanisms` (LLM #2 *or* CONFIRMED LLM #3) AND a non-empty `mechanism_of_impact`. Queries `code_nodes` for every qualified (`::`-bearing) same-file symbol with `node_type ∈ {Function, Method, Interface, TypeAlias, Enum, Class, Variable}`. Per-file candidate cap `settings.sibling_promotion_max_per_file = 12`. Candidates are injected **raw, unvalidated** into `cis.propagated_nodes` (`causal_chain=["CONTAINS"]`, `depth=1`, `justification_source="sibling_candidate"`). **No admission caps here.**

**Step 5.2b — Sibling PRUNE (V6+, deterministic).** The raw siblings are ranked by anchor-RRF and the top-K kept (`sibling_prune_top_k`, overridable via `SIBLING_PRUNE_TOP_K`). The anchor-RRF scorer beat PPR / semantic / flat alternatives in a bake-off.

**Step 5.3b — Sibling VALIDATION (V7, gated `enable_propagation_validation`).** A **distinct** LLM #4 call (`validate_siblings_for_file`) admits/rejects the pruned siblings per file using the file's anchors' mechanisms as context. Rejected → DROPPED; admitted → `justification_source="llm4_sibling"`. Per-file cap `settings.sibling_admit_max_per_file = 4`.

Net boundary: V4–V5 have no siblings; V6 carries pruned-but-unvalidated siblings (recall↑, precision↓); V7 LLM-validates them. Both arms gate additionally on `settings.enable_sibling_promotion=True`.

---

## 4. Retrieval Architecture (Phase 2 Detail)

The retriever is the deepest non-LLM lever in the pipeline. Several orthogonal mechanisms run inside `hybrid_search`.

### 4.1 RRF pool sizing

```
top_k_per_query = 30      # per dense/BM25 query per path
top_k_rrf_pool  = 200     # candidates entering the cross-encoder
max_admitted_seeds = 15   # cap after rerank (Step 2.4)
rrf_k = 60
```

The 200-candidate pool is wide on purpose: the cross-encoder is the actual selector. A narrow pool starves it of the right answer; a wide pool gives it a meaningful selection problem at the cost of ~30s extra reranker time per V3+ run.

### 4.2 Raw-CR multilingual dense pass

Gating: `settings.enable_raw_cr_dense_pass = True` AND `cr_text` is not None AND the dense path is enabled.

Inside the `dense_code` branch, after the LLM-#1-search-query loop, the retriever embeds the raw CR text once and queries `code_col` for `settings.raw_cr_dense_top_k = 60` nearest neighbours. Results merge into `seen_dc` via score-max.

Rationale: BGE-M3 is multilingual. The CR is Indonesian; the code identifiers are English. The LLM #1 search-query intermediation strips that direct signal and replaces it with concept-centric English queries; the raw-CR pass restores the direct embedding-space path.

### 4.3 Traceability pool seeding

Gating: `settings.enable_traceability_pool_seeding = True` AND `cr_interp.affected_layers` includes `"code"` AND ≥1 doc-chunk retrieved AND `ctx.conn` not None.

After dense_doc / bm25_doc retrieve doc-chunk lists, the retriever queries `doc_code_candidates` for those doc_ids with `weighted_similarity_score >= settings.traceability_seed_min_score = 0.40`, and seeds up to `settings.traceability_seed_top_k_per_doc = 5` code neighbours per doc into `dense_code_ids` (unless already present).

Rationale: the offline traceability matrix is the canonical "this doc chunk talks about this code" precomputation. Pool-membership seeding lets it introduce GT-correct candidates no LLM #1 search query happens to mention.

### 4.4 Layered code retrieval path

A ranked list, `layered_code`, is built when `cr_interp.layered_search_queries` is populated. For each canonical layer in `_CANONICAL_LAYERS = ("api_route", "page_component", "ui_component", "utility", "type_definition")`:

1. Look up the layer's queries (1–2 phrases per layer).
2. Run a dense BGE-M3 query against `code_col` with `where={"file_classification": <FileClassification>}`, up to `settings.per_layer_top_k = 12`.
3. Run a BM25 query, then filter post-hoc by `file_classification`.
4. Per-layer hits merged via score-max; top-K per layer feeds the global `layered_code` list, which joins RRF as a first-class path (weight 1.0).

This guarantees no architectural layer is starved when LLM #1's flat `search_queries` are biased toward one plane.

### 4.5 Anchor-priming BM25 path (Path 4)

When `cr_interp.anchor_candidates` is non-empty AND `settings.anchor_priming_bm25_boost > 1.0` (default 1.5), each anchor identifier is fed through the code BM25 index as an extra synthetic query scaled by `(boost − 1.0)`, surfacing anchor-matching code identifiers earlier in `bm25_code` before RRF fusion. A hallucinated anchor that surfaces a junk candidate still faces LLM #2. Gated by `variant_flags.anchor_priming` (default True); `with_anchor_priming(flags, False)` zeroes the boost.

---

## 5. Empirical Result (citrakara, cleaned index)

The pre-registered ablation runs V0–V7; the primary metric is entity-level set F1 (`f1_set`) against the full unpruned predicted set. The locked test is a one-sided paired Wilcoxon (V7 vs V5) at N ≥ 15.

### 5.1 Variant table (golden V0–V7, citrakara)

| Variant | What it adds | Entity F1 |
|---|---|---:|
| V0 | BM25-only retrieval | 0.198 |
| V1 | + dense retrieval | 0.184 |
| V2 | + RRF fusion | 0.239 |
| V3 | + cross-encoder rerank | 0.201 |
| V4 | + LLM #2 SIS validation | 0.404 |
| V5 | + LLM #3 trace validation | 0.404 |
| V6 | + deterministic propagation (both arms, pruned, unvalidated) | 0.206 |
| **V7** | + LLM #4 propagation validation (both arms) | **0.411** |

### 5.2 Mechanistic signatures

- **LLM #2 is the gain driver.** V3 → V4 is the single largest jump (0.201 → 0.404): the deterministic stack plateaus and the LLM validator delivers the precision.
- **The cross-encoder (V2 → V3) is net-negative** (−0.038 entity F1 on citrakara, −0.003 on nova). It is a pre-registered ablation stage representing standard retrieve-then-rerank practice; the ablation finds it contributes nothing positive on this task. Reported as a finding; retained in the ladder (not removed post-hoc). Mechanism: the corpus is JSDoc-sparse (≈26% overall, 34% on GT-relevant types), so the reranker ranks the majority of candidates on documentation-presence rather than structural necessity, and — running before the top-K cut — can evict structurally-required-but-topically-dull seeds.
- **V6 → V7 is the propagation-validation payoff.** V6 (deterministic, pruned, unvalidated) trades recall for precision and lands at 0.206; LLM #4 across both arms recovers precision to V7 = 0.411 > V5 = 0.404. The two-arm redesign lifted V6 from 0.150 to 0.206 at zero recall cost and projected ~26% fewer LLM #4 calls.
- **Structural recall ceiling ≈ 0.62.** Some GT lives behind edges the static graph cannot reach (e.g. UI components that fetch via API + Zod parse rather than importing schemas), an inherent limit of static CIA on decoupled architectures.

### 5.3 Thesis posture

The contribution is the **pre-registered ablation and its findings** — LLM validation (Phases 3/4 + 5.3a/5.3b) drives the gains; the deterministic apparatus (retrieval fusion, cross-encoder, graph propagation/pruning) trades recall for precision without a significant net win — not the tool's peak score. The in-between deterministic stages are apparatus/controls, defended by failure-mode rationale, not as variant boundaries.

---

## 6. Frozen Invariants

The following architectural invariants are FROZEN. Violating any requires updating this document AND `master_blueprint.md`.

1. **Node vocabulary** (`shared/models.py::NodeType`): the schema defines 8 node types (`File, Class, Function, Method, Interface, TypeAlias, Enum, Variable`). The live citrakara index emits 7 (no `Enum` instances on this corpus).
2. **Edge vocabulary** (`shared/constants.py::EDGE_CONFIG`, `persistence/sqlite_client.py` `structural_edges` CHECK): 9 propagation edge types, all walked by BFS — see §3.1. (There is no `EdgeType` Literal in `models.py`; the edge vocabulary is defined by `EDGE_CONFIG`'s keys and the SQLite CHECK constraint, not a typed enum.)
3. **5 canonical LLM stages in V7**: `interpret`, `validate_sis`, `validate_trace`, `validate_propagation`, `synthesize`. Per-CR call counts exceed 5 because the two-stage interpreter splits `interpret`, and Steps 5.3a / 5.3b each spawn per-child / per-file LLM #4 sub-calls. The five canonical stage names remain the contract.
4. **8 canonical ablation variants** (`evaluation/variant_flags.py::ALL_VARIANTS`). V3 = deterministic-filtering peak (cross-encoder rerank + cut, no LLM gating). V7 = full pipeline (both propagation arms + LLM #4 + LLM #5 aggregator).
5. **3 change_type values**: `ADDITION, MODIFICATION, DELETION`.
6. **Fail-CLOSED at every validator.** Per-item (drop on missing verdict) and per-batch (drop on exception, continue) at LLM #2, #3, #4 outward (5.3a), and #4 sibling (5.3b). The runner sets `degraded_run=True` when any drop fires.
7. **Distributed Justification Principle.** Entity-level justifications come VERBATIM from LLM #2 / #3 / #4 (including the sibling sub-call) or a synthetic `auto_exempt` string. LLM #5 never re-justifies entities. File-level justifications may be authored by LLM #5.
8. **Truncation decoupled from output.** The LLM #5 prompt may be truncated to fit the token budget; `impacted_entities` always contains the FULL validated CIS.
9. **CALLS reverse depth = 2.** Combined with the UTILITY-CALLS cutoff and per-node-type fan-in cap, this is the structural defence against graph flood.
10. **Semantic dedup (Step 2.2) is the only pre-validation gate**, run before rerank and the top-K cut. No score floor, plausibility, negative filter, traceability bonus, or RRF path weighting exists in the pipeline (all retired by ablation and removed from the code).
11. **The Wilcoxon test target is entity-level `f1_set`** (set-level F1). Bounded `F1@K` is absent from the codebase — it cannot detect graph floods.
12. **NFR-01 compares the validated SIS, not impacted_entities** — `trace_sink["step_5b_llm3_verdicts"]["validated_code_seeds"]` across two V7 runs. BFS + LLM #4 + sibling validation carry network-induced variance NFR-01 is not designed to test.
13. **File-type entities are filtered from `impacted_entities` at synthesis.** Every CIS node with `node_type == "File"` or without `::` is dropped; its `file_path` is still injected into `impacted_files`.
14. **TYPED_BY is NOT in `PROPAGATION_VALIDATION_EXEMPT_EDGES`.** Only `IMPLEMENTS` and `DEFINES_METHOD` are auto-exempt at depth 1.
15. **In-file sibling anchors (5.1b) require a non-empty LLM #2/#3 `mechanism_of_impact`.** Anchors without an articulate mechanism cannot drive lateral file-local expansion.
16. **LLM #1 is two-stage by default.** `interpret_cr_two_stage` runs whenever `variant_flags.two_stage_interpret=True` AND a skeleton file exists. The single-stage `interpret` path is the documented fallback. Both return the identical `CRInterpretation` shape.
17. **Anchor priming is a SOFT signal, never a hard pin.** `CRInterpretation.anchor_candidates` (when `variant_flags.anchor_priming=True`) contributes only a synthetic-BM25 boost (`anchor_priming_bm25_boost=1.5`). Any anchor-surfaced candidate still faces LLM #2.
18. **Stores are profile-namespaced under `./data/<profile>/`.** `get_settings(profile)` rewrites the five store paths after `Settings()` resolves `.env`, so the resolved profile (`--profile` > `IMPACTRACER_PROFILE` > `"citrakara"`) is authoritative. `load_pipeline_context` fails fast (`RuntimeError`) when the chosen profile's `code_nodes` table is empty.

---

## 7. Operational Surfaces

* **CLI** (`impactracer.cli`) — every command accepts `--profile/-p NAME` (default `citrakara`; `IMPACTRACER_PROFILE`-overridable) and echoes the resolved profile + paths to stderr:
  * `impactracer index <repo> [--profile NAME]` — offline indexer.
  * `impactracer analyze "<CR text>" --variant V7 [--output PATH] [--profile NAME]` — online analysis; writes `impact_report.json` + `impact_report_full.json`. Fails fast if the profile's index is empty.
  * `impactracer evaluate --dataset DIR [--mode full|retrieval-only|propagate-only] [--cache-from DIR] [--output DIR] [--profile NAME] [--verify-nfr]` — ablation harness. Produces `per_cr_per_variant_metrics.csv`, `summary_table.csv/md`, `statistical_tests.json`, `calibration_analysis.md`, and (with `--verify-nfr`) `nfr_verification.json`. Split-run: `retrieval-only` writes a reusable cache; `propagate-only` resumes it for V6/V7 without re-spending LLM on V0–V5.
  * `impactracer report [--output PATH] [--profile NAME]` — diagnostic indexing-quality report.
* **Persistent state files** (under the active profile's root `data/<profile>/`):
  * `impactracer.db` — SQLite (`code_nodes`, `structural_edges`, `doc_code_candidates`, `file_hashes`, `file_dependencies`, `index_metadata`).
  * `chroma_store/` — ChromaDB (`code_units`, `doc_chunks`).
  * `llm_audit.jsonl` — append-only LLM audit log (NFR-05 source; per-profile).

---

*End of analysis_implementation.md. Design specification in `master_blueprint.md`. Offline-indexer detail in `index_implementation.md`. Evaluation-corpus comparison in `INDEX_REPORT.md`.*
