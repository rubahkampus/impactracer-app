# Online Analysis Pipeline — Source of Truth

> **Status:** FROZEN as of the Crucible E2E Stress Test (5 of 5 CRs PASS all 5
> quality criteria). This document is the absolute reference for the online
> Change Impact Analysis pipeline. Any change to the runtime behaviour
> below MUST update this file.
>
> Companion: `master_blueprint.md` is the design specification.
> `implementation_report.md` is the append-only sprint memory.

---

## 1. Pipeline Architecture — Nine Steps

The online pipeline transforms an Indonesian/English Change Request (CR) into a
structured `ImpactReport` with deterministic entity-level rows and LLM-assisted
file-level summaries. It is orchestrated by
`impactracer/pipeline/runner.py:run_analysis`. Variant flags (V0..V7) toggle
each LLM call independently for the ablation study; the description below is V7.

```
              CR text (Indonesian or English)
                       │
   Step 1   ───────────▼──────────────────────  LLM #1 — interpret_cr
   (always-on)         CRInterpretation: is_actionable, change_type,
                       affected_layers, primary_intent, domain_concepts,
                       search_queries (EN), named_entry_points,
                       out_of_scope_operations, is_nfr.
                       │
                       │   Coherence soft-fix (Crucible Fix 14):
                       │     DELETION ⇒ ensure 'code' in affected_layers
                       │     ADDITION ⇒ ensure not code-only
                       ▼
   Step 2   ─── Adaptive RRF Hybrid Search ─────────────────────────────
   (V0+)               4 ranked lists fused per change_type:
                         • dense_doc   (BGE-M3 embedding × ChromaDB)
                         • bm25_doc    (rank_bm25 over chunked SRS/SDD)
                         • dense_code  (BGE-M3 embedding × ChromaDB)
                         • bm25_code   (rank_bm25 over code embed_text)
                       BM25 tokenizer: camelCase split, len≥2, EN+ID
                       stop-word list (Crucible Fix 5).
                       Output: top-K RRF pool (~50 candidates).
                       │
                       ▼
   Step 3   ─── Cross-Encoder Rerank (V3+) ────────────────────────────
                       BGE-Reranker-v2-m3, multi-query MAX scoring.
                       Output: top-15 admitted seeds.
                       Post-rerank score adjustments:
                         + Traceability bonus  (+0.10) for code candidates
                           found in any retrieved doc chunk's offline
                           doc_code_candidates row (Crucible Fix 12.2).
                         − Negative filter (additive −5.0) for candidates
                           whose name/snippet contains an out-of-scope
                           operation (Crucible Fix 13). Additive on the
                           cross-encoder logit so it works correctly across
                           positive AND negative scores.
                       │
                       ▼
   Step 3.5/3.6/3.7 ─── Pre-validation Gates (V3.5+) ──────────────────
                       3.5  Score floor (sanity-only, default −2.0).
                            Real precision is enforced by LLM #2; the
                            floor exists only to drop catastrophically
                            broken candidates (Crucible Fix 9).
                       3.6  Semantic dedup. Doc chunks whose top-1
                            code resolution is already in the pool are
                            collapsed into the code candidate;
                            (section_title, text) attached as
                            "Business Context" for the LLM #2 prompt.
                       3.7  Plausibility (density-only, threshold 0.50).
                            If a single file accounts for >50% of code
                            candidates, drop those candidates UNLESS
                            their name matches a named_entry_point.
                            The previous max_per_file=2 cap is removed
                            (Crucible Fix 9).
                       │
                       ▼
   Step 4   ─── LLM #2  validate_sis (V4+) ───────────────────────────
                       Batched ≤5 candidates per call.
                       Per-node fail-CLOSED, batch-level fail-CLOSED:
                         • missing verdict  → DROP that node
                         • exception        → DROP entire batch,
                                              continue with next batch
                                              (degraded_run flag set).
                       Captures verdict.{function_purpose,
                       mechanism_of_impact, justification} for each
                       admitted seed → propagated to NodeTrace.
                       │
                       ▼
   Step 5   ─── Doc → Code Resolution (V5+) ──────────────────────────
                       Direct code seeds pass through unchanged.
                       Doc-chunk seeds are looked up in
                       doc_code_candidates (offline traceability matrix);
                       top-K code candidates per doc are emitted as
                       resolution pairs.
                       │
                       ▼
   Step 5b  ─── LLM #3  validate_trace (V5+) ─────────────────────────
                       Batched ≤5 (doc, code) pairs per call.
                       Per-pair fail-CLOSED → REJECTED on missing
                       verdict; batch-level fail-CLOSED.
                       Each surviving code_id keeps the BEST decision's
                       justification (CONFIRMED > PARTIAL > REJECTED).
                       │
                       ▼
   Step 6   ─── BFS Propagation (V6+) ────────────────────────────────
                       Multi-seed BFS over the structural graph using
                       per-edge depth/direction rules (see §3 below).
                       Confidence tiers: top-N seeds by reranker score
                       are "high-conf"; low-conf seeds cap CALLS at
                       depth 1.
                       │
                       ▼
   Step 6.5 ─── CONTAINS Sub-Tree Collapse ──────────────────────────
                       InterfaceField nodes that reach the CIS via
                       CONTAINS-only paths are collapsed into their
                       parent Interface's NodeTrace.collapsed_children.
                       Reduces token cost without losing information.
                       │
                       ▼
   Step 7   ─── LLM #4  validate_propagation (V6.5+) ────────────────
                       Batched ≤5 propagated nodes per call.
                       Per-node fail-CLOSED, batch-level fail-CLOSED.
                       Auto-exempt: depth-1 IMPLEMENTS / DEFINES_METHOD
                       / TYPED_BY edges skip the LLM call (direct
                       structural contracts always kept; synthetic
                       justification = "Direct <edge> contract from
                       <seed> — auto-admitted exempt edge.").
                       Prompt is DE-BLINDED (Crucible Fix 2): the
                       causal chain is shown as factual context;
                       anti-tautology language forbids edge-type-as-
                       evidence reasoning.
                       Per-child collapsed validation: each surviving
                       parent's collapsed_children are individually
                       re-validated by an LLM #4-style call.
                       │
                       ▼
   Step 8   ─── Context Build ─────────────────────────────────────────
                       Backlinks (bidirectional doc↔code via the
                       offline traceability matrix), source snippets
                       (ILA preferred for Function/Method nodes), and
                       a CANONICAL FILE LIST in a "=== IMPACTED FILES
                       ===" header so LLM #5 knows exactly which files
                       it must produce file_justifications for.
                       Token-budget truncation is severity-aware:
                       (severity_rank ASC, depth ASC, node_id) — Tinggi
                       at depth 3 outranks Rendah at depth 2
                       (Crucible Fix 7). Truncation is DECOUPLED from
                       the output (Crucible Amendment 2): only the
                       LLM #5 prompt is trimmed; the report's
                       impacted_entities list always contains the FULL
                       validated CIS.
                       │
                       ▼
   Step 9   ─── LLM #5  synthesize (always-on) ──────────────────────
                       Aggregator-only role (Crucible Fix 3).
                       Input:  the truncated context + the canonical
                                file set.
                       Output: LLMSynthesisOutput =
                         { executive_summary,
                           documentation_conflicts,
                           file_justifications: list of
                                {file_path, justification} }.
                       The runner reconciles file_justifications
                       against the deterministic file set: hallucinated
                       files are dropped, omitted files get a
                       deterministic fallback summary derived from the
                       entity-level justifications already in that file.
                       │
                       ▼
              ImpactReport
              ├── executive_summary           (LLM #5)
              ├── impacted_files              (deterministic file_path
              │                                set + LLM #5 justification
              │                                or deterministic fallback)
              ├── impacted_entities           (deterministic; every
              │                                validated CIS node;
              │                                justification verbatim
              │                                from LLM #2/#3/#4 or
              │                                synthetic auto_exempt)
              ├── documentation_conflicts     (LLM #5)
              ├── estimated_scope             (deterministic from CIS size)
              ├── analysis_mode               ('retrieval_only' or
              │                                'retrieval_plus_propagation')
              └── degraded_run                (true if any LLM batch
                                               was dropped)
```

The runner additionally writes a per-step trace to `impact_report_full.json`
when `trace_sink` is provided (always populated by the CLI). Trace keys:
`step_1_interpretation`, `step_2_rrf_pool`, `step_3_reranked`,
`step_3_gates_survivors`, `step_4_llm2_verdicts`, `step_5_resolutions`,
`step_5b_llm3_verdicts`, `step_6_bfs_raw_cis`, `step_7_llm4_verdicts`,
`final_report`. Steps 5b/6/7 may be absent for variants that disable those
phases or for CRs that resolve to zero seeds before the corresponding stage.

---

## 2. The Five LLM Invocations — Roles, Prompts, Fail-Closed Logic

### LLM #1 — Interpret (`interpret_cr`)

* **Role:** parse the CR into a structured `CRInterpretation`. Single call,
  always-on, schema-constrained.
* **Output schema:** nine attributes (see `shared/models.py:CRInterpretation`)
  including `is_actionable`, `change_type` ∈ {ADDITION, MODIFICATION,
  DELETION}, `affected_layers`, `search_queries` (in English even when CR is
  Indonesian), `named_entry_points`, `out_of_scope_operations`, and the
  Crucible-added `is_nfr` flag.
* **Fail-closed:** Pydantic ValidationError on the response halts the run
  (rejection report). The `is_actionable=false` branch short-circuits to a
  minimal rejection report with no downstream calls.
* **Distributed Justification role:** none. LLM #1 produces metadata; it
  does not validate any node.

### LLM #2 — Validate SIS (`validate_sis`)

* **Role:** judge whether each retrieved candidate is DIRECTLY impacted by
  the CR. Operates on the cross-encoder rerank survivors (top-15 after
  gates). Batches of ≤5 candidates per call.
* **Prompt constraints** (`pipeline/validator.py:_SYSTEM_PROMPT`):
  * No retrieval scores in the prompt — anti-circular mandate.
  * Distinguishes code-node vs doc-chunk verdict criteria.
  * Forces concrete `mechanism_of_impact` ("vague 'related' justifications
    forbidden"). This text is the seed-level Distributed Justification
    propagated verbatim to `ImpactedEntity.justification` with
    `justification_source="llm2_sis"`.
  * Delimiter contract: copy node_id from BETWEEN
    `<<NODE_ID_START>>...<<NODE_ID_END>>`, do NOT include the markers in
    the JSON output. The runner sanitizes any leftover markers before
    lookup (Crucible E2E Task 1).
* **Fail-closed (Crucible Fix 1, FF-1):**
  * **Per node:** missing verdict → DROP that candidate. No silent
    admission.
  * **Per batch:** any uncaught exception (after retries exhaust in
    `LLMClient.call`) is caught by `validator.py`, the batch is recorded
    as DROPPED, `degraded=True` is set, and the loop continues with the
    next batch (Crucible Amendment 1: pipeline always finishes).
* **Captures:** `function_purpose`, `mechanism_of_impact`, `justification`
  for every confirmed seed → attached to `NodeTrace`.

### LLM #3 — Validate Trace (`validate_trace`)

* **Role:** for each `(doc_chunk, code_node)` pair produced by Step 5
  resolution, decide CONFIRMED / PARTIAL / REJECTED. Batches of ≤5 pairs
  per call.
* **Prompt constraints**
  (`pipeline/traceability_validator.py:_SYSTEM_PROMPT`):
  * Judge by AST structure and document semantics — never by score.
  * REJECTED requires a substantive feature-area mismatch; vocabulary
    overlap alone is not enough.
  * For ADDITION CRs: absence of current implementation does NOT mean
    REJECTED.
  * Delimiter contract identical to LLM #2; sanitization applied to both
    `doc_chunk_id` and `code_node_id`.
* **Fail-closed (Crucible Fix 1, FF-2):**
  * **Per pair:** missing verdict → REJECTED.
  * **Per batch:** exception → all pairs in that batch REJECTED, continue.
* **Captures:** the verdict justification of the BEST decision per code_id
  (CONFIRMED > PARTIAL > REJECTED) → propagated to
  `NodeTrace.justification` with `justification_source="llm3_trace"`.

### LLM #4 — Validate Propagation (`validate_propagation`)

* **Role:** for each BFS-propagated node that is NOT auto-exempt, decide
  whether the structural reach implies semantic impact.
* **Prompt constraints**
  (`pipeline/traversal_validator.py:_SYSTEM_PROMPT`):
  * **De-blinded chain** (Crucible Fix 2): the causal chain IS shown as
    factual context. Tautology safety is enforced by explicit anti-
    tautology language: *"Edge types are NOT impact evidence … Reject any
    node where the relationship is structurally present but the target's
    behaviour is unaffected by the CR."*
  * Forbidden justification patterns enumerated in the prompt
    ("function A calls function B" / "in the same module" / generic
    relation strings are explicitly rejected).
  * Required justification format demands a contract-breakage, behavioral-
    anomaly, or downstream-type-mismatch sentence.
  * Determinism: `random.seed(42); random.shuffle(to_validate)` before
    batching to neutralize positional bias without compromising
    reproducibility.
  * Delimiter contract + sanitization identical to LLM #2.
* **Auto-exempt edges:** depth-1 `IMPLEMENTS`, `DEFINES_METHOD`, `TYPED_BY`
  bypass the LLM call entirely. They receive a synthetic justification
  `"Direct <edge> contract from <seed> — auto-admitted exempt edge."` with
  `justification_source="auto_exempt"`. These are direct structural
  contracts whose impact is definitional, not stochastic.
* **Per-child collapse validation (Crucible Phase 2.2):** for each
  surviving parent with `collapsed_children`, an additional LLM #4-style
  call individually validates each child name. Same fail-closed semantics.
* **Fail-closed (Crucible Fix 1, FF-3):**
  * **Per node:** missing verdict → DROP.
  * **Per batch:** exception → DROP entire batch, continue.
  * **Per child batch:** exception → DROP entire child batch, continue.
* **Captures:** verdict justification → propagated to
  `NodeTrace.justification` with `justification_source="llm4_propagation"`.

### LLM #5 — Synthesize (`synthesize_summary`)

* **Role (FULLY DEMOTED, Crucible Fix 3):** aggregator only. Produces the
  executive summary, documentation conflicts list, and **per-file**
  justifications. NEVER produces per-entity justifications.
* **Distributed Justification Principle:**
  * `impacted_entities` is built deterministically by the runner from
    the validated CIS. Each entity's `justification` is propagated
    VERBATIM from the LLM (#2/#3/#4) that admitted it, or from the
    synthetic auto_exempt string. LLM #5 never sees nor authors these.
  * `impacted_files` is also deterministic with respect to its file_path
    set: every distinct file referenced by `impacted_entities` MUST have
    exactly one row in `impacted_files`. The `justification` field of
    each file row may be written by LLM #5 (because file-level
    summarization is by definition a summarization task, not a per-entity
    validation task). If LLM #5 omits a file or hallucinates one, the
    runner reconciles: hallucinated files dropped silently; omitted files
    receive a deterministic fallback that summarizes the entity-level
    justifications inside that file.
* **Prompt constraints** (`pipeline/synthesizer.py:SYSTEM_PROMPT`):
  * Explicit "DO NOT output an impacted_entities array" instruction.
  * Explicit "the runner builds entity-level rows" instruction.
  * One justification per file in the canonical "=== IMPACTED FILES ==="
    list shown in the user message.
* **Fail-closed (Crucible Amendment 1):** if `LLMClient.call` raises after
  retry exhaustion, the runner falls back to `build_minimal_summary`
  (deterministic fallback summary; `degraded_run=True`). The
  deterministic `impacted_entities` and `impacted_files` lists are still
  emitted — they exist independently of LLM #5.

---

## 3. Graph Constraints — BFS Propagation Rules

The structural graph is a `networkx.MultiDiGraph` materialized once per
pipeline context from the SQLite `structural_edges` table. BFS propagation
is governed by `EDGE_CONFIG` in `impactracer/shared/constants.py`. The
following rules are inviolable as of the Crucible refactor.

### 3.1 Per-edge direction & max_depth

| Edge type | Direction | Max depth | Rationale |
|---|---|---|---|
| `CALLS` | reverse | **2** (was 3) | Depth-3 fan-in regularly produces 200+ propagated nodes per seed in TS codebases. Depth-2 is the precision-recovery sweet spot (Crucible Fix 6 / AV-5). |
| `INHERITS` | reverse | 3 | Class hierarchies are typically shallow; 3 hops covers all real cases. |
| `IMPLEMENTS` | reverse | 3 | Interface contract graph. |
| `TYPED_BY` | reverse | 3 | Type-reference propagation. |
| `FIELDS_ACCESSED` | reverse | 2 | Field-level access has higher fan-out than method calls. |
| `DEFINES_METHOD` | forward | 1 | Definitional containment, not semantic propagation (Crucible Phase 1.2). |
| `PASSES_CALLBACK` | forward | 1 | |
| `HOOK_DEPENDS_ON` | reverse | 1 | React hook dependency edge. |
| `IMPORTS` | reverse | 1 | Module composition; no transitive impact assumed. |
| `RENDERS` | reverse | 1 | |
| `DEPENDS_ON_EXTERNAL` | reverse | 1 | |
| `CLIENT_API_CALLS` | reverse | 1 | |
| `DYNAMIC_IMPORT` | reverse | 1 | |
| `CONTAINS` | reverse | 1 | File ↔ symbol containment. Reverse-only: given a changed symbol, find which files contain it; do NOT enumerate sibling symbols (Crucible Phase 1.1). |

### 3.2 Confidence-tier CALLS cap

* `LOW_CONF_CAPPED_EDGES = {CALLS}`: low-confidence seeds (i.e., not in
  the top-N reranker scores AND not directly retrieved) cap CALLS depth
  to 1. Prevents low-quality seeds from emitting deep propagation chains.

### 3.3 Hub mitigation

* `_HUB_DEGREE_THRESHOLD = 20`: nodes whose total degree > 20 (typical for
  generic interfaces, framework primitives like `ext::react`) are capped
  at depth 1 for ALL edges when traversing FROM them. Prevents
  combinatorial explosion through framework hubs.

### 3.4 UTILITY-file CALLS cutoff (Crucible Fix 11.1)

* Seeds whose file_classification is `UTILITY` cap their reverse-CALLS
  chain at `UTILITY_FILE_CALLS_DEPTH_CAP = 1`. Utility functions are
  called from everywhere; deeper reverse-CALLS from a UTILITY seed is a
  near-guaranteed flood across unrelated features.

### 3.5 Per-node-type fan-in cap (Crucible Fix 11.2)

* `NODE_TYPE_MAX_FAN_IN`: a propagated neighbour with in-degree exceeding
  the type-specific cap is excluded from the CIS unless it is itself a
  SIS seed.

| Node type | Max fan-in |
|---|---|
| Function / Method / Class | 50 |
| Interface / TypeAlias / Enum | 100 |
| InterfaceField | 200 |
| File | 200 |
| ExternalPackage | 0 (always excluded — see 3.6) |

### 3.6 Excluded-type wholesale (Crucible Fix 11)

* `EXCLUDED_PROPAGATION_NODE_TYPES = {ExternalPackage}`. Third-party
  package nodes are dependency edges' targets, not units of impact;
  the pipeline observes `DEPENDS_ON_EXTERNAL` edges but never adds the
  external package itself to the propagated set.

### 3.7 Severity (last-hop rule, Crucible Phase 1.3)

* `severity_for_chain(causal_chain)` returns the severity of the LAST
  edge in the chain. SIS seeds (empty chain) are `Tinggi` by convention.
  This eliminates "severity laundering" where a chain like
  `CALLS → CALLS → IMPLEMENTS` would inherit the final IMPLEMENTS's
  Tinggi severity from an otherwise speculative path.

### 3.8 CONTAINS sub-tree collapse (Step 6.5)

* After BFS, parent nodes whose CONTAINS-only children are in the CIS
  receive those children's IDs in `NodeTrace.collapsed_children`; the
  children are removed from `propagated_nodes` to avoid token explosion.
  Each surviving collapsed child is individually re-validated by LLM #4
  (Crucible Phase 2.2), so this is a token-economy optimisation, not a
  short-circuit through validation.

### 3.9 Graph isolation invariant

* `bfs_propagate` does NOT mutate the shared graph (Crucible Phase 1.6).
  Sequential ablation runs (V0→V7 over the same CR) produce identical
  CIS results. Seeds absent from the graph are recorded as SIS-only
  terminal nodes (no expansion) but never inserted into the graph.

---

## 4. Attrition Topography — Empirical (Crucible E2E, 5 CRs)

Numbers are per-CR pipeline attrition counts collected from
`impact_report_full.json`. The pipeline was run against the indexed
`citrakara` repository (~400 code nodes, ~2,000 InterfaceField nodes) at
variant V7 with `google/gemini-2.5-flash-lite` as the LLM.

| CR | RRF | rerank | gates | LLM #2 | resolutions | LLM #3 | BFS prop. | LLM #4 | final entities |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CR-PIN-LISTING | 46 | 15 | 13 | 11 | 19 | 3 | 57 | 17 | 28 |
| CR-GRACE-DAYS | 49 | 15 | 9 | 6 | 14 | 3 | 48 | 12 | 19 |
| CR-INTERNAL-NOTES | 41 | 15 | 15 | 9 | 21 | 2 | 45 | 7 | 15 |
| CR-ESCROW-REFACTOR | 35 | 15 | 8 | 5 | 5 | 0 | 7 | 1 | 6 |
| CR-DASHBOARD-TANSTACK | 36 | 15 | 15 | 2 | 2 | 0 | 4 | 0 | 2 |
| **average** | **41.4** | **15.0** | **12.0** | **6.6** | **12.2** | **1.6** | **32.2** | **7.4** | **14.0** |

**Reading the funnel (averages):**

* RRF pool ≈ 41 candidates feeds the cross-encoder.
* Cross-encoder reranks down to a fixed top-15.
* Pre-validation gates (3.5/3.6/3.7) drop ≈ 3 candidates per CR.
* LLM #2 confirms ≈ 7 of 12 (~55% acceptance rate).
* Doc → code resolution expands to ≈ 12 pairs that LLM #3 prunes hard
  (~90% rejection — most doc resolutions are not impactful seeds).
* BFS adds ≈ 32 propagated nodes around the validated seeds.
* LLM #4 admits ≈ 7 of 32 (~23% acceptance rate). The other ~75% are the
  documented "graph flood": structurally reachable but semantically
  unrelated. LLM #4 is the precision-recovery valve.
* Final report ≈ 14 entities (median spread 2–28 depending on CR scope).

**Quality observations across the 5 runs:**

1. **No recall monsters.** The largest report (CR-PIN-LISTING, 28 entities,
   14 files) is justified by the CR's explicit cross-cutting nature
   ("model + service + component"). The smallest (CR-DASHBOARD-TANSTACK,
   2 entities, 2 files) reflects a tightly-localized refactor, exactly as
   the CR text suggests.
2. **No missed targets.** Every CR-named identifier (`ticket.model.ts`,
   `ticket.service.ts`, `CancelTicketDetails.tsx`, `AdminResolutionForm.tsx`,
   `escrowTransaction.repository.ts`, `escrowTransaction.service.ts`,
   `DashboardGalleryPage.tsx`, `commissionListing.model.ts`,
   `updateContractDeadline`) appears in either the entity set or the file
   set — verified by `tools/e2e_test.py` substring presence checks.
3. **Justification source distribution** is healthy:
   * `llm2_sis`: SIS-seed mechanisms (most concrete, model-level claims).
   * `llm3_trace`: doc → code structural grounding.
   * `llm4_propagation`: structural-contract-breakage rationales for
     propagated nodes.
   * `auto_exempt`: depth-1 IMPLEMENTS/TYPED_BY/DEFINES_METHOD edges.
   * `bfs_only`: appears only in V6-style runs without LLM #4.
   No entity in any of the 5 reports has an empty justification.
4. **Fail-closed integrity preserved.** Across the 5 CRs (≈ 50 LLM calls
   total), `degraded_run` was `false` in every report. The fail-closed
   guard rails are wired but did not need to fire on this run.
5. **Schema compliance is exact.** Every report has both
   `impacted_entities` and `impacted_files`, every file_path in
   `impacted_entities` has a corresponding `impacted_files` row, every
   `impact_report_full.json` contains the documented step keys.

---

## 5. Frozen Invariants

The following architectural invariants are FROZEN. Violating any of them
requires updating this document AND `master_blueprint.md`.

1. **9 node types**: `File, Class, Function, Method, Interface,
   TypeAlias, Enum, ExternalPackage, InterfaceField`.
2. **13 structural edge types**: see §3.1 above.
3. **5 LLM invocations in V7**: `interpret`, `validate_sis`,
   `validate_trace`, `validate_propagation`, `synthesize`. (Plus an
   internal child-validation call inside Step 7 that is parameterized
   from the same LLM #4 prompt; it does not count as a 6th distinct
   stage.)
4. **3 change_type values**: `ADDITION`, `MODIFICATION`, `DELETION`.
5. **Fail-CLOSED at every validator.** Both per-item (drop on missing
   verdict) and per-batch (drop on exception, continue) at LLM #2, #3,
   and #4. The runner annotates the report with `degraded_run=true` when
   any drop fires.
6. **Distributed Justification Principle.** Entity-level justifications
   come VERBATIM from LLM #2, LLM #3, LLM #4, or a synthetic auto_exempt
   string. LLM #5 never re-justifies entities. File-level justifications
   may be authored by LLM #5 because file summarization is
   summarization, not validation.
7. **Truncation decoupled from output.** The LLM #5 prompt may be
   truncated to fit the token budget; the report's `impacted_entities`
   list always contains the FULL validated CIS regardless.
8. **CALLS reverse depth = 2.** Combined with the UTILITY-CALLS cutoff
   and the per-node-type fan-in cap, this is the structural defence
   against graph flood.
9. **Negative filter is ADDITIVE (−5.0 on the cross-encoder logit).**
   A multiplicative penalty would invert sign on negative logits and
   inadvertently promote out-of-scope candidates.
10. **The Wilcoxon test target is `f1_set` (Total F1, set-level).**
    Bounded F1@K is removed from the codebase (Crucible Fix 4) because
    it cannot detect graph floods.

---

## 6. Operational Surfaces

* **CLI:**
  * `impactracer index <repo>` — offline indexer.
  * `impactracer analyze "<CR text>" --variant V7 [--output PATH]` —
    online analysis. Always writes both `impact_report.json` and the
    sibling `impact_report_full.json`.
  * `impactracer evaluate --dataset gt.json --output dir/` — ablation
    harness (V0..V7 × all CRs in the GT dataset; produces
    `per_cr_per_variant_metrics.csv` and the Wilcoxon test result on
    `f1_set`).
* **Diagnostic tools:**
  * `python tools/diagnose_pipeline.py --cr-text "..."` — V0..V7 attrition
    table for a single CR.
  * `python tools/e2e_test.py [--cr-id ID]` — runs the 5-CR Crucible E2E
    stress test and grades each report against the 5 quality criteria.
* **Persistent state files:**
  * `data/impactracer.db` — SQLite (code_nodes, structural_edges,
    doc_code_candidates, file_hashes, file_dependencies,
    index_metadata).
  * `data/chroma_store/` — ChromaDB (code_units, doc_chunks).
  * `data/llm_audit.jsonl` — append-only LLM audit log (NFR-05).
  * `data/locked_parameters.json` — frozen calibration values.

---

*End of source-of-truth document. Last updated: Crucible E2E
Stress Test, all 5 CRs PASS all 5 quality criteria.*
