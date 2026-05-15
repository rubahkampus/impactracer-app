# Online Analysis Pipeline — Source of Truth

> Operational reference for the V7 online pipeline as it currently runs.
> Every claim here either quotes a constant from `shared/constants.py` /
> `shared/config.py` or cites an empirical number from the canonical
> calibration run `eval/results_apex_v6/` (Sprint 16, anchor mechanism
> gate: V7 entity F1 = 0.263, V7 file F1 = 0.408 — the +31%/+8% lift
> over the Sprint 13-W2 baseline of 0.200 / 0.377 in `eval/results_v2/`).
> The prior canonical reference, Sprint 14 V4 (`eval/results_apex_v4/`,
> V7 entity F1 = 0.252), is preserved as the "Apex A+B before anchor
> gating" reference run.
>
> Companion: `master_blueprint.md` is the design specification.
> `index_implementation.md` is the offline-indexer operational reference.
> `implementation_report.md` is the append-only sprint memory.
>
> Sprint 14 (Apex Crucible Proposals A+B) and Sprint 15 (Proposal C, default-
> disabled) augment but do not replace this contract. Sprint 16 anchor-
> gating for sibling promotion is documented inline in §2 (LLM #4 sub-stage).

---

## 1. Pipeline Architecture — Nine Steps

The online pipeline transforms an Indonesian / English Change Request (CR) into a structured `ImpactReport` with deterministic entity-level rows and LLM-assisted file-level summaries. It is orchestrated by `impactracer/pipeline/runner.py::run_analysis`. Variant flags (V0..V7) toggle each LLM call independently for the ablation study; the description below is V7.

```
              CR text (Indonesian or English)
                       │
   Step 1   ───────────▼──────────────────────  LLM #1 — interpret_cr
   (always-on)         CRInterpretation: is_actionable, change_type,
                       affected_layers, primary_intent, domain_concepts,
                       search_queries (EN), named_entry_points,
                       out_of_scope_operations, is_nfr.
                       │
                       │   Coherence soft-fix:
                       │     DELETION ⇒ ensure 'code' in affected_layers
                       │     ADDITION ⇒ ensure not code-only
                       ▼
   Step 2   ─── Adaptive RRF Hybrid Search ─────────────────────────────
   (V0+)               Four ranked lists fused per change_type:
                         • dense_doc   (BGE-M3 embedding × ChromaDB)
                         • bm25_doc    (rank_bm25 over chunked SRS/SDD)
                         • dense_code  (BGE-M3 embedding × ChromaDB)
                         • bm25_code   (rank_bm25 over code embed_text)

                       dense_code additionally incorporates:
                         (a) RAW-CR multilingual dense pass — one extra
                             dense query against code_units using the raw
                             (Indonesian or mixed) CR text. BGE-M3 bridges
                             the Indonesian-CR ↔ English-identifier gap
                             without going through LLM #1's English
                             search queries.
                         (b) Traceability pool seeding — code nodes that
                             the offline doc_code_candidates table links
                             (≥ 0.40, ≤ 5 per doc) to any retrieved
                             doc-chunk are injected into dense_code with
                             a synthetic rank.

                       BM25 tokenizer: camelCase split, len ≥ 2, EN + ID
                       stop-word list.
                       Output: top-K RRF pool (top_k_rrf_pool = 200).
                       │
                       ▼
   Step 3   ─── Cross-Encoder Rerank (V3+) ────────────────────────────
                       BGE-Reranker-v2-m3, multi-query MAX scoring.
                       Output: top-15 admitted seeds.
                       Post-rerank score adjustments:
                         + Traceability bonus (+0.10) for code candidates
                           in any retrieved doc-chunk's offline
                           doc_code_candidates row.
                         − Negative filter (additive −5.0) for candidates
                           whose name/snippet contains an out-of-scope
                           operation. Additive on the cross-encoder
                           logit so it works correctly across positive
                           AND negative scores.
                       │
                       ▼
   Step 3.5/3.6/3.7 ─── Pre-validation Gates (V3+) ───────────────────
                       3.5  Score floor (sanity-only, default −2.0).
                            Real precision is enforced by LLM #2; the
                            floor exists only to drop catastrophically
                            broken candidates.
                       3.6  Semantic dedup. Doc chunks whose top-1 code
                            resolution is already in the pool are
                            collapsed into the code candidate;
                            (section_title, text) attached as
                            "Business Context" for the LLM #2 prompt.
                       3.7  Plausibility (density-only, threshold 0.50).
                            If a single file accounts for > 50 % of code
                            candidates, drop those candidates UNLESS
                            their name matches a named_entry_point.
                            No per-file count cap.
                       │
                       ▼
   Step 4   ─── LLM #2  validate_sis (V4+) ───────────────────────────
                       Batched ≤ 5 candidates per call.
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
                       Batched ≤ 5 (doc, code) pairs per call.
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
   Step 7   ─── LLM #4  validate_propagation (V7) ───────────────────
                       Batched ≤ 5 propagated nodes per call.
                       Per-node fail-CLOSED, batch-level fail-CLOSED.
                       Auto-exempt: depth-1 IMPLEMENTS / DEFINES_METHOD
                       edges skip the LLM call (TYPED_BY removed in
                       Sprint 14 — see §2 LLM #4 entry). Synthetic
                       justification = "Direct <edge> contract from
                       <seed> — auto-admitted exempt edge.".
                       Prompt is DE-BLINDED: the causal chain is shown
                       as factual context; anti-tautology language
                       forbids edge-type-as-evidence reasoning.
                       Per-child collapsed validation: each surviving
                       parent's collapsed_children are individually
                       re-validated by an LLM-#4-style call.
                       │
                       ▼
   Step 7.5 ─── Sibling Promotion (Sprint 14 Apex Crucible A) ───────
                       One LLM-#4-style call per file with a qualifying
                       anchor. Anchor = node with non-empty LLM #2
                       mechanism_of_impact (Sprint 16 — Option 1).
                       For each anchor's file, enumerate qualified
                       same-file siblings via CONTAINS and admit up to
                       4 per file by LLM-#4 sibling-batch verdict.
                       Admitted siblings become propagated_nodes with
                       causal_chain=["CONTAINS"] and
                       justification_source="llm4_sibling".
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
                       at depth 3 outranks Rendah at depth 2.
                       Truncation is DECOUPLED from the output: only
                       the LLM #5 prompt is trimmed; the report's
                       impacted_entities list always contains the FULL
                       validated CIS.
                       │
                       ▼
   Step 9   ─── LLM #5  synthesize (always-on) ──────────────────────
                       Aggregator-only role.
                       Input:  the truncated context + the canonical
                                file set.
                       Output: LLMSynthesisOutput =
                         { executive_summary,
                           documentation_conflicts,
                           file_justifications: list of
                                {file_path, justification} }.
                       The runner reconciles file_justifications
                       against the deterministic file set: hallucinated
                       files are dropped silently; omitted files
                       receive a deterministic fallback summarizing the
                       entity-level justifications inside that file.
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

The runner additionally writes a per-step trace to `impact_report_full.json` when `trace_sink` is provided (always populated by the CLI). Trace keys:

```
step_1_interpretation, step_2_rrf_pool, step_3_reranked,
step_3_gates_survivors, step_4_llm2_verdicts, step_5_resolutions,
step_5b_llm3_verdicts, step_6_bfs_raw_cis, step_7_llm4_verdicts,
step_7p5_sibling_promotion, final_report
```

Steps 5b / 6 / 7 / 7p5 may be absent for variants that disable those phases or for CRs that resolve to zero seeds before the corresponding stage. `step_7p5_sibling_promotion` is present only for V7 with `enable_sibling_promotion=True` and at least one qualifying anchor (i.e. at least one SIS seed with non-empty LLM #2 `mechanism_of_impact`).

---

## 2. The Five LLM Invocations — Roles, Prompts, Fail-Closed Logic

### LLM #1 — Interpret (`interpret_cr`)

* **Role:** parse the CR into a structured `CRInterpretation`. Single call, always-on, schema-constrained.
* **Output schema** (`shared/models.py::CRInterpretation`): `is_actionable`, `actionability_reason`, `primary_intent`, `change_type` ∈ {ADDITION, MODIFICATION, DELETION}, `affected_layers ⊆ {requirement, design, code}`, `domain_concepts`, `search_queries` (English even when CR is Indonesian), `named_entry_points`, `out_of_scope_operations`, `is_nfr`.
* **Fail-closed:** a Pydantic ValidationError on the response halts the run with a rejection report. The `is_actionable=False` branch short-circuits to a minimal rejection report with no downstream calls.
* **Distributed Justification role:** none. LLM #1 produces metadata; it does not validate any node.

### LLM #2 — Validate SIS (`validate_sis`)

* **Role:** judge whether each retrieved candidate is DIRECTLY impacted by the CR. Operates on the cross-encoder rerank survivors (top-15 after gates). Batches of ≤ 5 candidates per call.
* **Prompt constraints** (`pipeline/validator.py::_SYSTEM_PROMPT`):
  * No retrieval scores in the prompt — anti-circular mandate.
  * Distinguishes code-node vs doc-chunk verdict criteria.
  * Forces concrete `mechanism_of_impact` ("vague 'related' justifications forbidden"). This text becomes the seed-level Distributed Justification propagated verbatim to `ImpactedEntity.justification` with `justification_source="llm2_sis"`.
  * Delimiter contract: copy `node_id` from BETWEEN `<<NODE_ID_START>>...<<NODE_ID_END>>`, do NOT include the markers in the JSON output. The runner sanitises any leftover markers before lookup.
* **Fail-closed:**
  * **Per node:** missing verdict → DROP that candidate. No silent admission.
  * **Per batch:** any uncaught exception (after retries exhaust in `LLMClient.call`) is caught by `validator.py`, the batch is recorded as DROPPED, `degraded=True` is set, and the loop continues with the next batch.
* **Captures:** `function_purpose`, `mechanism_of_impact`, `justification` for every confirmed seed → attached to `NodeTrace`.

### LLM #3 — Validate Trace (`validate_trace`)

* **Role:** for each `(doc_chunk, code_node)` pair produced by Step 5 resolution, decide CONFIRMED / PARTIAL / REJECTED. Batches of ≤ 5 pairs per call.
* **Prompt constraints** (`pipeline/traceability_validator.py::_SYSTEM_PROMPT`):
  * Judge by AST structure and document semantics — never by score.
  * REJECTED requires a substantive feature-area mismatch; vocabulary overlap alone is not enough.
  * For ADDITION CRs: absence of current implementation does NOT mean REJECTED.
  * Delimiter contract identical to LLM #2; sanitisation applied to both `doc_chunk_id` and `code_node_id`.
* **Fail-closed:**
  * **Per pair:** missing verdict → REJECTED.
  * **Per batch:** exception → all pairs in that batch REJECTED, continue.
* **Captures:** the verdict justification of the BEST decision per code_id (CONFIRMED > PARTIAL > REJECTED) → propagated to `NodeTrace.justification` with `justification_source="llm3_trace"`.

### LLM #4 — Validate Propagation (`validate_propagation`) + sub-stages

* **Primary role:** for each BFS-propagated node that is NOT auto-exempt, decide whether the structural reach implies semantic impact. Plus, two parameterised reuses of the same module: `validate_collapsed_children` for collapsed sub-trees, and `validate_siblings` for Step 7.5 sibling promotion.
* **Prompt constraints** (`pipeline/traversal_validator.py::_SYSTEM_PROMPT`):
  * **De-blinded chain:** the causal chain IS shown as factual context. Tautology safety is enforced by explicit anti-tautology language: *"Edge types are NOT impact evidence … Reject any node where the relationship is structurally present but the target's behaviour is unaffected by the CR."*
  * Forbidden justification patterns enumerated in the prompt ("function A calls function B" / "in the same module" / generic relation strings are explicitly rejected).
  * Required justification format demands a contract-breakage, behavioural-anomaly, or downstream-type-mismatch sentence.
  * Determinism: `random.seed(42); random.shuffle(to_validate)` before batching to neutralise positional bias without compromising reproducibility.
  * Delimiter contract + sanitisation identical to LLM #2.
* **Auto-exempt edges (Sprint 14):** depth-1 `IMPLEMENTS` and `DEFINES_METHOD` bypass the LLM call entirely. They receive a synthetic justification `"Direct <edge> contract from <seed> — auto-admitted exempt edge."` with `justification_source="auto_exempt"`. `TYPED_BY` was REMOVED from the exempt set in Sprint 14 (Apex Crucible Proposal A): forensic on CR-01 V7 showed depth-1 TYPED_BY auto-admissions were producing 10 of 28 false positives (review.model.ts, galleryPost.model.ts, proposal.model.ts admitted because they referenced `CommissionListingSchema` but did not change). LLM #4 now adjudicates depth-1 TYPED_BY on the same footing as deeper chains.
* **Per-child collapse validation (`validate_collapsed_children`):** for each surviving parent with `collapsed_children`, an additional LLM-#4-style call individually validates each child name. Same fail-closed semantics.
* **Sibling promotion sub-stage (`validate_siblings`, Sprint 14 Apex Crucible Proposal A — Step 7.5):** after primary LLM #4 completes, the runner enumerates each qualifying anchor's same-file siblings via CONTAINS and submits them to an LLM-#4-prompted sibling-batch. **Anchor qualification (Sprint 16 — Option 1):** anchor must be in `sis_justifications` AND have a non-empty `mechanism_of_impact`. Multi-anchor batch (Sprint 14 V2 fix): the prompt receives ALL qualifying anchors in the file with their justifications, so LLM #4 sees the full multi-anchor contract surface. Per-file admission cap `settings.sibling_admit_max_per_file = 4`; no per-CR global cap. Admitted siblings receive `justification_source="llm4_sibling"`.
* **Fail-closed:**
  * **Per node:** missing verdict → DROP.
  * **Per batch:** exception → DROP entire batch, continue.
  * **Per child batch:** exception → DROP entire child batch, continue.
  * **Per sibling batch:** exception → DROP entire file's sibling batch, continue.
* **Captures:** verdict justification → propagated to `NodeTrace.justification` with `justification_source="llm4_propagation"` (primary) or `"llm4_sibling"` (Step 7.5 admissions).

### LLM #5 — Synthesize (`synthesize_summary`)

* **Role (aggregator-only):** produces the executive summary, documentation-conflicts list, and **per-file** justifications. NEVER produces per-entity justifications.
* **Distributed Justification Principle:**
  * `impacted_entities` is built deterministically by the runner from the validated CIS. Each entity's `justification` is propagated VERBATIM from the LLM (#2/#3/#4) that admitted it, or from the synthetic `auto_exempt` string. LLM #5 never sees nor authors these.
  * `impacted_files` is also deterministic with respect to its `file_path` set: every distinct file referenced by `impacted_entities` MUST have exactly one row in `impacted_files`. The `justification` field of each file row may be written by LLM #5 (file-level summarisation is by definition a summarisation task, not a per-entity validation task). If LLM #5 omits a file or hallucinates one, the runner reconciles: hallucinated files dropped silently; omitted files receive a deterministic fallback that summarises the entity-level justifications inside that file.
* **File-type filter (Sprint 14 Apex Crucible Proposal A):** `build_deterministic_impacted_entities` HARD-FILTERS every CIS node whose `node_type == "File"` or whose `node_id` lacks `::`. Ground Truth's `impacted_entities` only ever contains qualified `file::symbol` ids; emitting bare File nodes was producing 24/30 V7 predictions on CR-01 as guaranteed FPs. File-level impact is preserved separately: `assemble_impact_report(..., extra_impacted_file_paths=...)` ensures every File-type CIS node's path still drives `impacted_files`.
* **Prompt constraints** (`pipeline/synthesizer.py::SYSTEM_PROMPT`):
  * Explicit "DO NOT output an `impacted_entities` array" instruction.
  * Explicit "the runner builds entity-level rows" instruction.
  * One justification per file in the canonical "=== IMPACTED FILES ===" list shown in the user message.
* **Fail-closed:** if `LLMClient.call` raises after retry exhaustion, the runner falls back to `build_minimal_summary` (deterministic fallback summary; `degraded_run=True`). The deterministic `impacted_entities` and `impacted_files` lists are still emitted — they exist independently of LLM #5.

---

## 3. Graph Constraints — BFS Propagation Rules

The structural graph is a `networkx.MultiDiGraph` materialised once per pipeline context from the SQLite `structural_edges` table. BFS propagation is governed by `EDGE_CONFIG` in `impactracer/shared/constants.py`. The following rules are inviolable.

### 3.1 Per-edge direction and max_depth

| Edge type | Direction | Max depth | Rationale |
|---|---|---:|---|
| `CALLS` | reverse | **2** | Depth-3 fan-in regularly produces 200+ propagated nodes per seed in TS codebases. Depth-2 is the precision-recovery sweet spot. |
| `INHERITS` | reverse | 3 | Class hierarchies are typically shallow; 3 hops covers all real cases. |
| `IMPLEMENTS` | reverse | 3 | Interface contract graph. |
| `TYPED_BY` | reverse | 3 | Type-reference propagation. |
| `FIELDS_ACCESSED` | reverse | 2 | Field-level access has higher fan-out than method calls. |
| `DEFINES_METHOD` | forward | 1 | Definitional containment, not semantic propagation. |
| `PASSES_CALLBACK` | forward | 1 | |
| `HOOK_DEPENDS_ON` | reverse | 1 | React hook dependency edge. |
| `IMPORTS` | reverse | 1 | Module composition; no transitive impact assumed. |
| `RENDERS` | reverse | 1 | |
| `DEPENDS_ON_EXTERNAL` | reverse | 1 | |
| `CLIENT_API_CALLS` | reverse | 1 | |
| `DYNAMIC_IMPORT` | reverse | 1 | |
| `CONTAINS` | reverse | 1 | File ↔ symbol containment. Reverse-only: given a changed symbol, find which files contain it; do NOT enumerate sibling symbols. |

### 3.2 Confidence-tier CALLS cap

`LOW_CONF_CAPPED_EDGES = {CALLS}`: low-confidence seeds (i.e. not in the top-N reranker scores AND not directly retrieved) cap CALLS depth to 1. Prevents low-quality seeds from emitting deep propagation chains.

### 3.3 Hub mitigation

`_HUB_DEGREE_THRESHOLD = 20`: nodes whose total degree > 20 (typical for generic interfaces, framework primitives like `ext::react`) are capped at depth 1 for ALL edges when traversing FROM them. Prevents combinatorial explosion through framework hubs.

### 3.4 UTILITY-file CALLS cutoff

Seeds whose `file_classification == "UTILITY"` cap their reverse-CALLS chain at `UTILITY_FILE_CALLS_DEPTH_CAP = 1`. Utility functions are called from everywhere; deeper reverse-CALLS from a UTILITY seed is a near-guaranteed flood across unrelated features.

### 3.5 Per-node-type fan-in cap

`NODE_TYPE_MAX_FAN_IN`: a propagated neighbour with in-degree exceeding the type-specific cap is excluded from the CIS unless it is itself a SIS seed.

| Node type | Max fan-in |
|---|---:|
| Function / Method / Class | 50 |
| Interface / TypeAlias / Enum | 100 |
| InterfaceField | 200 |
| File | 200 |
| Variable | 80 |
| ExternalPackage | 0 (always excluded — see 3.6) |

### 3.6 Excluded-type wholesale

`EXCLUDED_PROPAGATION_NODE_TYPES = {ExternalPackage}`. Third-party package nodes are dependency edges' targets, not units of impact; the pipeline observes `DEPENDS_ON_EXTERNAL` edges but never adds the external package itself to the propagated set.

### 3.7 Severity (last-hop rule)

`severity_for_chain(causal_chain)` returns the severity of the LAST edge in the chain. SIS seeds (empty chain) are `Tinggi` by convention. This eliminates "severity laundering" where a chain like `CALLS → CALLS → IMPLEMENTS` would inherit the final IMPLEMENTS's Tinggi severity from an otherwise speculative path.

### 3.8 CONTAINS sub-tree collapse (Step 6.5)

After BFS, parent nodes whose CONTAINS-only children are in the CIS receive those children's ids in `NodeTrace.collapsed_children`; the children are removed from `propagated_nodes` to avoid token explosion. Each surviving collapsed child is individually re-validated by LLM #4 in Step 7, so this is a token-economy optimisation, not a short-circuit through validation.

### 3.9 Graph isolation invariant

`bfs_propagate` does NOT mutate the shared graph. Sequential ablation runs (V0 → V7 over the same CR) produce identical CIS results given identical inputs. Seeds absent from the graph are recorded as SIS-only terminal nodes (no expansion) but never inserted into the graph.

### 3.10 Step 7.5 — File-local sibling promotion via CONTAINS (Sprint 14 Apex Crucible Proposal A)

After LLM #4 validates the propagated set, `runner.py` invokes `collect_file_local_siblings` (defined in `graph_bfs.py`) to enumerate qualified siblings of every qualifying anchor. **Anchor qualification (Sprint 16):** anchor must be in `sis_justifications` AND have a non-empty LLM #2 `mechanism_of_impact`. The function fetches each anchor's `file_path`, then queries `code_nodes` for every qualified (`::`-bearing) symbol in the same files with `node_type ∈ {Function, Method, Interface, TypeAlias, Enum, Class, Variable}`. `InterfaceField` is excluded (already collapsed in Step 6.5 and never appears in GT).

Per-file candidate cap `settings.sibling_promotion_max_per_file = 12`. The list of `(sibling_id, node_type)` tuples per file is then submitted to `validate_siblings_for_file` (LLM #4 sibling-batch). Per-file admission cap `settings.sibling_admit_max_per_file = 4`; per-CR global cap is disabled (`sibling_admit_max_per_cr = 0`). Admitted siblings are injected into `cis.propagated_nodes` with `causal_chain=["CONTAINS"]`, `depth=1`, `source_seed=<primary anchor>`, `justification_source="llm4_sibling"`.

Step 7.5 fires only for V7 (`enable_propagation_validation=True`) AND `settings.enable_sibling_promotion=True`. Variants V4–V6 retain the unmodified post-LLM-#4 CIS.

---

## 4. Retrieval Architecture (Step 2 Detail)

The retriever is the deepest non-LLM lever in the pipeline. Three orthogonal mechanisms run inside `hybrid_search`; together they form the current Step 2 contract.

### 4.1 RRF pool sizing

```
top_k_per_query = 30      # per dense/BM25 query per path
top_k_rrf_pool  = 200     # candidates entering the cross-encoder
max_admitted_seeds = 15   # cap after rerank
rrf_k = 60
```

The 200-candidate pool is wide on purpose: the cross-encoder is the actual selector. A narrow pool starves the cross-encoder of the right answer; a wide pool gives it a meaningful selection problem at the cost of ~30s extra reranker time per V3+ run.

### 4.2 Raw-CR multilingual dense pass (W2B)

Gating: `settings.enable_raw_cr_dense_pass = True` AND `cr_text` is not None AND the dense path is enabled.

Implementation: inside the `dense_code` branch of `hybrid_search`, after the LLM-#1-search-query loop, the retriever embeds the raw CR text once and queries `code_col` for `settings.raw_cr_dense_top_k = 60` nearest neighbours. Results merge into the in-progress `seen_dc` dict via score-max (an id seen by both a search-query and the raw CR keeps the higher cosine).

Rationale: BGE-M3 is multilingual. The CR is Indonesian; the code identifiers are English. The shortest semantic path between them is the model's own multilingual embedding space. The LLM #1 search-query intermediation strips that direct signal and replaces it with concept-centric English queries; the raw-CR pass restores the direct path.

### 4.3 Traceability pool seeding (W2C)

Gating: `settings.enable_traceability_pool_seeding = True` AND `cr_interp.affected_layers` includes `"code"` AND at least one doc-chunk was retrieved AND `ctx.conn` is not None.

Implementation: after dense_doc / bm25_doc retrieve their respective doc-chunk lists, the retriever queries

```sql
SELECT doc_id, code_id, weighted_similarity_score
FROM doc_code_candidates
WHERE doc_id IN (...retrieved_doc_ids...)
  AND weighted_similarity_score >= ?            -- settings.traceability_seed_min_score = 0.40
ORDER BY doc_id, weighted_similarity_score DESC
```

For each retrieved doc chunk, up to `settings.traceability_seed_top_k_per_doc = 5` code neighbours are seeded into the `dense_code_ids` list (appended after the cosine-ranked entries) unless they are already in the dense_code or bm25_code lists.

Rationale: the offline traceability matrix is the canonical "this doc chunk talks about this code" precomputation. Treating it merely as a rerank +0.10 bonus wasted it on candidates already in the pool. Pool-membership seeding lets the matrix introduce GT-correct candidates that no LLM #1 search query happens to mention.

### 4.4 Post-rerank score adjustments (Step 3 detail)

After the cross-encoder produces `reranker_score` (sigmoid-normalised) and `raw_reranker_score` (raw logit):

- `apply_traceability_bonus(candidates, conn, bonus=0.10, top_k_per_doc=3)`: adds +0.10 to `raw_reranker_score` for each code candidate that any retrieved doc chunk's top-3 traceability neighbours include.
- `apply_negative_filter(candidates, out_of_scope_operations, penalty=1.0)` (Sprint 14 V2 softened): subtracts 1.0 from `raw_reranker_score` for each candidate whose **name** (not snippet) contains any out-of-scope operation. Needles shorter than 6 characters are dropped to avoid matching tokens like "log" or "add" in legitimate identifiers. **Additive on the cross-encoder logit**, never multiplicative — a multiplicative penalty would invert sign on a negative logit and inadvertently promote out-of-scope candidates.
  *Sprint 14 changed the default penalty from −5.0 to −1.0, dropped snippet matching, and added a ≥6-char needle filter after CR-02 forensics showed LLM #1's verbose post-Apex output listed phrases like "default grace period calculation" as out-of-scope; the −5.0 / name-or-snippet filter crushed legitimate "grace period" candidates. The −1.0 default acts as a tie-breaker rather than a kill switch.*

### 4.5 Layered code retrieval path (Sprint 14 Apex Crucible Proposal B)

A fifth ranked list, `layered_code`, is built inside `hybrid_search` when `cr_interp.layered_search_queries` is populated. For each canonical layer in `_CANONICAL_LAYERS = ("api_route", "page_component", "ui_component", "utility", "type_definition")`:

1. Look up the layer's queries from `cr_interp.layered_search_queries[layer_key]` (1–2 phrases per layer).
2. For each query, run a dense BGE-M3 query against `code_col` with `where={"file_classification": <FileClassification>}`, taking up to `settings.per_layer_top_k = 12` candidates.
3. For each query, run a BM25 query against `code_bm25` then filter post-hoc by `file_classification` in SQLite to enforce the same scope.
4. Per-layer hits are merged via score-max; top-K per layer feeds the global `layered_code` ranked list.

The fused list joins the RRF reducer as a first-class path with default weight 1.0 (RRF treats unknown labels as weight 1.0, parity with the other code paths). This guarantees no architectural layer is starved when LLM #1's flat `search_queries` are biased toward one plane.

**CR-02 calibration evidence:** before Sprint 14, LLM #1 emitted 3 service-layer queries for a service-described CR whose GT lived entirely in UI form components. The 200-pool contained 3 of 12 GT entities. With `layered_code`, the same CR's GT in-pool count rises (form-component layers contribute their own quotas), although CR-02 V7 final F1 remained 0.000 due to the structural form↔schema edge gap (see Sprint 15 / Proposal C postmortem in `implementation_report.md`).

---

## 5. Empirical Attrition Topography

Two reference calibration runs are documented here. Sprint 13-W2 (`eval/results_v2/`) is the pre-Apex baseline; **Sprint 16 (`eval/results_apex_v6/`) is the new canonical post-Apex result** with the anchor-mechanism gate enabled — this is the regime the thesis Chapter V cites for headline V7 numbers. The intermediate Sprint 14 V4 result (`eval/results_apex_v4/`, V7 F1 = 0.252) is retained as the "Apex A+B before anchor gating" reference. All runs: 5 CRs × canonical 8 variants = 40 cells, macro-averaged.

### 5.1 Variant table — Sprint 16 canonical (Apex A+B with anchor mechanism gate)

| Variant | Entity P | Entity R | Entity F1 | File P | File R | File F1 | Median entities | Median elapsed_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 0.107 | 0.107 | 0.107 | 0.172 | 0.300 | 0.219 | 10.0 |  ~6 |
| V1 | 0.138 | 0.184 | 0.134 | 0.158 | 0.348 | 0.205 | 11.0 |  ~7 |
| V2 | 0.255 | 0.249 | 0.244 | 0.184 | 0.438 | 0.250 | 10.0 |  ~7 |
| V3 | 0.199 | 0.271 | 0.207 | 0.163 | 0.381 | 0.221 |  7.0 | ~65 |
| V4 | 0.350 | 0.227 | 0.239 | 0.343 | 0.319 | 0.288 |  3.0 | ~70 |
| V5 | 0.386 | 0.227 | **0.267** | 0.270 | 0.319 | 0.281 |  3.0 | ~70 |
| V6 | 0.219 | 0.427 | 0.250 | 0.134 | **0.638** | 0.221 |  5.0 | ~70 |
| **V7** | **0.255** | **0.449** | **0.263** | **0.323** | **0.610** | **0.408** |  6.0 | ~80 |

### 5.2 Lift vs Sprint 13-W2 baseline

| Metric | Sprint 13-W2 (V7) | Sprint 14 V4 (V7) | Sprint 16 (V7) | Δ vs baseline |
|---|---:|---:|---:|---:|
| Entity F1 | 0.200 | 0.252 | **0.263** | **+31.5%** |
| Entity Precision | 0.149 | 0.226 | 0.255 | +71% |
| Entity Recall | 0.407 | 0.449 | 0.449 | +10% |
| File F1 | 0.377 | 0.442 | 0.408 | +8.2% |
| File Precision | 0.297 | 0.357 | 0.323 | +9% |
| File Recall | 0.605 | 0.638 | 0.610 | +1% |

### 5.3 Mechanistic signatures (Sprint 16 calibration)

**Graph Flood (V5 → V6).** Entity precision collapses 0.386 → 0.219 (ΔP = −0.167); recall lifts 0.227 → 0.427 (ΔR = +0.200); median entities 3.0 → 5.0. The flood signature is preserved.

**Precision Recovery (V6 → V7).** LLM #4 + Step 7.5 lifts entity precision 0.219 → 0.255 (ΔP = +0.036) AND recall 0.427 → 0.449 (ΔR = +0.022); median entities 5.0 → 6.0. **The Distributed Justification Principle remains empirically two-directional** even after the Sprint 16 anchor-gate tightening — LLM #4 raises both precision AND recall through Step 7.5 file-local sibling admissions.

**Variant ranking (Sprint 16):** V7 entity F1 = 0.263 is the highest precision+recall variant. V5 (0.267) is a tight 2-entity precision-heavy variant that edges V7 by 0.004 at n=5 — a sampling-noise gap (Cliff's δ = -0.12, median Δ ≈ 0). The 20-CR evaluation set will resolve V7-vs-V5 through the pre-registered Wilcoxon test.

**CR-04 stability (Sprint 16's design target).** The anchor mechanism gate eliminates the previously-observed CR-04 V7 = 0.000 failure mode caused by weak SIS anchors triggering 10-admit sibling overshoots. Sprint 16 calibration: CR-04 V7 = 0.250 with P=0.143, R=1.000. Cross-run stability is the load-bearing improvement of this regime, not the macro F1 lift.

**CR-02 is a structural limitation.** On the citrakara codebase CR-02 V7 = 0.000 across all 5 calibration iterations (Sprint 13-W2, Sprint 14 V1–V4, Sprint 15 α=0.7 / α=0.9). The CR text emphasises service-layer concepts ("grace period"); the GT lives in UI form components that fetch via API + Zod parse rather than importing schemas directly, severing the structural path. This is documented in `implementation_report.md` Sprint 15 as a fundamental limitation of static CIA on decoupled architectures and will be discussed in thesis Chapter V.

### 5.4 NFR results (Sprint 16 calibration run)

All 5 NFRs pass (`nfr_verification.json.all_passed = true`):

- **NFR-01 Determinism** — two V7 runs of the same CR produced identical validated SIS sets. The comparison target is `trace_sink["step_5b_llm3_verdicts"]["validated_code_seeds"]`, NOT `impacted_entities` (BFS + LLM #4 + Step 7.5 carry network variance NFR-01 does not test).
- **NFR-02 Local Execution** — stubbed per architect mandate; manual verification.
- **NFR-03 Latency** — overall median ~67s, p95 ~96s across 40 cells. V0 median 5.5s; V7 median 75.6s (Sprint 14 V4). Apex Crucible additions (`layered_code` path + Step 7.5 sibling promotion) added ~5–10s per V7 cell relative to Sprint 13-W2.
- **NFR-04 Cross-lingual** — Indonesian CR returned ≥1 entity with English identifier suffix (e.g. `CommissionListingPage`, `SearchCommissionListingPage`).
- **NFR-05 Config Consistency** — every audit entry within the run window shares one `config_hash`.

### 5.5 Statistical artefact (Sprint 16 calibration)

```json
{
  "status": "insufficient_pairs",
  "hypothesis": "V7.f1_set > V5.f1_set (one-sided paired Wilcoxon)",
  "variant_a": "V7",
  "variant_b": "V5",
  "metric": "f1_set",
  "n": 5,
  "min_required": 15,
  "median_diff_descriptive": 0.0000,
  "cliffs_delta_descriptive": -0.120
}
```

The pre-registered test is correctly **deferred** at n=5 < MIN_PAIRED_N=15. Sprint 16's anchor mechanism gate narrows V7-vs-V5 from δ=-0.28 (Sprint 14 V4) to **δ=-0.12** — the descriptive gap is now within sampling-noise bounds. V5 (2-entity tight precision) and V7 (6-entity precision+recall) sit on different points of the precision-recall trade-off; the 20-CR evaluation set will deliver the defensible p-value through the same harness.

### 5.6 Pipeline-stage funnel (one representative V7 CR in Sprint 14 V4)

- RRF pool → up to 200 candidates feed the cross-encoder (`top_k_rrf_pool=200`). The `layered_code` path contributes up to 60 (12 per layer × 5 layers).
- Cross-encoder reranks down to top-15 by raw_reranker_score.
- Pre-validation gates (3.5 / 3.6 / 3.7) drop a small handful per CR (typically 0–3).
- LLM #2 confirms ~half of survivors (acceptance ≈ 50–60 %).
- Doc → code resolution expands to ~10–20 pairs; LLM #3 prunes hard (~80–95 % rejection — most doc resolutions are not impactful seeds).
- BFS adds ~3–20 propagated nodes around the validated seeds (per CR; the W2 retrieval widening + Apex sibling promotion keeps this much tighter than pre-Apex's ~30).
- LLM #4 admits ~25 % of propagated nodes.
- Step 7.5 (sibling promotion) admits 0–4 file-local siblings per qualifying anchor; per-file cap=4, per-CR cap disabled. Sprint 14 V4 average across 5 CRs: ~3 admissions per V7 CR.
- File-type filter at synthesis drops bare File-type CIS nodes from `impacted_entities` (still emitted in `impacted_files`).
- Final entity-level report ~3–13 entities; final file-level report ~3–24 files, depending on the CR's cross-cutting scope.

`degraded_run` fires occasionally on CR-04 / CR-05 V5–V7 cells when an LLM batch hits schema-validation issues or rate limits after retries. On the n=5 calibration this can shift V7 entity F1 by ±0.01–0.02 — the LLM API noise floor smooths out at n=20.

---

## 6. Frozen Invariants

The following architectural invariants are FROZEN. Violating any of them requires updating this document AND `master_blueprint.md`.

1. **10 node types** (`shared/models.py::NodeType`): `File, Class, Function, Method, Interface, TypeAlias, Enum, ExternalPackage, InterfaceField, Variable`.
2. **14 structural edge types** (`shared/models.py::EdgeType`, `shared/constants.py::EDGE_CONFIG`): see §3.1 above.
3. **5 canonical LLM stages in V7**: `interpret`, `validate_sis`, `validate_trace`, `validate_propagation`, `synthesize`. Per-CR call counts can exceed 5 because Step 7 spawns `validate_collapsed_children` sub-calls and Step 7.5 (Sprint 14) spawns one `validate_siblings` call per file with a qualifying anchor. The five canonical stage names remain the architectural contract.
4. **8 canonical ablation variants** (`evaluation/variant_flags.py::ALL_VARIANTS = ["V0","V1","V2","V3","V4","V5","V6","V7"]`). V3 = deterministic-filtering peak (cross-encoder + all three gates, no LLM gating). V7 = full pipeline (BFS + LLM #4 + Step 7.5 sibling promotion + LLM #5 aggregator).
5. **3 change_type values**: `ADDITION, MODIFICATION, DELETION`.
6. **Fail-CLOSED at every validator.** Both per-item (drop on missing verdict) and per-batch (drop on exception, continue) at LLM #2, #3, #4 primary, #4 child-collapse, and #4 sibling-batch. The runner annotates the report with `degraded_run=True` when any drop fires.
7. **Distributed Justification Principle.** Entity-level justifications come VERBATIM from LLM #2 / LLM #3 / LLM #4 (including its sibling-batch sub-stage) or a synthetic `auto_exempt` string. LLM #5 never re-justifies entities. File-level justifications may be authored by LLM #5 because file summarisation is summarisation, not validation.
8. **Truncation decoupled from output.** The LLM #5 prompt may be truncated to fit the token budget; the report's `impacted_entities` list always contains the FULL validated CIS regardless.
9. **CALLS reverse depth = 2.** Combined with the UTILITY-CALLS cutoff and the per-node-type fan-in cap, this is the structural defence against graph flood.
10. **Negative filter is ADDITIVE (Sprint 14 default −1.0 on the cross-encoder logit, name-only, ≥6-char needle).** A multiplicative penalty would invert sign on negative logits and inadvertently promote out-of-scope candidates. Sprint 14 softened from −5.0 / name+snippet to the current default after CR-02 forensics.
11. **The Wilcoxon test target is entity-level `f1_set` (Total F1, set-level).** Bounded `F1@K` is absent from the codebase because it cannot detect graph floods.
12. **NFR-01 compares the validated SIS, not impacted_entities.** Specifically `trace_sink["step_5b_llm3_verdicts"]["validated_code_seeds"]` across two V7 runs. BFS + LLM #4 + Step 7.5 carry network-induced variance that NFR-01 is not designed to test.
13. **File-type entities are filtered from `impacted_entities` at synthesis** (Sprint 14). Every CIS node with `node_type == "File"` or without `::` in its id is dropped. Their `file_path` values are still injected into `impacted_files` via `extra_impacted_file_paths` so file-level reporting is preserved.
14. **TYPED_BY is NOT in `PROPAGATION_VALIDATION_EXEMPT_EDGES`** (Sprint 14). Only `IMPLEMENTS` and `DEFINES_METHOD` remain auto-exempt at depth 1. TYPED_BY now goes through LLM #4 like any other propagated chain.
15. **Sibling promotion (Step 7.5) anchors require non-empty LLM #2 `mechanism_of_impact`** (Sprint 16 — Option 1). Anchors without an articulate mechanism (e.g. CRUD funcs of unrelated domain entities) cannot drive lateral file-local expansion; this prevents the CR-04-style 10-admit overshoot observed in Apex V2.

---

## 7. Operational Surfaces

* **CLI** (`impactracer.cli`):
  * `impactracer index <repo>` — offline indexer.
  * `impactracer analyze "<CR text>" --variant V7 [--output PATH]` — online analysis. Always writes both `impact_report.json` and the sibling `impact_report_full.json`.
  * `impactracer evaluate --dataset DIR --output DIR [--run-full-ablation] [--verify-nfr]` — ablation harness over the canonical 8 variants × every CR in the GT directory; produces `per_cr_per_variant_metrics.csv`, `summary_table.csv/md`, `statistical_tests.json`, `calibration_analysis.md`, and (with `--verify-nfr`) `nfr_verification.json`.
  * `impactracer report [--output PATH]` — diagnostic indexing-quality report.
* **Diagnostic tools:**
  * `python tools/diagnose_pipeline.py --cr-text "..."` — V0..V7 attrition table for a single CR.
  * `python tools/e2e_test.py [--cr-id ID]` — runs the legacy 5-CR end-to-end stress test and grades each report against five quality criteria.
  * `python tools/reemit_eval_artifacts.py <output_dir> <dataset_dir> <run_start_iso>` — re-emit summary / statistical / NFR / analysis artefacts from an existing `per_cr_per_variant_metrics.csv` without re-running the ablation (used when only the post-CSV artefacts need refreshing).
* **Persistent state files:**
  * `data/impactracer.db` — SQLite (`code_nodes`, `structural_edges`, `doc_code_candidates`, `file_hashes`, `file_dependencies`, `index_metadata`).
  * `data/chroma_store/` — ChromaDB (`code_units`, `doc_chunks`).
  * `data/llm_audit.jsonl` — append-only LLM audit log (NFR-05 source).
  * `data/locked_parameters.json` — frozen calibration values written at evaluation start.

---

*End of analysis_implementation.md. Design specification in `master_blueprint.md`. Offline-indexer detail in `index_implementation.md`. Sprint history in `implementation_report.md`.*
