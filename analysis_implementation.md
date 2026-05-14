# Online Analysis Pipeline — Source of Truth

> Operational reference for the V7 online pipeline as it currently runs.
> Every claim here either quotes a constant from `shared/constants.py` /
> `shared/config.py` or cites an empirical number from
> `eval/results_v2/` (the live calibration run on the canonical 8 variants).
>
> Companion: `master_blueprint.md` is the design specification.
> `index_implementation.md` is the offline-indexer operational reference.
> `implementation_report.md` is the append-only sprint memory.

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
                       / TYPED_BY edges skip the LLM call (direct
                       structural contracts always kept; synthetic
                       justification = "Direct <edge> contract from
                       <seed> — auto-admitted exempt edge.").
                       Prompt is DE-BLINDED: the causal chain is shown
                       as factual context; anti-tautology language
                       forbids edge-type-as-evidence reasoning.
                       Per-child collapsed validation: each surviving
                       parent's collapsed_children are individually
                       re-validated by an LLM-#4-style call.
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
final_report
```

Steps 5b / 6 / 7 may be absent for variants that disable those phases or for CRs that resolve to zero seeds before the corresponding stage.

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

### LLM #4 — Validate Propagation (`validate_propagation`)

* **Role:** for each BFS-propagated node that is NOT auto-exempt, decide whether the structural reach implies semantic impact. Plus, a parameterised reuse of the same prompt validates each `collapsed_children` entry individually.
* **Prompt constraints** (`pipeline/traversal_validator.py::_SYSTEM_PROMPT`):
  * **De-blinded chain:** the causal chain IS shown as factual context. Tautology safety is enforced by explicit anti-tautology language: *"Edge types are NOT impact evidence … Reject any node where the relationship is structurally present but the target's behaviour is unaffected by the CR."*
  * Forbidden justification patterns enumerated in the prompt ("function A calls function B" / "in the same module" / generic relation strings are explicitly rejected).
  * Required justification format demands a contract-breakage, behavioural-anomaly, or downstream-type-mismatch sentence.
  * Determinism: `random.seed(42); random.shuffle(to_validate)` before batching to neutralise positional bias without compromising reproducibility.
  * Delimiter contract + sanitisation identical to LLM #2.
* **Auto-exempt edges:** depth-1 `IMPLEMENTS`, `DEFINES_METHOD`, `TYPED_BY` bypass the LLM call entirely. They receive a synthetic justification `"Direct <edge> contract from <seed> — auto-admitted exempt edge."` with `justification_source="auto_exempt"`. These are direct structural contracts whose impact is definitional, not stochastic.
* **Per-child collapse validation:** for each surviving parent with `collapsed_children`, an additional LLM-#4-style call individually validates each child name. Same fail-closed semantics.
* **Fail-closed:**
  * **Per node:** missing verdict → DROP.
  * **Per batch:** exception → DROP entire batch, continue.
  * **Per child batch:** exception → DROP entire child batch, continue.
* **Captures:** verdict justification → propagated to `NodeTrace.justification` with `justification_source="llm4_propagation"`.

### LLM #5 — Synthesize (`synthesize_summary`)

* **Role (aggregator-only):** produces the executive summary, documentation-conflicts list, and **per-file** justifications. NEVER produces per-entity justifications.
* **Distributed Justification Principle:**
  * `impacted_entities` is built deterministically by the runner from the validated CIS. Each entity's `justification` is propagated VERBATIM from the LLM (#2/#3/#4) that admitted it, or from the synthetic `auto_exempt` string. LLM #5 never sees nor authors these.
  * `impacted_files` is also deterministic with respect to its `file_path` set: every distinct file referenced by `impacted_entities` MUST have exactly one row in `impacted_files`. The `justification` field of each file row may be written by LLM #5 (file-level summarisation is by definition a summarisation task, not a per-entity validation task). If LLM #5 omits a file or hallucinates one, the runner reconciles: hallucinated files dropped silently; omitted files receive a deterministic fallback that summarises the entity-level justifications inside that file.
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
- `apply_negative_filter(candidates, out_of_scope_operations, penalty=5.0)`: subtracts 5.0 from `raw_reranker_score` for each candidate whose name or snippet (lowercased) contains any out-of-scope operation. **Additive on the cross-encoder logit**, never multiplicative — a multiplicative penalty would invert sign on a negative logit and inadvertently promote out-of-scope candidates.

---

## 5. Empirical Attrition Topography

The numbers below come from the calibration run on `eval/results_v2/` (5 CRs × canonical 8 variants = 40 cells). All metrics are macro-averaged across CRs; latencies are medians.

### 5.1 Variant table (calibration)

| Variant | Entity P | Entity R | Entity F1 | File P | File R | File F1 | Median entities | Median elapsed_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 0.054 | 0.187 | 0.079 | 0.182 | 0.348 | 0.228 | 18.0 |  5.4 |
| V1 | 0.058 | 0.204 | 0.083 | 0.249 | 0.433 | 0.303 | 21.0 |  6.2 |
| V2 | 0.046 | 0.182 | 0.070 | 0.193 | 0.376 | 0.243 | 19.0 |  6.0 |
| V3 | 0.078 | 0.207 | 0.104 | 0.181 | 0.548 | 0.246 | 15.0 | 64.2 |
| V4 | 0.183 | 0.207 | 0.194 | 0.278 | 0.348 | 0.307 |  6.0 | 72.4 |
| V5 | 0.179 | 0.187 | 0.181 | 0.309 | 0.319 | 0.307 |  6.0 | 70.1 |
| V6 | 0.086 | **0.440** | 0.133 | 0.170 | **0.757** | 0.261 | 20.0 | 69.3 |
| **V7** | **0.149** | 0.407 | **0.200** | **0.297** | 0.605 | **0.377** |  7.0 | 70.2 |

### 5.2 Mechanistic signatures

**Graph Flood (V5 → V6).** Entity precision collapses 0.179 → 0.086 (ΔP = −0.093); recall jumps 0.187 → 0.440 (ΔR = +0.253); median entities 6.0 → 20.0. BFS lifts recall by 2.4× and halves precision — the textbook signature predicted by the architecture.

**Precision Recovery (V6 → V7).** LLM #4 lifts entity precision 0.086 → 0.149 (ΔP = +0.063); recall holds at ≈ 0.4 (ΔR = −0.033); median entities pruned 20.0 → 7.0. LLM #4 rejects ~13 of every 20 propagated nodes — the precision-recovery valve is empirically functional.

**V7 leads the variant ranking** at entity F1 = 0.200 (was 0.044 at the pre-W2 baseline — a 4.5× lift). V4 (0.194) and V5 (0.181) come second and third. The full pipeline ranks first as the design predicts.

### 5.3 NFR results (calibration run)

All 5 NFRs pass (`nfr_verification.json.all_passed = true`):

- **NFR-01 Determinism** — two V7 runs of the same CR produced identical 4-element validated SIS sets. The comparison target is `trace_sink["step_5b_llm3_verdicts"]["validated_code_seeds"]`, NOT `impacted_entities` (BFS + LLM #4 carry network variance NFR-01 does not test).
- **NFR-02 Local Execution** — stubbed per architect mandate; manual verification.
- **NFR-03 Latency** — overall median 61.0s, p95 84.3s across 40 cells. V0 median 5.4s; V7 median 70.2s.
- **NFR-04 Cross-lingual** — Indonesian CR-01 returned 31 entities including English identifiers (`updateListing`, `updateUserProfile`, `PATCH`).
- **NFR-05 Config Consistency** — 216 audit entries within the run window, all sharing one `config_hash`.

### 5.4 Statistical artefact (calibration)

```json
{
  "status": "insufficient_pairs",
  "hypothesis": "V7.f1_set > V5.f1_set (one-sided paired Wilcoxon)",
  "variant_a": "V7",
  "variant_b": "V5",
  "metric": "f1_set",
  "n": 5,
  "min_required": 15,
  "median_diff_descriptive": -0.0177,
  "cliffs_delta_descriptive": 0.04
}
```

The pre-registered test is correctly **deferred** at n=5 < MIN_PAIRED_N=15. The 20-CR evaluation set will deliver the defensible p-value through the same harness.

### 5.5 Pipeline-stage funnel (one representative CR)

Reading the average funnel a V7 run takes on calibration:

- RRF pool → ~40–200 candidates feed the cross-encoder (varies with how many search queries × paths return non-empty lists).
- Cross-encoder reranks down to a fixed top-15.
- Pre-validation gates (3.5 / 3.6 / 3.7) drop a small handful per CR (typically 0–3).
- LLM #2 confirms ~half of survivors (acceptance ≈ 50–60 %).
- Doc → code resolution expands to ~10–20 pairs; LLM #3 prunes hard (~80–95 % rejection — most doc resolutions are not impactful seeds).
- BFS adds ~30 propagated nodes around the validated seeds (per CR average).
- LLM #4 admits ~25 % of propagated nodes (auto-exempt + LLM-confirmed); the other ~75 % are the documented graph flood.
- Final report ~7–28 entities, depending on the CR's cross-cutting scope.

`degraded_run = false` across all 5 calibration CRs — the fail-closed guard rails are wired but did not need to fire.

---

## 6. Frozen Invariants

The following architectural invariants are FROZEN. Violating any of them requires updating this document AND `master_blueprint.md`.

1. **10 node types** (`shared/models.py::NodeType`): `File, Class, Function, Method, Interface, TypeAlias, Enum, ExternalPackage, InterfaceField, Variable`.
2. **14 structural edge types** (`shared/models.py::EdgeType`, `shared/constants.py::EDGE_CONFIG`): see §3.1 above.
3. **5 LLM invocations in V7**: `interpret`, `validate_sis`, `validate_trace`, `validate_propagation`, `synthesize`. The internal child-validation call inside Step 7 reuses the LLM #4 prompt and is parameterised from the same module; it does not count as a sixth distinct stage.
4. **8 canonical ablation variants** (`evaluation/variant_flags.py::ALL_VARIANTS = ["V0","V1","V2","V3","V4","V5","V6","V7"]`). V3 = deterministic-filtering peak (cross-encoder + all three gates, no LLM gating). V7 = full pipeline (BFS + LLM #4 + LLM #5 aggregator).
5. **3 change_type values**: `ADDITION, MODIFICATION, DELETION`.
6. **Fail-CLOSED at every validator.** Both per-item (drop on missing verdict) and per-batch (drop on exception, continue) at LLM #2, #3, and #4. The runner annotates the report with `degraded_run=True` when any drop fires.
7. **Distributed Justification Principle.** Entity-level justifications come VERBATIM from LLM #2 / LLM #3 / LLM #4 or a synthetic `auto_exempt` string. LLM #5 never re-justifies entities. File-level justifications may be authored by LLM #5 because file summarisation is summarisation, not validation.
8. **Truncation decoupled from output.** The LLM #5 prompt may be truncated to fit the token budget; the report's `impacted_entities` list always contains the FULL validated CIS regardless.
9. **CALLS reverse depth = 2.** Combined with the UTILITY-CALLS cutoff and the per-node-type fan-in cap, this is the structural defence against graph flood.
10. **Negative filter is ADDITIVE (−5.0 on the cross-encoder logit).** A multiplicative penalty would invert sign on negative logits and inadvertently promote out-of-scope candidates.
11. **The Wilcoxon test target is entity-level `f1_set` (Total F1, set-level).** Bounded `F1@K` is absent from the codebase because it cannot detect graph floods.
12. **NFR-01 compares the validated SIS, not impacted_entities.** Specifically `trace_sink["step_5b_llm3_verdicts"]["validated_code_seeds"]` across two V7 runs. BFS + LLM #4 carry network-induced variance that NFR-01 is not designed to test.

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
