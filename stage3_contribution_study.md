# Stage-3 Contribution Study (Deterministic In-Between Ablation)

**Question.** Within the online pipeline's retrieval stage (Step 3), do the
deterministic "in-between" mechanisms — cross-encoder rerank, traceability
bonus, negative filter, graph-aware rerank, semantic dedup (3.6), and the
plausibility/density gate (3.7) — contribute measurable accuracy, and is the
V2→V3 cross-encoder "jump" justified? If not, they are removable complexity,
and the result reinforces the thesis's main claim that impact-analysis accuracy
is driven by the LLM-validation stages, not the deterministic retrieval-side
machinery.

## Method

- **Scope.** Variants V0–V3 only (the deterministic regime; no LLM #2/#3/#4).
  LLM #5 synthesis is skipped (`force_include_all_cis_nodes`), as the scored
  `impacted_entities` set is built deterministically and does not depend on it.
- **Datasets.** Both indexed repos: `citrakara` (24 CRs) and `nova` (18 CRs),
  N = 42 change requests.
- **Metric.** Set-level entity F1 (`f1_set`) against hand-annotated ground
  truth, the thesis's primary metric.
- **Variance control (critical).** A single LLM #1 interpretation is computed
  once per CR and **reused, identical, across every compared cell**. This is
  required because at this F1 scale (~0.12 on small GT sets) LLM #1
  nondeterminism perturbs the retrieved pool — and therefore F1 — by *more*
  than the deterministic mechanisms under test. Cells are verified to differ
  only in the intended toggle (interp-parity check). An earlier run that gave
  each cell its own interpret call was discarded as confounded.
- **Two complementary per-mechanism measures.**
  - *Leave-one-out (LOO):* production configuration minus one mechanism →
    its marginal contribution to the working system.
  - *Add-one-in (ADD):* bare configuration (top-K cap only) plus one
    mechanism → its standalone effect.
- **Bare vs. full ladder.** `bare` = only the inseparable top-K truncation;
  `full` = production (all mechanisms enabled; score floor already retired).
- **Note.** The Step-3.5 score floor is excluded: a separate 42-CR ablation
  found it strictly inert (it admits all normalized cross-encoder scores) and
  it has been retired.

## Results

### Bare-vs-full ladder (mean entity F1)

| Variant | bare (cap only) | full (production) | dressing Δ |
|---|---|---|---|
| V0 | 0.1220 | 0.1317 | +0.0097 |
| V1 | 0.1179 | 0.1147 | −0.0032 |
| V2 | 0.1339 | 0.1390 | +0.0051 |
| V3 | 0.1306 | 0.1227 | **−0.0079** |

**V2→V3 cross-encoder jump:** bare **−0.0033**, full **−0.0163** (combined).
Per repo, full: citrakara −0.0145, nova −0.0187. The cross-encoder rerank does
**not** improve entity F1 over plain RRF retrieval at the deterministic level;
the surrounding mechanisms slightly worsen it. All dressing deltas fall within
the measured LLM noise floor of ~0.01–0.02 F1.

### Per-mechanism contribution at V3 (combined, N=42)

| Mechanism | Leave-one-out (marginal) | Add-one-in (standalone) | Verdict |
|---|---|---|---|
| Cross-encoder rerank | — (defines V3; see jump above) | — | no positive contribution |
| Graph-aware rerank | +0.0000 (off in production) | +0.0229 (citra +0.028, nova +0.017) | repo-specific; not enabled |
| Traceability bonus | +0.0004 | −0.0032 | inert |
| Negative filter | +0.0000 | +0.0000 | inert |
| Semantic dedup (3.6) | +0.0000 | +0.0000 | inert |
| Plausibility/density (3.7) | −0.0047 | −0.0084 | **net-negative** |

## Conclusion

No deterministic Stage-3 mechanism produces a positive, above-noise,
repo-general contribution to entity F1. Three (negative filter, dedup,
traceability bonus) are inert; the plausibility/density gate is net-negative;
graph-aware rerank has a small standalone effect on one repo only and is not
enabled in the production configuration. The V2→V3 cross-encoder step is itself
slightly negative.

The deterministic retrieval-side machinery is therefore **removable
complexity** on accuracy grounds. This is a positive result for the thesis: it
localizes all genuine accuracy gains to the LLM-validation stages (the V3→V4
SIS-validation step in particular), and demonstrates — exhaustively and with
variance control — that the cheap deterministic scaffolding does not justify
itself.

### Engineering decision (what was retired vs. retained)

Acting on this finding, the following were **retired** (archival-only in
source, never invoked): the score floor (3.5), plausibility/density gate (3.7),
traceability bonus (3·b), negative filter (3·c), and named-entry pinning (3·e);
Top-K truncation (3·f) now selects a plain top-N by cross-encoder score.

**Semantic dedup (3.6) was retained** — explicitly *not* as an F1 contributor
(it measured within-noise) but on **engineering/robustness grounds**: it carries
the spec section text of a merged doc chunk into the LLM #2 prompt as "Business
Context" (a downstream effect the deterministic study, which stops before LLM #2,
does not capture), and it prevents the same impact being double-counted as both
a doc and a code candidate. **Graph-aware rerank was retained as an optional,
default-off knob** (it showed a small repo-specific standalone effect). The
cross-encoder rerank and Top-K truncation were retained as the core of Step 3.

### Threats to validity

1. **Deterministic layer only.** These findings concern V0–V3. They do *not*
   speak to the LLM-validation stages (V4–V7), where the real measured gains
   occur; "Stage-3 is inert" must not be read as "the pipeline is inert."
2. **Noise floor.** Most deltas lie within ±0.01–0.02 F1. The defensible claim
   is "no above-noise positive contribution," not the exact per-mechanism
   values.
3. **Per-repo scale.** Nova's absolute F1 (~0.04) makes its percentage deltas
   volatile; combined and citrakara figures are the more stable basis.

*Evidence: `eval/stage3_contrib_metrics.csv` (definitive ladder + per-knob),
`eval/stage3_sweep_metrics.csv` (44-combo sweep), `eval/pin_ablation_metrics.csv`
(named-entry pinning), `eval/floor_ablation_metrics.csv` (score floor).*
