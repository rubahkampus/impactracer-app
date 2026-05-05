"""LLM Call #5: ImpactReport synthesis (FR-E3).

Severity per node is precomputed from the causal chain; the LLM must not
override it. The prompt enforces this constraint.

Phase 3.3 (A-4): ``build_forced_inclusion_report`` constructs an ImpactReport
directly from the CIS without calling LLM #5. This is the "oracle output"
variant used to separate pipeline CIA recall from LLM #5 selection quality.
When ``force_include_all_cis_nodes=True`` in VariantFlags, runner.py calls
this function instead of ``synthesize_report``.

Reference: 07_online_pipeline.md §13.
"""

from __future__ import annotations

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import CISResult, ImpactReport, ImpactedNode


SYSTEM_PROMPT = """You are a software change impact analysis report generator.

Given a Change Request, a set of impacted code nodes with causal chains,
and traceability backlinks, produce a structured ImpactReport.

The severity of each impacted node is PRECOMPUTED from its causal chain:
- Tinggi: contract dependency (IMPLEMENTS, TYPED_BY, FIELDS_ACCESSED)
- Menengah: behavioral dependency (CALLS, INHERITS, DEFINES_METHOD,
  HOOK_DEPENDS_ON, PASSES_CALLBACK)
- Rendah: module composition (IMPORTS, RENDERS, DEPENDS_ON_EXTERNAL,
  CLIENT_API_CALLS, DYNAMIC_IMPORT)
- SIS seeds (empty causal chain): Tinggi

You MUST use the precomputed severity value for each node. Do not
override it.

For structural_justification, describe WHY this node is impacted by
referring to the specific edge chain. Do NOT invent relationships that
are not present in the data.

For documentation_conflicts, inspect the traceability backlinks and
identify documentation sections whose stated requirements may conflict
with the change.

SCHEMA CONSTRAINTS (must be satisfied in every response):
- executive_summary: string, max 800 characters, always required
- impacted_nodes: list of objects, each with:
    - node_id: string — copy verbatim from the NODE header in the context
    - node_type: string — copy verbatim from the node_type field in the context
    - file_path: string — copy verbatim from the file_path field; use "" if "(doc chunk)"
    - severity: one of "Tinggi", "Menengah", "Rendah" — copy from the Severity field
    - causal_chain: list of strings — copy the JSON array from "Causal chain (JSON array for output)"
    - structural_justification: string, max 200 characters
    - traceability_backlinks: list of strings — doc chunk IDs from the backlinks section
- documentation_conflicts: list of strings (use [] if none)
- estimated_scope: one of "terlokalisasi", "menengah", "ekstensif", always required

CRITICAL:
- file_path MUST be a string — use "" (empty string) if no file path is available, never null
- causal_chain MUST be a list — copy the JSON array exactly, never a plain string
- Do NOT wrap the response in any outer key (e.g. do not use {"impact_report": {...}})
- The JSON root must be the ImpactReport object itself
- analysis_mode is set by the pipeline, not by you — omit it from your response

Return valid JSON matching the schema exactly.
"""


def synthesize_report(context: str, client: LLMClient) -> ImpactReport:
    """Run LLM Call #5 and return the final report."""
    return client.call(
        system=SYSTEM_PROMPT,
        user=context,
        response_schema=ImpactReport,
        call_name="synthesize",
    )


def build_forced_inclusion_report(
    cis: CISResult,
    node_types: dict[str, str],
    node_file_paths: dict[str, str],
    estimated_scope: str,
) -> ImpactReport:
    """Build an ImpactReport from all CIS nodes without calling LLM #5.

    Phase 3.3 (A-4): the "oracle" or forced-inclusion variant. Used to
    measure the pipeline's raw CIA recall without LLM #5 selection bias.

    When LLM #5 selects only 7 of 208 CIS nodes, it is unclear whether:
    (a) the pipeline correctly identified 208 nodes but LLM #5 under-reports,
    or (b) the pipeline over-identified and LLM #5 correctly filters.

    This variant produces a deterministic report from the FULL CIS, allowing
    evaluation against ground truth at the CIS level. Comparing forced-inclusion
    F1 vs LLM-synthesized F1 quantifies LLM #5's contribution.

    All nodes are included as ImpactedNode records with severity computed
    from their causal chain. The executive_summary is a placeholder;
    this variant is for metric computation only, not for user presentation.
    """
    from impactracer.shared.constants import severity_for_chain

    combined = cis.combined()
    impacted: list[ImpactedNode] = []

    for node_id, trace in combined.items():
        impacted.append(ImpactedNode(
            node_id=node_id,
            node_type=node_types.get(node_id, "Unknown"),
            file_path=node_file_paths.get(node_id, ""),
            severity=severity_for_chain(trace.causal_chain),
            causal_chain=trace.causal_chain,
            structural_justification=f"Included via forced-inclusion variant (depth={trace.depth})",
            traceability_backlinks=[],
        ))

    return ImpactReport(
        executive_summary=(
            f"[FORCED INCLUSION] All {len(impacted)} CIS nodes included for "
            "evaluation. This variant bypasses LLM #5 synthesis."
        ),
        impacted_nodes=impacted,
        documentation_conflicts=[],
        estimated_scope=estimated_scope,  # type: ignore[arg-type]
        analysis_mode="retrieval_plus_propagation" if cis.propagated_nodes else "retrieval_only",
    )
