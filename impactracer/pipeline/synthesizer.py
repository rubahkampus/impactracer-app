"""LLM Call #5: ImpactReport synthesis (FR-E3).

Severity per node is precomputed from the causal chain; the LLM must not
override it. The prompt enforces this constraint.

Reference: 07_online_pipeline.md §13.
"""

from __future__ import annotations

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.models import ImpactReport


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
