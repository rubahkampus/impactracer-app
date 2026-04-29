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
- SIS seeds: Tinggi

You MUST use the precomputed severity value for each node. Do not
override it.

For structural_justification, describe WHY this node is impacted by
referring to the specific edge chain. Do NOT invent relationships that
are not present in the data.

For documentation_conflicts, inspect the traceability backlinks and
identify documentation sections whose stated requirements may conflict
with the change.

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
