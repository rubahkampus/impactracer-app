"""LLM Call #5: ImpactReport synthesis (FR-E3).

Crucible Fix 3 (Full Demotion) + Crucible E2E Schema Alignment:

  LLM #5 produces ONLY:
    - executive_summary (one paragraph)
    - documentation_conflicts (doc chunk IDs)
    - file_justifications: one row per impacted file, summarizing the
      entity-level impacts in that file. (LLM #5 IS allowed to write
      file-level prose, since file justification is a summarization
      task, not a per-entity validation task.)

  The runner builds:
    - impacted_entities (deterministic; justifications verbatim from
      LLM #2 / LLM #3 / LLM #4)
    - impacted_files (deterministic file_path set; LLM #5's
      file_justifications are matched against this set with
      "no information available" filled in for any missing files)

Crucible Amendment 2 (Truncation Decoupling): the runner emits the
FULL validated CIS in ImpactReport.impacted_entities regardless of
whether the corresponding nodes were truncated from the LLM #5 prompt
window.

Reference: 07_online_pipeline.md §13; Crucible E2E Plan TASK 2.
"""

from __future__ import annotations

from impactracer.pipeline.llm_client import LLMClient
from impactracer.shared.constants import severity_for_chain
from impactracer.shared.models import (
    CISResult,
    ImpactedEntity,
    ImpactedFile,
    ImpactReport,
    LLMSynthesisOutput,
)


SYSTEM_PROMPT = """You are an aggregator for a Change Impact Analysis pipeline.

INPUT TO YOU:
  1. The Change Request (CR) text and the system's interpretation.
  2. A pre-validated impact set: each ENTITY already carries a verbatim
     justification produced by an upstream validator (LLM #2 / #3 / #4).
     YOU MUST NOT re-justify individual entities.
  3. A list of distinct FILES referenced by those entities.

YOUR THREE RESPONSIBILITIES:

A. executive_summary (string, max ~800 chars):
   ONE paragraph for a non-technical stakeholder. Reference the most
   material change mechanisms drawn from the per-entity justifications
   already in the context. Do NOT invent mechanisms.

B. documentation_conflicts (list of strings, may be empty []):
   Doc-chunk IDs whose stated requirements may conflict with the CR.
   Inspect the traceability backlinks in the context.

C. file_justifications (list of objects):
   For EACH distinct file_path listed in the "=== IMPACTED FILES ==="
   section of the context, produce ONE row with:
     - file_path: copied verbatim from the list.
     - justification: ONE OR TWO sentences explaining why the FILE AS A
       WHOLE is impacted, summarizing the entity-level justifications
       attached to entities within that file.
   Cover EVERY file in the list. The justification should be concrete
   (refer to specific structural changes), not generic.

DO NOT:
  - Output an impacted_entities array (the runner builds it).
  - Override severity, scope, analysis_mode, or causal_chain.
  - Re-write or paraphrase the entity-level justifications.

OUTPUT — return exactly this JSON shape:
{
  "executive_summary": "...",
  "documentation_conflicts": ["doc_id_1", ...],
  "file_justifications": [
    {"file_path": "src/...", "justification": "..."},
    ...
  ]
}
"""


def synthesize_summary(context: str, client: LLMClient) -> LLMSynthesisOutput:
    """Run LLM Call #5 and return summary + per-file justifications."""
    return client.call(
        system=SYSTEM_PROMPT,
        user=context,
        response_schema=LLMSynthesisOutput,
        call_name="synthesize",
    )


def build_deterministic_impacted_entities(
    cis: CISResult,
    node_types: dict[str, str],
    node_file_paths: dict[str, str],
    justifications_extra: dict[str, str] | None = None,
    backlinks: dict[str, list[tuple[str, float]]] | None = None,
) -> list[ImpactedEntity]:
    """Build the entity-level rows DETERMINISTICALLY from the validated CIS.

    Justification source priority (Crucible Fix 3):
      1. trace.justification (set by runner from LLM #2/#3/#4 verdict)
      2. justifications_extra[node_id] (parallel map for legacy compat)
      3. trace.mechanism_of_impact (LLM #2 SIS attribute)
      4. empty string

    Severity is computed by ``severity_for_chain`` (last-hop rule).
    Backlinks are pulled from the assembled traceability map.
    """
    justifications_extra = justifications_extra or {}
    backlinks = backlinks or {}
    combined = cis.combined()
    rows: list[ImpactedEntity] = []

    for node_id, trace in combined.items():
        just = (
            trace.justification
            or justifications_extra.get(node_id, "")
            or trace.mechanism_of_impact
            or ""
        )
        if len(just) > 400:
            just = just[:400]

        bl = backlinks.get(node_id, [])
        backlink_ids = [doc_id for doc_id, _score in bl]

        rows.append(
            ImpactedEntity(
                node=node_id,
                node_type=node_types.get(node_id, "Unknown"),
                file_path=node_file_paths.get(node_id, ""),
                severity=severity_for_chain(trace.causal_chain),
                causal_chain=trace.causal_chain,
                justification=just,
                justification_source=trace.justification_source or "",
                traceability_backlinks=backlink_ids,
            )
        )

    return rows


# Backward-compat alias.
build_deterministic_impacted_nodes = build_deterministic_impacted_entities


def derive_impacted_files(
    entities: list[ImpactedEntity],
    file_justifications_from_llm5: dict[str, str] | None = None,
) -> list[ImpactedFile]:
    """Build the file-level rows DETERMINISTICALLY from impacted_entities.

    Invariant (Crucible E2E Task 2): ONE file row per distinct
    non-empty ``file_path`` referenced in ``entities``. Order: first
    appearance in entities (stable + reproducible).

    Justification source for each file_path:
      - LLM #5's file_justifications dict if it has a matching entry,
      - else a deterministic fallback summarizing the count of impacted
        entities and the worst severity.

    Note: LLM #5's prose is BOUNDED to the file set the runner derives.
    Files LLM #5 hallucinates that are not in the entity set are dropped;
    files in the entity set that LLM #5 omitted get the fallback string.
    """
    file_justifications_from_llm5 = file_justifications_from_llm5 or {}

    seen_order: list[str] = []
    seen_set: set[str] = set()
    file_to_entities: dict[str, list[ImpactedEntity]] = {}

    for ent in entities:
        fp = ent.file_path or ""
        if not fp:
            continue
        if fp not in seen_set:
            seen_set.add(fp)
            seen_order.append(fp)
        file_to_entities.setdefault(fp, []).append(ent)

    rows: list[ImpactedFile] = []
    severity_rank = {"Tinggi": 0, "Menengah": 1, "Rendah": 2}

    for fp in seen_order:
        ents = file_to_entities[fp]
        llm_just = file_justifications_from_llm5.get(fp, "").strip()
        if llm_just:
            justification = llm_just[:600]
        else:
            # Fallback: deterministic file summary.
            sev = min((severity_rank[e.severity] for e in ents), default=2)
            sev_label = ["Tinggi", "Menengah", "Rendah"][sev]
            n = len(ents)
            sample = ents[0]
            justification = (
                f"Contains {n} impacted entit{'y' if n == 1 else 'ies'} "
                f"(highest severity: {sev_label}). Representative impact: "
                f"{sample.justification[:300]}"
            )[:600]
        rows.append(ImpactedFile(file_path=fp, justification=justification))

    return rows


def assemble_impact_report(
    summary: LLMSynthesisOutput,
    impacted_entities: list[ImpactedEntity],
    estimated_scope: str,
    analysis_mode: str,
    degraded_run: bool,
) -> ImpactReport:
    """Assemble the final ImpactReport.

    Reconciles LLM #5's ``file_justifications`` against the deterministic
    file set derived from ``impacted_entities``. Files LLM #5 hallucinates
    that don't appear in the entity set are dropped silently; files in
    the entity set that LLM #5 omitted receive a deterministic fallback
    justification.
    """
    llm5_file_just = {
        item.file_path: item.justification
        for item in summary.file_justifications
        if item.file_path
    }
    impacted_files = derive_impacted_files(
        impacted_entities, file_justifications_from_llm5=llm5_file_just
    )

    return ImpactReport(
        executive_summary=summary.executive_summary,
        impacted_files=impacted_files,
        impacted_entities=impacted_entities,
        documentation_conflicts=list(summary.documentation_conflicts),
        estimated_scope=estimated_scope,  # type: ignore[arg-type]
        analysis_mode=analysis_mode,  # type: ignore[arg-type]
        degraded_run=degraded_run,
    )


def build_minimal_summary(
    text: str = "Change request analysis completed.",
    conflicts: list[str] | None = None,
) -> LLMSynthesisOutput:
    """Build a non-LLM summary for forced-inclusion / fallback paths."""
    return LLMSynthesisOutput(
        executive_summary=text[:800],
        documentation_conflicts=list(conflicts or []),
        file_justifications=[],
    )
