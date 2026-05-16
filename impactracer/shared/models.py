"""Pydantic v2 schemas, dataclasses, and enumerations.

This module is the authoritative data contract for the system. Every LLM
response schema inherits from :class:`TruncatingModel` so that overlong
string fields truncate silently instead of raising :class:`ValidationError`.

Schemas here are passed as ``response_schema`` to :class:`LLMClient.call`,
which enforces JSON mode via the OpenRouter API.

References:
    03_data_models.md - authoritative schema specification
    07_online_pipeline.md - how each schema is consumed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# =========================================================================
# Base model: graceful truncation
# =========================================================================


class TruncatingModel(BaseModel):
    """Base model that gracefully truncates overlong string fields.

    LLMs cannot reliably count characters; when a response slightly
    overshoots a ``max_length`` constraint, strict Pydantic validation
    raises ``ValidationError`` and crashes the pipeline. This validator
    inspects every string field's metadata for a declared ``max_length``
    and silently truncates the value to that limit before field-level
    validation runs.

    Every schema used as an LLM ``response_schema`` MUST inherit from
    this class.
    """

    @model_validator(mode="before")
    @classmethod
    def _truncate_overlong_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            value = data.get(field_name)
            if not isinstance(value, str):
                continue
            for meta in field_info.metadata:
                max_len = getattr(meta, "max_length", None)
                if max_len is not None and len(value) > max_len:
                    data[field_name] = value[:max_len]
                    break
        return data


# =========================================================================
# Enumerations
# =========================================================================

ChangeType = Literal["ADDITION", "MODIFICATION", "DELETION"]
"""Canonical change type values. Do not extend."""

ChunkType = Literal["FR", "NFR", "Design", "General"]
"""Markdown chunk classification."""

FileClassification = Literal[
    "API_ROUTE",
    "PAGE_COMPONENT",
    "UI_COMPONENT",
    "UTILITY",
    "TYPE_DEFINITION",
]
"""Source file architectural classification."""

NodeType = Literal[
    "File",
    "Class",
    "Function",
    "Method",
    "Interface",
    "TypeAlias",
    "Enum",
    "ExternalPackage",
    "InterfaceField",
    "Variable",
]
"""Code graph node kinds.

``Variable`` covers top-level ``const NAME = <RHS>`` declarations whose RHS is
NOT an arrow function (those are already captured as ``Function`` nodes).
Examples: Mongoose schemas (``const UserSchema = new Schema(...)``), constant
arrays / objects (``const TEMPLATES = [...]``), object-literal exports, and
``Object.freeze(...)`` constants. Added in Sprint 13-W1 to restore GT-existence
coverage that was missing under the original 9-type vocabulary.
"""

Severity = Literal["Tinggi", "Menengah", "Rendah"]
"""Impact severity category (Indonesian labels, thesis convention)."""

Scope = Literal["terlokalisasi", "menengah", "ekstensif"]
"""Impact scope classification."""

TraceDecision = Literal["CONFIRMED", "PARTIAL", "REJECTED"]
"""LLM #3 three-outcome decision for doc-to-code resolutions."""


# =========================================================================
# LLM #1: CR Interpretation
# =========================================================================


class CRInterpretation(BaseModel):
    """Output of LLM Call #1 (FR-B1 + FR-B2).

    Nine attributes exactly. The ``is_actionable`` flag gates the entire
    downstream pipeline: a False value halts analysis immediately with a
    minimal rejection report.
    """

    is_actionable: bool = Field(
        description=(
            "False if the CR is too ambiguous, contains no identifiable "
            "change intent, or is less than one full sentence."
        ),
    )
    actionability_reason: str | None = Field(
        default=None,
        description=(
            "One sentence explaining why the CR was rejected. "
            "Null when is_actionable is True."
        ),
    )
    primary_intent: str = Field(
        description="Single sentence describing what is being changed and why.",
    )
    change_type: ChangeType
    affected_layers: list[Literal["requirement", "design", "code"]]
    domain_concepts: list[str] = Field(
        description="Business-domain concepts, both explicit and implied.",
        min_length=1,
        max_length=10,
    )
    search_queries: list[str] = Field(
        description=(
            "2 to 5 English technical phrases optimized for vector search "
            "against code signatures and documentation section titles. "
            "Must be English even when the CR is in Indonesian."
        ),
        min_length=2,
        max_length=5,
    )
    named_entry_points: list[str] = Field(
        default_factory=list,
        description=(
            "1 to 4 specific function or component name patterns the CR "
            "explicitly describes. Used by the plausibility gate to exempt "
            "named elements from file-density limits."
        ),
        max_length=4,
    )
    out_of_scope_operations: list[str] = Field(
        default_factory=list,
        description=(
            "Business operations that share vocabulary with the CR but are "
            "NOT being changed. Injected into the SIS validator prompt as a "
            "hard exclusion list and used by the retriever to apply an "
            "additive demotion penalty (-5.0) on candidates whose names or "
            "snippets contain these substrings."
        ),
        max_length=4,
    )
    layered_search_queries: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Apex Crucible Proposal B: per-architectural-layer query grid. "
            "Keys MUST be one of: 'api_route', 'page_component', "
            "'ui_component', 'utility', 'type_definition'. Each value is a "
            "list of 1-2 English technical phrases that target THAT layer's "
            "naming conventions and structural patterns. The retriever runs "
            "every layer's queries against both the dense and BM25 indices "
            "and merges via per-layer RRF quotas, ensuring no layer is "
            "starved when one layer's signals dominate the LLM #1 output. "
            "When None, the retriever falls back to the flat search_queries."
        ),
    )
    is_nfr: bool = Field(
        default=False,
        description=(
            "True if the CR primarily concerns a non-functional requirement "
            "(performance, security, scalability, accessibility). NFR CRs "
            "still flow through the standard pipeline but are stratified in "
            "evaluation."
        ),
    )

    @model_validator(mode="after")
    def _check_actionable_fields(self) -> "CRInterpretation":
        """Enforce that actionable CRs have at least one affected layer.

        A zero-length affected_layers on an actionable CR causes all four
        retrieval paths to be skipped, producing zero candidates and a
        hallucinated synthesis result.
        """
        if self.is_actionable and len(self.affected_layers) == 0:
            # Coerce to a safe default rather than crashing — the LLM may
            # legitimately omit layers for ambiguous CRs. Default to all layers
            # so retrieval casts the widest net (maximum inclusivity mandate).
            self.affected_layers = ["requirement", "design", "code"]  # type: ignore[assignment]
        return self


# =========================================================================
# LLM #2: SIS Validation
# =========================================================================


class CandidateVerdict(TruncatingModel):
    """Per-candidate verdict from LLM Call #2 (FR-C5)."""

    node_id: str = Field(description="The candidate's node_id, copied verbatim.")
    function_purpose: str = Field(
        max_length=200,
        description="One sentence: what this node does.",
    )
    mechanism_of_impact: str = Field(
        max_length=400,
        description=(
            "Concrete modification mechanism if confirmed, or empty string "
            "if rejected. Propagated verbatim into the final "
            "ImpactReport.impacted_nodes[i].structural_justification."
        ),
    )
    justification: str = Field(
        max_length=400,
        description="One-sentence confirmation or rejection summary.",
    )
    confirmed: bool = Field(
        description=(
            "True only if mechanism_of_impact is non-empty and describes "
            "a structural, not merely topical, relevance."
        ),
    )


class SISValidationResult(TruncatingModel):
    """Envelope for LLM Call #2 response."""

    verdicts: list[CandidateVerdict]


# =========================================================================
# LLM #3: Trace Resolution Validation
# =========================================================================


class TraceVerdict(TruncatingModel):
    """Per-pair decision from LLM Call #3 (FR-C7)."""

    doc_chunk_id: str
    code_node_id: str
    decision: TraceDecision = Field(
        description=(
            "CONFIRMED: code node implements the document intent. "
            "PARTIAL: partial overlap, include with low-confidence marker. "
            "REJECTED: no implementation relationship."
        ),
    )
    justification: str = Field(
        max_length=400,
        description=(
            "One-sentence rationale for the decision. Propagated into "
            "ImpactedNode.structural_justification when this code seed "
            "is admitted via LLM #3."
        ),
    )


class TraceValidationResult(TruncatingModel):
    """Envelope for LLM Call #3 response."""

    verdicts: list[TraceVerdict]


# =========================================================================
# LLM #4: Propagation Validation
# =========================================================================


class PropagationVerdict(TruncatingModel):
    """Per-node decision from LLM Call #4 (FR-D2)."""

    node_id: str
    semantically_impacted: bool = Field(
        description=(
            "True if the structural dependency chain translates to actual "
            "semantic impact given the CR intent."
        ),
    )
    justification: str = Field(
        max_length=400,
        description=(
            "One sentence stating the contract breakage, behavioral anomaly, "
            "or downstream type-mismatch the change introduces. Crucible "
            "Fix 2 / Fix 3: must reference the chain factually but not be a "
            "bare-topology statement; propagated verbatim into the final "
            "ImpactedNode.structural_justification."
        ),
    )


class PropagationValidationResult(TruncatingModel):
    """Envelope for LLM Call #4 response."""

    verdicts: list[PropagationVerdict]


# =========================================================================
# LLM #5: ImpactReport Synthesis
# =========================================================================


class ImpactedEntity(TruncatingModel):
    """A single impacted code entity.

    Aligned with the Ground Truth ``GTEntry.impacted_entities`` shape.
    Each row carries the canonical ``node`` (= node_id) and a
    ``justification`` propagated VERBATIM from the validator that admitted
    the node (LLM #2 / LLM #3 / LLM #4 / auto_exempt). Justification is
    generated AT THE POINT OF VALIDATION, not retrospectively by LLM #5.
    """

    node: str = Field(description="The canonical node_id (matches GT format).")
    node_type: str
    file_path: str
    severity: Severity
    causal_chain: list[str] = Field(
        description="Ordered list of edge_type values from SIS root to this entity.",
    )
    justification: str = Field(
        max_length=400,
        description=(
            "One sentence explaining why this entity is impacted. "
            "PROPAGATED verbatim from the validator that admitted it "
            "(LLM #2 / LLM #3 / LLM #4 / auto_exempt)."
        ),
    )
    justification_source: str = Field(
        default="",
        description=(
            "Origin of the justification: 'llm2_sis', 'llm3_trace', "
            "'llm4_propagation', 'auto_exempt', 'bfs_only', "
            "or 'retrieval_only'."
        ),
    )
    traceability_backlinks: list[str] = Field(
        default_factory=list,
        description="Document chunk IDs linked to this entity.",
    )


class ImpactedFile(TruncatingModel):
    """A file-level impact entry.

    Aligned with ``GTEntry.impacted_files``: each row has ``file_path``
    and a ``justification`` summarizing why the file as a whole is
    impacted. The file-level justification MAY be written by LLM #5,
    summarizing the entity-level impacts within that file. This is the
    one place LLM #5 is permitted to generate prose — but it is bounded
    to the deterministic file-set computed by the runner.
    """

    file_path: str
    justification: str = Field(
        max_length=600,
        description=(
            "One-or-two-sentence rationale for why this file as a whole is "
            "impacted. May be generated by LLM #5 from the entity-level "
            "justifications already attached to entities in this file."
        ),
    )


# Backward-compat alias: legacy ImpactedNode points at ImpactedEntity. Some
# tests and the synthesizer still import ImpactedNode by name.
class ImpactedNode(ImpactedEntity):
    """Alias retained for backward compatibility. Prefer ImpactedEntity.

    Accepts both old field names (``node_id``, ``structural_justification``)
    and new field names (``node``, ``justification``) at construction time
    via a model-mode-before validator.
    """

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_field_names(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        # Map legacy -> new only if new is absent.
        if "node" not in data and "node_id" in data:
            data["node"] = data.pop("node_id")
        if "justification" not in data and "structural_justification" in data:
            data["justification"] = data.pop("structural_justification")
        return data

    @property
    def node_id(self) -> str:  # type: ignore[override]
        return self.node

    @property
    def structural_justification(self) -> str:
        return self.justification


class FileJustificationItem(TruncatingModel):
    """One file-level justification produced by LLM #5."""

    file_path: str
    justification: str = Field(
        max_length=600,
        description="One-or-two-sentence rationale for why this file is impacted.",
    )


class LLMSynthesisOutput(TruncatingModel):
    """Structured output from LLM Call #5.

    LLM #5 produces:
      - executive_summary (one paragraph)
      - documentation_conflicts (list of doc chunk IDs)
      - file_justifications: ONE entry per file in the deterministic
        impacted-file set, each with a one-or-two-sentence summary of
        why that file as a whole is impacted.

    LLM #5 NEVER generates entity-level justifications; those are
    propagated verbatim from LLM #2/#3/#4 (Distributed Justification
    Principle).
    """

    executive_summary: str = Field(
        max_length=800,
        description="One paragraph suitable for non-technical stakeholders.",
    )
    documentation_conflicts: list[str] = Field(
        default_factory=list,
        description=(
            "Doc chunk IDs whose stated requirements may conflict with the CR."
        ),
    )
    file_justifications: list[FileJustificationItem] = Field(
        default_factory=list,
        description=(
            "Per-file justifications. The runner enforces that the set of "
            "file_paths here matches the deterministic impacted-file set "
            "derived from impacted_entities."
        ),
    )


class ImpactReport(TruncatingModel):
    """Final user-visible artifact (FR-E3).

    Two entity arrays aligned with the Ground Truth shape:

      - ``impacted_entities`` (list[ImpactedEntity]): every entity in the
        validated CIS, with a justification propagated VERBATIM from
        LLM #2 / LLM #3 / LLM #4. 100% deterministic content.

      - ``impacted_files`` (list[ImpactedFile]): one row per distinct
        ``file_path`` referenced in ``impacted_entities``. The
        ``justification`` field is written by LLM #5 (file-level summary).

    Invariant: every file_path in any impacted_entity has a corresponding
    entry in impacted_files. Both arrays contain 100% of the validated CIS
    regardless of whether nodes were truncated from the LLM #5 prompt window.
    """

    executive_summary: str = Field(
        max_length=800,
        description="One paragraph suitable for non-technical stakeholders.",
    )
    impacted_files: list[ImpactedFile] = Field(
        default_factory=list,
        description=(
            "File-level impact rows. ONE entry per unique file_path in "
            "impacted_entities. Justifications are written by LLM #5 "
            "summarizing entity-level impacts within each file."
        ),
    )
    impacted_entities: list[ImpactedEntity] = Field(
        default_factory=list,
        description=(
            "Entity-level impact rows (functions, classes, interfaces, "
            "etc.). 100% deterministic content; justifications are "
            "VERBATIM from LLM #2 / LLM #3 / LLM #4."
        ),
    )
    documentation_conflicts: list[str] = Field(
        default_factory=list,
        description=(
            "Documentation sections whose stated requirements may conflict "
            "with the proposed change."
        ),
    )
    estimated_scope: Scope
    analysis_mode: Literal["retrieval_only", "retrieval_plus_propagation"] = Field(
        default="retrieval_only",
        description=(
            "retrieval_only: SIS seeds only (BFS not applied). "
            "retrieval_plus_propagation: BFS propagation was applied. "
            "Set by the pipeline runner, not by the LLM."
        ),
    )
    degraded_run: bool = Field(
        default=False,
        description=(
            "True if any LLM batch failed all retries and was dropped "
            "(fail-closed batch). The CR completed "
            "but the impact set is potentially incomplete."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_impacted_nodes(cls, data: Any) -> Any:
        # Legacy callers may pass impacted_nodes=. Map to impacted_entities.
        if isinstance(data, dict) and "impacted_nodes" in data and "impacted_entities" not in data:
            data["impacted_entities"] = data.pop("impacted_nodes")
        return data

    # Backward-compat: legacy callers iterate `report.impacted_nodes`.
    @property
    def impacted_nodes(self) -> list[ImpactedEntity]:
        return self.impacted_entities


# =========================================================================
# BFS Output Types (dataclasses, not LLM schemas)
# =========================================================================


@dataclass
class NodeTrace:
    """Per-node BFS trace record.

    ``justification`` is populated at the validation step that admits the node:
    LLM #2 for SIS seeds, LLM #3 for trace-resolved seeds, LLM #4 for
    propagated nodes, or a synthetic auto-exempt string for single-hop contract
    edges. ``justification_source`` records which validator produced it.
    """

    depth: int
    causal_chain: list[str]
    path: list[str]
    source_seed: str
    low_confidence_seed: bool = False
    # After BFS, parent nodes whose CONTAINS-only children are in the CIS
    # receive those children's IDs here; children are removed from
    # propagated_nodes to avoid token explosion.
    collapsed_children: list[str] = field(default_factory=list)
    justification: str = ""
    justification_source: str = ""  # 'llm2_sis' | 'llm3_trace' | 'llm4_propagation' | 'auto_exempt'
    function_purpose: str = ""      # Populated for SIS seeds from LLM #2 verdict.
    mechanism_of_impact: str = ""   # Populated for SIS seeds from LLM #2 verdict.


@dataclass
class CISResult:
    """Complete BFS output separating seeds from propagated nodes."""

    sis_nodes: dict[str, NodeTrace] = field(default_factory=dict)
    propagated_nodes: dict[str, NodeTrace] = field(default_factory=dict)

    def combined(self) -> dict[str, NodeTrace]:
        """Return a merged view of SIS seeds and propagated nodes.

        Merge contract (Phase 1 — E-4): SIS seeds take priority over
        propagated traces for the same node_id.  The dict unpacking order
        ``{**propagated, **sis}`` means sis_nodes keys overwrite any matching
        propagated_nodes key, preserving the seed's depth=0 / empty causal_chain
        attributes instead of a BFS-derived trace that would misrepresent the
        node as a propagated result.

        Invariant: for any node_id that appears in both dicts,
        ``combined()[node_id] is self.sis_nodes[node_id]``.
        """
        # propagated_nodes written first so sis_nodes overwrites on collision.
        return {**self.propagated_nodes, **self.sis_nodes}

    def all_node_ids(self) -> list[str]:
        """Return all node IDs as a flat list."""
        return list(self.sis_nodes.keys()) + list(self.propagated_nodes.keys())


# =========================================================================
# Retrieval Candidate DTO
# =========================================================================


@dataclass
class Candidate:
    """Internal DTO passed between retriever, reranker, gates, validator."""

    node_id: str
    node_type: str
    collection: str                     # "doc_chunks" or "code_units"
    rrf_score: float
    reranker_score: float = 0.0
    # raw_reranker_score: the cross-encoder score BEFORE min-max normalization.
    # Preserved so the score floor gate operates on absolute quality, not rank
    # position within the top-15 window (B4).
    raw_reranker_score: float = 0.0
    file_path: str = ""
    file_classification: str | None = None
    chunk_type: str | None = None
    name: str = ""
    text_snippet: str = ""
    internal_logic_abstraction: str | None = None
    # merged_doc_ids: node IDs of doc chunks deduplicated into this code node
    # during Step 3.6 (semantic dedup).
    merged_doc_ids: list[str] = field(default_factory=list)
    # merged_doc_contexts: parallel list of (section_title, text) tuples for
    # each merged doc chunk.  Injected into the LLM #2 validator prompt as
    # "Business Context" blocks so the LLM sees the requirement that makes
    # this code node relevant (B1).
    merged_doc_contexts: list[tuple[str, str]] = field(default_factory=list)
    bm25_score: float = 0.0
    cosine_score: float = 0.0
