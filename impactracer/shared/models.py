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
]
"""Code graph node kinds (nine types)."""

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
            "hard exclusion list."
        ),
        max_length=4,
    )


# =========================================================================
# LLM #2: SIS Validation
# =========================================================================


class CandidateVerdict(TruncatingModel):
    """Per-candidate verdict from LLM Call #2 (FR-C5)."""

    node_id: str = Field(description="The candidate's node_id, copied verbatim.")
    function_purpose: str = Field(
        max_length=150,
        description="One sentence: what this node does.",
    )
    mechanism_of_impact: str = Field(
        max_length=200,
        description=(
            "Concrete modification mechanism if confirmed, or empty string "
            "if rejected."
        ),
    )
    justification: str = Field(
        max_length=200,
        description="One-sentence confirmation or rejection summary.",
    )
    confirmed: bool = Field(
        description=(
            "True only if mechanism_of_impact is non-empty and describes "
            "a structural, not merely topical, relevance."
        ),
    )


class SISValidationResult(BaseModel):
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
        max_length=200,
        description="One-sentence rationale for the decision.",
    )


class TraceValidationResult(BaseModel):
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
        max_length=200,
        description="One-sentence rationale referencing the edge chain.",
    )


class PropagationValidationResult(BaseModel):
    """Envelope for LLM Call #4 response."""

    verdicts: list[PropagationVerdict]


# =========================================================================
# LLM #5: ImpactReport Synthesis
# =========================================================================


class ImpactedNode(TruncatingModel):
    """A single impacted code element in the final report."""

    node_id: str
    node_type: str
    file_path: str
    severity: Severity
    causal_chain: list[str] = Field(
        description="Ordered list of edge_type values from SIS root to this node.",
    )
    structural_justification: str = Field(
        max_length=200,
        description="One sentence explaining why this node is impacted.",
    )
    traceability_backlinks: list[str] = Field(
        default_factory=list,
        description="Document chunk IDs linked to this code node.",
    )


class ImpactReport(TruncatingModel):
    """Final user-visible artifact (FR-E3). Output of LLM Call #5."""

    executive_summary: str = Field(
        max_length=800,
        description="One paragraph suitable for non-technical stakeholders.",
    )
    impacted_nodes: list[ImpactedNode]
    documentation_conflicts: list[str] = Field(
        default_factory=list,
        description=(
            "Documentation sections whose stated requirements may conflict "
            "with the proposed change."
        ),
    )
    estimated_scope: Scope


# =========================================================================
# BFS Output Types (dataclasses, not LLM schemas)
# =========================================================================


@dataclass
class NodeTrace:
    """Per-node BFS trace record."""

    depth: int
    causal_chain: list[str]
    path: list[str]
    source_seed: str
    low_confidence_seed: bool = False


@dataclass
class CISResult:
    """Complete BFS output separating seeds from propagated nodes."""

    sis_nodes: dict[str, NodeTrace] = field(default_factory=dict)
    propagated_nodes: dict[str, NodeTrace] = field(default_factory=dict)

    def combined(self) -> dict[str, NodeTrace]:
        """Return a merged view of SIS seeds and propagated nodes."""
        return {**self.sis_nodes, **self.propagated_nodes}

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
    file_path: str = ""
    file_classification: str | None = None
    chunk_type: str | None = None
    name: str = ""
    text_snippet: str = ""
    internal_logic_abstraction: str | None = None
    merged_doc_ids: list[str] = field(default_factory=list)
    bm25_score: float = 0.0
    cosine_score: float = 0.0
