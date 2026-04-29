"""Shared cross-cutting contracts: models, config, constants.

This package contains everything consumed by two or more of the domain
packages (indexer, pipeline, persistence, evaluation). It has no
dependencies on the other packages in the project.
"""

from impactracer.shared.models import (
    TruncatingModel,
    ChangeType,
    ChunkType,
    FileClassification,
    NodeType,
    CRInterpretation,
    CandidateVerdict,
    SISValidationResult,
    TraceDecision,
    TraceVerdict,
    TraceValidationResult,
    PropagationVerdict,
    PropagationValidationResult,
    Severity,
    Scope,
    ImpactedNode,
    ImpactReport,
    NodeTrace,
    CISResult,
    Candidate,
)
from impactracer.shared.constants import (
    RRF_PATH_WEIGHTS,
    LAYER_COMPAT,
    layer_compat,
    EDGE_CONFIG,
    LOW_CONF_CAPPED_EDGES,
    PROPAGATION_VALIDATION_EXEMPT_EDGES,
    SEVERITY_BY_EDGE_CHAIN_TYPE,
    severity_for_chain,
    BUILTIN_PATTERNS,
    PRIMITIVE_TYPES,
    HOOK_NAMES,
)
from impactracer.shared.config import Settings, get_settings

__all__ = [
    "TruncatingModel",
    "ChangeType",
    "ChunkType",
    "FileClassification",
    "NodeType",
    "CRInterpretation",
    "CandidateVerdict",
    "SISValidationResult",
    "TraceDecision",
    "TraceVerdict",
    "TraceValidationResult",
    "PropagationVerdict",
    "PropagationValidationResult",
    "Severity",
    "Scope",
    "ImpactedNode",
    "ImpactReport",
    "NodeTrace",
    "CISResult",
    "Candidate",
    "RRF_PATH_WEIGHTS",
    "LAYER_COMPAT",
    "layer_compat",
    "EDGE_CONFIG",
    "LOW_CONF_CAPPED_EDGES",
    "PROPAGATION_VALIDATION_EXEMPT_EDGES",
    "SEVERITY_BY_EDGE_CHAIN_TYPE",
    "severity_for_chain",
    "BUILTIN_PATTERNS",
    "PRIMITIVE_TYPES",
    "HOOK_NAMES",
    "Settings",
    "get_settings",
]
